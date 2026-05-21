import time

import numpy as np
import cvxpy as cp

from .base import BaseController, ControlLimits
from obstacles import normalize_obstacles


class MPCController(BaseController):
    """
    基于线性化自行车模型的 MPC 路径跟踪控制器 (Parameter 化版本)

    状态: x = [x, y, phi, v]^T
    控制: u = [a, delta]^T

    依赖 VehicleModel 提供线性化:
        A, B, c = vehicle_model.linearize(x_ref, u_ref)
        满足 x_{k+1} ≈ A @ x_k + B @ u_k + c

    实现要点:
        QP 结构在 __init__ 里一次性建好, 全部"会变"的量都用 cp.Parameter,
        solve() 只更新 .value 再调一次 prob.solve(warm_start=True)。
        cvxpy 的 canonicalization 只跑一次, 单步求解时间从 ~80ms 降到 ~10ms。

    用法:
        mpc = MPCController(vehicle_model, N=20)
        u, info = mpc.solve(state, ref, nearest_idx, u_prev=u_prev)
    """

    N_STATE = 4
    N_CTRL = 2

    def __init__(self, vehicle_model, N=20,
                 Q=None, R=None, Qf=None, Rd=None,
                 a_max=3.0, a_min=-5.0,
                 delta_max=np.deg2rad(30),
                 da_max=2.0, ddelta_max=np.deg2rad(30),
                 obstacles=None, r_car=1.25, margin=0.3,
                 avoid_side=None, avoid_bias_amp=1.5,
                 alpha_uncert=0.0,
                 avoidance_mode='hard', lambda_soft=1000.0):
        """
        参数:
            vehicle_model : VehicleModel 实例,用于线性化(也隐含 dt, L)
            N             : 预测时域(步数)
            Q             : 状态误差权重 (4x4)
            R             : 控制偏离参考的权重 (2x2)
            Qf            : 末端状态权重 (4x4); 严格要求闭环稳定时,
                            可解 DARE 得到 LQR 终端代价矩阵, 这里不做。
            Rd            : 控制增量权重 (2x2)
            a_max, a_min  : 加速度上下限
            delta_max     : 前轮转角绝对值上限
            da_max        : 每步加速度变化量上限
            ddelta_max    : 每步前轮转角变化量上限
            obstacles     : list of CircleObstacle | (cx, cy, r); None = 不加避障
                            非 None 时 QP 多 N*M 个半空间约束 (B 方案):
                                n_k · (x_var[k][:2] - p_obs) >= r_safe
                            n_k 在每次 solve() 时由 warm-start 预测点重新算
            r_car         : 车体等效半径 (用于膨胀 r_safe = r + r_car + margin)
                            后轴中心模型下取 L/2 = 1.25m 是常用近似
            margin        : 安全冗余 (m)
            avoid_side    : 'left' | 'right' | None
                            破对称 hint, 当障碍正前方 + 参考也走直线时,
                            B 方案半空间约束在 y 方向无分量 → MPC 会"停"
                            而非"绕". 给第一拍 warm-start 一个有偏的预测轨迹
                            (左/右弧), 让法向 n_k 含 y 分量, 把 QP 推向绕行解.
                            None: 不破对称 (适合参考已避障的场景, 如 A* + MPC)
            avoid_bias_amp: 破对称的初始横向偏移幅度 (m), 默认 1.5m
            alpha_uncert  : 不确定性锥增长率 (m/s). r_safe 在每个 stage k
                            额外膨胀 α * k * dt, 表达"预测越远越不准". 静态障碍
                            建议 0.0 (无需额外膨胀, r_inflate 已经够); 动态障碍
                            (恒速预测) 建议 0.3 ~ 1.0. 默认 0.0 → 与阶段 3 行为
                            完全一致, 老场景回归 byte-perfect.
            avoidance_mode: 'hard' (B 半空间, 阶段 3 默认) | 'soft' (A 软约束)
                            'hard': n·x >= b, QP 严格保证(线性化点附近), 但
                                    多障碍夹击时可能 infeasible
                            'soft': 加非负 slack 变量, n·x + s >= b, cost 加
                                    λ·Σs², 永远 feasible 但不保证不撞 (靠 λ 大)
                            默认 'hard' → 阶段 3 行为不变
            lambda_soft   : soft 模式下的 slack 二次惩罚权重. 越大越接近硬约束
        """
        self.vm = vehicle_model
        self.N = N
        self.dt = vehicle_model.dt

        # BaseController 接口属性
        self.limits = ControlLimits(a_min=a_min, a_max=a_max, delta_max=delta_max)

        # 默认权重
        self.Q = Q if Q is not None else np.diag([10.0, 10.0, 5.0, 1.0])
        self.R = R if R is not None else np.diag([1.0, 10.0])
        self.Qf = Qf if Qf is not None else self.Q.copy()
        self.Rd = Rd if Rd is not None else np.diag([0.1, 10.0])

        # 物理约束 (保留原属性, 内部约束构造和上层兼容查询都用)
        self.a_max = a_max
        self.a_min = a_min
        self.delta_max = delta_max
        self.da_max = da_max
        self.ddelta_max = ddelta_max

        # 避障配置 (B 方案: 半空间线性化, 迭代式 SCP)
        self.obstacles = (normalize_obstacles(obstacles)
                           if obstacles else [])
        self.r_car = r_car
        self.margin = margin
        if avoid_side not in (None, 'left', 'right'):
            raise ValueError(f"avoid_side 取 None/'left'/'right', 实际={avoid_side!r}")
        self.avoid_side = avoid_side
        self.avoid_bias_amp = avoid_bias_amp
        self.alpha_uncert = alpha_uncert
        if avoidance_mode not in ('hard', 'soft'):
            raise ValueError(f"avoidance_mode 取 'hard'/'soft', 实际={avoidance_mode!r}")
        self.avoidance_mode = avoidance_mode
        self.lambda_soft = float(lambda_soft)
        # warm-start 预测轨迹 (上拍 solve 的 x_var.value), 给避障约束做线性化
        # 没解过或上次失败时为 None, solve() 里退化为用当前 state 当线性化点
        self._x_pred_prev = None

        # ---- 一次性构建 QP ----
        self._build_problem()

    # =====================================================================
    # 一次性构建参数化 QP
    # =====================================================================

    def _build_problem(self):
        N, nx, nu = self.N, self.N_STATE, self.N_CTRL

        # 决策变量
        self.x_var = cp.Variable((N + 1, nx))
        self.u_var = cp.Variable((N, nu))

        # Parameter: 每个 solve() 都会更新 .value
        self.x0_p     = cp.Parameter(nx)
        self.x_ref_p  = cp.Parameter((N + 1, nx))
        self.u_ref_p  = cp.Parameter((N, nu))
        self.u_prev_p = cp.Parameter(nu)
        # 每个 stage 一组线性化矩阵
        self.A_p = [cp.Parameter((nx, nx)) for _ in range(N)]
        self.B_p = [cp.Parameter((nx, nu)) for _ in range(N)]
        self.c_p = [cp.Parameter(nx)       for _ in range(N)]

        cost = 0
        constraints = [self.x_var[0] == self.x0_p]

        for k in range(N):
            cost += cp.quad_form(self.x_var[k] - self.x_ref_p[k], self.Q)
            cost += cp.quad_form(self.u_var[k] - self.u_ref_p[k], self.R)

            u_prev_k = self.u_var[k - 1] if k > 0 else self.u_prev_p
            cost += cp.quad_form(self.u_var[k] - u_prev_k, self.Rd)

            # 线性化动力学
            constraints += [
                self.x_var[k + 1] ==
                self.A_p[k] @ self.x_var[k] + self.B_p[k] @ self.u_var[k] + self.c_p[k]
            ]

            # 控制量上下界
            constraints += [
                self.u_var[k, 0] >= self.a_min,
                self.u_var[k, 0] <= self.a_max,
                self.u_var[k, 1] >= -self.delta_max,
                self.u_var[k, 1] <=  self.delta_max,
            ]

            # 控制增量上下界
            du = self.u_var[k] - u_prev_k
            constraints += [
                du[0] >= -self.da_max,
                du[0] <=  self.da_max,
                du[1] >= -self.ddelta_max,
                du[1] <=  self.ddelta_max,
            ]

        # 末端代价
        cost += cp.quad_form(self.x_var[N] - self.x_ref_p[N], self.Qf)

        # ---- 避障约束 ----
        # 对每个 stage k=1..N (k=0 已被 x_0 == x0_p 钉死, 加约束反而易 infeasible)
        # 对每个障碍 m, 共同的几何近似 (B 半空间线性化):
        #     n_p[k][m] @ x_var[k][:2] >= b_p[k][m]
        # 等价于 n · (x_k - p_obs) >= r_safe, 其中:
        #     n        = 单位外法线 (warm-start 点 → 障碍中心 的反向)
        #     b_p      = r_safe + n · p_obs   (在 solve() 里预算, 避开 DPP 限制)
        #
        # 'hard' 模式 (B 方案): 上面这条作为硬约束, 严格保证. 可能 infeasible.
        # 'soft' 模式 (A 方案): 加非负 slack s_p[k][m] >= 0, 约束放松为
        #     n · x_k + s >= b, cost += λ·Σs². 永远 feasible 但不保证不撞.
        # 没有障碍时不加任何约束, QP 形态与无避障版严格一致 → 老场景回归不变
        if self.obstacles:
            M = len(self.obstacles)
            self.n_p = [[cp.Parameter(2) for _ in range(M)]
                        for _ in range(N + 1)]
            self.b_p = [[cp.Parameter()  for _ in range(M)]
                        for _ in range(N + 1)]
            # 给一个安全的初始 .value, 避免 cvxpy 拒解 (Parameter 必须 set value)
            for k in range(N + 1):
                for m in range(M):
                    self.n_p[k][m].value = np.array([1.0, 0.0])
                    self.b_p[k][m].value = -1e9   # 极松, 等价"无约束"

            if self.avoidance_mode == 'soft':
                # 整张矩阵化的 slack, k=0 不约束但留位置便于索引 (cost 只罚 1..N)
                self.slack = cp.Variable((N + 1, M), nonneg=True)
                cost = cost + self.lambda_soft * cp.sum_squares(self.slack[1:])

            for k in range(1, N + 1):
                for m in range(M):
                    if self.avoidance_mode == 'hard':
                        constraints += [
                            self.n_p[k][m] @ self.x_var[k][:2]
                            >= self.b_p[k][m]
                        ]
                    else:  # soft
                        constraints += [
                            self.n_p[k][m] @ self.x_var[k][:2]
                            + self.slack[k, m]
                            >= self.b_p[k][m]
                        ]

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    # =====================================================================
    # 求解一次 MPC (BaseController 接口)
    # =====================================================================

    def solve(self, state, ref, nearest_idx, u_prev=None):
        """
        参数:
            state       : np.ndarray (4,)        当前车辆状态
            ref         : ReferenceTrajectory    参考轨迹对象
            nearest_idx : int                    最近参考点索引
            u_prev      : np.ndarray (2,) 或 None 上一拍施加的控制

        返回:
            u0   : shape (2,)
            info : dict 含 status / cost / solve_time / x_pred / u_pred

        注:
            参考窗口的提取(按时间)和 u_ref 的反推(从 kappa)都在内部完成,
            外部不需要关心 N+1 / N 的 shape 区别。
        """
        N, nx, nu = self.N, self.N_STATE, self.N_CTRL

        # ---------- 取参考窗口 (按时间, 不按索引, 关键 trick) ----------
        # 参考轨迹是按弧长 ds 等距离散, 而 MPC 是按 dt 时间步推进。
        # 直接按 idx 取会让 ref_window[k] 和 "车 k 个 dt 之后" 错位,
        # 弯道跟踪反而恶化 (双移线场景已验证)。
        ref_window_full = ref.get_reference_window_by_time(
            idx_start=nearest_idx, dt=self.dt, N=N + 1,
        )
        x_ref_window = ref_window_full[:, :4]   # MPC 状态只用前 4 列

        # 反推参考控制: a_ref = 0 (恒速假设), delta_ref = arctan(L * kappa)
        kappa_ref = ref_window_full[:N, ref.IDX_KAPPA]
        delta_ref = np.arctan(ref.L * kappa_ref)
        u_ref_window = np.column_stack([np.zeros(N), delta_ref])

        # ---------- u_prev 处理 (状态外置, 不再依赖 self.u_prev) ----------
        if u_prev is None:
            u_prev_arr = np.zeros(nu)
        else:
            u_prev_arr = np.asarray(u_prev, dtype=float).reshape(nu)

        # ---------- 写入 Parameter ----------
        self.x0_p.value     = np.asarray(state, dtype=float)
        self.x_ref_p.value  = np.asarray(x_ref_window, dtype=float)
        self.u_ref_p.value  = np.asarray(u_ref_window, dtype=float)
        self.u_prev_p.value = u_prev_arr

        for k in range(N):
            A, B, c = self.vm.linearize(x_ref_window[k], u_ref_window[k])
            self.A_p[k].value = A
            self.B_p[k].value = B
            self.c_p[k].value = c

        # ---------- 避障约束: 用 warm-start 预测点重新线性化半空间 ----------
        if self.obstacles:
            self._update_avoidance_params(state)

        # ---------- 求解 ----------
        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                            max_iter=20000)
        except Exception as e:
            solve_time = time.perf_counter() - t0
            print(f"[MPC] 求解器异常: {e}")
            self._x_pred_prev = None   # 失败丢弃 warm start
            return u_prev_arr.copy(), {
                'status': 'error', 'cost': np.nan,
                'solve_time': solve_time,
                'x_pred': None, 'u_pred': None,
            }
        solve_time = time.perf_counter() - t0

        if self.prob.status not in ['optimal', 'optimal_inaccurate']:
            print(f"[MPC] 求解失败, status = {self.prob.status}, 沿用上次控制")
            self._x_pred_prev = None   # 不可靠的 x_var 不能拿来作下一步 warm start
            return u_prev_arr.copy(), {
                'status': self.prob.status, 'cost': np.nan,
                'solve_time': solve_time,
                'x_pred': None, 'u_pred': None,
            }

        u0 = self.u_var.value[0]
        # 缓存这次的预测轨迹给下一拍避障线性化
        if self.obstacles:
            self._x_pred_prev = self.x_var.value.copy()

        return u0, {
            'status': self.prob.status,
            'cost': self.prob.value,
            'solve_time': solve_time,
            'x_pred': self.x_var.value,
            'u_pred': self.u_var.value,
        }

    # =====================================================================
    # 避障: 半空间约束的线性化点更新
    # =====================================================================

    def _update_avoidance_params(self, state):
        """
        在每个 stage k=1..N, 对每个障碍 m, 用 warm-start 预测点 x̂_k 重新算
        外法线 n_k 和 RHS 标量 b_k:

            n_k = (x̂_k - p_obs) / ||x̂_k - p_obs||      单位外法线
            b_k = r_safe + n_k · p_obs                  把 'n·(x-p_obs)>=r_safe'
                                                         整理到 'n·x >= b' 形式
            r_safe = obs.r + r_car + margin

        warm-start 来源:
            - 有上拍预测 → self._x_pred_prev[k]
            - 第一次 / 上拍失败 → 全部 stage 都用当前 state[:2] 退化
              (n 在所有 k 上是同一方向, MPC 第二拍才会真正"看到" 沿预测轨迹的避让)

        退化保护:
            如果 ||x̂_k - p_obs|| < 1e-3, 说明车几乎踩在障碍中心上 (一般是仿真
            出错或场景太极端). 任意取 n=[1,0], r_safe 顶住, 让求解还能继续。

        DPP 友好:
            n_k @ x_var[k][:2] >= b_k 中, n_k 与 b_k 都是 cp.Parameter, x_var 是
            Variable. 不存在 Parameter*Parameter 的乘法 (b 在 numpy 里预算完才赋
            value), cvxpy 的 DPP 检查通过。
        """
        N = self.N
        if self._x_pred_prev is not None:
            x_lin = self._x_pred_prev
        elif self.avoid_side is not None:
            # 第一拍破对称: 沿当前航向往前推, 同时叠加正弦横向偏置, 让 n_k 含 y 分量
            x_lin = self._biased_warm_start(state)
        else:
            # 退化: 全 stage 用当前 state. 复制 N+1 份。
            x_lin = np.tile(np.asarray(state, dtype=float), (N + 1, 1))

        # 阶段 5: 每个 stage k 取障碍的 *预测位置* p_obs(k*dt), 而非当前位置;
        # r_safe 也随 k 膨胀 (不确定性锥), 表达"预测越远越不准".
        # 静态障碍 (CircleObstacle) 的 predict 恒返回 self.xy → 当 alpha_uncert=0
        # 时与阶段 3 行为完全一致, 老场景回归 byte-perfect.
        r_base = self.r_car + self.margin
        alpha = self.alpha_uncert
        for m, obs in enumerate(self.obstacles):
            for k in range(1, N + 1):
                future_dt = k * self.dt
                p_obs_k = obs.predict(future_dt)
                r_safe_k = obs.r + r_base + alpha * future_dt
                diff = x_lin[k][:2] - p_obs_k
                d = np.linalg.norm(diff)
                if d < 1e-3:
                    n = np.array([1.0, 0.0])
                else:
                    n = diff / d
                self.n_p[k][m].value = n
                self.b_p[k][m].value = r_safe_k + n @ p_obs_k

    def _biased_warm_start(self, state):
        """
        第一拍 warm-start 退化时, 用一个有横向偏置的"假想轨迹"做线性化点,
        让法向 n_k 在 y 方向有非零分量, MPC 才会绕而不是停。

        构造:
            x[k] = x_now + v * cos(phi) * k * dt    (沿航向直行)
            y[k] = y_now + v * sin(phi) * k * dt + amp * sin(πk/N) * sign
            phi[k] = phi_now (近似)
            v[k] = v_now (近似)

        sign = +1 ('left') 或 -1 ('right').
        amp = self.avoid_bias_amp, 默认 1.5m.
        正弦形态: 起点终点偏置都为 0, 中间最大, 模拟"顺利绕开后回到原线"。
        """
        N = self.N
        x, y, phi, v = state
        sign = +1.0 if self.avoid_side == 'left' else -1.0
        amp = self.avoid_bias_amp
        x_lin = np.zeros((N + 1, 4))
        for k in range(N + 1):
            t = k * self.dt
            x_lin[k, 0] = x + v * np.cos(phi) * t
            x_lin[k, 1] = (y + v * np.sin(phi) * t
                           + amp * np.sin(np.pi * k / N) * sign)
            x_lin[k, 2] = phi
            x_lin[k, 3] = v
        return x_lin


# ========================================================================
# 单元测试 (保留, 用绝对 import)
# ========================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import matplotlib.pyplot as plt
    from core.reference_trajectory import ReferenceTrajectory
    from core.vehicle_model import VehicleModel

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    vm = VehicleModel(L=2.5, dt=0.05)
    ref = ReferenceTrajectory(L=2.5).generate_circle(radius=20, v_ref=5.0)
    mpc = MPCController(vm, N=20)

    x_current = np.array([18.0, 0.0, np.pi / 2, 4.0])
    idx = ref.find_nearest(x_current[0], x_current[1])

    u0, info = mpc.solve(x_current, ref, idx)
    print(f"求解状态: {info['status']}, 耗时 {info['solve_time']*1000:.2f} ms")
    print(f"当前控制: a = {u0[0]:.3f} m/s^2, delta = {np.rad2deg(u0[1]):.2f}°")
    print(f"代价: {info['cost']:.4f}")

    # warm start 效果
    times = []
    for _ in range(20):
        _, inf = mpc.solve(x_current, ref, idx, u_prev=u0)
        times.append(inf['solve_time'] * 1000)
    print(f"后续 20 次平均求解耗时: {np.mean(times):.2f} ms (含 warm start)")
