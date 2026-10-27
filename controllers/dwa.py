"""
DWA (Dynamic Window Approach) 局部规划控制器

与 MPC 同一个 BaseController 接口, 定位是"采样类"局部规划, 拿来和"优化类"的
MPC 避障做对比:
    MPC: 在线优化整条控制序列. 圆障碍的"圆外"是非凸集合, 要先线性化成半空间才能
         放进 QP; 障碍正对路径时还需要 avoid_side 之类的提示破对称
    DWA: 在一步可达的控制量集合 (动态窗口) 里采样, 每组控制量保持不变往前推,
         逐条做碰撞检查再打分. 不需要凸化, 绕左还是绕右由打分自然选出来

动态窗口 (控制量 u = [a, delta], 输入/增量上下限与 MPC 一致):
    a     ∈ [a_prev - da_max,         a_prev + da_max]         ∩ [a_min, a_max]
    delta ∈ [delta_prev - ddelta_max, delta_prev + ddelta_max] ∩ [-delta_max, delta_max]

    经典 DWA 给差速机器人设计, 在 (v, ω) 速度空间采样. 这里输入是 [a, delta],
    一步可达的 (v, delta) 正好由这两个窗口决定, 所以直接在控制空间采样。
    a 在窗口内均匀采样; delta 以上一拍为中心按几何间隔采样 (中间密两头疏, 小修正和
    急转都能采到); 再补"保持上一拍""保持车速""追参考车速""曲率前馈"几个名义候选。

rollout:
    每组 (a, delta) 在 horizon 秒内保持不变, 用与 VehicleModel.step 相同的前向欧拉
    批量前推. 保持 a 不变 (经典做法是保持 v 不变) 是为了让"刹车让行"也能成为
    候选, 横穿行人场景需要。

碰撞检查 (与 MPC 的 r_safe 定义一致, 含不确定性锥):
    第 k 步 (t = k*dt) 车与障碍预测位置 obs.predict(t) 的距离须满足
        d >= r_safe(t) = obs.r + r_car + margin + alpha_uncert * t
    任意一步违反 → 该候选不可行

打分 (越小越好, 只在可行候选里选):
    J = w_lat    * mean_k |e_lat,k|          路径横向偏差
      + w_head   * |e_phi,K|                 rollout 终点航向偏差
      + w_speed  * mean_k |v_k - v_ref,k|    速度偏差
      + w_obs    * max(0, 1/c - 1/c_max)     离障碍越近越大, 超过 c_max 不计
      + w_ddelta * |delta - delta_prev|      转角变化
      + w_da     * |a - a_prev|              加速度变化
      + w_goal   * min_k |p_k - p_goal|      终点在 rollout 可达范围内时才计, 防止从终点旁错过
    c = min_{k,m} (d - r_safe): 整条 rollout 上的最小剩余安全距离

    没有可行候选时 (例如障碍突然贴近): 选 c 最大 (最不危险) 的那组,
    info['status'] = 'no_admissible'
"""

import time
import numpy as np

from .base import BaseController, ControlLimits
from obstacles import normalize_obstacles


def _wrap_to_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


class DWAController(BaseController):
    """
    DWA 局部规划: 动态窗口采样 → 批量 rollout → 碰撞检查 → 打分 → 执行最优候选的第一步。

    无内部状态 (上一拍控制量由 u_prev 传入)。
    info 里的 'x_pred' 是最优 rollout (K+1, 4), 画图 / ROS2 直接复用 MPC 预测轨迹的显示。
    """

    def __init__(self, vehicle_model,
                 horizon=1.8, n_a=5, n_delta=8, delta_ratio=1.0 / 128,
                 w_lat=2.0, w_head=0.5, w_speed=3.0, w_obs=1.0,
                 w_ddelta=1.0, w_da=0.05, w_goal=3.0, clearance_max=3.0,
                 a_max=3.0, a_min=-5.0, delta_max=np.deg2rad(30),
                 da_max=2.0, ddelta_max=np.deg2rad(30),
                 obstacles=None, r_car=1.25, margin=0.3, alpha_uncert=0.0,
                 proj_stride=5):
        """
        参数:
            vehicle_model : VehicleModel, rollout 用它的 L / dt / v_min / v_max
            horizon       : rollout 时长 (s). block / oncoming / crossing 上 1.5~2.0 s 都能
                            绕开并回到终点, 取中间值; 再长 (2.2 s 起) block 绕完回线太慢
            n_a           : 动态窗口内 a 的均匀采样数
            n_delta       : delta 每侧的几何间隔偏移个数, 最小偏移 = ddelta_max * delta_ratio
                            (默认 8 个, 30°/步窗口时为 0.23° ... 30°)
            w_*           : 打分权重, 见模块 docstring. w_speed 要压过"刹停在障碍前"的
                            诱惑: 权重低时, 刹车候选的代价比绕行候选的路径偏差代价小,
                            DWA 会停在障碍前出不来 (局部极小)
            clearance_max : 剩余安全距离超过它就不再计障碍代价 (m)
            a_*, delta_max, da_max, ddelta_max : 与 MPC 同名参数一致
            obstacles     : list of CircleObstacle | DynamicObstacle | tuple; None = 无障碍
            r_car, margin, alpha_uncert : r_safe 的组成, 与 MPC 一致
            proj_stride   : 算横向误差时参考点的抽稀步长 (ds=0.1 时 5 → 0.5 m)
        """
        self.vm = vehicle_model
        self.dt = vehicle_model.dt
        self.limits = ControlLimits(a_min=a_min, a_max=a_max, delta_max=delta_max)
        self.da_max = da_max
        self.ddelta_max = ddelta_max

        self.K = int(round(horizon / self.dt))
        self.n_a = n_a
        self.n_delta = n_delta
        self.delta_ratio = delta_ratio
        self.w_lat, self.w_head, self.w_speed = w_lat, w_head, w_speed
        self.w_obs, self.w_ddelta, self.w_da = w_obs, w_ddelta, w_da
        self.w_goal = w_goal
        self.clearance_max = clearance_max
        self.proj_stride = proj_stride

        self.obstacles = normalize_obstacles(obstacles) if obstacles else []
        self.r_car = r_car
        self.margin = margin
        self.alpha_uncert = alpha_uncert

    def solve(self, state, ref, nearest_idx, u_prev=None):
        t0 = time.perf_counter()
        state = np.asarray(state, dtype=float)
        u_prev = (np.zeros(2) if u_prev is None
                  else np.asarray(u_prev, dtype=float).reshape(2))
        # 静止时负加速度不起作用 (VehicleModel 把 v 截在 v_min, 车不会倒退), 上一拍的
        # 刹车指令按 0 处理; 否则窗口里全是负加速度, 车会一直停着出不来
        if (self.vm.v_min is not None and state[3] <= self.vm.v_min + 1e-6
                and u_prev[0] < 0.0):
            u_prev = np.array([0.0, u_prev[1]])

        # ---------- 1. 动态窗口采样 ----------
        a_c, d_c = self._sample_window(u_prev, state, ref, nearest_idx)
        A, D = (g.ravel() for g in np.meshgrid(a_c, d_c, indexing='ij'))

        # ---------- 2. 批量 rollout ----------
        traj = self._rollout(state, A, D)                 # (M, K, 4)

        # ---------- 3. 碰撞检查 ----------
        clearance = self._clearance(traj)                 # (M,)
        admissible = clearance >= 0.0

        # ---------- 4. 打分 ----------
        e_lat, e_phi, v_ref = self._tracking_errors(traj, ref, nearest_idx, state)
        c_eff = np.maximum(clearance, 0.05)               # 防 1/0, 无障碍时为 inf → 代价 0
        obs_term = np.maximum(0.0, 1.0 / c_eff - 1.0 / self.clearance_max)
        cost = (self.w_lat * np.abs(e_lat).mean(axis=1)
                + self.w_head * np.abs(e_phi[:, -1])
                + self.w_speed * np.abs(traj[:, :, 3] - v_ref).mean(axis=1)
                + self.w_obs * obs_term
                + self.w_ddelta * np.abs(D - u_prev[1])
                + self.w_da * np.abs(A - u_prev[0]))
        # 终点进入 rollout 可达范围后, 按 rollout 离终点的最近距离计代价 (经典 DWA 的
        # goal 项); 否则路径在弯道中间结束时, 切弯会让车从终点旁边错过去
        goal = ref.points[-1, :2]
        reach = traj[:, :, 3].sum(axis=1).max() * self.dt
        if np.hypot(state[0] - goal[0], state[1] - goal[1]) <= reach:
            cost += self.w_goal * np.hypot(traj[:, :, 0] - goal[0],
                                           traj[:, :, 1] - goal[1]).min(axis=1)

        info = {}
        if admissible.any():
            best = int(np.argmin(np.where(admissible, cost, np.inf)))
        else:
            best = int(np.argmax(clearance))
            info['status'] = 'no_admissible'

        u = np.array([A[best], D[best]])
        info.update({
            'solve_time': time.perf_counter() - t0,
            'cost': float(cost[best]),
            'x_pred': np.vstack([state, traj[best]]),
        })
        return u, info

    # =====================================================================
    # 内部步骤
    # =====================================================================

    def _sample_window(self, u_prev, state, ref, nearest_idx):
        """
        动态窗口内采样, 都截到窗口内:
            a     : 均匀 n_a 个, 再补上一拍的 a、a=0 (保持车速)、追参考车速的 a
                    (rollout 中点追平 v_ref)
            delta : 以上一拍为中心按几何间隔取偏移 ±ddelta_max·[r, ..., 1], 中间密
                    两头疏, 再补窗口两端和最近参考点的曲率前馈 arctan(L·κ)
        rollout 里控制量全程不变, 均匀格点太粗时稳态会卡在离目标最近的格点上:
        8 m/s 下 2° 的转角间隔, 一个 rollout 就偏出两三米, 小修正采不到;
        a 的间隔 1 m/s² 时, 0.6 m/s 的车速差也降不下去。
        """
        a_prev, d_prev = u_prev
        a_lo = max(self.limits.a_min, a_prev - self.da_max)
        a_hi = min(self.limits.a_max, a_prev + self.da_max)
        d_lo = max(-self.limits.delta_max, d_prev - self.ddelta_max)
        d_hi = min(self.limits.delta_max, d_prev + self.ddelta_max)
        v_ref = ref.points[nearest_idx, ref.IDX_V]
        a_track = (v_ref - state[3]) / (0.5 * self.K * self.dt)
        d_ff = np.arctan(self.vm.L * ref.points[nearest_idx, ref.IDX_KAPPA])
        a_c = np.unique(np.concatenate([np.linspace(a_lo, a_hi, self.n_a),
                                        np.clip([a_prev, 0.0, a_track], a_lo, a_hi)]))
        offs = self.ddelta_max * np.geomspace(self.delta_ratio, 1.0, self.n_delta)
        d_c = d_prev + np.concatenate([-offs[::-1], [0.0], offs])
        d_c = np.unique(np.clip(np.concatenate([d_c, [d_lo, d_hi, d_ff]]), d_lo, d_hi))
        return a_c, d_c

    def _rollout(self, state, A, D):
        """批量版 VehicleModel.step: M 组控制量各自保持不变, 前推 K 步. 返回 (M, K, 4)。"""
        L, dt = self.vm.L, self.dt
        M = len(A)
        x, y, phi, v = (np.full(M, s) for s in state)
        tan_d = np.tan(D)
        out = np.empty((M, self.K, 4))
        for k in range(self.K):
            # 与 VehicleModel.step 同序: 位置/航向都用本步开始时的 phi、v
            x, y, phi, v = (x + v * np.cos(phi) * dt,
                            y + v * np.sin(phi) * dt,
                            phi + v / L * tan_d * dt,
                            v + A * dt)
            if self.vm.v_min is not None:
                v = np.maximum(v, self.vm.v_min)
            if self.vm.v_max is not None:
                v = np.minimum(v, self.vm.v_max)
            out[:, k, 0], out[:, k, 1], out[:, k, 2], out[:, k, 3] = x, y, phi, v
        return out

    def _clearance(self, traj):
        """每个候选整条 rollout 上的最小剩余安全距离 min_{k,m}(d - r_safe); 无障碍时 +inf。"""
        M = traj.shape[0]
        if not self.obstacles:
            return np.full(M, np.inf)
        t = np.arange(1, self.K + 1) * self.dt            # rollout 第 k 个点对应的时刻
        r_base = self.r_car + self.margin
        c = np.full(M, np.inf)
        for obs in self.obstacles:
            p = np.array([obs.predict(tk) for tk in t])   # (K, 2) 障碍预测位置
            r_safe = obs.r + r_base + self.alpha_uncert * t
            d = np.hypot(traj[:, :, 0] - p[:, 0], traj[:, :, 1] - p[:, 1]) - r_safe
            c = np.minimum(c, d.min(axis=1))
        return c

    def _tracking_errors(self, traj, ref, nearest_idx, state):
        """
        rollout 每个点相对参考路径的横向误差 / 航向误差 / 该处参考速度, 均为 (M, K)。

        在 nearest_idx 附近截一段参考 (覆盖本拍 rollout 最远能到的范围), 每 proj_stride
        个点取一个做最近点搜索, 再按该点切线投影求横向误差: 投影消掉了抽稀带来的
        沿路径方向误差, 弯道上剩下的是 O(Δs²·κ) 量级。
        """
        pts = ref.points
        n = len(pts)
        ds = (pts[-1, ref.IDX_S] - pts[0, ref.IDX_S]) / max(1, n - 1)
        travel = traj[:, :, 3].sum(axis=1).max() * self.dt    # rollout 最远行驶距离
        i_lo = max(0, nearest_idx - int(2.0 / ds))
        i_hi = min(n, nearest_idx + int((travel + 2.0) / ds) + 1)
        idx = np.arange(i_lo, i_hi, self.proj_stride)
        if idx[-1] != i_hi - 1:
            idx = np.append(idx, i_hi - 1)
        seg = pts[idx]

        px, py = traj[:, :, 0], traj[:, :, 1]
        d2 = ((px[..., None] - seg[:, ref.IDX_X]) ** 2
              + (py[..., None] - seg[:, ref.IDX_Y]) ** 2)  # (M, K, S)
        near = seg[np.argmin(d2, axis=-1)]                  # (M, K, 6)
        phi_r = near[..., ref.IDX_PHI]
        e_lat = (-np.sin(phi_r) * (px - near[..., ref.IDX_X])
                 + np.cos(phi_r) * (py - near[..., ref.IDX_Y]))
        e_phi = _wrap_to_pi(traj[:, :, 2] - phi_r)
        return e_lat, e_phi, near[..., ref.IDX_V]


# ========================================================================
# 单元测试
# ========================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.vehicle_model import VehicleModel
    from core.reference_trajectory import ReferenceTrajectory

    vm = VehicleModel(L=2.5, dt=0.1)
    ref = ReferenceTrajectory(L=2.5).generate_straight_line(length=60, v_ref=6.0)
    dwa = DWAController(vm)

    # 测试 1: 批量 rollout 与 VehicleModel.step 逐步推进结果一致
    s0 = np.array([0.0, 0.5, 0.1, 5.0])
    A = np.array([1.0, -2.0])
    D = np.array([0.1, -0.2])
    traj = dwa._rollout(s0, A, D)
    for m in range(len(A)):
        s = s0.copy()
        for k in range(dwa.K):
            s = vm.step(s, [A[m], D[m]])
            assert np.allclose(traj[m, k], s), f"rollout 第 {m} 组第 {k} 步不一致"
    print("rollout 与 VehicleModel.step 一致")

    # 测试 2: 无障碍, 车在路径上且车速 = v_ref → 应直行保速
    state = np.array([0.0, 0.0, 0.0, 6.0])
    u, info = dwa.solve(state, ref, 0)
    print(f"无障碍: a={u[0]:.3f}  delta={np.rad2deg(u[1]):.2f}°  "
          f"耗时 {info['solve_time']*1e3:.2f} ms")
    assert abs(u[0]) < 1e-9 and abs(u[1]) < 1e-9, "完美跟踪时应保持 a=0, delta=0"

    # 测试 3: 正前方障碍 → 选中的 rollout 全程满足 r_safe
    obs = (15.0, 0.0, 1.0)
    dwa_obs = DWAController(vm, obstacles=[obs])
    u, info = dwa_obs.solve(state, ref, 0)
    xp = info['x_pred']
    d_min = np.min(np.hypot(xp[1:, 0] - obs[0], xp[1:, 1] - obs[1]))
    r_safe = obs[2] + dwa_obs.r_car + dwa_obs.margin
    print(f"正前方障碍: a={u[0]:.3f}  delta={np.rad2deg(u[1]):.2f}°  "
          f"rollout 最小距离 {d_min:.2f} m (r_safe={r_safe:.2f} m)")
    assert 'status' not in info and d_min >= r_safe, "应选到满足 r_safe 的候选"

    # 测试 4: 求解时间
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        dwa_obs.solve(state, ref, 0)
        times.append((time.perf_counter() - t0) * 1e3)
    a_c, d_c = dwa_obs._sample_window(np.zeros(2), state, ref, 0)
    print(f"50 次平均: {np.mean(times):.2f} ms  "
          f"(候选 {len(a_c)}x{len(d_c)} 组, rollout {dwa_obs.K} 步)")
