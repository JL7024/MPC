"""
误差状态 LQR 控制器 (教科书路径跟踪 LQR)

状态:    e = [e_lat, e_phi, e_v]^T   (车相对参考点的误差)
控制:    du = [a, delta - delta_ff]^T
最终输出 u = u_ref + du, 其中 u_ref = [0, arctan(L*kappa)]

误差动力学推导 (连续时间, 在参考点局部线性化, 假设 e_phi 小):
    e_lat_dot = v * sin(e_phi)              ≈ v * e_phi
    e_phi_dot = v/L * tan(delta) - kappa*v  ≈ (v/L) * delta_err   (扣掉 delta_ff = arctan(L*kappa))
    e_v_dot   = a

    所以连续时间误差动力学 (以 v 作为时变参数, 不是状态):
        Ac = [[0, v, 0],
              [0, 0, 0],
              [0, 0, 0]]
        Bc = [[0, 0],
              [0, v/L],
              [1, 0]]

    离散化 (前向欧拉, 与车辆模型一致):
        A_d = I + Ac * dt
        B_d = Bc * dt

LQR 求解:
    P  = solve_discrete_are(A_d, B_d, Q, R)
    K  = (R + B_dᵀ P B_d)⁻¹ B_dᵀ P A_d
    du = -K @ e

由于 v 是时变的, A_d/B_d 每步都变, DARE 每步都重解。3x3 的 DARE 很快 (<1ms)。

为什么 v 不放进状态:
    放进去要写 v = v_ref + e_v 的非线性耦合, 线性化后变成 LTV 系统, 推导更绕。
    把 v 当 frozen parameter 是工业界 path-tracking LQR 的标准简化。

为什么需要 u_ref 前馈:
    弯道上 delta_ref = arctan(L*kappa) 才是稳态正确转角;
    如果没有前馈, e=0 时 du=0, u=0 → 直行 → 偏离 → e 增大 → 跟踪不稳。
"""

import time
import numpy as np
from scipy.linalg import solve_discrete_are

from .base import BaseController, ControlLimits


def _wrap_to_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


class LQRController(BaseController):
    """
    误差状态 LQR. 每步在参考点局部线性化 + 解 DARE. 无约束, saturation 后置。

    参数语义:
        Q = diag([q_lat, q_phi, q_v])  : 误差权重
        R = diag([r_a,  r_delta])      : 控制权重
        与 MPC 的权重 spirit 对齐: q_lat=q_phi=10/5, q_v=1, r_delta>>r_a
    """

    def __init__(self, vehicle_model,
                 Q=None, R=None,
                 v_min_for_lin=0.5,
                 a_max=3.0, a_min=-5.0,
                 delta_max=np.deg2rad(30)):
        """
        参数:
            vehicle_model  : VehicleModel
            Q              : (3,3) 误差状态权重 [e_lat, e_phi, e_v]
            R              : (2,2) 控制权重 [a, delta]
            v_min_for_lin  : 线性化时 v 的下限 (避免 v=0 时 B 退化, 解 DARE 失败)
            a_*, delta_*   : saturation 上下限
        """
        self.vm = vehicle_model
        self.dt = vehicle_model.dt
        self.limits = ControlLimits(a_min=a_min, a_max=a_max, delta_max=delta_max)

        self.Q = Q if Q is not None else np.diag([10.0, 5.0, 1.0])
        self.R = R if R is not None else np.diag([1.0, 10.0])
        self.v_min_for_lin = v_min_for_lin

        # 缓存上一拍的 K, DARE 解算失败时回退用
        self._last_K = None

    def _compute_K(self, v_lin):
        """在线性化速度 v_lin 处求 LQR 增益 K (3x3 DARE)"""
        L = self.vm.L
        Ac = np.array([
            [0.0, v_lin, 0.0],
            [0.0, 0.0,   0.0],
            [0.0, 0.0,   0.0],
        ])
        Bc = np.array([
            [0.0, 0.0],
            [0.0, v_lin / L],
            [1.0, 0.0],
        ])
        A_d = np.eye(3) + Ac * self.dt
        B_d = Bc * self.dt

        # DARE: P = A^T P A - A^T P B (R + B^T P B)^-1 B^T P A + Q
        P = solve_discrete_are(A_d, B_d, self.Q, self.R)
        K = np.linalg.solve(self.R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
        return K

    def solve(self, state, ref, nearest_idx, u_prev=None):
        t0 = time.perf_counter()

        x, y, phi, v = state
        ref_state = ref.get_reference_state(nearest_idx)
        x_r, y_r, phi_r, v_r = ref_state
        kappa = ref.points[nearest_idx, ref.IDX_KAPPA]
        L = self.vm.L

        # ---------- 误差 ----------
        e_lat = -np.sin(phi_r) * (x - x_r) + np.cos(phi_r) * (y - y_r)
        e_phi = _wrap_to_pi(phi - phi_r)
        e_v   = v - v_r   # 注意方向: v - v_ref (Q 是对称二次型, 方向影响在 K 上一致)
        e = np.array([e_lat, e_phi, e_v])

        # ---------- 线性化速度: 用 v_ref 比 v 更稳 ----------
        # 用 v 的话 v=0 时 B 全 0 → DARE 不可解; 用 v_ref 单调连续。
        v_lin = max(v_r, self.v_min_for_lin)

        # ---------- 求 K (失败时回退到上一拍) ----------
        try:
            K = self._compute_K(v_lin)
            self._last_K = K
        except Exception as ex:
            if self._last_K is None:
                # 第一拍就失败, 直接 0 控制
                return np.zeros(2), {
                    'solve_time': time.perf_counter() - t0,
                    'status': f'dare_failed_init: {ex}',
                }
            K = self._last_K

        # ---------- 控制律 ----------
        du = -K @ e
        u_ref = np.array([0.0, np.arctan(L * kappa)])
        u = u_ref + du

        # saturation
        u[0] = np.clip(u[0], self.limits.a_min, self.limits.a_max)
        u[1] = np.clip(u[1], -self.limits.delta_max, self.limits.delta_max)

        return u, {'solve_time': time.perf_counter() - t0}


# ========================================================================
# 单元测试
# ========================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.vehicle_model import VehicleModel
    from core.reference_trajectory import ReferenceTrajectory

    vm = VehicleModel(L=2.5, dt=0.1)
    ref = ReferenceTrajectory(L=2.5).generate_circle(radius=20, v_ref=5.0)
    lqr = LQRController(vm)

    # 测试 1: 完美跟踪状态 (e=0), 控制应等于前馈
    perfect_state = ref.get_reference_state(0)
    u, info = lqr.solve(perfect_state, ref, 0)
    delta_ff_expected = np.arctan(vm.L / 20.0)   # arctan(L*kappa) = arctan(L/R)
    print(f"完美跟踪: u = {u}, 期望 delta ≈ {delta_ff_expected:.4f}, "
          f"实际 {u[1]:.4f}, 耗时 {info['solve_time']*1e6:.1f} μs")
    assert abs(u[0]) < 1e-6, "完美跟踪 a 应为 0"
    assert abs(u[1] - delta_ff_expected) < 1e-6, "完美跟踪 delta 应等于前馈"

    # 测试 2: 偏离参考, K 应是有限值
    state = np.array([18.0, 0.0, np.pi / 2, 4.0])
    idx = ref.find_nearest(state[0], state[1])
    u, info = lqr.solve(state, ref, idx)
    print(f"偏离跟踪: u = {u}  (a={u[0]:.3f}, delta={np.rad2deg(u[1]):.2f}°), "
          f"耗时 {info['solve_time']*1e6:.1f} μs")

    # 测试 3: 求解时间统计
    import time as t
    times = []
    for _ in range(100):
        t0 = t.perf_counter()
        lqr.solve(state, ref, idx)
        times.append((t.perf_counter() - t0) * 1e3)
    print(f"100 次平均: {np.mean(times):.3f} ms (DARE 重解每步)")
