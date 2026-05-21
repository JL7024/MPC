"""
PID 控制器: Stanley 横向 + 速度 PI 纵向

横向: Stanley 几何控制器 (经典自动驾驶横向控制)
    delta = -e_phi - arctan(k_lat * e_lat / (v + eps)) + delta_ff

    其中:
        e_phi   : 航向误差 (车头 phi - 参考 phi_r), 用项目里 compute_errors 的定义
        e_lat   : 横向误差 (左正右负), Frenet 横向分量
        k_lat   : Stanley 增益, 越大对横向误差反应越激进
        v + eps : 防除零, 低速时给个保底速度
        delta_ff: 前馈, arctan(L * kappa) 让弯道稳态零误差

    符号推导:
        - e_phi > 0  → 车头偏左于参考 → 需要右转 (delta < 0) → 公式里 -e_phi ✓
        - e_lat > 0  → 车在参考左边   → 需要右转 (delta < 0) → 公式里 -arctan(...) ✓

纵向: 速度 PI + anti-windup
    a = Kp * e_v + Ki * ∫e_v dt
    其中 e_v = v_ref - v (注意方向, 速度落后时 e_v > 0 → 加速)
    饱和时不积分 (clamping anti-windup), 防止积分爆掉

设计上的取舍:
    - 不用 PD on lateral error (那是普通 PID, 不如 Stanley 几何意义清晰)
    - Stanley 单参数 k_lat 调起来比 Kp/Ki/Kd 三参数简单
    - 速度环不用 D, 速度噪声会被 D 放大; PI 足够
"""

import time
import numpy as np

from .base import BaseController, ControlLimits


def _wrap_to_pi(a):
    """归一化角度到 [-pi, pi]"""
    return np.arctan2(np.sin(a), np.cos(a))


class PIDController(BaseController):
    """
    Stanley 横向 + 速度 PI 纵向。

    内部状态: 仅速度 PI 的积分项 (self._integral_v)。
    控制器实例每次仿真新建一次, 保证积分清零。
    """

    def __init__(self, vehicle_model,
                 k_lat=2.0,
                 kp_v=1.0, ki_v=0.5,
                 v_eps=1.0,
                 a_max=3.0, a_min=-5.0,
                 delta_max=np.deg2rad(30)):
        """
        参数:
            vehicle_model : VehicleModel 实例
            k_lat         : Stanley 横向增益
            kp_v, ki_v    : 速度环 PI 增益
            v_eps         : Stanley 分母保底速度 (防除零, 也避免低速过激进)
            a_*, delta_*  : saturation 上下限 (与 MPC 对齐)
        """
        self.vm = vehicle_model
        self.dt = vehicle_model.dt
        self.limits = ControlLimits(a_min=a_min, a_max=a_max, delta_max=delta_max)

        self.k_lat = k_lat
        self.kp_v = kp_v
        self.ki_v = ki_v
        self.v_eps = v_eps

        # 算法内部状态 (不通过 u_prev 外置, 因为这是控制器自己的"记忆")
        self._integral_v = 0.0

    def solve(self, state, ref, nearest_idx, u_prev=None):
        t0 = time.perf_counter()

        x, y, phi, v = state
        ref_state = ref.get_reference_state(nearest_idx)
        x_r, y_r, phi_r, v_r = ref_state
        kappa = ref.points[nearest_idx, ref.IDX_KAPPA]

        # ---------- 误差计算 (与 viz/plot.py compute_errors 同惯例) ----------
        e_lat = -np.sin(phi_r) * (x - x_r) + np.cos(phi_r) * (y - y_r)
        e_phi = _wrap_to_pi(phi - phi_r)
        e_v   = v_r - v   # 注意方向: v_ref - v, 落后时 e_v > 0 → 加速

        # ---------- Stanley 横向 ----------
        v_safe = max(v, self.v_eps)
        delta_fb = -e_phi - np.arctan(self.k_lat * e_lat / v_safe)
        delta_ff = np.arctan(self.vm.L * kappa)
        delta = delta_ff + delta_fb
        delta = np.clip(delta, -self.limits.delta_max, self.limits.delta_max)

        # ---------- 速度 PI + clamping anti-windup ----------
        # 先试积分, 算未饱和 a
        new_integral = self._integral_v + e_v * self.dt
        a_unsat = self.kp_v * e_v + self.ki_v * new_integral
        a = np.clip(a_unsat, self.limits.a_min, self.limits.a_max)

        # anti-windup: 只在"控制未饱和" 或 "饱和但积分项让控制往饱和反方向走" 时积分
        # 简化版 clamping: 如果饱和, 不更新积分
        if a == a_unsat:
            self._integral_v = new_integral
        # else: 积分保持上一拍值

        u = np.array([a, delta])
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
    pid = PIDController(vm)

    state = np.array([18.0, 0.0, np.pi / 2, 4.0])
    idx = ref.find_nearest(state[0], state[1])
    u, info = pid.solve(state, ref, idx)
    print(f"PID 输出: a={u[0]:.3f}  delta={np.rad2deg(u[1]):.2f}°  "
          f"耗时 {info['solve_time']*1e6:.1f} μs")

    # 跑 5 步看积分能不能逐步累积 (起始 v=4, v_ref=5, e_v=1)
    for _ in range(5):
        u, _ = pid.solve(state, ref, idx)
    print(f"5 步后积分: {pid._integral_v:.3f} (应≈ 6 * 0.1 * 1 = 0.6)")
