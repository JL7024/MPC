"""
扩展卡尔曼滤波器 (EKF) for 自行车运动学模型

状态: x = [x, y, phi, v]^T
过程: x_{k+1} = f(x_k, u_k) + w,   w ~ N(0, Q)

测量 (按到时执行 update):
    GPS:   z_gps   = H_gps   @ x + v_gps,   v_gps ~ N(0, R_gps)
           H_gps   = [[1,0,0,0],[0,1,0,0]]
    轮速:  z_wheel = H_wheel @ x + v_wheel, v_wheel ~ N(0, R_wheel)
           H_wheel = [[0,0,0,1]]

关键实现要点:
    1. predict 用真非线性 f (vehicle_model.step) 推均值, 用 Jacobian F = A_d
       推协方差。这是 EKF 的标准做法 (mean by nonlinear, cov by linearization)。
    2. F 在当前估计 x_hat 处取, 不在参考点取——这是 EKF vs LQR 的核心区别。
    3. phi 误差 wrap: 残差 y = z - h(x) 没有 phi (传感器都不直接测 phi),
       但 state.phi 仍要周期性归一化, 防止跨 ±π 时数值漂移。
    4. P 数值上保持对称: 每次 update 后 P = (P + P.T) / 2 (Joseph form 也行,
       但对称化更便宜)。
    5. 协方差初值 P0 给大 = "我不太确定起点", EKF 会快速被测量纠正。

为什么 Q != 0 即使仿真无过程噪声:
    Q 表达"对模型的不信任", 不是"实际过程的扰动"。
    如果 Q = 0, EKF 100% 信 predict, 测量被忽略 → 退化成开环积分。
    Q 越大, 测量权重越大 (对应 Kalman gain K 增大)。
    所以 Q 是"调参"参数, 不是物理量。
"""

import numpy as np


def _wrap_to_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


class ExtendedKalmanFilter:
    """
    自行车模型 EKF, 支持多种测量类型 (GPS / 轮速 / ...) 异步 update。

    用法:
        ekf = ExtendedKalmanFilter(vehicle_model, x0, P0, Q)
        # 主循环每步:
        ekf.predict(u_last)               # 用上一拍施加的 u 预测
        if measurements.gps is not None:
            ekf.update_gps(z_gps, R_gps)  # GPS 来了就 update
        if measurements.wheel is not None:
            ekf.update_wheel(z_wheel, R_wheel)
        x_est = ekf.x                     # 控制器拿这个
    """

    NX = 4

    def __init__(self, vehicle_model, x0, P0, Q):
        """
        参数:
            vehicle_model : VehicleModel 实例 (复用 step 和 linearize)
            x0            : (4,)  初始状态估计
            P0            : (4,4) 初始协方差
            Q             : (4,4) 过程噪声协方差
        """
        self.vm = vehicle_model
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.Q = np.asarray(Q, dtype=float).copy()

    # =====================================================================
    # 预测
    # =====================================================================

    def predict(self, u):
        """
        x_hat_{k+1|k}  = f(x_hat_{k|k}, u_k)         (用真非线性)
        P_{k+1|k}      = F P F^T + Q                  (用 Jacobian)
        其中 F = ∂f/∂x |_{x_hat, u}  (在 *当前估计* 处线性化, 不在 ref)
        """
        u = np.asarray(u, dtype=float)
        # F 由 vehicle_model.linearize 提供 (它的 A_d 就是 F)
        # 注意: linearize 还返回 c_d, 但 mean propagation 用真 step 就行
        F, _, _ = self.vm.linearize(self.x, u)
        # 均值用真非线性
        self.x = self.vm.step(self.x, u)
        self.x[2] = _wrap_to_pi(self.x[2])
        # 协方差用线性化
        self.P = F @ self.P @ F.T + self.Q
        # 数值对称化 (浮点累积非对称会让后面 inv 出问题)
        self.P = 0.5 * (self.P + self.P.T)

    # =====================================================================
    # 更新 (每个传感器一个方法)
    # =====================================================================

    def _update(self, z, H, R):
        """
        通用 EKF update:
            y = z - H x       (innovation, 残差)
            S = H P H^T + R   (innovation covariance)
            K = P H^T S^{-1}
            x ← x + K y
            P ← (I - K H) P
        """
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        # solve 比 inv 更稳: K = P H^T S^{-1}  ⇔  K S^T = P H^T  (S 对称)
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = self.x + K @ y
        self.x[2] = _wrap_to_pi(self.x[2])
        I = np.eye(self.NX)
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def update_gps(self, z, R):
        """z: (2,) [x, y] 测量, R: (2,2)"""
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        self._update(np.asarray(z, dtype=float), H, np.asarray(R, dtype=float))

    def update_wheel(self, z, R):
        """z: (1,) [v] 测量, R: (1,1)"""
        H = np.array([[0.0, 0.0, 0.0, 1.0]])
        self._update(np.asarray(z, dtype=float).reshape(1),
                     H, np.asarray(R, dtype=float).reshape(1, 1))

    # =====================================================================
    # 便利方法
    # =====================================================================

    def state(self):
        """返回当前估计 (拷贝)"""
        return self.x.copy()

    def cov_diag(self):
        """返回协方差对角线 (4,) — 各分量的方差, 开根号是 1σ"""
        return np.diag(self.P).copy()


# ========================================================================
# 自测: 单步 predict + GPS update, 看 K 是否合理
# ========================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.vehicle_model import VehicleModel

    vm = VehicleModel(L=2.5, dt=0.1)
    x_truth = np.array([0.0, 0.0, 0.0, 5.0])

    # EKF 初值故意偏 1m, 速度差 1 m/s
    x_init = np.array([1.0, 0.5, 0.05, 4.0])
    P0 = np.diag([1.0, 1.0, 0.1, 1.0])
    Q = np.diag([0.01, 0.01, 0.001, 0.05])

    ekf = ExtendedKalmanFilter(vm, x_init, P0, Q)
    R_gps = np.diag([0.5**2, 0.5**2])

    print(f"初始: 真值={x_truth}, 估计={ekf.x}")
    print(f"初始 P 对角: {ekf.cov_diag()}")
    print()

    u = np.array([0.0, 0.0])  # 直行
    for step in range(20):
        # 真值推进
        x_truth = vm.step(x_truth, u)
        # EKF predict
        ekf.predict(u)
        # 模拟 GPS (加噪声)
        rng = np.random.default_rng(step)
        z_gps = x_truth[:2] + rng.normal(0, 0.5, 2)
        ekf.update_gps(z_gps, R_gps)
        if step % 4 == 0:
            err = ekf.x - x_truth
            print(f"step {step:2d}: 真={x_truth.round(3)}  估={ekf.x.round(3)}  "
                  f"err={err.round(3)}  σ={np.sqrt(ekf.cov_diag()).round(3)}")

    print("\n应当看到: x/y 误差被 GPS 拽小, σ_x/σ_y 收敛到 ~0.3 (R_gps^0.5 量级)")
