import numpy as np


class VehicleModel:
    """
    自行车运动学模型

    状态量：
        车辆后轴中心位置x，y
        航向角phi
        纵向速度v

    控制量：
        纵向加速度a
        前轮转角delta
    """

    def __init__(self, L=2.5, dt=0.1, v_min=0.0, v_max=None):
        self.L = L
        self.dt = dt
        self.v_min = v_min
        self.v_max = v_max
        self.state = np.zeros(4)

    def reset(self, x=0.0, y=0.0, phi=0.0, v=0.0):
        # 重置车辆到指定初始状态
        self.state = np.array([x, y, phi, v])
        return self.state.copy()

    def step(self, state, u):
        """
        施加控制量 u,前进一个时间步 dt
        参数：
            当前状态，控制量
        返回：
            新的状态
        """

        x, y, phi, v = state
        a, delta = u

        x_next = x + v * np.cos(phi) * self.dt
        y_next = y + v * np.sin(phi) * self.dt
        phi_next = phi + v / self.L * np.tan(delta) * self.dt
        v_next = v + a * self.dt
        if self.v_min is not None:
            v_next = max(self.v_min, v_next)
        if self.v_max is not None:
            v_next = min(self.v_max, v_next)

        return np.array([x_next, y_next, phi_next, v_next])

    def linearize(self, state_ref, u_ref):
        """
        在参考点 (state_ref, u_ref) 处对离散动力学做一阶泰勒展开线性化。

        离散动力学形式: x_{k+1} ≈ A_d @ x_k + B_d @ u_k + c_d

        参数:
            state_ref: 参考状态 [px, py, phi, v], shape (4,)
            u_ref:     参考控制 [a, delta],       shape (2,)

        返回:
            A_d: 离散状态矩阵,     shape (4, 4)
            B_d: 离散控制矩阵,     shape (4, 2)
            c_d: 离散仿射项,       shape (4,)
        """

        _, _, phi, v = state_ref
        a, delta = u_ref
        L, dt = self.L, self.dt

        A_c = np.array([
            [0.0, 0.0, -v * np.sin(phi), np.cos(phi)],
            [0.0, 0.0, v * np.cos(phi), np.sin(phi)],
            [0.0, 0.0, 0.0, np.tan(delta) / L],
            [0.0, 0.0, 0.0, 0.0],
        ])

        B_c = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, v / (L * np.cos(delta) ** 2)],
            [1.0, 0.0],
        ])

        f_ref = np.array([
            v * np.cos(phi),
            v * np.sin(phi),
            v / L * np.tan(delta),
            a
        ])

        A_d = np.eye(4) +A_c * dt
        B_d = B_c * dt
        c_d = (f_ref - A_c @ state_ref - B_c @ u_ref) * dt

        return A_d, B_d, c_d


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    car = VehicleModel(L=2.5, dt=0.1)
    state = car.reset(x=0, y=0, phi=3.1, v=5.0)

    trajectory = [state]
    for _ in range(200):
        state = car.step(state, u=[0, 0.1])
        trajectory.append(state)


    trajectory = np.array(trajectory)

    plt.figure(figsize=(8, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-')
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.title('车辆轨迹测试')
    plt.show()

    # ===== 线性化正确性测试 =====
    # 思路:在参考点附近取一个小扰动,比较 "真实 step" 和 "线性化预测" 的差距
    # 扰动很小时,两者应该非常接近(差距是 O(扰动^2))

    state_ref = np.array([1.0, 2.0, 0.3, 5.0])
    u_ref = np.array([0.5, 0.1])

    A_d, B_d, c_d = car.linearize(state_ref, u_ref)

    # 情况 1: 就在参考点本身,线性化应该等于真实 step
    x_true_ref = car.step(state_ref, u_ref)
    x_lin_ref = A_d @ state_ref + B_d @ u_ref + c_d
    print("在参考点:  真实 =", x_true_ref)
    print("在参考点:  线性 =", x_lin_ref)
    print("差距(应接近 0):", np.linalg.norm(x_true_ref - x_lin_ref))

    # 情况 2: 小扰动,线性化应该很接近但不完全相等
    dx = np.array([0.01, 0.01, 0.01, 0.01])
    du = np.array([0.01, 0.01])
    x_true = car.step(state_ref + dx, u_ref + du)
    x_lin = A_d @ (state_ref + dx) + B_d @ (u_ref + du) + c_d
    print("\n小扰动:    真实 =", x_true)
    print("小扰动:    线性 =", x_lin)
    print("差距(应很小):", np.linalg.norm(x_true - x_lin))
