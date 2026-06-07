import numpy as np


class ReferenceTrajectory:
    """
    参考轨迹类

    存储一条离散的参考路径，每个路径点包含：
        x,y     ：位置
        phi     ：航向角
        v       ：参考速度
        kappa   ：曲率（带符号，左转为正）
        s       ：累积弧长

        内部用shape（N，6）的numpy数组存储，列顺序为[x, y, phi, v, kappa, s]

        用法:
            ref = RefenceTrajectory
            ref.generate_circle(radius=20, v_vef=5.0)
            idx = ref.find_nearest(car_x, car_y)
            window = ref.get_reference_window(idx, N=20)
    """

    # 约定每列在points数组中的位置，避免魔法数字
    IDX_X, IDX_Y, IDX_PHI, IDX_V, IDX_KAPPA, IDX_S = 0, 1, 2, 3, 4, 5
    N_COLS = 6

    def __init__(self, L=2.5):
        self.points = None  # shape(N, 6)
        self.L = L  # 用于反推参考前轮转角

    # ==========================================================================
    # 生成参考轨迹
    # ==========================================================================

    def generate_straight_line(self, length=50.0, v_ref=5.0, ds=0.1):
        """
        生成一条沿x轴直线参考轨迹，从（0,0）到（length，0）
        参数:
            length  : 参考轨迹长度（m）
            v_ref   : 参考速度（m/s）
                      - 标量    : 全程恒速 (向后兼容)
                      - callable: f(s) -> v_array, s 为弧长 ndarray
                        典型用法: 加速-巡航-制动梯形剖面
            ds      : 弧长采样间距（m）
        """

        N = int(length / ds) + 1
        s = np.linspace(0, length, N)

        points = np.zeros((N, self.N_COLS))
        points[:, self.IDX_X] = s
        points[:, self.IDX_Y] = 0.0
        points[:, self.IDX_PHI] = 0.0
        # v_ref 支持标量或 callable, 后者用于变速场景 (沿 s 的速度剖面)
        if callable(v_ref):
            points[:, self.IDX_V] = v_ref(s)
        else:
            points[:, self.IDX_V] = v_ref
        points[:, self.IDX_KAPPA] = 0.0
        points[:, self.IDX_S] = s

        self.points = points
        return self

    def generate_circle(self, radius=20.0, v_ref=5.0, ds=0.1, n_lap=1.0):
        """
        生成一个以原点为圆心的圆 (从 (radius, 0) 出发，逆时针)
        参数:
            radius  : 参考轨迹半径（m）
            v_ref   : 参考速度（m/s）
            ds      : 时间间隔（m）
            n_lap   : 圈数
        """
        total_s = 2 * np.pi * radius * n_lap
        N = int(total_s / ds) + 1
        s = np.linspace(0, total_s, N)

        theta = s / radius      # 圆心角

        points = np.zeros((N, self.N_COLS))
        points[:, self.IDX_X] = radius * np.cos(theta)
        points[:, self.IDX_Y] = radius * np.sin(theta)
        points[:, self.IDX_PHI] = theta + np.pi / 2
        points[:, self.IDX_V] = v_ref
        points[:, self.IDX_KAPPA] = 1.0 / radius
        points[:, self.IDX_S] = s

        self.points = points
        return self

    def generate_from_waypoints(self, waypoints, v_ref=5.0, ds=0.1,
                                smooth_window=2.0):
        """
        从一系列点生成参考轨迹,用线性插值 + 数值差分算航向和曲率

        参数:
            waypoints     : 参考点列表
            v_ref         : 参考速度 (m/s)，支持标量或 f(s) callable
            ds            : 弧长重采样间距 (m)
            smooth_window : 曲率平滑窗口 (m); <=0 表示不平滑。
                            折线 waypoint 在拐角处的二阶差分 kappa 会出尖刺,
                            喂给 MPC 当 delta_ref 时反而把控制带偏。这里用一个
                            汉宁窗滑动平均把毛刺抹平。

        流程:
            1. 累积弧长
            2. 按 ds 等间距重采样 x, y
            3. 一阶差分得 phi
            4. 二阶差分得 kappa, 再做窗口平滑
        """
        waypoints = np.asarray(waypoints, dtype=float)

        # 1. 原始路点的累积弧长
        seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        s_raw = np.concatenate([[0.0], np.cumsum(seg)])
        total_s = s_raw[-1]

        # 2. 按 ds 重采样
        N = int(total_s / ds) + 1
        s = np.linspace(0, total_s, N)
        x = np.interp(s, s_raw, waypoints[:, 0])
        y = np.interp(s, s_raw, waypoints[:, 1])

        # 3. 航向: 中心差分
        dx = np.gradient(x, s)
        dy = np.gradient(y, s)
        phi = np.arctan2(dy, dx)

        # 4. 曲率: kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
        ddx = np.gradient(dx, s)
        ddy = np.gradient(dy, s)
        kappa = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

        # 4.5 平滑 kappa: 汉宁窗滑动平均, 反射边界避免端点偏移
        if smooth_window > 0:
            win_len = max(3, int(smooth_window / ds) | 1)  # 强制奇数
            if win_len < N:
                window = np.hanning(win_len)
                window /= window.sum()
                pad = win_len // 2
                # 反射填充, 卷积后裁回原长
                kappa_padded = np.concatenate([
                    kappa[pad:0:-1], kappa, kappa[-2:-pad - 2:-1]
                ])
                kappa = np.convolve(kappa_padded, window, mode='valid')

        points = np.zeros((N, self.N_COLS))
        points[:, self.IDX_X] = x
        points[:, self.IDX_Y] = y
        points[:, self.IDX_PHI] = phi
        if callable(v_ref):
            points[:, self.IDX_V] = v_ref(s)
        else:
            points[:, self.IDX_V] = v_ref
        points[:, self.IDX_KAPPA] = kappa
        points[:, self.IDX_S] = s

        self.points = points
        return self

    # ============================================================
    # 查询接口
    # ============================================================

    def __len__(self):
        return 0 if self.points is None else len(self.points)

    def find_nearest(self, x, y, search_from=0, search_window=None):
        """
        找到距离 (x, y) 最近的参考点索引

        参数：
            x, y           : 车辆当前位置
            search_from    : 从哪个索引开始搜索（上一时刻的最近点，用于加速）
            search_window  : 只在 [search_from, search_from + search_window] 内搜索
                             None 表示全范围搜索

        注意：
            实时控制时强烈建议传 search_from 和 search_window，
            避免车辆绕回后被"抄近路"锁定到错误的参考点。
        """
        assert self.points is not None, "请先生成轨迹"

        N = len(self.points)
        # 边界保护:search_from 已经越界就直接返回末尾点,
        # 否则下面的切片会得到空数组, np.argmin 抛 ValueError
        if search_from >= N:
            return N - 1

        if search_window is None:
            candidates = self.points[search_from:]
            offset = search_from
        else:
            end = min(search_from + search_window, N)
            candidates = self.points[search_from:end]
            offset = search_from

        # 极端情况:窗口长度 0, 退化为返回 search_from(已知 < N)
        if len(candidates) == 0:
            return search_from

        dx = candidates[:, self.IDX_X] - x
        dy = candidates[:, self.IDX_Y] - y
        dist_sq = dx ** 2 + dy ** 2
        local_idx = int(np.argmin(dist_sq))
        return offset + local_idx

    def get_reference_window_by_time(self, idx_start, dt, N, fallback_v_end=0.0):
        """
        从索引 idx_start 处的参考点开始,按"未来 N 个 dt 时间步"取参考窗口

        ⚠ 给 MPC 用的就是这个方法,而不是 get_reference_window!

        关键:参考轨迹是按弧长 ds 等距离散的(空间步长),
        而 MPC 是按 dt 时间步推进的(时间步长)。两者不是一回事:
            - 直接按 idx 取窗口 → 相邻两点距离是 ds (例如 0.1m)
            - 按时间取窗口      → 相邻两点距离是 v_ref*dt (例如 0.8m)
        如果用错,MPC 看到的"未来 1.5 秒"只覆盖了参考轨迹前 1.5m,
        导致末端位置和速度互相打架,MPC 会被迫减速/失稳。

        实现:用前向欧拉沿弧长推进
            s_{k+1} = s_k + v_ref(s_k) * dt
        然后在累积弧长 s 上做线性插值,不要求 v*dt 是 ds 的整数倍。

        参数:
            idx_start       : 起始参考点索引 (通常是 find_nearest 的返回)
            dt              : MPC 时间步长
            N               : 取多少个点
            fallback_v_end  : 越过参考轨迹末端后,填充段的速度
                              默认 0 表示"到终点该停了"

        返回:
            window : shape (N, 6) 的参考状态序列 [x, y, phi, v, kappa, s]
                     window[k] 表示车 k 个 dt 之后应到达的状态
        """
        assert self.points is not None, "请先生成轨迹"
        assert 0 <= idx_start < len(self.points), \
            f"idx_start={idx_start} 越界 [0, {len(self.points)})"

        s_all = self.points[:, self.IDX_S]
        v_all = self.points[:, self.IDX_V]
        s_max = s_all[-1]

        # 1. 沿弧长前向欧拉, 算每个 stage 的目标 s
        s_targets = np.zeros(N)
        s_cur = s_all[idx_start]
        for k in range(N):
            s_targets[k] = s_cur
            # 当前点的参考速度 (越过末端就用末端的)
            v_at_cur = np.interp(min(s_cur, s_max), s_all, v_all)
            s_cur += v_at_cur * dt

        # 2. 在 s 上做线性插值得到状态
        s_clip = np.clip(s_targets, 0.0, s_max)
        window = np.zeros((N, self.N_COLS))
        for col in [self.IDX_X, self.IDX_Y, self.IDX_V,
                    self.IDX_KAPPA, self.IDX_S]:
            window[:, col] = np.interp(s_clip, s_all, self.points[:, col])
        # phi 单独 unwrap, 避免在 ±pi 跳变处插值出问题 (圆轨迹必须)
        phi_unwrapped = np.unwrap(self.points[:, self.IDX_PHI])
        window[:, self.IDX_PHI] = np.interp(s_clip, s_all, phi_unwrapped)

        # 3. 越过末端的点 → 速度置为 fallback (默认 0,表示停车)
        overshoot = s_targets > s_max
        window[overshoot, self.IDX_V] = fallback_v_end

        return window



    def get_reference_window(self, idx, N):
        """
        从索引 idx 开始取 N 个参考点，供 MPC 预测时域使用
        如果到达轨迹末尾，用最后一个点填充（让车停在终点）

        返回：shape (N, 6) 的数组
        """
        assert self.points is not None, "请先生成轨迹"

        end = idx + N
        if end <= len(self.points):
            return self.points[idx:end].copy()
        else:
            # 末尾填充：保持最后一个点
            window = np.zeros((N, self.N_COLS))
            n_valid = len(self.points) - idx
            window[:n_valid] = self.points[idx:]
            window[n_valid:] = self.points[-1]
            # 填充段速度置 0，表示到终点该停了
            window[n_valid:, self.IDX_V] = 0.0
            return window

    def get_reference_control(self, idx):
        """
        反推索引 idx 处的参考控制量 (a_ref, delta_ref)

        推导：
            自行车模型：phi_dot = v / L * tan(delta)
            又 phi_dot = v * kappa
            所以 delta = arctan(L * kappa)

            参考加速度：由于我们用的是恒定 v_ref，所以 a_ref = 0
            如果以后改成变速参考，这里要改成 a = v * dv/ds

        返回：np.array([a_ref, delta_ref])
        """
        assert self.points is not None, "请先生成轨迹"

        kappa = self.points[idx, self.IDX_KAPPA]
        delta_ref = np.arctan(self.L * kappa)
        a_ref = 0.0
        return np.array([a_ref, delta_ref])


    def get_reference_state(self, idx):
        """返回索引 idx 处的参考状态 (x, y, phi, v)"""
        assert self.points is not None, "请先生成轨迹"
        p = self.points[idx]
        return np.array([p[self.IDX_X], p[self.IDX_Y],
                         p[self.IDX_PHI], p[self.IDX_V]])





# ================================================================
# 单元测试 & 可视化
# ================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---------- 测试 1: 直线 ----------
    ref1 = ReferenceTrajectory().generate_straight_line(length=30, v_ref=5.0)
    axes[0].plot(ref1.points[:, 0], ref1.points[:, 1], 'b-')
    axes[0].set_title(f'直线 (N={len(ref1)})')
    axes[0].axis('equal'); axes[0].grid(True)

    # ---------- 测试 2: 圆 ----------
    ref2 = ReferenceTrajectory().generate_circle(radius=15, v_ref=5.0)
    axes[1].plot(ref2.points[:, 0], ref2.points[:, 1], 'r-')
    # 每隔一段画个航向箭头，肉眼验证 phi 对不对
    step = len(ref2) // 20
    for i in range(0, len(ref2), step):
        x, y, phi = ref2.points[i, :3]
        axes[1].arrow(x, y, 1.5 * np.cos(phi), 1.5 * np.sin(phi),
                      head_width=0.4, color='k', alpha=0.6)
    axes[1].set_title(f'圆 r=15 (N={len(ref2)})')
    axes[1].axis('equal'); axes[1].grid(True)

    # ---------- 测试 3: 双移线 (经典换道测试) ----------
    waypoints = np.array([
        [0, 0], [10, 0], [20, 3.5], [30, 3.5],
        [40, 0], [60, 0]
    ])
    ref3 = ReferenceTrajectory().generate_from_waypoints(waypoints, ds=0.1)
    axes[2].plot(ref3.points[:, 0], ref3.points[:, 1], 'g-', label='重采样轨迹')
    axes[2].plot(waypoints[:, 0], waypoints[:, 1], 'ko', label='路径点')
    axes[2].set_title(f'双移线 (N={len(ref3)})')
    axes[2].axis('equal'); axes[2].grid(True); axes[2].legend()

    plt.tight_layout(); plt.show()

    # ---------- 测试 4: 查询接口 ----------
    print("\n=== 查询接口测试 ===")
    car_x, car_y = 14.0, 2.0
    idx = ref3.find_nearest(car_x, car_y)
    print(f"车辆位置 ({car_x}, {car_y})")
    print(f"最近点索引: {idx}")
    print(f"最近点坐标: ({ref3.points[idx, 0]:.2f}, {ref3.points[idx, 1]:.2f})")
    print(f"参考状态: {ref3.get_reference_state(idx)}")
    print(f"参考控制 (a_ref, delta_ref): {ref3.get_reference_control(idx)}")

    # ---------- 测试 5: 预测窗口 ----------
    window = ref3.get_reference_window(idx, N=20)
    print(f"\n预测窗口 shape: {window.shape}")
    print(f"窗口首点: {window[0, :2]}")
    print(f"窗口末点: {window[-1, :2]}")

    # ---------- 测试 6: 曲率画出来看看 ----------
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ref3.points[:, 5], ref3.points[:, 4])
    ax2.set_xlabel('s (m)'); ax2.set_ylabel('曲率 (1/m)')
    ax2.set_title('双移线轨迹沿弧长的曲率')
    ax2.grid(True)
    plt.show()
