"""
仿真场景定义。

用 dataclass 表达一个场景的所有可配置项, 后面阶段加新字段(障碍/EKF/目标点)
时, 老场景用默认值即可, 不用改函数签名。
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from core.reference_trajectory import ReferenceTrajectory


@dataclass
class Scenario:
    """
    一个仿真场景的完整描述。

    字段:
        name        : 场景标识 (用于文件名)
        title       : 图标题
        ref         : 参考轨迹
        init_state  : 车辆初始状态 [x, y, phi, v]
        obstacles   : 障碍物列表 (阶段 3 引入, 现在留空)
        enable_ekf  : 是否启用状态估计 (阶段 2 引入)
        goal        : 目标点 (阶段 4 A* 用)
    """
    name: str
    title: str
    ref: ReferenceTrajectory
    init_state: np.ndarray
    obstacles: List = field(default_factory=list)
    enable_ekf: bool = False
    goal: Optional[np.ndarray] = None


# ==================================================================
# 内置场景
# ==================================================================

def straight_scene() -> Scenario:
    """直线 40m, 起点偏离 2m + 航向偏 10° + 速度低 5m/s"""
    ref = ReferenceTrajectory().generate_straight_line(
        length=40.0, v_ref=8.0, ds=0.1)
    return Scenario(
        name='line',
        title='Straight line tracking',
        ref=ref,
        init_state=np.array([0.0, 2.0, np.deg2rad(10), 3.0]),
    )


def circle_scene() -> Scenario:
    """圆轨迹 R=15m, v=6m/s, 车从圆内起步"""
    ref = ReferenceTrajectory().generate_circle(
        radius=15.0, v_ref=6.0, ds=0.1, n_lap=1.0)
    return Scenario(
        name='circle',
        title='Circle tracking (R=15m, v=6m/s)',
        ref=ref,
        init_state=np.array([13.0, 1.0, np.pi / 2, 0.0]),
    )


def lane_change_scene() -> Scenario:
    """经典双移线: 变道超车"""
    waypoints = np.array([
        [0, 0], [15, 0], [25, 3.5], [45, 3.5],
        [55, 0], [80, 0],
    ])
    ref = ReferenceTrajectory().generate_from_waypoints(
        waypoints, v_ref=10.0, ds=0.1)
    return Scenario(
        name='lane',
        title='Double lane change  (v_ref=10m/s)',
        ref=ref,
        init_state=np.array([0.0, 0.3, 0.0, 8.0]),
    )


def accel_cruise_brake_scene() -> Scenario:
    """
    变速直线: 梯形速度剖面 (加速 → 巡航 → 减速 → 慢速)
        s ∈ [0,   20m] : v 从 V_MIN 线性升到 V_MAX
        s ∈ [20,  60m] : v = V_MAX (巡航)
        s ∈ [60,  80m] : v 从 V_MAX 线性降到 V_MIN

    为什么不是 0 → 12 → 0:
        参考窗口是按时间 (s_cur += v*dt) 推进的, v=0 时 s_cur 不动,
        整个 N+1 预测窗口塌缩到同一点 → MPC 看到静止参考 → 死锁。
        用 V_MIN=1.0 m/s 模拟"creep speed", 既保留减速测试又避免数值边界。

    设计意图:
        - 直线场景里"速度跟踪"是真正在工作的, 而不是稳态零误差
        - 减速段考验 anti-windup (积分项不能在饱和时继续涨)
        - MPC 的预瞄能"看到"将要减速 → 提前松油门; PID/LQR 反应式
    """
    LENGTH = 80.0
    V_MAX = 12.0
    V_MIN = 1.0
    S_ACCEL_END = 20.0
    S_BRAKE_START = 60.0

    def v_profile(s):
        v = np.where(
            s < S_ACCEL_END,
            V_MIN + (V_MAX - V_MIN) * s / S_ACCEL_END,
            np.where(
                s < S_BRAKE_START,
                V_MAX,
                V_MAX - (V_MAX - V_MIN) * (s - S_BRAKE_START)
                                          / (LENGTH - S_BRAKE_START),
            ),
        )
        return np.maximum(v, V_MIN)

    ref = ReferenceTrajectory().generate_straight_line(
        length=LENGTH, v_ref=v_profile, ds=0.1)
    return Scenario(
        name='accel_brake',
        title='Accel-Cruise-Decel straight (v: 1 → 12 → 1 m/s)',
        ref=ref,
        # 起点在路径上, 初速 = V_MIN 匹配 ref 起始, 聚焦纵向控制
        init_state=np.array([0.0, 0.0, 0.0, V_MIN]),
    )


def serpentine_scene() -> Scenario:
    """
    S 曲线蛇形: 周期 30m、振幅 ±2m 的连续正弦摆动 v=8 m/s

    设计意图:
        - 高频转向输入, 暴露控制器的平滑性差异
        - MPC 的 Rd (控制增量惩罚) → 转角光滑
        - PID/Stanley 是反应式 → 转角抖
        - 同时是连续曲率切换 (左/右), 比单一圆更真实
    """
    AMP = 2.0
    PERIOD = 30.0
    LENGTH = 80.0
    N_SAMPLES = 200

    x = np.linspace(0, LENGTH, N_SAMPLES)
    y = AMP * np.sin(2 * np.pi * x / PERIOD)
    waypoints = np.column_stack([x, y])

    ref = ReferenceTrajectory().generate_from_waypoints(
        waypoints, v_ref=8.0, ds=0.1)
    return Scenario(
        name='serpentine',
        title='S-curve serpentine (period=30m, amp=2m, v=8m/s)',
        ref=ref,
        # 起点轻微偏离 + 起步速度低于参考, 综合考验
        init_state=np.array([0.0, 0.3, 0.0, 6.0]),
    )


def astar_scene() -> Scenario:
    """
    A* 全局规划 + B-spline 平滑 → 参考轨迹

    工作区 80m x 30m, 10 个不同半径的圆障碍组成障碍场。靠近直线路径的
    3 个关键障碍让 A* 连续绕行, 其余障碍限定可行空间并丰富规划场景。
    考虑 r_inflate=1.55m + 平滑后曲率限制, 关键障碍之间保留足够间距,
    避免 B-spline 平滑后超过 |kappa|<0.23 1/m 的车辆极限。

    pipeline:
        OccupancyGrid → astar → smooth_path → generate_from_waypoints → MPC

    设计意图:
        - 验收 ROADMAP 阶段 4: 端到端规划+控制闭环可演示
        - 障碍是规划层处理的, MPC 此时还看不到障碍 (阶段 3 才加避障约束),
          所以参考轨迹必须先把障碍绕开
    """
    from planning import OccupancyGrid, astar, smooth_path

    # 工作区 + 障碍
    obstacles = [
        (12.0, -8.5, 1.8),
        (20.0,  3.0, 2.0),
        (24.0,  9.5, 2.2),
        (33.0, -9.5, 2.1),
        (40.0, -3.0, 2.5),
        (47.0,  9.5, 1.9),
        (56.0, -9.0, 2.2),
        (60.0,  4.0, 2.0),
        (70.0,  9.0, 2.0),
        (76.0, -8.5, 1.7),
    ]
    gm = OccupancyGrid(x_min=-5, x_max=85, y_min=-15, y_max=15,
                       resolution=0.5)
    R_INFLATE = 1.55   # L/2 (1.25) + margin (0.3)
    gm.add_circles(obstacles, r_inflate=R_INFLATE)

    # A* + smooth
    start_xy = (0.0, 0.0)
    goal_xy = (80.0, 0.0)
    raw_path, info_a = astar(gm, start_xy, goal_xy)
    if raw_path is None:
        raise RuntimeError(f"A* 失败: {info_a['reason']}")
    # smooth_factor=20: 经验值. 默认值 (=M, M=路径点数, 这里 ~160) 平滑过头,
    # 偏离 A* 太多 → 撞 inflated 边界. sf=20 让曲率刚好低于车辆极限 0.23 1/m,
    # 经 generate_from_waypoints 的二次汉宁平滑后 κ_max ≈ 0.218 < 0.231。
    smooth_xy, info_s = smooth_path(raw_path, ds=0.3, smooth_factor=20,
                                     grid_map=gm)

    # 喂给 ReferenceTrajectory (会再做一次 ds=0.1 重采样 + 汉宁平滑)
    ref = ReferenceTrajectory().generate_from_waypoints(
        smooth_xy, v_ref=6.0, ds=0.1)

    # 把规划元数据塞进场景, 给可视化用
    sc = Scenario(
        name='astar',
        title=f'A* + B-spline 全局规划  ({len(obstacles)} 障碍)',
        ref=ref,
        init_state=np.array([0.0, 0.0, 0.0, 0.0]),
        obstacles=[(cx, cy, r) for (cx, cy, r) in obstacles],
        goal=np.array([80.0, 0.0]),
    )
    # 附加属性 (Scenario dataclass 没定义这些, 用 setattr 即可)
    sc.grid_map = gm
    sc.raw_astar_path = raw_path
    sc.smoothed_path = smooth_xy
    sc.r_inflate = R_INFLATE
    sc.astar_info = info_a
    sc.smooth_info = info_s
    return sc


def oncoming_vehicle_scene() -> Scenario:
    """
    阶段 5 demo 之一: 直线 + 对向车迎面.

    ego: (0, 0), v=8m/s 沿 +x
    obs: (50, 0), 恒速 (-5, 0) m/s, r=1.0   (closing speed 13m/s, T2C ~3.8s)

    设计意图:
        - 验证 MPC 用 obs.predict(k*dt) 提前看到对向车而不是用静态当前位置
        - 不确定性锥 α=0.5 让远期 r_safe 膨胀, 给 swerve 更多余量
        - 预期: ego 提前左变道避开, swerve 后回到 y≈0
    """
    ref = ReferenceTrajectory().generate_straight_line(
        length=80.0, v_ref=8.0, ds=0.1)
    sc = Scenario(
        name='oncoming',
        title='Oncoming vehicle (动态避障, 对向车 -5 m/s)',
        ref=ref,
        init_state=np.array([0.0, 0.0, 0.0, 6.0]),
        # (cx, cy, vx, vy, r) → DynamicObstacle
        obstacles=[(50.0, 0.3, -5.0, 0.0, 1.0)],
    )
    sc.alpha_uncert = 0.5    # 不确定性锥 0.5 m/s
    return sc


def crossing_pedestrian_scene() -> Scenario:
    """
    阶段 5 demo 之二: 直线 + 行人横穿.

    ego: (0, 0), v=6m/s 沿 +x
    obs: (25, -6), 恒速 (0, 1) m/s, r=0.4   (行人比车小, 慢)

    设计意图:
        - ego 直行 → 6 秒到 x=36; 行人 6 秒后到 y=0, x=25 → 错开了
          但行人在 ~4 秒时到 y=-2, ego 在 x=24 处, 距离 d=√(1+4)≈2.2 < r_safe=1.7m
          → 必须减速等行人过去, 或微微 swerve
        - 与对向车不同: 这里"减速"是更自然的 swerve 也行的反应
        - α=0.3 (行人不像车, 更难预测但速度慢, 锥小一点)
    """
    ref = ReferenceTrajectory().generate_straight_line(
        length=50.0, v_ref=6.0, ds=0.1)
    sc = Scenario(
        name='crossing',
        title='Crossing pedestrian (动态避障, 行人横穿 +1 m/s)',
        ref=ref,
        init_state=np.array([0.0, 0.0, 0.0, 4.0]),
        obstacles=[(25.0, -6.0, 0.0, 1.0, 0.4)],
    )
    sc.alpha_uncert = 0.3
    return sc


def obstacle_blocking_scene() -> Scenario:
    """
    参考是直线, 障碍**靠近路径中央** —— MPC 必须自己绕 (上层规划没救场)。

    设计意图:
        - 阶段 3 的"硬核 demo": 没有 A* 帮忙, 只能靠 MPC 的避障约束
        - 验收: min(dist_to_obs) >= r_safe AND 车的 y 偏移峰值 > 0 (说明真的绕了)

    为什么障碍偏 0.5m 而不是正中:
        B 半空间方案在"warm-start 与障碍距离 >> obstacle radius"时, 线性化把
        圆障碍近似成一个平面, 这个平面的法向几乎只有 -x 分量 (车-障连线方向).
        如果车也在 y=0 直线上前进, 法向就完全没有 y 分量, QP 找不到"侧让"的
        梯度 → 局部极小, MPC 只能减速到停在切平面前。
        把障碍稍偏 0.5m, 法向有微弱 y 分量, 给 QP 一个明确的避让方向 → 顺利绕开。
        (实际工程里这种"破对称"由上游 planner 提供, 这里手动安排。)

    场景配置:
        参考: x ∈ [0, 40] 直线, v_ref=6m/s
        障碍: (22, 0.5, 1.5)  靠近 y=0 但偏一点 → 自然让车走 y<0 一侧
        起点: (0, 0, 0, 4)
    """
    ref = ReferenceTrajectory().generate_straight_line(
        length=40.0, v_ref=6.0, ds=0.1)
    sc = Scenario(
        name='block',
        title='Single obstacle blocking (MPC alone avoids)',
        ref=ref,
        init_state=np.array([0.0, 0.0, 0.0, 4.0]),
        obstacles=[(22.0, 0.5, 1.5)],
    )
    return sc


# 名字 -> 工厂函数, main.py / compare.py 用这个查表
SCENE_REGISTRY = {
    'line':        straight_scene,
    'circle':      circle_scene,
    'lane':        lane_change_scene,
    'accel_brake': accel_cruise_brake_scene,
    'serpentine':  serpentine_scene,
    'astar':       astar_scene,
    'block':       obstacle_blocking_scene,
    'oncoming':    oncoming_vehicle_scene,
    'crossing':    crossing_pedestrian_scene,
}
