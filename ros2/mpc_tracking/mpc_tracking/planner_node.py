"""
规划节点: 生成参考轨迹并发布。

scenario=astar 时真正做规划: 用收到的静态障碍建占据栅格 → A* → B-spline 平滑
→ ReferenceTrajectory 等弧长重采样。流程与 scenarios.astar_scene 相同, 参数在 yaml 里。
其它场景没有规划层, 直接发布场景自带的参考轨迹。

发布:
    /planning/reference  自定义 ReferenceTrajectory (含速度和曲率, 给控制器)
    /planning/path       参考轨迹 nav_msgs/Path (给 RViz)
    /planning/raw_path   A* 原始折线 (给 RViz)
    /planning/grid       膨胀后的占据栅格 nav_msgs/OccupancyGrid (标准消息)
    /planning/grid_markers  同一张栅格画成方块 + 规划区域边框 (给 RViz)
    /planning/goal       目标点 Marker (给 RViz)
都用"锁存" QoS, 并每秒重发一次, 晚打开的 RViz 也能显示。

为什么栅格要另外画成 Marker: WSLg 的显卡驱动下, RViz 的 Map 显示项着色器链接失败,
地图显示不出来; 用 Marker 方块画不依赖那个着色器。
"""

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from mpc_tracking import _paths
from mpc_tracking.common import QOS_LATCHED, make_path, run_node, steady_clock
from mpc_tracking_interfaces.msg import ObstacleArray
from mpc_tracking_interfaces.msg import ReferenceTrajectory as ReferenceTrajectoryMsg

_paths.setup()
from core.reference_trajectory import ReferenceTrajectory  # noqa: E402
from planning import OccupancyGrid as GridMap, astar, smooth_path  # noqa: E402
from scenarios import SCENE_REGISTRY  # noqa: E402


class PlannerNode(Node):

    def __init__(self):
        super().__init__('planner_node')
        self.scenario = self.declare_parameter('scenario', 'astar').value
        self.wheelbase = self.declare_parameter('wheelbase', 2.5).value
        self.frame_id = self.declare_parameter('frame_id', 'map').value
        # 以下只在 A* 规划时使用, 默认值与 scenarios.astar_scene 一致
        self.bounds = [self.declare_parameter(n, v).value for n, v in
                       (('x_min', -5.0), ('x_max', 85.0), ('y_min', -15.0), ('y_max', 15.0))]
        self.resolution = self.declare_parameter('resolution', 0.5).value
        self.r_inflate = self.declare_parameter('r_inflate', 1.55).value
        self.start = list(self.declare_parameter('start', [0.0, 0.0]).value)
        self.goal = list(self.declare_parameter('goal', [80.0, 0.0]).value)
        self.smooth_ds = self.declare_parameter('smooth_ds', 0.3).value
        self.smooth_factor = self.declare_parameter('smooth_factor', 20.0).value
        self.ref_ds = self.declare_parameter('ref_ds', 0.1).value
        self.v_ref = self.declare_parameter('v_ref', 6.0).value

        self.pub_ref = self.create_publisher(ReferenceTrajectoryMsg, '/planning/reference',
                                             QOS_LATCHED)
        self.pub_path = self.create_publisher(Path, '/planning/path', QOS_LATCHED)
        self.pub_raw = self.create_publisher(Path, '/planning/raw_path', QOS_LATCHED)
        self.pub_grid = self.create_publisher(OccupancyGrid, '/planning/grid', QOS_LATCHED)
        self.pub_grid_markers = self.create_publisher(MarkerArray, '/planning/grid_markers',
                                                      QOS_LATCHED)
        self.pub_goal = self.create_publisher(Marker, '/planning/goal', QOS_LATCHED)

        self.plan_id = 0
        self.outputs = []           # [(publisher, msg), ...] 最近一次规划的全部输出

        if self.scenario == 'astar':
            self.get_logger().info('A* 规划模式: 等待障碍物信息...')
            self.create_subscription(ObstacleArray, '/sim/obstacles', self._on_obstacles,
                                     QOS_LATCHED)
        else:
            scene = SCENE_REGISTRY[self.scenario]()
            self.get_logger().info(f'场景 {self.scenario} 没有规划层, 直接发布场景自带的参考轨迹')
            self._publish_plan(scene.ref)
        self.create_timer(1.0, self._republish, clock=steady_clock())

    def _on_obstacles(self, msg):
        if self.plan_id > 0:
            return      # 只在启动时规划一次; 障碍物后续更新不触发重规划
        circles = [(o.x, o.y, o.radius) for o in msg.obstacles
                   if o.vx == 0.0 and o.vy == 0.0]     # 动态障碍交给 MPC 处理
        gm = GridMap(*self.bounds, resolution=self.resolution)
        gm.add_circles(circles, r_inflate=self.r_inflate)
        raw, info_a = astar(gm, tuple(self.start), tuple(self.goal))
        if raw is None:
            self.get_logger().error(f'A* 失败: {info_a["reason"]}')
            return
        smooth_xy, info_s = smooth_path(raw, ds=self.smooth_ds,
                                        smooth_factor=self.smooth_factor, grid_map=gm)
        ref = ReferenceTrajectory(L=self.wheelbase).generate_from_waypoints(
            smooth_xy, v_ref=self.v_ref, ds=self.ref_ds)
        self.get_logger().info(
            f'A* 完成: {len(circles)} 个障碍, 展开 {info_a["expanded"]} 个节点, '
            f'用时 {info_a["time"] * 1000:.1f}ms, 折线 {len(raw)} 点; '
            f'平滑 {info_s["mode"]}; 参考轨迹 {len(ref)} 点, 长 {ref.points[-1, 5]:.1f}m')
        self._publish_plan(ref, raw_path=raw, grid=gm)

    def _publish_plan(self, ref, raw_path=None, grid=None):
        self.plan_id += 1
        stamp = Time()      # 规划结果画在固定的 map 坐标系里, 不需要时间戳
        pts = ref.points
        msg = ReferenceTrajectoryMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.plan_id = self.plan_id
        msg.wheelbase = float(ref.L)
        for name, col in (('x', ref.IDX_X), ('y', ref.IDX_Y), ('phi', ref.IDX_PHI),
                          ('v', ref.IDX_V), ('kappa', ref.IDX_KAPPA), ('s', ref.IDX_S)):
            setattr(msg, name, pts[:, col].astype(float).tolist())

        outputs = [(self.pub_ref, msg),
                   (self.pub_path, make_path(stamp, self.frame_id, pts[:, :2],
                                             pts[:, ref.IDX_PHI]))]
        if raw_path is not None:
            outputs.append((self.pub_raw, make_path(stamp, self.frame_id, raw_path)))
        if grid is not None:
            outputs.append((self.pub_grid, self._grid_msg(grid, stamp)))
            outputs.append((self.pub_grid_markers, self._grid_markers(grid, stamp)))
        outputs.append((self.pub_goal, self._goal_marker(pts[-1, :2], stamp)))
        self.outputs = outputs
        self._republish()

    def _republish(self):
        for pub, msg in self.outputs:
            pub.publish(msg)

    def _grid_msg(self, gm, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.info.resolution = float(gm.res)
        msg.info.width = gm.n_x
        msg.info.height = gm.n_y
        msg.info.origin.position.x = gm.x_min
        msg.info.origin.position.y = gm.y_min
        msg.info.origin.orientation.w = 1.0
        # grid[i, j]: i 是 y 方向行号, j 是 x 方向列号, 与 OccupancyGrid 的行优先排列一致
        msg.data = (gm.grid.astype(np.int8) * 100).ravel().tolist()
        return msg

    def _grid_markers(self, gm, stamp):
        cells = Marker()
        cells.header.stamp = stamp
        cells.header.frame_id = self.frame_id
        cells.ns = 'grid_cells'
        cells.type = Marker.CUBE_LIST
        cells.action = Marker.ADD
        cells.pose.orientation.w = 1.0
        cells.scale.x = cells.scale.y = float(gm.res)
        cells.scale.z = 0.02
        cells.color.r, cells.color.g, cells.color.b, cells.color.a = 0.6, 0.6, 0.65, 0.45
        ii, jj = np.nonzero(gm.grid)
        for i, j in zip(ii, jj):
            x, y = gm.grid_to_world(int(i), int(j))
            cells.points.append(Point(x=float(x), y=float(y), z=0.0))

        bounds = Marker()
        bounds.header = cells.header
        bounds.ns = 'grid_bounds'
        bounds.type = Marker.LINE_STRIP
        bounds.action = Marker.ADD
        bounds.pose.orientation.w = 1.0
        bounds.scale.x = 0.08
        bounds.color.r, bounds.color.g, bounds.color.b, bounds.color.a = 0.6, 0.6, 0.65, 0.8
        x0, x1, y0, y1 = gm.x_min, gm.x_max, gm.y_min, gm.y_max
        bounds.points = [Point(x=x, y=y, z=0.0)
                         for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
        return MarkerArray(markers=[cells, bounds])

    def _goal_marker(self, xy, stamp):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.frame_id
        m.ns = 'goal'
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(xy[0])
        m.pose.position.y = float(xy[1])
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 1.0     # 直径 1 m = main.run_simulation 的到达判定圈
        m.scale.z = 0.1
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.3, 0.8
        return m


def main(args=None):
    run_node(PlannerNode, args)


if __name__ == '__main__':
    main()
