"""
仿真节点: 代替真车和真实世界, 并作为整个系统的时钟源 (/clock)。

每推进一步 (dt) 做的事和 main.run_simulation 的一拍相同:
    1. 用收到的控制指令 [a, delta] 推进车辆真值 (VehicleModel.step)
    2. 推进障碍物 (动态障碍按恒速模型移动, 静态不动)
    3. 用 SensorSuite 生成带噪声的 GPS (5 Hz) 和轮速 (10 Hz) 测量
然后发布: /clock、障碍物、车辆真值 (+TF)、GPS、轮速。
同一拍里先发 GPS 再发轮速, EKF 节点依赖这个顺序 (见 ekf_node)。

lockstep (默认开启):
    推进第 k+1 步之前, 先等"基于第 k 步状态算出的控制指令" (时间戳 = t_k)。
    这样闭环结果与求解耗时、进程调度无关, 每次运行都一样, 能和离线仿真逐步对比。
    real_time_factor 只决定推进节奏: 1.0 = 实时, 0.5 = 慢放, 0 = 尽快跑完。
"""

import time

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (Point, PoseStamped, PoseWithCovarianceStamped,
                               TransformStamped, TwistWithCovarianceStamped)
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from mpc_tracking import _paths
from mpc_tracking.common import (QOS_LATCHED, QOS_STREAM, make_odometry,
                                 ns_to_time, run_node, steady_clock, time_to_ns,
                                 yaw_to_quaternion)
from mpc_tracking_interfaces.msg import Obstacle, ObstacleArray

_paths.setup()
from core.vehicle_model import VehicleModel  # noqa: E402
from estimation import SensorSuite  # noqa: E402
from obstacles import DynamicObstacle, normalize_obstacles  # noqa: E402
from scenarios import SCENE_REGISTRY  # noqa: E402

# 车体外形 (只用于显示), 与 viz/animate.py 的 _vehicle_polygon 一致:
# 后轴中心为原点, 车身从 -0.3 m 到 L + 0.3 m, 宽 1.4 m
_REAR_OVERHANG = 0.3
_CAR_WIDTH = 1.4
_MAX_TRAIL = 5000


class SimNode(Node):

    def __init__(self):
        super().__init__('sim_node')
        self.scenario_name = self.declare_parameter('scenario', 'astar').value
        dt = self.declare_parameter('dt', 0.1).value
        wheelbase = self.declare_parameter('wheelbase', 2.5).value
        self.rtf = self.declare_parameter('real_time_factor', 1.0).value
        self.lockstep = self.declare_parameter('lockstep', True).value
        self.cmd_timeout = self.declare_parameter('cmd_timeout', 1.0).value
        self.start_timeout = self.declare_parameter('start_timeout', 30.0).value
        gps_sigma = self.declare_parameter('gps_sigma', 0.5).value
        gps_rate = self.declare_parameter('gps_rate', 2).value
        wheel_sigma = self.declare_parameter('wheel_sigma', 0.1).value
        wheel_rate = self.declare_parameter('wheel_rate', 1).value
        seed = self.declare_parameter('sensor_seed', 42).value
        self.r_safe_extra = (self.declare_parameter('r_car', 1.25).value
                             + self.declare_parameter('margin', 0.3).value)
        self.frame_id = self.declare_parameter('frame_id', 'map').value
        self.child_frame_id = self.declare_parameter('child_frame_id', 'base_link').value

        if self.rtf <= 0 and not self.lockstep:
            raise ValueError('real_time_factor=0 (尽快跑完) 需要 lockstep=true')
        if self.scenario_name not in SCENE_REGISTRY:
            raise ValueError(f'未知场景 {self.scenario_name!r}, 可选: {list(SCENE_REGISTRY)}')
        scene = SCENE_REGISTRY[self.scenario_name]()

        self.car = VehicleModel(L=wheelbase, dt=dt)
        self.state = self.car.reset(*scene.init_state)
        # normalize 会把 (cx,cy,r)/(cx,cy,vx,vy,r) 元组转成可 step() 的障碍对象
        self.obstacles = normalize_obstacles(scene.obstacles) if scene.obstacles else []
        self.sensors = SensorSuite(gps_sigma=gps_sigma, gps_rate=gps_rate,
                                   wheel_sigma=wheel_sigma, wheel_rate=wheel_rate,
                                   seed=seed)

        # 仿真时间用整数纳秒累加, 保证每拍时间戳严格等于 k * dt (浮点累加会漂)
        self.dt_ns = int(round(dt * 1e9))
        self.k = 0
        self.cmds = {}                      # 指令时间戳 (ns) -> [a, delta]
        self.latest_cmd = np.zeros(2)
        self.started = False
        self.created_wall = time.monotonic()
        self.tick_wall = self.created_wall  # 本拍数据发布的墙钟时刻 (lockstep 超时用)
        self.next_step_wall = 0.0
        self.timeout_warned = False
        self.trail = Path()
        self.trail.header.frame_id = self.frame_id

        self.pub_clock = self.create_publisher(Clock, '/clock', 10)
        self.pub_odom = self.create_publisher(Odometry, '/sim/odom_truth', QOS_STREAM)
        self.pub_gps = self.create_publisher(PoseWithCovarianceStamped, '/sim/gps', QOS_STREAM)
        self.pub_wheel = self.create_publisher(TwistWithCovarianceStamped,
                                               '/sim/wheel_speed', QOS_STREAM)
        self.pub_obs = self.create_publisher(ObstacleArray, '/sim/obstacles', QOS_LATCHED)
        self.pub_obs_markers = self.create_publisher(MarkerArray, '/sim/obstacle_markers',
                                                     QOS_STREAM)
        self.pub_vehicle = self.create_publisher(Marker, '/sim/vehicle_marker', QOS_STREAM)
        self.pub_trail = self.create_publisher(Path, '/sim/trajectory', QOS_STREAM)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(AckermannDriveStamped, '/control/cmd', self._on_cmd,
                                 QOS_STREAM)

        self._publish_world()
        # 开始前每 0.5 s 重发 t=0 的世界状态, 让晚启动的节点 (控制器、RViz) 也能拿到
        self.create_timer(0.5, self._on_idle, clock=steady_clock())
        self.create_timer(0.005, self._try_step, clock=steady_clock())
        self.get_logger().info(
            f'场景 {self.scenario_name}: 初始状态 {np.round(self.state, 3).tolist()}, '
            f'障碍 {len(self.obstacles)} 个, dt={dt}s, real_time_factor={self.rtf}, '
            f'lockstep={self.lockstep}。等待第一条控制指令...')

    @property
    def t_ns(self):
        return self.k * self.dt_ns

    # ------------------------------------------------------------ 控制指令

    def _on_cmd(self, msg):
        t_ns = time_to_ns(msg.header.stamp)
        # 车辆模型的输入是加速度和前轮转角 (AckermannDrive 的 speed 字段不使用)
        u = np.array([msg.drive.acceleration, msg.drive.steering_angle], dtype=float)
        self.cmds[t_ns] = u
        self.latest_cmd = u
        for old in [t for t in self.cmds if t < self.t_ns - 10 * self.dt_ns]:
            del self.cmds[old]
        if not self.started and t_ns == self.t_ns:
            self._start('收到第一条控制指令, 开始仿真')
        self._try_step()

    def _start(self, reason):
        self.started = True
        self.next_step_wall = time.monotonic()
        self.tick_wall = self.next_step_wall
        self.get_logger().info(reason)

    def _on_idle(self):
        if self.started:
            return
        if time.monotonic() - self.created_wall > self.start_timeout:
            self.get_logger().warn(
                f'{self.start_timeout:.0f}s 内没收到控制指令, 以零输入开始仿真')
            self._start('开始仿真 (无控制器)')
            return
        self._publish_world()

    # ------------------------------------------------------------ 推进

    def _try_step(self):
        if not self.started:
            return
        now = time.monotonic()
        if self.rtf > 0 and now < self.next_step_wall:
            return
        if self.lockstep and self.t_ns not in self.cmds:
            if now - self.tick_wall < self.cmd_timeout:
                return
            if not self.timeout_warned:
                self.get_logger().warn(
                    f't={self.t_ns / 1e9:.1f}s 等控制指令超过 {self.cmd_timeout}s, '
                    '沿用上一条指令 (控制节点是否在运行?)')
                self.timeout_warned = True
        else:
            self.timeout_warned = False

        u = self.cmds.get(self.t_ns, self.latest_cmd)
        self._step(u)

        self.tick_wall = time.monotonic()
        if self.rtf > 0:
            period = self.car.dt / self.rtf
            # 落后超过一个周期 (例如卡顿) 时不追赶, 避免连推好几步
            self.next_step_wall = max(self.next_step_wall + period, self.tick_wall - period)

    def _step(self, u):
        self.state = self.car.step(self.state, u)
        for obs in self.obstacles:
            obs.step(self.car.dt)
        self.k += 1
        meas = self.sensors.measure(self.state)
        self._publish_world()
        stamp = ns_to_time(self.t_ns)
        if meas.gps is not None:
            self._publish_gps(stamp, meas.gps)
        if meas.wheel is not None:
            self._publish_wheel(stamp, meas.wheel)

    # ------------------------------------------------------------ 发布

    def _publish_world(self):
        stamp = ns_to_time(self.t_ns)
        self.pub_clock.publish(Clock(clock=stamp))
        self._publish_obstacles(stamp)

        self.pub_odom.publish(make_odometry(stamp, self.frame_id, self.child_frame_id,
                                            self.state))
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.frame_id
        tf.child_frame_id = self.child_frame_id
        tf.transform.translation.x = float(self.state[0])
        tf.transform.translation.y = float(self.state[1])
        tf.transform.rotation = yaw_to_quaternion(float(self.state[2]))
        self.tf_broadcaster.sendTransform(tf)
        self._publish_vehicle_marker()

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.frame_id
        ps.pose.position.x = float(self.state[0])
        ps.pose.position.y = float(self.state[1])
        ps.pose.orientation = tf.transform.rotation
        if not self.trail.poses or self.trail.poses[-1].header.stamp != stamp:
            self.trail.poses.append(ps)
            del self.trail.poses[:-_MAX_TRAIL]
        self.trail.header.stamp = stamp
        self.pub_trail.publish(self.trail)

    def _publish_gps(self, stamp, z):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.pose.pose.position.x = float(z[0])
        msg.pose.pose.position.y = float(z[1])
        msg.pose.pose.orientation.w = 1.0
        cov = np.zeros(36)
        cov[[0, 1, 6, 7]] = self.sensors.gps.R.ravel()
        cov[[14, 21, 28, 35]] = 1e6     # z 和姿态没有测量, 给极大方差
        msg.pose.covariance = cov.tolist()
        self.pub_gps.publish(msg)

    def _publish_wheel(self, stamp, z):
        msg = TwistWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.child_frame_id
        msg.twist.twist.linear.x = float(z[0])
        cov = np.zeros(36)
        cov[0] = self.sensors.wheel.R[0, 0]
        cov[[7, 14, 21, 28, 35]] = 1e6  # 只测纵向速度
        msg.twist.covariance = cov.tolist()
        self.pub_wheel.publish(msg)

    def _publish_obstacles(self, stamp):
        arr = ObstacleArray()
        arr.header.stamp = stamp
        arr.header.frame_id = self.frame_id
        markers = MarkerArray()
        for i, obs in enumerate(self.obstacles):
            vx, vy = ((obs.vx, obs.vy) if isinstance(obs, DynamicObstacle) else (0.0, 0.0))
            arr.obstacles.append(Obstacle(id=i, x=float(obs.cx), y=float(obs.cy),
                                          vx=float(vx), vy=float(vy), radius=float(obs.r)))
            markers.markers.append(self._cylinder(
                stamp, 3 * i, obs.cx, obs.cy, obs.r, 1.0, (0.85, 0.2, 0.2, 0.9)))
            # 浅色圆盘 = MPC 避障约束用的安全半径 r + r_car + margin
            markers.markers.append(self._cylinder(
                stamp, 3 * i + 1, obs.cx, obs.cy, obs.r + self.r_safe_extra, 0.04,
                (1.0, 0.6, 0.1, 0.25)))
            if vx != 0.0 or vy != 0.0:
                arrow = self._marker(stamp, 3 * i + 2, Marker.ARROW, (1.0, 1.0, 0.2, 0.9))
                arrow.points = [Point(x=float(obs.cx), y=float(obs.cy), z=1.1),
                                Point(x=float(obs.cx + vx), y=float(obs.cy + vy), z=1.1)]
                arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.15, 0.35, 0.35
                markers.markers.append(arrow)
        self.pub_obs.publish(arr)
        if markers.markers:
            self.pub_obs_markers.publish(markers)

    def _marker(self, stamp, marker_id, marker_type, rgba, ns='obstacles'):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.frame_id
        m.ns = ns
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def _cylinder(self, stamp, marker_id, x, y, radius, height, rgba):
        m = self._marker(stamp, marker_id, Marker.CYLINDER, rgba)
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = height / 2.0
        m.scale.x = m.scale.y = 2.0 * float(radius)
        m.scale.z = float(height)
        return m

    def _publish_vehicle_marker(self):
        # 画在 base_link 坐标系里, 时间戳给 0 = 用最新 TF, 车体跟着 TF 走
        m = Marker()
        m.header.stamp = Time()
        m.header.frame_id = self.child_frame_id
        m.ns = 'vehicle'
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.frame_locked = True
        length = self.car.L + 2 * _REAR_OVERHANG
        m.pose.position.x = self.car.L / 2.0
        m.pose.position.z = 0.5
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = length, _CAR_WIDTH, 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.45, 0.95, 0.9
        self.pub_vehicle.publish(m)


def main(args=None):
    run_node(SimNode, args)


if __name__ == '__main__':
    main()
