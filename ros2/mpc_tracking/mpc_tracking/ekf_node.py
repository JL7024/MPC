"""
EKF 状态估计节点: 融合 GPS (位置) 和轮速 (速度), 发布 /ekf/odom。

滤波算法直接复用 estimation.ExtendedKalmanFilter, 本节点只决定什么时候 predict / update:
    - 测量消息带仿真时间戳。先把估计 predict 到该时间戳 (每次推进一个 dt,
      用这段时间里实际生效的控制指令), 再做对应传感器的 update。
    - 以轮速 (每拍都有, 最高频的传感器) 为节拍: 处理完轮速就发布一次估计。
      仿真节点同一拍里先发 GPS 再发轮速, 所以发布时这一拍的测量都已用上,
      顺序与离线仿真的 "predict → GPS update → 轮速 update" 相同。

订阅的创建顺序是刻意的: 控制指令最先, 然后 GPS、轮速。多条消息同时就绪时,
rclpy 按订阅创建顺序处理, 保证 predict 之前已经拿到这段时间的控制指令。
"""

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from mpc_tracking import _paths
from mpc_tracking.common import (QOS_STREAM, make_odometry, ns_to_time, run_node,
                                 steady_clock, time_to_ns)

_paths.setup()
from core.vehicle_model import VehicleModel  # noqa: E402
from estimation import ExtendedKalmanFilter  # noqa: E402
from scenarios import SCENE_REGISTRY  # noqa: E402


class EkfNode(Node):

    def __init__(self):
        super().__init__('ekf_node')
        scenario = self.declare_parameter('scenario', 'astar').value
        dt = self.declare_parameter('dt', 0.1).value
        wheelbase = self.declare_parameter('wheelbase', 2.5).value
        init_offset = self.declare_parameter('init_offset', [0.5, -0.3, 0.05, 0.5]).value
        p0_diag = self.declare_parameter('p0_diag', [1.0, 1.0, 0.1, 1.0]).value
        q_diag = self.declare_parameter('q_diag', [0.01, 0.01, 0.001, 0.05]).value
        self.frame_id = self.declare_parameter('frame_id', 'map').value
        self.child_frame_id = self.declare_parameter('child_frame_id', 'base_link').value

        # 初值 = 场景初始状态 + 偏差, 与 main.make_ekf_and_sensors 相同, 让 EKF 真正干活
        init_state = np.asarray(SCENE_REGISTRY[scenario]().init_state, dtype=float)
        x0 = init_state + np.asarray(init_offset, dtype=float)
        self.ekf = ExtendedKalmanFilter(VehicleModel(L=wheelbase, dt=dt), x0,
                                        np.diag(p0_diag), np.diag(q_diag))

        self.dt_ns = int(round(dt * 1e9))
        self.t_est_ns = 0           # 当前估计对应的仿真时间
        self.cmds = {}              # 指令时间戳 (ns) -> [a, delta]
        self.latest_cmd = np.zeros(2)
        self.got_measurement = False
        self.last_wheel_ns = None

        self.pub = self.create_publisher(Odometry, '/ekf/odom', QOS_STREAM)
        self.create_subscription(AckermannDriveStamped, '/control/cmd', self._on_cmd,
                                 QOS_STREAM)
        self.create_subscription(PoseWithCovarianceStamped, '/sim/gps', self._on_gps,
                                 QOS_STREAM)
        self.create_subscription(TwistWithCovarianceStamped, '/sim/wheel_speed',
                                 self._on_wheel, QOS_STREAM)

        # 收到第一条测量之前, 定期发布初始估计, 让控制器能算出 t=0 的第一条指令
        self._publish()
        self.create_timer(0.5, self._on_idle, clock=steady_clock())
        self.get_logger().info(f'EKF 初值 {np.round(x0, 3).tolist()} (真值 + 偏差 {init_offset})')

    def _on_idle(self):
        if not self.got_measurement:
            self._publish()

    def _on_cmd(self, msg):
        t_ns = time_to_ns(msg.header.stamp)
        u = np.array([msg.drive.acceleration, msg.drive.steering_angle], dtype=float)
        self.cmds[t_ns] = u
        self.latest_cmd = u
        for old in [t for t in self.cmds if t < self.t_est_ns - 10 * self.dt_ns]:
            del self.cmds[old]

    def _predict_to(self, t_ns):
        # 每次推进一个 dt; 区间 [t, t+dt) 内生效的是时间戳为 t 的那条指令
        while self.t_est_ns + self.dt_ns // 2 < t_ns:
            u = self.cmds.get(self.t_est_ns)
            if u is None:
                u = self.latest_cmd
                self.get_logger().warn(
                    f'缺少 t={self.t_est_ns / 1e9:.1f}s 的控制指令, 用最近一条代替',
                    throttle_duration_sec=2.0)
            self.ekf.predict(u)
            self.t_est_ns += self.dt_ns

    def _on_gps(self, msg):
        self._predict_to(time_to_ns(msg.header.stamp))
        c = msg.pose.covariance
        z = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        R = np.array([[c[0], c[1]], [c[6], c[7]]])
        self.ekf.update_gps(z, R)
        self.got_measurement = True

    def _on_wheel(self, msg):
        t_ns = time_to_ns(msg.header.stamp)
        if self.last_wheel_ns is not None and t_ns - self.last_wheel_ns > self.dt_ns * 3 // 2:
            self.get_logger().warn('轮速不是每拍都有, EKF 会缺拍发布; 建议 wheel_rate=1',
                                   throttle_duration_sec=10.0)
        self.last_wheel_ns = t_ns
        self._predict_to(t_ns)
        z = np.array([msg.twist.twist.linear.x])
        R = np.array([[msg.twist.covariance[0]]])
        self.ekf.update_wheel(z, R)
        self.got_measurement = True
        self._publish()

    def _publish(self):
        self.pub.publish(make_odometry(ns_to_time(self.t_est_ns), self.frame_id,
                                       self.child_frame_id, self.ekf.state(), self.ekf.P))


def main(args=None):
    run_node(EkfNode, args)


if __name__ == '__main__':
    main()
