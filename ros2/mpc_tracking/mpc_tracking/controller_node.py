"""
控制节点: 每收到一次车辆状态就算一次控制, 发布 [a, delta] 指令和 MPC 预测时域。

为什么由状态消息触发, 而不是自己开定时器:
    EKF 每拍发布一次估计。收到就算, 控制总是基于这一拍最新的估计;
    如果控制器另开定时器, 两个定时器的相位差会随机多出一拍延迟。
    指令的时间戳 = 所用状态的时间戳, 仿真节点 lockstep 时靠它对齐。

每拍的处理与 main.run_simulation 相同: find_nearest (带回退余量) → controller.solve。
控制器对象直接用 controllers/ 下的 MPCController / PIDController / LQRController。
"""

# 必须最先导入: 在 numpy / rclpy 创建任何线程之前屏蔽 SIGINT, 原因见 sigint_guard
from mpc_tracking import sigint_guard  # noqa: I100

import math

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

from mpc_tracking import _paths
from mpc_tracking.common import (QOS_LATCHED, QOS_STREAM, make_path, ns_to_time,
                                 odometry_to_state, run_node, time_to_ns)
from mpc_tracking_interfaces.msg import ControllerStatus, ObstacleArray
from mpc_tracking_interfaces.msg import ReferenceTrajectory as ReferenceTrajectoryMsg

_paths.setup()
from controllers import LQRController, MPCController, PIDController  # noqa: E402
from core.reference_trajectory import ReferenceTrajectory  # noqa: E402
from core.vehicle_model import VehicleModel  # noqa: E402
from obstacles import CircleObstacle, DynamicObstacle  # noqa: E402
from scenarios import SCENE_REGISTRY  # noqa: E402

# 与 main.run_simulation 相同: 最近点搜索允许回退 20 个点 (ds=0.1 → 2 m),
# 吸收 EKF 估计的瞬时跳动, 避免在闭合曲线上越跑越前
N_ROLLBACK = 20
SEARCH_WINDOW = 200


class ControllerNode(Node):

    def __init__(self):
        super().__init__('controller_node')
        scenario = self.declare_parameter('scenario', 'astar').value
        self.ctrl_name = self.declare_parameter('controller', 'mpc').value
        dt = self.declare_parameter('dt', 0.1).value
        wheelbase = self.declare_parameter('wheelbase', 2.5).value
        use_ekf = self.declare_parameter('use_ekf', True).value
        self.mpc_avoid = self.declare_parameter('mpc_avoid', True).value
        self.avoidance_mode = self.declare_parameter('avoidance_mode', 'hard').value
        self.lambda_soft = self.declare_parameter('lambda_soft', 1000.0).value
        self.horizon = self.declare_parameter('horizon', 0).value
        avoid_side = self.declare_parameter('avoid_side', '').value
        alpha_uncert = self.declare_parameter('alpha_uncert', -1.0).value
        self.stop_decel = self.declare_parameter('stop_decel', 3.0).value
        self.frame_id = self.declare_parameter('frame_id', 'map').value
        if self.ctrl_name not in ('mpc', 'pid', 'lqr'):
            raise ValueError(f'未知控制器 {self.ctrl_name!r} (可选: mpc / pid / lqr)')

        # avoid_side / alpha_uncert 默认取场景里的设置, 与 main.run_case 相同
        scene = SCENE_REGISTRY[scenario]()
        self.avoid_side = (getattr(scene, 'avoid_side', None) if avoid_side == ''
                           else (None if avoid_side == 'none' else avoid_side))
        self.alpha_uncert = (getattr(scene, 'alpha_uncert', 0.0) if alpha_uncert < 0
                             else alpha_uncert)

        self.car = VehicleModel(L=wheelbase, dt=dt)
        self.ctrl = None
        self.ref = None
        self.ref_phi = None             # 展开 (unwrap) 后的参考航向, 用来对齐状态航向
        self.plan_id = None
        self.obstacles = None           # 最近一次收到的障碍物 (对象列表)
        self.last_idx = 0
        self.u_prev = None
        self.stopping = False
        self.last_stamp = -1
        self.last_cmd = None
        self.pending = None             # 控制器就绪前收到的状态

        self.pub_cmd = self.create_publisher(AckermannDriveStamped, '/control/cmd', QOS_STREAM)
        self.pub_pred = self.create_publisher(Path, '/control/mpc_prediction', QOS_STREAM)
        self.pub_window = self.create_publisher(Path, '/control/reference_window', QOS_STREAM)
        self.pub_status = self.create_publisher(ControllerStatus, '/control/status', QOS_STREAM)

        self.create_subscription(ReferenceTrajectoryMsg, '/planning/reference',
                                 self._on_reference, QOS_LATCHED)
        self.create_subscription(ObstacleArray, '/sim/obstacles', self._on_obstacles,
                                 QOS_LATCHED)
        state_topic = '/ekf/odom' if use_ekf else '/sim/odom_truth'
        self.create_subscription(Odometry, state_topic, self._on_state, QOS_STREAM)
        self.get_logger().info(f'控制器 {self.ctrl_name}, 状态来源 {state_topic}, '
                               f'MPC 避障 {"开" if self.mpc_avoid else "关"}')

    @property
    def _uses_obstacles(self):
        return self.ctrl_name == 'mpc' and self.mpc_avoid

    # ------------------------------------------------------------ 输入

    def _on_reference(self, msg):
        if msg.plan_id == self.plan_id:
            return      # 规划节点每秒重发一次, 同一个计划不重复处理
        pts = np.column_stack([msg.x, msg.y, msg.phi, msg.v, msg.kappa, msg.s])
        ref = ReferenceTrajectory(L=msg.wheelbase)
        ref.points = pts.astype(float)
        self.ref = ref
        self.ref_phi = np.unwrap(ref.points[:, ref.IDX_PHI])
        self.plan_id = msg.plan_id
        self.last_idx = 0
        self.u_prev = None
        self.stopping = False
        self.get_logger().info(f'收到参考轨迹 (plan_id={msg.plan_id}, {len(ref)} 点)')
        self._maybe_build()

    def _on_obstacles(self, msg):
        obs = [CircleObstacle(o.x, o.y, o.radius) if (o.vx == 0.0 and o.vy == 0.0)
               else DynamicObstacle(o.x, o.y, o.vx, o.vy, o.radius)
               for o in msg.obstacles]
        self.obstacles = obs
        if self.ctrl is None or not self._uses_obstacles:
            self._maybe_build()
            return
        if len(obs) != len(self.ctrl.obstacles):
            # 障碍数量决定 QP 的约束个数, 变了只能重建 (一般不会发生)
            self.get_logger().warn(f'障碍数量变为 {len(obs)}, 重建 MPC')
            self.ctrl = None
            self._maybe_build()
            return
        # MPC 每次 solve 都从 self.obstacles 读取 predict(), 原地替换即可
        self.ctrl.obstacles[:] = obs

    def _maybe_build(self):
        if self.ctrl is not None or self.ref is None:
            return
        if self._uses_obstacles and self.obstacles is None:
            return
        if self.ctrl_name == 'mpc':
            obstacles = list(self.obstacles) if (self._uses_obstacles and self.obstacles) else None
            # 预测步数规则与 main.make_controller 相同: 默认 15, 动态障碍场景 25
            N = self.horizon if self.horizon > 0 else (25 if self.alpha_uncert > 0 else 15)
            self.ctrl = MPCController(vehicle_model=self.car, N=N, obstacles=obstacles,
                                      avoid_side=self.avoid_side,
                                      alpha_uncert=self.alpha_uncert,
                                      avoidance_mode=self.avoidance_mode,
                                      lambda_soft=self.lambda_soft)
            self.get_logger().info(
                f'MPC 就绪: N={N}, 障碍 {len(self.ctrl.obstacles)} 个, '
                f'避障 {self.avoidance_mode}, alpha_uncert={self.alpha_uncert}')
        elif self.ctrl_name == 'pid':
            self.ctrl = PIDController(vehicle_model=self.car)
            self.get_logger().info('PID 就绪')
        else:
            self.ctrl = LQRController(vehicle_model=self.car)
            self.get_logger().info('LQR 就绪')
        if self.pending is not None:
            t_ns, state = self.pending
            self.pending = None
            self._compute(t_ns, state)

    def _on_state(self, msg):
        t_ns = time_to_ns(msg.header.stamp)
        if t_ns == self.last_stamp:
            # 仿真开始前 EKF 会重复发布初始估计; 重发同一条指令, 防止仿真节点错过它
            if self.last_cmd is not None:
                self.pub_cmd.publish(self.last_cmd)
            return
        if t_ns < self.last_stamp:
            return
        state = odometry_to_state(msg)
        if self.ctrl is None:
            self.pending = (t_ns, state)
            return
        self._compute(t_ns, state)

    # ------------------------------------------------------------ 控制

    def _compute(self, t_ns, state):
        ref = self.ref
        search_from = max(0, self.last_idx - N_ROLLBACK)
        idx = ref.find_nearest(state[0], state[1], search_from=search_from,
                               search_window=SEARCH_WINDOW + N_ROLLBACK)
        self.last_idx = idx

        # 消息里的航向被四元数折回 [-pi, pi], 参考航向是连续的 (圆轨迹会超过 pi)。
        # 只在差值超过 pi 时整圈平移, 正常情况下不改动数值
        d = state[2] - self.ref_phi[idx]
        if abs(d) > math.pi:
            state[2] -= 2.0 * math.pi * round(d / (2.0 * math.pi))

        # PID / LQR 跟踪的是恒定参考速度, 越过终点后不会自己停; 越过终点就进入制动
        if not self.stopping and idx >= len(ref) - 1:
            end = ref.points[-1]
            heading = np.array([math.cos(end[ref.IDX_PHI]), math.sin(end[ref.IDX_PHI])])
            if float(np.dot(state[:2] - end[:2], heading)) > 0.0:
                self.stopping = True
                self.get_logger().info('已越过参考轨迹终点, 制动停车')

        if self.stopping:
            u = np.array([-self.stop_decel if state[3] > 0.05 else 0.0, 0.0])
            info = {'status': 'stopping', 'solve_time': 0.0}
        else:
            with sigint_guard.deferred():
                u, info = self.ctrl.solve(state, ref, idx, u_prev=self.u_prev)
            u = np.asarray(u, dtype=float)
        self.u_prev = u
        self.last_stamp = t_ns
        self._publish(t_ns, state, u, idx, info)

    def _publish(self, t_ns, state, u, idx, info):
        stamp = ns_to_time(t_ns)
        cmd = AckermannDriveStamped()
        cmd.header.stamp = stamp
        cmd.header.frame_id = 'base_link'
        cmd.drive.acceleration = float(u[0])
        cmd.drive.steering_angle = float(u[1])
        cmd.drive.speed = float(max(0.0, state[3] + u[0] * self.car.dt))   # 仅供参考
        self.pub_cmd.publish(cmd)
        self.last_cmd = cmd

        x_pred = info.get('x_pred')
        if x_pred is not None:
            self.pub_pred.publish(make_path(stamp, self.frame_id, x_pred[:, :2], x_pred[:, 2]))
        N = getattr(self.ctrl, 'N', 20)
        window = self.ref.get_reference_window_by_time(idx, self.car.dt, N + 1)
        self.pub_window.publish(make_path(stamp, self.frame_id, window[:, :2], window[:, 2]))

        st = ControllerStatus()
        st.header.stamp = stamp
        st.header.frame_id = self.frame_id
        st.controller = self.ctrl_name
        st.solver_status = str(info.get('status', 'n/a'))
        st.solve_time_ms = float((info.get('solve_time') or 0.0) * 1000.0)
        cost = info.get('cost')
        st.cost = float(cost) if cost is not None else float('nan')
        st.nearest_idx = int(idx)
        st.progress = float(idx / max(1, len(self.ref) - 1))
        st.stopping = self.stopping
        self.pub_status.publish(st)


def main(args=None):
    run_node(ControllerNode, args, before_spin=sigint_guard.allow)


if __name__ == '__main__':
    main()
