"""
监控节点: 只订阅、不参与控制。用真值评估跟踪效果, 车到终点时打印与 main.py 同口径的指标。

    - 到达判定与 main.run_simulation 相同: 真值距参考终点 < stop_dist_to_end
    - 跟踪误差用 viz.plot.compute_errors 计算 (第 k 拍: 真值 x_{k+1} 对比第 k 拍的参考点)
    - EKF 误差: 估计 x̂_k 对比真值 x_k, k = 0..T
因此这里的数字可以和离线的 python main.py --case astar --ekf --mpc-avoid 直接对比。
"""

import json
import math
import os
import time

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from mpc_tracking import _paths
from mpc_tracking.common import (QOS_LATCHED, odometry_to_state, run_node,
                                 steady_clock, time_to_ns)
from mpc_tracking_interfaces.msg import ControllerStatus
from mpc_tracking_interfaces.msg import ReferenceTrajectory as ReferenceTrajectoryMsg

_paths.setup()
os.environ.setdefault('MPLBACKEND', 'Agg')     # viz.plot 会 import matplotlib, 这里不需要窗口
from viz.plot import compute_errors  # noqa: E402

# 监控只记录数据, 订阅队列放大一些, 仿真全速跑时也不丢
_QOS_MONITOR = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
_OK_STATUS = ('optimal', 'optimal_inaccurate', 'n/a', 'stopping')


class MonitorNode(Node):

    def __init__(self):
        super().__init__('monitor_node')
        dt = self.declare_parameter('dt', 0.1).value
        self.stop_dist = self.declare_parameter('stop_dist_to_end', 0.5).value
        self.exit_on_finish = self.declare_parameter('exit_on_finish', False).value
        self.summary_path = self.declare_parameter('summary_path', '').value
        self.progress_every = self.declare_parameter('progress_every', 50).value

        self.dt_ns = int(round(dt * 1e9))
        self.ref = None
        self.truth = {}         # t_ns -> [x, y, phi, v]
        self.est = {}
        self.status = {}        # t_ns -> ControllerStatus
        self.last_status = None
        self.finish_t = None    # 真值进入终点圈的时刻
        self.finish_wall = None
        self.done = False

        self.create_subscription(ReferenceTrajectoryMsg, '/planning/reference',
                                 self._on_reference, QOS_LATCHED)
        self.create_subscription(Odometry, '/sim/odom_truth', self._on_truth, _QOS_MONITOR)
        self.create_subscription(Odometry, '/ekf/odom', self._on_est, _QOS_MONITOR)
        self.create_subscription(ControllerStatus, '/control/status', self._on_status,
                                 _QOS_MONITOR)
        self.create_timer(0.5, self._try_finish, clock=steady_clock())

    def _on_reference(self, msg):
        if self.ref is None:
            self.ref = np.column_stack([msg.x, msg.y, msg.phi, msg.v])

    def _on_est(self, msg):
        self.est[time_to_ns(msg.header.stamp)] = odometry_to_state(msg)
        self._try_finish()

    def _on_status(self, msg):
        self.status[time_to_ns(msg.header.stamp)] = msg
        self.last_status = msg
        self._try_finish()

    def _on_truth(self, msg):
        t_ns = time_to_ns(msg.header.stamp)
        state = odometry_to_state(msg)
        self.truth[t_ns] = state
        if self.ref is None or self.finish_t is not None:
            return
        k = t_ns // self.dt_ns
        if self.progress_every > 0 and k > 0 and k % self.progress_every == 0:
            self._log_progress(t_ns, state)
        if np.linalg.norm(state[:2] - self.ref[-1, :2]) < self.stop_dist:
            self.finish_t = t_ns
            self.finish_wall = time.monotonic()
            self._try_finish()

    def _log_progress(self, t_ns, state):
        # 上一拍的控制状态可能比这一拍的真值晚到, 没到就用最近收到的一条 (只是进度提示)
        st = self.status.get(t_ns - self.dt_ns) or self.last_status
        if st is None:
            return
        x_r, y_r, phi_r, _ = self.ref[st.nearest_idx]
        e_lat = -math.sin(phi_r) * (state[0] - x_r) + math.cos(phi_r) * (state[1] - y_r)
        self.get_logger().info(
            f't={t_ns / 1e9:5.1f}s  进度 {st.progress * 100:3.0f}%  速度 {state[3]:4.2f}m/s  '
            f'横向误差 {e_lat:+.3f}m  求解 {st.solve_time_ms:.1f}ms ({st.solver_status})')

    # ------------------------------------------------------------ 汇总

    def _try_finish(self):
        if self.finish_t is None or self.done:
            return
        T = self.finish_t // self.dt_ns
        stamps = [k * self.dt_ns for k in range(T + 1)]
        use_ekf = bool(self.est)
        complete = (all(t in self.status for t in stamps[:-1])
                    and (not use_ekf or all(t in self.est for t in stamps)))
        # 估计和控制状态比真值晚到一点; 最多再等 2 s
        if not complete and time.monotonic() - self.finish_wall < 2.0:
            return
        self._summarize(T, stamps, use_ekf)

    def _summarize(self, T, stamps, use_ekf):
        ok = [k for k in range(T) if stamps[k] in self.status
              and stamps[k] in self.truth and stamps[k + 1] in self.truth]
        if len(ok) < T:
            self.get_logger().warn(f'有 {T - len(ok)} 拍数据没收到, 指标只用收到的部分')
        sts = [self.status[stamps[k]] for k in ok]
        hist = {
            'state': np.array([self.truth[stamps[0]]] + [self.truth[stamps[k + 1]] for k in ok]),
            'ref_state': np.array([self.ref[st.nearest_idx] for st in sts]),
        }
        e_lat, e_phi, e_v = compute_errors(hist)
        solve_ms = np.array([st.solve_time_ms for st in sts])
        failures = sum(st.solver_status not in _OK_STATUS for st in sts)
        dist = float(np.linalg.norm(self.truth[stamps[-1]][:2] - self.ref[-1, :2]))
        controller = sts[0].controller if sts else '?'

        summary = {
            'controller': controller,
            'state_source': 'ekf' if use_ekf else 'truth',
            'steps': int(T),
            'final_dist_m': dist,
            'solve_ms_avg': float(solve_ms.mean()) if len(solve_ms) else float('nan'),
            'solve_ms_max': float(solve_ms.max()) if len(solve_ms) else float('nan'),
            'lat_rms_m': float(np.sqrt(np.mean(e_lat ** 2))),
            'lat_max_m': float(np.max(np.abs(e_lat))),
            'phi_rms_deg': float(np.rad2deg(np.sqrt(np.mean(e_phi ** 2)))),
            'v_rms_mps': float(np.sqrt(np.mean(e_v ** 2))),
            'solver_failures': int(failures),
        }
        lines = [
            '==================== ROS2 运行结果 ====================',
            f'  控制器 {controller}, 状态来源 {"EKF 估计" if use_ekf else "真值"}',
            f'  到达终点 @ step {T}, 距终点 {dist:.2f}m',
            f'  仿真步数:     {T}',
            f'  平均求解:     {summary["solve_ms_avg"]:.2f}ms / step',
            f'  最大求解:     {summary["solve_ms_max"]:.2f}ms / step',
            f'  横向 RMS:     {summary["lat_rms_m"]:.3f}m',
            f'  横向 max:     {summary["lat_max_m"]:.3f}m',
            f'  航向 RMS:     {summary["phi_rms_deg"]:.2f}°',
            f'  速度 RMS:     {summary["v_rms_mps"]:.3f}m/s',
        ]
        if use_ekf:
            ks = [t for t in stamps if t in self.est and t in self.truth]
            err = np.array([self.est[t] - self.truth[t] for t in ks])
            err[:, 2] = np.arctan2(np.sin(err[:, 2]), np.cos(err[:, 2]))
            rms = np.sqrt(np.mean(err ** 2, axis=0))
            summary['ekf_rms'] = {'x_m': float(rms[0]), 'y_m': float(rms[1]),
                                  'phi_deg': float(np.rad2deg(rms[2])), 'v_mps': float(rms[3])}
            lines.append(f'  EKF 估计 RMS: x={rms[0]:.3f}m  y={rms[1]:.3f}m  '
                         f'phi={np.rad2deg(rms[2]):.2f}°  v={rms[3]:.3f}m/s')
        lines.append(f'  求解失败:     {failures} 拍')
        self.get_logger().info('\n'.join(lines))

        if self.summary_path:
            with open(os.path.expanduser(self.summary_path), 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        self.done = True


def main(args=None):
    # exit_on_finish 时打印完指标就退出, launch 文件据此关闭整个系统
    run_node(MonitorNode, args,
             should_exit=lambda node: node.done and node.exit_on_finish)


if __name__ == '__main__':
    main()
