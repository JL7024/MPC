"""节点间共用的小工具: QoS 配置、时间戳、消息与 numpy 状态之间的转换。"""

import math
import signal

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from rclpy.clock import Clock, ClockType
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

# 普通数据流 (传感器、状态、控制指令): 可靠传输, 不保留历史
QOS_STREAM = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20,
                        reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.VOLATILE)

# "锁存"话题 (参考轨迹、地图、障碍物): 晚启动的订阅者也能拿到最后一条
QOS_LATCHED = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)

NS_PER_SEC = 1_000_000_000


def steady_clock():
    """
    墙钟 (steady) 时钟, 给"等待启动 / 定期重发"这类定时器用。
    节点开了 use_sim_time 后, 默认定时器跟着 /clock 走; 仿真还没开始时 /clock
    不前进, 这类定时器就永远不触发, 所以要显式指定 steady 时钟。
    """
    return Clock(clock_type=ClockType.STEADY_TIME)


def run_node(node_cls, args=None, before_spin=None, should_exit=None):
    """
    节点 main() 的公共部分: 初始化、spin、收到 Ctrl+C / SIGTERM 时有序退出。

    为什么不用 rclpy 自带的信号处理: 它在另一个线程里异步关闭上下文, 主线程
    可能正好在取消息, 于是关闭时偶尔抛 RuntimeError。这里信号只设一个标志,
    主循环看到后再按顺序销毁节点、关闭 rclpy。
    """
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    received = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: received.append(signum))
    executor = SingleThreadedExecutor()
    node = None
    try:
        node = node_cls()
        executor.add_node(node)
        if before_spin is not None:
            before_spin()
        while not received and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if should_exit is not None and should_exit(node):
                break
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


# ---------------------------------------------------------------- 时间戳

def ns_to_time(t_ns):
    return Time(sec=int(t_ns // NS_PER_SEC), nanosec=int(t_ns % NS_PER_SEC))


def time_to_ns(stamp):
    return stamp.sec * NS_PER_SEC + stamp.nanosec


# ---------------------------------------------------------------- 角度

def yaw_to_quaternion(yaw):
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def quaternion_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# ---------------------------------------------------------------- 状态 <-> Odometry

def make_odometry(stamp, frame_id, child_frame_id, state, cov=None):
    """
    车辆状态 [x, y, phi, v] → nav_msgs/Odometry。
    v 是车体纵向速度, 放在 twist.linear.x (child_frame_id 坐标系)。
    cov: 可选 4x4 状态协方差, 映射到 pose (x, y, yaw) 和 twist (vx) 的协方差。
    """
    x, y, phi, v = (float(s) for s in state)
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation = yaw_to_quaternion(phi)
    msg.twist.twist.linear.x = v
    if cov is not None:
        P = np.asarray(cov, dtype=float)
        pc = np.zeros(36)
        # pose 协方差按 (x, y, z, roll, pitch, yaw) 排列, 状态里的 phi 对应 yaw (第 5 维)
        idx = [0, 1, 5]
        for r in range(3):
            for c in range(3):
                pc[idx[r] * 6 + idx[c]] = P[r, c]
        msg.pose.covariance = pc.tolist()
        tc = np.zeros(36)
        tc[0] = P[3, 3]
        msg.twist.covariance = tc.tolist()
    return msg


def odometry_to_state(msg):
    p = msg.pose.pose
    return np.array([p.position.x, p.position.y,
                     quaternion_to_yaw(p.orientation),
                     msg.twist.twist.linear.x], dtype=float)


# ---------------------------------------------------------------- Path

def make_path(stamp, frame_id, xy, phi=None):
    """
    (M, 2) 点列 → nav_msgs/Path。phi 为 None 时用相邻点连线方向当朝向。
    """
    xy = np.asarray(xy, dtype=float)
    if phi is None and len(xy) >= 2:
        d = np.diff(xy, axis=0)
        phi = np.arctan2(d[:, 1], d[:, 0])
        phi = np.append(phi, phi[-1])
    msg = Path()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    for k, (x, y) in enumerate(xy):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        if phi is not None:
            ps.pose.orientation = yaw_to_quaternion(float(phi[k]))
        else:
            ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg
