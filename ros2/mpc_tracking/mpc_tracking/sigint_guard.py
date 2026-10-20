"""
让控制节点在 MPC 求解途中也能被 Ctrl+C 正常关闭。

问题: OSQP 求解时会临时接管 SIGINT, 求解中途收到的 Ctrl+C 被它吞掉 (只打断这次求解),
rclpy 收不到关闭信号, 节点就不退出, launch 等 5 秒后才强杀。

做法: 进程一开始就屏蔽 SIGINT, 之后创建的线程 (DDS、numpy 的 OpenBLAS 线程池) 都继承屏蔽;
只有主线程在不求解时放开。求解期间到达的 Ctrl+C 先挂起, 求解结束放开屏蔽后
再交给 rclpy 的信号处理, 节点正常退出。

必须在 import numpy / rclpy 之前导入本模块。
"""

import signal
from contextlib import contextmanager

_SIGINT = {signal.SIGINT}

signal.pthread_sigmask(signal.SIG_BLOCK, _SIGINT)


def allow():
    """主线程开始 spin 之前调用。"""
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _SIGINT)


@contextmanager
def deferred():
    """包住 OSQP 求解: 期间收到的 Ctrl+C 挂起, 结束后再处理。"""
    signal.pthread_sigmask(signal.SIG_BLOCK, _SIGINT)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _SIGINT)
