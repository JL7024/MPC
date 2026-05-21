"""
控制器抽象接口

所有控制器(MPC/PID/LQR)都实现这个接口,使得 main.py 的仿真循环
不需要知道具体控制器类型,可以即插即换。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class ControlLimits:
    """
    控制量物理约束。控制器内部用它做 saturation,
    绘图模块用它画约束上下限横线。
    """
    a_min: float = -5.0
    a_max: float = 3.0
    delta_max: float = np.deg2rad(30)


class BaseController(ABC):
    """
    控制器基类。

    子类需要:
        1. 在 __init__ 中设置 self.dt 和 self.limits
        2. 实现 solve(state, ref, nearest_idx, u_prev) -> (u, info)

    info 字典约定:
        必含: 'solve_time' (秒)
        可选: 'x_pred' (N+1, 4), 'cost', 'status'
        没有的字段不要放进去, 让上层用 .get() 取。
    """

    dt: float
    limits: ControlLimits

    @abstractmethod
    def solve(self, state, ref, nearest_idx, u_prev=None):
        """
        参数:
            state       : np.ndarray (4,)        当前车辆状态 [x, y, phi, v]
            ref         : ReferenceTrajectory    完整参考轨迹对象,控制器自取所需窗口
            nearest_idx : int                    最近参考点索引(外部已算,避免重复搜)
            u_prev      : np.ndarray (2,) 或 None 上一拍施加的控制 [a, delta]

        返回:
            u    : np.ndarray (2,)  本拍要施加的控制
            info : dict             见类 docstring
        """
        raise NotImplementedError
