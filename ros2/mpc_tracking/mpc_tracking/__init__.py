"""
MPC 轨迹跟踪仿真的 ROS2 封装。

节点只负责消息收发和时序, 算法全部复用仓库根目录下的模块
(core/ controllers/ estimation/ planning/ obstacles/ scenarios.py), 不复制一份。
"""
