"""
一条命令启动整条链路: 仿真 + 规划 + EKF + 控制 + 监控 (+ RViz2)。

    ros2 launch mpc_tracking full_pipeline.launch.py
    ros2 launch mpc_tracking full_pipeline.launch.py scenario:=oncoming
    ros2 launch mpc_tracking full_pipeline.launch.py controller:=pid use_ekf:=false
    ros2 launch mpc_tracking full_pipeline.launch.py rviz:=false real_time_factor:=0 exit_on_finish:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_SHARE = get_package_share_directory('mpc_tracking')

_ARGS = [
    ('scenario', 'astar',
     '场景名 (scenarios.SCENE_REGISTRY): astar / line / circle / lane / accel_brake / '
     'serpentine / block / oncoming / crossing'),
    ('controller', 'mpc', '控制器: mpc / pid / lqr'),
    ('use_ekf', 'true', 'true = 控制器用 EKF 估计; false = 直接用真值'),
    ('mpc_avoid', 'true', 'MPC 是否加避障约束 (场景里有障碍时才起作用)'),
    ('real_time_factor', '1.0', '仿真速度倍率: 1.0 实时, 0.5 慢放, 0 尽快跑完'),
    ('rviz', 'true', '是否打开 RViz2'),
    ('exit_on_finish', 'false', '车到终点、打印完指标后自动退出'),
    ('params_file', os.path.join(_SHARE, 'config', 'pipeline.yaml'), '参数文件'),
]


def _as_bool(text):
    return text.strip().lower() in ('1', 'true', 'yes', 'on')


def _setup(context):
    cfg = {name: LaunchConfiguration(name).perform(context) for name, _, _ in _ARGS}
    params = cfg['params_file']
    use_ekf = _as_bool(cfg['use_ekf'])
    scenario = {'scenario': cfg['scenario']}
    # 仿真节点发布 /clock, 其它节点 (包括 RViz) 都用仿真时间
    sim_time = {'use_sim_time': True}
    # 注意: 这里用字典传的参数是"通配"参数, 优先级低于 yaml 里按节点名写的同名参数,
    # 所以由 launch 参数控制的开关 (real_time_factor、controller 等) 不要写进 yaml 的节点段

    def node(executable, parameters, **kwargs):
        # PYTHONUNBUFFERED: 算法代码里的 print (如 MPC 求解失败提示) 能实时显示
        return Node(package='mpc_tracking', executable=executable, name=executable,
                    output='screen', parameters=parameters,
                    additional_env={'PYTHONUNBUFFERED': '1'}, **kwargs)

    actions = [
        node('sim_node', [params, scenario,
                          {'real_time_factor': float(cfg['real_time_factor'])}]),
        node('planner_node', [params, scenario, sim_time]),
        node('controller_node', [params, scenario, sim_time,
                                 {'controller': cfg['controller'], 'use_ekf': use_ekf,
                                  'mpc_avoid': _as_bool(cfg['mpc_avoid'])}]),
    ]
    if use_ekf:
        actions.append(node('ekf_node', [params, scenario, sim_time]))

    monitor_kwargs = {}
    if _as_bool(cfg['exit_on_finish']):
        monitor_kwargs['on_exit'] = Shutdown(reason='监控节点打印完指标, 退出')
    actions.append(node('monitor_node',
                        [params, sim_time, {'exit_on_finish': _as_bool(cfg['exit_on_finish'])}],
                        **monitor_kwargs))

    if _as_bool(cfg['rviz']):
        # 关掉 RViz 窗口 = 结束整个仿真
        actions.append(Node(package='rviz2', executable='rviz2', name='rviz2',
                            arguments=['-d', os.path.join(_SHARE, 'rviz', 'pipeline.rviz')],
                            parameters=[sim_time], output='log',
                            on_exit=Shutdown(reason='RViz 窗口已关闭')))
    return actions


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default, description=desc)
         for name, default, desc in _ARGS]
        + [OpaqueFunction(function=_setup)])
