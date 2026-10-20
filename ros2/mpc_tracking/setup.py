import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'mpc_tracking'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JL7024',
    maintainer_email='jl7024@todo.todo',
    description='ROS2 wrapper of the MPC trajectory tracking simulation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sim_node = mpc_tracking.sim_node:main',
            'planner_node = mpc_tracking.planner_node:main',
            'ekf_node = mpc_tracking.ekf_node:main',
            'controller_node = mpc_tracking.controller_node:main',
            'monitor_node = mpc_tracking.monitor_node:main',
        ],
    },
)
