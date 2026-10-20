#!/usr/bin/env bash
# 在 Ubuntu (WSL) 里编译 ROS2 包。源码留在仓库里, 编译产物放在 ~/mpc_ws (Linux 文件系统, 更快)。
# 用法: bash ros2/scripts/build.sh [额外的 colcon 参数]
set -eo pipefail

ROS2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${MPC_WS:-$HOME/mpc_ws}"

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
mkdir -p "$WS"
cd "$WS"
# --symlink-install: 安装目录里放指回源码的链接, 改 Python / launch / yaml / rviz 不用重新编译;
# 节点也靠这些链接找到仓库根目录下的算法代码 (见 mpc_tracking/_paths.py)
colcon build --symlink-install --base-paths "$ROS2_DIR" "$@"
