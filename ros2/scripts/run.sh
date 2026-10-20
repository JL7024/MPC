#!/usr/bin/env bash
# 编译 (没改动时几秒就结束) 并启动整条链路。参数原样传给 ros2 launch, 例如:
#   bash ros2/scripts/run.sh
#   bash ros2/scripts/run.sh scenario:=oncoming
#   bash ros2/scripts/run.sh rviz:=false real_time_factor:=0 exit_on_finish:=true
set -eo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${MPC_WS:-$HOME/mpc_ws}"
export MPC_VENV="${MPC_VENV:-$WS/.venv}"
export DISPLAY="${DISPLAY:-:0}"     # WSLg 的显示, RViz2 窗口靠它弹到 Windows 桌面

if [ ! -d "$MPC_VENV" ]; then
  echo "[run] 首次运行: 创建 Python 虚拟环境并安装 cvxpy ..."
  bash "$SCRIPTS/setup_venv.sh"
fi

echo "[run] 编译 ROS2 包 ..."
mkdir -p "$WS/log"
if ! bash "$SCRIPTS/build.sh" > "$WS/log/last_build.txt" 2>&1; then
  cat "$WS/log/last_build.txt"
  echo "[run] 编译失败, 完整日志: $WS/log/last_build.txt"
  exit 1
fi

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source "$WS/install/setup.bash"
echo "[run] 启动: ros2 launch mpc_tracking full_pipeline.launch.py $*"
exec ros2 launch mpc_tracking full_pipeline.launch.py "$@"
