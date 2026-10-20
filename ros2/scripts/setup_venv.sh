#!/usr/bin/env bash
# 一次性: 建 Python 虚拟环境并安装 cvxpy, 节点运行时会自动加载这里的包。
# numpy 刻意不装进虚拟环境、并固定为系统版本: ROS 的 Python 消息模块是按系统
# numpy 1.x 编译的, pip 一旦把 numpy 升到 2.x, 这些模块就会导入失败。
set -eo pipefail

WS="${MPC_WS:-$HOME/mpc_ws}"
VENV="${MPC_VENV:-$WS/.venv}"

mkdir -p "$WS"
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
touch "$VENV/COLCON_IGNORE"     # 别让 colcon 把虚拟环境当成包去扫描
NUMPY_VERSION="$(python3 -c 'import numpy; print(numpy.__version__)')"
"$VENV/bin/python" -m pip install --disable-pip-version-check "cvxpy>=1.5" "numpy==$NUMPY_VERSION"
