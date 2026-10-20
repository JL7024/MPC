"""
让节点能 import 仓库根目录下的算法模块, 以及虚拟环境里的 cvxpy。

为什么不把算法代码搬进 ROS 包:
    算法模块是项目主体, main.py / compare*.py 也在用。ROS 包只做一层薄封装,
    直接 import 同一份代码, 两边永远一致。

仓库根目录怎么找:
    用 colcon build --symlink-install 编译时, 安装目录里的 .py 是指回源码的
    符号链接, realpath(__file__) = <repo>/ros2/mpc_tracking/mpc_tracking/_paths.py,
    往上 3 层目录就是 <repo>。也可以用环境变量 MPC_REPO_ROOT 显式指定。

为什么把虚拟环境的 site-packages 插到 sys.path 最前面:
    cvxpy 1.7 需要比 Ubuntu 自带 scipy 1.11 更新的 scipy, 两者只装在 venv 里。
    numpy 刻意没装进 venv, 仍用系统的 1.26: ROS 的 Python 消息模块是按
    numpy 1.x 编译的, 换成 numpy 2.x 会导入失败。
"""

import glob
import os
import sys

_done = False


def _find_repo_root():
    candidates = []
    if os.environ.get('MPC_REPO_ROOT'):
        candidates.append(os.environ['MPC_REPO_ROOT'])
    here = os.path.dirname(os.path.realpath(__file__))
    candidates.append(os.path.abspath(os.path.join(here, '..', '..', '..')))
    for root in candidates:
        if os.path.isfile(os.path.join(root, 'controllers', 'mpc.py')):
            return root
    raise RuntimeError(
        '找不到算法代码所在的仓库根目录 (应包含 controllers/mpc.py)。'
        '请用 colcon build --symlink-install 编译, 或设置环境变量 MPC_REPO_ROOT。'
        f'已尝试: {candidates}')


def setup():
    """幂等: 每个节点在 import 算法模块之前调用一次。"""
    global _done
    if _done:
        return
    venv = os.environ.get('MPC_VENV', os.path.expanduser('~/mpc_ws/.venv'))
    for site in sorted(glob.glob(os.path.join(venv, 'lib', 'python3*', 'site-packages'))):
        if site not in sys.path:
            sys.path.insert(0, site)
    root = _find_repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    _done = True
