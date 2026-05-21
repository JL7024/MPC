"""
A* 全局规划结果可视化

4 层叠加:
    1. OccupancyGrid (灰色, 含膨胀)
    2. 原始障碍圆 (红色, 未膨胀)
    3. A* 折线 (蓝点)
    4. B-spline 平滑路径 (绿色)
    + 起点/终点
    + 可选: 车辆实际行驶轨迹 (传 hist 才画)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_planning(scene, hist=None, save_path=None):
    """
    画 A* 规划场景的层叠图。

    参数:
        scene     : 由 astar_scene() 返回的 Scenario, 必须有附加属性
                    .grid_map / .raw_astar_path / .smoothed_path / .obstacles / .r_inflate
        hist      : 可选, run_simulation 返回的 dict; 不为 None 时叠加车辆实际轨迹
        save_path : 可选, 保存路径
    """
    gm = scene.grid_map
    obstacles = scene.obstacles

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(scene.title, fontsize=13, fontweight='bold')

    # 1. 膨胀后栅格 (灰)
    ax.imshow(gm.grid, cmap='Greys', origin='lower',
              extent=(gm.x_min, gm.x_max, gm.y_min, gm.y_max),
              interpolation='nearest', alpha=0.4,
              zorder=0)

    # 2. 原始障碍 + 安全圈虚线
    for cx, cy, r in obstacles:
        ax.add_patch(plt.Circle((cx, cy), r, color='red', fill=True, alpha=0.4,
                                zorder=2, label='_nolegend_'))
        ax.add_patch(plt.Circle((cx, cy), r + scene.r_inflate, color='red',
                                fill=False, ls='--', lw=1.0, alpha=0.6,
                                zorder=2, label='_nolegend_'))
    # 单独 legend entry
    ax.plot([], [], 'r-', lw=8, alpha=0.4, label='障碍 (实)')
    ax.plot([], [], 'r--', lw=1.0, label=f'安全圈 (+{scene.r_inflate:.2f}m)')

    # 3. A* 折线 (淡蓝)
    raw = scene.raw_astar_path
    ax.plot(raw[:, 0], raw[:, 1], 'b.-', ms=3, lw=0.8, alpha=0.45,
            label=f'A* 折线 ({len(raw)} 点)', zorder=3)

    # 4. 平滑路径 (深绿)
    sm = scene.smoothed_path
    ax.plot(sm[:, 0], sm[:, 1], color='tab:green', lw=2.0,
            label=f'B-spline 平滑 ({len(sm)} 点)', zorder=4)

    # ReferenceTrajectory (二次汉宁平滑后, 给 MPC 跟的)
    ax.plot(scene.ref.points[:, 0], scene.ref.points[:, 1],
            'k--', lw=0.8, alpha=0.7,
            label=f'参考轨迹 ({len(scene.ref)} 点)', zorder=5)

    # 5. 车辆实际轨迹 (有 hist 时)
    if hist is not None:
        ax.plot(hist['state'][:, 0], hist['state'][:, 1],
                'tab:purple', lw=1.8, alpha=0.85,
                label=f'车辆实际', zorder=6)

    # 起点/终点
    start = scene.init_state[:2]
    goal = scene.goal
    ax.plot(start[0], start[1], 'go', ms=12, label='起点', zorder=7)
    ax.plot(goal[0], goal[1], 'r*', ms=15, label='终点', zorder=7)

    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    # 信息框
    info_text = (
        f"A*: {scene.astar_info['expanded']} 节点, "
        f"{scene.astar_info['time']*1000:.1f}ms, "
        f"路径 {scene.astar_info['path_cost']:.1f}m\n"
        f"smooth: {scene.smooth_info['mode']}\n"
        f"碰撞: {scene.smooth_info['collisions']}"
    )
    ax.text(0.01, 0.97, info_text, transform=ax.transAxes,
            va='top', fontsize=8,
            bbox=dict(boxstyle='round', fc='white', alpha=0.85))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  规划图已保存: {save_path}')
    return fig
