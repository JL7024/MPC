"""
MPC 避障结果可视化 (静态 + 动态)

2 面板:
    上: XY 轨迹 + 参考 + 障碍 + 起终点
        静态障碍: 实心 + r_safe 虚线圈
        动态障碍: 起点实心 + 终点淡色 + 中间几个快照虚线圈 + 障碍轨迹线
    下: 距各障碍 (真实位置) 中心的距离时间序列
        虚线 = 静态 r_safe 基础值 (不含不确定性锥, 阶段 5 锥仅 MPC 内部用)
        斜线 (可选) = r_safe(k) 包络, 用 alpha_uncert 显式画
"""

import numpy as np
import matplotlib.pyplot as plt

from obstacles import normalize_obstacles, DynamicObstacle


def plot_avoidance(hist, ref, obstacles, dt, r_car=1.25, margin=0.3,
                    alpha_uncert=0.0, title='', save_path=None):
    """
    画避障对比图。静态/动态自动判别 (基于 hist 是否含 obs_traj).

    参数:
        hist        : run_simulation 输出. 动态场景下含 'obs_traj' (T+1, M, 2)
        ref         : ReferenceTrajectory
        obstacles   : 障碍列表 (用于读 r 和初始位置)
        dt          : 仿真步长
        r_car, margin: 与 MPC 对齐, 用来算 r_safe 基础值
        alpha_uncert: 不确定性锥增长率, 用来在距离图上画斜线包络 (>0 才画)
        title       : 图标题
        save_path   : 保存路径或 None
    """
    obs = normalize_obstacles(obstacles)
    M = len(obs)
    car_state = np.asarray(hist['state'])           # (T+1, 4)
    T = len(car_state)
    t = np.arange(T) * dt

    # 障碍真实位置历史. 静态: shape (T+1, M, 2), 每步都等于初始 xy
    has_obs_traj = 'obs_traj' in hist and hist['obs_traj'] is not None
    if has_obs_traj:
        obs_traj = np.asarray(hist['obs_traj'])      # (T+1, M, 2)
    else:
        # 静态: 重构 (用 normalize 后的 obs.xy 复制 T+1 份)
        obs_traj = np.tile(np.array([[o.xy for o in obs]]), (T, 1, 1))
        # 注意 (T, M, 2), 不是 T+1; 静态版没有 step, 用 T (state 长度) 即可
        # 这里 T 已经 = len(car_state) = T+1 in 仿真术语. 所以正好.

    # 是否动态: 任一障碍 obs_traj 起止不同就是动态
    is_dynamic = bool(np.any(obs_traj[0] != obs_traj[-1]))

    # 距各障碍真实位置的距离 (T, M)
    dists = np.linalg.norm(car_state[:, None, :2] - obs_traj, axis=-1)
    r_safes = np.array([o.r + r_car + margin for o in obs])
    min_dist_per_step = dists.min(axis=1)
    overall_min = float(min_dist_per_step.min())
    violations = int(((dists - r_safes[None, :]) < 0).any(axis=1).sum())

    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                              gridspec_kw={'height_ratios': [1.5, 1]})
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ============== 上: XY ==============
    ax = axes[0]
    ax.plot(ref.points[:, 0], ref.points[:, 1], 'k--',
            lw=1.0, alpha=0.6, label='参考')
    ax.plot(car_state[:, 0], car_state[:, 1], 'tab:blue',
            lw=1.8, label='车辆轨迹')

    if is_dynamic:
        # 动态障碍: 画起点实心 + 终点淡色 + 中间几个快照 (虚线圈) + 轨迹线
        n_snapshots = 5
        snap_idx = np.linspace(0, T - 1, n_snapshots, dtype=int)
        for m, o in enumerate(obs):
            r_safe = r_safes[m]
            traj = obs_traj[:, m, :]                  # (T+1, 2)
            # 障碍真实路径线
            ax.plot(traj[:, 0], traj[:, 1], 'r:', lw=1.0, alpha=0.6)
            for ki, k in enumerate(snap_idx):
                # alpha 从浅 (起点) 到深 (终点) — 让人看出运动方向
                a_fill = 0.15 + 0.5 * (ki / max(1, n_snapshots - 1))
                a_edge = 0.4 + 0.4 * (ki / max(1, n_snapshots - 1))
                ax.add_patch(plt.Circle(traj[k], o.r, color='red', fill=True,
                                        alpha=a_fill, zorder=2))
                ax.add_patch(plt.Circle(traj[k], r_safe, color='red',
                                        fill=False, ls='--', lw=0.7,
                                        alpha=a_edge * 0.6, zorder=2))
            # 标记最近接触时刻
            k_min = int(np.argmin(dists[:, m]))
            ax.add_patch(plt.Circle(traj[k_min], o.r, color='magenta',
                                    fill=False, lw=2.0, zorder=3))
            # 速度箭头 (起点) 让人看出运动方向
            if isinstance(o, DynamicObstacle):
                v = np.array([o.vx, o.vy])  # 注意 obs 已被 step 推进, vx/vy 不变
                if np.linalg.norm(v) > 1e-3:
                    ax.annotate('', xy=traj[0] + v * 1.2, xytext=traj[0],
                                arrowprops=dict(arrowstyle='->', color='red', lw=1.4))
        ax.plot([], [], 'r-', lw=4, alpha=0.4, label='障碍快照 (浅→深=时间)')
        ax.plot([], [], 'magenta', lw=2, label='最近接触时刻')
        ax.plot([], [], 'r:', lw=1.0, label='障碍真实路径')
    else:
        # 静态: 老画法
        for m, o in enumerate(obs):
            ax.add_patch(plt.Circle(o.xy, o.r, color='red', fill=True,
                                    alpha=0.45, zorder=2))
            ax.add_patch(plt.Circle(o.xy, r_safes[m], color='red', fill=False,
                                    ls='--', lw=1.0, alpha=0.7, zorder=2))
        ax.plot([], [], 'r-', lw=8, alpha=0.45, label='障碍')
        ax.plot([], [], 'r--', lw=1.0, label=f'r_safe (=r + r_car + margin)')

    ax.plot(car_state[0, 0], car_state[0, 1], 'go', ms=10, label='起点', zorder=5)
    ax.plot(car_state[-1, 0], car_state[-1, 1], 'r*', ms=14, label='终点', zorder=5)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.set_title(f'XY 轨迹  —  min(dist) = {overall_min:.2f}m,  '
                 f'r_safe 违反 = {violations} 步')

    # ============== 下: 距离时间序列 ==============
    ax2 = axes[1]
    colors = plt.cm.tab10(np.arange(M))
    for m in range(M):
        lbl = (f'obs {m} 起点 ({obs[m].xy[0]:.1f},{obs[m].xy[1]:.1f})'
               if is_dynamic else
               f'obs {m} ({obs[m].xy[0]:.1f},{obs[m].xy[1]:.1f})')
        ax2.plot(t, dists[:, m], color=colors[m], lw=1.4, label=lbl)
        ax2.axhline(r_safes[m], color=colors[m], ls=':', lw=1.0, alpha=0.7)
        # 不确定性锥包络: r_safe(t) = r_safe + α*t
        if alpha_uncert > 0:
            r_cone = r_safes[m] + alpha_uncert * t
            ax2.plot(t, r_cone, color=colors[m], ls='-.', lw=0.9, alpha=0.6,
                     label=f'r_safe(k) 锥, α={alpha_uncert}')
    ax2.plot(t, min_dist_per_step, 'k-', lw=0.8, alpha=0.7, label='min over obs')
    ax2.set_xlabel('t (s)'); ax2.set_ylabel('|x_车 - x_obs真实| (m)')
    title_dyn = '距各障碍真实位置距离' if is_dynamic else '距各障碍中心距离'
    ax2.set_title(f'{title_dyn}  (虚线 = r_safe; 应永远在虚线之上)')
    ax2.grid(True, alpha=0.3); ax2.legend(loc='best', fontsize=8, ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  避障图已保存: {save_path}')
    return fig
