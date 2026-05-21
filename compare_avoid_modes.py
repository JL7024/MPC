"""
避障约束 hard vs soft 对比脚本

跑同场景多次:
    hard:                          B 方案半空间硬约束
    soft @ λ ∈ {100, 1000, 10000}: A 方案二次惩罚, 不同权重

输出:
    XY 轨迹叠合 (车辆经过障碍时的不同绕行幅度)
    距障碍中心距离的时间曲线 + r_safe 横线
    终端指标表 (steps / e_lat_rms / min_dist / 是否撞)

用法:
    python compare_avoid_modes.py                      # 默认 block 场景
    python compare_avoid_modes.py --case oncoming      # 切场景
    python compare_avoid_modes.py --save-dir results   # 保存图
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from main import run_simulation, make_controller
from core.vehicle_model import VehicleModel
from viz.plot import compute_errors
from scenarios import SCENE_REGISTRY


# 模式 -> (label, color)
MODES = [
    ('hard', None,    'tab:red',    'hard (B 半空间)'),
    ('soft', 100.0,   'tab:orange', 'soft λ=100'),
    ('soft', 1000.0,  'tab:green',  'soft λ=1000'),
    ('soft', 10000.0, 'tab:blue',   'soft λ=10000'),
]


def run_one(scene_fn, mode, lam):
    sc = scene_fn()
    car = VehicleModel(L=2.5, dt=0.1)
    obs = sc.obstacles
    alpha = getattr(sc, 'alpha_uncert', 0.0)
    avoid_side = getattr(sc, 'avoid_side', None)
    kw = dict(obstacles=obs, avoid_side=avoid_side, alpha_uncert=alpha)
    if mode == 'soft':
        kw['avoidance_mode'] = 'soft'
        kw['lambda_soft'] = lam
    ctrl = make_controller('mpc', car, **kw)
    obs_to_step = ctrl.obstacles
    hist = run_simulation(car, sc.ref, ctrl, sc.init_state, max_steps=3000,
                          obstacles_to_step=obs_to_step)
    return sc, ctrl, hist


def metrics_of(sc, ctrl, hist):
    states = hist['state']
    obs_traj = hist.get('obs_traj')
    if obs_traj is None:
        # 静态: 用 ctrl.obstacles 的 xy 复制 T+1 份
        obs_traj = np.tile(np.array([[o.xy for o in ctrl.obstacles]]),
                            (len(states), 1, 1))
    dists = np.linalg.norm(states[:, None, :2] - obs_traj, axis=-1)
    r_obs_arr = np.array([o.r for o in ctrl.obstacles])
    r_safe_arr = r_obs_arr + ctrl.r_car + ctrl.margin
    r_collide_arr = r_obs_arr + ctrl.r_car
    min_per_obs = dists.min(axis=0)
    overall_min = float(min_per_obs.min())
    n_violate_safe = int(((dists - r_safe_arr[None, :]) < 0).any(axis=1).sum())
    n_collide = int(((dists - r_collide_arr[None, :]) < 0).any(axis=1).sum())
    e_lat, e_phi, e_v = compute_errors(hist)
    return {
        'steps':         len(hist['u']),
        'e_lat_rms':     float(np.sqrt(np.mean(e_lat ** 2))),
        'e_lat_max':     float(np.max(np.abs(e_lat))),
        'min_dist':      overall_min,
        'r_safe':        float(r_safe_arr[np.argmin(min_per_obs)]),
        'collided':      bool(n_collide > 0),
        'safe_violations': n_violate_safe,
        'avg_solve_ms':  float(np.mean(hist['solve_time']) * 1000),
        'states':        states,
        'obs_traj':      obs_traj,
        'dists':         dists,
        'r_safe_arr':    r_safe_arr,
    }


def plot_compare(sc, results, dt=0.1, save_path=None):
    """
    results: list of (mode, lam, color, label, metrics)
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                              gridspec_kw={'height_ratios': [1.4, 1]})
    fig.suptitle(f'{sc.title}  —  hard vs soft 对比',
                 fontsize=13, fontweight='bold')

    # ---------- 上: XY ----------
    ax = axes[0]
    ax.plot(sc.ref.points[:, 0], sc.ref.points[:, 1], 'k--',
            lw=1.0, alpha=0.5, label='参考')
    # 用第一组的障碍画起点位置 (动态时只画起点圆)
    obs_traj0 = results[0][4]['obs_traj']
    is_dynamic = bool(np.any(obs_traj0[0] != obs_traj0[-1]))
    for m, o in enumerate(sc.obstacles_normalized):
        # 起点障碍 + r_safe
        r_safe = o.r + 1.25 + 0.3   # 与 MPC 默认一致
        ax.add_patch(plt.Circle(obs_traj0[0, m], o.r, color='red',
                                fill=True, alpha=0.45, zorder=2))
        ax.add_patch(plt.Circle(obs_traj0[0, m], r_safe, color='red',
                                fill=False, ls='--', lw=1.0, alpha=0.6, zorder=2))
        if is_dynamic:
            # 终点障碍 (淡色)
            ax.add_patch(plt.Circle(obs_traj0[-1, m], o.r, color='red',
                                    fill=True, alpha=0.18, zorder=2))
            # 障碍轨迹线
            ax.plot(obs_traj0[:, m, 0], obs_traj0[:, m, 1],
                    'r:', lw=0.8, alpha=0.5)
    for mode, lam, color, label, mtr in results:
        ax.plot(mtr['states'][:, 0], mtr['states'][:, 1], color=color,
                lw=1.7, alpha=0.85,
                label=f'{label}  RMS={mtr["e_lat_rms"]:.2f}m  min={mtr["min_dist"]:.2f}m')
    ax.plot(sc.init_state[0], sc.init_state[1], 'go', ms=10, label='起点', zorder=5)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.set_title('XY 轨迹叠合 (障碍越远 = 越保守)')

    # ---------- 下: 距离时间序列 ----------
    ax2 = axes[1]
    # r_safe 取第一个 (假设场景只 1 个障碍, 多障碍只画 min)
    r_safe = results[0][4]['r_safe']
    for mode, lam, color, label, mtr in results:
        T = len(mtr['states'])
        t = np.arange(T) * dt
        min_d = mtr['dists'].min(axis=1)
        ax2.plot(t, min_d, color=color, lw=1.5,
                 label=f'{label}  min_dist={mtr["min_dist"]:.2f}m')
    ax2.axhline(r_safe, color='red', ls='--', lw=1.0, alpha=0.7,
                label=f'r_safe={r_safe:.2f}m (期望 ≥)')
    # 物理碰撞线
    r_collide = r_safe - 0.3   # margin
    ax2.axhline(r_collide, color='maroon', ls=':', lw=1.0, alpha=0.7,
                label=f'r_collide={r_collide:.2f}m (撞了)')
    ax2.set_xlabel('t (s)'); ax2.set_ylabel('min(|x_车 - x_obs|) (m)')
    ax2.set_title('距障碍最小距离时间序列  (越靠下 = 越贴 = 越激进)')
    ax2.grid(True, alpha=0.3); ax2.legend(loc='best', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  对比图已保存: {save_path}')
    return fig


def print_table(sc, results):
    print(f"\n=== {sc.title} ===")
    head = (f"{'mode':<18}{'steps':<7}{'e_lat_rms':<11}{'e_lat_max':<11}"
            f"{'min_dist':<10}{'r_safe':<8}{'撞?':<5}{'safe_violate':<13}{'avg_ms':<8}")
    print(head)
    print('-' * len(head))
    for mode, lam, color, label, mtr in results:
        coll = '是' if mtr['collided'] else '否'
        print(f"{label:<18}{mtr['steps']:<7}{mtr['e_lat_rms']:<11.4f}"
              f"{mtr['e_lat_max']:<11.4f}{mtr['min_dist']:<10.3f}"
              f"{mtr['r_safe']:<8.2f}{coll:<5}{mtr['safe_violations']:<13}"
              f"{mtr['avg_solve_ms']:<8.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='block',
                        choices=['block', 'oncoming', 'crossing', 'astar'])
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--save-dir', default=None)
    args = parser.parse_args()

    scene_fn = SCENE_REGISTRY[args.case]
    print(f'\n--- 跑 {args.case}: 4 个模式 (hard / soft λ=100 / 1000 / 10000) ---')

    results = []
    for mode, lam, color, label in MODES:
        sc, ctrl, hist = run_one(scene_fn, mode, lam)
        # 方便 plot 用: 记下 normalize 后的 obstacles
        sc.obstacles_normalized = ctrl.obstacles
        mtr = metrics_of(sc, ctrl, hist)
        results.append((mode, lam, color, label, mtr))

    print_table(sc, results)
    save_path = (f'{args.save_dir}/compare_avoid_{args.case}.png'
                 if args.save_dir else None)
    fig = plot_compare(sc, results, dt=ctrl.dt, save_path=save_path)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
