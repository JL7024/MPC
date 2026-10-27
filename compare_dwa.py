"""
MPC vs DWA 避障对比脚本

同一场景各跑一次:
    MPC: 半空间硬约束避障 (B 方案, 与 compare_avoid_modes 的 hard 相同),
         用场景的 avoid_side / alpha_uncert
    DWA: 动态窗口采样局部规划, r_car / margin / alpha_uncert 与 MPC 一致

输出:
    上: XY 轨迹叠合 + 障碍 (动态障碍画起点、轨迹, 以及各控制器离它最近那一刻的位置)
    下: 距障碍最小距离 / 车速 / 前轮转角 三条时间曲线
    终端表格: 步数 / 终止原因 / 横向误差 / 最小距离 / 是否撞 / 转角速率 / 求解时间

用法:
    python compare_dwa.py                          # block / oncoming / crossing 三个场景
    python compare_dwa.py --case oncoming          # 只跑一个
    python compare_dwa.py --save-dir results/dwa   # 保存对比图
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from main import run_simulation, make_controller
from core.vehicle_model import VehicleModel
from scenarios import SCENE_REGISTRY
from compare_avoid_modes import metrics_of


# 控制器 -> (color, 图例, 表格名)
CONTROLLERS = [
    ('mpc', 'tab:red',  'MPC (硬约束)', 'MPC(hard)'),
    ('dwa', 'tab:blue', 'DWA',          'DWA'),
]
DEFAULT_CASES = ['block', 'oncoming', 'crossing']


def run_one(scene_fn, ctrl_name):
    sc = scene_fn()
    car = VehicleModel(L=2.5, dt=0.1)
    ctrl = make_controller(ctrl_name, car, obstacles=sc.obstacles,
                           avoid_side=getattr(sc, 'avoid_side', None),
                           alpha_uncert=getattr(sc, 'alpha_uncert', 0.0))
    # stop_divergence_dist: 绕完障碍没回到终点附近、越走越远时提前结束, 不空跑到 max_steps
    hist = run_simulation(car, sc.ref, ctrl, sc.init_state, max_steps=1500,
                          obstacles_to_step=ctrl.obstacles,
                          stop_divergence_dist=5.0)
    return sc, ctrl, hist


def extra_metrics(hist, dt):
    """metrics_of 之外再补: 终止原因 / 转角平滑性 / 最低车速 / 最大求解时间 / 失败拍数"""
    u = hist['u']
    ddelta = np.diff(np.concatenate([[0.0], u[:, 1]])) / dt
    st = np.asarray(hist['solve_time']) * 1000
    return {
        'termination':    hist['termination_reason'],
        'ddelta_rms_dps': float(np.rad2deg(np.sqrt(np.mean(ddelta ** 2)))),
        'ddelta_max_dps': float(np.rad2deg(np.max(np.abs(ddelta)))),
        'v_min':          float(np.min(hist['state'][:, 3])),
        # MPC 首拍含 cvxpy 编译 (~100 ms), 不计入
        'max_solve_ms':   float(np.max(st[1:])) if len(st) > 1 else float(st[0]),
        # MPC: 求解失败 (触发兜底); DWA: 没有可行候选
        'n_fail':         int(sum(s not in ('optimal', 'optimal_inaccurate', 'n/a')
                                  for s in hist['status'])),
    }


def plot_compare(sc, results, dt=0.1, save_path=None):
    """
    results: list of (name, color, label, ctrl, hist, metrics)
    """
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
    fig.suptitle(f'{sc.title}  —  MPC vs DWA', fontsize=13, fontweight='bold')

    # ---------- 上: XY ----------
    ax = fig.add_subplot(gs[0, :])
    ax.plot(sc.ref.points[:, 0], sc.ref.points[:, 1], 'k--',
            lw=1.0, alpha=0.5, label='参考')
    # 障碍按跑得最久的那组画 (动态障碍的轨迹最完整)
    longest = max(results, key=lambda r: len(r[5]['obs_traj']))
    obs_traj, obs_ctrl = longest[5]['obs_traj'], longest[3]
    is_dynamic = bool(np.any(obs_traj[0] != obs_traj[-1]))
    for m, o in enumerate(obs_ctrl.obstacles):
        r_safe = o.r + obs_ctrl.r_car + obs_ctrl.margin
        ax.add_patch(plt.Circle(obs_traj[0, m], o.r, color='gray',
                                alpha=0.5, zorder=2))
        ax.add_patch(plt.Circle(obs_traj[0, m], r_safe, color='gray',
                                fill=False, ls='--', lw=1.0, zorder=2))
        if is_dynamic:
            ax.plot(obs_traj[:, m, 0], obs_traj[:, m, 1], ':',
                    color='gray', lw=0.9)
    for name, color, label, ctrl, hist, mtr in results:
        S = mtr['states']
        ax.plot(S[:, 0], S[:, 1], color=color, lw=1.8, alpha=0.9,
                label=f"{label}  最大横向偏移={mtr['e_lat_max']:.2f}m  "
                      f"min_dist={mtr['min_dist']:.2f}m")
        if is_dynamic:
            # 离障碍最近的那一刻: 车的位置 (实心点) + 障碍当时的位置 (同色虚线圆)
            i = int(np.argmin(mtr['dists'].min(axis=1)))
            m = int(np.argmin(mtr['dists'][i]))
            ax.plot(S[i, 0], S[i, 1], 'o', color=color, ms=6, zorder=6)
            ax.add_patch(plt.Circle(mtr['obs_traj'][i, m], ctrl.obstacles[m].r,
                                    color=color, fill=False, ls='--', lw=1.2,
                                    zorder=5))
    ax.plot(sc.init_state[0], sc.init_state[1], 'go', ms=9, label='起点', zorder=6)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    # XY 要等比例, 而场景多是 80 m × 6 m 这种细长条: 按长宽比定上排高度,
    # 否则上排格子里大半是空白 (13.5 ≈ 上排坐标轴宽度 inch, +1.0 留给标题和坐标轴文字)
    ax.autoscale_view()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    top_h = float(np.clip(13.5 * (y1 - y0) / (x1 - x0), 1.2, 5.0)) + 1.0
    gs.set_height_ratios([top_h, 4.0])
    fig.set_size_inches(15, top_h + 5.0)
    title = 'XY 轨迹 (灰: 障碍起始位置与 r_safe'
    title += '; 圆点 + 同色虚线圆: 离障碍最近时车和障碍的位置)' if is_dynamic else ')'
    ax.set_title(title)

    # ---------- 下左: 距障碍最小距离 ----------
    ax_d = fig.add_subplot(gs[1, 0])
    r_safe = results[0][5]['r_safe']
    r_collide = r_safe - results[0][3].margin
    for name, color, label, ctrl, hist, mtr in results:
        t = np.arange(len(mtr['states'])) * dt
        ax_d.plot(t, mtr['dists'].min(axis=1), color=color, lw=1.5, label=label)
    ax_d.axhline(r_safe, color='k', ls='--', lw=1.0, alpha=0.7,
                 label=f'r_safe={r_safe:.2f}m')
    ax_d.axhline(r_collide, color='maroon', ls=':', lw=1.0, alpha=0.8,
                 label=f'r_collide={r_collide:.2f}m')
    ax_d.set_ylim(0, 12)
    ax_d.set_xlabel('t (s)'); ax_d.set_ylabel('距障碍中心 (m)')
    ax_d.set_title('距障碍最小距离')
    ax_d.grid(True, alpha=0.3); ax_d.legend(loc='best', fontsize=8)

    # ---------- 下中: 车速 ----------
    ax_v = fig.add_subplot(gs[1, 1])
    for name, color, label, ctrl, hist, mtr in results:
        t = np.arange(len(hist['state'])) * dt
        ax_v.plot(t, hist['state'][:, 3], color=color, lw=1.5, label=label)
    ref_v = longest[4]['ref_state'][:, 3]
    ax_v.plot(np.arange(1, len(ref_v) + 1) * dt, ref_v, 'k--', lw=1.0,
              alpha=0.6, label='参考')
    ax_v.set_xlabel('t (s)'); ax_v.set_ylabel('v (m/s)')
    ax_v.set_title('车速')
    ax_v.grid(True, alpha=0.3); ax_v.legend(loc='best', fontsize=8)

    # ---------- 下右: 前轮转角 ----------
    ax_s = fig.add_subplot(gs[1, 2])
    for name, color, label, ctrl, hist, mtr in results:
        t = np.arange(len(hist['u'])) * dt
        ax_s.plot(t, np.rad2deg(hist['u'][:, 1]), color=color, lw=1.3,
                  label=f"{label}  转角速率 RMS={mtr['ddelta_rms_dps']:.1f}°/s")
    ax_s.set_xlabel('t (s)'); ax_s.set_ylabel('δ (°)')
    ax_s.set_title('前轮转角')
    ax_s.grid(True, alpha=0.3); ax_s.legend(loc='best', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  对比图已保存: {save_path}')
    return fig


def print_table(sc, results):
    print(f"\n=== {sc.title} ===")
    head = (f"{'controller':<11}{'steps':<7}{'end':<13}{'e_lat_rms':<11}"
            f"{'e_lat_max':<11}{'min_dist':<10}{'r_safe':<8}{'collide':<9}"
            f"{'v_min':<7}{'ddelta_rms':<12}{'ddelta_max':<12}{'avg_ms':<8}"
            f"{'max_ms*':<9}{'fail':<5}")
    print(head)
    print('-' * len(head))
    for name, color, label, ctrl, hist, mtr in results:
        short = next(c[3] for c in CONTROLLERS if c[0] == name)
        print(f"{short:<11}{mtr['steps']:<7}{mtr['termination']:<13}"
              f"{mtr['e_lat_rms']:<11.3f}{mtr['e_lat_max']:<11.3f}"
              f"{mtr['min_dist']:<10.3f}{mtr['r_safe']:<8.2f}"
              f"{('yes' if mtr['collided'] else 'no'):<9}{mtr['v_min']:<7.2f}"
              f"{mtr['ddelta_rms_dps']:<12.1f}{mtr['ddelta_max_dps']:<12.1f}"
              f"{mtr['avg_solve_ms']:<8.2f}{mtr['max_solve_ms']:<9.2f}"
              f"{mtr['n_fail']:<5}")
    print("  ddelta: 转角速率 °/s;  max_ms*: 不含首拍;  "
          "fail: MPC 求解失败 / DWA 无可行候选的拍数")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='all',
                        choices=['all'] + DEFAULT_CASES + ['astar'],
                        help='all = block / oncoming / crossing')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--save-dir', default=None)
    args = parser.parse_args()

    cases = DEFAULT_CASES if args.case == 'all' else [args.case]
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for case in cases:
        print(f'\n--- 跑 {case}: MPC (硬约束) / DWA ---')
        results = []
        for name, color, label, _ in CONTROLLERS:
            sc, ctrl, hist = run_one(SCENE_REGISTRY[case], name)
            mtr = metrics_of(sc, ctrl, hist)
            mtr.update(extra_metrics(hist, ctrl.dt))
            results.append((name, color, label, ctrl, hist, mtr))

        print_table(sc, results)
        save_path = (f'{args.save_dir}/compare_dwa_{case}.png'
                     if args.save_dir else None)
        fig = plot_compare(sc, results, dt=0.1, save_path=save_path)
        if args.no_show:
            plt.close(fig)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
