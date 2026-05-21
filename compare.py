"""
多控制器同场景对比脚本

用法:
    python compare.py                         # 三场景, 三控制器
    python compare.py --case lane             # 只跑双移线
    python compare.py --case lane --save out  # 保存对比图

输出:
    每个场景一张 4 面板对比图:
        (0,0) XY 轨迹    : 三条轨迹叠合
        (0,1) 横向误差   : 三条曲线叠合
        (1,0) 航向误差   : 三条曲线叠合
        (1,1) 速度跟踪   : 三条曲线 + 参考速度
    再加一张终端表格统计 RMS / max / 求解时间。
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from main import run_simulation, make_controller
from viz.plot import compute_errors
from core.vehicle_model import VehicleModel
from scenarios import SCENE_REGISTRY


CONTROLLER_NAMES = ['mpc', 'pid', 'lqr']
COLORS = {'mpc': 'tab:blue', 'pid': 'tab:orange', 'lqr': 'tab:green'}
LABELS = {'mpc': 'MPC', 'pid': 'PID (Stanley+PI)', 'lqr': 'LQR (误差状态)'}


def run_one(scene_fn, ctrl_name):
    """跑一个 (场景, 控制器) 组合, 返回 hist + 错误指标"""
    sc = scene_fn()
    car = VehicleModel(L=2.5, dt=0.1)
    ctrl = make_controller(ctrl_name, car)
    hist = run_simulation(car, sc.ref, ctrl, sc.init_state, max_steps=3000)
    e_lat, e_phi, e_v = compute_errors(hist)
    metrics = {
        'steps': len(hist['u']),
        'e_lat_rms': float(np.sqrt(np.mean(e_lat**2))),
        'e_lat_max': float(np.max(np.abs(e_lat))),
        'e_phi_rms_deg': float(np.rad2deg(np.sqrt(np.mean(e_phi**2)))),
        'e_v_rms': float(np.sqrt(np.mean(e_v**2))),
        'solve_ms_avg': float(np.mean(hist['solve_time']) * 1000),
        'solve_ms_max': float(np.max(hist['solve_time']) * 1000),
    }
    return sc, hist, (e_lat, e_phi, e_v), metrics


def plot_overlay(scene, results, save_path=None):
    """
    画 4 面板对比图。
    results: dict {ctrl_name: (hist, errors, metrics)}
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{scene.title}  —  控制器对比', fontsize=14, fontweight='bold')

    # ---------- (0,0) XY 轨迹 ----------
    ax = axes[0, 0]
    ax.plot(scene.ref.points[:, 0], scene.ref.points[:, 1], 'k--',
            lw=1.0, label='参考', alpha=0.5)
    for cname in CONTROLLER_NAMES:
        hist = results[cname][0]
        ax.plot(hist['state'][:, 0], hist['state'][:, 1],
                color=COLORS[cname], lw=1.6, label=LABELS[cname], alpha=0.85)
    ax.plot(scene.init_state[0], scene.init_state[1], 'go', ms=10, label='起点', zorder=5)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('行驶轨迹')
    ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(loc='best')

    # ---------- (0,1) 横向误差 ----------
    ax = axes[0, 1]
    for cname in CONTROLLER_NAMES:
        hist, (e_lat, _, _), m = results[cname]
        t = np.arange(len(e_lat)) * 0.1
        ax.plot(t, e_lat, color=COLORS[cname], lw=1.5,
                label=f'{LABELS[cname]}  RMS={m["e_lat_rms"]:.3f}m')
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('t (s)'); ax.set_ylabel('横向误差 (m)')
    ax.set_title('横向误差对比')
    ax.grid(True, alpha=0.3); ax.legend(loc='best')

    # ---------- (1,0) 航向误差 ----------
    ax = axes[1, 0]
    for cname in CONTROLLER_NAMES:
        hist, (_, e_phi, _), m = results[cname]
        t = np.arange(len(e_phi)) * 0.1
        ax.plot(t, np.rad2deg(e_phi), color=COLORS[cname], lw=1.5,
                label=f'{LABELS[cname]}  RMS={m["e_phi_rms_deg"]:.2f}°')
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('t (s)'); ax.set_ylabel('航向误差 (°)')
    ax.set_title('航向误差对比')
    ax.grid(True, alpha=0.3); ax.legend(loc='best')

    # ---------- (1,1) 速度跟踪 ----------
    ax = axes[1, 1]
    # 参考速度
    first = results[CONTROLLER_NAMES[0]][0]
    t_ref = np.arange(len(first['ref_state'])) * 0.1
    ax.plot(t_ref, first['ref_state'][:, 3], 'k--', lw=1.2, label='参考', alpha=0.6)
    for cname in CONTROLLER_NAMES:
        hist, (_, _, e_v), m = results[cname]
        t = np.arange(len(hist['u'])) * 0.1
        ax.plot(t, hist['state'][1:, 3], color=COLORS[cname], lw=1.5,
                label=f'{LABELS[cname]}  RMS={m["e_v_rms"]:.3f}m/s')
    ax.set_xlabel('t (s)'); ax.set_ylabel('v (m/s)')
    ax.set_title('速度跟踪对比')
    ax.grid(True, alpha=0.3); ax.legend(loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  对比图已保存: {save_path}')
    return fig


def print_table(scene_name, results):
    """终端表格输出"""
    print(f"\n=== {scene_name} ===")
    header = f"{'controller':<14}{'steps':<8}{'e_lat_rms':<12}{'e_lat_max':<12}" \
             f"{'e_phi_rms°':<13}{'e_v_rms':<10}{'solve_ms':<12}"
    print(header)
    print('-' * len(header))
    for cname in CONTROLLER_NAMES:
        m = results[cname][2]
        print(f"{LABELS[cname]:<14}{m['steps']:<8}"
              f"{m['e_lat_rms']:<12.4f}{m['e_lat_max']:<12.4f}"
              f"{m['e_phi_rms_deg']:<13.3f}{m['e_v_rms']:<10.4f}"
              f"{m['solve_ms_avg']:<6.3f}avg/{m['solve_ms_max']:<.2f}max")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['all'] + list(SCENE_REGISTRY.keys()),
                        default='all')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--save-dir', default=None,
                        help='保存对比图到此目录')
    args = parser.parse_args()

    todo = list(SCENE_REGISTRY.keys()) if args.case == 'all' else [args.case]

    for sname in todo:
        scene_fn = SCENE_REGISTRY[sname]
        results = {}
        for cname in CONTROLLER_NAMES:
            sc, hist, errs, metrics = run_one(scene_fn, cname)
            results[cname] = (hist, errs, metrics)
        scene = sc  # 任一 results 里的 scene 都行 (同一场景)

        print_table(scene.title, results)
        save_path = f"{args.save_dir}/compare_{sname}.png" if args.save_dir else None
        fig = plot_overlay(scene, results, save_path=save_path)
        if args.no_show:
            plt.close(fig)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
