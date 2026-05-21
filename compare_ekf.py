"""
EKF on vs off 对比脚本

跑同一场景两次:
    EKF off: 控制器拿真值 (理想反馈)
    EKF on : 控制器拿 EKF 估计 (实际工程场景, 测量带噪)
两次仿真共享同一控制器和场景, 只在 "feedback 来源" 上不同。

用法:
    python compare_ekf.py                       # 全场景, 默认 MPC
    python compare_ekf.py --case lane           # 只跑双移线
    python compare_ekf.py --controller pid      # 切控制器
    python compare_ekf.py --save-dir results    # 保存对比图

输出:
    每个场景一张 4 面板对比图:
        (0,0) XY 轨迹  : 参考 + truth(off) + truth(on) + GPS 散点
        (0,1) 横向误差 : off vs on
        (1,0) 速度跟踪 : 参考 + off + on
        (1,1) EKF 估计 vs 真值的 4 分量误差 (仅 EKF on 的曲线)
    终端表格: 跟踪 RMS / EKF 估计 RMS (off 行没有估计)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from main import run_simulation, make_controller, make_ekf_and_sensors
from viz.plot import compute_errors
from core.vehicle_model import VehicleModel
from scenarios import SCENE_REGISTRY


def run_pair(scene_fn, ctrl_name):
    """同一场景跑 EKF off 和 EKF on 两次, 返回 (scene, hist_off, hist_on)"""
    # ---- off ----
    sc = scene_fn()
    car = VehicleModel(L=2.5, dt=0.1)
    ctrl = make_controller(ctrl_name, car)
    hist_off = run_simulation(car, sc.ref, ctrl, sc.init_state, max_steps=3000)

    # ---- on ----
    sc2 = scene_fn()  # 干净的 ref/init
    car2 = VehicleModel(L=2.5, dt=0.1)
    ctrl2 = make_controller(ctrl_name, car2)
    ekf, sensors = make_ekf_and_sensors(car2, sc2.init_state)
    hist_on = run_simulation(car2, sc2.ref, ctrl2, sc2.init_state, max_steps=3000,
                             estimator=ekf, sensors=sensors)
    return sc, hist_off, hist_on


def metrics_of(hist):
    """跟踪 RMS / EKF 估计 RMS (后者仅在 hist 有 state_est 时计算)"""
    e_lat, e_phi, e_v = compute_errors(hist)
    out = {
        'steps':           len(hist['u']),
        'e_lat_rms':       float(np.sqrt(np.mean(e_lat ** 2))),
        'e_lat_max':       float(np.max(np.abs(e_lat))),
        'e_phi_rms_deg':   float(np.rad2deg(np.sqrt(np.mean(e_phi ** 2)))),
        'e_v_rms':         float(np.sqrt(np.mean(e_v ** 2))),
    }
    if 'state_est' in hist:
        est = hist['state_est']
        truth = hist['state']
        err = est - truth
        err[:, 2] = np.arctan2(np.sin(err[:, 2]), np.cos(err[:, 2]))
        out['ekf_x_rms']       = float(np.sqrt(np.mean(err[:, 0] ** 2)))
        out['ekf_y_rms']       = float(np.sqrt(np.mean(err[:, 1] ** 2)))
        out['ekf_phi_rms_deg'] = float(np.rad2deg(np.sqrt(np.mean(err[:, 2] ** 2))))
        out['ekf_v_rms']       = float(np.sqrt(np.mean(err[:, 3] ** 2)))
    return out


def plot_overlay(scene, hist_off, hist_on, dt=0.1, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{scene.title}  —  EKF on vs off',
                 fontsize=14, fontweight='bold')

    e_lat_off, e_phi_off, e_v_off = compute_errors(hist_off)
    e_lat_on,  e_phi_on,  e_v_on  = compute_errors(hist_on)

    # ---------- (0,0) XY ----------
    ax = axes[0, 0]
    ax.plot(scene.ref.points[:, 0], scene.ref.points[:, 1], 'k--',
            lw=1.0, alpha=0.5, label='参考')
    ax.plot(hist_off['state'][:, 0], hist_off['state'][:, 1],
            color='tab:green', lw=1.6, label='truth (EKF off)', alpha=0.85)
    ax.plot(hist_on['state'][:, 0], hist_on['state'][:, 1],
            color='tab:blue', lw=1.6, label='truth (EKF on)', alpha=0.85)
    # GPS 散点 (仅 on)
    gps_pts = np.array([m for m in hist_on.get('meas_gps', []) if m is not None])
    if len(gps_pts) > 0:
        ax.plot(gps_pts[:, 0], gps_pts[:, 1], 'r.', ms=3, alpha=0.35,
                label='GPS 测量')
    ax.plot(scene.init_state[0], scene.init_state[1], 'go', ms=9,
            label='起点', zorder=5)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('行驶轨迹')
    ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=8)

    # ---------- (0,1) 横向误差 ----------
    ax = axes[0, 1]
    t_off = np.arange(len(e_lat_off)) * dt
    t_on  = np.arange(len(e_lat_on))  * dt
    ax.plot(t_off, e_lat_off, color='tab:green', lw=1.4,
            label=f'EKF off  RMS={np.sqrt(np.mean(e_lat_off**2)):.3f}m')
    ax.plot(t_on,  e_lat_on,  color='tab:blue', lw=1.4,
            label=f'EKF on   RMS={np.sqrt(np.mean(e_lat_on**2)):.3f}m')
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('t (s)'); ax.set_ylabel('横向误差 (m)')
    ax.set_title('横向误差对比 (实际偏离参考)')
    ax.grid(True, alpha=0.3); ax.legend(loc='best')

    # ---------- (1,0) 速度跟踪 ----------
    ax = axes[1, 0]
    t_ref = np.arange(len(hist_off['ref_state'])) * dt
    ax.plot(t_ref, hist_off['ref_state'][:, 3], 'k--',
            lw=1.0, alpha=0.6, label='参考')
    ax.plot(t_off, hist_off['state'][1:, 3], color='tab:green',
            lw=1.4, label=f'EKF off  RMS={np.sqrt(np.mean(e_v_off**2)):.3f}m/s')
    ax.plot(t_on, hist_on['state'][1:, 3], color='tab:blue',
            lw=1.4, label=f'EKF on   RMS={np.sqrt(np.mean(e_v_on**2)):.3f}m/s')
    ax.set_xlabel('t (s)'); ax.set_ylabel('v (m/s)')
    ax.set_title('速度跟踪对比')
    ax.grid(True, alpha=0.3); ax.legend(loc='best')

    # ---------- (1,1) EKF 估计误差 (仅 on) ----------
    ax = axes[1, 1]
    if 'state_est' in hist_on:
        est = hist_on['state_est']
        truth = hist_on['state']
        T = len(truth)
        t = np.arange(T) * dt
        err = est - truth
        err[:, 2] = np.arctan2(np.sin(err[:, 2]), np.cos(err[:, 2]))
        ax.plot(t, err[:, 0], lw=1.2, label=f'x  RMS={np.sqrt(np.mean(err[:,0]**2)):.3f}m')
        ax.plot(t, err[:, 1], lw=1.2, label=f'y  RMS={np.sqrt(np.mean(err[:,1]**2)):.3f}m')
        ax.plot(t, err[:, 3], lw=1.2, label=f'v  RMS={np.sqrt(np.mean(err[:,3]**2)):.3f}m/s')
        # phi 用第二个 y 轴会乱, 单独画 deg
        ax2 = ax.twinx()
        ax2.plot(t, np.rad2deg(err[:, 2]), color='tab:red', lw=1.0, alpha=0.6,
                 label=f'phi  RMS={np.rad2deg(np.sqrt(np.mean(err[:,2]**2))):.2f}°')
        ax2.set_ylabel('phi 误差 (°)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='lower right', fontsize=8)
        ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel('t (s)'); ax.set_ylabel('估计误差 (m, m/s)')
        ax.set_title('EKF 估计 - 真值 (越接近 0 越好)')
        ax.grid(True, alpha=0.3); ax.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'no EKF data', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  对比图已保存: {save_path}')
    return fig


def print_table(scene_title, m_off, m_on):
    print(f"\n=== {scene_title} ===")
    head = f"{'mode':<10}{'steps':<7}{'e_lat_rms':<12}{'e_lat_max':<12}" \
           f"{'e_phi_rms°':<13}{'e_v_rms':<10}"
    print(head)
    print('-' * len(head))
    print(f"{'EKF off':<10}{m_off['steps']:<7}{m_off['e_lat_rms']:<12.4f}"
          f"{m_off['e_lat_max']:<12.4f}{m_off['e_phi_rms_deg']:<13.3f}"
          f"{m_off['e_v_rms']:<10.4f}")
    print(f"{'EKF on':<10}{m_on['steps']:<7}{m_on['e_lat_rms']:<12.4f}"
          f"{m_on['e_lat_max']:<12.4f}{m_on['e_phi_rms_deg']:<13.3f}"
          f"{m_on['e_v_rms']:<10.4f}")
    if 'ekf_x_rms' in m_on:
        print(f"  EKF 估计 RMS: x={m_on['ekf_x_rms']:.3f}m  "
              f"y={m_on['ekf_y_rms']:.3f}m  "
              f"phi={m_on['ekf_phi_rms_deg']:.2f}°  "
              f"v={m_on['ekf_v_rms']:.3f}m/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['all'] + list(SCENE_REGISTRY.keys()),
                        default='all')
    parser.add_argument('--controller', choices=['mpc', 'pid', 'lqr'],
                        default='mpc')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--save-dir', default=None)
    args = parser.parse_args()

    todo = list(SCENE_REGISTRY.keys()) if args.case == 'all' else [args.case]
    figs = []
    for sname in todo:
        scene_fn = SCENE_REGISTRY[sname]
        print(f"\n--- 跑 {sname} (controller={args.controller}) ---")
        scene, hist_off, hist_on = run_pair(scene_fn, args.controller)
        m_off = metrics_of(hist_off)
        m_on  = metrics_of(hist_on)
        print_table(scene.title, m_off, m_on)
        save_path = (f"{args.save_dir}/compare_ekf_{sname}.png"
                     if args.save_dir else None)
        fig = plot_overlay(scene, hist_off, hist_on,
                           dt=0.1, save_path=save_path)
        figs.append(fig)
        if args.no_show:
            plt.close(fig)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
