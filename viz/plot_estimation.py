"""
状态估计结果可视化

输入: hist 字典需含
    'state'         : (T+1, 4) 真值历史
    'state_est'     : (T+1, 4) EKF 估计历史
    'cov_diag'      : (T+1, 4) 协方差对角线 (各分量方差)
    'meas_gps'      : (T,)     每步 GPS 测量 [(2,) 或 None]
    'meas_wheel'    : (T,)     每步轮速测量 [(1,) 或 None]

输出: 4 面板图
    (0,0) x 真值/估计/GPS 测量 + ±2σ 置信带
    (0,1) y 真值/估计/GPS 测量 + ±2σ
    (1,0) phi 真值/估计 + ±2σ          (无直接测量, 间接可观)
    (1,1) v 真值/估计/轮速测量 + ±2σ

外加一张轨迹叠合图 (XY 平面真值 vs 估计 vs GPS 散点)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_estimation_results(hist, dt, title='', save_path=None):
    """
    画 EKF 估计结果。要求 hist 已含 state_est / cov_diag / meas_gps / meas_wheel。
    """
    truth = np.asarray(hist['state'])         # (T+1, 4)
    est   = np.asarray(hist['state_est'])     # (T+1, 4)
    cov   = np.asarray(hist['cov_diag'])      # (T+1, 4)
    sigma = np.sqrt(cov)
    T = len(truth)
    t = np.arange(T) * dt

    # 把 measurements 列表展平成数组 + mask
    meas_gps = hist.get('meas_gps', [])
    meas_wh  = hist.get('meas_wheel', [])
    gps_t, gps_x, gps_y = [], [], []
    for k, m in enumerate(meas_gps):
        if m is not None:
            # 测量是在 step k 时刻拿的, 对应时间 (k+1)*dt (predict 后才 update)
            gps_t.append((k + 1) * dt)
            gps_x.append(m[0])
            gps_y.append(m[1])
    wh_t, wh_v = [], []
    for k, m in enumerate(meas_wh):
        if m is not None:
            wh_t.append((k + 1) * dt)
            wh_v.append(m[0])
    gps_t, gps_x, gps_y = map(np.array, (gps_t, gps_x, gps_y))
    wh_t, wh_v = map(np.array, (wh_t, wh_v))

    # ============== 4 面板: 各状态分量时间历程 ==============
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'{title}  —  EKF 状态估计', fontsize=14, fontweight='bold')

    labels = ['x (m)', 'y (m)', 'phi (rad)', 'v (m/s)']
    for i, (ax, lbl) in enumerate(zip(axes.flat, labels)):
        ax.plot(t, truth[:, i], 'k-', lw=1.6, label='真值', alpha=0.85)
        ax.plot(t, est[:, i], 'tab:blue', lw=1.4, label='EKF 估计', alpha=0.9)
        # ±2σ 置信带
        ax.fill_between(t, est[:, i] - 2 * sigma[:, i],
                            est[:, i] + 2 * sigma[:, i],
                        color='tab:blue', alpha=0.15, label='±2σ')
        # 测量散点
        if i == 0 and len(gps_t) > 0:
            ax.plot(gps_t, gps_x, 'r.', ms=4, alpha=0.5, label='GPS 测量')
        elif i == 1 and len(gps_t) > 0:
            ax.plot(gps_t, gps_y, 'r.', ms=4, alpha=0.5, label='GPS 测量')
        elif i == 3 and len(wh_t) > 0:
            ax.plot(wh_t, wh_v, 'g.', ms=3, alpha=0.4, label='轮速测量')
        ax.set_xlabel('t (s)'); ax.set_ylabel(lbl)
        # 标题: RMS 估计误差
        err = est[:, i] - truth[:, i]
        if i == 2:   # phi: wrap
            err = np.arctan2(np.sin(err), np.cos(err))
        rms = np.sqrt(np.mean(err ** 2))
        ax.set_title(f'{lbl.split()[0]}  RMS估计误差={rms:.3f}')
        ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  EKF 估计图已保存: {save_path}')
    return fig


def plot_xy_with_gps(hist, ref, title='', save_path=None):
    """XY 平面单图: 真值轨迹 / EKF 估计 / GPS 散点 / 参考"""
    truth = np.asarray(hist['state'])
    est   = np.asarray(hist['state_est'])
    meas_gps = hist.get('meas_gps', [])
    gps_pts = np.array([m for m in meas_gps if m is not None])

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f'{title}  —  EKF XY 轨迹', fontsize=13, fontweight='bold')
    ax.plot(ref.points[:, 0], ref.points[:, 1], 'k--', lw=1.0,
            alpha=0.5, label='参考')
    ax.plot(truth[:, 0], truth[:, 1], 'k-', lw=1.8, label='真值')
    ax.plot(est[:, 0], est[:, 1], 'tab:blue', lw=1.4, label='EKF 估计', alpha=0.9)
    if len(gps_pts) > 0:
        ax.plot(gps_pts[:, 0], gps_pts[:, 1], 'r.', ms=4,
                alpha=0.4, label='GPS 测量')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  XY 估计图已保存: {save_path}')
    return fig
