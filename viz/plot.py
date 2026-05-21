"""
仿真结果可视化与误差度量。

与控制器解耦: 接受 dt 和 ControlLimits, 不依赖具体控制器对象。
"""

import numpy as np
import matplotlib.pyplot as plt


# ==================================================================
# 误差计算
# ==================================================================

def compute_errors(hist):
    """
    从仿真历史计算跟踪误差

    - e_lat: 横向误差 = -sin(phi_ref)*(x - x_ref) + cos(phi_ref)*(y - y_ref)
             左正右负, 是车辆当前位置相对参考点的 Frenet 坐标横向分量
    - e_phi: 航向误差, 归一化到 [-pi, pi]
    - e_v  : 速度误差
    """
    # 车辆状态从 t=1 开始 (t=0 是初值), 和控制量/参考对齐
    car_state = hist['state'][1:]              # shape (T, 4)
    ref_state = hist['ref_state']              # shape (T, 4)

    dx = car_state[:, 0] - ref_state[:, 0]
    dy = car_state[:, 1] - ref_state[:, 1]
    phi_ref = ref_state[:, 2]

    e_lat = -np.sin(phi_ref) * dx + np.cos(phi_ref) * dy
    e_phi = np.arctan2(
        np.sin(car_state[:, 2] - phi_ref),
        np.cos(car_state[:, 2] - phi_ref),
    )
    e_v = car_state[:, 3] - ref_state[:, 3]

    return e_lat, e_phi, e_v


# ==================================================================
# 6 面板对比图
# ==================================================================

def plot_results(hist, ref, dt, limits, title, save_path=None):
    """
    画 6 面板仿真结果

    参数:
        hist      : run_simulation 返回的历史字典
        ref       : ReferenceTrajectory
        dt        : 控制周期 (s), 决定时间轴
        limits    : ControlLimits, 决定约束横线位置
        title     : 图标题
        save_path : 不为 None 则保存
    """
    e_lat, e_phi, e_v = compute_errors(hist)

    u = hist['u']                          # shape (T, 2)
    car_state = hist['state']              # shape (T+1, 4)
    t = np.arange(len(u)) * dt             # 时间轴 (秒)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---------- (0,0) XY 轨迹 ----------
    ax = axes[0, 0]
    ax.plot(ref.points[:, 0], ref.points[:, 1], 'k--',
            lw=1.2, label='参考轨迹', alpha=0.7)
    ax.plot(car_state[:, 0], car_state[:, 1], 'b-',
            lw=1.8, label='车辆')
    ax.plot(car_state[0, 0], car_state[0, 1], 'go',
            ms=10, label='起点', zorder=5)
    ax.plot(car_state[-1, 0], car_state[-1, 1], 'r*',
            ms=14, label='终点', zorder=5)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('行驶轨迹')
    ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(loc='best')

    # ---------- (0,1) 横向误差 ----------
    ax = axes[0, 1]
    ax.plot(t, e_lat, 'b-', lw=1.5)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('t (s)'); ax.set_ylabel('横向误差 (m)')
    ax.set_title(f'横向误差  (RMS={np.sqrt(np.mean(e_lat**2)):.3f}m, '
                 f'最大={np.max(np.abs(e_lat)):.3f}m)')
    ax.grid(True, alpha=0.3)

    # ---------- (0,2) 航向误差 ----------
    ax = axes[0, 2]
    ax.plot(t, np.rad2deg(e_phi), 'b-', lw=1.5)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('t (s)'); ax.set_ylabel('航向误差 (°)')
    ax.set_title(f'航向误差  (RMS={np.rad2deg(np.sqrt(np.mean(e_phi**2))):.2f}°)')
    ax.grid(True, alpha=0.3)

    # ---------- (1,0) 速度跟踪 ----------
    ax = axes[1, 0]
    ax.plot(t, hist['ref_state'][:, 3], 'k--', lw=1.2, label='参考速度')
    ax.plot(t, car_state[1:, 3], 'b-', lw=1.5, label='实际速度')
    ax.set_xlabel('t (s)'); ax.set_ylabel('v (m/s)')
    ax.set_title('速度跟踪')
    ax.grid(True, alpha=0.3); ax.legend()

    # ---------- (1,1) 加速度命令 ----------
    ax = axes[1, 1]
    ax.plot(t, u[:, 0], 'b-', lw=1.5)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(limits.a_max, color='r', ls=':', lw=0.8, alpha=0.5, label='约束上下限')
    ax.axhline(limits.a_min, color='r', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('t (s)'); ax.set_ylabel('a (m/s^2)')
    ax.set_title('加速度控制量')
    ax.grid(True, alpha=0.3); ax.legend(loc='best')

    # ---------- (1,2) 前轮转角命令 ----------
    ax = axes[1, 2]
    delta_max_deg = np.rad2deg(limits.delta_max)
    ax.plot(t, np.rad2deg(u[:, 1]), 'b-', lw=1.5)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(delta_max_deg, color='r', ls=':', lw=0.8, alpha=0.5, label='约束上下限')
    ax.axhline(-delta_max_deg, color='r', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('t (s)'); ax.set_ylabel('δ (°)')
    ax.set_title('前轮转角控制量')
    ax.grid(True, alpha=0.3); ax.legend(loc='best')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  图已保存: {save_path}")
    return fig
