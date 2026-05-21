"""
轨迹跟踪动画 (含静态/动态障碍 + EKF 估计 + A* 栅格背景)

依赖 run_simulation 返回的 hist 字典里有:
    'state'    : (T+1, 4)            车辆历史状态 [x, y, phi, v]
    'x_pred'   : (T, N+1, 4)         每步 MPC 预测序列, 可选 (PID/LQR 为 None)
    'state_est': (T+1, 4)            EKF 估计, 可选
    'obs_traj' : (T+1, M, 2)         障碍真实位置历史, 可选

可选 scene 参数:
    obstacles         : 障碍列表, 用于读 r 和初始位置
    grid_map          : OccupancyGrid, 给 A* 场景画栅格底图
    raw_astar_path    : (M, 2) A* 折线
    smoothed_path     : (M', 2) B-spline 平滑路径
    r_safe_per_obs    : (M,) 每个障碍的 r_safe (车体半径 + margin 后)

用法:
    from viz.animate import animate_results
    anim = animate_results(hist, ref, dt, title, scene=sc)
    plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon, Circle


def _vehicle_polygon(state, length=2.5, width=1.4):
    """根据状态生成一个矩形车体的 4 个角点 (后轴中心为 state 原点)"""
    x, y, phi, _ = state
    rear_overhang = 0.3
    corners_body = np.array([
        [-rear_overhang,        -width / 2],
        [length + rear_overhang, -width / 2],
        [length + rear_overhang,  width / 2],
        [-rear_overhang,         width / 2],
    ])
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    corners_world = (R @ corners_body.T).T + np.array([x, y])
    return corners_world


def animate_results(hist, ref, dt, title='', save_path=None,
                    interval_ms=None, trail_length=None,
                    scene=None,
                    show_ekf=True, show_obstacles=True, show_grid=True,
                    r_car=1.25, margin=0.3):
    """
    返回 FuncAnimation 实例。调用方负责 plt.show() 或保存。

    参数:
        hist          : run_simulation 返回的字典
        ref           : ReferenceTrajectory
        dt            : 仿真步长 (s)
        title         : 图标题
        save_path     : 不为 None 则尝试保存到 mp4/gif
        interval_ms   : 帧间隔(毫秒), 默认按 dt 实时播放
        trail_length  : 轨迹拖尾长度(步), None 表示全程不消失
        scene         : Scenario 对象, 可选;
                        从中取 obstacles / grid_map / smoothed_path 等附加属性
        show_ekf      : EKF 估计存在时是否画半透明"幻影车" (覆盖在真车上)
        show_obstacles: 是否画障碍 (有 obs_traj 自动随时间更新)
        show_grid     : 是否画 A* 栅格背景 (scene.grid_map 存在时)
        r_car, margin : 用于算 r_safe = obs.r + r_car + margin
    """
    car_state = np.asarray(hist['state'])
    x_pred = hist.get('x_pred')
    has_pred = x_pred is not None and len(x_pred) > 0
    if has_pred:
        x_pred = np.asarray(x_pred, dtype=object)   # 可能含 None
    state_est = hist.get('state_est')
    has_ekf = show_ekf and state_est is not None
    if has_ekf:
        state_est = np.asarray(state_est)

    obs_traj = hist.get('obs_traj')
    obstacles = (getattr(scene, 'obstacles', None)
                  if scene is not None else None) if show_obstacles else None
    if obstacles is not None:
        from obstacles import normalize_obstacles
        obstacles = normalize_obstacles(obstacles)

    has_obs = obstacles is not None and len(obstacles) > 0
    has_obs_traj = obs_traj is not None and len(obs_traj) > 0
    if has_obs_traj:
        obs_traj = np.asarray(obs_traj)

    grid_map = (getattr(scene, 'grid_map', None)
                 if (scene is not None and show_grid) else None)
    smoothed_path = (getattr(scene, 'smoothed_path', None)
                      if scene is not None else None)

    T = len(hist['u'])
    if interval_ms is None:
        interval_ms = int(dt * 1000)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # ---------- 静态背景 ----------
    if grid_map is not None:
        ax.imshow(grid_map.grid, cmap='Greys', origin='lower',
                  extent=(grid_map.x_min, grid_map.x_max,
                          grid_map.y_min, grid_map.y_max),
                  interpolation='nearest', alpha=0.3, zorder=0)

    # 参考轨迹 (虚线)
    ax.plot(ref.points[:, 0], ref.points[:, 1], 'k--',
            lw=1.0, alpha=0.5, label='参考', zorder=1)
    ax.plot(ref.points[0, 0], ref.points[0, 1], 'g^', ms=9, alpha=0.5, zorder=2)
    ax.plot(ref.points[-1, 0], ref.points[-1, 1], 'r^', ms=9, alpha=0.5, zorder=2)

    # ---------- 动态元素 ----------
    (trail_line,)   = ax.plot([], [], 'b-', lw=1.5, label='已行驶', zorder=3)
    pred_label = 'MPC 预测' if has_pred else '_nolegend_'
    (pred_line,)    = ax.plot([], [], 'r.-', lw=1.0, ms=2.5, alpha=0.8,
                               label=pred_label, zorder=4)
    car_patch = MplPolygon(_vehicle_polygon(car_state[0]),
                            closed=True, fc='royalblue', ec='navy',
                            alpha=0.9, zorder=6)
    ax.add_patch(car_patch)
    # EKF 估计幻影车
    ekf_patch = None
    if has_ekf:
        ekf_patch = MplPolygon(_vehicle_polygon(state_est[0]),
                                closed=True, fc='orange', ec='darkorange',
                                alpha=0.35, zorder=5)
        ax.add_patch(ekf_patch)
        ax.plot([], [], 's', mfc='orange', mec='darkorange', alpha=0.6,
                 ms=10, label='EKF 估计 (幻影)')

    # 障碍: 每个 (圆 + 安全圈) 都做成可更新的 patch
    obs_circles = []
    if has_obs:
        for o in obstacles:
            r_safe = o.r + r_car + margin
            c_obs = Circle(o.xy, o.r, color='red', fill=True, alpha=0.55,
                            zorder=4)
            c_safe = Circle(o.xy, r_safe, color='red', fill=False,
                             ls='--', lw=1.0, alpha=0.5, zorder=4)
            ax.add_patch(c_obs)
            ax.add_patch(c_safe)
            obs_circles.append((c_obs, c_safe))
        ax.plot([], [], 'r-', lw=8, alpha=0.55, label='障碍')

    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        va='top', fontsize=10,
                        bbox=dict(boxstyle='round', fc='white', alpha=0.75))

    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    # 边距: 同时考虑 ref 和障碍范围
    pad = 3.0
    xs = [ref.points[:, 0].min(), ref.points[:, 0].max()]
    ys = [ref.points[:, 1].min(), ref.points[:, 1].max()]
    if has_obs_traj:
        xs += [obs_traj[..., 0].min(), obs_traj[..., 0].max()]
        ys += [obs_traj[..., 1].min(), obs_traj[..., 1].max()]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    def init():
        trail_line.set_data([], [])
        pred_line.set_data([], [])
        time_text.set_text('')
        artists = [trail_line, pred_line, car_patch, time_text]
        if ekf_patch is not None: artists.append(ekf_patch)
        for co, cs in obs_circles:
            artists += [co, cs]
        return artists

    def update(frame):
        # 拖尾
        if trail_length is None:
            trail = car_state[: frame + 2]
        else:
            start = max(0, frame + 1 - trail_length)
            trail = car_state[start: frame + 2]
        trail_line.set_data(trail[:, 0], trail[:, 1])

        if has_pred:
            pred = x_pred[frame]
            if pred is not None and not np.any(np.isnan(np.asarray(pred, dtype=float))):
                pred_arr = np.asarray(pred)
                pred_line.set_data(pred_arr[:, 0], pred_arr[:, 1])
            else:
                pred_line.set_data([], [])
        else:
            pred_line.set_data([], [])

        car_patch.set_xy(_vehicle_polygon(car_state[frame + 1]))

        if ekf_patch is not None:
            ekf_patch.set_xy(_vehicle_polygon(state_est[frame + 1]))

        # 障碍位置 (动态时随时间更新)
        if has_obs and has_obs_traj:
            for m, (c_obs, c_safe) in enumerate(obs_circles):
                pos = obs_traj[frame + 1, m]    # 推进后的位置
                c_obs.center = (pos[0], pos[1])
                c_safe.center = (pos[0], pos[1])

        v = car_state[frame + 1, 3]
        time_text.set_text(f't = {(frame + 1) * dt:.2f} s\n'
                            f'v = {v:.2f} m/s')

        artists = [trail_line, pred_line, car_patch, time_text]
        if ekf_patch is not None: artists.append(ekf_patch)
        for co, cs in obs_circles:
            artists += [co, cs]
        return artists

    anim = FuncAnimation(fig, update, frames=T, init_func=init,
                         interval=interval_ms, blit=False, repeat=False)
    anim._fig = fig

    if save_path:
        try:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=int(1 / dt))
            else:
                anim.save(save_path, writer='ffmpeg', fps=int(1 / dt))
            print(f"  动画已保存: {save_path}")
        except Exception as e:
            print(f"  动画保存失败({e}), 继续显示")

    return anim


# 自测
if __name__ == "__main__":
    print("animate.py 是工具模块, 请通过 main.py --animate 调用。")
