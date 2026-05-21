"""
路径平滑: A* 折线 → 曲率连续的光滑曲线 (B-spline) → 等弧长重采样

为什么需要:
    A* 输出是 8 邻域折线, 在拐点曲率不连续 (实际是 ±∞ 的脉冲)。
    直接喂给 generate_from_waypoints 即使做 hanning 平滑也会留尖刺,
    MPC 的 delta_ref = arctan(L*kappa) 会跳变 → 转向命令震荡。

方案选型:
    a) B-spline (scipy.interpolate.splprep)  ★ 推荐
       - 参数化曲线, x(t)/y(t) 各自三次 B-spline
       - 平滑因子 s 控制紧贴 vs 平滑的折中
       - 端点 fix (起点/终点保持原位)
    b) 三阶分段多项式: 可控但实现繁
    c) 直接用 generate_from_waypoints 内部的 hanning: 已存在但效果有限

实现要点:
    1. splprep 需 M >= k+1 个点 (k=3 → M>=4); 短路径退化为 cubic 插值
    2. 平滑后做碰撞检测: 沿密集采样点查 grid 是否 free, 命中就减半 s 重试
    3. 重采样到等弧长间距 ds: 先在密集 t 上算累积弧长, 再 np.interp
    4. 保留起点/终点严格不变 (用 splprep 的 nest 参数办不到, 这里手动覆盖端点)
"""

import numpy as np
from scipy.interpolate import splprep, splev


def smooth_path(path_xy, ds=0.5, smooth_factor=None,
                grid_map=None, max_collision_retry=5,
                dense_factor=10):
    """
    把 A* 折线平滑到曲率连续, 再按 ds 等弧长重采样。

    参数:
        path_xy             : (M, 2) A* 输出折线 (世界坐标)
        ds                  : 重采样弧长间距 (m), 与 ReferenceTrajectory 一致 (默认 0.1 ?)
                              这里默认 0.5, 调用方再喂给 generate_from_waypoints
                              做最终 0.1 重采样, 两次平滑更稳。
        smooth_factor       : splprep 的 s 参数. None → M (经验值, 文档推荐)
                              s=0 严格插值; s 越大越平滑越偏离原路径。
        grid_map            : 不为 None 时做碰撞检测; 平滑后路径碰到障碍就减小 s 重试
        max_collision_retry : 减 s 重试次数上限
        dense_factor        : 内部高密度采样倍数 (ds_dense = ds / dense_factor)

    返回:
        smoothed_xy : (M', 2) 平滑后等弧长路径, 端点与输入一致
        info        : dict {'s_used', 'collisions', 'arc_length'}
    """
    path_xy = np.asarray(path_xy, dtype=float)
    M = len(path_xy)
    assert M >= 2, f"路径至少 2 点, 实际 {M}"

    # 短路径: B-spline 退化, 直接线性插值 + 等弧长重采样
    if M < 4:
        return _resample_linear(path_xy, ds), {
            's_used': None, 'collisions': 0,
            'arc_length': _path_length(path_xy),
            'mode': 'linear (M<4)',
        }

    if smooth_factor is None:
        smooth_factor = float(M)

    s_try = smooth_factor
    for retry in range(max_collision_retry + 1):
        try:
            tck, u = splprep([path_xy[:, 0], path_xy[:, 1]],
                             s=s_try, k=3)
        except Exception as e:
            # splprep 偶尔在节点重合时炸, 退化为线性
            return _resample_linear(path_xy, ds), {
                's_used': None, 'collisions': 0,
                'arc_length': _path_length(path_xy),
                'mode': f'linear (splprep failed: {e})',
            }

        # 高密度采样 → 累积弧长 → 等弧长重采样
        u_dense = np.linspace(0.0, 1.0, M * dense_factor)
        x_dense, y_dense = splev(u_dense, tck)
        smoothed = _resample_uniform_arc(np.column_stack([x_dense, y_dense]), ds)

        # 端点对齐: splprep 平滑会让端点偏离, 这里强制贴回
        smoothed[0]  = path_xy[0]
        smoothed[-1] = path_xy[-1]

        # 碰撞检测
        if grid_map is None:
            return smoothed, {
                's_used': s_try, 'collisions': 0,
                'arc_length': _path_length(smoothed),
                'mode': f'spline (s={s_try:.2f})',
            }

        collisions = _count_collisions(smoothed, grid_map)
        if collisions == 0:
            return smoothed, {
                's_used': s_try, 'collisions': 0,
                'arc_length': _path_length(smoothed),
                'mode': f'spline (s={s_try:.2f}, no collision)',
            }

        # 命中障碍, s 减半重试
        s_try *= 0.5

    # 重试光了还撞, 用最后一次结果但报告
    return smoothed, {
        's_used': s_try, 'collisions': collisions,
        'arc_length': _path_length(smoothed),
        'mode': f'spline (s={s_try:.2f}, STILL collides)',
    }


# ========================================================================
# 辅助函数
# ========================================================================

def _path_length(path_xy):
    return float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))


def _resample_uniform_arc(path_xy, ds):
    """沿弧长按 ds 等距重采样 (线性插值)"""
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    s_total = s_cum[-1]
    if s_total < ds:
        # 路径比 ds 还短: 给两个端点
        return path_xy[[0, -1]].copy()
    n_out = max(2, int(np.ceil(s_total / ds)) + 1)
    s_new = np.linspace(0.0, s_total, n_out)
    x_new = np.interp(s_new, s_cum, path_xy[:, 0])
    y_new = np.interp(s_new, s_cum, path_xy[:, 1])
    return np.column_stack([x_new, y_new])


def _resample_linear(path_xy, ds):
    """折线 + 等弧长重采样 (M<4 或 spline 失败时用)"""
    return _resample_uniform_arc(path_xy, ds)


def _count_collisions(path_xy, grid_map):
    """点查 grid: 多少个采样点落在障碍 cell 上"""
    cnt = 0
    for x, y in path_xy:
        i, j = grid_map.world_to_grid(x, y)
        if not grid_map.is_free(i, j):
            cnt += 1
    return cnt


# ========================================================================
# 自测
# ========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from planning.grid_map import OccupancyGrid
    from planning.astar import astar

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 空地图直对角, 平滑后基本是直线
    gm = OccupancyGrid(0, 50, -10, 10, resolution=0.5)
    raw, _ = astar(gm, (1, -8), (48, 8))
    smooth, info = smooth_path(raw, ds=0.3)
    print(f"测试 1 (空地图): 原 {len(raw)} 点 长 {_path_length(raw):.2f}m → "
          f"平滑后 {len(smooth)} 点 长 {info['arc_length']:.2f}m  [{info['mode']}]")

    # 2. 带障碍, 验证平滑后无碰撞
    gm2 = OccupancyGrid(0, 50, -10, 10, resolution=0.5)
    obstacles = [(15, 0, 2.0), (25, -3, 1.5), (35, 2, 2.5)]
    gm2.add_circles(obstacles, r_inflate=1.55)
    raw2, _ = astar(gm2, (1, 0), (48, 0))
    smooth2, info2 = smooth_path(raw2, ds=0.3, grid_map=gm2)
    print(f"测试 2 (3 障碍): 原 {len(raw2)} 点 → 平滑 {len(smooth2)} 点  "
          f"碰撞 {info2['collisions']}  [{info2['mode']}]")

    # 3. 曲率检查: 数值微分算最大曲率, 对照 r_min = L/tan(δ_max)
    L_car, delta_max = 2.5, np.deg2rad(30)
    kappa_max_car = np.tan(delta_max) / L_car      # ~0.231
    dx = np.gradient(smooth2[:, 0])
    dy = np.gradient(smooth2[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    kappa = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + 1e-9) ** 1.5
    print(f"曲率 max={kappa.max():.4f} 1/m, 车辆极限 |kappa|<{kappa_max_car:.4f}, "
          f"{'OK' if kappa.max() < kappa_max_car else 'VIOLATES'}")

    # 4. 短路径退化测试 (M<4)
    short = np.array([[0, 0], [1, 0], [2, 0]])
    s_short, info_s = smooth_path(short, ds=0.5)
    print(f"测试 4 (短路径 M=3): {len(s_short)} 点  [{info_s['mode']}]")

    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上: 路径对比
    ax = axes[0]
    ax.imshow(gm2.grid, cmap='Greys', origin='lower',
              extent=(gm2.x_min, gm2.x_max, gm2.y_min, gm2.y_max),
              interpolation='nearest', alpha=0.5)
    for cx, cy, r in obstacles:
        ax.add_patch(plt.Circle((cx, cy), r, color='red', fill=False, lw=2))
    ax.plot(raw2[:, 0], raw2[:, 1], 'b.-', ms=3, lw=0.8, alpha=0.5,
            label=f'A* 折线 ({len(raw2)} 点)')
    ax.plot(smooth2[:, 0], smooth2[:, 1], 'g-', lw=2,
            label=f'B-spline 平滑 ({len(smooth2)} 点)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title('A* 折线 vs B-spline 平滑')

    # 下: 沿弧长曲率
    ax2 = axes[1]
    s_arc = np.concatenate([[0],
                            np.cumsum(np.linalg.norm(np.diff(smooth2, axis=0),
                                                     axis=1))])
    ax2.plot(s_arc, kappa, 'g-', lw=1.5, label='平滑后曲率')
    ax2.axhline(kappa_max_car, color='r', ls='--', alpha=0.7,
                label=f'车辆极限 κ_max={kappa_max_car:.3f}')
    ax2.set_xlabel('s (m)'); ax2.set_ylabel('|κ| (1/m)')
    ax2.grid(True, alpha=0.3); ax2.legend()
    ax2.set_title('平滑路径曲率 vs 车辆极限')

    plt.tight_layout(); plt.show()
