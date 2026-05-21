"""
A* 路径搜索 (栅格地图)

接口:
    path = astar(grid_map, start_xy, goal_xy)
        grid_map  : OccupancyGrid
        start_xy  : (x, y) 世界坐标
        goal_xy   : (x, y) 世界坐标
        path      : np.ndarray (M, 2) 世界坐标序列, 起点到终点
                    None 表示无路径

实现要点:
    1. 8 邻域: 4 直邻 cost=res, 4 对角 cost=√2*res
       (如果对角和直邻都给 cost=1, 启发就要相应改成对角距离才一致)
    2. 启发函数 h: 欧氏距离 (admissible, 保证最优)
    3. openset: heapq, 元素 (f, counter, node)  counter 防 f 相等时
       heapq 比较 dict 报 TypeError
    4. closed: set of (i, j)
    5. 路径回溯: parent 字典 {(i,j): (i_parent, j_parent)}
    6. 起点/终点都不能在障碍上 (调用方负责或这里 fallback)

复杂度:
    最坏 O(N log N), N = 自由 cell 数. 50x50 free 地图 < 50ms。
"""

import heapq
import numpy as np


# 8 邻域: (di, dj, step_cost_in_cell_units)
# 注意 cost 是"以 res 为单位"的 1 或 √2, 真实距离 = cost * res
_NEIGHBORS = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, np.sqrt(2)),
    (-1,  1, np.sqrt(2)),
    ( 1, -1, np.sqrt(2)),
    ( 1,  1, np.sqrt(2)),
]


def _heuristic(i1, j1, i2, j2):
    """欧氏距离 (cell 单位); admissible 当 cost 为真实距离时"""
    return np.hypot(i1 - i2, j1 - j2)


def astar(grid_map, start_xy, goal_xy, allow_diag_corner_cut=False):
    """
    A* 主循环。

    参数:
        grid_map               : OccupancyGrid
        start_xy, goal_xy      : (x, y) 世界坐标
        allow_diag_corner_cut  : 是否允许对角穿越两个相邻直邻被占的"窄缝"
                                 默认 False (更安全, 实车工程惯例)

    返回:
        (path_world, info)
            path_world : np.ndarray (M, 2) 世界坐标 (cell 中心), None 表示无解
            info       : dict 含 'expanded' (展开节点数), 'time' (秒), 'reason'
    """
    import time
    t0 = time.perf_counter()

    si, sj = grid_map.world_to_grid(*start_xy)
    gi, gj = grid_map.world_to_grid(*goal_xy)

    # 起终点合法性
    if not grid_map.in_bounds(si, sj):
        return None, {'expanded': 0, 'time': 0.0, 'reason': 'start out of bounds'}
    if not grid_map.in_bounds(gi, gj):
        return None, {'expanded': 0, 'time': 0.0, 'reason': 'goal out of bounds'}
    if grid_map.grid[si, sj] != grid_map.FREE:
        return None, {'expanded': 0, 'time': 0.0, 'reason': 'start in obstacle'}
    if grid_map.grid[gi, gj] != grid_map.FREE:
        return None, {'expanded': 0, 'time': 0.0, 'reason': 'goal in obstacle'}

    # openset: 堆顶 = (f, counter, (i,j))  counter 防 tie-break 时比较元组报错
    counter = 0
    open_heap = []
    g_score = {(si, sj): 0.0}
    parent = {}
    h0 = _heuristic(si, sj, gi, gj)
    heapq.heappush(open_heap, (h0, counter, (si, sj)))
    closed = set()
    expanded = 0

    while open_heap:
        f_cur, _, (ci, cj) = heapq.heappop(open_heap)
        if (ci, cj) in closed:
            continue
        if (ci, cj) == (gi, gj):
            # 找到, 回溯
            path_ij = [(ci, cj)]
            while (ci, cj) in parent:
                ci, cj = parent[(ci, cj)]
                path_ij.append((ci, cj))
            path_ij.reverse()
            path_world = np.array(
                [grid_map.grid_to_world(i, j) for (i, j) in path_ij]
            )
            return path_world, {
                'expanded': expanded,
                'time': time.perf_counter() - t0,
                'reason': 'ok',
                'path_len_cells': len(path_ij),
                'path_cost': g_score[(gi, gj)] * grid_map.res,
            }

        closed.add((ci, cj))
        expanded += 1

        for di, dj, step in _NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if not grid_map.in_bounds(ni, nj):
                continue
            if grid_map.grid[ni, nj] != grid_map.FREE:
                continue
            # corner cutting: 对角移动时, 检查共享的两个直邻是否被占
            if di != 0 and dj != 0 and not allow_diag_corner_cut:
                if grid_map.grid[ci, cj + dj] != grid_map.FREE or \
                   grid_map.grid[ci + di, cj] != grid_map.FREE:
                    continue
            tentative_g = g_score[(ci, cj)] + step
            if tentative_g < g_score.get((ni, nj), np.inf):
                g_score[(ni, nj)] = tentative_g
                parent[(ni, nj)] = (ci, cj)
                f = tentative_g + _heuristic(ni, nj, gi, gj)
                counter += 1
                heapq.heappush(open_heap, (f, counter, (ni, nj)))

    return None, {
        'expanded': expanded,
        'time': time.perf_counter() - t0,
        'reason': 'no path',
    }


# ========================================================================
# 自测: 简单地图 + 中央障碍
# ========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from planning.grid_map import OccupancyGrid

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 测试 1: 空地图, 应直接出对角线
    gm = OccupancyGrid(0, 50, -10, 10, resolution=0.5)
    path, info = astar(gm, (1.0, -8.0), (48.0, 8.0))
    print(f"空地图: 展开 {info['expanded']} 个节点, 耗时 {info['time']*1000:.2f}ms, "
          f"路径 {len(path)} 点")
    assert path is not None

    # 测试 2: 中央 3 障碍, 看绕行
    gm2 = OccupancyGrid(0, 50, -10, 10, resolution=0.5)
    obstacles = [(15, 0, 2.0), (25, -3, 1.5), (35, 2, 2.5)]
    gm2.add_circles(obstacles, r_inflate=1.55)
    path2, info2 = astar(gm2, (1.0, 0.0), (48.0, 0.0))
    print(f"3 障碍: 展开 {info2['expanded']} 个节点, 耗时 {info2['time']*1000:.2f}ms, "
          f"路径长 {info2.get('path_cost', 0):.2f}m")
    assert path2 is not None

    # 测试 3: 不可达 (起点在障碍内)
    path3, info3 = astar(gm2, (15.0, 0.0), (48.0, 0.0))
    print(f"起点在障碍: {info3['reason']}")
    assert path3 is None

    # 测试 4: 100x100 cell 性能
    import time
    gm4 = OccupancyGrid(0, 50, 0, 50, resolution=0.5)   # 100x100
    np.random.seed(0)
    for _ in range(20):
        cx, cy, r = np.random.uniform(5, 45), np.random.uniform(5, 45), \
                    np.random.uniform(1, 3)
        gm4.add_circle(cx, cy, r, r_inflate=1.0)
    t0 = time.perf_counter()
    path4, info4 = astar(gm4, (1.0, 1.0), (48.0, 48.0))
    dt_ms = (time.perf_counter() - t0) * 1000
    if path4 is not None:
        print(f"100x100 + 20 随机障碍: {info4['expanded']} 节点, "
              f"{dt_ms:.1f}ms, 路径 {len(path4)} 点")
    else:
        print(f"100x100 没找到路径: {info4['reason']}")

    # 可视化测试 2
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(gm2.grid, cmap='Greys', origin='lower',
              extent=(gm2.x_min, gm2.x_max, gm2.y_min, gm2.y_max),
              interpolation='nearest', alpha=0.6)
    for cx, cy, r in obstacles:
        ax.add_patch(plt.Circle((cx, cy), r, color='red', fill=False, lw=2))
    ax.plot(path2[:, 0], path2[:, 1], 'b.-', ms=4, lw=1.5, label='A* 路径')
    ax.plot(path2[0, 0], path2[0, 1], 'go', ms=12, label='起点')
    ax.plot(path2[-1, 0], path2[-1, 1], 'r*', ms=14, label='终点')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'A* 在 3 圆障碍中找路径  (展开 {info2["expanded"]} 节点)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.show()
