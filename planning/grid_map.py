"""
2D 占据栅格地图 (OccupancyGrid)

存储:
    0 = free, 1 = occupied   (uint8 ndarray)
    形状 (n_y, n_x), 即 grid[i, j] 中 i 是行索引(y 方向), j 是列索引(x 方向)

世界坐标系约定:
    cell (i, j) 的中心在世界坐标 (x_min + (j+0.5)*res, y_min + (i+0.5)*res)
    相邻 cell 的世界距离 = res; 对角 cell 的世界距离 = sqrt(2)*res

为什么这样安排索引:
    matplotlib imshow / numpy 的内存连续性都是 (row, col) = (y, x),
    所以 grid[i,j] 用 (y_idx, x_idx) 顺手, 同时和 OpenCV/图像处理一致。

膨胀 (inflate):
    A* 用的是车形质点假设 → 障碍要预先膨胀 r_car + margin。
    用迭代膨胀: 每个被占 cell 把它周围半径 r 的所有 cell 也标占。
    这一步在 add_circle 里直接做(传 r_inflate), 不需要单独 inflate 步。
"""

import numpy as np


class OccupancyGrid:
    """
    占据栅格地图。

    用法:
        gm = OccupancyGrid(x_min=0, x_max=50, y_min=-10, y_max=10, resolution=0.5)
        gm.add_circle(20.0, 2.0, r=2.0, r_inflate=1.55)   # 障碍 + 车体半径膨胀
        i, j = gm.world_to_grid(15.0, 0.0)
        if gm.is_free(i, j):
            ...
    """

    FREE = 0
    OCCUPIED = 1

    def __init__(self, x_min, x_max, y_min, y_max, resolution=0.5):
        """
        参数:
            x_min, x_max, y_min, y_max : 世界坐标范围 (米)
            resolution : 每 cell 边长 (米); 越小越精细但 A* 越慢

        n_x = ceil((x_max - x_min) / res), 同理 n_y。
        最大值取闭区间, 即 cell 边界刚好压在 x_max/y_max 上时也算地图内。
        """
        assert x_max > x_min and y_max > y_min, "bounds 不合法"
        assert resolution > 0
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.res = float(resolution)

        self.n_x = int(np.ceil((x_max - x_min) / resolution))
        self.n_y = int(np.ceil((y_max - y_min) / resolution))
        self.grid = np.zeros((self.n_y, self.n_x), dtype=np.uint8)

    # =====================================================================
    # 坐标转换
    # =====================================================================

    def world_to_grid(self, x, y):
        """
        世界坐标 → cell 索引 (i, j). 不做边界检查, 调用方需 in_bounds.

        用 floor 而不是 round: 落在 cell 内的点就属于该 cell, 落在边界上属于
        +x/+y 方向的下一个 cell (惯例)。
        """
        j = int(np.floor((x - self.x_min) / self.res))
        i = int(np.floor((y - self.y_min) / self.res))
        return i, j

    def grid_to_world(self, i, j):
        """cell (i, j) 的中心点世界坐标"""
        x = self.x_min + (j + 0.5) * self.res
        y = self.y_min + (i + 0.5) * self.res
        return x, y

    def in_bounds(self, i, j):
        return 0 <= i < self.n_y and 0 <= j < self.n_x

    def is_free(self, i, j):
        """超界视为非 free (相当于把地图外围当作墙)"""
        if not self.in_bounds(i, j):
            return False
        return self.grid[i, j] == self.FREE

    # =====================================================================
    # 添加障碍
    # =====================================================================

    def add_circle(self, cx, cy, r, r_inflate=0.0):
        """
        添加一个圆形障碍。世界坐标系下圆心 (cx, cy), 半径 r。
        r_inflate: 在 r 之外再额外膨胀的距离 (车体半径 + margin)。

        实现: 直接遍历可能受影响的 cell 块, O((r+r_inflate)^2 / res^2),
              对单个圆很快; 多个圆累加即可。
        """
        r_eff = r + r_inflate
        # 影响的 cell 范围 (用 cell 中心是否在 r_eff 内判定)
        x_lo, x_hi = cx - r_eff, cx + r_eff
        y_lo, y_hi = cy - r_eff, cy + r_eff
        i_lo, j_lo = self.world_to_grid(x_lo, y_lo)
        i_hi, j_hi = self.world_to_grid(x_hi, y_hi)
        i_lo = max(0, i_lo); j_lo = max(0, j_lo)
        i_hi = min(self.n_y - 1, i_hi); j_hi = min(self.n_x - 1, j_hi)

        # 向量化: 在范围块内一次算所有 cell 中心到圆心的距离
        ii, jj = np.meshgrid(np.arange(i_lo, i_hi + 1),
                             np.arange(j_lo, j_hi + 1), indexing='ij')
        cx_block = self.x_min + (jj + 0.5) * self.res
        cy_block = self.y_min + (ii + 0.5) * self.res
        d2 = (cx_block - cx) ** 2 + (cy_block - cy) ** 2
        mask = d2 <= r_eff ** 2
        self.grid[i_lo:i_hi + 1, j_lo:j_hi + 1][mask] = self.OCCUPIED

    def add_circles(self, circles, r_inflate=0.0):
        """批量加障碍. circles: [(cx, cy, r), ...]"""
        for cx, cy, r in circles:
            self.add_circle(cx, cy, r, r_inflate=r_inflate)

    # =====================================================================
    # 工具
    # =====================================================================

    def occupancy_ratio(self):
        return float(self.grid.sum()) / self.grid.size

    def __repr__(self):
        return (f"OccupancyGrid(n=({self.n_y},{self.n_x}), res={self.res}, "
                f"occ={self.occupancy_ratio()*100:.1f}%)")


# ========================================================================
# 自测: 建图 + 加障碍 + 坐标转换正确性
# ========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 50m x 20m 工作区, 0.5m 分辨率
    gm = OccupancyGrid(x_min=0, x_max=50, y_min=-10, y_max=10, resolution=0.5)
    print(gm, f"  期望 n_y=40, n_x=100  实际 ({gm.n_y}, {gm.n_x})")

    # 加 3 个圆障碍, 车体半径 1.25m + 0.3m margin = 1.55m 膨胀
    obstacles = [(15, 0, 2.0), (25, -3, 1.5), (35, 2, 2.5)]
    gm.add_circles(obstacles, r_inflate=1.55)
    print(f"加 3 障碍后: {gm}")

    # 坐标转换往返一致性 (cell 中心)
    for i, j in [(0, 0), (10, 50), (20, 99)]:
        wx, wy = gm.grid_to_world(i, j)
        i2, j2 = gm.world_to_grid(wx, wy)
        assert (i, j) == (i2, j2), f"往返失败: ({i},{j}) → ({wx},{wy}) → ({i2},{j2})"
    print("坐标往返一致性: OK")

    # 检查障碍中心 cell 一定是占用的
    for cx, cy, _ in obstacles:
        i, j = gm.world_to_grid(cx, cy)
        assert gm.grid[i, j] == 1, f"障碍中心 ({cx},{cy}) 不是占用?"
    print("障碍中心 cell 占用: OK")

    # 边界外查询应返回非 free
    assert not gm.is_free(-1, 0)
    assert not gm.is_free(0, -1)
    assert not gm.is_free(gm.n_y, 0)
    assert not gm.is_free(0, gm.n_x)
    print("边界外 is_free=False: OK")

    # 可视化
    fig, ax = plt.subplots(figsize=(12, 5))
    # imshow: extent 用世界坐标, origin='lower' 让 y 向上为正
    ax.imshow(gm.grid, cmap='Greys', origin='lower',
              extent=(gm.x_min, gm.x_max, gm.y_min, gm.y_max),
              interpolation='nearest', alpha=0.7)
    # 画原始障碍圆 (未膨胀)
    for cx, cy, r in obstacles:
        circle = plt.Circle((cx, cy), r, color='red', fill=False, lw=2)
        ax.add_patch(circle)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('OccupancyGrid 自测 (灰=膨胀后占用, 红圈=原始障碍)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
