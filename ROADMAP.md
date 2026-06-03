# MPC 轨迹跟踪项目扩展路线图

> 本文档是项目从"纯 MPC 跟踪 demo"扩展为"完整自动驾驶决策-规划-控制-估计 pipeline"的分阶段计划。
> 每个阶段独立可交付、可验收、可写入简历。每条任务包含设计决策和面试角度，便于边做边理解。

---

## 0. 现状盘点（baseline）

当前已有：

| 模块 | 文件 | 功能 |
|---|---|---|
| 车辆模型 | `vehicle_model.py` | 后轴中心运动学自行车，离散欧拉积分 + 一阶线性化 |
| 参考轨迹 | `reference_trajectory.py` | 直线 / 圆 / 路点三种生成器，弧长重采样，曲率平滑，按时间取窗口 |
| MPC 控制器 | `mpc_controller.py` | cvxpy 参数化 QP，OSQP 求解，warm start，~10 ms/step |
| 仿真 + 可视化 | `main.py`, `animate.py` | 闭环仿真，6 面板分析图，车体多边形动画 |

baseline 的关键设计（已实现，**面试要会讲**）：
- 状态：`[x, y, phi, v]`，控制：`[a, delta]`
- 离散线性化提供 `A_d, B_d, c_d`，仿射项 `c_d` 吸收线性化偏差
- 参考窗口按"未来 N 个 dt"取（不是按 idx），保证位置和速度时刻一致
- `delta_ref = arctan(L * kappa)` 作为前馈，让 QP 只解扰动
- QP 用 cp.Parameter 一次构建多次复用，避免重复 canonicalization

---

## 阶段 0：预备重构（开工必做）

### 为什么需要这步

baseline 代码本身写得很干净，但**接口和组织方式是为单控制器/无估计/无避障设计的**。直接在上面加阶段 1-6 的功能，会反复改同一处接口。提前重构 4-5 小时，能让后面所有阶段都顺。

### 目标
把现有代码从"MPC 单一专用"改造成"可扩展的控制-估计-规划框架"，但**功能行为不变**——baseline 三个场景跑出来的轨迹和误差曲线必须和重构前一致（regression test）。

### 涉及文件
- 重组：把现有文件搬进新目录
- 新增：`controllers/base.py`、`scenarios.py`、`viz/`
- 修改：`main.py`、`mpc_controller.py` 的接口

### 重构清单

#### R1. 目录结构调整
```
improve/
  core/
    __init__.py
    vehicle_model.py        # 搬过来，不改
    reference_trajectory.py # 搬过来，不改
  controllers/
    __init__.py
    base.py                 # 新增 BaseController
    mpc.py                  # 现 mpc_controller.py 搬进来 + 改接口
  viz/
    __init__.py
    plot.py                 # 现 main.py 的 plot_results / compute_errors
    animate.py              # 搬过来，不改
  scenarios.py              # 现 main.py 的 make_*_scene 改成 dataclass
  main.py                   # 只剩 argparse + run_simulation
  ROADMAP.md
```

#### R2. 抽 BaseController 接口
```python
# controllers/base.py
class BaseController:
    def solve(self, state, ref, nearest_idx, u_prev=None):
        """
        state       : np.ndarray (4,)  当前车辆状态
        ref         : ReferenceTrajectory 实例（不切窗口，控制器自己取）
        nearest_idx : int  最近参考点索引（外面已算好，避免重复搜）
        u_prev      : np.ndarray (2,) 或 None  上一拍施加的控制
        return      : (u, info) ; info 必含 'solve_time'，可选 'x_pred', 'cost', 'status'
        """
        raise NotImplementedError
```
MPC 适配后，PID/LQR 沿用同一接口。

#### R3. MPC.solve 接口改造
- 旧签名：`solve(x0, x_ref_window, u_prev=None, u_ref_window=None)`
- 新签名：`solve(state, ref, nearest_idx, u_prev=None)`
- 参考窗口（`x_ref_window`、`u_ref_window`）的取法挪进 MPC 内部（现在在 main.py 里）
- 删掉 `self.u_prev` 这个内部状态，u_prev 只从外部传入

#### R4. run_simulation 解耦
- 接受 `controller` 对象，循环里只调 `controller.solve(...)`
- `hist` 中 MPC 特有字段（`x_pred`, `cost`, `status`）改成 optional，PID/LQR 时填 None
- 控制器无关的字段：`state`, `u`, `ref_state`, `ref_idx`, `solve_time`

#### R5. plot_results 与控制器解耦
- 当前 `plot_results(hist, ref, mpc, title)` 用了 `mpc.a_max / a_min / delta_max`、`mpc.dt`
- 改成显式传入约束（或从一个 `ControlLimits` dataclass 拿）：
  ```python
  plot_results(hist, ref, dt, limits, title)
  ```
- `compute_errors` 挪到 `viz/plot.py` 或独立 `metrics.py`

#### R6. Scenario dataclass
```python
# scenarios.py
@dataclass
class Scenario:
    name: str
    ref: ReferenceTrajectory
    init_state: np.ndarray
    obstacles: list = field(default_factory=list)   # 阶段 3 用
    enable_ekf: bool = False                         # 阶段 2 用
    goal: np.ndarray = None                          # 阶段 4 用

def straight_scene() -> Scenario: ...
def circle_scene() -> Scenario: ...
def lane_change_scene() -> Scenario: ...
```
后面阶段加新字段时，老场景用默认值即可，不用动。

### 实现要点
- **每改一步立刻跑 baseline**，确认轨迹完全一致（hash 对比 hist['state'] 也行）
- import 路径全部改成 `from core.vehicle_model import ...`、`from controllers.mpc import ...`
- `__init__.py` 暴露常用类：`from controllers import MPCController`

### 坑
- 目录搬动后 `if __name__ == "__main__":` 自测里的相对 import 会坏，要改成绝对 import 并加 `sys.path` 或者用 `python -m core.vehicle_model` 跑
- `plot_results` 解耦时容易漏掉某些字段，建议跑一遍三场景把图保存下来，肉眼对比重构前后

### 验收标准
- 三场景跑出来的 `e_lat RMS / max`, `e_phi RMS`, `e_v RMS` 与重构前**完全一致**（数值精度内）
- 6 面板图肉眼看不出差异
- `python -m core.vehicle_model`、`python -m controllers.mpc` 自测能跑
- `from controllers.base import BaseController` 接口已就位（PID/LQR 后续直接继承）

### 面试角度
- "为什么先重构？" → 接口设计的提前投资，避免后续反复改；YAGNI 不是借口（这里有明确的扩展需求）
- "BaseController 抽象的好处？" → 开闭原则、单元测试隔离、对比实验可复现
- "状态外置（u_prev）的好处？" → 纯函数更易测试，无副作用，多实例切换干净

---

## 阶段 1：控制器对比（PID + LQR）

### 目标
基于阶段 0 的 BaseController 接口，实现 PID 和 LQR 两个控制器，与 MPC 在同样场景下对比性能。

### 涉及文件（新增）
```
controllers/
    pid.py              # 新增 PIDController(BaseController)
    lqr.py              # 新增 LQRController(BaseController)
compare.py              # 多控制器同场景对比脚本
```

### 设计决策

**1. 控制器统一接口**
```python
class BaseController:
    def solve(self, state, ref_window, u_prev=None, u_ref_window=None):
        """返回 (u, info)，info 至少含 status、solve_time"""
```
所有控制器实现这个接口，main.py 加 `--controller {mpc,pid,lqr}` 切换。

**2. PID 设计：纵横解耦**
- 横向：用 Stanley 或 PID(e_lat) + PD(e_phi) → `delta`
  - **推荐 Stanley**：`delta = e_phi + arctan(k * e_lat / v)`，单参数 k，工程界用得最多
  - 加上前馈 `delta_ref = arctan(L * kappa)`，弯道跟踪才不会滞后
- 纵向：PI(e_v) → `a`
- 输出做 saturation（`a_min/a_max`、`±delta_max`）

**3. LQR 设计：误差状态 LQR（推荐）**
- 误差状态 `e = [e_lat, e_phi, e_v]`（车辆相对参考点的 Frenet 偏差）
- 在参考点局部线性化得误差动力学 `e_{k+1} = Ã·e_k + B̃·δu_k`
- 每步解一次 DARE 得增益 `K`，`u = u_ref - K·e`
- `u_ref` 复用 baseline 的 `[0, arctan(L*kappa)]`

> 备选：直接对 `[x, y, phi, v]` 做 LQR + 前馈，但坐标系不直观，对比时不如误差 LQR 干净。

### 实现要点
- DARE 用 `scipy.linalg.solve_discrete_are`
- `Q, R` 权重对齐 MPC 的 `Q, R`（让对比公平）
- LQR 出来的 `delta` 必须做 saturation，否则圆弧大曲率会爆
- PID 的积分项需要 anti-windup（控制饱和时停止积分）

### 坑
- **公平对比**：MPC 已经调过参，PID/LQR 也必须调到合理水平，否则结论是"调参偏差"不是"算法差异"
- LQR 的误差动力学推导容易错符号，建议先在直线/圆场景单测验证 K 是否合理
- PID 在双移线场景的预瞄不足，会有滞后；MPC 因为有预测窗口天然有预瞄优势——这是 MPC 的核心卖点之一

### 验收标准
- `python compare.py --case lane` 输出三条轨迹叠合图 + 误差对比表
- 三种控制器都能完成 3 个场景（直线/圆/双移线）
- 对比图能清晰看出：MPC 横向 RMS 最小，LQR 次之，PID 在双移线滞后明显
- 求解时间对比：PID < LQR < MPC（数量级差异）

### 面试角度
- "为什么用 MPC 而不是 PID/LQR？" → 显式约束、预测能力、多目标统一优化
- "LQR 和 MPC 关系？" → 无约束、无限时域 LQR 是 MPC 的特殊情况；MPC 终端代价矩阵 `Qf` 通常取 DARE 解
- "Stanley 控制器原理？" → 几何法，前轴对准前方目标点
- "PID 调参顺序？" → P → D → I，anti-windup 的实现
- "DARE 求解原理？" → 迭代 Riccati 方程或矩阵符号函数法

---

## 阶段 2：EKF 状态估计

### 目标
在仿真闭环中插入 EKF：真值仿真 → 加噪声测量 → EKF 估计 → 喂给控制器（替代真值）。

### 涉及文件（新增）
```
estimation/
    __init__.py
    ekf.py              # ExtendedKalmanFilter 类
    sensors.py          # GPS/IMU/轮速 噪声模型
```
修改 `main.py`：在仿真循环里插入 EKF，控制器收到的 state 来自 EKF 而非真值。

### 设计决策

**1. 状态与测量模型**
- 状态：`[x, y, phi, v]`（与车辆模型一致）
- 测量：
  - GPS：`z_gps = [x, y] + N(0, σ_gps²)`，σ_gps ≈ 0.5 m，10 Hz
  - IMU：`z_imu = [phi_dot, a] + 噪声`（这里 phi_dot 通过欧拉积分进 phi），100 Hz
  - 轮速：`z_v = v + 噪声`，σ_v ≈ 0.1 m/s，50 Hz

**2. EKF 流程**
- 预测：`x̂_k+1 = f(x̂_k, u_k)` 用 `vehicle_model.step`，雅可比 `F = A_d`（复用 baseline 的 `linearize`）
- 协方差预测：`P = F P Fᵀ + Q`
- 更新（每个传感器到时执行一次）：
  - `y = z - h(x̂)`，`S = H P Hᵀ + R`
  - `K = P Hᵀ S⁻¹`
  - `x̂ = x̂ + K y`，`P = (I - K H) P`

**3. 多速率融合**
- 不同传感器频率不同 → 不能每步全更新
- 用时间戳调度：每个 `dt` 检查"哪些传感器有新测量"，对应执行更新

### 实现要点
- `phi` 是周期变量，更新时残差 `y_phi` 必须 wrap 到 `[-π, π]`
- 初始 P 给大（比如 `diag([10, 10, 1, 5])`），表示开始不确定
- Q（过程噪声）和 R（测量噪声）调参——Q 大跟得快但抖，Q 小平滑但滞后
- IMU 的加速度测量可作为 `u` 的另一种来源（融合命令值与 IMU 测量），简化版直接用命令值即可

### 坑
- **EKF 开了之后跟踪误差会变大**——这是真实的，正好体现估计的不确定性影响控制
- 雅可比写错是最常见 bug，建议用数值微分单测对比
- 协方差矩阵数值上要保持对称正定，可以 `P = (P + Pᵀ)/2` 周期 reset
- 传感器异步时序，调试用统一时间戳

### 验收标准
- 输出对比图：真值 vs EKF 估计 vs 噪声测量（4 个状态分量分别画）
- 估计误差（与真值差）平均值接近 0，标准差小于测量噪声
- 跟踪误差（用 EKF 状态喂 MPC）与用真值的对比，差距 < 50%
- NEES 检验（normalized estimation error squared）大致在 χ² 分布内（高级验证）

### 面试角度
- "EKF vs UKF 区别？" → 一阶线性化 vs sigma 点采样；UKF 对强非线性更好但计算贵
- "Q 和 R 怎么调？" → Q 反映模型不确定性，R 反映传感器噪声方差（出厂参数或实测）
- "可观性分析？" → 只有 GPS 时 phi 是弱可观的，需要运动激励（车不动时 phi 不收敛）
- "为什么协方差要保持正定？" → 概率解释要求；数值上用 Joseph 形式更稳定
- "传感器异步怎么处理？" → 按时间戳调度更新；或者用 IMU 高频做 dead reckoning，GPS 低频校正

---

## 阶段 3：静态避障

### 目标
在工作区内放置若干圆形障碍物，MPC 在跟踪参考轨迹的同时绕开。

### 涉及文件（新增）
```
obstacles/
    __init__.py
    static.py           # CircleObstacle 类
controllers/mpc.py      # 修改：接受 obstacles 参数，加避障约束/惩罚
viz/
    __init__.py
    plot_obstacles.py   # 在轨迹图和动画上画障碍
```

### 设计决策

避障在 MPC 中是**非凸约束**（要求点在圆外），OSQP（QP 求解器）解不了硬非凸约束。三种主流处理：

| 方案 | 凸性 | 改动 | 鲁棒性 |
|---|---|---|---|
| **软约束惩罚** | 凸（二次） | 最小 | 不保证不撞，靠权重 |
| **半空间线性化** | 凸（每步迭代） | 中 | 接近硬约束，可能震荡 |
| **凸走廊** | 凸 | 大，需规划配合 | 最强，但要 A* 配合 |

**推荐：半空间线性化（CILQR / SCP 思想）**
- 在每步预测点 `x_k`，对每个障碍 `o`，取车-障连线方向 `n = (x_k - x_o) / ||x_k - x_o||`
- 加约束：`n · (x_k - x_o) ≥ r_safe`
- 这是线性约束，QP 能解；下一拍重新线性化（roll-out 风格）
- 第一次迭代用上一帧的预测点做线性化（warm start），收敛快

> 简化版：先做软约束（在 cost 里加 `λ * max(0, r_safe - d_k)²`），改动最小，约 20 行代码

### 实现要点
- `r_safe = r_obs + r_car + margin`，`r_car ≈ L/2`，`margin ≈ 0.3 m`
- 多障碍时约束数 = `N × M`，M 大时 QP 变慢；可以先按距离筛选只保留 top-K 个最近障碍
- 软约束的权重 λ 要大（>1000），否则 MPC 宁愿撞也不绕
- 可视化：在 XY 图上画圆障碍 + 安全圈（虚线表示 r_safe）

### 坑
- 静态避障最常见的是**陷在局部极小**：障碍正前方时左右等概率，MPC 可能左右抖动
  - 缓解：在 cost 里加左右偏向的微小不对称项，或者用规划层提供"应该从哪边绕"
- 半空间线性化在车被障碍包围时会不可行，要加 slack 变量做软化
- 多障碍场景下 OSQP 求解时间可能从 10 ms 涨到 50 ms，记录下来对比

### 验收标准
- 在直线场景中央放 1~3 个障碍，MPC 能绕过且不撞
- 在双移线场景叠加障碍，MPC 能完成换道 + 避障
- 输出避障距离曲线（车到最近障碍的距离），最小距离 ≥ r_safe
- 求解时间对比（无避障 vs 有避障 N 个）

### 面试角度
- "MPC 怎么处理避障？" → 软约束 / 线性化半空间 / 凸走廊三种
- "为什么不用硬约束？" → 非凸，QP 解不了；非线性求解器（IPOPT/acados）能但慢
- "局部极小怎么避免？" → 上层规划层提供 homotopy（绕行方向），或 multi-shooting + 多初值
- "实时性如何保证？" → 早停、热启动、约束剪枝、求解器选择（OSQP 比 quadprog 快）

---

## 阶段 4：A* 全局规划

### 目标
给定起点、终点、障碍地图，A* 生成 waypoints，喂给 `generate_from_waypoints` 做参考轨迹，再由 MPC 跟踪。

### 涉及文件（新增）
```
planning/
    __init__.py
    grid_map.py         # OccupancyGrid 类（栅格化障碍地图）
    astar.py            # A* 搜索算法
    smoothing.py        # 路径平滑（B-spline / Bezier）
```

### 设计决策

**1. 地图表示**
- OccupancyGrid：二维 numpy 数组，`0=free, 1=occupied`
- 分辨率：0.5 m/cell（精度和搜索时间的折中）
- 障碍输入：圆障碍 → 膨胀（车体半径 + margin）→ 栅格化

**2. A* 实现**
- 8 邻域（上下左右 + 4 对角）
- 启发函数：欧氏距离（admissible）；对角距离（更紧但实现复杂）
- 数据结构：openset 用 `heapq`，closed set 用 set
- 节点 cost：`g(n) + h(n)`，g 是真实代价，h 是启发

**3. 路径平滑**
A* 出来的折线**曲率不连续**，直接给 MPC 跟会很难：
- 选项 a：B-spline 拟合（推荐，scipy.interpolate）
- 选项 b：三阶多项式分段
- 选项 c：保持原 waypoints，靠 `generate_from_waypoints` 的 hanning 平滑（最省事但效果有限）

> 进阶：Hybrid A*（考虑车辆运动学，节点带朝向，扩展用 Reeds-Shepp 曲线），输出可执行路径，但代码量翻倍。本阶段不做，作为后续 bonus。

### 实现要点
- A* 节点定义：`(i, j, g, h, parent)`
- 启发用欧氏：`h = sqrt((i-i_goal)² + (j-j_goal)²) * resolution`
- 路径回溯：从 goal 沿 parent 链到 start，反转
- 平滑后的路径要重采样回均匀 ds，再喂 `generate_from_waypoints`

### 坑
- 8 邻域 A* 走对角时实际距离是 `√2 * resolution`，不是 `resolution`，否则 cost 估算错
- 启发函数不 admissible（高估）会导致路径不最优
- B-spline 平滑后可能切到障碍内（平滑参数过大），需后置碰撞检测，不行就降低平滑度
- A* 在大地图（>500x500）会慢，可考虑 JPS（Jump Point Search）加速

### 验收标准
- 在 50x50 地图（1m 分辨率）随机障碍，A* 能在 <100 ms 内找到路径
- 平滑后路径曲率连续，最大曲率 < 1/r_min（r_min 是车的最小转弯半径）
- pipeline 跑通：A* → smoothing → ReferenceTrajectory → MPC，车能从起点开到终点
- 可视化：地图 + 障碍 + A* 折线 + 平滑后路径 + 车实际轨迹四层叠加

### 面试角度
- "A* vs Dijkstra？" → A* 是 Dijkstra + 启发；启发 admissible 时保证最优
- "A* vs RRT？" → A* 完备最优但需离散化；RRT 适合高维连续空间但非最优
- "Hybrid A* 是什么？" → 状态带朝向，扩展用车辆模型，输出可执行路径
- "为什么需要平滑？" → 折线曲率不连续 → 转向不连续 → 控制器跟不上
- "怎么保证规划-控制一致？" → 规划考虑车辆约束（最小转弯半径、加速度）；或上下层迭代

---

## 阶段 5：动态避障

### 目标
在阶段 3 静态避障基础上，把障碍变成运动的（如对向车、行人横穿），MPC 用预测时域内的障碍轨迹做时变约束。

### 涉及文件
```
obstacles/
    dynamic.py          # DynamicObstacle 类，含 predict(k) 方法
controllers/mpc.py      # 修改：避障约束按 stage 取障碍预测位置
```

### 设计决策

**1. 障碍运动模型**
- 简化：恒速直线 `x_o(k) = x_o(0) + v_o * k * dt`
- 进阶：恒加速、转弯（CTRV 模型）
- 预测时域内，每个 stage k 用预测位置 `x_o(k)` 做约束

**2. 预测不确定性**
- 障碍未来位置不确定，可让安全半径随预测步长增大：
  `r_safe(k) = r_obs + r_car + margin + α * k`
- 这就是"不确定性锥"，是无人车业界标准做法

**3. 与静态避障代码复用**
- 把静态障碍当作 `v_o = 0` 的动态障碍
- 静态/动态用同一接口 `obstacle.position_at(k * dt)`

### 实现要点
- 障碍每步先 step 一次更新真实位置（仿真）
- MPC 取每个 stage 的**预测位置**（不是当前位置）做约束
- 多障碍时按当前距离筛选 top-K，但要考虑运动方向（迎面来的优先级高）

### 坑
- 预测时域 N 不够长，看不到对方 → 反应不及时
  - 例：v_obs=10 m/s，dt=0.1，N=15 → 看 1.5s 即 15m，不够远
  - 解决：增大 N 或拉长 dt（但 dt 大会牺牲控制精度）
- "rude prediction"——障碍其实会拐弯，恒速预测会失效
  - 实际系统用学习方法（如 Trajectron++）做 multi-modal 预测
- 死锁问题：两车迎面互让 → 都左 → 还是会撞，需要协议或社会力模型

### 验收标准
- 直线场景，对向车以 5 m/s 迎面来，自车能换道避开
- 横穿场景，行人以 1 m/s 从右侧穿过，自车能减速等待或绕行
- 不确定性锥可视化：动画里画障碍未来位置 + 不确定性圆
- 极限场景测试：多个动态障碍同时出现

### 面试角度
- "动态避障 vs 静态避障核心差异？" → 障碍预测；时变约束
- "为什么用恒速预测？" → 简单、对短预测时域够用；长时域要更复杂模型
- "怎么处理预测不确定性？" → 不确定性锥、CVaR、scenario-based MPC、随机 MPC
- "感知-预测-规划-控制串联怎么解耦？" → 模块接口（消息），但耦合点（如规划-控制一致性）需要联调

---

## 阶段 6：ROS2 包装

### 目标
把整个 pipeline 包成 ROS2 节点，能用 `ros2 launch` 启动，RViz2 可视化。

### 环境前置（**必须先确认**）
- ROS2 Humble (LTS) **不支持 Windows**
- 你的环境是 Win11，三选一：
  - **WSL2 + Ubuntu 22.04**（推荐，最省事，Windows 直接装）
  - **Linux 双系统**（最纯净，但麻烦）
  - **Docker + osrf/ros:humble-desktop**（隔离好，但 GUI 转发要配）

### 涉及文件（新增）
```
mpc_tracking_ws/
  src/
    mpc_tracking/
      mpc_tracking/
        __init__.py
        sim_node.py            # 真值仿真，发 odom/imu/gps
        ekf_node.py            # 状态估计
        planner_node.py        # A* 规划
        controller_node.py     # MPC/PID/LQR
        obstacle_node.py       # 障碍物发布
        # 复用阶段 1-5 的算法模块（直接 import）
      launch/
        full_pipeline.launch.py
      config/
        params.yaml
      rviz/
        default.rviz
      package.xml
      setup.py
      setup.cfg
```

### 设计决策

**1. 节点划分（数据流）**
```
[obstacle_node] → /obstacles ──┐
                               ↓
[planner_node]  ─────── /path → [controller_node] → /cmd_vel
                               ↑
[sim_node] → /odom_truth, /imu, /gps
              ↓
          [ekf_node] → /odom_estimated ─→ [controller_node]
                                       ↓
[sim_node] ←───────────────── /cmd_vel
```

**2. 消息选型**
| Topic | 类型 |
|---|---|
| /odom_truth, /odom_estimated | nav_msgs/Odometry |
| /imu | sensor_msgs/Imu |
| /gps | sensor_msgs/NavSatFix 或自定义 PoseWithCovariance |
| /path | nav_msgs/Path |
| /obstacles | visualization_msgs/MarkerArray + 自定义 ObstacleArray |
| /cmd_vel | ackermann_msgs/AckermannDriveStamped（推荐）或 geometry_msgs/Twist |

**3. 时序**
- sim_node 用 timer (10 Hz) 推进车辆
- ekf_node 触发式（订阅到测量就更新）
- controller_node 用 timer (10 Hz) 求解 MPC
- planner_node 触发式（goal 改变时重规划）

### 实现要点
- 现有 Python 算法模块**不需要改**，节点里 `import` 即可
- 节点构造时实例化算法对象，回调里调用 `solve()`
- TF 树：`map → odom → base_link`，sim_node 发 `odom→base_link`，map 是固定的
- launch 文件统一启动所有节点 + RViz2

### 坑
- **同步循环 → 异步消息驱动**：原来是 for 循环每步 step+solve，现在每个节点独立，需要重新设计时序
- **RViz2 配置**：每个 marker 的 frame_id 要正确，否则不显示
- **ROS2 Python 节点性能**：rclpy 比 rclcpp 慢，但 MPC 求解才是瓶颈，问题不大
- **包间依赖**：ackermann_msgs 不是默认装的，`apt install ros-humble-ackermann-msgs`
- **WSL2 GUI**：Win11 + WSL2 已支持 WSLg，RViz2 直接能显示；Win10 需要 X server

### 验收标准
- `colcon build` 通过
- `ros2 launch mpc_tracking full_pipeline.launch.py` 一条命令起飞
- RViz2 中能看到：地图、障碍、规划路径、车辆、MPC 预测、TF 树
- `ros2 topic echo /cmd_vel` 能看到控制命令
- 录制 rosbag 回放能复现仿真

### 面试角度
- "ROS2 vs ROS1？" → DDS 通信、QoS 策略、生命周期节点、无 master
- "TF 树原理？" → 树状坐标系，自动找最短路径，时间戳插值
- "rclpy vs rclcpp 选型？" → 性能 vs 开发效率；混合方案（节点用 cpp，算法用 py 通过 service 调用）
- "QoS 策略？" → reliable vs best_effort，volatile vs transient_local，深度
- "怎么调试 ROS2 系统？" → ros2 topic echo / ros2 node info / rosbag 回放 / rqt_graph

---

## 阶段顺序与依赖

```
阶段0 (预备重构) ──→ 阶段1 (PID/LQR) ─┐
                 ──→ 阶段2 (EKF)     ─┤
                                      ├─→ 阶段6 (ROS2 包装)
                 ──→ 阶段3 (静态避障) ─┤
                 ──→ 阶段4 (A*)      ─┤
                 ──→ 阶段5 (动态避障) ─┘
                                  ↑
                              依赖阶段3
```

- **阶段 0 是所有后续阶段的前置**，必须先做
- 阶段 1, 2, 3, 4 互相独立，做完阶段 0 后可以任意顺序
- 阶段 5 依赖阶段 3（静态避障基础）
- 阶段 6 在所有算法稳定后做，避免重复改 ROS 节点

---

## 通用约定（每阶段都遵守）

1. **每阶段完成后**：
   - 在 baseline 场景跑一遍，确认没有 regression
   - 输出对比图（before/after），保存到 `results/phase_N/`
   - README 更新一段简介

2. **代码风格**：
   - 沿用现有的中文注释风格，关键 trick 必须注释 why
   - 每个新模块带 `if __name__ == "__main__":` 自测

3. **目录重构时机**：
   - 阶段 0 已完成所有目录拆分
   - 之后每阶段按 ROADMAP 里的目录新增即可

4. **测试**：
   - 单元测试不强求，但每个模块的 `__main__` 自测必须能跑
   - 系统级用对比图验证

---

## 简历表述参考

当前已实现（阶段 1–5）可以这样写：

> **自动驾驶轨迹跟踪与决策仿真系统** | Python, cvxpy, OSQP, NumPy/SciPy
> - 实现基于线性化自行车模型的 MPC 控制器，cvxpy 参数化 QP + OSQP 热启动，单步求解 ~5-15 ms
> - 对比 PID / LQR / MPC 在直线、圆、双移线场景的跟踪性能，量化预测控制的预瞄优势
> - 集成扩展卡尔曼滤波器（EKF）异步融合 GPS（5 Hz）/ 轮速（10 Hz）多速率传感器，估计误差 < 0.3 m
> - MPC 增加障碍线性化半空间约束（hard / soft 双模式），支持静态 / 动态避障，验证不确定性锥设计
> - A* 全局规划 + B-spline 平滑生成参考路径，与 MPC 跟踪联合验证
> - （扩展）tabular Q-learning 循迹实验，与经典控制器在同一车辆模型下对照

> ⚠️ 诚信提醒：**ROS2 (Humble) 多节点封装 + RViz2 / rosbag** 是本 ROADMAP 阶段 6 的*规划项*，
> 当前代码**尚未实现**；`estimation/sensors.py` 也只有 GPS + 轮速、**没有 IMU**。写进简历前请确认
> 对应功能已完成，避免面试被追问时表述与代码对不上。
