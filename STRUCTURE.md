# 项目结构

## 目录树

```
MPC/
├── main.py                          # 仿真入口 (CLI)
├── scenarios.py                     # 8 个内置场景的工厂函数 + 注册表
├── compare.py                       # 多控制器对比脚本 (MPC vs PID vs LQR)
├── compare_ekf.py                   # EKF on/off 对比脚本
├── compare_avoid_modes.py           # 避障 hard vs soft 对比脚本
│
├── core/
│   ├── __init__.py
│   ├── vehicle_model.py             # 自行车运动学模型 + 一阶线性化
│   └── reference_trajectory.py      # 参考轨迹生成与查询
│
├── controllers/
│   ├── __init__.py
│   ├── base.py                      # BaseController 抽象 + ControlLimits
│   ├── mpc.py                       # MPC (cvxpy + OSQP, 含避障约束)
│   ├── pid.py                       # PID (Stanley 横向 + PI 纵向)
│   └── lqr.py                       # LQR (误差状态 + DARE 每步重解)
│
├── estimation/
│   ├── __init__.py
│   ├── ekf.py                       # 扩展卡尔曼滤波器
│   └── sensors.py                   # GPS / 轮速传感器仿真
│
├── planning/
│   ├── __init__.py
│   ├── grid_map.py                  # 占据栅格地图
│   ├── astar.py                     # A* 路径搜索 (8 邻域)
│   └── smoothing.py                 # B-spline 平滑 + 等弧长重采样
│
├── obstacles/
│   ├── __init__.py
│   ├── static.py                    # CircleObstacle (静态)
│   └── dynamic.py                   # DynamicObstacle (恒速 CV 模型)
│
├── viz/
│   ├── __init__.py
│   ├── plot.py                      # 6 面板单场景图 + compute_errors
│   ├── animate.py                   # 轨迹动画 (车体 / 障碍 / EKF / 栅格)
│   ├── plot_estimation.py           # EKF 估计误差 4 面板图
│   ├── plot_planning.py             # A* 规划层叠图
│   └── plot_avoidance.py            # 避障 XY + min_dist 时间序列
│
├── rl/                              # 扩展实验: tabular Q-learning 循迹 (复用 core/)
│   ├── line_tracking_env.py         # 离散化直线循迹环境
│   ├── q_learning.py                # Q-learning 训练 + rollout
│   ├── demo_q_learning.py           # 训练 + 出图 demo
│   ├── compare_episode_counts.py    # 不同 episode 数对比
│   └── results/                     # reward 曲线 / 策略热力图
│
├── docs/                            # 交互式流程图 (HTML, 浏览器打开)
│   ├── mpc_flow.html
│   └── astar_mpc_precise.html
│
├── ROADMAP.md                       # 项目分阶段规划文档
├── baseline_metrics.json            # 5 个 baseline 场景的回归数值
├── STRUCTURE.md                     # (本文档)
├── README.md                        # 项目说明
└── results/                         # 输出图/动画 (运行时生成)
    ├── animations/                  # GIF 动画
    ├── phase_1/                     # 控制器对比图
    ├── phase_3/                     # 避障对比图
    ├── phase_4/                     # A* 规划图
    └── phase_5/                     # 动态避障图
```

---

## 文件职责说明

### 根目录

| 文件 | 职责 | 关键 API |
|---|---|---|
| `main.py` | **程序入口** —— `argparse` 解析 CLI、`run_simulation` 跑闭环、`run_case` 调度单场景、`make_controller` 工厂 | `run_simulation(car, ref, controller, ...)`, `make_controller(name, ...)`, `make_ekf_and_sensors(...)` |
| `scenarios.py` | 8 个场景的工厂函数 (`straight_scene` / `circle_scene` / `lane_change_scene` / `accel_cruise_brake_scene` / `serpentine_scene` / `astar_scene` / `obstacle_blocking_scene` / `oncoming_vehicle_scene` / `crossing_pedestrian_scene`)，每个返回一个 `Scenario` dataclass。`SCENE_REGISTRY` 字典是名字 → 工厂的查表 | `Scenario`, `SCENE_REGISTRY` |
| `compare.py` | 同场景下跑 MPC/PID/LQR 三控制器，4 面板对比图 + 终端表格 | `main()` |
| `compare_ekf.py` | 同场景跑 EKF on / off 两次，对比跟踪退化 | `main()` |
| `compare_avoid_modes.py` | 同场景跑 hard + soft (λ=100/1000/10000) 四模式，对比避障策略 | `main()` |
| `baseline_metrics.json` | 5 个 baseline 场景 (line/circle/lane/accel_brake/serpentine) 的精确回归数值 (步数/RMS/终态)，用于改动后的 byte-perfect 校验 | (数据文件) |

### `core/` —— 物理与几何

| 文件 | 职责 | 关键 API |
|---|---|---|
| `vehicle_model.py` | 后轴中心自行车运动学：状态 `[x, y, phi, v]`，控制 `[a, delta]`。提供前向 `step` 和一阶线性化 `linearize` (返回 A_d, B_d, c_d 仿射形式) | `VehicleModel.step(state, u)`, `VehicleModel.linearize(x_ref, u_ref)` |
| `reference_trajectory.py` | 三种参考轨迹生成器 (直线 / 圆 / 路点+B-spline)，弧长重采样，曲率汉宁平滑。**关键** trick：`get_reference_window_by_time` 按"未来 N 个 dt"取窗口而非按 idx 取，避免 MPC 末端漂移 | `ReferenceTrajectory.generate_*()`, `find_nearest()`, `get_reference_window_by_time()`, `get_reference_state()` |

### `controllers/` —— 控制器

所有子类继承 `BaseController`，实现 `solve(state, ref, nearest_idx, u_prev) -> (u, info)`。

| 文件 | 职责 | 关键 API |
|---|---|---|
| `base.py` | 抽象基类 + `ControlLimits` dataclass (a_max/a_min/delta_max) | `BaseController`, `ControlLimits` |
| `mpc.py` | **核心控制器** —— cvxpy 参数化 QP、OSQP 求解、warm start。支持 obstacles + 半空间避障 (hard/soft 双模式) + 不确定性锥 (`alpha_uncert`) + 破对称 hint (`avoid_side`)。所有"会变"的量都是 `cp.Parameter`，单步求解 ~5-15ms | `MPCController(vehicle_model, N=15, obstacles=None, avoidance_mode='hard', ...)` |
| `pid.py` | Stanley 横向 (`-e_phi - arctan(k·e_lat/v)` + 弯道前馈) + PI 纵向 (含 anti-windup) | `PIDController(vehicle_model, k_lat=2.0, ...)` |
| `lqr.py` | 误差状态 LQR：`e=[e_lat, e_phi, e_v]`，每步在参考点局部线性化 + DARE。控制律 `u = u_ref - K·e` | `LQRController(vehicle_model, Q=None, R=None, ...)` |

### `estimation/` —— 状态估计

| 文件 | 职责 | 关键 API |
|---|---|---|
| `ekf.py` | 扩展卡尔曼滤波：`predict` 用真非线性 `f` 推均值 + 用 Jacobian `F` 推协方差；`update_gps` / `update_wheel` 异步触发 | `ExtendedKalmanFilter(vehicle_model, x0, P0, Q)`, `predict(u)`, `update_gps(z, R)`, `update_wheel(z, R)` |
| `sensors.py` | 多速率传感器仿真：`GPSSensor` (位置, 5Hz, σ=0.5m) + `WheelSpeedSensor` (速度, 10Hz, σ=0.1m/s)。`SensorSuite` 把它们打包，每步返回 `Measurements` (任一字段可为 None) | `GPSSensor`, `WheelSpeedSensor`, `SensorSuite`, `Measurements` |

### `planning/` —— 规划

| 文件 | 职责 | 关键 API |
|---|---|---|
| `grid_map.py` | 占据栅格地图 (二维 uint8 数组)。`add_circle` 直接在膨胀后栅格化 (`r_inflate=r_car+margin`)，避免单独膨胀步 | `OccupancyGrid(x_min, x_max, ...)`, `add_circles()`, `world_to_grid()`, `is_free()` |
| `astar.py` | 8 邻域 A* (4 直邻 cost=1, 4 对角 cost=√2)，欧氏启发，heapq + counter 防 tie-break。返回世界坐标路径 + 元数据 | `astar(grid_map, start_xy, goal_xy)` |
| `smoothing.py` | A* 折线 → B-spline 平滑 (`scipy.interpolate.splprep`) → 等弧长重采样 ds。带碰撞检测，撞了减半 s 重试 | `smooth_path(path_xy, ds=0.5, smooth_factor=None, grid_map=None)` |

### `obstacles/` —— 障碍

| 文件 | 职责 | 关键 API |
|---|---|---|
| `static.py` | `CircleObstacle` dataclass + `predict(future_dt)` (恒返回 xy) + `step(dt)` (no-op)。`normalize_obstacles` 把混合 (CircleObstacle / DynamicObstacle / 3-tuple / 5-tuple) 列表归一 | `CircleObstacle`, `normalize_obstacles()` |
| `dynamic.py` | `DynamicObstacle` (恒速 CV 模型)：`predict(future_dt)` 返回 `xy + v·t`，`step(dt)` 推进真实位置。与 `CircleObstacle` 同接口，MPC 不分静/动 | `DynamicObstacle(cx, cy, vx, vy, r)` |

### `viz/` —— 可视化

| 文件 | 职责 | 关键 API |
|---|---|---|
| `plot.py` | `compute_errors` 算横向/航向/速度误差 (Frenet 坐标系)；`plot_results` 画 6 面板单场景图 (XY + 横向/航向/速度 + a/δ 控制量) | `compute_errors(hist)`, `plot_results(hist, ref, dt, limits, title, ...)` |
| `animate.py` | matplotlib `FuncAnimation`，画车体多边形 + 拖尾 + MPC 预测线 + EKF 幻影车 + 静/动障碍 (从 `hist['obs_traj']` 自动取每帧位置) + A* 栅格底图 | `animate_results(hist, ref, dt, ..., scene=None, ...)` |
| `plot_estimation.py` | EKF 估计 vs 真值的 4 面板时间序列 (x/y/phi/v 各一格 + ±2σ 置信带 + 测量散点) | `plot_estimation_results(hist, dt, title, ...)` |
| `plot_planning.py` | A* 规划层叠图：栅格 + 原始障碍圆 + 安全圈 + A* 折线 + B-spline 平滑路径 + 参考轨迹 + 车实际轨迹 | `plot_planning(scene, hist=None, ...)` |
| `plot_avoidance.py` | 避障双面板：上 XY (静态 → 实心圆+r_safe虚线；动态 → 5 时间快照浅深渐变 + 障碍轨迹线) + 下 min_dist 时间序列 (含 r_safe(t) 不确定性锥包络) | `plot_avoidance(hist, ref, obstacles, dt, ...)` |

### `rl/` —— 强化学习扩展（教学性质，非主线）

最小化的 tabular Q-learning 循迹实验，**复用 `core/` 的车辆模型与参考轨迹**，与经典控制器处在同一物理环境下对照。刻意不依赖 Gym，便于读懂 RL 主循环。

| 文件 | 职责 | 关键 API |
|---|---|---|
| `line_tracking_env.py` | 离散化直线循迹环境：误差 `(e_lat, e_phi, e_v)` 分箱成状态，7 档离散转向为动作 | `LineTrackingEnv.reset()`, `.step(action)` |
| `q_learning.py` | Q-learning 训练 + 贪心 rollout 评估 | `train_q_learning(...)`, `rollout(...)` |
| `demo_q_learning.py` | 训练 + 出 reward 曲线 / 策略热力图 / 循迹图 | `python -m rl.demo_q_learning` |
| `compare_episode_counts.py` | 不同 episode 数收敛对比 | `python -m rl.compare_episode_counts` |

### `docs/` —— 交互式流程图

`mpc_flow.html` / `astar_mpc_precise.html` 两个独立 HTML，浏览器打开可看 MPC 求解与 A\*+MPC 联动流程，供 README / 面试讲解配图用。

---

## Entry Point

唯一的程序入口是 `main.py`。所有功能都通过它的 CLI 触发：

```bash
python main.py --case <scene_name> [--controller mpc|pid|lqr] [--ekf] [--mpc-avoid] [--avoid-mode hard|soft] [--lambda-soft <float>] [--animate] [--save-dir <path>]
```

`compare.py`、`compare_ekf.py`、`compare_avoid_modes.py` 是独立的对比脚本，**复用** `main.py` 里的 `run_simulation` 和 `make_controller`，但有自己的 `if __name__ == '__main__'`。

每个模块 (`controllers/mpc.py`、`estimation/ekf.py`、`planning/astar.py`、…) 自带 `if __name__ == "__main__":` 自测，可以单独跑：

```bash
python -m controllers.mpc        # MPC 自测
python -m estimation.ekf          # EKF 自测
python -m planning.astar          # A* 自测
```

---

## 主要调用链

### 1. 一次仿真的数据流（最常见用例）

```
main.py CLI
  → run_case(scene_fn, ...)
      → scene_fn() 返回 Scenario (含 ref, init_state, obstacles?, grid_map?)
      → make_controller('mpc', car, obstacles=..., avoidance_mode=...)
          → MPCController.__init__ → _build_problem (一次性 cvxpy QP 构建)
      → make_ekf_and_sensors(...)  # 可选
      → run_simulation(car, ref, controller, init_state,
                       estimator=ekf, sensors=sensors,
                       obstacles_to_step=ctrl.obstacles)
          ── 每拍 ──
          ├── controller.solve(state_or_estimate, ref, nearest_idx, u_prev)
          │       └── MPC: 取 ref window → 写 Parameter → prob.solve(OSQP)
          │           └── _update_avoidance_params (warm-start 重新线性化半空间)
          ├── car.step(state, u)            # 真值前向推进
          ├── obs.step(dt) for obs in ...   # 动态障碍真实推进 (CV)
          └── ekf.predict(u); ekf.update_gps/wheel(measurement, R)  # 可选
      → compute_errors / plot_results / plot_estimation_results / plot_planning / plot_avoidance
```

### 2. MPC 求解一次的内部流程

```
MPCController.solve(state, ref, nearest_idx, u_prev)
  → ref.get_reference_window_by_time(idx, dt, N+1)   # 按时间取窗口
  → 反推 u_ref (a_ref=0, delta_ref=arctan(L·κ))
  → 写 Parameter: x0_p, x_ref_p, u_ref_p, u_prev_p
  → 每个 stage k 调 vm.linearize(x_ref[k], u_ref[k]) → 写 A_p[k], B_p[k], c_p[k]
  → (有障碍时) _update_avoidance_params(state)
       └── 用 self._x_pred_prev 或 _biased_warm_start 当线性化点
       └── 每个 (k, m) 算 n=(x̂_k - p_obs_k)/||...||, 写 n_p[k][m] 和 b_p[k][m]
  → prob.solve(OSQP, warm_start=True)
  → 缓存 self._x_pred_prev = self.x_var.value
  → 返回 (u_var.value[0], info)
```

### 3. 阶段 4 + 5 完整 pipeline (规划 + 估计 + 避障)

```
astar_scene()
  → OccupancyGrid + add_circles(r_inflate=1.55)
  → astar(grid, start, goal)                    # 折线
  → smooth_path(folded, ds=0.3, smooth_factor=20, grid)  # B-spline
  → ReferenceTrajectory.generate_from_waypoints(smoothed, v_ref, ds=0.1)
  → 把 grid_map / raw_path / smoothed / 障碍打包到 Scenario 上

run_case(astar_scene, use_ekf=True, use_mpc_avoidance=True)
  → make_controller('mpc', car, obstacles=sc.obstacles, avoidance_mode='hard')
  → make_ekf_and_sensors → ekf 用噪声 GPS+轮速, MPC 喂估计值
  → run_simulation: ekf 估计驱动 controller, 障碍约束兜底
  → 同时画 plot_results / plot_estimation / plot_planning / plot_avoidance
```

---

## 模块依赖图（顶向下）

```
main.py / compare*.py
    ↓
scenarios.py ──→ planning/ (仅 astar_scene 用)
    ↓
core/ (vehicle_model, reference_trajectory)
    ↓
controllers/ ──→ obstacles/ (仅 mpc.py 用)
    ↓
estimation/ (独立, 不依赖 controllers)
    ↓
viz/ (从 hist 字典反推, 不依赖具体控制器/估计器实现)
```

`obstacles/` 和 `planning/` 是独立模块，互不依赖；它们都被 `scenarios.py` 和/或 `controllers/mpc.py` 引用。`viz/` 全部模块只读 `hist` 字典 + scenario 属性，不依赖控制器或 estimator 类型，方便扩展。
