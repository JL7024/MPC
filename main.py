"""
闭环仿真入口: VehicleModel + Scenario + Controller (+ 可选 EKF)

用法:
    python main.py                            # 跑所有场景, 默认 MPC, 无 EKF
    python main.py --case lane                # 只跑双移线
    python main.py --controller pid           # 选控制器 mpc/pid/lqr
    python main.py --ekf                      # 启用 EKF 状态估计
    python main.py --animate                  # 额外播放动画

仿真循环对控制器无感, 只调用 BaseController.solve(state, ref, idx, u_prev)。
EKF 通过 estimator 参数注入, 不传就走老路径 (regression byte-perfect)。
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from core.vehicle_model import VehicleModel
from controllers import MPCController, PIDController, LQRController
from estimation import ExtendedKalmanFilter, SensorSuite
from viz.plot import compute_errors, plot_results
from viz.plot_estimation import plot_estimation_results, plot_xy_with_gps
from scenarios import SCENE_REGISTRY


# ==================================================================
# 闭环仿真主循环 (与控制器解耦, 估计器可选)
# ==================================================================

def run_simulation(car, ref, controller, init_state, max_steps=2000,
                   stop_dist_to_end=0.5,
                   estimator=None, sensors=None,
                   obstacles_to_step=None):
    """
    跑一次闭环仿真到车辆到达参考终点 (或超出步数上限)

    参数:
        car              : VehicleModel
        ref              : ReferenceTrajectory
        controller       : BaseController 子类
        init_state       : 车辆初始状态 [x, y, phi, v]
        max_steps        : 最大仿真步数
        stop_dist_to_end : 车辆距参考终点多近时判定完成
        estimator        : ExtendedKalmanFilter 或 None
                           - None: 控制器拿真值 (老行为, regression 一致)
                           - 不为 None: 控制器拿 estimator.x (估计)
        sensors          : SensorSuite 或 None (estimator 不为 None 时必传)
        obstacles_to_step: 阶段 5: 每拍 step(dt) 推进的障碍列表 (静态可包含, 不动)
                           注意: 这些对象与 controller.obstacles 是同一份引用,
                           推进后下次 solve 自动看到, 不需要再传

    返回:
        history : dict
            必有: state / u / ref_state / ref_idx / solve_time / cost / status / x_pred
            estimator 启用时多: state_est / cov_diag / meas_gps / meas_wheel
            障碍非空时多: obs_traj (T+1, M, 2) 每步每障碍的真实位置历史
    """
    use_ekf = estimator is not None
    if use_ekf and sensors is None:
        raise ValueError("estimator 启用时必须提供 sensors")
    if obstacles_to_step is None:
        obstacles_to_step = []

    state = car.reset(*init_state)
    u_prev = None
    last_idx = 0

    hist = {
        'state':      [state.copy()],
        'u':          [],
        'ref_state':  [],
        'ref_idx':    [],
        'solve_time': [],
        'cost':       [],
        'status':     [],
        'x_pred':     [],
    }
    has_pred_all = True

    if use_ekf:
        # 估计器在 t=0 已经初始化; 记录 t=0 的估计 (与 state 同长度 T+1)
        hist['state_est'] = [estimator.state()]
        hist['cov_diag']  = [estimator.cov_diag()]
        hist['meas_gps']   = []   # 长度 T (每步一个或 None)
        hist['meas_wheel'] = []

    if obstacles_to_step:
        # 每个时刻所有障碍的真实 (cx, cy) 位置, 给 plot 画轨迹用. shape (T+1, M, 2)
        hist['obs_traj'] = [
            np.array([o.xy for o in obstacles_to_step])
        ]

    end_xy = ref.points[-1, :2]

    for step in range(max_steps):
        # ---------- 1. 选控制器输入: 真值 or EKF 估计 ----------
        if use_ekf:
            state_for_ctrl = estimator.state()
        else:
            state_for_ctrl = state

        # ---------- 2. find_nearest 用控制器看到的状态 (估计或真值) ----------
        # 注意: EKF 启用时, 控制器看到的是估计值, 它对 ref 的认知也基于此
        #
        # 单调性 + 回退余量: 原版 search_from=last_idx 强制单调前进, 但 EKF 噪声
        # 在闭曲线 (circle) 上会让估计位置瞬时跳跃 → last_idx 累计前推得比真车
        # 进度还快 → 几百步后越过参考末端, 仿真死锁。
        # 修法: 允许 search_from 比 last_idx 后退 N_ROLLBACK 个 cell. 在开放轨迹
        # (line/lane/serpentine/...) 上, 真正最近点本来就在 last_idx 之后, np.argmin
        # 找的还是同一个 idx → 回归 byte-perfect; 只在闭曲线 + 噪声组合才有效。
        N_ROLLBACK = 20   # ds=0.1 → 2m 余量, 足够吸收 GPS 0.5m σ 的瞬时跳跃
        search_from = max(0, last_idx - N_ROLLBACK)
        nearest_idx = ref.find_nearest(
            state_for_ctrl[0], state_for_ctrl[1],
            search_from=search_from,
            search_window=200 + N_ROLLBACK,
        )
        last_idx = nearest_idx

        # ---------- 3. 终止判定: 用真值 (公平度量) ----------
        dist_to_end = np.linalg.norm(state[:2] - end_xy)
        if dist_to_end < stop_dist_to_end:
            print(f"  到达终点 @ step {step}, 距终点 {dist_to_end:.2f}m")
            break

        # ---------- 4. 控制器求解 ----------
        u_opt, info = controller.solve(state_for_ctrl, ref, nearest_idx,
                                        u_prev=u_prev)

        # ---------- 5. 真值推进 ----------
        state = car.step(state, u_opt)
        u_prev = u_opt

        # ---------- 5.5. 障碍真值推进 (阶段 5) ----------
        # 静态障碍 step 是 no-op; 动态障碍按恒速更新 cx, cy.
        # 推进发生在 car.step 之后, EKF 之前 — 时序: 控制基于上一拍状态, 推进
        # 都按 dt 同步发生.
        for obs in obstacles_to_step:
            obs.step(car.dt)

        # ---------- 6. EKF: predict + measure + update ----------
        if use_ekf:
            estimator.predict(u_opt)
            m = sensors.measure(state)
            if m.gps is not None:
                estimator.update_gps(m.gps, sensors.gps.R)
            if m.wheel is not None:
                estimator.update_wheel(m.wheel, sensors.wheel.R)
            hist['meas_gps'].append(m.gps)
            hist['meas_wheel'].append(m.wheel)
            hist['state_est'].append(estimator.state())
            hist['cov_diag'].append(estimator.cov_diag())

        # ---------- 7. 记录 ----------
        hist['state'].append(state.copy())
        hist['u'].append(u_opt.copy())
        hist['ref_state'].append(ref.get_reference_state(nearest_idx))
        hist['ref_idx'].append(nearest_idx)
        hist['solve_time'].append(info.get('solve_time', 0.0) or 0.0)
        hist['cost'].append(info.get('cost', np.nan) or np.nan)
        hist['status'].append(info.get('status', 'n/a'))
        if obstacles_to_step:
            hist['obs_traj'].append(
                np.array([o.xy for o in obstacles_to_step]))

        x_pred = info.get('x_pred')
        if x_pred is None:
            has_pred_all = False
            hist['x_pred'].append(None)
        else:
            hist['x_pred'].append(x_pred)
    else:
        print(f"  达到最大步数 {max_steps}, 仿真中断")

    # 转 numpy
    for key in ('state', 'u', 'ref_state', 'ref_idx', 'solve_time', 'cost'):
        hist[key] = np.asarray(hist[key])
    if use_ekf:
        for key in ('state_est', 'cov_diag'):
            hist[key] = np.asarray(hist[key])
        # meas_gps / meas_wheel 保持 list (含 None), 给 plot 自己处理
    if obstacles_to_step:
        hist['obs_traj'] = np.asarray(hist['obs_traj'])   # (T+1, M, 2)

    if has_pred_all and len(hist['x_pred']) > 0:
        hist['x_pred'] = np.asarray(hist['x_pred'])
    else:
        hist['x_pred'] = None

    return hist


# ==================================================================
# 工厂函数
# ==================================================================

def make_controller(name, vehicle_model, obstacles=None, avoid_side=None,
                    alpha_uncert=0.0, avoidance_mode='hard',
                    lambda_soft=1000.0):
    """
    obstacles      : list of (cx,cy,r)/(cx,cy,vx,vy,r)/CircleObstacle/DynamicObstacle
                     或 None. 只对 MPC 生效, PID/LQR 不支持避障约束 (会忽略).
    avoid_side     : 破对称 hint, 仅 MPC 用. 'left'/'right'/None.
    alpha_uncert   : 不确定性锥增长率 (m/s), 阶段 5 动态障碍场景应 > 0.
    avoidance_mode : 'hard' (B 半空间) | 'soft' (A 二次惩罚)
    lambda_soft    : soft 模式 slack 惩罚权重
    """
    if name == 'mpc':
        N = 15
        # 动态障碍场景 (alpha>0) 默认延长预测时域, 让 MPC 提前看到对向/横穿
        if alpha_uncert > 0:
            N = 25
        return MPCController(vehicle_model=vehicle_model, N=N,
                             obstacles=obstacles, avoid_side=avoid_side,
                             alpha_uncert=alpha_uncert,
                             avoidance_mode=avoidance_mode,
                             lambda_soft=lambda_soft)
    if name == 'pid':
        return PIDController(vehicle_model=vehicle_model)
    if name == 'lqr':
        return LQRController(vehicle_model=vehicle_model)
    raise ValueError(f"未知控制器: {name} (可选: mpc / pid / lqr)")


def make_ekf_and_sensors(vehicle_model, init_state, init_offset=None,
                          ekf_seed=42):
    """
    构造 EKF + SensorSuite, 默认配置参考 ROADMAP 阶段 2。

    init_offset: EKF 初值相对真值的偏差 (4,), 默认给非零偏差让 EKF 真正干活。
                 None → 用 [0.5, -0.3, 0.05, 0.5] 的默认偏差。
    """
    if init_offset is None:
        init_offset = np.array([0.5, -0.3, 0.05, 0.5])
    x0_ekf = np.asarray(init_state, dtype=float) + init_offset
    P0 = np.diag([1.0, 1.0, 0.1, 1.0])
    Q  = np.diag([0.01, 0.01, 0.001, 0.05])
    ekf = ExtendedKalmanFilter(vehicle_model, x0_ekf, P0, Q)
    sensors = SensorSuite(seed=ekf_seed)
    return ekf, sensors


# ==================================================================
# 单场景执行
# ==================================================================

def run_case(scene_fn, controller_name='mpc', use_ekf=False,
             show=True, save_path=None, ekf_save_path=None,
             planning_save_path=None,
             use_mpc_avoidance=False,
             avoidance_mode='hard', lambda_soft=1000.0,
             animate=False, anim_save=None):
    scenario = scene_fn()

    # 透传场景里的障碍 + avoid_side hint 到 MPC (仅当 use_mpc_avoidance 时)
    # 默认 False: 即使场景含障碍 (如 astar_scene), MPC 也只跟踪不约束 → 老行为
    obs_for_mpc = None
    avoid_side = None
    alpha_uncert = 0.0
    if use_mpc_avoidance and getattr(scenario, 'obstacles', None):
        obs_for_mpc = scenario.obstacles
        avoid_side = getattr(scenario, 'avoid_side', None)
        # 场景可指定不确定性锥增长率 (动态障碍场景应 > 0)
        alpha_uncert = getattr(scenario, 'alpha_uncert', 0.0)

    car = VehicleModel(L=2.5, dt=0.1)
    controller = make_controller(controller_name, car,
                                  obstacles=obs_for_mpc,
                                  avoid_side=avoid_side,
                                  alpha_uncert=alpha_uncert,
                                  avoidance_mode=avoidance_mode,
                                  lambda_soft=lambda_soft)
    # 把控制器内部 normalize 后的障碍列表拿出来, 主循环用它 step (同一份对象 →
    # 推进后 MPC 下次 solve 自动看到)
    obs_to_step = (controller.obstacles
                    if (use_mpc_avoidance and hasattr(controller, 'obstacles'))
                    else [])

    estimator, sensors = (None, None)
    if use_ekf:
        estimator, sensors = make_ekf_and_sensors(car, scenario.init_state)

    print(f"\n=== {scenario.title} [controller={controller_name}, "
          f"ekf={'on' if use_ekf else 'off'}] ===")
    print(f"参考轨迹点数: {len(scenario.ref)}, 初始状态: {scenario.init_state}")

    t0 = time.time()
    hist = run_simulation(car, scenario.ref, controller, scenario.init_state,
                          max_steps=3000,
                          estimator=estimator, sensors=sensors,
                          obstacles_to_step=obs_to_step)
    wall_time = time.time() - t0

    e_lat, e_phi, e_v = compute_errors(hist)
    avg_solve = np.mean(hist['solve_time']) * 1000
    max_solve = np.max(hist['solve_time']) * 1000
    print(f"  仿真步数:     {len(hist['u'])}")
    print(f"  墙钟时间:     {wall_time:.2f}s")
    print(f"  平均求解:     {avg_solve:.2f}ms / step")
    print(f"  最大求解:     {max_solve:.2f}ms / step")
    print(f"  横向 RMS:     {np.sqrt(np.mean(e_lat**2)):.3f}m")
    print(f"  横向 max:     {np.max(np.abs(e_lat)):.3f}m")
    print(f"  航向 RMS:     {np.rad2deg(np.sqrt(np.mean(e_phi**2))):.2f}°")
    print(f"  速度 RMS:     {np.sqrt(np.mean(e_v**2)):.3f}m/s")
    if use_ekf:
        # EKF 估计误差 (估计 vs 真值)
        est = hist['state_est']
        truth = hist['state']
        ekf_err = est - truth
        ekf_err[:, 2] = np.arctan2(np.sin(ekf_err[:, 2]), np.cos(ekf_err[:, 2]))
        print(f"  EKF 估计 RMS: x={np.sqrt(np.mean(ekf_err[:,0]**2)):.3f}m  "
              f"y={np.sqrt(np.mean(ekf_err[:,1]**2)):.3f}m  "
              f"phi={np.rad2deg(np.sqrt(np.mean(ekf_err[:,2]**2))):.2f}°  "
              f"v={np.sqrt(np.mean(ekf_err[:,3]**2)):.3f}m/s")

    fig = plot_results(hist, scenario.ref, controller.dt, controller.limits,
                       scenario.title, save_path=save_path)

    fig_ekf = None
    if use_ekf:
        fig_ekf = plot_estimation_results(hist, controller.dt,
                                          scenario.title,
                                          save_path=ekf_save_path)

    # 规划场景: 多画一张 4 层叠加 (栅格/障碍/A*/平滑/参考/实际)
    fig_plan = None
    if hasattr(scenario, 'grid_map'):
        from viz.plot_planning import plot_planning
        fig_plan = plot_planning(scenario, hist=hist,
                                  save_path=planning_save_path)

    # 避障场景 (含 obstacles 但无 grid_map): 画 XY+障碍+min_dist 时间序列
    # 也可在 grid_map 场景额外画一张, 看 r_safe 是否被违反
    fig_avoid = None
    if obs_for_mpc:
        from viz.plot_avoidance import plot_avoidance
        avoid_save = (planning_save_path.replace('_planning', '_avoidance')
                      if planning_save_path else None)
        # 不论 grid_map 场景还是 block/oncoming/crossing 场景, 只要 MPC 加了避障
        # 约束就都画一张. 动态场景靠 hist['obs_traj'] 和 alpha_uncert 自动启用
        # 时变障碍 + r_safe(t) 包络绘制.
        fig_avoid = plot_avoidance(hist, scenario.ref,
                                    controller.obstacles,
                                    controller.dt,
                                    r_car=controller.r_car,
                                    margin=controller.margin,
                                    alpha_uncert=controller.alpha_uncert,
                                    title=scenario.title + ' — 避障',
                                    save_path=avoid_save)

    if animate:
        from viz.animate import animate_results
        anim = animate_results(hist, scenario.ref, controller.dt,
                               scenario.title, save_path=anim_save,
                               scene=scenario,
                               r_car=getattr(controller, 'r_car', 1.25),
                               margin=getattr(controller, 'margin', 0.3))
        if not show:
            plt.close(anim._fig if hasattr(anim, '_fig') else plt.gcf())

    if show:
        plt.show()
    else:
        plt.close(fig)
        if fig_ekf is not None:
            plt.close(fig_ekf)
        if fig_plan is not None:
            plt.close(fig_plan)
        if fig_avoid is not None:
            plt.close(fig_avoid)
    return hist


# ==================================================================
# CLI
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['all'] + list(SCENE_REGISTRY.keys()),
                        default='all', help='要跑的场景')
    parser.add_argument('--controller', choices=['mpc', 'pid', 'lqr'],
                        default='mpc', help='选用的控制器')
    parser.add_argument('--ekf', action='store_true',
                        help='启用 EKF 状态估计 (控制器收估计值而非真值)')
    parser.add_argument('--mpc-avoid', action='store_true',
                        help='MPC 加避障约束; 仅当场景含 obstacles 才有效')
    parser.add_argument('--avoid-mode', choices=['hard', 'soft'], default='hard',
                        help="避障约束类型: hard=B 半空间硬约束, soft=A 二次惩罚 (默认 hard)")
    parser.add_argument('--lambda-soft', type=float, default=1000.0,
                        help='soft 模式 slack 惩罚权重 (越大越接近 hard, 默认 1000)')
    parser.add_argument('--no-show', action='store_true',
                        help='不弹窗显示 (只保存)')
    parser.add_argument('--save-dir', default=None,
                        help='图保存目录 (默认不保存)')
    parser.add_argument('--animate', action='store_true',
                        help='额外播放轨迹跟踪动画')
    parser.add_argument('--anim-save', default=None,
                        help='保存动画到 mp4/gif (需 ffmpeg 或 pillow)')
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    todo = list(SCENE_REGISTRY.keys()) if args.case == 'all' else [args.case]

    for name in todo:
        scene_fn = SCENE_REGISTRY[name]
        suffix = '_ekf' if args.ekf else ''
        save_path = (f"{args.save_dir}/{name}{suffix}.png"
                     if args.save_dir else None)
        ekf_save_path = (f"{args.save_dir}/{name}_ekf_est.png"
                         if args.save_dir and args.ekf else None)
        planning_save_path = (f"{args.save_dir}/{name}_planning.png"
                              if args.save_dir else None)
        anim_save = (f"{args.save_dir}/{name}{suffix}.mp4"
                     if args.save_dir and args.anim_save else args.anim_save)
        run_case(scene_fn, controller_name=args.controller,
                 use_ekf=args.ekf,
                 use_mpc_avoidance=args.mpc_avoid,
                 avoidance_mode=args.avoid_mode,
                 lambda_soft=args.lambda_soft,
                 show=not args.no_show, save_path=save_path,
                 ekf_save_path=ekf_save_path,
                 planning_save_path=planning_save_path,
                 animate=args.animate, anim_save=anim_save)
