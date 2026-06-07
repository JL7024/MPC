from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from controllers import LQRController, MPCController, PIDController
from core.reference_trajectory import ReferenceTrajectory
from core.vehicle_model import VehicleModel
from estimation import ExtendedKalmanFilter, SensorSuite
from main import make_ekf_and_sensors, run_simulation
from obstacles import normalize_obstacles
from planning import OccupancyGrid, astar, smooth_path
from scenarios import SCENE_REGISTRY
from viz.plot import compute_errors


ControllerName = Literal["mpc", "pid", "lqr"]
AvoidMode = Literal["none", "hard", "soft"]
ReferenceMode = Literal["preset", "direct"]


class PointModel(BaseModel):
    x: float
    y: float


class StartModel(PointModel):
    theta: float = 0.0
    v: float = 4.0


class BoundsModel(BaseModel):
    x_min: float = -5.0
    x_max: float = 85.0
    y_min: float = -15.0
    y_max: float = 15.0


class ObstacleModel(BaseModel):
    id: str | None = None
    kind: Literal["static", "dynamic"] = "static"
    x: float
    y: float
    r: float = Field(default=1.0, gt=0.0)
    vx: float = 0.0
    vy: float = 0.0


class MpcWeightsModel(BaseModel):
    q_x: float = Field(default=10.0, ge=0.0)
    q_y: float = Field(default=10.0, ge=0.0)
    q_phi: float = Field(default=5.0, ge=0.0)
    q_v: float = Field(default=1.0, ge=0.0)
    r_a: float = Field(default=1.0, ge=0.0)
    r_delta: float = Field(default=10.0, ge=0.0)
    rd_a: float = Field(default=0.1, ge=0.0)
    rd_delta: float = Field(default=10.0, ge=0.0)


class VehicleParamsModel(BaseModel):
    target_speed: float = Field(default=6.0, gt=0.0)
    dt: float = Field(default=0.1, gt=0.0)
    wheelbase: float = Field(default=2.5, gt=0.0)
    a_max: float = Field(default=3.0, gt=0.0)
    a_min: float = -5.0
    delta_max_deg: float = Field(default=30.0, gt=0.0)


class SimulateRequest(BaseModel):
    scene: str = "block"
    controller: ControllerName = "mpc"
    reference_mode: ReferenceMode = "preset"
    astar_enabled: bool = False
    ekf_enabled: bool = False
    avoid_mode: AvoidMode = "hard"
    lambda_soft: float = Field(default=1000.0, ge=0.0)
    horizon: int = Field(default=15, ge=3, le=60)
    alpha_uncert: float = Field(default=0.0, ge=0.0)
    safety_margin: float = Field(default=0.3, ge=0.0)
    car_radius: float = Field(default=1.25, gt=0.0)
    max_steps: int = Field(default=800, ge=20, le=4000)
    start: StartModel = Field(default_factory=lambda: StartModel(x=0.0, y=0.0))
    goal: PointModel = Field(default_factory=lambda: PointModel(x=40.0, y=0.0))
    bounds: BoundsModel = Field(default_factory=BoundsModel)
    obstacles: list[ObstacleModel] = Field(default_factory=list)
    weights: MpcWeightsModel = Field(default_factory=MpcWeightsModel)
    vehicle: VehicleParamsModel = Field(default_factory=VehicleParamsModel)


@dataclass
class BuiltReference:
    ref: ReferenceTrajectory
    raw_path: np.ndarray | None = None
    smooth_path: np.ndarray | None = None
    astar_info: dict | None = None
    smooth_info: dict | None = None


def scene_presets() -> list[dict]:
    presets = []
    for name, factory in SCENE_REGISTRY.items():
        scenario = factory()
        goal_xy = getattr(scenario, "goal", None)
        if goal_xy is None:
            goal_xy = scenario.ref.points[-1, :2]
        obstacles = [_obstacle_tuple_to_dict(i, obs)
                     for i, obs in enumerate(getattr(scenario, "obstacles", []))]
        bounds = _bounds_for_scene(scenario.init_state[:2], goal_xy, obstacles,
                                   scenario.ref.points[:, :2])
        presets.append({
            "name": name,
            "title": _safe_title(getattr(scenario, "title", name)),
            "start": _state_to_start(scenario.init_state),
            "goal": {"x": float(goal_xy[0]), "y": float(goal_xy[1])},
            "obstacles": obstacles,
            "bounds": bounds,
            "astarRecommended": hasattr(scenario, "grid_map"),
            "avoidRecommended": name in {"block", "oncoming", "crossing"},
            "alphaUncert": float(getattr(scenario, "alpha_uncert", 0.0)),
            "targetSpeed": float(np.nanmedian(scenario.ref.points[:, ReferenceTrajectory.IDX_V])),
        })
    return presets


def run_from_request(req: SimulateRequest) -> dict:
    t0 = time.perf_counter()
    car = VehicleModel(L=req.vehicle.wheelbase, dt=req.vehicle.dt)
    built = build_reference(req)
    max_steps = max(req.max_steps, _estimate_max_steps_from_reference(built.ref, req.vehicle.dt))
    obstacle_tuples = [_obstacle_to_tuple(o) for o in req.obstacles]
    use_mpc_avoidance = (
        req.controller == "mpc"
        and req.avoid_mode != "none"
        and len(obstacle_tuples) > 0
    )

    controller = _make_controller(req, car,
                                  obstacle_tuples if use_mpc_avoidance else None)
    if use_mpc_avoidance:
        obstacles_to_step = controller.obstacles
    else:
        obstacles_to_step = normalize_obstacles(obstacle_tuples)

    estimator: ExtendedKalmanFilter | None = None
    sensors: SensorSuite | None = None
    init_state = np.array([req.start.x, req.start.y, req.start.theta, req.start.v],
                          dtype=float)
    if req.ekf_enabled:
        estimator, sensors = make_ekf_and_sensors(car, init_state)

    hist = run_simulation(
        car,
        built.ref,
        controller,
        init_state,
        max_steps=max_steps,
        stop_dist_to_end=1.5,
        stop_min_ref_progress=0.85,
        stop_divergence_dist=8.0,
        stop_divergence_patience=12,
        stop_capture_dist=3.0,
        stop_capture_speed=0.5,
        stop_solver_failure_patience=8,
        estimator=estimator,
        sensors=sensors,
        obstacles_to_step=obstacles_to_step,
    )
    wall_time = time.perf_counter() - t0
    metrics = _compute_metrics(hist, req, controller, wall_time, max_steps)

    return {
        "ok": True,
        "scene": req.scene,
        "controller": req.controller,
        "referenceMode": req.reference_mode,
        "astarEnabled": req.astar_enabled,
        "avoidMode": req.avoid_mode if use_mpc_avoidance else "none",
        "dt": req.vehicle.dt,
        "metrics": metrics,
        "series": _series(hist, req, controller),
        "trajectory": _points_xy(hist["state"]),
        "estimatedTrajectory": _points_xy(hist["state_est"]) if "state_est" in hist else [],
        "reference": _points_xy(built.ref.points),
        "refStates": _records(hist.get("ref_state", np.empty((0, 4))),
                              ["x", "y", "theta", "v"]),
        "controls": _records(hist.get("u", np.empty((0, 2))), ["a", "delta"]),
        "predictions": _predictions(hist.get("x_pred")),
        "obstacleTraj": _obstacle_traj(hist.get("obs_traj")),
        "obstacles": [_obstacle_for_response(o) for o in obstacles_to_step],
        "planning": _planning_response(built),
        "bounds": req.bounds.model_dump(),
    }


def build_reference(req: SimulateRequest) -> BuiltReference:
    if req.astar_enabled:
        return _build_astar_reference(req)
    if req.reference_mode == "preset" and req.scene in SCENE_REGISTRY:
        scenario = SCENE_REGISTRY[req.scene]()
        ref = scenario.ref
        ref.points[:, ReferenceTrajectory.IDX_V] = req.vehicle.target_speed
        return BuiltReference(ref=ref)
    return BuiltReference(ref=_direct_reference(req))


def _build_astar_reference(req: SimulateRequest) -> BuiltReference:
    bounds = req.bounds
    grid = OccupancyGrid(bounds.x_min, bounds.x_max,
                         bounds.y_min, bounds.y_max,
                         resolution=0.5)
    circles = [(o.x, o.y, o.r) for o in req.obstacles]
    grid.add_circles(circles, r_inflate=req.car_radius + req.safety_margin)
    start_xy = np.array([req.start.x, req.start.y], dtype=float)
    goal_xy = np.array([req.goal.x, req.goal.y], dtype=float)
    raw_path, astar_info = astar(grid, start_xy, goal_xy)
    if raw_path is None:
        raise ValueError(f"A* failed: {astar_info.get('reason', 'unknown')}")
    raw_path = _force_path_endpoints(raw_path, start_xy, goal_xy)
    smooth_xy, smooth_info = smooth_path(raw_path, ds=0.3, smooth_factor=20,
                                          grid_map=grid)
    smooth_xy = _force_path_endpoints(smooth_xy, start_xy, goal_xy)
    speed_profile = _terminal_speed_profile(
        req.vehicle.target_speed,
        comfortable_decel=min(abs(req.vehicle.a_min), 2.5),
    )
    ref = ReferenceTrajectory().generate_from_waypoints(
        smooth_xy, v_ref=speed_profile, ds=0.1
    )
    return BuiltReference(ref=ref, raw_path=raw_path, smooth_path=smooth_xy,
                          astar_info=astar_info, smooth_info=smooth_info)


def _direct_reference(req: SimulateRequest) -> ReferenceTrajectory:
    start = np.array([req.start.x, req.start.y], dtype=float)
    goal = np.array([req.goal.x, req.goal.y], dtype=float)
    if np.linalg.norm(goal - start) < 1.0:
        goal = start + np.array([1.0, 0.0])
    speed_profile = _terminal_speed_profile(
        req.vehicle.target_speed,
        comfortable_decel=min(abs(req.vehicle.a_min), 2.5),
    )
    return ReferenceTrajectory().generate_from_waypoints(
        np.vstack([start, goal]),
        v_ref=speed_profile,
        ds=0.1,
        smooth_window=0.0,
    )


def _terminal_speed_profile(target_speed: float, comfortable_decel: float):
    target_speed = float(target_speed)
    decel = max(0.5, float(comfortable_decel))

    def profile(s):
        s = np.asarray(s, dtype=float)
        remaining = np.maximum(0.0, s[-1] - s)
        braking_limit = np.sqrt(2.0 * decel * remaining)
        return np.minimum(target_speed, braking_limit)

    return profile


def _make_controller(req: SimulateRequest, car: VehicleModel, obstacles):
    delta_max = math.radians(req.vehicle.delta_max_deg)
    common = {
        "a_max": req.vehicle.a_max,
        "a_min": req.vehicle.a_min,
        "delta_max": delta_max,
    }
    if req.controller == "mpc":
        w = req.weights
        return MPCController(
            vehicle_model=car,
            N=req.horizon,
            Q=np.diag([w.q_x, w.q_y, w.q_phi, w.q_v]),
            R=np.diag([w.r_a, w.r_delta]),
            Rd=np.diag([w.rd_a, w.rd_delta]),
            obstacles=obstacles,
            r_car=req.car_radius,
            margin=req.safety_margin,
            alpha_uncert=req.alpha_uncert,
            avoidance_mode="soft" if req.avoid_mode == "soft" else "hard",
            lambda_soft=req.lambda_soft,
            **common,
        )
    if req.controller == "pid":
        return PIDController(vehicle_model=car, **common)
    return LQRController(vehicle_model=car, **common)


def _compute_metrics(hist, req: SimulateRequest, controller, wall_time: float,
                     max_steps: int) -> dict:
    solve_time = np.asarray(hist.get("solve_time", []), dtype=float)
    u = np.asarray(hist.get("u", []), dtype=float)
    e_lat, e_phi, e_v = compute_errors(hist) if len(u) else (
        np.array([]), np.array([]), np.array([]))
    clearance = _clearance_series(hist, req, controller)
    status_counts = Counter(str(s) for s in hist.get("status", []))
    path = np.asarray(hist.get("state", np.empty((0, 4))))[:, :2]
    path_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
    final_xy = path[-1] if len(path) else np.array([np.nan, np.nan])
    goal_xy = np.array([req.goal.x, req.goal.y], dtype=float)
    reached_max_steps = len(u) >= max_steps
    return {
        "steps": int(len(u)),
        "maxSteps": int(max_steps),
        "reachedMaxSteps": bool(reached_max_steps),
        "terminationReason": str(hist.get("termination_reason", "unknown")),
        "wallTimeMs": wall_time * 1000.0,
        "totalSolveMs": float(np.sum(solve_time) * 1000.0) if len(solve_time) else 0.0,
        "avgSolveMs": float(np.mean(solve_time) * 1000.0) if len(solve_time) else 0.0,
        "p95SolveMs": float(np.percentile(solve_time, 95) * 1000.0) if len(solve_time) else 0.0,
        "maxSolveMs": float(np.max(solve_time) * 1000.0) if len(solve_time) else 0.0,
        "rmsLat": _rms(e_lat),
        "maxLat": float(np.max(np.abs(e_lat))) if len(e_lat) else 0.0,
        "rmsHeadingDeg": math.degrees(_rms(e_phi)),
        "rmsSpeed": _rms(e_v),
        "pathLength": path_len,
        "finalDistanceToGoal": float(np.linalg.norm(final_xy - goal_xy)) if len(path) else None,
        "finalDistanceToReferenceEnd": float(np.linalg.norm(final_xy - goal_xy)) if len(path) else None,
        "minClearance": float(np.min(clearance)) if len(clearance) else None,
        "collision": bool(len(clearance) and np.min(clearance) < 0.0),
        "statusCounts": dict(status_counts),
    }


def _clearance_series(hist, req: SimulateRequest, controller) -> np.ndarray:
    states = np.asarray(hist.get("state", np.empty((0, 4))))[:, :2]
    if not req.obstacles or len(states) == 0:
        return np.array([])
    obs_traj = hist.get("obs_traj")
    obstacles = getattr(controller, "obstacles", None)
    if not obstacles:
        obstacles = normalize_obstacles([_obstacle_to_tuple(o) for o in req.obstacles])
    clearances = []
    for t, xy in enumerate(states):
        step_clearances = []
        for m, obs in enumerate(obstacles):
            if obs_traj is not None and t < len(obs_traj) and m < obs_traj.shape[1]:
                obs_xy = obs_traj[t, m]
            else:
                obs_xy = obs.xy
            r_safe = obs.r + req.car_radius + req.safety_margin
            step_clearances.append(float(np.linalg.norm(xy - obs_xy) - r_safe))
        if step_clearances:
            clearances.append(min(step_clearances))
    return np.asarray(clearances, dtype=float)


def _estimate_max_steps_from_reference(ref: ReferenceTrajectory, dt: float) -> int:
    if ref.points is None or len(ref.points) < 2:
        return 220
    length = float(ref.points[-1, ReferenceTrajectory.IDX_S])
    speeds = ref.points[:, ReferenceTrajectory.IDX_V]
    positive = speeds[speeds > 0.2]
    v_ref = float(np.median(positive)) if len(positive) else 3.0
    nominal = int(math.ceil(length / max(v_ref, 0.2) / dt))
    return int(np.clip(nominal * 3 + 80, 220, 3000))


def _series(hist, req: SimulateRequest, controller) -> dict:
    u = np.asarray(hist.get("u", []), dtype=float)
    solve = np.asarray(hist.get("solve_time", []), dtype=float) * 1000.0
    clearance = _clearance_series(hist, req, controller)
    if len(u):
        e_lat, e_phi, e_v = compute_errors(hist)
    else:
        e_lat = e_phi = e_v = np.array([])
    n = len(u)
    if len(clearance) > n:
        clearance = clearance[:n]
    t = np.arange(n, dtype=float) * req.vehicle.dt
    return {
        "t": _clean_list(t),
        "eLat": _clean_list(e_lat),
        "eHeadingDeg": _clean_list(np.rad2deg(e_phi)),
        "eSpeed": _clean_list(e_v),
        "clearance": _clean_list(clearance),
        "solveMs": _clean_list(solve),
        "accel": _clean_list(u[:, 0] if len(u) else []),
        "steerDeg": _clean_list(np.rad2deg(u[:, 1]) if len(u) else []),
    }


def _planning_response(built: BuiltReference) -> dict:
    return {
        "rawPath": _points_xy(built.raw_path) if built.raw_path is not None else [],
        "smoothPath": _points_xy(built.smooth_path) if built.smooth_path is not None else [],
        "astarInfo": built.astar_info or {},
        "smoothInfo": built.smooth_info or {},
    }


def _force_path_endpoints(path_xy: np.ndarray, start_xy: np.ndarray,
                          goal_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) == 0:
        return np.vstack([start_xy, goal_xy])
    out = path_xy.copy()
    out[0] = start_xy
    out[-1] = goal_xy
    return out


def _predictions(x_pred) -> list[list[dict]]:
    if x_pred is None:
        return []
    arr = np.asarray(x_pred)
    if arr.ndim != 3:
        return []
    return [[{"x": _num(p[0]), "y": _num(p[1])} for p in frame]
            for frame in arr]


def _obstacle_traj(obs_traj) -> list[list[dict]]:
    if obs_traj is None:
        return []
    arr = np.asarray(obs_traj)
    if arr.ndim != 3:
        return []
    return [[{"x": _num(p[0]), "y": _num(p[1])} for p in frame]
            for frame in arr]


def _records(arr, names: list[str]) -> list[dict]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return []
    return [{name: _num(row[i]) for i, name in enumerate(names)}
            for row in arr]


def _points_xy(arr) -> list[dict]:
    if arr is None:
        return []
    arr = np.asarray(arr)
    if arr.size == 0:
        return []
    return [{"x": _num(row[0]), "y": _num(row[1])} for row in arr]


def _clean_list(values) -> list[float | None]:
    return [_num(v) for v in values]


def _num(value):
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else 0.0


def _obstacle_to_tuple(o: ObstacleModel):
    if o.kind == "dynamic":
        return (o.x, o.y, o.vx, o.vy, o.r)
    return (o.x, o.y, o.r)


def _obstacle_tuple_to_dict(index: int, obs) -> dict:
    if len(obs) == 5:
        x, y, vx, vy, r = obs
        return {
            "id": f"obs-{index}",
            "kind": "dynamic",
            "x": float(x),
            "y": float(y),
            "r": float(r),
            "vx": float(vx),
            "vy": float(vy),
        }
    x, y, r = obs
    return {
        "id": f"obs-{index}",
        "kind": "static",
        "x": float(x),
        "y": float(y),
        "r": float(r),
        "vx": 0.0,
        "vy": 0.0,
    }


def _obstacle_for_response(obs) -> dict:
    data = {
        "x": float(obs.xy[0]),
        "y": float(obs.xy[1]),
        "r": float(obs.r),
        "kind": "dynamic" if hasattr(obs, "velocity") else "static",
    }
    if hasattr(obs, "velocity"):
        data["vx"] = float(obs.velocity[0])
        data["vy"] = float(obs.velocity[1])
    else:
        data["vx"] = 0.0
        data["vy"] = 0.0
    return data


def _state_to_start(state) -> dict:
    return {
        "x": float(state[0]),
        "y": float(state[1]),
        "theta": float(state[2]),
        "v": float(state[3]),
    }


def _bounds_for_scene(start_xy, goal_xy, obstacles, ref_xy) -> dict:
    points = [np.asarray(start_xy), np.asarray(goal_xy)]
    if ref_xy is not None and len(ref_xy):
        points.extend([np.min(ref_xy, axis=0), np.max(ref_xy, axis=0)])
    for obs in obstacles:
        r = obs["r"] + 4.0
        points.extend([np.array([obs["x"] - r, obs["y"] - r]),
                       np.array([obs["x"] + r, obs["y"] + r])])
    pts = np.vstack(points)
    lo = np.min(pts, axis=0) - 5.0
    hi = np.max(pts, axis=0) + 5.0
    if hi[0] - lo[0] < 20.0:
        mid = (hi[0] + lo[0]) / 2.0
        lo[0], hi[0] = mid - 10.0, mid + 10.0
    if hi[1] - lo[1] < 12.0:
        mid = (hi[1] + lo[1]) / 2.0
        lo[1], hi[1] = mid - 6.0, mid + 6.0
    return {
        "x_min": float(lo[0]),
        "x_max": float(hi[0]),
        "y_min": float(lo[1]),
        "y_max": float(hi[1]),
    }


def _safe_title(title: str) -> str:
    return title.encode("ascii", errors="ignore").decode("ascii") or "Scenario"
