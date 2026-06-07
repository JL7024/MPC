const canvas = document.getElementById("sandbox");
const ctx = canvas.getContext("2d");

const el = (id) => document.getElementById(id);
const controls = {
  sceneSelect: el("sceneSelect"),
  obstacleScene: el("obstacleScene"),
  controller: el("controller"),
  ekfEnabled: el("ekfEnabled"),
  avoidMode: el("avoidMode"),
  capabilityHint: el("capabilityHint"),
  diagnostics: el("diagnostics"),
  comparisonSummary: el("comparisonSummary"),
  horizon: el("horizon"),
  qPos: el("qPos"),
  qPhi: el("qPhi"),
  rDelta: el("rDelta"),
  lambdaSoft: el("lambdaSoft"),
  targetSpeed: el("targetSpeed"),
  aMax: el("aMax"),
  safetyMargin: el("safetyMargin"),
  alphaUncert: el("alphaUncert"),
  selectedRadius: el("selectedRadius"),
  selectedDynamic: el("selectedDynamic"),
  selectedVx: el("selectedVx"),
  selectedVy: el("selectedVy"),
  frameSlider: el("frameSlider"),
  layerReference: el("layerReference"),
  layerRaw: el("layerRaw"),
  layerSmooth: el("layerSmooth"),
  layerTrajectory: el("layerTrajectory"),
  layerPrediction: el("layerPrediction"),
  layerSafety: el("layerSafety"),
  compareControllers: el("compareControllers"),
  compareAvoidance: el("compareAvoidance"),
  sweepHorizon: el("sweepHorizon"),
};

const state = {
  presets: [],
  scene: null,
  start: { x: 0, y: 0, theta: 0, v: 4 },
  goal: { x: 40, y: 0 },
  bounds: { x_min: -5, x_max: 85, y_min: -15, y_max: 15 },
  obstacles: [],
  selected: null,
  drag: null,
  result: null,
  comparison: [],
  frame: 0,
  playing: false,
  lastTick: 0,
};

const TRAJECTORY_OPTIONS = [
  ["line", "直线跟踪"],
  ["circle", "圆轨迹"],
  ["lane", "双移线"],
  ["accel_brake", "加速-巡航-制动"],
  ["serpentine", "S 曲线"],
  ["direct", "起点到终点"],
  ["astar", "A* 全局规划"],
];

const OBSTACLE_OPTIONS = [
  ["none", "无障碍"],
  ["block", "静态障碍"],
  ["oncoming", "对向来车"],
  ["crossing", "横穿行人"],
];

const COMPARISON_COLORS = ["#2563eb", "#0f766e", "#ea580c", "#7c3aed"];

const CHART_SPECS = [
  { id: "chartLat", field: "eLat", color: "#2563eb", xLabel: "时间 (s)", yLabel: "误差 (m)" },
  { id: "chartHeading", field: "eHeadingDeg", color: "#7c3aed", xLabel: "时间 (s)", yLabel: "误差 (deg)" },
  { id: "chartSpeed", field: "eSpeed", color: "#0891b2", xLabel: "时间 (s)", yLabel: "误差 (m/s)" },
  { id: "chartSteer", field: "steerDeg", color: "#16a34a", xLabel: "时间 (s)", yLabel: "转角 (deg)" },
  { id: "chartClearance", field: "clearance", color: "#b45309", xLabel: "时间 (s)", yLabel: "余量 (m)" },
  { id: "chartSolve", field: "solveMs", color: "#ea580c", xLabel: "时间 (s)", yLabel: "耗时 (ms)" },
];

function layerEnabled(name) {
  return controls[name]?.checked ?? true;
}

function syncSliderLabels() {
  const pairs = [
    ["horizon", ""],
    ["qPos", ""],
    ["qPhi", ""],
    ["rDelta", ""],
    ["lambdaSoft", ""],
    ["targetSpeed", " m/s"],
    ["aMax", " m/s2"],
    ["safetyMargin", " m"],
    ["alphaUncert", " m/s"],
  ];
  for (const [id, suffix] of pairs) {
    const value = controls[id].value;
    el(`${id}Value`).textContent = `${value}${suffix}`;
  }
}

function fitCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(500, Math.round(rect.width * dpr));
  const h = Math.max(320, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  draw();
}

function worldToScreen(p) {
  const b = state.bounds;
  const pad = 34;
  const sx = (canvas.width - pad * 2) / (b.x_max - b.x_min);
  const sy = (canvas.height - pad * 2) / (b.y_max - b.y_min);
  const s = Math.min(sx, sy);
  const ox = pad + (canvas.width - pad * 2 - (b.x_max - b.x_min) * s) / 2;
  const oy = pad + (canvas.height - pad * 2 - (b.y_max - b.y_min) * s) / 2;
  return {
    x: ox + (p.x - b.x_min) * s,
    y: canvas.height - (oy + (p.y - b.y_min) * s),
    s,
  };
}

function screenToWorld(x, y) {
  const b = state.bounds;
  const map = worldToScreen({ x: b.x_min, y: b.y_min });
  return {
    x: b.x_min + (x - map.x) / map.s,
    y: b.y_min + (canvas.height - y - (canvas.height - map.y)) / map.s,
  };
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  const result = state.result;
  if (state.comparison.length) {
    drawComparison();
  } else {
    if (result?.planning?.rawPath?.length && layerEnabled("layerRaw")) drawPath(result.planning.rawPath, "#94a3b8", 1.4, [5, 5]);
    if (result?.planning?.smoothPath?.length && layerEnabled("layerSmooth")) drawPath(result.planning.smoothPath, "#16a34a", 2.2, []);
    if (result?.reference?.length && layerEnabled("layerReference")) drawPath(result.reference, "#111827", 1.6, [8, 6]);
    if (result?.estimatedTrajectory?.length) drawPath(result.estimatedTrajectory.slice(0, state.frame + 1), "#a855f7", 1.8, []);
    if (result?.trajectory?.length && layerEnabled("layerTrajectory")) drawPath(result.trajectory.slice(0, state.frame + 1), "#2563eb", 2.6, []);
  }
  drawObstacles();
  if (!state.comparison.length && result?.predictions?.[state.frame] && layerEnabled("layerPrediction")) {
    drawPath(result.predictions[state.frame], "#ea580c", 2, []);
  }
  drawHandle(state.start, "#15803d", "S");
  drawHandle(state.goal, "#b42318", "G");
  if (!state.comparison.length) drawCar();
}

function drawComparison() {
  const first = state.comparison.find((item) => item.result)?.result;
  if (first?.planning?.rawPath?.length && layerEnabled("layerRaw")) drawPath(first.planning.rawPath, "#94a3b8", 1.2, [5, 5]);
  if (first?.planning?.smoothPath?.length && layerEnabled("layerSmooth")) drawPath(first.planning.smoothPath, "#16a34a", 1.6, []);
  if (first?.reference?.length && layerEnabled("layerReference")) drawPath(first.reference, "#111827", 1.4, [8, 6]);
  if (!layerEnabled("layerTrajectory")) return;
  state.comparison.forEach((item) => {
    if (item.result?.trajectory?.length) {
      drawPath(item.result.trajectory, item.color, 2.4, []);
    }
  });
}

function drawGrid() {
  const b = state.bounds;
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  const step = 5;
  for (let x = Math.ceil(b.x_min / step) * step; x <= b.x_max; x += step) {
    const p0 = worldToScreen({ x, y: b.y_min });
    const p1 = worldToScreen({ x, y: b.y_max });
    line(p0.x, p0.y, p1.x, p1.y);
  }
  for (let y = Math.ceil(b.y_min / step) * step; y <= b.y_max; y += step) {
    const p0 = worldToScreen({ x: b.x_min, y });
    const p1 = worldToScreen({ x: b.x_max, y });
    line(p0.x, p0.y, p1.x, p1.y);
  }
  ctx.strokeStyle = "#94a3b8";
  const z0 = worldToScreen({ x: b.x_min, y: 0 });
  const z1 = worldToScreen({ x: b.x_max, y: 0 });
  line(z0.x, z0.y, z1.x, z1.y);
}

function drawPath(points, color, width, dash) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width * (window.devicePixelRatio || 1);
  ctx.setLineDash(dash.map((v) => v * (window.devicePixelRatio || 1)));
  ctx.beginPath();
  points.forEach((p, i) => {
    const s = worldToScreen(p);
    if (i === 0) ctx.moveTo(s.x, s.y);
    else ctx.lineTo(s.x, s.y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawObstacles() {
  const obsFrame = state.result?.obstacleTraj?.[state.frame];
  const activeRole = obstacleActiveRole();
  state.obstacles.forEach((obs, i) => {
    const center = obsFrame?.[i] || obs;
    const p = worldToScreen(center);
    const r = obs.r * p.s;
    const visualOnly = activeRole === "visual";
    ctx.beginPath();
    ctx.fillStyle = visualOnly
      ? "rgba(100, 116, 139, .10)"
      : (obs.kind === "dynamic" ? "rgba(234, 88, 12, .22)" : "rgba(185, 28, 28, .20)");
    ctx.strokeStyle = state.selected === obs.id
      ? "#0f766e"
      : (visualOnly ? "rgba(100, 116, 139, .65)" : (obs.kind === "dynamic" ? "#ea580c" : "#b91c1c"));
    ctx.lineWidth = state.selected === obs.id ? 3 : 2;
    ctx.setLineDash(visualOnly ? [6, 5] : []);
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);
    if (layerEnabled("layerSafety")) {
      const margin = Number(controls.safetyMargin.value);
      const alpha = Number(controls.alphaUncert.value);
      // alpha is m/s growth rate; max-horizon growth = alpha * N * dt
      const alphaGrowth = (obs.kind === "dynamic" && alpha > 0)
        ? alpha * Number(controls.horizon.value) * 0.1 : 0;
      const inflated = obs.r + 1.25 + margin + alphaGrowth;
      ctx.beginPath();
      ctx.strokeStyle = visualOnly ? "rgba(100, 116, 139, .35)" : "rgba(180, 83, 9, .45)";
      ctx.lineWidth = 1.4;
      ctx.setLineDash([5, 5]);
      ctx.arc(p.x, p.y, inflated * p.s, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (obs.kind === "dynamic") drawVelocity(obs, center);
  });
}

function obstacleActiveRole() {
  const hasObstacle = controls.obstacleScene.value !== "none";
  if (!hasObstacle) return "none";
  if (controls.sceneSelect.value === "astar") return "planning";
  if (controls.controller.value === "mpc" && controls.avoidMode.value !== "none") return "local";
  return "visual";
}

function drawVelocity(obs, center) {
  const start = worldToScreen(center);
  const end = worldToScreen({ x: center.x + obs.vx, y: center.y + obs.vy });
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const len = Math.hypot(dx, dy);
  if (len < 2) return;
  ctx.strokeStyle = "#ea580c";
  ctx.fillStyle = "#ea580c";
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  line(start.x, start.y, end.x, end.y);
  const angle = Math.atan2(dy, dx);
  const headLen = Math.min(len * 0.35, 10 * (window.devicePixelRatio || 1));
  ctx.beginPath();
  ctx.moveTo(end.x, end.y);
  ctx.lineTo(end.x - headLen * Math.cos(angle - Math.PI / 6), end.y - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(end.x - headLen * Math.cos(angle + Math.PI / 6), end.y - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

function drawHandle(point, color, label) {
  const p = worldToScreen(point);
  const rr = 8 * (window.devicePixelRatio || 1);
  ctx.beginPath();
  ctx.fillStyle = color;
  ctx.arc(p.x, p.y, rr, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = `${11 * (window.devicePixelRatio || 1)}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, p.x, p.y + .5);
}

function drawCar() {
  const traj = state.result?.trajectory;
  if (!traj?.length) return;
  const i = Math.min(state.frame, traj.length - 1);
  const pos = traj[i];
  const ctrlState = state.result?.refStates?.[Math.max(0, i - 1)];
  const next = traj[Math.min(i + 1, traj.length - 1)];
  const theta = Math.atan2((next?.y ?? pos.y) - pos.y, (next?.x ?? pos.x) - pos.x) || ctrlState?.theta || 0;
  const p = worldToScreen(pos);
  const scale = p.s;
  ctx.save();
  ctx.translate(p.x, p.y);
  ctx.rotate(-theta);
  ctx.fillStyle = "#0f766e";
  ctx.strokeStyle = "#064e3b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.rect(-1.2 * scale, -.55 * scale, 2.4 * scale, 1.1 * scale);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#d1fae5";
  ctx.beginPath();
  ctx.moveTo(1.25 * scale, 0);
  ctx.lineTo(.55 * scale, -.35 * scale);
  ctx.lineTo(.55 * scale, .35 * scale);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function line(x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function hitTest(world) {
  const radius = 0.8;
  if (dist(world, state.start) < radius) return { type: "start" };
  if (dist(world, state.goal) < radius) return { type: "goal" };
  for (let i = state.obstacles.length - 1; i >= 0; i--) {
    const obs = state.obstacles[i];
    if (dist(world, obs) <= obs.r + 0.35) return { type: "obstacle", id: obs.id };
    if (obs.kind === "dynamic" && dist(world, { x: obs.x + obs.vx, y: obs.y + obs.vy }) < 0.6) {
      return { type: "velocity", id: obs.id };
    }
  }
  return null;
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

canvas.addEventListener("mousedown", (event) => {
  const p = pointer(event);
  const world = screenToWorld(p.x, p.y);
  const hit = hitTest(world);
  if (hit?.type === "obstacle" || hit?.type === "velocity") {
    state.selected = hit.id;
    syncSelectedControls();
  }
  state.drag = hit;
  draw();
});

canvas.addEventListener("mousemove", (event) => {
  if (!state.drag) return;
  const p = pointer(event);
  const world = screenToWorld(p.x, p.y);
  if (state.drag.type === "start") {
    state.start.x = world.x;
    state.start.y = world.y;
    if (controls.sceneSelect.value !== "astar") controls.sceneSelect.value = "direct";
  } else if (state.drag.type === "goal") {
    state.goal.x = world.x;
    state.goal.y = world.y;
    if (controls.sceneSelect.value !== "astar") controls.sceneSelect.value = "direct";
  } else {
    const obs = state.obstacles.find((o) => o.id === state.drag.id);
    if (!obs) return;
    if (state.drag.type === "velocity") {
      obs.vx = world.x - obs.x;
      obs.vy = world.y - obs.y;
      syncSelectedControls();
    } else {
      obs.x = world.x;
      obs.y = world.y;
    }
  }
  clearComputedResult();
  draw();
});

window.addEventListener("mouseup", () => { state.drag = null; });

function pointer(event) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  return { x: (event.clientX - rect.left) * dpr, y: (event.clientY - rect.top) * dpr };
}

function buildRequest() {
  const trajectory = controls.sceneSelect.value;
  return {
    scene: trajectoryPresetName(trajectory),
    controller: controls.controller.value,
    reference_mode: trajectory === "direct" || trajectory === "astar" ? "direct" : "preset",
    astar_enabled: trajectory === "astar",
    ekf_enabled: controls.ekfEnabled.checked,
    avoid_mode: controls.avoidMode.value,
    lambda_soft: Number(controls.lambdaSoft.value),
    horizon: Number(controls.horizon.value),
    alpha_uncert: Number(controls.alphaUncert.value),
    safety_margin: Number(controls.safetyMargin.value),
    car_radius: 1.25,
    max_steps: estimateMaxSteps(),
    start: state.start,
    goal: state.goal,
    bounds: state.bounds,
    obstacles: state.obstacles,
    weights: {
      q_x: Number(controls.qPos.value),
      q_y: Number(controls.qPos.value),
      q_phi: Number(controls.qPhi.value),
      q_v: 1,
      r_a: 1,
      r_delta: Number(controls.rDelta.value),
      rd_a: 0.1,
      rd_delta: Number(controls.rDelta.value),
    },
    vehicle: {
      target_speed: Number(controls.targetSpeed.value),
      dt: 0.1,
      wheelbase: 2.5,
      a_max: Number(controls.aMax.value),
      a_min: -5,
      delta_max_deg: 30,
    },
  };
}

function estimateMaxSteps() {
  const d = Math.hypot(state.goal.x - state.start.x, state.goal.y - state.start.y);
  const v = Math.max(1, Number(controls.targetSpeed.value));
  const dt = 0.1;
  return Math.max(220, Math.min(900, Math.ceil((d / v / dt) * 3)));
}

async function simulatePayload(payload) {
  const response = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Simulation failed");
  return data;
}

async function runSimulation() {
  state.comparison = [];
  renderComparisonSummary([]);
  setStatus("正在仿真...");
  setBatchButtonsDisabled(true);
  try {
    const data = await simulatePayload(buildRequest());
    state.result = data;
    state.frame = 0;
    controls.frameSlider.max = Math.max(0, (data.trajectory?.length || 1) - 1);
    controls.frameSlider.value = 0;
    renderMetrics(data.metrics);
    renderDiagnostics(data.metrics);
    drawCharts(data.series);
    const suffix = statusSuffix(data.metrics);
    setStatus(`完成：${data.metrics.steps} 步，平均求解 ${fmt(data.metrics.avgSolveMs, 2)} ms${suffix}`);
    draw();
  } catch (error) {
    renderDiagnostics(null, [`仿真失败：${error.message}`]);
    setStatus(`失败：${error.message}`);
  } finally {
    setBatchButtonsDisabled(false);
  }
}

function renderMetrics(m) {
  if (!m) {
    el("metrics").innerHTML = `<div class="metric">
      <div class="label">性能指标</div>
      <div class="value">待运行</div>
    </div>`;
    return;
  }
  const items = [
    ["步数", m.steps, ""],
    ["路径长度", m.pathLength, " m"],
    ["墙钟耗时", m.wallTimeMs / 1000, " s"],
    ["平均求解", m.avgSolveMs, " ms"],
    ["横向 RMS", m.rmsLat, " m"],
    ["航向 RMS", m.rmsHeadingDeg, " deg"],
    ["最小安全余量", m.minClearance, " m"],
    ["终点距离", m.finalDistanceToGoal, " m"],
  ];
  el("metrics").innerHTML = items.map(([label, value, unit]) => {
    const danger = label === "最小安全余量" && value !== null && value < 0;
    return `<div class="metric ${danger ? "danger" : ""}">
      <div class="label">${label}</div>
      <div class="value">${value === null ? "无" : fmt(value, 2)}${unit}</div>
    </div>`;
  }).join("");
}

function renderDiagnostics(m, extraErrors = []) {
  const messages = extraErrors.map((text) => ["bad", text]);
  const hasObstacle = controls.obstacleScene.value !== "none" && state.obstacles.length > 0;
  const role = obstacleActiveRole();
  if (!m && !messages.length) {
    controls.diagnostics.innerHTML = "";
    return;
  }
  if (m?.terminationReason === "diverged") {
    messages.push(["bad", "轨迹已经偏离终点方向，仿真被提前中断。优先尝试 A* 全局规划，或降低目标速度、减小障碍半径。"]);
  }
  if (m?.terminationReason === "solver_failed") {
    messages.push(["bad", "MPC 硬约束连续多拍不可行，系统已紧急制动并中断。多障碍或大障碍需要先用 A* 选择绕行路线。"]);
  }
  if (m?.reachedMaxSteps) {
    messages.push(["warn", "到达步数上限仍未收敛。可能是参考轨迹闭合、终点停止条件过严，或局部避障把车推离了终点。"]);
  }
  if (m?.minClearance !== null && Number.isFinite(Number(m.minClearance)) && m.minClearance < 0) {
    const label = controls.avoidMode.value === "soft"
      ? "软约束允许短暂违反安全距离；当前最小安全余量为负，说明轨迹进入了膨胀安全圈。"
      : "最小安全余量为负，说明轨迹与障碍安全圈发生重叠，需要增大安全余量或切换 A*。";
    messages.push(["bad", label]);
  }
  if (m?.finalDistanceToGoal !== null && Number.isFinite(Number(m.finalDistanceToGoal)) && m.finalDistanceToGoal > 4) {
    messages.push(["warn", `终点距离仍有 ${fmt(m.finalDistanceToGoal, 2)} m，当前参数组合没有稳定到达目标。`]);
  }
  if (m?.avgSolveMs !== null && Number.isFinite(Number(m.avgSolveMs)) && m.avgSolveMs > 80) {
    messages.push(["warn", "平均求解耗时偏高。可以降低预测时域 N，或先用 A* 生成更顺的参考路径。"]);
  }
  const failedSolves = Object.entries(m?.statusCounts || {})
    .filter(([status]) => !["optimal", "optimal_inaccurate", "n/a"].includes(status))
    .reduce((sum, [, count]) => sum + Number(count || 0), 0);
  if (failedSolves > 0) {
    messages.push(["warn", `本次有 ${failedSolves} 拍求解不可行或失败；失败时车辆会紧急制动并回正。`]);
  }
  if (hasObstacle && role === "visual") {
    messages.push(["warn", "当前障碍物只参与显示，不进入 PID/LQR 或无避障 MPC 的约束。若要真正绕开，需要使用 A* 或 MPC 避障。"]);
  }
  if (hasObstacle && role === "planning") {
    messages.push(["good", "障碍物已进入 A* 全局规划，控制器负责跟踪规划后的路径。"]);
  }
  if (hasObstacle && role === "local") {
    messages.push(["good", "障碍物已进入 MPC 局部 hard/soft 约束，可以观察局部避障对轨迹的推开效果。"]);
  }
  if (!messages.length) {
    messages.push(["good", "本次运行没有明显异常。"]);
  }
  controls.diagnostics.innerHTML = messages.map(([level, text]) => (
    `<div class="diagnostic ${level}">${escapeHtml(text)}</div>`
  )).join("");
}

function statusSuffix(m) {
  if (m.terminationReason === "diverged") {
    return "，偏离终点过远已中断，建议使用 A* 全局规划或减小障碍/改硬约束";
  }
  if (m.terminationReason === "solver_failed") {
    return "，硬约束连续不可行，已紧急制动并中断，建议改用 A*";
  }
  if (m.reachedMaxSteps) {
    return "，达到步数上限，可能未收敛";
  }
  return "";
}

async function runComparison(kind) {
  const base = buildRequest();
  const variants = comparisonVariants(kind, base);
  if (!variants.length) return;
  state.playing = false;
  el("playPause").textContent = "播放";
  state.comparison = [];
  renderComparisonSummary([]);
  setBatchButtonsDisabled(true);
  try {
    const results = [];
    for (const [index, variant] of variants.entries()) {
      setStatus(`正在对比：${variant.label}`);
      try {
        const data = await simulatePayload(variant.payload);
        results.push({
          label: variant.label,
          color: COMPARISON_COLORS[index % COMPARISON_COLORS.length],
          result: data,
          error: null,
        });
      } catch (error) {
        results.push({
          label: variant.label,
          color: COMPARISON_COLORS[index % COMPARISON_COLORS.length],
          result: null,
          error: error.message,
        });
      }
    }
    state.comparison = results.filter((item) => item.result);
    const first = state.comparison[0]?.result || null;
    state.result = first;
    state.frame = 0;
    controls.frameSlider.max = Math.max(0, (first?.trajectory?.length || 1) - 1);
    controls.frameSlider.value = 0;
    renderMetrics(first?.metrics || null);
    renderDiagnostics(first?.metrics || null, results.filter((item) => item.error).map((item) => `${item.label} 失败：${item.error}`));
    renderComparisonSummary(results);
    drawCharts(results);
    draw();
    setStatus(`对比完成：${state.comparison.length}/${results.length} 个方案可显示`);
  } catch (error) {
    renderDiagnostics(null, [`对比失败：${error.message}`]);
    setStatus(`对比失败：${error.message}`);
  } finally {
    setBatchButtonsDisabled(false);
  }
}

function comparisonVariants(kind, base) {
  if (kind === "controllers") {
    const mpcAvoidMode = base.avoid_mode !== "none"
      ? base.avoid_mode
      : ((base.obstacles?.length || 0) > 0 && !base.astar_enabled ? "hard" : "none");
    return [
      ["PID", { controller: "pid", avoid_mode: "none" }],
      ["LQR", { controller: "lqr", avoid_mode: "none" }],
      ["MPC", { controller: "mpc", avoid_mode: mpcAvoidMode }],
    ].map(([label, patch]) => ({ label, payload: makePayloadVariant(base, patch) }));
  }
  if (kind === "avoidance") {
    return [
      ["无避障", { controller: "mpc", avoid_mode: "none" }],
      ["硬约束", { controller: "mpc", avoid_mode: "hard" }],
      ["软约束", { controller: "mpc", avoid_mode: "soft" }],
    ].map(([label, patch]) => ({ label, payload: makePayloadVariant(base, patch) }));
  }
  if (kind === "horizon") {
    const n = Number(controls.horizon.value);
    const avoidMode = base.avoid_mode !== "none"
      ? base.avoid_mode
      : ((base.obstacles?.length || 0) > 0 && !base.astar_enabled ? "hard" : "none");
    const values = [...new Set([Math.max(5, n - 7), n, Math.min(40, n + 10)])].sort((a, b) => a - b);
    return values.map((horizon) => ({
      label: `N=${horizon}`,
      payload: makePayloadVariant(base, { controller: "mpc", avoid_mode: avoidMode, horizon }),
    }));
  }
  return [];
}

function makePayloadVariant(base, patch) {
  const payload = structuredClone(base);
  Object.assign(payload, patch);
  payload.max_steps = Math.min(payload.max_steps, 520);
  return payload;
}

function renderComparisonSummary(results) {
  if (!results.length) {
    controls.comparisonSummary.innerHTML = "";
    return;
  }
  controls.comparisonSummary.innerHTML = results.map((item) => {
    if (item.error || !item.result) {
      return `<div class="comparison-row">
        <div class="comparison-name"><span class="swatch" style="background:${item.color}"></span>${escapeHtml(item.label)}</div>
        <div class="comparison-stat">失败</div>
        <div class="comparison-stat">${escapeHtml(item.error || "")}</div>
      </div>`;
    }
    const m = item.result.metrics;
    return `<div class="comparison-row">
      <div class="comparison-name"><span class="swatch" style="background:${item.color}"></span>${escapeHtml(item.label)}</div>
      <div class="comparison-stat">终点 ${fmt(m.finalDistanceToGoal, 1)} m</div>
      <div class="comparison-stat">均值 ${fmt(m.avgSolveMs, 1)} ms</div>
    </div>`;
  }).join("");
}

function setBatchButtonsDisabled(disabled) {
  el("runSim").disabled = disabled;
  controls.compareControllers.disabled = disabled;
  controls.compareAvoidance.disabled = disabled;
  controls.sweepHorizon.disabled = disabled;
}

function drawCharts(input) {
  const runs = normalizeChartRuns(input);
  CHART_SPECS.forEach((spec) => {
    const lines = runs.map((run) => ({
      label: run.label,
      color: run.color || spec.color,
      x: run.series?.t || [],
      y: run.series?.[spec.field] || [],
    }));
    drawChart(el(spec.id), lines, spec);
  });
}

function normalizeChartRuns(input) {
  if (Array.isArray(input)) {
    return input
      .filter((item) => item.result?.series)
      .map((item) => ({
        label: item.label,
        color: item.color,
        series: item.result.series,
      }));
  }
  if (input) return [{ label: "", color: null, series: input }];
  return [];
}

function drawChart(chart, lines, spec) {
  const c = chart.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = chart.getBoundingClientRect();
  chart.width = Math.max(240, Math.round(rect.width * dpr));
  chart.height = Math.max(120, Math.round(rect.height * dpr));
  c.clearRect(0, 0, chart.width, chart.height);
  c.fillStyle = "#f8fafc";
  c.fillRect(0, 0, chart.width, chart.height);

  const cleanY = [];
  const cleanX = [];
  for (const lineDef of lines) {
    (lineDef.y || []).forEach((v, i) => {
      const xVal = lineDef.x?.[i] ?? i;
      if (Number.isFinite(v) && Number.isFinite(xVal)) {
        cleanY.push(v);
        cleanX.push(xVal);
      }
    });
  }
  const showLegend = lines.filter((lineDef) => lineDef.label).length > 1;
  const left = 42 * dpr;
  const right = 10 * dpr;
  const top = showLegend ? 24 * dpr : 8 * dpr;
  const bottom = 28 * dpr;
  const plotW = chart.width - left - right;
  const plotH = chart.height - top - bottom;
  c.strokeStyle = "#cbd5e1";
  c.lineWidth = 1 * dpr;
  lineOn(c, left, top, left, top + plotH);
  lineOn(c, left, top + plotH, left + plotW, top + plotH);
  c.fillStyle = "#64748b";
  c.font = `${10 * dpr}px sans-serif`;
  c.textAlign = "center";
  c.fillText(spec.xLabel, left + plotW / 2, chart.height - 6 * dpr);
  c.save();
  c.translate(10 * dpr, top + plotH / 2);
  c.rotate(-Math.PI / 2);
  c.fillText(spec.yLabel, 0, 0);
  c.restore();

  if (showLegend) drawChartLegend(c, lines, left, 8 * dpr, dpr);
  if (cleanY.length < 2) {
    c.fillStyle = "#94a3b8";
    c.textAlign = "center";
    c.fillText("暂无数据", left + plotW / 2, top + plotH / 2);
    return;
  }
  const min = Math.min(...cleanY);
  const max = Math.max(...cleanY);
  const pad = Math.max(1e-6, (max - min) * 0.08);
  const yMin = min === max ? min - 1 : min - pad;
  const yMax = min === max ? max + 1 : max + pad;
  const span = Math.max(1e-6, yMax - yMin);
  const xMin = cleanX.length ? Math.min(...cleanX) : 0;
  const xMax = cleanX.length ? Math.max(...cleanX) : 1;
  const xSpan = Math.max(1e-6, xMax - xMin);
  c.strokeStyle = "#e2e8f0";
  c.lineWidth = 1;
  c.fillStyle = "#64748b";
  c.textAlign = "right";
  for (let i = 0; i <= 3; i++) {
    const y = top + plotH - (i / 3) * plotH;
    const tick = yMin + (i / 3) * span;
    lineOn(c, left, y, left + plotW, y);
    c.fillText(formatTick(tick), left - 5 * dpr, y + 3 * dpr);
  }
  if (yMin < 0 && yMax > 0) {
    const zeroY = top + plotH - ((0 - yMin) / span) * plotH;
    c.strokeStyle = "#94a3b8";
    c.setLineDash([4 * dpr, 4 * dpr]);
    lineOn(c, left, zeroY, left + plotW, zeroY);
    c.setLineDash([]);
  }
  c.textAlign = "center";
  for (let i = 0; i <= 2; i++) {
    const x = left + (i / 2) * plotW;
    const tick = xMin + (i / 2) * xSpan;
    c.fillText(formatTick(tick), x, top + plotH + 13 * dpr);
  }

  lines.forEach((lineDef) => {
    c.strokeStyle = lineDef.color;
    c.lineWidth = 2 * dpr;
    c.beginPath();
    let started = false;
    (lineDef.y || []).forEach((v, i) => {
      const xVal = lineDef.x?.[i] ?? i;
      if (!Number.isFinite(v) || !Number.isFinite(xVal)) {
        started = false;
        return;
      }
      const x = left + ((xVal - xMin) / xSpan) * plotW;
      const y = top + plotH - ((v - yMin) / span) * plotH;
      if (!started) {
        c.moveTo(x, y);
        started = true;
      } else {
        c.lineTo(x, y);
      }
    });
    c.stroke();
  });
}

function drawChartLegend(c, lines, x, y, dpr) {
  let dx = 0;
  c.font = `${10 * dpr}px sans-serif`;
  c.textAlign = "left";
  c.textBaseline = "middle";
  lines.filter((lineDef) => lineDef.label).forEach((lineDef) => {
    c.fillStyle = lineDef.color;
    c.fillRect(x + dx, y, 9 * dpr, 9 * dpr);
    c.fillStyle = "#475569";
    c.fillText(lineDef.label, x + dx + 13 * dpr, y + 4.5 * dpr);
    dx += (lineDef.label.length * 8 + 34) * dpr;
  });
  c.textBaseline = "alphabetic";
}

function formatTick(value) {
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function lineOn(c, x1, y1, x2, y2) {
  c.beginPath();
  c.moveTo(x1, y1);
  c.lineTo(x2, y2);
  c.stroke();
}

function setFrame(frame) {
  const max = Number(controls.frameSlider.max);
  state.frame = Math.max(0, Math.min(max, Math.round(frame)));
  controls.frameSlider.value = state.frame;
  el("frameLabel").textContent = `${fmt(state.frame * 0.1, 1)}s`;
  draw();
}

function tick(ts) {
  if (state.playing && state.result) {
    if (ts - state.lastTick > 50) {
      const next = state.frame + 1;
      if (next > Number(controls.frameSlider.max)) {
        state.playing = false;
        el("playPause").textContent = "播放";
      } else {
        setFrame(next);
      }
      state.lastTick = ts;
    }
  }
  requestAnimationFrame(tick);
}

function syncSelectedControls() {
  const obs = state.obstacles.find((o) => o.id === state.selected);
  const disabled = !obs;
  controls.selectedRadius.disabled = disabled;
  controls.selectedDynamic.disabled = disabled;
  controls.selectedVx.disabled = disabled;
  controls.selectedVy.disabled = disabled;
  const xyGrid = document.querySelector(".xy-grid");
  if (xyGrid) xyGrid.style.display = (obs && obs.kind === "dynamic") ? "" : "none";
  if (!obs) return;
  controls.selectedRadius.value = obs.r;
  controls.selectedDynamic.checked = obs.kind === "dynamic";
  controls.selectedVx.value = obs.vx || 0;
  controls.selectedVy.value = obs.vy || 0;
}

function loadPreset(name) {
  controls.sceneSelect.value = name;
  loadConfiguration();
}

function loadConfiguration() {
  const trajectory = controls.sceneSelect.value;
  const obstacleMode = controls.obstacleScene.value;
  const basePreset = getPreset(trajectoryPresetName(trajectory));
  const obstaclePreset = obstacleMode === "none" ? null : getPreset(obstacleMode);
  if (!basePreset) return;
  const posePreset = (trajectory === "direct" || trajectory === "astar") && obstaclePreset
    ? obstaclePreset
    : basePreset;
  state.scene = basePreset;
  state.start = structuredClone(posePreset.start);
  state.goal = structuredClone(posePreset.goal);
  state.bounds = structuredClone(basePreset.bounds);
  if (obstaclePreset) state.bounds = mergeBounds(state.bounds, obstaclePreset.bounds);
  state.obstacles = obstaclePreset ? structuredClone(obstaclePreset.obstacles) : [];
  state.selected = state.obstacles[0]?.id || null;
  controls.avoidMode.value = obstaclePreset && trajectory !== "astar" ? "hard" : "none";
  controls.alphaUncert.value = obstaclePreset?.alphaUncert || 0;
  controls.targetSpeed.value = Math.max(1, Math.min(14, posePreset.targetSpeed || 6));
  syncSelectedControls();
  syncSliderLabels();
  updateControlAvailability();
  state.result = null;
  state.comparison = [];
  state.frame = 0;
  renderMetrics(null);
  renderDiagnostics(null);
  renderComparisonSummary([]);
  drawCharts({ t: [], eLat: [], eHeadingDeg: [], solveMs: [] });
  draw();
}

function getPreset(name) {
  return state.presets.find((p) => p.name === name);
}

function trajectoryPresetName(trajectory) {
  if (trajectory === "direct") return "line";
  if (trajectory === "astar") return "astar";
  return trajectory;
}

function mergeBounds(a, b) {
  return {
    x_min: Math.min(a.x_min, b.x_min),
    x_max: Math.max(a.x_max, b.x_max),
    y_min: Math.min(a.y_min, b.y_min),
    y_max: Math.max(a.y_max, b.y_max),
  };
}

function updateControlAvailability() {
  const isMpc = controls.controller.value === "mpc";
  const hasObstacle = controls.obstacleScene.value !== "none";
  const usesAstar = controls.sceneSelect.value === "astar";
  const complexLocalAvoidance = state.obstacles.length > 1
    || state.obstacles.some((obs) => obs.r + Number(controls.safetyMargin.value) + 1.25 > 4);
  controls.avoidMode.disabled = !isMpc;
  controls.horizon.disabled = !isMpc;
  controls.qPos.disabled = !isMpc;
  controls.qPhi.disabled = !isMpc;
  controls.rDelta.disabled = !isMpc;
  controls.lambdaSoft.disabled = !isMpc || controls.avoidMode.value !== "soft";
  if (!isMpc) controls.avoidMode.value = "none";

  // A* raw/smooth layers are only meaningful in astar mode
  controls.layerRaw.disabled = !usesAstar;
  controls.layerSmooth.disabled = !usesAstar;
  [controls.layerRaw, controls.layerSmooth].forEach((cb) => {
    const label = cb.closest("label");
    if (label) label.style.opacity = usesAstar ? "" : "0.38";
  });

  if (!hasObstacle) {
    controls.capabilityHint.textContent = "当前无障碍物；可从左面板添加或选择障碍模板。";
  } else if (!isMpc && usesAstar) {
    controls.capabilityHint.textContent = "PID/LQR 不支持局部避障；障碍物会参与 A* 规划，控制器只负责跟踪规划后的轨迹。";
  } else if (!isMpc) {
    controls.capabilityHint.textContent = "PID/LQR 不支持局部避障；障碍物仅作显示，建议选择 A* 全局规划或切回 MPC。";
  } else if (usesAstar && controls.avoidMode.value === "none") {
    controls.capabilityHint.textContent = "障碍物参与 A* 全局规划；MPC 当前只跟踪规划后的轨迹。";
  } else if (controls.avoidMode.value === "none") {
    controls.capabilityHint.textContent = "MPC 局部避障未启用；障碍物仅作显示。";
  } else if (!usesAstar && controls.avoidMode.value === "hard" && complexLocalAvoidance) {
    controls.capabilityHint.textContent = "当前多障碍/大障碍容易让局部硬约束不可行；建议使用 A* 全局规划选择绕行路线。";
  } else if (controls.sceneSelect.value === "direct" && controls.avoidMode.value === "soft") {
    controls.capabilityHint.textContent = "MPC 软约束会尽量避障但允许违反安全距离；复杂或大障碍建议改用 A* 全局规划。";
  } else {
    const modeLabel = controls.avoidMode.value === "soft" ? "软约束" : "硬约束";
    controls.capabilityHint.textContent = `MPC ${modeLabel}避障已启用，障碍物进入控制器约束。`;
  }
  draw();
}

function fmt(value, digits) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(text) {
  el("statusText").textContent = text;
}

function clearComputedResult() {
  if (!state.result && !state.comparison.length) return;
  state.result = null;
  state.comparison = [];
  state.frame = 0;
  controls.frameSlider.max = 0;
  controls.frameSlider.value = 0;
  el("frameLabel").textContent = "0.0s";
  renderMetrics(null);
  renderDiagnostics(null);
  renderComparisonSummary([]);
  drawCharts({ t: [], eLat: [], eHeadingDeg: [], solveMs: [] });
  setStatus("配置已修改，点击运行更新仿真");
}

function isDisplayOnlyControl(target) {
  return target === controls.frameSlider
    || target === controls.layerReference
    || target === controls.layerRaw
    || target === controls.layerSmooth
    || target === controls.layerTrajectory
    || target === controls.layerPrediction
    || target === controls.layerSafety;
}

function addObstacle() {
  const id = `obs-${Date.now()}`;
  state.obstacles.push({ id, kind: "static", x: (state.start.x + state.goal.x) / 2, y: 1.5, r: 1.2, vx: 0, vy: 0 });
  state.selected = id;
  syncSelectedControls();
  clearComputedResult();
  draw();
}

function deleteSelected() {
  if (!state.selected) return;
  state.obstacles = state.obstacles.filter((o) => o.id !== state.selected);
  state.selected = state.obstacles[0]?.id || null;
  syncSelectedControls();
  clearComputedResult();
  draw();
}

async function init() {
  const response = await fetch("/api/presets");
  const data = await response.json();
  state.presets = data.scenes;
  controls.sceneSelect.innerHTML = TRAJECTORY_OPTIONS
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");
  controls.obstacleScene.innerHTML = OBSTACLE_OPTIONS
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");
  controls.sceneSelect.value = "line";
  controls.obstacleScene.value = "block";
  loadConfiguration();
  renderMetrics(null);
  drawCharts({ t: [], eLat: [], eHeadingDeg: [], solveMs: [] });
  setStatus("就绪：调整配置后点击运行开始仿真");
}

for (const input of document.querySelectorAll("input, select")) {
  input.addEventListener("input", (event) => {
    syncSliderLabels();
    if (!isDisplayOnlyControl(event.target)) clearComputedResult();
    updateControlAvailability();
  });
}
controls.sceneSelect.addEventListener("change", loadConfiguration);
controls.obstacleScene.addEventListener("change", loadConfiguration);
el("runSim").addEventListener("click", runSimulation);
el("resetScene").addEventListener("click", loadConfiguration);
el("addObstacle").addEventListener("click", addObstacle);
el("deleteSelected").addEventListener("click", deleteSelected);
controls.compareControllers.addEventListener("click", () => runComparison("controllers"));
controls.compareAvoidance.addEventListener("click", () => runComparison("avoidance"));
controls.sweepHorizon.addEventListener("click", () => runComparison("horizon"));
el("playPause").addEventListener("click", () => {
  state.playing = !state.playing;
  el("playPause").textContent = state.playing ? "暂停" : "播放";
});
controls.frameSlider.addEventListener("input", () => setFrame(Number(controls.frameSlider.value)));
controls.selectedRadius.addEventListener("input", () => {
  const obs = state.obstacles.find((o) => o.id === state.selected);
  if (obs) obs.r = Number(controls.selectedRadius.value);
  draw();
});
controls.selectedDynamic.addEventListener("change", () => {
  const obs = state.obstacles.find((o) => o.id === state.selected);
  if (obs) {
    obs.kind = controls.selectedDynamic.checked ? "dynamic" : "static";
    syncSelectedControls();
  }
  draw();
});
controls.selectedVx.addEventListener("input", () => {
  const obs = state.obstacles.find((o) => o.id === state.selected);
  if (obs) obs.vx = Number(controls.selectedVx.value);
  draw();
});
controls.selectedVy.addEventListener("input", () => {
  const obs = state.obstacles.find((o) => o.id === state.selected);
  if (obs) obs.vy = Number(controls.selectedVy.value);
  draw();
});

document.addEventListener("keydown", (event) => {
  if (isEditingTarget(event.target)) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (state.result) {
      state.playing = !state.playing;
      el("playPause").textContent = state.playing ? "暂停" : "播放";
    }
  } else if (event.code === "ArrowLeft" || event.code === "ArrowRight") {
    if (!state.result) return;
    event.preventDefault();
    state.playing = false;
    el("playPause").textContent = "播放";
    setFrame(state.frame + (event.code === "ArrowRight" ? 1 : -1));
  }
});

function isEditingTarget(target) {
  return ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName) || target.isContentEditable;
}

window.addEventListener("resize", fitCanvas);
syncSliderLabels();
syncSelectedControls();
fitCanvas();
requestAnimationFrame(tick);
init();
