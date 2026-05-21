"""
传感器仿真模型

每个传感器接受真实状态, 返回带噪声的测量 (或 None 表示这一步没数据)。

设计:
    - 用 numpy Generator 持有独立 RNG, 同一 seed 跨 run 可复现
    - rate_divider 让传感器 N 步触发一次, 模拟多速率
        (例: dt=0.1s, rate_divider=5 → 0.5s 一次 = 2Hz)
    - 每个传感器 measure() 必返回 (z 或 None), 上层 EKF 自己判断要不要 update
"""

from dataclasses import dataclass
import numpy as np


class GPSSensor:
    """
    GPS 模拟器: 测 [x, y], 加高斯白噪声
        z_gps = state[:2] + N(0, sigma² I)
    典型: sigma ≈ 0.5 m (民用 RTK ~5cm, 商业 GPS ~5m, 这里取消费级中等水平)
    """
    def __init__(self, sigma=0.5, rate_divider=1, seed=0):
        self.sigma = sigma
        self.R = sigma ** 2 * np.eye(2)   # 测量噪声协方差, EKF 用
        self.rate_divider = rate_divider
        self.rng = np.random.default_rng(seed)
        self._counter = 0

    def measure(self, state):
        """返回 (2,) 测量值, 或 None"""
        out = None
        if self._counter % self.rate_divider == 0:
            out = state[:2] + self.rng.normal(0, self.sigma, 2)
        self._counter += 1
        return out


class WheelSpeedSensor:
    """
    轮速计模拟器: 测 v (纵向速度)
        z_wheel = state[3] + N(0, sigma²)
    典型: sigma ≈ 0.1 m/s (轮速编码器分辨率)
    """
    def __init__(self, sigma=0.1, rate_divider=1, seed=1):
        self.sigma = sigma
        self.R = np.array([[sigma ** 2]])   # 1x1
        self.rate_divider = rate_divider
        self.rng = np.random.default_rng(seed)
        self._counter = 0

    def measure(self, state):
        """返回标量 (作为 (1,) 形 array, 方便统一处理), 或 None"""
        out = None
        if self._counter % self.rate_divider == 0:
            out = np.array([state[3] + self.rng.normal(0, self.sigma)])
        self._counter += 1
        return out


@dataclass
class Measurements:
    """一次 measure() 的所有传感器结果, 任一字段可为 None"""
    gps: np.ndarray = None         # (2,)
    wheel: np.ndarray = None       # (1,)


class SensorSuite:
    """
    传感器集合: 一次 measure(state) 调用, 把所有传感器都触发一遍。

    默认配置 (10Hz 仿真下):
        GPS   每 2 步一次 = 5Hz, sigma=0.5m
        轮速  每 1 步一次 = 10Hz, sigma=0.1m/s
        → GPS 间隙时, EKF 只能靠 predict + 轮速; GPS 来了再校正位置
        这就是 sensor fusion 的标准设置, 也能让对比图看到 GPS 的"跳"
    """
    def __init__(self,
                 gps_sigma=0.5, gps_rate=2,
                 wheel_sigma=0.1, wheel_rate=1,
                 seed=42):
        self.gps = GPSSensor(sigma=gps_sigma, rate_divider=gps_rate, seed=seed)
        self.wheel = WheelSpeedSensor(sigma=wheel_sigma, rate_divider=wheel_rate,
                                      seed=seed + 1)

    def measure(self, state) -> Measurements:
        return Measurements(
            gps=self.gps.measure(state),
            wheel=self.wheel.measure(state),
        )


# ========================================================================
# 自测
# ========================================================================
if __name__ == "__main__":
    state = np.array([10.0, 5.0, 0.5, 8.0])
    suite = SensorSuite(seed=42)

    print("10 步测量 (GPS rate=2, wheel rate=1):")
    for k in range(10):
        m = suite.measure(state)
        gps_str = f"({m.gps[0]:6.3f}, {m.gps[1]:6.3f})" if m.gps is not None else "  None  "
        wheel_str = f"{m.wheel[0]:.3f}" if m.wheel is not None else "None"
        print(f"  step {k}: gps={gps_str}  wheel={wheel_str}")

    # 检查 RNG 复现: 同 seed 跑两遍, 第 11 次测量值应完全一致
    suite2 = SensorSuite(seed=42)
    m1 = suite.measure(state)              # suite 的第 11 次
    for _ in range(10):
        suite2.measure(state)
    m2 = suite2.measure(state)             # suite2 的第 11 次
    print(f"\n复现性检查 (同 seed, 第 11 次): gps_diff = {np.linalg.norm(m1.gps - m2.gps):.6e}")
