"""
动态圆形障碍 (恒速 / Constant Velocity 模型)

设计:
    - 与 CircleObstacle 同接口: predict(future_dt) / step(dt) / xy
      → MPC 不需要区分静态/动态, 一份代码 work both
    - 状态可变: cx, cy 随 step(dt) 推进; vx, vy 不变 (CV 模型)
    - predict 是相对当前状态的预测, 不是绝对时间. main 主循环每拍调一次
      step(dt), MPC solve 时用 predict(k*dt) 取每个 stage 的"未来位置"

为什么不带绝对时间戳:
    主循环时序明确——MPC solve 完, car step, obstacle step, 进下一拍。
    用相对时间 future_dt 让 MPC 不需要持有全局 clock, 接口干净。

CV 模型局限:
    真实障碍可能拐弯/加速/停车, 恒速预测在长时域 (>2s) 会失真。
    阶段 5 用"不确定性锥" r_safe(k) = r_base + α*k*dt 兜底——
    α 越大表示越不信任预测, 越保守。这是工业界标准做法 (Waymo / Mobileye)。
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DynamicObstacle:
    """
    恒速动态圆形障碍。

    字段:
        cx, cy : 当前障碍中心世界坐标 (会随 step 变化)
        vx, vy : 障碍速度 (世界系, m/s; 静态用 0,0 即可, 但建议直接用 CircleObstacle)
        r      : 几何半径 (不含 r_car / margin 膨胀)
    """
    cx: float
    cy: float
    vx: float
    vy: float
    r: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)

    def predict(self, future_dt) -> np.ndarray:
        """
        future_dt 秒后的位置预测 (基于当前 self.cx, cy 和 vx, vy).
        恒速模型: p(t) = p_now + v * t
        """
        return np.array([self.cx + self.vx * future_dt,
                         self.cy + self.vy * future_dt], dtype=float)

    def step(self, dt):
        """
        真实推进 dt 秒. 主循环每拍调一次, 让障碍在仿真世界里真的动起来。
        速度 vx, vy 不变 (CV 模型).
        """
        self.cx += self.vx * dt
        self.cy += self.vy * dt

    # 向后兼容: position_at 是 CircleObstacle 旧名字
    def position_at(self, future_dt):
        return self.predict(future_dt)


# ========================================================================
# 自测
# ========================================================================
if __name__ == "__main__":
    obs = DynamicObstacle(cx=10.0, cy=0.0, vx=-2.0, vy=0.5, r=1.0)
    print(f"初始: xy={obs.xy}, v={obs.velocity}")

    # predict 不应改变状态
    p_at_1s = obs.predict(1.0)
    print(f"predict(1.0) = {p_at_1s}  期望 [8.0, 0.5]")
    assert np.allclose(p_at_1s, [8.0, 0.5])
    assert np.allclose(obs.xy, [10.0, 0.0]), "predict 不能改 state"

    # step 推进
    obs.step(0.5)
    print(f"step(0.5) 后 xy={obs.xy}  期望 [9.0, 0.25]")
    assert np.allclose(obs.xy, [9.0, 0.25])

    # step 后再 predict
    p_at_1s_after = obs.predict(1.0)
    print(f"step 后 predict(1.0) = {p_at_1s_after}  期望 [7.0, 0.75]")
    assert np.allclose(p_at_1s_after, [7.0, 0.75])

    print("OK")
