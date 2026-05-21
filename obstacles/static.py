"""
静态圆形障碍.

阶段 5 (动态避障) 会扩展出 DynamicObstacle, 接口约定:
    - 必须有 .xy (np.ndarray shape (2,)) 当前世界坐标
    - 必须有 .r 标量半径
    - 静态: position_at(t) 恒返回 .xy, 动态: 按运动模型预测

为什么用 dataclass 而不是 namedtuple:
    后续可能加属性 (颜色, 名字, 是否致命...)
    namedtuple 不可变, 改字段就要换签名; dataclass 加字段不破坏 __init__
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CircleObstacle:
    """
    圆形静态障碍。

    字段:
        cx, cy : 障碍中心世界坐标
        r      : 障碍半径 (实际几何, 不含 r_car / margin 膨胀)

    安全距离 r_safe 在控制器里算, 不存这: r_safe = r + r_car + margin

    predict/step 接口:
        与 DynamicObstacle 同名同签名, 让 MPC 不需要分静态/动态分支:
            predict(future_dt) → 未来 future_dt 秒后的位置 (静态恒等于 xy)
            step(dt)          → 真实推进, 静态啥也不做
        阶段 5 动态版会重写这两个方法。
    """
    cx: float
    cy: float
    r: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=float)

    def predict(self, future_dt) -> np.ndarray:
        """未来 future_dt 秒后的位置. 静态: 永远 self.xy."""
        return self.xy

    def step(self, dt):
        """真实推进 dt 秒. 静态: 不动."""
        pass

    # 向后兼容旧名字 (老代码可能调用 position_at)
    def position_at(self, future_dt):
        return self.predict(future_dt)


def normalize_obstacles(obs_list):
    """
    把 [CircleObstacle | DynamicObstacle | (cx, cy, r) | (cx, cy, vx, vy, r),
        ...] 混合列表规整为 [Obstacle, ...] (CircleObstacle 或 DynamicObstacle)
    便于上层 main.py / scenarios.py 写起来灵活。

    元组规则:
        len 3: (cx, cy, r)            → CircleObstacle
        len 5: (cx, cy, vx, vy, r)    → DynamicObstacle
    """
    # 局部 import 避免循环 (dynamic.py 不引用 static)
    from .dynamic import DynamicObstacle

    out = []
    for o in obs_list:
        if isinstance(o, (CircleObstacle, DynamicObstacle)):
            out.append(o)
        elif isinstance(o, (tuple, list)):
            if len(o) == 3:
                out.append(CircleObstacle(*o))
            elif len(o) == 5:
                out.append(DynamicObstacle(*o))
            else:
                raise TypeError(f"元组长度只支持 3 或 5, 实际 {len(o)}: {o!r}")
        else:
            raise TypeError(f"无法解析障碍: {o!r} "
                            f"(支持 CircleObstacle/DynamicObstacle "
                            f"或 (cx,cy,r) / (cx,cy,vx,vy,r) tuple)")
    return out


# ========================================================================
# 自测
# ========================================================================
if __name__ == "__main__":
    obs1 = CircleObstacle(10.0, 5.0, 2.0)
    print(f"obs1: {obs1}, xy={obs1.xy}, r={obs1.r}")
    print(f"position_at(t=0): {obs1.position_at(0)}")
    print(f"position_at(t=10): {obs1.position_at(10)}  (静态, 不变)")

    mixed = [obs1, (20.0, -3.0, 1.5), CircleObstacle(30.0, 0.0, 2.5)]
    norm = normalize_obstacles(mixed)
    print(f"normalize: {norm}")
    assert all(isinstance(x, CircleObstacle) for x in norm)
    print("OK")
