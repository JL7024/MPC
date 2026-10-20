# Circular obstacle with a constant-velocity motion model, world (map) frame.
# Static obstacles have vx = vy = 0.
uint32 id
float64 x        # center x [m]
float64 y        # center y [m]
float64 vx       # velocity x [m/s]
float64 vy       # velocity y [m/s]
float64 radius   # geometric radius [m], without vehicle radius / safety margin
