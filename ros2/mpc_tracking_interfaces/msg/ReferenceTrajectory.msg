# Reference trajectory sampled at equal arc length.
# Same columns as core.reference_trajectory.ReferenceTrajectory.points:
# [x, y, phi, v, kappa, s]. nav_msgs/Path cannot carry v and kappa, which MPC needs.
std_msgs/Header header
uint32 plan_id       # increases every time the planner publishes a new plan
float64 wheelbase    # L used for delta_ref = atan(L * kappa)
float64[] x          # [m]
float64[] y          # [m]
float64[] phi        # heading [rad]
float64[] v          # reference speed [m/s]
float64[] kappa      # signed curvature [1/m], left turn positive
float64[] s          # cumulative arc length [m]
