# Published by controller_node once per control step.
std_msgs/Header header    # stamp = time of the state the command was computed from
string controller         # mpc / pid / lqr
string solver_status      # optimal / optimal_inaccurate / infeasible / n/a / stopping
float64 solve_time_ms
float64 cost              # NaN when not available
uint32 nearest_idx        # index into the reference trajectory
float64 progress          # nearest_idx / (len - 1)
bool stopping             # true after the vehicle passed the end of the reference
