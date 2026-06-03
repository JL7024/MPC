"""A tiny tabular-RL environment for straight-line vehicle tracking.

This is not a Gym/Gymnasium environment. It is deliberately small so the RL
loop is easy to read: reset -> observe -> choose discrete action -> step.
"""

import numpy as np

from core.reference_trajectory import ReferenceTrajectory
from core.vehicle_model import VehicleModel


def wrap_to_pi(angle):
    """Normalize an angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class LineTrackingEnv:
    """Straight-line tracking task with discretized observations/actions."""

    def __init__(self, length=30.0, v_ref=5.0, dt=0.1, seed=0):
        self.length = float(length)
        self.v_ref = float(v_ref)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)

        self.vehicle = VehicleModel(L=2.5, dt=dt)
        self.ref = ReferenceTrajectory(L=2.5).generate_straight_line(
            length=length, v_ref=v_ref, ds=0.1
        )

        # Discrete action set: RL chooses steering only. A small built-in speed
        # loop keeps this teaching demo focused on learning lateral behavior.
        steer_values = np.deg2rad([-20.0, -12.0, -6.0, 0.0, 6.0, 12.0, 20.0])
        self.actions = np.column_stack([
            np.zeros_like(steer_values),
            steer_values,
        ])

        # Bins convert continuous tracking errors into a small Q-table index.
        self.e_lat_bins = np.array(
            [-3.0, -2.0, -1.2, -0.6, -0.25, -0.08,
              0.08, 0.25, 0.6, 1.2, 2.0, 3.0]
        )
        self.e_phi_bins = np.deg2rad(
            [-35.0, -20.0, -12.0, -6.0, -2.0,
               2.0,   6.0,  12.0, 20.0, 35.0]
        )
        self.e_v_bins = np.array([-4.0, -2.0, -1.0, -0.3, 0.3, 1.0, 2.0, 4.0])

        self.max_steps = int(np.ceil(length / max(v_ref, 1e-6) / dt)) + 50
        self.offtrack_limit = 4.0
        self.state = None
        self.step_count = 0
        self.last_ref_idx = 0

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def q_shape(self):
        return (
            len(self.e_lat_bins) + 1,
            len(self.e_phi_bins) + 1,
            len(self.e_v_bins) + 1,
            self.n_actions,
        )

    def reset(self, initial_state=None, randomize=True):
        """Reset the vehicle and return the first discrete observation."""
        self.step_count = 0
        self.last_ref_idx = 0

        if initial_state is not None:
            state = np.asarray(initial_state, dtype=float)
        elif randomize:
            state = np.array([
                0.0,
                self.rng.uniform(-2.5, 2.5),
                self.rng.uniform(np.deg2rad(-18.0), np.deg2rad(18.0)),
                self.v_ref + self.rng.uniform(-2.0, 1.0),
            ])
        else:
            state = np.array([0.0, 2.0, np.deg2rad(10.0), 3.0])

        state[3] = np.clip(state[3], 0.0, 12.0)
        self.state = self.vehicle.reset(*state)
        return self.observe()

    def tracking_errors(self, state=None):
        """Return (e_lat, e_phi, e_v, ref_idx) for a continuous state."""
        if state is None:
            state = self.state
        x, y, phi, v = state

        search_from = max(0, self.last_ref_idx - 5)
        idx = self.ref.find_nearest(
            x, y, search_from=search_from, search_window=120
        )
        ref_state = self.ref.get_reference_state(idx)
        x_r, y_r, phi_r, v_r = ref_state

        e_lat = -np.sin(phi_r) * (x - x_r) + np.cos(phi_r) * (y - y_r)
        e_phi = wrap_to_pi(phi - phi_r)
        e_v = v - v_r
        return float(e_lat), float(e_phi), float(e_v), int(idx)

    def observe(self):
        """Return the discretized observation tuple used as Q-table index."""
        e_lat, e_phi, e_v, idx = self.tracking_errors()
        self.last_ref_idx = idx
        return (
            int(np.digitize(e_lat, self.e_lat_bins)),
            int(np.digitize(e_phi, self.e_phi_bins)),
            int(np.digitize(e_v, self.e_v_bins)),
        )

    def step(self, action_id):
        """Apply one discrete action and return obs, reward, done, info."""
        old_state = self.state.copy()
        action = self.actions[int(action_id)].copy()
        action[0] = np.clip(1.2 * (self.v_ref - old_state[3]), -2.0, 2.0)
        new_state = self.vehicle.step(old_state, action)
        new_state[3] = np.clip(new_state[3], 0.0, 12.0)
        self.state = new_state
        self.step_count += 1

        old_e_lat, old_e_phi, _, _ = self.tracking_errors(old_state)
        obs = self.observe()
        e_lat, e_phi, e_v, idx = self.tracking_errors()
        progress = max(0.0, new_state[0] - old_state[0])
        lateral_improvement = abs(old_e_lat) - abs(e_lat)
        heading_improvement = abs(old_e_phi) - abs(e_phi)

        reward = (
            2.0 * progress
            + 3.0 * lateral_improvement
            + 0.8 * heading_improvement
            - 0.40 * abs(e_lat)
            - 0.20 * abs(e_phi)
            - 0.05 * abs(e_v)
            - 0.02 * abs(action[1])
        )

        success = new_state[0] >= self.length and abs(e_lat) < 2.0
        offtrack = abs(e_lat) > self.offtrack_limit
        timeout = self.step_count >= self.max_steps
        done = bool(success or offtrack or timeout)

        if success:
            reward += 50.0
        elif offtrack:
            reward -= 25.0

        info = {
            "state": new_state.copy(),
            "action": action.copy(),
            "errors": np.array([e_lat, e_phi, e_v], dtype=float),
            "ref_idx": idx,
            "success": success,
            "offtrack": offtrack,
            "timeout": timeout,
        }
        return obs, float(reward), done, info
