"""Minimal tabular Q-learning implementation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainStats:
    rewards: list
    lengths: list
    successes: list
    epsilons: list


def greedy_action(q_table, obs, rng):
    """Choose a random argmax action for the current observation."""
    values = q_table[obs]
    best = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(best))


def epsilon_greedy(q_table, obs, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[-1]))
    return greedy_action(q_table, obs, rng)


def train_q_learning(env, episodes=1200, alpha=0.20, gamma=0.98,
                     epsilon_start=1.0, epsilon_end=0.05,
                     epsilon_decay=0.995, seed=1):
    """Train a Q-table on the provided environment."""
    rng = np.random.default_rng(seed)
    q_table = np.zeros(env.q_shape, dtype=float)
    stats = TrainStats(rewards=[], lengths=[], successes=[], epsilons=[])

    epsilon = float(epsilon_start)
    for _ in range(int(episodes)):
        obs = env.reset(randomize=True)
        total_reward = 0.0
        success = False

        for step in range(env.max_steps):
            action_id = epsilon_greedy(q_table, obs, epsilon, rng)
            next_obs, reward, done, info = env.step(action_id)

            old_value = q_table[obs + (action_id,)]
            target = reward + gamma * np.max(q_table[next_obs]) * (not done)
            q_table[obs + (action_id,)] = (
                old_value + alpha * (target - old_value)
            )

            obs = next_obs
            total_reward += reward
            if done:
                success = bool(info["success"])
                break

        stats.rewards.append(float(total_reward))
        stats.lengths.append(step + 1)
        stats.successes.append(success)
        stats.epsilons.append(float(epsilon))
        epsilon = max(float(epsilon_end), epsilon * float(epsilon_decay))

    return q_table, stats


def rollout(env, q_table=None, initial_state=None, random_policy=False,
            seed=2):
    """Run one episode and record states/actions/rewards/errors."""
    rng = np.random.default_rng(seed)
    obs = env.reset(initial_state=initial_state, randomize=initial_state is None)

    history = {
        "state": [env.state.copy()],
        "u": [],
        "reward": [],
        "errors": [],
        "action_id": [],
        "success": False,
    }

    for _ in range(env.max_steps):
        if random_policy or q_table is None:
            action_id = int(rng.integers(env.n_actions))
        else:
            action_id = greedy_action(q_table, obs, rng)

        obs, reward, done, info = env.step(action_id)
        history["state"].append(info["state"])
        history["u"].append(info["action"])
        history["reward"].append(reward)
        history["errors"].append(info["errors"])
        history["action_id"].append(action_id)

        if done:
            history["success"] = bool(info["success"])
            break

    for key in ("state", "u", "reward", "errors", "action_id"):
        history[key] = np.asarray(history[key])
    return history

