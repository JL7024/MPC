"""Run the minimal Q-learning line-tracking demo.

Usage:
    python -m rl.demo_q_learning
    python -m rl.demo_q_learning --episodes 2000 --show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl.line_tracking_env import LineTrackingEnv
from rl.q_learning import rollout, train_q_learning


def moving_average(values, window=50):
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training(stats, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    rewards = np.asarray(stats.rewards)
    ax.plot(rewards, color="tab:blue", alpha=0.25, lw=0.8, label="episode")
    ma = moving_average(rewards, window=50)
    ax.plot(np.arange(len(ma)) + 49, ma, color="tab:blue", lw=2.0,
            label="50-episode moving average")
    ax.set_title("Q-learning reward curve")
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    return fig


def plot_rollouts(env, random_hist, learned_hist, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Before vs after tabular Q-learning", fontweight="bold")

    ax = axes[0, 0]
    ax.plot(env.ref.points[:, 0], env.ref.points[:, 1], "k--",
            lw=1.0, label="reference")
    ax.plot(random_hist["state"][:, 0], random_hist["state"][:, 1],
            color="tab:red", lw=1.4, alpha=0.8, label="random policy")
    ax.plot(learned_hist["state"][:, 0], learned_hist["state"][:, 1],
            color="tab:blue", lw=1.8, label="learned policy")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("XY trajectory")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(random_hist["errors"][:, 0], color="tab:red",
            alpha=0.8, label="random")
    ax.plot(learned_hist["errors"][:, 0], color="tab:blue",
            label="learned")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("lateral error (m)")
    ax.set_title("Lateral error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(np.rad2deg(random_hist["u"][:, 1]), color="tab:red",
            alpha=0.8, label="random")
    ax.plot(np.rad2deg(learned_hist["u"][:, 1]), color="tab:blue",
            label="learned")
    ax.set_xlabel("step")
    ax.set_ylabel("steering delta (deg)")
    ax.set_title("Steering commands")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(random_hist["state"][:, 3], color="tab:red",
            alpha=0.8, label="random")
    ax.plot(learned_hist["state"][:, 3], color="tab:blue",
            label="learned")
    ax.axhline(env.v_ref, color="k", lw=0.8, ls="--", alpha=0.5,
               label="v_ref")
    ax.set_xlabel("step")
    ax.set_ylabel("speed (m/s)")
    ax.set_title("Speed")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    return fig


def plot_policy_heatmap(env, q_table, save_path):
    # Look at the greedy steering choice when speed error is near zero.
    v_idx = int(np.digitize(0.0, env.e_v_bins))
    steer = np.zeros((len(env.e_lat_bins) + 1, len(env.e_phi_bins) + 1))
    for i in range(steer.shape[0]):
        for j in range(steer.shape[1]):
            action_id = int(np.argmax(q_table[i, j, v_idx]))
            steer[i, j] = np.rad2deg(env.actions[action_id, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(steer, origin="lower", cmap="coolwarm",
                   vmin=-15, vmax=15, aspect="auto")
    ax.set_title("Learned greedy steering map (speed error near 0)")
    ax.set_xlabel("heading-error bin")
    ax.set_ylabel("lateral-error bin")
    fig.colorbar(im, ax=ax, label="steering delta (deg)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = LineTrackingEnv(seed=args.seed)
    q_table, stats = train_q_learning(env, episodes=args.episodes,
                                      seed=args.seed + 1)

    initial_state = np.array([0.0, 2.0, np.deg2rad(10.0), 3.0])
    random_hist = rollout(env, q_table=None, initial_state=initial_state,
                          random_policy=True, seed=args.seed + 2)
    learned_hist = rollout(env, q_table=q_table, initial_state=initial_state,
                           random_policy=False, seed=args.seed + 3)

    plot_training(stats, save_dir / "q_learning_reward_curve.png")
    plot_rollouts(env, random_hist, learned_hist,
                  save_dir / "q_learning_line_tracking.png")
    plot_policy_heatmap(env, q_table, save_dir / "q_learning_policy_heatmap.png")

    success_rate = np.mean(stats.successes[-100:]) if stats.successes else 0.0
    print("Q-learning demo finished")
    print(f"episodes: {args.episodes}")
    print(f"last-100 success rate: {success_rate:.2%}")
    print(f"random rollout success: {random_hist['success']}")
    print(f"learned rollout success: {learned_hist['success']}")
    print(f"figures saved to: {save_dir}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

