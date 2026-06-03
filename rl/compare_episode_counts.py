"""Compare Q-learning behavior after different training budgets.

Usage:
    python -m rl.compare_episode_counts
    python -m rl.compare_episode_counts --episodes 0,50,100,200,400,800
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl.line_tracking_env import LineTrackingEnv
from rl.q_learning import TrainStats, epsilon_greedy, rollout


def parse_episode_counts(text):
    counts = []
    for part in text.split(","):
        part = part.strip()
        if part:
            counts.append(int(part))
    return counts


def train_snapshots(env, counts, seed):
    """Train once and store Q-table snapshots at requested episode counts."""
    max_episodes = max(counts)
    wanted = set(counts)
    rng = np.random.default_rng(seed)
    q_table = np.zeros(env.q_shape, dtype=float)
    snapshots = {0: q_table.copy()}
    rewards = []

    alpha = 0.20
    gamma = 0.98
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995

    for episode in range(1, max_episodes + 1):
        obs = env.reset(randomize=True)
        total_reward = 0.0

        for _ in range(env.max_steps):
            action_id = epsilon_greedy(q_table, obs, epsilon, rng)
            next_obs, reward, done, _ = env.step(action_id)

            old_value = q_table[obs + (action_id,)]
            target = reward + gamma * np.max(q_table[next_obs]) * (not done)
            q_table[obs + (action_id,)] = (
                old_value + alpha * (target - old_value)
            )

            obs = next_obs
            total_reward += reward
            if done:
                break

        rewards.append(float(total_reward))
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode in wanted:
            snapshots[episode] = q_table.copy()

    return snapshots, rewards


def evaluate_many(env, q_table, episodes, seed, n_eval=20):
    successes = []
    final_errors = []
    for i in range(n_eval):
        hist = rollout(env, q_table=q_table, initial_state=None,
                       random_policy=(episodes == 0), seed=seed + i)
        successes.append(bool(hist["success"]))
        final_errors.append(float(abs(hist["errors"][-1, 0])))
    return float(np.mean(successes)), float(np.mean(final_errors))


def evaluate_counts(counts, seed):
    initial_state = np.array([0.0, 2.0, np.deg2rad(10.0), 3.0])
    env = LineTrackingEnv(seed=seed)
    snapshots, training_rewards = train_snapshots(env, counts, seed + 101)
    results = []

    for i, episodes in enumerate(counts):
        q_table = snapshots.get(episodes)
        hist = rollout(env, q_table=q_table, initial_state=initial_state,
                       random_policy=(episodes == 0), seed=seed + 1000 + i)
        final_error = float(abs(hist["errors"][-1, 0]))
        min_error = float(np.min(np.abs(hist["errors"][:, 0])))
        success_rate, mean_final_error = evaluate_many(
            env, q_table, episodes, seed + 2000 + i * 100
        )
        stats = None
        if episodes > 0:
            stats = TrainStats(
                rewards=training_rewards[:episodes],
                lengths=[],
                successes=[],
                epsilons=[],
            )
        results.append({
            "episodes": episodes,
            "env": env,
            "q_table": q_table,
            "stats": stats,
            "hist": hist,
            "success": bool(hist["success"]),
            "final_error": final_error,
            "min_error": min_error,
            "success_rate": success_rate,
            "mean_final_error": mean_final_error,
        })

    return results


def plot_episode_comparison(results, save_path):
    env = results[-1]["env"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Effect of training episode count", fontweight="bold")
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(results)))

    ax = axes[0, 0]
    ax.plot(env.ref.points[:, 0], env.ref.points[:, 1], "k--",
            lw=1.0, label="reference")
    for color, item in zip(colors, results):
        h = item["hist"]
        label = f"{item['episodes']} eps"
        if item["success"]:
            label += " ✓"
        ax.plot(h["state"][:, 0], h["state"][:, 1],
                color=color, lw=1.7, label=label)
    ax.set_title("XY trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for color, item in zip(colors, results):
        h = item["hist"]
        ax.plot(np.abs(h["errors"][:, 0]), color=color, lw=1.6,
                label=f"{item['episodes']} eps")
    ax.set_title("Absolute lateral error")
    ax.set_xlabel("step")
    ax.set_ylabel("|e_lat| (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    episodes = [item["episodes"] for item in results]
    success_rates = [100.0 * item["success_rate"] for item in results]
    mean_final_errors = [item["mean_final_error"] for item in results]
    ax2 = ax.twinx()
    ax.bar(episodes, success_rates, width=32, alpha=0.45,
           color="tab:blue", label="success rate")
    ax2.plot(episodes, mean_final_errors, "o-", color="tab:orange",
             label="mean final |e_lat|")
    ax.set_title("20 random evaluations vs training budget")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("success rate (%)")
    ax2.set_ylabel("mean final error (m)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax = axes[1, 1]
    for color, item in zip(colors, results):
        stats = item["stats"]
        if stats is None:
            continue
        rewards = np.asarray(stats.rewards, dtype=float)
        if len(rewards) >= 25:
            kernel = np.ones(25) / 25
            rewards = np.convolve(rewards, kernel, mode="valid")
            x = np.arange(len(rewards)) + 24
        else:
            x = np.arange(len(rewards))
        ax.plot(x, rewards, color=color, lw=1.5,
                label=f"{item['episodes']} eps")
    ax.set_title("Training reward moving averages")
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="0,50,100,200,400,800",
                        help="comma-separated training budgets")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--save-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    counts = parse_episode_counts(args.episodes)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_counts(counts, args.seed)
    out = save_dir / "q_learning_episode_count_comparison.png"
    plot_episode_comparison(results, out)

    print("Episode-count comparison finished")
    for item in results:
        print(
            f"{item['episodes']:>4} episodes | "
            f"success={item['success']!s:<5} | "
            f"final |e_lat|={item['final_error']:.3f} m | "
            f"best |e_lat|={item['min_error']:.3f} m | "
            f"20-eval success={item['success_rate']:.0%}"
        )
    print(f"figure saved to: {out}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
