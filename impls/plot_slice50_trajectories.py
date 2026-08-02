"""Plot a sample of length-50 trajectories from an ogbench *-stitch-slice50 dataset.

The slice50 datasets are the stitch datasets chopped into 50-step episodes, so
`terminals` fires every 51st transition. This script draws a handful of those
short trajectories in the maze's xy plane, on top of the maze wall layout.

Usage:
    python plot_slice50_trajectories.py \
        --env_name antmaze-medium-stitch-slice50-v0 --num_traj 12
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Fixed categorical hue order (never cycled beyond its length -- we subsample
# trajectories rather than generating new hues).
PALETTE = [
    "#3b6fd4", "#d4703b", "#3ba05a", "#b5479b", "#c9a227",
    "#4aa8b8", "#9b5de5", "#c0392b", "#6b8e23", "#e07a5f",
    "#2a9d8f", "#8d6e63",
]


def episode_bounds(terminals):
    """Return (start, stop) index pairs for each episode, stop exclusive."""
    ends = np.flatnonzero(terminals)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return list(zip(starts, ends + 1))


def maze_walls(env_name):
    """Pull (maze_map, unit, offset_x, offset_y) from the env, or None if unavailable."""
    try:
        import ogbench
        from utils.dataset_slicing import parse_slice_token
        base_name, _ = parse_slice_token(env_name)
        env = ogbench.make_env_and_datasets(base_name, env_only=True)
        base = env.unwrapped
        return (
            np.asarray(base.maze_map),
            float(getattr(base, "_maze_unit", 4.0)),
            float(getattr(base, "_offset_x", 4.0)),
            float(getattr(base, "_offset_y", 4.0)),
        )
    except Exception as e:  # pragma: no cover - purely cosmetic background
        print(f"[warn] could not load maze layout ({e}); plotting without walls")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="antmaze-medium-stitch-slice50-v0")
    p.add_argument("--data_path", default=None,
                   help="Defaults to ~/.ogbench/data/<env_name>.npz")
    p.add_argument("--num_traj", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_path", default=None)
    args = p.parse_args()

    data_path = args.data_path or os.path.expanduser(
        f"~/.ogbench/data/{args.env_name}.npz")
    data = np.load(data_path)
    obs, terminals = data["observations"], data["terminals"]

    bounds = episode_bounds(terminals)
    lengths = np.array([b - a for a, b in bounds])
    print(f"{len(bounds)} episodes, length min/median/max = "
          f"{lengths.min()}/{int(np.median(lengths))}/{lengths.max()}")

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(bounds), size=min(args.num_traj, len(bounds)), replace=False)
    trajs = [obs[bounds[i][0]:bounds[i][1], :2] for i in picks]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    walls = maze_walls(args.env_name)
    if walls is not None:
        maze_map, unit, offx, offy = walls
        rows, cols = maze_map.shape
        for i in range(rows):
            for j in range(cols):
                if maze_map[i, j] == 1:
                    ax.add_patch(Rectangle(
                        (j * unit - offx - unit / 2, i * unit - offy - unit / 2),
                        unit, unit, facecolor="#9aa0a6", edgecolor="none", alpha=0.28,
                        zorder=0))

    for k, xy in enumerate(trajs):
        color = PALETTE[k % len(PALETTE)]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.0, alpha=0.95,
                solid_capstyle="round", zorder=2, label=f"traj {picks[k]} ({len(xy)} steps)")
        ax.scatter(*xy[0], s=48, facecolor=color, edgecolor="white", linewidths=1.4,
                   marker="o", zorder=3)
        ax.scatter(*xy[-1], s=70, facecolor=color, edgecolor="white", linewidths=1.4,
                   marker="X", zorder=3)

    # Identity is not color-alone: legend lists every series, and markers
    # disambiguate start (circle) from end (X).
    ax.scatter([], [], s=48, facecolor="none", edgecolor="#3c4043", marker="o", label="start")
    ax.scatter([], [], s=70, facecolor="none", edgecolor="#3c4043", marker="X", label="end")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{args.env_name}: {len(trajs)} sampled 50-step trajectories")
    ax.grid(True, color="#dadce0", linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)

    save_path = args.save_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"{args.env_name}_traj50.png")
    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
