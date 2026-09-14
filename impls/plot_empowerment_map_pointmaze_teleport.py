import argparse
import glob
import json
import os
import re
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from agents import agents as agent_registry
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent
from plot_empowerment_map_antmaze import (
    _draw_interval_dots,
    _parse_xy,
    _raise_time_limit,
    rollout_skill_ant,
)
from matplotlib.patches import Rectangle, Circle


def _latest_run_dir(ckpt_root: str) -> str:
    run_dirs = [p for p in glob.glob(os.path.join(ckpt_root, "*")) if os.path.isdir(p)]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {ckpt_root}")
    run_dirs.sort(key=os.path.getmtime)
    return run_dirs[-1]


def _latest_epoch(run_dir: str) -> int:
    ckpts = glob.glob(os.path.join(run_dir, "params_*.pkl"))
    if not ckpts:
        raise FileNotFoundError(f"No params_*.pkl found in {run_dir}")
    epochs = []
    for path in ckpts:
        m = re.search(r"params_(\d+)\.pkl$", os.path.basename(path))
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        raise RuntimeError(f"Could not parse checkpoint epochs in {run_dir}")
    return max(epochs)


def _parse_int_list(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def plot_point_paths(xy_per_skill, start_xy, overlay_maze, overlay_teleporters,
                     extent, output_path, title=None):
    """One 2D plot: a thin line per skill (point xy over time), start marked.

    Same figure as the AntMaze skill-path plot, plus the teleporter rings and
    PointMaze axis labels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    overlay_maze(ax)
    overlay_teleporters(ax)

    K = len(xy_per_skill)
    cmap = plt.get_cmap("hsv")
    for z, xy in enumerate(xy_per_skill):
        ax.plot(xy[:, 0], xy[:, 1], color=cmap(z / max(K, 1)), linewidth=0.6,
                alpha=0.9, label=f"skill {z}")

    ax.scatter([start_xy[0]], [start_xy[1]], c="black", s=40, marker="o",
               edgecolors="white", linewidths=0.8, zorder=5, label="Point start")

    interval_handles = _draw_interval_dots(ax, xy_per_skill)
    interval_handles = list(interval_handles) + [
        plt.Line2D([0], [0], color="cyan", linewidth=2.0, label="teleport in"),
        plt.Line2D([0], [0], color="red", linewidth=2.0, linestyle="--",
                   label="teleport out"),
    ]

    x_lo, x_hi, y_lo, y_hi = extent
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Point x")
    ax.set_ylabel("Point y")
    if title is not None:
        ax.set_title(title)
    if K <= 15:
        skill_leg = ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        ax.add_artist(skill_leg)
    ax.legend(handles=interval_handles, loc="lower left", fontsize=6,
              framealpha=0.85, title="interval")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PointMaze-Teleport empowerment map (same computation as Ant Soccer / AntMaze).")
    parser.add_argument("--ckpt_root", type=str, default="ckpts", help="Root checkpoint directory.")
    parser.add_argument("--run_dir", type=str, default=None, help="Explicit run dir (overrides latest in ckpt_root).")
    parser.add_argument("--epoch", type=int, default=None, help="Explicit epoch (overrides latest params_*.pkl).")
    parser.add_argument("--grid_res", type=int, default=200, help="Grid resolution for point XY map.")
    parser.add_argument("--x_min", type=float, default=0.0, help="Grid min x for point position.")
    parser.add_argument("--x_max", type=float, default=36.0, help="Grid max x for point position.")
    parser.add_argument("--y_min", type=float, default=0.0, help="Grid min y for point position.")
    parser.add_argument("--y_max", type=float, default=24.0, help="Grid max y for point position.")
    parser.add_argument(
        "--num_splus_samples",
        type=int,
        default=384,
        help="Number of s+ samples used in empowerment Monte Carlo estimate.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for per-point RNG.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of grid points to evaluate per empowerment batch (avoids OOM on large grids).",
    )
    parser.add_argument("--output", type=str, default=None, help="Output image path (.png). Defaults to run dir.")
    # -- Skill-rollout flags (mirror the antmaze script) --------------------
    parser.add_argument(
        "--skill_paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a 2D plot with one point-path line per skill. "
             "On by default; pass --no-skill_paths to disable.",
    )
    parser.add_argument(
        "--skill_map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the empowerment map. On by default; pass --no-skill_map "
             "to skip the (slow) map computation.",
    )
    parser.add_argument("--path_steps", type=int, default=3000,
                        help="Env steps per skill rollout for --skill_paths.")
    parser.add_argument("--point_xy", type=str, default=None,
                        help="Fixed point x,y start for the skill rollouts (e.g. '12,12'). "
                             "Defaults to a random valid (non-wall) cell.")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir is not None else _latest_run_dir(args.ckpt_root)
    epoch = args.epoch if args.epoch is not None else _latest_epoch(run_dir)

    flags_path = os.path.join(run_dir, "flags.json")
    if not os.path.exists(flags_path):
        raise FileNotFoundError(f"flags.json not found in {run_dir}")
    with open(flags_path, "r") as f:
        flags = json.load(f)

    agent_cfg = flags["agent"]
    # Increase MC fidelity for plotting.
    agent_cfg["num_splus_samples"] = int(args.num_splus_samples)
    env_name = flags["env_name"]

    # Build PointMaze env/dataset and agent
    env, train_dataset, _ = make_env_and_datasets(env_name, frame_stack=agent_cfg.get("frame_stack"))
    example_batch = train_dataset.sample(1)
    if agent_cfg.get("discrete"):
        example_batch["actions"] = np.full_like(example_batch["actions"], env.action_space.n - 1)

    agent_class = agent_registry[agent_cfg["agent_name"]]
    agent = agent_class.create(
        seed=flags.get("seed", 0),
        ex_observations=example_batch["observations"],
        ex_actions=example_batch["actions"],
        config=agent_cfg,
    )
    agent = restore_agent(agent, run_dir, epoch)

    obs0, _ = env.reset()
    obs0 = np.asarray(obs0, dtype=np.float32)
    base_env = env.unwrapped

    x_low, x_high = float(args.x_min), float(args.x_max)
    y_low, y_high = float(args.y_min), float(args.y_max)
    unit = getattr(base_env, "_maze_unit", 4.0)
    half = float(unit) / 2.0
    # Always expand the plotted extent by half a cell on all borders.
    x_low_plot, x_high_plot = x_low - half, x_high + half
    y_low_plot, y_high_plot = y_low - half, y_high + half
    # Overlay helpers: draw maze walls and teleporter rings
    maze_map = getattr(base_env, "maze_map", None)
    offx = getattr(base_env, "_offset_x", 4.0)
    offy = getattr(base_env, "_offset_y", 4.0)
    teleport_info = getattr(base_env, "_teleport_info", None)

    def overlay_maze(ax):
        if maze_map is None:
            return
        rows, cols = maze_map.shape
        for i in range(rows):
            for j in range(cols):
                if maze_map[i, j] == 1:
                    cx = j * unit - offx
                    cy = i * unit - offy
                    llx = cx - unit / 2.0
                    lly = cy - unit / 2.0
                    rect = Rectangle(
                        (llx, lly),
                        unit,
                        unit,
                        facecolor="black",
                        edgecolor="black",
                        linewidth=0.3,
                        alpha=0.2,
                    )
                    ax.add_patch(rect)

    def overlay_teleporters(ax):
        if teleport_info is None:
            return
        radius = float(teleport_info.get("teleport_radius", 1.0))
        for (x, y) in teleport_info.get("teleport_in_xys", []):
            ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="cyan", linewidth=2.0, label="_teleport_in"))
        for (x, y) in teleport_info.get("teleport_out_xys", []):
            ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="red", linewidth=2.0, linestyle="--", label="_teleport_out"))

    # -- Skill-rollout branch (per-skill point paths) -----------------------
    def is_valid_xy(x, y):
        if maze_map is None:
            return True
        j = int(round((x + offx) / unit))
        i_ = int(round((y + offy) / unit))
        rows, cols = maze_map.shape
        if i_ < 0 or i_ >= rows or j < 0 or j >= cols:
            return False
        return int(maze_map[i_, j]) != 1

    def sample_valid_xy():
        for _ in range(10000):
            x = float(np.random.uniform(x_low, x_high))
            y = float(np.random.uniform(y_low, y_high))
            if is_valid_xy(x, y):
                return np.array([x, y], dtype=np.float64)
        raise RuntimeError("Could not sample a valid (non-wall) point position.")

    if args.skill_paths:
        num_skills = int(agent_cfg.get("num_skills"))
        # Lift the 1000-step TimeLimit so paths run the full requested horizon.
        _raise_time_limit(env, args.path_steps)

        if args.point_xy is not None:
            px_, py_ = _parse_xy(args.point_xy)
            point_xy = np.array([px_, py_], dtype=np.float64)
        else:
            point_xy = sample_valid_xy()

        print(f"Skill rollouts: K={num_skills}, steps={args.path_steps}, "
              f"point={point_xy.tolist()}")

        xy_per_skill = []
        for z in range(num_skills):
            print(f"  rolling out skill {z + 1}/{num_skills}...")
            _, xy_traj = rollout_skill_ant(
                env=env, agent=agent, num_skills=num_skills, skill_id=z,
                ant_xy=point_xy, n_steps=args.path_steps, frame_skip=1,
                temperature=0.0, seed=z, collect_frames=False, collect_xy=True,
            )
            xy_per_skill.append(xy_traj)

        paths_out = os.path.join(run_dir, f"skill_point_paths_e{epoch}.png")
        plot_point_paths(
            xy_per_skill=xy_per_skill,
            start_xy=point_xy,
            overlay_maze=overlay_maze,
            overlay_teleporters=overlay_teleporters,
            extent=(x_low_plot, x_high_plot, y_low_plot, y_high_plot),
            output_path=paths_out,
            title=(f"Point paths | run={os.path.basename(run_dir.rstrip('/'))} | epoch={epoch}\n"
                   f"K={num_skills}, steps={args.path_steps}, "
                   f"start=({point_xy[0]:.2f}, {point_xy[1]:.2f})"),
        )
        print(f"Saved skill point-path plot: {paths_out}")

    if not args.skill_map:
        return

    xs = np.linspace(x_low, x_high, args.grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, args.grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)

    # Pre-assemble observation batch template and just overwrite XY
    obs_batch = np.repeat(obs0[None, :], args.grid_res * args.grid_res, axis=0)
    obs_batch[:, 0] = flat_x
    obs_batch[:, 1] = flat_y
    obs_batch_jnp = jnp.asarray(obs_batch)

    # Per-point RNG root
    point_root_seed = int(np.random.default_rng(args.seed).integers(0, 2**31 - 1))
    point_root_key = jax.random.PRNGKey(point_root_seed)
    num_points = obs_batch_jnp.shape[0]
    point_keys = jax.random.split(point_root_key, num_points)
    # Compute empowerment exactly like Ant Soccer: agent.empowerment averaged over skills internally.
    # Batch over grid points to avoid OOM on large grids / large num_splus_samples.
    @jax.jit
    def _emp_batch(obs_b, keys_b):
        return jax.vmap(
            lambda ob, key: agent.empowerment(ob[None, ...], rng=key).squeeze(),
            in_axes=(0, 0),
        )(obs_b, keys_b)

    batch_size = max(1, int(args.batch_size))
    emp_chunks = []
    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        emp_chunks.append(np.asarray(_emp_batch(obs_batch_jnp[start:end], point_keys[start:end])))
        print(f"  empowerment batch {start}:{end} / {num_points}")
    emp_vals = np.concatenate(emp_chunks, axis=0)
    emp_map = emp_vals.reshape(args.grid_res, args.grid_res)

    # Plot single heatmap (same style as AntMaze)
    out_img = args.output if args.output is not None else os.path.join(run_dir, f"empowerment_pointmaze_teleport_e{epoch}.png")
    out_npy = os.path.splitext(out_img)[0] + ".npy"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(
        emp_map,
        origin="lower",
        extent=[x_low_plot, x_high_plot, y_low_plot, y_high_plot],
        aspect="auto",
        cmap="viridis",
    )
    overlay_maze(ax)
    overlay_teleporters(ax)
    # Manual legend entries for the teleporter rings.
    legend_handles = [
        plt.Line2D([0], [0], color="cyan", linewidth=2.0, label="teleport in"),
        plt.Line2D([0], [0], color="red", linewidth=2.0, linestyle="--", label="teleport out"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_xlabel("Point x")
    ax.set_ylabel("Point y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"PointMaze-Teleport empowerment | run={os.path.basename(run_dir.rstrip('/'))} | epoch={epoch}")
    plt.tight_layout()
    plt.savefig(out_img, dpi=180)
    np.save(out_npy, emp_map)
    print(f"Saved image: {out_img}")
    print(f"Saved array: {out_npy} (shape {emp_map.shape})")


if __name__ == "__main__":
    main()
