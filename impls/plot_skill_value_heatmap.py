import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import glob
import json
import re
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from agents import agents as agent_registry
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent


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


def _parse_xy(text: str) -> tuple[float, float]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'x,y', got: {text}")
    return float(parts[0]), float(parts[1])


def _maze_center_xy(maze_map, unit, offx, offy):
    """World xy of the free cell closest to the maze's grid center."""
    rows, cols = maze_map.shape
    center_i, center_j = (rows - 1) / 2.0, (cols - 1) / 2.0
    free_cells = [(i, j) for i in range(rows) for j in range(cols) if maze_map[i, j] == 0]
    if not free_cells:
        raise RuntimeError("Maze has no free cells.")
    bi, bj = min(free_cells, key=lambda c: (c[0] - center_i) ** 2 + (c[1] - center_j) ** 2)
    return float(bj * unit - offx), float(bi * unit - offy)


def main():
    parser = argparse.ArgumentParser(
        description="Plot the learned skill-conditioned value V(s, z, s+) as a heatmap over s+, "
                    "for a fixed start state s and a handful of skills."
    )
    parser.add_argument("--ckpt_root", type=str, default="ckpts", help="Root checkpoint directory.")
    parser.add_argument("--run_dir", type=str, default=None, help="Explicit run dir (overrides latest in ckpt_root).")
    parser.add_argument("--epoch", type=int, default=None, help="Explicit epoch (overrides latest params_*.pkl).")
    parser.add_argument("--grid_res", type=int, default=150, help="Grid resolution for the s+ sweep.")
    parser.add_argument("--x_min", type=float, default=None, help="Grid min x (default: derived from maze_map).")
    parser.add_argument("--x_max", type=float, default=None, help="Grid max x (default: derived from maze_map).")
    parser.add_argument("--y_min", type=float, default=None, help="Grid min y (default: derived from maze_map).")
    parser.add_argument("--y_max", type=float, default=None, help="Grid max y (default: derived from maze_map).")
    parser.add_argument("--start_xy", type=str, default=None,
                        help="Fixed start state s's x,y (e.g. '8,8'). Default: the free cell "
                             "closest to the maze's grid center.")
    parser.add_argument("--skills", type=str, default=None,
                        help="Comma-separated skill indices to plot (default: a spread of "
                             "--num_display_skills evenly spaced skills).")
    parser.add_argument("--num_display_skills", type=int, default=6,
                        help="Number of skills to auto-select when --skills is not given.")
    parser.add_argument("--goal_batch_size", type=int, default=4096,
                        help="Number of s+ grid points embedded per forward pass.")
    parser.add_argument("--output", type=str, default=None, help="Output image path (.png). Defaults to run dir.")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir is not None else _latest_run_dir(args.ckpt_root)
    epoch = args.epoch if args.epoch is not None else _latest_epoch(run_dir)

    flags_path = os.path.join(run_dir, "flags.json")
    if not os.path.exists(flags_path):
        raise FileNotFoundError(f"flags.json not found in {run_dir}")
    with open(flags_path, "r") as f:
        flags = json.load(f)

    agent_cfg = flags["agent"]
    env_name = flags["env_name"]

    env, train_dataset, _ = make_env_and_datasets(env_name, frame_stack=agent_cfg.get("frame_stack"))
    example_batch = train_dataset.sample(1)
    if agent_cfg.get("discrete"):
        example_batch["actions"] = np.full_like(example_batch["actions"], env.action_space.n - 1)

    agent_class = agent_registry[agent_cfg["agent_name"]]
    if not hasattr(agent_class, "skill_values_cross"):
        raise SystemExit(
            f'Agent "{agent_cfg["agent_name"]}" has no `skill_values_cross` hook -- this script only '
            f"supports skill-conditioned value agents (see agents/empowerment_skill.py)."
        )
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

    maze_map = getattr(base_env, "maze_map", None)
    if maze_map is None:
        raise SystemExit(f'"{env_name}" has no maze_map -- this script is maze-only.')
    unit = getattr(base_env, "_maze_unit", 4.0)
    offx = getattr(base_env, "_offset_x", 4.0)
    offy = getattr(base_env, "_offset_y", 4.0)
    rows, cols = maze_map.shape

    half = float(unit) / 2.0
    x_low = args.x_min if args.x_min is not None else -float(offx)
    x_high = args.x_max if args.x_max is not None else float((cols - 1) * unit - offx)
    y_low = args.y_min if args.y_min is not None else -float(offy)
    y_high = args.y_max if args.y_max is not None else float((rows - 1) * unit - offy)
    x_low_plot, x_high_plot = x_low - half, x_high + half
    y_low_plot, y_high_plot = y_low - half, y_high + half

    if args.start_xy is not None:
        start_x, start_y = _parse_xy(args.start_xy)
    else:
        start_x, start_y = _maze_center_xy(maze_map, unit, offx, offy)

    s_obs = obs0.copy()
    s_obs[0], s_obs[1] = start_x, start_y
    s_jnp = jnp.asarray(s_obs)[None, :]

    num_skills = int(agent_cfg["num_skills"])
    if args.skills is not None:
        skill_ids = _parse_int_list(args.skills)
        if not all(0 <= z < num_skills for z in skill_ids):
            raise SystemExit(f"--skills must be indices in [0, {num_skills}), got {args.skills}.")
    else:
        n_show = max(1, min(args.num_display_skills, num_skills))
        skill_ids = sorted(set(np.linspace(0, num_skills - 1, n_show).round().astype(int).tolist()))

    xs = np.linspace(x_low, x_high, args.grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, args.grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    num_points = flat_x.shape[0]

    goal_batch = np.repeat(obs0[None, :], num_points, axis=0)
    goal_batch[:, 0] = flat_x
    goal_batch[:, 1] = flat_y
    goal_batch_jnp = jnp.asarray(goal_batch)

    print(
        f"[{env_name}] agent={agent_cfg['agent_name']} epoch={epoch} K={num_skills} "
        f"start=({start_x:.2f}, {start_y:.2f}) grid={args.grid_res}x{args.grid_res} "
        f"skills={skill_ids}"
    )

    chunk = max(1, int(args.goal_batch_size))
    embed_chunks = []
    for start in range(0, num_points, chunk):
        end = min(start + chunk, num_points)
        embed_chunks.append(np.asarray(agent.value_goal_embeddings(goal_batch_jnp[start:end])))
        print(f"  embedding s+ {end}/{num_points}")
    goal_embeddings = jnp.asarray(np.concatenate(embed_chunks, axis=0))

    values = np.asarray(agent.skill_values_from_goal_embeddings(s_jnp, goal_embeddings))[0]  # [G, K]
    value_map = values.reshape(args.grid_res, args.grid_res, num_skills)

    def overlay_maze(ax):
        for i in range(rows):
            for j in range(cols):
                if maze_map[i, j] == 1:
                    cx = j * unit - offx
                    cy = i * unit - offy
                    rect = Rectangle(
                        (cx - unit / 2.0, cy - unit / 2.0), unit, unit,
                        facecolor="black", edgecolor="white", linewidth=0.4, alpha=0.55,
                    )
                    ax.add_patch(rect)

    # Robust shared color scale across the displayed skills -- a handful of
    # never-visited goal cells can otherwise dominate the min/max and wash out
    # the contrast everywhere else.
    shown = value_map[..., skill_ids]
    vmin, vmax = np.percentile(shown, [1, 99])

    n = len(skill_ids)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False)

    im = None
    for idx, z in enumerate(skill_ids):
        ax = axes[idx // ncols][idx % ncols]
        im = ax.imshow(
            value_map[..., z], origin="lower",
            extent=[x_low_plot, x_high_plot, y_low_plot, y_high_plot],
            aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        overlay_maze(ax)
        ax.scatter([start_x], [start_y], c="red", s=60, marker="*",
                   edgecolors="white", linewidths=0.6, zorder=5)
        ax.set_title(f"skill {z}")
        ax.set_xlim(x_low_plot, x_high_plot)
        ax.set_ylim(y_low_plot, y_high_plot)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        f"V(s, z, s+) | {env_name} | run={os.path.basename(run_dir.rstrip('/'))} | epoch={epoch}\n"
        f"s = ({start_x:.2f}, {start_y:.2f})"
    )
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.88, hspace=0.4, wspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cbar_ax, label="log V")

    out_img = args.output if args.output is not None else os.path.join(
        run_dir, f"skill_value_heatmap_e{epoch}.png"
    )
    out_npy = os.path.splitext(out_img)[0] + ".npy"
    plt.savefig(out_img, dpi=180)
    np.save(out_npy, value_map)
    print(f"Saved image: {out_img}")
    print(f"Saved array: {out_npy} (shape {value_map.shape}, axis order [y, x, skill])")


if __name__ == "__main__":
    main()
