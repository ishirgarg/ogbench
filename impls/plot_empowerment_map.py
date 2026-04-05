import argparse
import glob
import json
import os
import re

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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


def _parse_indices(text: str) -> tuple[int, int]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated indices, got: {text}")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(description="Plot empowerment map from latest checkpoint.")
    parser.add_argument("--ckpt_root", type=str, default="ckpts", help="Root checkpoint directory.")
    parser.add_argument("--run_dir", type=str, default=None, help="Explicit run dir (overrides latest in ckpt_root).")
    parser.add_argument("--epoch", type=int, default=None, help="Explicit epoch (overrides latest params_*.pkl).")
    parser.add_argument("--grid_res", type=int, default=40, help="Grid resolution for ant XY map.")
    # Ant Soccer only; indices are not needed.
    parser.add_argument(
        "--ball_xy",
        type=str,
        default=None,
        help="Fixed ball x,y as 'x,y'. If omitted, uses 9 random positions and plots a 3x3 grid.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output image path (.png). Defaults to run dir.")
    parser.add_argument(
        "--num_splus_samples",
        type=int,
        default=192,
        help="Number of s+ samples used in empowerment Monte Carlo estimate.",
    )
    parser.add_argument("--x_min", type=float, default=-12.0, help="Grid min x for ant position.")
    parser.add_argument("--x_max", type=float, default=12.0, help="Grid max x for ant position.")
    parser.add_argument("--y_min", type=float, default=-12.0, help="Grid min y for ant position.")
    parser.add_argument("--y_max", type=float, default=12.0, help="Grid max y for ant position.")
    parser.add_argument(
        "--goal_xy",
        type=str,
        default=None,
        help="Optional goal x,y as 'x,y'. If provided and env supports set_goal, sets goal once before sweep.",
    )
    parser.add_argument("--fix_ball", action="store_true", help="Fix ball position; sample 9 random goals.")
    parser.add_argument("--fix_goal", action="store_true", help="Fix goal position; sample 9 random balls.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to make ball/goal sampling deterministic across runs.",
    )
    parser.add_argument(
        "--use_rel4_fallback",
        action="store_true",
        help="Use fallback that overwrites only the last 4 obs entries (ball-agent, goal-ball).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir is not None else _latest_run_dir(args.ckpt_root)
    epoch = args.epoch if args.epoch is not None else _latest_epoch(run_dir)

    flags_path = os.path.join(run_dir, "flags.json")
    if not os.path.exists(flags_path):
        raise FileNotFoundError(f"flags.json not found in {run_dir}")
    with open(flags_path, "r") as f:
        flags = json.load(f)

    agent_cfg = flags["agent"]
    # Override MC sample count for plotting fidelity.
    agent_cfg["num_splus_samples"] = int(args.num_splus_samples)
    env_name = flags["env_name"]

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

    x_low, x_high = args.x_min, args.x_max
    y_low, y_high = args.y_min, args.y_max

    xs = np.linspace(x_low, x_high, args.grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, args.grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    def compute_empowerment_map(fixed_ball_xy: tuple[float, float], goal_xy_override: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        # Build observations for Ant Soccer directly via env state, or explicit rel4 fallback.
        flat_x = xx.reshape(-1)
        flat_y = yy.reshape(-1)
        obs_list = []
        # Set a goal for this map: override if provided, else random within bounds.
        if goal_xy_override is not None:
            goal_xy = goal_xy_override.astype(np.float64)
        else:
            goal_xy = np.array(
                [rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)],
                dtype=np.float64,
            )
        if not args.use_rel4_fallback:
            base_env.set_goal(goal_xy=goal_xy)
            for x, y in zip(flat_x, flat_y):
                base_env.set_agent_ball_xy(np.array([x, y], dtype=np.float64), np.array(fixed_ball_xy, dtype=np.float64))
                obs_single = np.asarray(base_env.get_ob(), dtype=np.float32)
                obs_list.append(obs_single)
            obs_batch = np.stack(obs_list, axis=0)
        else:
            # Fallback: directly overwrite Ant and Ball XY in qpos portion of obs = [qpos, qvel].
            # qpos layout: [ant x, ant y, ..., ball x, ball y, ball z, ball quat_w, quat_x, quat_y, quat_z]
            # So ball XY are qpos[-7:-5].
            obs_batch = np.repeat(obs0[None, :], args.grid_res * args.grid_res, axis=0)
            nq = int(base_env.data.qpos.size)
            # Overwrite ant XY (qpos[:2]) per grid point
            obs_batch[:, 0] = flat_x
            obs_batch[:, 1] = flat_y
            # Overwrite ball XY (qpos[-7:-5]) with fixed_ball_xy for the entire map
            obs_batch[:, nq - 7] = float(fixed_ball_xy[0])
            obs_batch[:, nq - 6] = float(fixed_ball_xy[1])

        print(obs_batch[:3])

        # Use a different PRNG key for every point to avoid correlated estimates.
        obs_batch_jnp = jnp.asarray(obs_batch)
        num_points = obs_batch_jnp.shape[0]
        root_seed = int(np.random.randint(0, 2**31 - 1))
        root_key = jax.random.PRNGKey(root_seed)
        point_keys = jax.random.split(root_key, num_points)
        emp = np.asarray(
            jax.vmap(
                lambda ob, key: agent.empowerment(ob[None, ...], rng=key).squeeze(),
                in_axes=(0, 0),
            )(obs_batch_jnp, point_keys)
        )
        # Return the empowerment map and the goal used (if any; otherwise NaNs to force explicitness).
        goal_used = goal_xy.astype(np.float32) if 'goal_xy' in locals() else np.array([np.nan, np.nan], dtype=np.float32)
        return emp.reshape(args.grid_res, args.grid_res), goal_used

    # Determine plotting scenario based on flags
    if args.fix_ball and args.fix_goal:
        raise ValueError("Both --fix_ball and --fix_goal provided; please set only one.")

    rng = np.random.default_rng(args.seed)
    force_grid = False
    goal_overrides: list[np.ndarray | None]

    if args.fix_ball:
        # Fix ball position (from args if provided, else sample once), sample 9 random goals.
        if args.ball_xy is not None:
            bx, by = _parse_indices(args.ball_xy)
            fixed_ball = (float(bx), float(by))
        else:
            fixed_ball = (float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high)))
        ball_positions = [fixed_ball for _ in range(9)]
        goal_overrides = [
            np.array([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)], dtype=np.float64) for _ in range(9)
        ]
        force_grid = True
    elif args.fix_goal:
        # Fix goal position (from args if provided, else sample once), sample 9 random balls.
        if args.goal_xy is not None:
            gx, gy = _parse_indices(args.goal_xy)
            fixed_goal = np.array([float(gx), float(gy)], dtype=np.float64)
        else:
            fixed_goal = np.array([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)], dtype=np.float64)
        ball_positions = [(float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high))) for _ in range(9)]
        goal_overrides = [fixed_goal for _ in range(9)]
        force_grid = True
    else:
        # Default behavior: single plot if ball fixed, else 3x3 random balls and random goals.
        if args.ball_xy is not None:
            bx, by = _parse_indices(args.ball_xy)
            ball_positions = [(float(bx), float(by))]
            goal_overrides = [None]
            force_grid = False
        else:
            ball_positions = [(float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high))) for _ in range(9)]
            goal_overrides = [None for _ in range(9)]
            force_grid = True

    results = [compute_empowerment_map(bp, goal_xy_override=go) for bp, go in zip(ball_positions, goal_overrides)]
    maps = [m for (m, _) in results]
    goals = [g for (_, g) in results]

    out_img = args.output if args.output is not None else os.path.join(run_dir, f"empowerment_map_e{epoch}.png")
    out_npy = os.path.splitext(out_img)[0] + ".npy"

    if not force_grid and len(ball_positions) == 1:
        fixed_ball = ball_positions[0]
        goal_used = goals[0]
        plt.figure(figsize=(7, 6))
        im = plt.imshow(
            maps[0],
            origin="lower",
            extent=[x_low, x_high, y_low, y_high],
            aspect="auto",
            cmap="viridis",
        )
        plt.scatter(
            [fixed_ball[0]],
            [fixed_ball[1]],
            c="red",
            s=60,
            marker="o",
            edgecolors="white",
            linewidths=1.0,
            label="Ball",
        )
        # Plot goal used for this map.
        plt.scatter(
            [goal_used[0]],
            [goal_used[1]],
            c="yellow",
            s=70,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            label="Goal",
        )
        plt.legend(loc="upper right")
        plt.colorbar(im, label="Empowerment")
        plt.xlabel(f"Ant x")
        plt.ylabel(f"Ant y")
        plt.title(
            f"Empowerment map | run={os.path.basename(run_dir)} | epoch={epoch}\n"
            f"fixed ball=({fixed_ball[0]:.3f}, {fixed_ball[1]:.3f})"
        )
        plt.tight_layout()
        plt.savefig(out_img, dpi=180)
        np.save(out_npy, maps[0])
        print(f"Saved image: {out_img}")
        print(f"Saved array: {out_npy}")
    else:
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        axes = axes.flatten()
        for i, (ax, emp_map, bp, goal_used) in enumerate(zip(axes, maps, ball_positions, goals)):
            im = ax.imshow(
                emp_map,
                origin="lower",
                extent=[x_low, x_high, y_low, y_high],
                aspect="auto",
                cmap="viridis",
            )
            ax.scatter(
                [bp[0]],
                [bp[1]],
                c="red",
                s=50,
                marker="o",
                edgecolors="white",
                linewidths=0.8,
                label="Ball",
            )
            ax.scatter(
                [goal_used[0]],
                [goal_used[1]],
                c="yellow",
                s=60,
                marker="*",
                edgecolors="black",
                linewidths=0.6,
                label="Goal",
            )
            ax.set_title(f"Ball=({bp[0]:.2f}, {bp[1]:.2f})")
            ax.set_xlabel(f"Ant x")
            ax.set_ylabel(f"Ant y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Empowerment maps | run={os.path.basename(run_dir)} | epoch={epoch}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_img, dpi=180)
        np.save(out_npy, np.stack(maps, axis=0))
        print(f"Saved 3x3 image: {out_img}")
        print(f"Saved array stack: {out_npy}")


if __name__ == "__main__":
    main()

