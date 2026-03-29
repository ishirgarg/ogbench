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
    parser.add_argument("--ant_xy_indices", type=str, default="0,1", help="Observation indices for ant x,y.")
    parser.add_argument("--ball_xy_indices", type=str, default="2,3", help="Observation indices for ball x,y.")
    parser.add_argument(
        "--ball_xy",
        type=str,
        default=None,
        help="Fixed ball x,y as 'x,y'. If omitted, uses ball position from reset observation.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output image path (.png). Defaults to run dir.")
    parser.add_argument(
        "--num_splus_samples",
        type=int,
        default=192,
        help="Number of s+ samples used in empowerment Monte Carlo estimate.",
    )
    parser.add_argument("--x_min", type=float, default=-22.0, help="Grid min x for ant position.")
    parser.add_argument("--x_max", type=float, default=22.0, help="Grid max x for ant position.")
    parser.add_argument("--y_min", type=float, default=-22.0, help="Grid min y for ant position.")
    parser.add_argument("--y_max", type=float, default=22.0, help="Grid max y for ant position.")
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

    ant_x_idx, ant_y_idx = _parse_indices(args.ant_xy_indices)
    ball_x_idx, ball_y_idx = _parse_indices(args.ball_xy_indices)

    obs0, _ = env.reset()
    obs0 = np.asarray(obs0, dtype=np.float32)

    base_env = env.unwrapped
    # Decide ball positions:
    # - If provided, use that single fixed position.
    # - If omitted, create 4 random positions within [x_min,x_max]×[y_min,y_max] and
    #   plot a 2×2 grid of empowerment maps.
    if args.ball_xy is not None:
        bx, by = _parse_indices(args.ball_xy)
        ball_positions = [(float(bx), float(by))]
    else:
        # Unseeded randomness: rely on numpy global RNG state.
        ball_positions = []
        for _ in range(4):
            rx = float(np.random.uniform(low=float(args.x_min), high=float(args.x_max)))
            ry = float(np.random.uniform(low=float(args.y_min), high=float(args.y_max)))
            ball_positions.append((rx, ry))

    x_low, x_high = args.x_min, args.x_max
    y_low, y_high = args.y_min, args.y_max

    xs = np.linspace(x_low, x_high, args.grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, args.grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    def compute_empowerment_map(fixed_ball_xy: tuple[float, float]) -> np.ndarray:
        # Build observations using the env's own state->observation path when available.
        # This guarantees the state vector matches exactly what the environment produces.
        flat_x = xx.reshape(-1)
        flat_y = yy.reshape(-1)
        obs_list = []
        if hasattr(base_env, "set_agent_ball_xy") and (hasattr(base_env, "get_ob") or hasattr(base_env, "_get_obs")):
            for x, y in zip(flat_x, flat_y):
                base_env.set_agent_ball_xy(np.array([x, y], dtype=np.float64), np.array(fixed_ball_xy, dtype=np.float64))
                if hasattr(base_env, "get_ob"):
                    obs_single = np.asarray(base_env.get_ob(), dtype=np.float32)
                else:
                    obs_single = np.asarray(base_env._get_obs(), dtype=np.float32)
                # If frame stacking is enabled, mimic wrapper output by repeating the base observation.
                if obs_single.shape[0] != obs0.shape[0]:
                    stack = obs0.shape[0] // obs_single.shape[0]
                    if stack > 1 and stack * obs_single.shape[0] == obs0.shape[0]:
                        obs_single = np.concatenate([obs_single] * stack, axis=-1)
                obs_list.append(obs_single)
            obs_batch = np.stack(obs_list, axis=0)
        else:
            # Fallback: direct index overwrite (less exact, but keeps script generic).
            obs_batch = np.repeat(obs0[None, :], args.grid_res * args.grid_res, axis=0)
            obs_batch[:, ant_x_idx] = flat_x
            obs_batch[:, ant_y_idx] = flat_y
            obs_batch[:, ball_x_idx] = fixed_ball_xy[0]
            obs_batch[:, ball_y_idx] = fixed_ball_xy[1]
        emp = np.asarray(agent.empowerment(jnp.asarray(obs_batch), rng=jax.random.PRNGKey(0)))
        return emp.reshape(args.grid_res, args.grid_res)

    maps = [compute_empowerment_map(bp) for bp in ball_positions]

    out_img = args.output if args.output is not None else os.path.join(run_dir, f"empowerment_map_e{epoch}.png")
    out_npy = os.path.splitext(out_img)[0] + ".npy"

    if len(ball_positions) == 1:
        fixed_ball = ball_positions[0]
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
        plt.legend(loc="upper right")
        plt.colorbar(im, label="Empowerment")
        plt.xlabel(f"Ant x (obs[{ant_x_idx}])")
        plt.ylabel(f"Ant y (obs[{ant_y_idx}])")
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
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i, (ax, emp_map, bp) in enumerate(zip(axes, maps, ball_positions)):
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
            ax.set_title(f"Ball=({bp[0]:.2f}, {bp[1]:.2f})")
            ax.set_xlabel(f"Ant x (obs[{ant_x_idx}])")
            ax.set_ylabel(f"Ant y (obs[{ant_y_idx}])")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Empowerment maps | run={os.path.basename(run_dir)} | epoch={epoch}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_img, dpi=180)
        np.save(out_npy, np.stack(maps, axis=0))
        print(f"Saved 2x2 image: {out_img}")
        print(f"Saved array stack: {out_npy}")


if __name__ == "__main__":
    main()

