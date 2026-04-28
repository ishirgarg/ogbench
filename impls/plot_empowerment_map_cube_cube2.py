import argparse
import glob
import json
import os
import re

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from agents import agents as agent_registry
from ogbench.manipspace import lie
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


def _parse_xy(text: str) -> tuple[float, float]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'x,y', got: {text}")
    return float(parts[0]), float(parts[1])


def _set_gripper_pose(base_env, xyz, yaw=0.0):
    eff_ori = lie.SO3.from_z_radians(yaw) @ base_env._effector_down_rotation
    T_wp = lie.SE3.from_rotation_and_translation(eff_ori, np.asarray(xyz, dtype=np.float64))
    T_wa = T_wp @ base_env._T_pa
    qpos = base_env._ik.solve(
        pos=T_wa.translation(),
        quat=T_wa.rotation().wxyz,
        curr_qpos=base_env._data.qpos[base_env._arm_joint_ids],
    )
    base_env._data.qpos[base_env._arm_joint_ids] = qpos


def _set_gripper_openness(base_env, openness):
    base_env._data.qpos[base_env._gripper_opening_joint_id] = float(openness) * 0.8


def _set_cube(base_env, i, xyz):
    base_env._data.joint(f'object_joint_{i}').qpos[:3] = np.asarray(xyz, dtype=np.float64)
    base_env._data.joint(f'object_joint_{i}').qpos[3:] = lie.SO3.identity().wxyz


def main():
    parser = argparse.ArgumentParser(
        description="Cube-double empowerment map: fix gripper near center (jittered) and cube1, sweep cube2 XY."
    )
    parser.add_argument("--ckpt_root", type=str, default="ckpts/cube2")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--grid_res", type=int, default=30)
    parser.add_argument("--num_splus_samples", type=int, default=192)
    parser.add_argument("--x_min", type=float, default=0.30, help="Cube2 sweep min x.")
    parser.add_argument("--x_max", type=float, default=0.55, help="Cube2 sweep max x.")
    parser.add_argument("--y_min", type=float, default=-0.30, help="Cube2 sweep min y.")
    parser.add_argument("--y_max", type=float, default=0.30, help="Cube2 sweep max y.")
    parser.add_argument("--cube_x_min", type=float, default=0.30, help="Cube1 random-sampling min x.")
    parser.add_argument("--cube_x_max", type=float, default=0.55)
    parser.add_argument("--cube_y_min", type=float, default=-0.30)
    parser.add_argument("--cube_y_max", type=float, default=0.30)
    parser.add_argument(
        "--gripper_center",
        type=str,
        default="0.425,0.0",
        help="Gripper xy center. Each panel jitters this by --gripper_jitter.",
    )
    parser.add_argument("--gripper_jitter", type=float, default=0.03)
    parser.add_argument("--gripper_z", type=float, default=0.15)
    parser.add_argument("--openness", type=float, default=1.0)
    parser.add_argument(
        "--cube1_xy",
        type=str,
        default=None,
        help="Fixed cube1 'x,y'. If omitted, samples a random cube1 per panel.",
    )
    parser.add_argument(
        "--mask_radius",
        type=float,
        default=0.05,
        help="Grid cells within this radius of cube1 are masked (NaN) to avoid penetration artifacts.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Number of grid points to evaluate per empowerment batch (avoids OOM on large grids).",
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

    env.reset()
    base_env = env.unwrapped
    num_cubes = base_env._num_cubes
    if num_cubes != 2:
        raise ValueError(
            f"This script is for cube-double (num_cubes=2), but the loaded env has num_cubes={num_cubes}."
        )

    rng = np.random.default_rng(args.seed)
    gripper_center = _parse_xy(args.gripper_center)

    x_low, x_high = args.x_min, args.x_max
    y_low, y_high = args.y_min, args.y_max
    xs = np.linspace(x_low, x_high, args.grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, args.grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    num_points = flat_x.shape[0]

    @jax.jit
    def _emp_batch(obs_b, keys_b):
        return jax.vmap(
            lambda ob, key: agent.empowerment(ob[None, ...], rng=key).squeeze(),
            in_axes=(0, 0),
        )(obs_b, keys_b)

    def compute_empowerment_map(gripper_xy, cube1_xy):
        obs_list = []
        for cx2, cy2 in zip(flat_x, flat_y):
            _set_cube(base_env, 0, (cube1_xy[0], cube1_xy[1], 0.02))
            _set_cube(base_env, 1, (float(cx2), float(cy2), 0.02))
            _set_gripper_pose(base_env, (gripper_xy[0], gripper_xy[1], args.gripper_z))
            _set_gripper_openness(base_env, args.openness)
            mujoco.mj_forward(base_env._model, base_env._data)
            obs_list.append(np.asarray(base_env.compute_observation(), dtype=np.float32))
        obs_batch = np.stack(obs_list, axis=0)

        obs_batch_jnp = jnp.asarray(obs_batch)
        root_key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
        point_keys = jax.random.split(root_key, num_points)
        # Batch over grid points to avoid OOM on large grids / large num_splus_samples.
        batch_size = max(1, int(args.batch_size))
        emp_chunks = []
        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            emp_chunks.append(np.asarray(_emp_batch(obs_batch_jnp[start:end], point_keys[start:end])))
            print(f"  empowerment batch {start}:{end} / {num_points}")
        emp = np.concatenate(emp_chunks, axis=0)
        emp_map = emp.reshape(args.grid_res, args.grid_res)

        if args.mask_radius > 0.0:
            dx = xx - cube1_xy[0]
            dy = yy - cube1_xy[1]
            mask = (dx * dx + dy * dy) < (args.mask_radius ** 2)
            emp_map = emp_map.copy()
            emp_map[mask] = np.nan
        return emp_map

    fixed_cube1 = _parse_xy(args.cube1_xy) if args.cube1_xy is not None else None

    if fixed_cube1 is not None:
        panels = [
            (gripper_center, fixed_cube1),
        ]
        force_grid = False
    else:
        panels = []
        for _ in range(9):
            gx = gripper_center[0] + float(rng.uniform(-args.gripper_jitter, args.gripper_jitter))
            gy = gripper_center[1] + float(rng.uniform(-args.gripper_jitter, args.gripper_jitter))
            c1 = (
                float(rng.uniform(args.cube_x_min, args.cube_x_max)),
                float(rng.uniform(args.cube_y_min, args.cube_y_max)),
            )
            panels.append(((gx, gy), c1))
        force_grid = True

    maps = [compute_empowerment_map(g, c1) for (g, c1) in panels]

    out_img = args.output if args.output is not None else os.path.join(
        run_dir, f"empowerment_cube_cube2_e{epoch}.png"
    )
    out_npy = os.path.splitext(out_img)[0] + ".npy"

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#444444")

    if not force_grid and len(panels) == 1:
        (g, c1) = panels[0]
        plt.figure(figsize=(7, 6))
        im = plt.imshow(
            maps[0],
            origin="lower",
            extent=[x_low, x_high, y_low, y_high],
            aspect="auto",
            cmap=cmap,
        )
        plt.scatter([c1[0]], [c1[1]], c="red", s=80, marker="s",
                    edgecolors="white", linewidths=1.0, label="cube 1")
        plt.scatter([g[0]], [g[1]], c="yellow", s=90, marker="*",
                    edgecolors="black", linewidths=0.8, label="gripper")
        plt.legend(loc="upper right")
        plt.colorbar(im, label="Empowerment")
        plt.xlabel("Cube 2 x")
        plt.ylabel("Cube 2 y")
        plt.title(
            f"Empowerment (cube2 sweep) | "
            f"gripper=({g[0]:.2f},{g[1]:.2f},z={args.gripper_z:.2f}) open={args.openness:.2f}\n"
            f"cube1=({c1[0]:.2f},{c1[1]:.2f}) | run={os.path.basename(run_dir)} | epoch={epoch}"
        )
        plt.tight_layout()
        plt.savefig(out_img, dpi=180)
        np.save(out_npy, maps[0])
    else:
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        axes = axes.flatten()
        for ax, emp_map, (g, c1) in zip(axes, maps, panels):
            im = ax.imshow(
                emp_map,
                origin="lower",
                extent=[x_low, x_high, y_low, y_high],
                aspect="auto",
                cmap=cmap,
            )
            ax.scatter([c1[0]], [c1[1]], c="red", s=50, marker="s",
                       edgecolors="white", linewidths=0.8)
            ax.scatter([g[0]], [g[1]], c="yellow", s=60, marker="*",
                       edgecolors="black", linewidths=0.6)
            ax.set_title(
                f"c1=({c1[0]:.2f},{c1[1]:.2f}) g=({g[0]:.2f},{g[1]:.2f})",
                fontsize=9,
            )
            ax.set_xlabel("Cube 2 x")
            ax.set_ylabel("Cube 2 y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"Empowerment (cube2 sweep) | gripper z={args.gripper_z:.2f}, open={args.openness:.2f} | "
            f"run={os.path.basename(run_dir)} | epoch={epoch}"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_img, dpi=180)
        np.save(out_npy, np.stack(maps, axis=0))

    print(f"Saved image: {out_img}")
    print(f"Saved array: {out_npy}")


if __name__ == "__main__":
    main()
