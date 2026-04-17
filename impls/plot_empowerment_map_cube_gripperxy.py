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


def _parse_xys(text: str) -> list[tuple[float, float]]:
    return [_parse_xy(p) for p in text.split(";")]


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
        description="Cube empowerment map: sweep gripper XY with cubes fixed, at a single gripper openness."
    )
    parser.add_argument("--ckpt_root", type=str, default="ckpts/cube1")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--grid_res", type=int, default=30)
    parser.add_argument("--num_splus_samples", type=int, default=192)
    parser.add_argument("--x_min", type=float, default=0.25, help="Gripper sweep min x.")
    parser.add_argument("--x_max", type=float, default=0.60, help="Gripper sweep max x.")
    parser.add_argument("--y_min", type=float, default=-0.35, help="Gripper sweep min y.")
    parser.add_argument("--y_max", type=float, default=0.35, help="Gripper sweep max y.")
    parser.add_argument("--cube_x_min", type=float, default=0.30)
    parser.add_argument("--cube_x_max", type=float, default=0.55)
    parser.add_argument("--cube_y_min", type=float, default=-0.30)
    parser.add_argument("--cube_y_max", type=float, default=0.30)
    parser.add_argument("--gripper_z", type=float, default=0.05, help="Gripper z during the sweep.")
    parser.add_argument(
        "--openness",
        type=float,
        default=0.0,
        help="Gripper openness in [0, 1]. 0=closed, 1=open.",
    )
    parser.add_argument(
        "--cube_xys",
        type=str,
        default=None,
        help="Fixed cube positions as 'x,y' or 'x1,y1;x2,y2'. If omitted, samples 9 random layouts.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
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

    rng = np.random.default_rng(args.seed)

    x_low, x_high = args.x_min, args.x_max
    y_low, y_high = args.y_min, args.y_max
    xs = np.linspace(x_low, x_high, args.grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, args.grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    num_points = flat_x.shape[0]

    def compute_empowerment_map(cube_xys):
        obs_list = []
        for gx, gy in zip(flat_x, flat_y):
            for i in range(num_cubes):
                _set_cube(base_env, i, (cube_xys[i][0], cube_xys[i][1], 0.02))
            _set_gripper_pose(base_env, (float(gx), float(gy), args.gripper_z))
            _set_gripper_openness(base_env, args.openness)
            mujoco.mj_forward(base_env._model, base_env._data)
            obs_list.append(np.asarray(base_env.compute_observation(), dtype=np.float32))
        obs_batch = np.stack(obs_list, axis=0)

        obs_batch_jnp = jnp.asarray(obs_batch)
        root_key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
        point_keys = jax.random.split(root_key, num_points)
        emp = np.asarray(
            jax.vmap(
                lambda ob, key: agent.empowerment(ob[None, ...], rng=key).squeeze(),
                in_axes=(0, 0),
            )(obs_batch_jnp, point_keys)
        )
        return emp.reshape(args.grid_res, args.grid_res)

    if args.cube_xys is not None:
        parsed = _parse_xys(args.cube_xys)
        if len(parsed) != num_cubes:
            raise ValueError(f"--cube_xys has {len(parsed)} entries but env has {num_cubes} cubes.")
        layouts = [parsed]
        force_grid = False
    else:
        layouts = []
        for _ in range(9):
            layouts.append(
                [
                    (
                        float(rng.uniform(args.cube_x_min, args.cube_x_max)),
                        float(rng.uniform(args.cube_y_min, args.cube_y_max)),
                    )
                    for _ in range(num_cubes)
                ]
            )
        force_grid = True

    maps = [compute_empowerment_map(layout) for layout in layouts]

    state_tag = "closed" if args.openness < 0.5 else "open"
    out_img = args.output if args.output is not None else os.path.join(
        run_dir, f"empowerment_cube_gripperxy_{state_tag}_e{epoch}.png"
    )
    out_npy = os.path.splitext(out_img)[0] + ".npy"

    cube_colors = ["red", "blue", "orange", "green"]

    if not force_grid and len(layouts) == 1:
        layout = layouts[0]
        plt.figure(figsize=(7, 6))
        im = plt.imshow(
            maps[0],
            origin="lower",
            extent=[x_low, x_high, y_low, y_high],
            aspect="auto",
            cmap="viridis",
        )
        for i, (cx, cy) in enumerate(layout):
            plt.scatter(
                [cx], [cy],
                c=cube_colors[i % len(cube_colors)],
                s=80, marker="s", edgecolors="white", linewidths=1.0,
                label=f"cube {i}",
            )
        plt.legend(loc="upper right")
        plt.colorbar(im, label="Empowerment")
        plt.xlabel("Gripper x")
        plt.ylabel("Gripper y")
        plt.title(
            f"Empowerment | gripper {state_tag} (openness={args.openness:.2f}), z={args.gripper_z:.3f}\n"
            f"run={os.path.basename(run_dir)} | epoch={epoch}"
        )
        plt.tight_layout()
        plt.savefig(out_img, dpi=180)
        np.save(out_npy, maps[0])
    else:
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        axes = axes.flatten()
        for ax, emp_map, layout in zip(axes, maps, layouts):
            im = ax.imshow(
                emp_map,
                origin="lower",
                extent=[x_low, x_high, y_low, y_high],
                aspect="auto",
                cmap="viridis",
            )
            for i, (cx, cy) in enumerate(layout):
                ax.scatter(
                    [cx], [cy],
                    c=cube_colors[i % len(cube_colors)],
                    s=50, marker="s", edgecolors="white", linewidths=0.8,
                )
            ax.set_title(
                ", ".join(f"c{i}=({cx:.2f},{cy:.2f})" for i, (cx, cy) in enumerate(layout)),
                fontsize=9,
            )
            ax.set_xlabel("Gripper x")
            ax.set_ylabel("Gripper y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"Empowerment | gripper {state_tag} (openness={args.openness:.2f}), z={args.gripper_z:.3f} | "
            f"run={os.path.basename(run_dir)} | epoch={epoch}"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_img, dpi=180)
        np.save(out_npy, np.stack(maps, axis=0))

    print(f"Saved image: {out_img}")
    print(f"Saved array: {out_npy}")


if __name__ == "__main__":
    main()
