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
        description="Cube empowerment map: gripper hovers over one cube, sweeps height vs openness."
    )
    parser.add_argument("--ckpt_root", type=str, default="ckpts")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--grid_res", type=int, default=30)
    parser.add_argument("--num_splus_samples", type=int, default=192)
    parser.add_argument("--z_min", type=float, default=0.03, help="Gripper z min (above cube center).")
    parser.add_argument("--z_max", type=float, default=0.25, help="Gripper z max.")
    parser.add_argument("--open_min", type=float, default=0.0)
    parser.add_argument("--open_max", type=float, default=1.0)
    parser.add_argument("--cube_x_min", type=float, default=0.35)
    parser.add_argument("--cube_x_max", type=float, default=0.50)
    parser.add_argument("--cube_y_min", type=float, default=-0.20)
    parser.add_argument("--cube_y_max", type=float, default=0.20)
    parser.add_argument(
        "--cube_xy",
        type=str,
        default=None,
        help="Fixed target-cube 'x,y'. If omitted, samples 9 random cube xys and plots a 3x3 grid.",
    )
    parser.add_argument(
        "--park_xy",
        type=str,
        default="0.525,0.24",
        help="Parking xy for non-target cubes in double/multi-cube envs.",
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
    park_xy = _parse_xy(args.park_xy)

    z_low, z_high = args.z_min, args.z_max
    o_low, o_high = args.open_min, args.open_max
    zs = np.linspace(z_low, z_high, args.grid_res, dtype=np.float32)
    os_ = np.linspace(o_low, o_high, args.grid_res, dtype=np.float32)
    zz, oo = np.meshgrid(zs, os_)
    flat_z = zz.reshape(-1)
    flat_o = oo.reshape(-1)
    num_points = flat_z.shape[0]

    def compute_empowerment_map(cube_xy):
        obs_list = []
        for z, op in zip(flat_z, flat_o):
            _set_cube(base_env, 0, (cube_xy[0], cube_xy[1], 0.02))
            for i in range(1, num_cubes):
                _set_cube(base_env, i, (park_xy[0], park_xy[1], 0.02))
            _set_gripper_pose(base_env, (float(cube_xy[0]), float(cube_xy[1]), float(z)))
            _set_gripper_openness(base_env, float(op))
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

    if args.cube_xy is not None:
        cube_xys = [_parse_xy(args.cube_xy)]
        force_grid = False
    else:
        cube_xys = [
            (
                float(rng.uniform(args.cube_x_min, args.cube_x_max)),
                float(rng.uniform(args.cube_y_min, args.cube_y_max)),
            )
            for _ in range(9)
        ]
        force_grid = True

    maps = [compute_empowerment_map(cxy) for cxy in cube_xys]

    out_img = args.output if args.output is not None else os.path.join(
        run_dir, f"empowerment_cube_height_aperture_e{epoch}.png"
    )
    out_npy = os.path.splitext(out_img)[0] + ".npy"

    if not force_grid and len(cube_xys) == 1:
        cx, cy = cube_xys[0]
        plt.figure(figsize=(7, 6))
        im = plt.imshow(
            maps[0],
            origin="lower",
            extent=[z_low, z_high, o_low, o_high],
            aspect="auto",
            cmap="viridis",
        )
        plt.colorbar(im, label="Empowerment")
        plt.xlabel("Gripper height z (above table)")
        plt.ylabel("Gripper openness (0=closed, 1=open)")
        parked_str = f" | park=({park_xy[0]:.2f},{park_xy[1]:.2f})" if num_cubes > 1 else ""
        plt.title(
            f"Empowerment | cube=({cx:.2f},{cy:.2f}){parked_str}\n"
            f"run={os.path.basename(run_dir)} | epoch={epoch}"
        )
        plt.tight_layout()
        plt.savefig(out_img, dpi=180)
        np.save(out_npy, maps[0])
    else:
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        axes = axes.flatten()
        for ax, emp_map, (cx, cy) in zip(axes, maps, cube_xys):
            im = ax.imshow(
                emp_map,
                origin="lower",
                extent=[z_low, z_high, o_low, o_high],
                aspect="auto",
                cmap="viridis",
            )
            ax.set_title(f"cube=({cx:.2f},{cy:.2f})", fontsize=9)
            ax.set_xlabel("z (height)")
            ax.set_ylabel("openness")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        parked_str = f" | park=({park_xy[0]:.2f},{park_xy[1]:.2f})" if num_cubes > 1 else ""
        fig.suptitle(
            f"Empowerment | height vs openness{parked_str} | "
            f"run={os.path.basename(run_dir)} | epoch={epoch}"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_img, dpi=180)
        np.save(out_npy, np.stack(maps, axis=0))

    print(f"Saved image: {out_img}")
    print(f"Saved array: {out_npy}")


if __name__ == "__main__":
    main()
