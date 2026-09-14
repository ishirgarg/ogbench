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
        description="Cube empowerment vs. horizontal distance-to-cube, averaged over approach "
                    "angles, compared for a closed vs. open gripper."
    )
    parser.add_argument("--ckpt_root", type=str, default="ckpts/cube1")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--num_distances", type=int, default=30)
    parser.add_argument("--num_angles", type=int, default=16, help="Angles averaged over at each distance.")
    parser.add_argument("--dist_min", type=float, default=0.03)
    parser.add_argument("--dist_max", type=float, default=0.30)
    parser.add_argument("--num_splus_samples", type=int, default=192)
    parser.add_argument("--emp_sample_chunk_size", type=int, default=64,
                         help="Chunk size for the internal successor-sample loop in "
                              "agent.empowerment (avoids materializing all splus samples at once).")
    parser.add_argument("--gripper_z", type=float, default=0.05, help="Gripper z during the sweep.")
    parser.add_argument(
        "--opennesses",
        type=str,
        default="0.0,1.0",
        help="Comma-separated gripper closedness values (0=open, 1=closed; despite the "
             "flag name) to compare as separate curves.",
    )
    parser.add_argument("--cube_xy", type=str, default="0.425,0.0", help="Fixed cube position 'x,y'.")
    parser.add_argument("--park_xy", type=str, default="0.525,0.24",
                         help="Parking xy for non-target cubes in multi-cube envs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    run_dir = (args.run_dir if args.run_dir is not None else _latest_run_dir(args.ckpt_root)).rstrip("/")
    epoch = args.epoch if args.epoch is not None else _latest_epoch(run_dir)

    flags_path = os.path.join(run_dir, "flags.json")
    if not os.path.exists(flags_path):
        raise FileNotFoundError(f"flags.json not found in {run_dir}")
    with open(flags_path, "r") as f:
        flags = json.load(f)

    agent_cfg = flags["agent"]
    agent_cfg["num_splus_samples"] = int(args.num_splus_samples)
    agent_cfg["emp_sample_chunk_size"] = int(args.emp_sample_chunk_size)
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
    cube_xy = _parse_xy(args.cube_xy)
    park_xy = _parse_xy(args.park_xy)
    opennesses = [float(x) for x in args.opennesses.split(",")]

    distances = np.linspace(args.dist_min, args.dist_max, args.num_distances, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, args.num_angles, endpoint=False, dtype=np.float32)

    @jax.jit
    def _emp_batch(obs_b, keys_b):
        return jax.vmap(
            lambda ob, key: agent.empowerment(ob[None, ...], rng=key).squeeze(),
            in_axes=(0, 0),
        )(obs_b, keys_b)

    def compute_curve(openness):
        obs_list = []
        for d in distances:
            for a in angles:
                gx = cube_xy[0] + float(d) * np.cos(a)
                gy = cube_xy[1] + float(d) * np.sin(a)
                _set_cube(base_env, 0, (cube_xy[0], cube_xy[1], 0.02))
                for i in range(1, num_cubes):
                    _set_cube(base_env, i, (park_xy[0], park_xy[1], 0.02))
                _set_gripper_pose(base_env, (gx, gy, args.gripper_z))
                _set_gripper_openness(base_env, openness)
                mujoco.mj_forward(base_env._model, base_env._data)
                obs_list.append(np.asarray(base_env.compute_observation(), dtype=np.float32))
        obs_batch = np.stack(obs_list, axis=0)
        num_points = obs_batch.shape[0]

        obs_batch_jnp = jnp.asarray(obs_batch)
        root_key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
        point_keys = jax.random.split(root_key, num_points)
        batch_size = max(1, int(args.batch_size))
        emp_chunks = []
        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            emp_chunks.append(np.asarray(_emp_batch(obs_batch_jnp[start:end], point_keys[start:end])))
            print(f"  openness={openness:.2f} empowerment batch {start}:{end} / {num_points}")
        emp = np.concatenate(emp_chunks, axis=0)
        return emp.reshape(args.num_distances, args.num_angles)

    curves = {openness: compute_curve(openness) for openness in opennesses}

    out_img = args.output if args.output is not None else os.path.join(
        run_dir, f"empowerment_cube_distance_curve_e{epoch}.png"
    )
    out_npy = os.path.splitext(out_img)[0] + ".npz"

    plt.figure(figsize=(7, 5.5))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(opennesses)))
    for color, openness in zip(colors, opennesses):
        emp_map = curves[openness]
        mean = emp_map.mean(axis=1)
        std = emp_map.std(axis=1)
        # openness is really closedness: 0=open, 1=closed.
        state_word = "open" if openness < 0.5 else "closed" if openness > 0.5 else "half"
        label = f"{state_word} (closedness={openness:.2f})"
        plt.plot(distances, mean, color=color, linewidth=2.0, label=label)
        plt.fill_between(distances, mean - std, mean + std, color=color, alpha=0.15)
    plt.xlabel("Horizontal distance from gripper to cube (m)")
    plt.ylabel("Empowerment")
    plt.legend(loc="best")
    plt.title(
        f"Empowerment vs. distance-to-cube | z={args.gripper_z:.3f}, avg over {args.num_angles} angles\n"
        f"run={os.path.basename(run_dir)} | epoch={epoch}"
    )
    plt.tight_layout()
    plt.savefig(out_img, dpi=180)
    np.savez(out_npy, distances=distances, angles=angles, **{f"openness_{o}": curves[o] for o in opennesses})

    print(f"Saved image: {out_img}")
    print(f"Saved array: {out_npy}")


if __name__ == "__main__":
    main()
