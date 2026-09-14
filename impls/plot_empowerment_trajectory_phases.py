"""
Plot empowerment-vs-time over a real (offline dataset) cube manipulation trajectory,
with background shading for automatically-detected approach / near-cube-closed /
lifted phases derived from the observation itself (gripper-cube distance, gripper
opening, cube height above the table).

This intentionally uses real on-distribution states from the play dataset rather
than a synthetic grid sweep, since empowerment estimators can be unreliable off
their training distribution.
"""

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


def _episode_bounds(terminals):
    """Compact-dataset terminals come in consecutive pairs at episode boundaries;
    collapse runs of adjacent terminal indices and use the last index of each run
    as the true episode end (same convention as plot_empowerment_trajectory_video.py)."""
    term_idx = np.nonzero(terminals > 0)[0]
    if len(term_idx) == 0:
        raise RuntimeError("Offline dataset has no terminals; cannot slice into trajectories.")
    split_points = np.where(np.diff(term_idx) > 1)[0] + 1
    term_groups = np.split(term_idx, split_points)
    end_locs = np.array([int(g[-1]) for g in term_groups])
    start_locs = np.concatenate([[0], end_locs[:-1] + 1])
    return start_locs, end_locs


def _extract_cube_features(obs, n_arm):
    """Decode effector position, gripper opening, and cube position from the raw
    cube_env observation layout (see ogbench/manipspace/envs/cube_env.py)."""
    xyz_center = np.array([0.425, 0.0, 0.0])
    xyz_scaler = 10.0
    gripper_scaler = 3.0
    effector_pos = obs[2 * n_arm:2 * n_arm + 3] / xyz_scaler + xyz_center
    gripper_opening = float(obs[2 * n_arm + 5]) / gripper_scaler
    cube_start = 2 * n_arm + 7
    cube_pos = obs[cube_start:cube_start + 3] / xyz_scaler + xyz_center
    return effector_pos, gripper_opening, cube_pos


def compute_empowerment_timeseries(agent, observations, batch_size=256):
    """Compute empowerment at every timestep, chunked at a fixed `batch_size`.

    agent.empowerment() now streams over skills (lax.scan) and successor-state
    samples (lax.map, chunked via config['emp_sample_chunk_size']) internally,
    so this outer chunking is just to bound the number of states processed per
    XLA call — no special-casing of the shape is needed any more.
    """
    obs_array = np.stack(observations, axis=0).astype(np.float32)
    n = obs_array.shape[0]
    emp_values = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        obs_batch = jnp.asarray(obs_array[start:end])
        rng = jax.random.PRNGKey(start)
        keys = jax.random.split(rng, end - start)
        emp = jax.vmap(
            lambda ob, key: agent.empowerment(ob[None, ...], rng=key).squeeze(),
            in_axes=(0, 0),
        )(obs_batch, keys)
        emp_values.append(np.asarray(emp))
    return np.concatenate(emp_values, axis=0)


def classify_phases(distances, openings, cube_heights, table_z, dist_thresh, lift_thresh):
    """Per-timestep phase labels derived purely from decoded state, not privileged info.

    `openings` is the raw `gripper_opening` observation field, which — despite its
    name — is really a closedness signal: 0=open, 1=closed (it maps directly onto
    the Robotiq driver-joint range 0=open/0.8=closed and the ctrl=255*value command
    convention, where 0=open/255=closed). So "near + closed" means distance small
    AND openings large (close to 1), not small.
    """
    lifted = cube_heights > (table_z + lift_thresh)
    near_and_closed = (distances < dist_thresh) & (openings > 0.5)
    phases = np.full(len(distances), "approach", dtype=object)
    phases[near_and_closed] = "near+closed"
    phases[lifted] = "lifted"
    return phases


PHASE_COLORS = {"approach": "#f0f0f0", "near+closed": "#ffd966", "lifted": "#93c47d"}


def plot_episode(distances, openings, cube_heights, emp_values, phases, table_z, title, out_img):
    n = len(emp_values)
    t = np.arange(n)

    fig, (ax_emp, ax_state) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1]})

    # Shade contiguous phase runs.
    for ax in (ax_emp, ax_state):
        start = 0
        for i in range(1, n + 1):
            if i == n or phases[i] != phases[start]:
                ax.axvspan(start - 0.5, i - 0.5, color=PHASE_COLORS[phases[start]], alpha=0.5, zorder=0)
                start = i

    ax_emp.plot(t, emp_values, color="#1f4e79", linewidth=1.8, zorder=2)
    ax_emp.set_ylabel("Empowerment")
    ax_emp.set_title(title, fontsize=10)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5, label=name)
                      for name, c in PHASE_COLORS.items()]
    ax_emp.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax_state.plot(t, distances, color="#cc0000", linewidth=1.5, label="gripper-cube distance (m)")
    ax_state.plot(t, cube_heights - table_z, color="#38761d", linewidth=1.5, label="cube height above table (m)")
    ax_state.plot(t, openings, color="#674ea7", linewidth=1.2, linestyle="--",
                  label="gripper closedness [0=open,1=closed]")
    ax_state.set_xlabel("Timestep")
    ax_state.set_ylabel("State")
    ax_state.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_img, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Empowerment-vs-time over a real offline cube trajectory, with phase shading."
    )
    parser.add_argument("--ckpt_root", type=str, default="ckpts/cube1")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_splus_samples", type=int, default=192)
    parser.add_argument("--emp_batch_size", type=int, default=256)
    parser.add_argument("--emp_sample_chunk_size", type=int, default=64,
                         help="Chunk size for the internal successor-sample loop in "
                              "agent.empowerment (avoids materializing all splus samples at once).")
    parser.add_argument("--dist_thresh", type=float, default=0.06,
                         help="Gripper-cube distance (m) below which we call it 'near'.")
    parser.add_argument("--lift_thresh", type=float, default=0.03,
                         help="Height (m) above the resting table height that counts as 'lifted'.")
    parser.add_argument("--table_z", type=float, default=0.02)
    parser.add_argument("--require_lift", action="store_true", default=True,
                         help="Resample episodes until one contains a lift event (more informative figure).")
    parser.add_argument("--no_require_lift", dest="require_lift", action="store_false")
    parser.add_argument("--max_search", type=int, default=500,
                         help="Max episodes to search through when --require_lift is set.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_dir = (args.run_dir if args.run_dir is not None else _latest_run_dir(args.ckpt_root)).rstrip("/")
    epoch = args.epoch if args.epoch is not None else _latest_epoch(run_dir)

    flags_path = os.path.join(run_dir, "flags.json")
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
    n_arm = len(base_env._arm_joint_ids)

    terminals = np.asarray(train_dataset["terminals"])
    start_locs, end_locs = _episode_bounds(terminals)
    rng = np.random.default_rng(args.seed)

    chosen = []
    tried = set()
    attempts = 0
    while len(chosen) < args.num_episodes and attempts < max(args.max_search, args.num_episodes):
        idx = int(rng.integers(len(end_locs)))
        attempts += 1
        if idx in tried and args.require_lift:
            continue
        tried.add(idx)
        start, end = int(start_locs[idx]), int(end_locs[idx])
        if args.max_steps is not None:
            end = min(end, start + args.max_steps)
        obs_arr = np.asarray(train_dataset["observations"][start:end + 1])

        if args.require_lift:
            heights = np.array([_extract_cube_features(o, n_arm)[2][2] for o in obs_arr])
            if not np.any(heights > (args.table_z + args.lift_thresh)):
                continue
        chosen.append((idx, start, end, obs_arr))

    if not chosen:
        raise RuntimeError(
            f"Could not find an episode with a lift event after {attempts} attempts; "
            f"try --no_require_lift or a lower --lift_thresh."
        )

    for ep_i, (idx, start, end, obs_arr) in enumerate(chosen):
        print(f"Episode {ep_i}: dataset idx={idx}, start={start}, end={end}, len={len(obs_arr)}")

        features = [_extract_cube_features(o, n_arm) for o in obs_arr]
        distances = np.array([np.linalg.norm(eff - cube) for eff, _, cube in features])
        openings = np.array([op for _, op, _ in features])
        cube_heights = np.array([cube[2] for _, _, cube in features])

        emp_values = compute_empowerment_timeseries(agent, list(obs_arr), batch_size=args.emp_batch_size)

        phases = classify_phases(distances, openings, cube_heights, args.table_z,
                                  args.dist_thresh, args.lift_thresh)

        title = (f"Empowerment over a real trajectory (offline dataset idx={idx})\n"
                 f"run={os.path.basename(run_dir)} | epoch={epoch}")

        if args.output is not None:
            if args.num_episodes == 1:
                out_img = args.output
            else:
                base, ext = os.path.splitext(args.output)
                out_img = f"{base}_ep{ep_i}{ext}"
        else:
            out_img = os.path.join(run_dir, f"empowerment_trajectory_phases_ep{ep_i}_e{epoch}.png")

        plot_episode(distances, openings, cube_heights, emp_values, phases, args.table_z, title, out_img)

        out_npz = os.path.splitext(out_img)[0] + ".npz"
        np.savez(out_npz, distances=distances, openings=openings, cube_heights=cube_heights,
                 emp_values=emp_values, phases=phases)

        print(f"  Saved image: {out_img}")
        print(f"  Saved array: {out_npz}")


if __name__ == "__main__":
    main()
