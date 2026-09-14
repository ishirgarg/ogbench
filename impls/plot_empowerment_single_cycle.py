"""
Plot empowerment-vs-phase over ONE real (offline dataset) pick-and-place cycle,
with an equal-width "warped" axis per phase instead of raw timesteps, so short
phases (e.g. near+closing) are just as visible as long ones (e.g. approach).

Phases (derived from decoded state, not privileged info):
  approach     -- distance shrinking, gripper relaxed, cube untouched
  near+closing -- gripper arrives near cube, closedness rising
  lift         -- cube rising, gripper closed, xy ~fixed at the pick-up spot
  arc-up       -- cube still rising but xy has started moving away from pick-up spot
  arc-down     -- cube falling and xy still moving (hasn't reached place spot yet)
  descent      -- cube falling, xy ~fixed at the place spot
  release      -- closedness falling back to open

"xy ~fixed" is measured against the xy position at the moment lift starts
(for `lift`) or at the moment release starts (for `descent`), each within
`--xy_move_thresh` meters. Either of `lift`/`descent` can end up empty (zero
width) if the real trajectory never holds xy still at that end of the arc --
that's a legitimate finding, not a bug.
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
    term_idx = np.nonzero(terminals > 0)[0]
    if len(term_idx) == 0:
        raise RuntimeError("Offline dataset has no terminals; cannot slice into trajectories.")
    split_points = np.where(np.diff(term_idx) > 1)[0] + 1
    term_groups = np.split(term_idx, split_points)
    end_locs = np.array([int(g[-1]) for g in term_groups])
    start_locs = np.concatenate([[0], end_locs[:-1] + 1])
    return start_locs, end_locs


def _extract_cube_features(obs, n_arm):
    xyz_center = np.array([0.425, 0.0, 0.0])
    xyz_scaler = 10.0
    gripper_scaler = 3.0
    effector_pos = obs[2 * n_arm:2 * n_arm + 3] / xyz_scaler + xyz_center
    gripper_closedness = float(obs[2 * n_arm + 5]) / gripper_scaler
    cube_start = 2 * n_arm + 7
    cube_pos = obs[cube_start:cube_start + 3] / xyz_scaler + xyz_center
    return effector_pos, gripper_closedness, cube_pos


def compute_empowerment_timeseries(agent, observations, batch_size=256):
    """Compute empowerment at every timestep, padding every chunk (by repeating
    its last row) up to exactly `batch_size`.

    A single grasp cycle is usually much shorter than 256 steps, so a naive
    chunk of the cycle's own (novel, never-before-compiled) length can hit an
    idiosyncratic slow/pathological XLA compile even with the streaming
    skills/samples fix in agent.empowerment() -- some shapes just get an
    unlucky autotune. Padding to a fixed, already-exercised batch_size (256,
    matching plot_empowerment_trajectory_phases.py's default) reuses that
    warm shape instead of risking a fresh cold compile.
    """
    obs_array = np.stack(observations, axis=0).astype(np.float32)
    n = obs_array.shape[0]
    emp_values = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        actual = end - start
        obs_chunk = obs_array[start:end]
        if actual < batch_size:
            pad = np.repeat(obs_chunk[-1:], batch_size - actual, axis=0)
            obs_chunk = np.concatenate([obs_chunk, pad], axis=0)
        obs_batch = jnp.asarray(obs_chunk)
        rng = jax.random.PRNGKey(start)
        keys = jax.random.split(rng, batch_size)
        emp = jax.vmap(
            lambda ob, key: agent.empowerment(ob[None, ...], rng=key).squeeze(),
            in_axes=(0, 0),
        )(obs_batch, keys)
        emp_values[start:end] = np.asarray(emp)[:actual]
    return emp_values


PHASE_NAMES = ["approach", "near", "closing", "lift", "arc-up", "arc-down", "descent", "release"]
PHASE_COLORS = {
    "approach": "#f0f0f0",
    "near": "#ffe599",
    "closing": "#f6b26b",
    "lift": "#93c47d",
    "arc-up": "#76a5af",
    "arc-down": "#6fa8dc",
    "descent": "#8e7cc3",
    "release": "#e69138",
}


def find_first_cycle(distances, closedness, heights, cube_xy, table_z,
                      dist_thresh=0.03, touch_thresh=0.01, closed_thresh=0.5, open_thresh=0.15,
                      xy_move_thresh=0.01):
    """Locate one approach->near->closing->lift->arc->descent->release cycle
    and its phase boundaries.

    `near` (dist < dist_thresh, not yet in contact) and `closing` (dist <
    touch_thresh, i.e. fingers already touching the cube, closedness rising
    toward closed_thresh) split what would otherwise be one "near+closing"
    span -- empirically empowerment dips exactly during the touching-but-not-
    yet-firmly-gripped window, so it's useful to isolate it as its own phase.

    Within the grasped span [grasp_idx, release_start), the pick-up xy anchor
    is cube_xy[grasp_idx] and the place-down xy anchor is cube_xy[release_start].
    `lift` is the longest prefix that stays within `xy_move_thresh` of the
    pick-up anchor; `descent` is the longest suffix that stays within
    `xy_move_thresh` of the place-down anchor. The height peak (argmax) splits
    whatever remains in between into `arc-up` (before the peak) and `arc-down`
    (after it). Either `lift` or `descent` can come out empty if the real
    trajectory never holds xy still at that end.

    Returns (start, end, segments) or None if no clean cycle could be found.
    """
    n = len(distances)
    grasp_idx = next((t for t in range(n) if closedness[t] >= closed_thresh), None)
    if grasp_idx is None:
        return None

    near_candidates = [t for t in range(grasp_idx + 1) if distances[t] < dist_thresh]
    near_idx = near_candidates[0] if near_candidates else max(0, grasp_idx - 1)

    touch_candidates = [t for t in range(near_idx, grasp_idx + 1) if distances[t] < touch_thresh]
    touch_idx = touch_candidates[0] if touch_candidates else grasp_idx

    release_start = next((t for t in range(grasp_idx + 1, n) if closedness[t] < closed_thresh), n - 1)
    if release_start <= grasp_idx:
        return None
    release_end = next((t for t in range(release_start, n) if closedness[t] < open_thresh), n - 1)

    # lift: longest prefix of [grasp_idx, release_start) within xy_move_thresh of the pick-up spot.
    pickup_xy = cube_xy[grasp_idx]
    lift_end = grasp_idx
    for t in range(grasp_idx, release_start):
        if np.linalg.norm(cube_xy[t] - pickup_xy) <= xy_move_thresh:
            lift_end = t + 1
        else:
            break

    # descent: longest suffix of [grasp_idx, release_start) within xy_move_thresh of the place-down spot.
    place_xy = cube_xy[release_start]
    descent_start = release_start
    for t in range(release_start - 1, lift_end - 1, -1):
        if np.linalg.norm(cube_xy[t] - place_xy) <= xy_move_thresh:
            descent_start = t
        else:
            break

    if descent_start < lift_end:
        descent_start = lift_end  # degenerate overlap guard

    # height peak splits the remaining arc into arc-up / arc-down.
    if descent_start > lift_end:
        peak_idx = lift_end + int(np.argmax(heights[lift_end:descent_start]))
        peak_idx = min(max(peak_idx, lift_end), descent_start)
    else:
        peak_idx = lift_end

    start = 0
    end = release_end
    segments = {
        "approach": (start, near_idx),
        "near": (near_idx, touch_idx),
        "closing": (touch_idx, grasp_idx),
        "lift": (grasp_idx, lift_end),
        "arc-up": (lift_end, peak_idx),
        "arc-down": (peak_idx, descent_start),
        "descent": (descent_start, release_start),
        "release": (release_start, end + 1),
    }
    return start, end, segments


def warp_axis(n, segments):
    """Map each raw timestep in [0, n) to an x-coordinate where every named
    phase occupies exactly one unit of width, regardless of its step count."""
    warped = np.zeros(n)
    for i, name in enumerate(PHASE_NAMES):
        lo, hi = segments[name]
        lo, hi = max(lo, 0), min(hi, n)
        if hi <= lo:
            continue
        span = hi - lo
        for t in range(lo, hi):
            frac = (t - lo) / span if span > 0 else 0.0
            warped[t] = i + frac
    return warped


def plot_cycle(warped_x, distances, closedness, heights, emp_values, table_z, title, out_img):
    fig, (ax_emp, ax_state) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1]})

    for ax in (ax_emp, ax_state):
        for i, name in enumerate(PHASE_NAMES):
            ax.axvspan(i, i + 1, color=PHASE_COLORS[name], alpha=0.5, zorder=0)
            ax.axvline(i, color="gray", linewidth=0.6, zorder=1)

    ax_emp.plot(warped_x, emp_values, color="#1f4e79", linewidth=2.0, marker="o", markersize=2, zorder=2)
    ax_emp.set_ylabel("Empowerment")
    ax_emp.set_title(title, fontsize=10, pad=28)
    for i, name in enumerate(PHASE_NAMES):
        ax_emp.text(i + 0.5, 1.06, name, transform=ax_emp.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, clip_on=False)

    ax_state.plot(warped_x, distances, color="#cc0000", linewidth=1.5, label="gripper-cube distance (m)")
    ax_state.plot(warped_x, heights - table_z, color="#38761d", linewidth=1.5, label="cube height above table (m)")
    ax_state.plot(warped_x, closedness, color="#674ea7", linewidth=1.2, linestyle="--",
                  label="gripper closedness [0=open,1=closed]")
    ax_state.set_xlabel("Phase (equal-width axis; NOT proportional to real time)")
    ax_state.set_ylabel("State")
    ax_state.legend(loc="upper right", fontsize=7)
    ax_state.set_xlim(0, len(PHASE_NAMES))
    ax_state.set_xticks(range(len(PHASE_NAMES) + 1))
    ax_state.set_xticklabels([])

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_img, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Empowerment over ONE real pick-and-place cycle, equal-width phase axis."
    )
    parser.add_argument("--ckpt_root", type=str, default="ckpts/cube1")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--num_splus_samples", type=int, default=384)
    parser.add_argument("--emp_sample_chunk_size", type=int, default=64)
    parser.add_argument("--emp_batch_size", type=int, default=256)
    parser.add_argument("--dist_thresh", type=float, default=0.03)
    parser.add_argument("--touch_thresh", type=float, default=0.01,
                         help="Distance (m) below which the fingers are considered already "
                              "touching the cube, splitting 'near' from 'closing'.")
    parser.add_argument("--closed_thresh", type=float, default=0.5)
    parser.add_argument("--open_thresh", type=float, default=0.15)
    parser.add_argument("--xy_move_thresh", type=float, default=0.01,
                         help="Meters of cube-xy drift from the pick-up/place-down spot "
                              "still counted as 'xy fixed' for the lift/descent phases.")
    parser.add_argument("--table_z", type=float, default=0.02)
    parser.add_argument("--max_search", type=int, default=500,
                         help="Max episodes to try before giving up on finding a clean cycle.")
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

    found = None
    tried = set()
    attempts = 0
    while found is None and attempts < args.max_search:
        idx = int(rng.integers(len(end_locs)))
        attempts += 1
        if idx in tried:
            continue
        tried.add(idx)
        ep_start, ep_end = int(start_locs[idx]), int(end_locs[idx])
        obs_arr = np.asarray(train_dataset["observations"][ep_start:ep_end + 1])

        features = [_extract_cube_features(o, n_arm) for o in obs_arr]
        distances = np.array([np.linalg.norm(eff - cube) for eff, _, cube in features])
        closedness = np.array([c for _, c, _ in features])
        heights = np.array([cube[2] for _, _, cube in features])
        cube_xy = np.array([cube[:2] for _, _, cube in features])

        result = find_first_cycle(distances, closedness, heights, cube_xy, args.table_z,
                                   dist_thresh=args.dist_thresh, touch_thresh=args.touch_thresh,
                                   closed_thresh=args.closed_thresh, open_thresh=args.open_thresh,
                                   xy_move_thresh=args.xy_move_thresh)
        if result is None:
            continue
        start, end, segments = result
        # Require a real approach phase (not degenerate) for a clean/informative example.
        if segments["approach"][1] - segments["approach"][0] < 5:
            continue
        found = (idx, ep_start, obs_arr, distances, closedness, heights, start, end, segments)

    if found is None:
        raise RuntimeError(f"Could not find a clean single cycle after {attempts} episodes.")

    idx, ep_start, obs_arr, distances, closedness, heights, start, end, segments = found
    print(f"Episode idx={idx}, cycle spans dataset steps {ep_start + start}:{ep_start + end}")
    for name in PHASE_NAMES:
        lo, hi = segments[name]
        print(f"  {name:16s} steps [{lo}, {hi})  (len={max(0, hi - lo)})")

    obs_cycle = obs_arr[start:end + 1]
    distances_c = distances[start:end + 1]
    closedness_c = closedness[start:end + 1]
    heights_c = heights[start:end + 1]
    segments_c = {name: (lo - start, hi - start) for name, (lo, hi) in segments.items()}

    emp_values = compute_empowerment_timeseries(agent, list(obs_cycle), batch_size=args.emp_batch_size)
    warped_x = warp_axis(len(obs_cycle), segments_c)

    title = (f"Empowerment over one pick-and-place cycle (offline dataset idx={idx})\n"
             f"run={os.path.basename(run_dir)} | epoch={epoch}")

    out_img = args.output if args.output is not None else os.path.join(
        run_dir, f"empowerment_single_cycle_e{epoch}.png"
    )
    plot_cycle(warped_x, distances_c, closedness_c, heights_c, emp_values, args.table_z, title, out_img)

    out_npz = os.path.splitext(out_img)[0] + ".npz"
    np.savez(out_npz, warped_x=warped_x, distances=distances_c, closedness=closedness_c,
             heights=heights_c, emp_values=emp_values,
             **{f"segment_{k}": v for k, v in segments_c.items()})

    print(f"Saved image: {out_img}")
    print(f"Saved array: {out_npz}")


if __name__ == "__main__":
    main()
