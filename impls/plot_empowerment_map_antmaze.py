import os
# Force EGL (headless GL) before any mujoco import so env.render() works on a server.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import glob
import inspect
import json
import re
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.patches import Rectangle

from agents import agents as agent_registry
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent
from utils.log_utils import reshape_video


# Step intervals at which to drop a color-coded marker along each skill path.
INTERVAL_STEPS = [500, 1000, 1500, 2000, 2500, 3000]


def _raise_time_limit(env, min_steps):
    """Lift the env's TimeLimit so rollouts can run the full requested horizon.

    These envs wrap a gymnasium TimeLimit with max_episode_steps=1000, so a
    rollout would truncate at step 1000 no matter how many steps we ask for.
    Walk the wrapper chain and bump every _max_episode_steps (and the spec) to
    at least min_steps + 1 so `truncated` only fires at our horizon.
    """
    target = int(min_steps) + 1
    e = env
    while e is not None:
        if getattr(e, "_max_episode_steps", None) is not None and e._max_episode_steps < target:
            e._max_episode_steps = target
        e = getattr(e, "env", None)
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None) is not None \
            and spec.max_episode_steps < target:
        spec.max_episode_steps = target


def _draw_interval_dots(ax, xy_per_skill, intervals=INTERVAL_STEPS):
    """Drop a small color-coded dot on every skill path at each step interval.

    xy_traj index t == env step t (index 0 is the start state), so the dot for
    interval `step` lives at xy[step] when the trajectory ran that long. All
    dots for a given interval share one color (from tab10) so it's clear which
    marker corresponds to which step across skills. Returns legend handles.
    """
    cmap = plt.get_cmap('tab10')
    handles = []
    for j, step in enumerate(intervals):
        color = cmap(j % 10)
        xs, ys = [], []
        for xy in xy_per_skill:
            if len(xy) > step:
                xs.append(float(xy[step, 0]))
                ys.append(float(xy[step, 1]))
        if xs:
            ax.scatter(xs, ys, c=[color], s=14, marker='o', edgecolors='black',
                       linewidths=0.3, zorder=7)
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='', color=color,
                                   markeredgecolor='black', markersize=5,
                                   label=f"step {step}"))
    return handles


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


def rollout_skill_ant(env, agent, num_skills, skill_id, ant_xy,
                      n_steps, frame_skip, temperature=0.0, seed=0,
                      collect_frames=True, collect_xy=False):
    """Roll out the empowerment-maximizing policy for a single skill in AntMaze.

    Resets the env, overrides ant XY (via set_xy — no ball in AntMaze), then
    runs deterministic policy(.|s, skill_onehot) for n_steps. Returns
    (frames, xy_traj), where each is None if not requested.
    """
    K = num_skills
    skill_onehot = jnp.eye(K, dtype=jnp.float32)[skill_id][None, :]

    @jax.jit
    def policy_action(obs, key):
        obs_b = obs[None, ...]
        dist = agent.network.select('policy')(
            obs_b, skill_onehot, temperature=temperature
        )
        if temperature == 0.0:
            action = dist.mode()
        else:
            action = dist.sample(seed=key)
        return jnp.clip(action[0], -1.0, 1.0)

    base_env = env.unwrapped

    env.reset()
    base_env.set_xy(np.asarray(ant_xy, dtype=np.float64))
    obs = np.asarray(base_env.get_ob(), dtype=np.float32)

    frames = [env.render().copy()] if collect_frames else None
    xy_traj = [np.asarray(base_env.get_xy(), dtype=np.float32)] if collect_xy else None
    rng = jax.random.PRNGKey(int(seed))

    for step in range(1, n_steps + 1):
        rng, key = jax.random.split(rng)
        action = np.asarray(policy_action(jnp.asarray(obs), key))
        obs, _, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        if collect_frames and (step % frame_skip == 0 or step == n_steps):
            frames.append(env.render().copy())
        if collect_xy:
            xy_traj.append(np.asarray(base_env.get_xy(), dtype=np.float32))
        if terminated or truncated:
            break

    frames_out = np.asarray(frames, dtype=np.uint8) if collect_frames else None
    xy_out = np.stack(xy_traj, axis=0) if collect_xy else None
    return frames_out, xy_out


def compose_skill_grid(renders_per_skill, n_cols=None):
    """Pad + border per-skill rollouts and tile into a near-square grid.

    Mirrors utils.log_utils.get_wandb_video padding/border behavior, then
    uses utils.log_utils.reshape_video for the tiling.
    """
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(len(renders_per_skill))))

    max_length = max(len(r) for r in renders_per_skill)
    padded = []
    for render in renders_per_skill:
        assert render.dtype == np.uint8
        if len(render) < max_length:
            final = render[-1]
            dim = np.array(
                ImageEnhance.Brightness(Image.fromarray(final)).enhance(0.5)
            )
            pad = np.repeat(dim[None, ...], max_length - len(render), axis=0)
            render = np.concatenate([render, pad], axis=0)
        render = np.pad(
            render, ((0, 0), (1, 1), (1, 1), (0, 0)),
            mode='constant', constant_values=0,
        )
        padded.append(render)

    arr = np.array(padded)  # [n, t, h, w, c]
    tiled = reshape_video(arr, n_cols)  # [t, c, H, W]
    tiled = np.transpose(tiled, (0, 2, 3, 1))  # [t, H, W, c]
    return tiled.astype(np.uint8)


def plot_skill_paths_ant(xy_per_skill, ant_start_xy, overlay_maze, extent,
                         output_path, title=None):
    """One 2D plot: a thin line per skill (ant xy over time), black dot at start.

    AntMaze has no ball, so this is a pared-down version of the antsoccer
    variant.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    overlay_maze(ax)

    K = len(xy_per_skill)
    cmap = plt.get_cmap('hsv')
    for z, xy in enumerate(xy_per_skill):
        color = cmap(z / max(K, 1))
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.6, alpha=0.9,
                label=f"skill {z}")

    ax.scatter([ant_start_xy[0]], [ant_start_xy[1]], c='black', s=40,
               marker='o', edgecolors='white', linewidths=0.8, zorder=5,
               label='Ant start')

    interval_handles = _draw_interval_dots(ax, xy_per_skill)

    x_lo, x_hi, y_lo, y_hi = extent
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    ax.set_xlabel('Ant x')
    ax.set_ylabel('Ant y')
    if title is not None:
        ax.set_title(title)
    if K <= 15:
        skill_leg = ax.legend(loc='upper right', fontsize=7, framealpha=0.85)
        ax.add_artist(skill_leg)
    ax.legend(handles=interval_handles, loc='lower left', fontsize=6,
              framealpha=0.85, title='interval')

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def save_mp4(frames, output_path, fps=15):
    """Save frames as an mp4 via imageio-ffmpeg (h264 / yuv420p)."""
    import imageio
    h, w = frames[0].shape[:2]
    h -= h % 2
    w -= w % 2
    writer = imageio.get_writer(
        output_path, format='FFMPEG', mode='I',
        fps=fps, codec='libx264',
        output_params=['-pix_fmt', 'yuv420p'],
    )
    for f in frames:
        writer.append_data(f[:h, :w])
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Plot AntMaze empowerment map (same computation as Ant Soccer).")
    parser.add_argument("--ckpt_root", type=str, default="ckpts", help="Root checkpoint directory.")
    parser.add_argument("--run_dir", type=str, default=None, help="Explicit run dir (overrides latest in ckpt_root).")
    parser.add_argument("--epoch", type=int, default=None, help="Explicit epoch (overrides latest params_*.pkl).")
    parser.add_argument("--grid_res", type=int, default=200, help="Grid resolution for ant XY map.")
    parser.add_argument("--x_min", type=float, default=0.0, help="Grid min x for ant position.")
    parser.add_argument("--x_max", type=float, default=20.0, help="Grid max x for ant position.")
    parser.add_argument("--y_min", type=float, default=0.0, help="Grid min y for ant position.")
    parser.add_argument("--y_max", type=float, default=20.0, help="Grid max y for ant position.")
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
    # ── Skill-rollout flags (mirror antsoccer) ──────────────────────────────
    parser.add_argument(
        "--skill_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a near-square grid (one video per skill) of the empowerment-"
             "maximizing policy rolled out from a single ant start. "
             "On by default; pass --no-skill_video to disable.",
    )
    parser.add_argument(
        "--skill_paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a 2D plot with a thin ant-path line per skill. "
             "On by default; pass --no-skill_paths to disable.",
    )
    parser.add_argument(
        "--skill_map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the empowerment map. On by default; pass --no-skill_map "
             "to skip the (slow) map computation.",
    )
    parser.add_argument("--video_steps", type=int, default=3000)
    parser.add_argument("--video_frame_skip", type=int, default=3)
    parser.add_argument("--video_fps", type=int, default=15)
    parser.add_argument("--video_ant_xy", type=str, default=None,
                        help="Fixed ant x,y for the skill rollouts (e.g. '8,8'). "
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
    # Increase MC fidelity for plotting. Different agents key their per-state
    # sample count differently (e.g. empowerment_dads uses est_num_joints, not
    # num_splus_samples) -- set whichever key this agent's saved config has.
    agent_cfg["num_splus_samples"] = int(args.num_splus_samples)
    if "est_num_joints" in agent_cfg:
        agent_cfg["est_num_joints"] = int(args.num_splus_samples)
    env_name = flags["env_name"]

    # Build AntMaze env/dataset and agent
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

    # Fresh np.random global so env.reset() picks a different task / start each
    # invocation (env.reset uses np.random internally).
    np.random.seed(None)

    obs0, _ = env.reset()
    obs0 = np.asarray(obs0, dtype=np.float32)
    base_env = env.unwrapped

    x_low, x_high = float(args.x_min), float(args.x_max)
    y_low, y_high = float(args.y_min), float(args.y_max)
    unit = getattr(base_env, "_maze_unit", 4.0)
    half = float(unit) / 2.0
    x_low_plot, x_high_plot = x_low - half, x_high + half
    y_low_plot, y_high_plot = y_low - half, y_high + half

    # Maze overlay (hoisted so the path plot can reuse it).
    maze_map = getattr(base_env, "maze_map", None)
    offx = getattr(base_env, "_offset_x", 4.0)
    offy = getattr(base_env, "_offset_y", 4.0)

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

    def is_valid_xy(x: float, y: float) -> bool:
        if maze_map is None:
            return True
        j = int(round((x + offx) / unit))
        i = int(round((y + offy) / unit))
        rows, cols = maze_map.shape
        if i < 0 or i >= rows or j < 0 or j >= cols:
            return False
        return int(maze_map[i, j]) != 1

    def sample_valid_xy() -> np.ndarray:
        for _ in range(10000):
            x = float(np.random.uniform(x_low, x_high))
            y = float(np.random.uniform(y_low, y_high))
            if is_valid_xy(x, y):
                return np.array([x, y], dtype=np.float64)
        raise RuntimeError("Could not sample a valid (non-wall) ant position.")

    # ── Skill-rollout branch (video grid + ant-path plot) ──────────────────
    if args.skill_video or args.skill_paths:
        num_skills = int(agent_cfg.get('num_skills'))

        # Lift the 1000-step TimeLimit so paths run the full requested horizon.
        _raise_time_limit(env, args.video_steps)

        if args.video_ant_xy is not None:
            ax_, ay_ = _parse_xy(args.video_ant_xy)
            ant_xy = np.array([ax_, ay_], dtype=np.float64)
        else:
            ant_xy = sample_valid_xy()

        print(
            f"Skill rollouts: K={num_skills}, steps={args.video_steps}, "
            f"frame_skip={args.video_frame_skip}, ant={ant_xy.tolist()}, "
            f"video={args.skill_video}, paths={args.skill_paths}"
        )

        renders_per_skill = []
        xy_per_skill = []
        for z in range(num_skills):
            print(f"  rolling out skill {z + 1}/{num_skills}...")
            frames, xy_traj = rollout_skill_ant(
                env=env,
                agent=agent,
                num_skills=num_skills,
                skill_id=z,
                ant_xy=ant_xy,
                n_steps=args.video_steps,
                frame_skip=args.video_frame_skip,
                temperature=0.0,
                seed=z,
                collect_frames=args.skill_video,
                collect_xy=args.skill_paths,
            )
            if args.skill_video:
                renders_per_skill.append(frames)
            if args.skill_paths:
                xy_per_skill.append(xy_traj)

        if args.skill_video:
            grid = compose_skill_grid(renders_per_skill)
            video_out = os.path.join(run_dir, f"skill_grid_video_e{epoch}.mp4")
            save_mp4(grid, video_out, fps=args.video_fps)
            print(f"Saved skill-grid video: {video_out}")

        if args.skill_paths:
            paths_out = os.path.join(run_dir, f"skill_ant_paths_e{epoch}.png")
            plot_skill_paths_ant(
                xy_per_skill=xy_per_skill,
                ant_start_xy=ant_xy,
                overlay_maze=overlay_maze,
                extent=(x_low_plot, x_high_plot, y_low_plot, y_high_plot),
                output_path=paths_out,
                title=(
                    f"Ant paths | run={os.path.basename(run_dir.rstrip("/"))} | epoch={epoch}\n"
                    f"K={num_skills}, steps={args.video_steps}, "
                    f"start=({ant_xy[0]:.2f}, {ant_xy[1]:.2f})"
                ),
            )
            print(f"Saved skill ant-path plot: {paths_out}")

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
    # Some agents (e.g. empowerment_crl) expose a deterministic E(s) with no rng arg.
    _emp_takes_rng = "rng" in inspect.signature(agent.empowerment).parameters

    @jax.jit
    def _emp_batch(obs_b, keys_b):
        if _emp_takes_rng:
            fn = lambda ob, key: agent.empowerment(ob[None, ...], rng=key).squeeze()
        else:
            fn = lambda ob, key: agent.empowerment(ob[None, ...]).squeeze()
        return jax.vmap(fn, in_axes=(0, 0))(obs_b, keys_b)

    batch_size = max(1, int(args.batch_size))
    emp_chunks = []
    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        emp_chunks.append(np.asarray(_emp_batch(obs_batch_jnp[start:end], point_keys[start:end])))
        print(f"  empowerment batch {start}:{end} / {num_points}")
    emp_vals = np.concatenate(emp_chunks, axis=0)
    emp_map = emp_vals.reshape(args.grid_res, args.grid_res)

    out_img = args.output if args.output is not None else os.path.join(run_dir, f"empowerment_antmaze_e{epoch}.png")
    out_npy = os.path.splitext(out_img)[0] + ".npy"

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(
        emp_map,
        origin="lower",
        extent=[x_low_plot, x_high_plot, y_low_plot, y_high_plot],
        aspect="auto",
        cmap="viridis",
    )
    overlay_maze(ax)
    ax.set_xlabel("Ant x")
    ax.set_ylabel("Ant y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"AntMaze empowerment | run={os.path.basename(run_dir.rstrip("/"))} | epoch={epoch}")
    plt.tight_layout()
    plt.savefig(out_img, dpi=180)
    np.save(out_npy, emp_map)
    print(f"Saved image: {out_img}")
    print(f"Saved array: {out_npy} (shape {emp_map.shape})")


if __name__ == "__main__":
    main()

