import os
# Force EGL (headless GL) before any mujoco import so env.render() works on a server.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import glob
import json
import re

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance

from agents import agents as agent_registry
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent
from utils.log_utils import reshape_video


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


def rollout_skill(env, agent, num_skills, skill_id, ant_xy, ball_xy,
                  n_steps, frame_skip, temperature=0.0, seed=0,
                  goal_xy=None, collect_frames=True, collect_xy=False):
    """Roll out the empowerment-maximizing policy for a single fixed skill.

    Resets the env, overrides ant + ball XY (and optionally the goal) so the
    starting state is identical across skills, then runs deterministic
    policy(.|s, skill_onehot) for n_steps. Returns (frames, ant_xy_traj),
    where each is None if not requested. Frames are captured every
    frame_skip steps to match OGBench's eval video cadence; the xy
    trajectory captures every step.
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
    if goal_xy is not None:
        base_env.set_goal(goal_xy=np.asarray(goal_xy, dtype=np.float64))
    base_env.set_agent_ball_xy(
        np.asarray(ant_xy, dtype=np.float64),
        np.asarray(ball_xy, dtype=np.float64),
    )
    obs = np.asarray(base_env.get_ob(), dtype=np.float32)

    frames = [env.render().copy()] if collect_frames else None
    xy_traj = [np.asarray(base_env.get_agent_ball_xy()[0], dtype=np.float32)] \
        if collect_xy else None
    rng = jax.random.PRNGKey(int(seed))

    for step in range(1, n_steps + 1):
        rng, key = jax.random.split(rng)
        action = np.asarray(policy_action(jnp.asarray(obs), key))
        obs, _, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        if collect_frames and (step % frame_skip == 0 or step == n_steps):
            frames.append(env.render().copy())
        if collect_xy:
            xy_traj.append(np.asarray(base_env.get_agent_ball_xy()[0], dtype=np.float32))
        if terminated or truncated:
            break

    frames_out = np.asarray(frames, dtype=np.uint8) if collect_frames else None
    xy_out = np.stack(xy_traj, axis=0) if collect_xy else None
    return frames_out, xy_out


def compose_skill_grid(renders_per_skill, n_cols=None):
    """Pad + border per-skill rollouts and tile into a near-square grid.

    Mirrors utils.log_utils.get_wandb_video padding/border behavior, then
    uses utils.log_utils.reshape_video for the actual tiling, so the layout
    matches what OGBench produces during training.
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


def plot_skill_paths(xy_per_skill, ball_xy, ant_start_xy, overlay_maze, extent,
                     output_path, title=None):
    """One 2D plot: a thin line per skill (ant xy over time), plus the ball.

    Args:
        xy_per_skill: list of np.ndarray[T_i, 2] giving the ant (x, y) per step.
        ball_xy: (x, y) of the fixed ball.
        ant_start_xy: (x, y) of the fixed starting ant position.
        overlay_maze: callable that draws maze walls onto an Axes.
        extent: (x_lo, x_hi, y_lo, y_hi) plot bounds.
        output_path: where to write the PNG.
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
    ax.scatter([ball_xy[0]], [ball_xy[1]], c='red', s=70, marker='o',
               edgecolors='white', linewidths=1.0, zorder=6, label='Ball')

    x_lo, x_hi, y_lo, y_hi = extent
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    ax.set_xlabel('Ant x')
    ax.set_ylabel('Ant y')
    if title is not None:
        ax.set_title(title)

    # Legend only sensible for small K; collapse otherwise.
    if K <= 15:
        ax.legend(loc='upper right', fontsize=7, framealpha=0.85)

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
    parser = argparse.ArgumentParser(description="Plot empowerment map from latest checkpoint.")
    parser.add_argument("--ckpt_root", type=str, default="ckpts", help="Root checkpoint directory.")
    parser.add_argument("--run_dir", type=str, default=None, help="Explicit run dir (overrides latest in ckpt_root).")
    parser.add_argument("--epoch", type=int, default=None, help="Explicit epoch (overrides latest params_*.pkl).")
    parser.add_argument("--grid_res", type=int, default=200, help="Grid resolution for ant XY map.")
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
        default=384,
        help="Number of s+ samples used in empowerment Monte Carlo estimate.",
    )
    parser.add_argument("--x_min", type=float, default=0, help="Grid min x for ant position.")
    parser.add_argument("--x_max", type=float, default=20, help="Grid max x for ant position.")
    parser.add_argument("--y_min", type=float, default=0, help="Grid min y for ant position.")
    parser.add_argument("--y_max", type=float, default=20, help="Grid max y for ant position.")
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of grid points to evaluate per empowerment batch (avoids OOM on large grids).",
    )
    # ── Skill-grid video flags ──────────────────────────────────────────────
    parser.add_argument(
        "--skill_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a near-square grid (one video per skill) of the empowerment-"
             "maximizing policy rolled out from a fixed ant+ball start. "
             "On by default; pass --no-skill_video to disable.",
    )
    parser.add_argument(
        "--skill_paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a 2D plot with a thin ant-path line per skill (plus the "
             "fixed ball). Triggers rollouts; shares them with --skill_video. "
             "On by default; pass --no-skill_paths to disable.",
    )
    parser.add_argument(
        "--skill_map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the empowerment map(s). On by default; pass --no-skill_map "
             "to skip the (slow) map computation.",
    )
    parser.add_argument("--video_steps", type=int, default=500,
                        help="Env steps to roll out per skill.")
    parser.add_argument("--video_frame_skip", type=int, default=3,
                        help="Frame-skip cadence for captured frames (OGBench default = 3).")
    parser.add_argument("--video_fps", type=int, default=15,
                        help="Output mp4 fps.")
    parser.add_argument("--video_ant_xy", type=str, default=None,
                        help="Fixed ant x,y for the skill grid (e.g. '8,8'). "
                             "Defaults to the env's reset ant position.")
    parser.add_argument("--video_ball_xy", type=str, default=None,
                        help="Fixed ball x,y for the skill grid. "
                             "Defaults to the env's reset ball position.")
    parser.add_argument("--video_goal_xy", type=str, default=None,
                        help="Optional fixed goal x,y for the skill grid.")
    args = parser.parse_args()

    # Force a fresh np.random global seed on every invocation, so the env's
    # task / start sampling differs run-to-run (env.reset uses np.random
    # internally). --seed only affects the per-grid-point empowerment RNG.
    np.random.seed(None)

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

    # Maze geometry for wall overlay (auto-detected from env: arena, medium, etc.)
    unit = getattr(base_env, "_maze_unit", 4.0)
    half = float(unit) / 2.0
    offx = getattr(base_env, "_offset_x", 4.0)
    offy = getattr(base_env, "_offset_y", 4.0)
    maze_map = getattr(base_env, "maze_map", None)
    x_low_plot, x_high_plot = x_low - half, x_high + half
    y_low_plot, y_high_plot = y_low - half, y_high + half

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
                        alpha=0.6,
                    )
                    ax.add_patch(rect)

    # ── Skill-rollout branch (video grid + ant-path plot) ──────────────────
    # Runs once if either output is requested; produces video and/or the
    # path plot from a shared rollout pass. Does not short-circuit the
    # empowerment map below.
    if args.skill_video or args.skill_paths:
        num_skills = int(agent_cfg.get('num_skills'))

        # Determine the fixed ant+ball+goal from CLI overrides, otherwise
        # read them off the env's reset state (so the grid uses a natural
        # task start when nothing is specified).
        default_ant, default_ball = base_env.get_agent_ball_xy()

        if args.video_ant_xy is not None:
            ax_, ay_ = _parse_indices(args.video_ant_xy)
            ant_xy = np.array([float(ax_), float(ay_)], dtype=np.float64)
        else:
            ant_xy = np.asarray(default_ant, dtype=np.float64)
        if args.video_ball_xy is not None:
            bx_, by_ = _parse_indices(args.video_ball_xy)
            ball_xy = np.array([float(bx_), float(by_)], dtype=np.float64)
        else:
            ball_xy = np.asarray(default_ball, dtype=np.float64)
        if args.video_goal_xy is not None:
            gx_, gy_ = _parse_indices(args.video_goal_xy)
            goal_xy = np.array([float(gx_), float(gy_)], dtype=np.float64)
        else:
            goal_xy = None

        print(
            f"Skill rollouts: K={num_skills}, steps={args.video_steps}, "
            f"frame_skip={args.video_frame_skip}, ant={ant_xy.tolist()}, "
            f"ball={ball_xy.tolist()}, goal={None if goal_xy is None else goal_xy.tolist()}, "
            f"video={args.skill_video}, paths={args.skill_paths}"
        )

        renders_per_skill = []
        xy_per_skill = []
        for z in range(num_skills):
            print(f"  rolling out skill {z + 1}/{num_skills}...")
            frames, xy_traj = rollout_skill(
                env=env,
                agent=agent,
                num_skills=num_skills,
                skill_id=z,
                ant_xy=ant_xy,
                ball_xy=ball_xy,
                n_steps=args.video_steps,
                frame_skip=args.video_frame_skip,
                temperature=0.0,
                seed=z,
                goal_xy=goal_xy,
                collect_frames=args.skill_video,
                collect_xy=args.skill_paths,
            )
            if args.skill_video:
                renders_per_skill.append(frames)
            if args.skill_paths:
                xy_per_skill.append(xy_traj)

        if args.skill_video:
            grid = compose_skill_grid(renders_per_skill)
            video_out = args.output if args.output is not None else os.path.join(
                run_dir, f"skill_grid_video_e{epoch}.mp4"
            )
            save_mp4(grid, video_out, fps=args.video_fps)
            print(f"Saved skill-grid video: {video_out}")

        if args.skill_paths:
            paths_out = os.path.join(run_dir, f"skill_ant_paths_e{epoch}.png")
            plot_skill_paths(
                xy_per_skill=xy_per_skill,
                ball_xy=ball_xy,
                ant_start_xy=ant_xy,
                overlay_maze=overlay_maze,
                extent=(x_low_plot, x_high_plot, y_low_plot, y_high_plot),
                output_path=paths_out,
                title=(
                    f"Ant paths | run={os.path.basename(run_dir)} | epoch={epoch}\n"
                    f"K={num_skills}, steps={args.video_steps}, "
                    f"ball=({ball_xy[0]:.2f}, {ball_xy[1]:.2f})"
                ),
            )
            print(f"Saved skill ant-path plot: {paths_out}")

    def is_valid_xy(x: float, y: float) -> bool:
        if maze_map is None:
            return True
        j = int(round((x + offx) / unit))
        i = int(round((y + offy) / unit))
        rows, cols = maze_map.shape
        if i < 0 or i >= rows or j < 0 or j >= cols:
            return False
        return int(maze_map[i, j]) != 1

    def sample_valid_ball_xy(rng_) -> tuple[float, float]:
        for _ in range(10000):
            x = float(rng_.uniform(x_low, x_high))
            y = float(rng_.uniform(y_low, y_high))
            if is_valid_xy(x, y):
                return x, y
        raise RuntimeError("Could not sample a valid (non-wall) ball position within bounds.")

    xs = np.linspace(x_low, x_high, args.grid_res, dtype=np.float32)
    ys = np.linspace(y_low, y_high, args.grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    @jax.jit
    def _emp_batch(obs_b, keys_b):
        return jax.vmap(
            lambda ob, key: agent.empowerment(ob[None, ...], rng=key).squeeze(),
            in_axes=(0, 0),
        )(obs_b, keys_b)

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
        # Batch over grid points to avoid OOM on large grids / large num_splus_samples.
        batch_size = max(1, int(args.batch_size))
        emp_chunks = []
        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            emp_chunks.append(np.asarray(_emp_batch(obs_batch_jnp[start:end], point_keys[start:end])))
            print(f"  empowerment batch {start}:{end} / {num_points}")
        emp = np.concatenate(emp_chunks, axis=0)
        # Return the empowerment map and the goal used (if any; otherwise NaNs to force explicitness).
        goal_used = goal_xy.astype(np.float32) if 'goal_xy' in locals() else np.array([np.nan, np.nan], dtype=np.float32)
        return emp.reshape(args.grid_res, args.grid_res), goal_used

    if not args.skill_map:
        return

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
            fixed_ball = sample_valid_ball_xy(rng)
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
        ball_positions = [sample_valid_ball_xy(rng) for _ in range(9)]
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
            ball_positions = [sample_valid_ball_xy(rng) for _ in range(9)]
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
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        im = ax.imshow(
            maps[0],
            origin="lower",
            extent=[x_low_plot, x_high_plot, y_low_plot, y_high_plot],
            aspect="auto",
            cmap="viridis",
        )
        overlay_maze(ax)
        ax.scatter(
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
        ax.scatter(
            [goal_used[0]],
            [goal_used[1]],
            c="yellow",
            s=70,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            label="Goal",
        )
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax, label="Empowerment")
        ax.set_xlabel(f"Ant x")
        ax.set_ylabel(f"Ant y")
        ax.set_title(
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
                extent=[x_low_plot, x_high_plot, y_low_plot, y_high_plot],
                aspect="auto",
                cmap="viridis",
            )
            overlay_maze(ax)
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

