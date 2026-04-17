"""
Create a video of an OGBench trajectory with a synchronized empowerment-vs-time plot.

Top half: environment render at each timestep.
Bottom half: empowerment curve up to the current timestep (with a moving cursor).

Usage:
    python plot_empowerment_trajectory_video.py --ckpt_root ckpts/cube1 \
        --num_episodes 1 --fps 15 --output trajectory_emp.mp4
"""

# Must set EGL backend before any mujoco import (headless rendering).
import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import glob
import json
import os
import re

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from agents import agents as agent_registry
from utils.env_utils import make_env_and_datasets
from utils.evaluation import supply_rng
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
    return max(epochs)


def rollout_trajectory(agent, env, config, max_steps=None, eval_temperature=0):
    """Roll out one episode, collecting frames, observations, and actions."""
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    observation, info = env.reset()
    goal = info.get('goal')
    done = False
    step = 0

    frames = []
    observations = []
    actions_list = []

    while not done:
        if max_steps is not None and step >= max_steps:
            break

        frame = env.render().copy()
        frames.append(frame)
        observations.append(observation.copy())

        action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
        action = np.array(action)
        if not config.get('discrete'):
            action = np.clip(action, -1, 1)
        actions_list.append(action.copy())

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    # Capture final frame
    frames.append(env.render().copy())
    observations.append(observation.copy())

    return frames, observations, actions_list


def compute_empowerment_timeseries(agent, observations, batch_size=64):
    """Compute empowerment for each observation in a trajectory."""
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


def fig_to_array(fig):
    """Convert a matplotlib figure to a numpy RGB array."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    return arr


def compose_video_frames(env_frames, emp_values, fps):
    """Create composite frames: env render on top, empowerment plot on bottom."""
    n_frames = len(env_frames)
    timesteps = np.arange(n_frames)

    # Get env frame dimensions to match height for side-by-side layout
    env_h, env_w = env_frames[0].shape[:2]

    # Fixed y-axis: 1 to 2
    y_lo = 1.3
    y_hi = 1.5

    # Figure size: match env height, reasonable width for a tall plot
    fig_h_inches = env_h / 100.0
    fig_w_inches = fig_h_inches * 0.6
    plot_w = int(fig_w_inches * 100)

    composed = []
    for t in range(n_frames):
        # Create the empowerment plot for this timestep
        fig, ax = plt.subplots(figsize=(fig_w_inches, fig_h_inches), dpi=100)

        # Plot the full trajectory in light gray as context
        ax.plot(timesteps, emp_values, color='lightgray', linewidth=1.0, zorder=1)

        # Plot up to current timestep in color
        if t > 0:
            ax.plot(timesteps[:t+1], emp_values[:t+1], color='#2196F3', linewidth=2.0, zorder=2)

        # Current point marker
        ax.scatter([t], [emp_values[t]], color='#F44336', s=40, zorder=3, edgecolors='white', linewidths=0.5)

        ax.set_xlim(-0.5, n_frames - 0.5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel('Timestep', fontsize=9)
        ax.set_ylabel('Empowerment', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_title(f't={t}  emp={emp_values[t]:.3f}', fontsize=10, pad=3)
        fig.tight_layout(pad=0.5)

        plot_img = fig_to_array(fig)
        plt.close(fig)

        # Resize plot to match env height
        from PIL import Image
        plot_pil = Image.fromarray(plot_img)
        plot_pil = plot_pil.resize((plot_w, env_h), Image.LANCZOS)
        plot_img = np.array(plot_pil)

        # Stack horizontally: env on left, plot on right
        composite = np.concatenate([env_frames[t], plot_img], axis=1)
        composed.append(composite)

    return composed


def save_mp4(frames, output_path, fps=15):
    """Save frames as an MP4 video using imageio-ffmpeg."""
    import imageio
    # Ensure even dimensions for h264
    h, w = frames[0].shape[:2]
    h = h - (h % 2)
    w = w - (w % 2)
    writer = imageio.get_writer(
        output_path, format='FFMPEG', mode='I',
        fps=fps, codec='libx264',
        output_params=['-pix_fmt', 'yuv420p'],
    )
    for frame in frames:
        writer.append_data(frame[:h, :w])
    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create a trajectory video with empowerment-vs-time overlay."
    )
    parser.add_argument("--ckpt_root", type=str, default="ckpts/cube1")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of trajectory videos to produce.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (None = run to termination).")
    parser.add_argument("--eval_temperature", type=float, default=0,
                        help="Action sampling temperature.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Only keep every N-th frame (1 = keep all).")
    parser.add_argument("--num_splus_samples", type=int, default=192,
                        help="Monte-Carlo samples for empowerment estimation.")
    parser.add_argument("--emp_batch_size", type=int, default=64,
                        help="Batch size for empowerment computation.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto in run_dir).")
    args = parser.parse_args()

    # Resolve checkpoint
    run_dir = args.run_dir if args.run_dir is not None else _latest_run_dir(args.ckpt_root)
    epoch = args.epoch if args.epoch is not None else _latest_epoch(run_dir)
    print(f"Using run_dir={run_dir}, epoch={epoch}")

    # Load flags and agent
    flags_path = os.path.join(run_dir, "flags.json")
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
    print("Agent restored.")

    for ep_idx in range(args.num_episodes):
        print(f"\n--- Episode {ep_idx + 1}/{args.num_episodes} ---")

        # Roll out trajectory
        print("Rolling out trajectory...")
        frames, observations, actions = rollout_trajectory(
            agent, env, agent_cfg,
            max_steps=args.max_steps,
            eval_temperature=args.eval_temperature,
        )
        print(f"  {len(frames)} frames, {len(observations)} observations")

        # Apply frame skip
        if args.frame_skip > 1:
            indices = list(range(0, len(frames), args.frame_skip))
            if indices[-1] != len(frames) - 1:
                indices.append(len(frames) - 1)
            frames = [frames[i] for i in indices]
            observations = [observations[i] for i in indices]

        # Compute empowerment
        print("Computing empowerment at each timestep...")
        emp_values = compute_empowerment_timeseries(
            agent, observations, batch_size=args.emp_batch_size
        )
        print(f"  Empowerment range: [{emp_values.min():.4f}, {emp_values.max():.4f}]")

        # Compose video frames
        print("Composing video frames...")
        composed = compose_video_frames(frames, emp_values, args.fps)

        # Save
        if args.output is not None:
            if args.num_episodes == 1:
                out_path = args.output
            else:
                base, ext = os.path.splitext(args.output)
                out_path = f"{base}_ep{ep_idx}{ext}"
        else:
            out_path = os.path.join(run_dir, f"trajectory_empowerment_ep{ep_idx}_e{epoch}.mp4")

        print(f"Saving video to {out_path}...")
        save_mp4(composed, out_path, fps=args.fps)
        print(f"Done: {out_path}")

        # Also save empowerment timeseries as .npy
        npy_path = os.path.splitext(out_path)[0] + "_emp.npy"
        np.save(npy_path, emp_values)
        print(f"Saved empowerment data: {npy_path}")


if __name__ == "__main__":
    main()
