"""Per-skill ant-path plot for DDS-trained policies.

DDS agents (agent_name='dds') do NOT expose a skill-conditioned 'policy'
network the way empowerment_skill agents do, so the plot_empowerment_map_*
scripts crash on them (KeyError: 'policy'). Here we roll out each *codebook
skill* directly: fix skill index k, look up its code z_k = codebook[k], and
generate every action with the low-level decoder a ~ D(z_k, s) (DDPM for
continuous envs). The skill is held fixed for the whole episode, matching the
empowerment skill-path plots (skill_ant_paths_e*.png).

Outputs a single PNG: dds_skill_paths_e{epoch}.png in the run dir.

Works for AntMaze (set_xy/get_xy) and Ant Soccer (set_agent_ball_xy /
get_agent_ball_xy) navigate + stitch envs.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # no rendering here, but keep env consistent

import argparse
import glob
import json
import re

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from agents import agents as agent_registry
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent


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


def _latest_epoch(run_dir):
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


def _parse_xy(text):
    parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
    if len(parts) != 2:
        raise ValueError(f"Expected 'x,y', got {text!r}")
    return float(parts[0]), float(parts[1])


def rollout_dds_skill(env, agent, skill_vec, is_antsoccer, ant_xy, ball_xy,
                      n_steps, temperature=0.0, seed=0):
    """Roll out a single fixed DDS codebook skill; return the agent xy trajectory."""
    skills = jnp.asarray(skill_vec, dtype=jnp.float32)[None, :]  # [1, D_z]

    @jax.jit
    def policy_action(obs, key):
        actions = agent._decode(obs[None, ...], skills, key, temperature)  # [1, A]
        return jnp.clip(actions[0], -1.0, 1.0)

    base_env = env.unwrapped
    env.reset()
    if is_antsoccer:
        base_env.set_agent_ball_xy(
            np.asarray(ant_xy, dtype=np.float64),
            np.asarray(ball_xy, dtype=np.float64),
        )
        get_xy = lambda: np.asarray(base_env.get_agent_ball_xy()[0], dtype=np.float32)
    else:
        base_env.set_xy(np.asarray(ant_xy, dtype=np.float64))
        get_xy = lambda: np.asarray(base_env.get_xy(), dtype=np.float32)

    obs = np.asarray(base_env.get_ob(), dtype=np.float32)
    xy_traj = [get_xy()]
    rng = jax.random.PRNGKey(int(seed))
    for _ in range(n_steps):
        rng, key = jax.random.split(rng)
        action = np.asarray(policy_action(jnp.asarray(obs), key))
        obs, _, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        xy_traj.append(get_xy())
        if terminated or truncated:
            break
    return np.stack(xy_traj, axis=0)


def plot_skill_paths(xy_per_skill, ant_start_xy, overlay_maze, extent,
                     output_path, title=None, ball_xy=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    overlay_maze(ax)
    K = len(xy_per_skill)
    cmap = plt.get_cmap('hsv')
    for z, xy in enumerate(xy_per_skill):
        ax.plot(xy[:, 0], xy[:, 1], color=cmap(z / max(K, 1)),
                linewidth=0.6, alpha=0.9, label=f"skill {z}")
    ax.scatter([ant_start_xy[0]], [ant_start_xy[1]], c='black', s=40, marker='o',
               edgecolors='white', linewidths=0.8, zorder=5, label='Ant start')
    if ball_xy is not None:
        ax.scatter([ball_xy[0]], [ball_xy[1]], c='red', s=70, marker='o',
                   edgecolors='white', linewidths=0.8, zorder=5, label='Ball')
    x_lo, x_hi, y_lo, y_hi = extent
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    ax.set_xlabel('Ant x')
    ax.set_ylabel('Ant y')
    if title is not None:
        ax.set_title(title)
    interval_handles = _draw_interval_dots(ax, xy_per_skill)
    if K <= 16:
        skill_leg = ax.legend(loc='upper right', fontsize=7, framealpha=0.85)
        ax.add_artist(skill_leg)
    ax.legend(handles=interval_handles, loc='lower left', fontsize=6,
              framealpha=0.85, title='interval')
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Per-skill ant-path plot for DDS policies.")
    p.add_argument("--run_dir", type=str, required=True, help="DDS checkpoint run dir.")
    p.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch (default: latest).")
    p.add_argument("--steps", type=int, default=3000, help="Env steps per skill rollout.")
    p.add_argument("--temperature", type=float, default=0.0, help="Decoder sampling temperature.")
    p.add_argument("--x_min", type=float, default=0.0)
    p.add_argument("--x_max", type=float, default=20.0)
    p.add_argument("--y_min", type=float, default=0.0)
    p.add_argument("--y_max", type=float, default=20.0)
    p.add_argument("--ant_xy", type=str, default=None,
                   help="Fixed ant start x,y (default: env reset position).")
    p.add_argument("--ball_xy", type=str, default=None,
                   help="Fixed ball x,y for antsoccer (default: env reset position).")
    p.add_argument("--output", type=str, default=None, help="Output PNG (default: run dir).")
    args = p.parse_args()

    run_dir = args.run_dir
    epoch = args.epoch if args.epoch is not None else _latest_epoch(run_dir)

    with open(os.path.join(run_dir, "flags.json")) as f:
        flags = json.load(f)
    agent_cfg = flags["agent"]
    env_name = flags["env_name"]
    is_antsoccer = "antsoccer" in env_name

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

    # Lift the 1000-step TimeLimit so paths can run the full requested horizon.
    _raise_time_limit(env, args.steps)

    np.random.seed(None)
    env.reset()
    base_env = env.unwrapped

    unit = getattr(base_env, "_maze_unit", 4.0)
    offx = getattr(base_env, "_offset_x", 4.0)
    offy = getattr(base_env, "_offset_y", 4.0)
    maze_map = getattr(base_env, "maze_map", None)
    half = float(unit) / 2.0

    def overlay_maze(ax):
        if maze_map is None:
            return
        rows, cols = maze_map.shape
        for i in range(rows):
            for j in range(cols):
                if maze_map[i, j] == 1:
                    cx = j * unit - offx
                    cy = i * unit - offy
                    ax.add_patch(Rectangle((cx - half, cy - half), unit, unit,
                                           facecolor="black", edgecolor="black",
                                           linewidth=0.3, alpha=0.2))

    # Start positions.
    if is_antsoccer:
        default_ant, default_ball = base_env.get_agent_ball_xy()
        ant_xy = np.array(_parse_xy(args.ant_xy)) if args.ant_xy else np.asarray(default_ant, np.float64)
        ball_xy = np.array(_parse_xy(args.ball_xy)) if args.ball_xy else np.asarray(default_ball, np.float64)
    else:
        ant_xy = np.array(_parse_xy(args.ant_xy)) if args.ant_xy else np.asarray(base_env.get_xy(), np.float64)
        ball_xy = None

    # Codebook: one skill per code.
    codebook = np.asarray(agent._codebook_table())  # [K, D_z]
    num_skills = codebook.shape[0]
    print(f"[{env_name}] epoch={epoch} K={num_skills} steps={args.steps} "
          f"ant={np.asarray(ant_xy).tolist()}"
          + (f" ball={np.asarray(ball_xy).tolist()}" if is_antsoccer else ""))

    xy_per_skill = []
    for z in range(num_skills):
        print(f"  rolling out skill {z + 1}/{num_skills}...")
        xy_per_skill.append(rollout_dds_skill(
            env=env, agent=agent, skill_vec=codebook[z],
            is_antsoccer=is_antsoccer, ant_xy=ant_xy, ball_xy=ball_xy,
            n_steps=args.steps, temperature=args.temperature, seed=z,
        ))

    x_low, x_high = args.x_min - half, args.x_max + half
    y_low, y_high = args.y_min - half, args.y_max + half
    out = args.output or os.path.join(run_dir, f"dds_skill_paths_e{epoch}.png")
    plot_skill_paths(
        xy_per_skill=xy_per_skill, ant_start_xy=ant_xy, overlay_maze=overlay_maze,
        extent=(x_low, x_high, y_low, y_high), output_path=out, ball_xy=ball_xy,
        title=(f"DDS ant paths | {os.path.basename(run_dir)} | epoch={epoch}\n"
               f"K={num_skills}, steps={args.steps}, "
               f"start=({float(ant_xy[0]):.2f}, {float(ant_xy[1]):.2f})"),
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
