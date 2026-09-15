"""Render videos of a trained offline high-level controller executing in the env.

Loads a controller run directory (the one holding `flags.json` / `params_*.pkl`,
e.g. a `controller_awr_sweep/alpha10/OGBench/Debug/sd000_*` folder), rebuilds the
agent exactly as `main.py` would, restores the checkpoint, and rolls it out with
rendering on. One mp4 per episode, plus a per-task grid mp4 of all episodes.

The env is built with `make_env_only`, so no offline dataset is touched: the
agent only needs example observation/action SHAPES, which come from the env's
spaces.

Usage:
    MUJOCO_GL=osmesa python render_controller_rollouts.py \
        --run_dir ckpts/final/.../controller_awr_sweep/alpha10/OGBench/Debug/sd000_* \
        --out_dir videos/emp_antmaze --tasks 1 2 3 4 5 --episodes 2
"""

import argparse
import glob
import json
import os
import re

import imageio
import numpy as np
from PIL import Image, ImageEnhance

from agents import agents
from utils.env_utils import make_env_only
from utils.evaluation import init_eval_state, supply_rng
from utils.flax_utils import restore_agent


def restore_config(config, saved):
    """Overwrite `config` in place with the values recorded in `saved` (main.py's copy)."""
    for key, value in saved.items():
        if key not in config:
            continue
        current = config[key]
        if hasattr(current, 'items') and isinstance(value, dict):
            restore_config(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            config[key] = tuple(value)
        else:
            config[key] = value


def resolve_run_dir(pattern):
    matches = sorted(glob.glob(pattern))
    matches = [m for m in matches if os.path.exists(os.path.join(m, 'flags.json'))]
    if len(matches) != 1:
        raise ValueError(f'--run_dir must match exactly one run folder with flags.json, got {matches}')
    return matches[0].rstrip('/')


def latest_epoch(run_dir):
    epochs = [int(re.search(r'params_(\d+)\.pkl', p).group(1))
              for p in glob.glob(os.path.join(run_dir, 'params_*.pkl'))]
    if not epochs:
        raise FileNotFoundError(f'no params_*.pkl in {run_dir}')
    return max(epochs)


def fix_skill_path(path):
    """Checkpoints recorded pre-move paths (`ckpts/<x>`); they now live under `ckpts/final/<x>`."""
    if path is None or os.path.exists(os.path.join(path, 'flags.json')):
        return path
    alt = path.replace('ckpts/', 'ckpts/final/', 1)
    if os.path.exists(os.path.join(alt, 'flags.json')):
        print(f'[render] skill checkpoint moved: {path} -> {alt}')
        return alt
    return path


def build_agent(run_dir, restore_epoch, seed=0):
    with open(os.path.join(run_dir, 'flags.json')) as f:
        saved = json.load(f)
    agent_name = saved['agent']['agent_name']
    env_name = saved['env_name']

    module = __import__(f'agents.{agent_name}', fromlist=['get_config'])
    base_name = saved['agent'].get('base_agent_name')
    config = module.get_config(base_name) if base_name is not None else module.get_config()
    restore_config(config, saved['agent'])
    if 'skill_checkpoint_path' in config:
        config['skill_checkpoint_path'] = fix_skill_path(config['skill_checkpoint_path'])

    env = make_env_only(env_name, frame_stack=config.get('frame_stack'))
    obs_dim = env.observation_space.shape
    ex_observations = np.zeros((1, *obs_dim), dtype=np.float32)
    if config.get('discrete'):
        ex_actions = np.full((1,), env.action_space.n - 1, dtype=np.int32)
    else:
        ex_actions = np.zeros((1, *env.action_space.shape), dtype=np.float32)

    agent = agents[agent_name].create(seed, ex_observations, ex_actions, config)
    agent = restore_agent(agent, run_dir, restore_epoch)
    return agent, env, config, env_name


def rollout(agent, env, config, task_id, seed, frame_skip, max_steps=None):
    """One rendered episode. Returns (frames, success)."""
    actor_fn = supply_rng(agent.sample_actions_with_state,
                          rng=__import__('jax').random.PRNGKey(seed))
    observation, info = env.reset(seed=seed, options=dict(task_id=task_id, render_goal=True))
    goal = info.get('goal')
    goal_frame = info.get('goal_rendered')
    agent_state = init_eval_state(agent, env)
    frames, step, done = [], 0, False
    while not done:
        action, agent_state = actor_fn(observations=observation, goals=goal,
                                       temperature=0, agent_state=agent_state)
        action = np.array(action)
        if not config.get('discrete'):
            action = np.clip(action, -1, 1)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        if max_steps is not None and step >= max_steps:
            done = True
        if step % frame_skip == 0 or done:
            frame = env.render().copy()
            frames.append(np.concatenate([goal_frame, frame], axis=0) if goal_frame is not None else frame)
    return np.array(frames), float(info.get('success', 0.0))


def grid_video(renders, n_cols):
    """Pad every clip to the same length (dimming the freeze frame) and tile them."""
    max_length = max(len(r) for r in renders)
    padded = []
    for render in renders:
        final = np.array(ImageEnhance.Brightness(Image.fromarray(render[-1])).enhance(0.5))
        pad = np.repeat(final[np.newaxis], max_length - len(render), axis=0)
        clip = np.concatenate([render, pad], axis=0) if len(pad) else render
        padded.append(np.pad(clip, ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=0))
    n_rows = int(np.ceil(len(padded) / n_cols))
    blank = np.zeros_like(padded[0])
    padded += [blank] * (n_rows * n_cols - len(padded))
    rows = [np.concatenate(padded[r * n_cols:(r + 1) * n_cols], axis=2) for r in range(n_rows)]
    return np.concatenate(rows, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True, help='Controller run folder (glob allowed).')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--restore_epoch', type=int, default=None)
    parser.add_argument('--tasks', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--episodes', type=int, default=2, help='Episodes per task.')
    parser.add_argument('--frame_skip', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    epoch = args.restore_epoch if args.restore_epoch is not None else latest_epoch(run_dir)
    agent, env, config, env_name = build_agent(run_dir, epoch)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[render] {env_name} | {run_dir} @ epoch {epoch}')

    all_clips, successes = [], {}
    for task_id in args.tasks:
        for ep in range(args.episodes):
            seed = args.seed + 1000 * task_id + ep
            frames, success = rollout(agent, env, config, task_id, seed, args.frame_skip, args.max_steps)
            successes[(task_id, ep)] = success
            path = os.path.join(args.out_dir, f'task{task_id}_ep{ep}_{"success" if success else "fail"}.mp4')
            imageio.mimsave(path, frames, fps=args.fps, macro_block_size=1)
            all_clips.append(frames)
            print(f'  task {task_id} ep {ep}: success={success:.0f}  {len(frames)} frames -> {path}')

    grid = grid_video(all_clips, n_cols=args.episodes)  # one row per task
    grid_path = os.path.join(args.out_dir, 'all_tasks_grid.mp4')
    imageio.mimsave(grid_path, grid, fps=args.fps, macro_block_size=1)
    print(f'[render] grid -> {grid_path}')
    print(f'[render] success rate: {np.mean(list(successes.values())):.2f} over {len(successes)} episodes')


if __name__ == '__main__':
    main()
