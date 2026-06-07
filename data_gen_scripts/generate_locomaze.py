import glob
import json
import os
import pathlib
import re
from collections import defaultdict

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from agents import SACAgent
from agents import agents as agent_registry
from tqdm import trange
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent
from utils.log_utils import setup_wandb

import ogbench.locomaze  # noqa

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'navigate', 'Dataset type.')
flags.DEFINE_string('restore_path', 'experts/ant', 'Expert agent restore path.')
flags.DEFINE_integer('restore_epoch', 400000, 'Expert agent restore epoch.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_float('noise', 0.2, 'Gaussian action noise level.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Maximum number of steps in an episode.')

# Empowerment-based initial-state sampling.
flags.DEFINE_string(
    'empowerment_ckpt', None,
    'Path to a pretrained empowerment run folder (containing flags.json and params_*.pkl). '
    'When set, initial cells are sampled proportionally to empowerment instead of uniformly.',
)
flags.DEFINE_integer('empowerment_epoch', None, 'Empowerment checkpoint epoch (defaults to latest params_*.pkl).')
flags.DEFINE_enum(
    'empowerment_sampling', 'linear', ['linear', 'softmax'],
    "How to turn empowerment into start-cell probabilities. "
    "'linear': p ∝ emp (assumed non-negative, no clipping). 'softmax': p ∝ exp(emp / temp).",
)
flags.DEFINE_float('empowerment_temp', 1.0, 'Softmax temperature (only used when empowerment_sampling=softmax).')
flags.DEFINE_integer('empowerment_splus_samples', 384, 'Number of s+ samples for the empowerment Monte Carlo estimate.')

# Weights & Biases logging (start-state KDE heatmap).
flags.DEFINE_bool('log_wandb', False, 'Whether to log a KDE heatmap of start states to wandb.')
flags.DEFINE_string('wandb_project', 'ogbench_datagen', 'wandb project name.')
flags.DEFINE_string('wandb_group', None, 'wandb group name.')
flags.DEFINE_string('wandb_name', None, 'wandb run name.')
flags.DEFINE_enum('wandb_mode', 'online', ['online', 'offline', 'disabled'], 'wandb mode.')


def _latest_epoch(run_dir):
    """Return the largest epoch among params_*.pkl checkpoints in run_dir."""
    ckpts = glob.glob(os.path.join(run_dir, 'params_*.pkl'))
    if not ckpts:
        raise FileNotFoundError(f'No params_*.pkl found in {run_dir}')
    epochs = []
    for path in ckpts:
        m = re.search(r'params_(\d+)\.pkl$', os.path.basename(path))
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        raise RuntimeError(f'Could not parse checkpoint epochs in {run_dir}')
    return max(epochs)


def log_start_state_empowerment_scatter(cell_xys, cell_emp, env):
    """Log a scatter of candidate start cells (x, y) colored by precomputed empowerment to wandb."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import wandb
    from matplotlib.patches import Rectangle

    cell_xys = np.asarray(cell_xys)
    cell_emp = np.asarray(cell_emp)
    n = len(cell_xys)

    base_env = env.unwrapped
    unit = float(getattr(base_env, '_maze_unit', 4.0))
    offx = float(getattr(base_env, '_offset_x', 4.0))
    offy = float(getattr(base_env, '_offset_y', 4.0))
    maze_map = getattr(base_env, 'maze_map', None)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    if maze_map is not None:
        rows, cols = maze_map.shape
        for i in range(rows):
            for j in range(cols):
                if maze_map[i, j] == 1:
                    cx = j * unit - offx
                    cy = i * unit - offy
                    ax.add_patch(Rectangle(
                        (cx - unit / 2.0, cy - unit / 2.0), unit, unit,
                        facecolor='black', edgecolor='black', linewidth=0.3, alpha=0.2,
                    ))
    sc = ax.scatter(
        cell_xys[:, 0], cell_xys[:, 1], c=cell_emp, s=120, marker='s',
        cmap='viridis', edgecolors='white', linewidths=0.5,
    )
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Start-cell empowerment — {n} cells')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='empowerment')
    plt.tight_layout()
    wandb.log({'start_cell_empowerment': wandb.Image(fig)})
    plt.close(fig)


def log_start_state_kde(start_xys, env):
    """Log a KDE heatmap of all realized start states to wandb (title shows the sample count)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import wandb
    from matplotlib.patches import Rectangle
    from scipy.stats import gaussian_kde

    start_xys = np.asarray(start_xys, dtype=np.float64)
    n = len(start_xys)

    base_env = env.unwrapped
    unit = float(getattr(base_env, '_maze_unit', 4.0))
    offx = float(getattr(base_env, '_offset_x', 4.0))
    offy = float(getattr(base_env, '_offset_y', 4.0))
    maze_map = getattr(base_env, 'maze_map', None)

    if maze_map is not None:
        rows, cols = maze_map.shape
        x_lo = -offx - unit / 2.0
        x_hi = (cols - 1) * unit - offx + unit / 2.0
        y_lo = -offy - unit / 2.0
        y_hi = (rows - 1) * unit - offy + unit / 2.0
    else:
        rows = cols = 0
        x_lo, x_hi = float(start_xys[:, 0].min()), float(start_xys[:, 0].max())
        y_lo, y_hi = float(start_xys[:, 1].min()), float(start_xys[:, 1].max())

    res = 200
    xs = np.linspace(x_lo, x_hi, res)
    ys = np.linspace(y_lo, y_hi, res)
    xx, yy = np.meshgrid(xs, ys)
    kde = gaussian_kde(start_xys.T)
    dens = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(res, res)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(
        dens, origin='lower', extent=[x_lo, x_hi, y_lo, y_hi], aspect='auto', cmap='viridis'
    )
    if maze_map is not None:
        for i in range(rows):
            for j in range(cols):
                if maze_map[i, j] == 1:
                    cx = j * unit - offx
                    cy = i * unit - offy
                    ax.add_patch(Rectangle(
                        (cx - unit / 2.0, cy - unit / 2.0), unit, unit,
                        facecolor='black', edgecolor='black', linewidth=0.3, alpha=0.3,
                    ))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Start-state density (KDE) — N={n} samples')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    wandb.log({'start_state_kde': wandb.Image(fig)})
    plt.close(fig)


def main(_):
    assert FLAGS.dataset_type in ['path', 'navigate', 'stitch', 'explore']
    # 'path': Reach a single goal and stay there.
    # 'navigate': Repeatedly reach randomly sampled goals in a single episode.
    # 'stitch': Reach a nearby goal that is 4 cells away and stay there.
    # 'explore': Repeatedly follow random directions sampled every 10 steps.

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        max_episode_steps=FLAGS.max_episode_steps,
    )
    ob_dim = env.observation_space.shape[0]

    # Initialize oracle agent.
    if 'point' in FLAGS.env_name:

        def actor_fn(ob, temperature):
            return ob[-2:]
    else:
        # Load agent config.
        restore_path = FLAGS.restore_path
        candidates = glob.glob(restore_path)
        assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

        with open(candidates[0] + '/flags.json', 'r') as f:
            agent_config = json.load(f)['agent']

        # Load agent.
        agent = SACAgent.create(
            FLAGS.seed,
            np.zeros(ob_dim),
            env.action_space.sample(),
            agent_config,
        )
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        actor_fn = supply_rng(agent.sample_actions, rng=agent.rng)

    # Store all empty cells and vertex cells.
    all_cells = []
    vertex_cells = []
    maze_map = env.unwrapped.maze_map
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                all_cells.append((i, j))

                # Exclude hallway cells.
                if (
                    maze_map[i - 1, j] == 0
                    and maze_map[i + 1, j] == 0
                    and maze_map[i, j - 1] == 1
                    and maze_map[i, j + 1] == 1
                ):
                    continue
                if (
                    maze_map[i, j - 1] == 0
                    and maze_map[i, j + 1] == 0
                    and maze_map[i - 1, j] == 1
                    and maze_map[i + 1, j] == 1
                ):
                    continue

                vertex_cells.append((i, j))

    # Set up empowerment-based initial-state sampling.
    use_empowerment = FLAGS.empowerment_ckpt is not None
    cell_xys = None
    cell_emp = None  # Precomputed empowerment per candidate start cell.
    cell_probs = None  # Fixed sampling distribution over start cells.
    if use_empowerment:
        run_dir = FLAGS.empowerment_ckpt
        with open(os.path.join(run_dir, 'flags.json'), 'r') as f:
            emp_flags = json.load(f)
        agent_cfg = emp_flags['agent']
        agent_cfg['num_splus_samples'] = FLAGS.empowerment_splus_samples
        if agent_cfg.get('frame_stack') is not None:
            raise ValueError(
                'Empowerment estimators trained with frame_stack are not supported by this script '
                '(the data-collection env produces unstacked observations).'
            )
        epoch = FLAGS.empowerment_epoch if FLAGS.empowerment_epoch is not None else _latest_epoch(run_dir)

        # Build an example batch from the data-collection env (obs/action dims match the estimator's env).
        example_obs = np.zeros((1, ob_dim), dtype=np.float32)
        if agent_cfg.get('discrete'):
            example_act = np.full((1,), env.action_space.n - 1)
        else:
            example_act = np.asarray(env.action_space.sample(), dtype=np.float32)[None]

        agent_class = agent_registry[agent_cfg['agent_name']]
        emp_agent = agent_class.create(
            seed=FLAGS.seed,
            ex_observations=example_obs,
            ex_actions=example_act,
            config=agent_cfg,
        )
        emp_agent = restore_agent(emp_agent, run_dir, epoch)

        @jax.jit
        def emp_for_cells(obs_batch, rng):
            return emp_agent.empowerment(obs_batch, rng)

        # Precompute empowerment once over the finite set of start cells. Only x-y differs between
        # cells; the proprioceptive part comes from a single default reset observation.
        template_ob, _ = env.reset()
        template_ob = np.asarray(template_ob, dtype=np.float32)
        cell_xys = np.array([env.unwrapped.ij_to_xy(c) for c in all_cells], dtype=np.float32)
        obs_batch = np.repeat(template_ob[None, :], len(all_cells), axis=0)
        obs_batch[:, 0] = cell_xys[:, 0]
        obs_batch[:, 1] = cell_xys[:, 1]
        cell_emp = np.asarray(emp_for_cells(jnp.asarray(obs_batch), jax.random.PRNGKey(FLAGS.seed)))

        # Convert empowerment into a fixed sampling distribution over cells.
        if FLAGS.empowerment_sampling == 'softmax':
            logits = cell_emp / FLAGS.empowerment_temp
            cell_probs = np.exp(logits - logits.max())
            cell_probs = cell_probs / cell_probs.sum()
        else:
            if (cell_emp < 0).any():
                raise ValueError(
                    'Negative empowerment encountered with empowerment_sampling=linear; '
                    'use --empowerment_sampling=softmax instead.'
                )
            cell_probs = cell_emp / cell_emp.sum()

        print(
            f'Empowerment sampling enabled: ckpt={run_dir}, epoch={epoch}, '
            f'mode={FLAGS.empowerment_sampling}, temp={FLAGS.empowerment_temp}, '
            f'num_cells={len(all_cells)}, num_splus_samples={FLAGS.empowerment_splus_samples}, '
            f'emp[min/mean/max]={cell_emp.min():.4f}/{cell_emp.mean():.4f}/{cell_emp.max():.4f}',
            flush=True,
        )

    if FLAGS.log_wandb:
        setup_wandb(
            project=FLAGS.wandb_project,
            group=FLAGS.wandb_group,
            name=FLAGS.wandb_name,
            mode=FLAGS.wandb_mode,
        )
        # Log the precomputed per-cell empowerment landscape once at the start.
        if use_empowerment:
            log_start_state_empowerment_scatter(cell_xys, cell_emp, env)

    # Collect data.
    dataset = defaultdict(list)
    start_xys = []  # Realized start x-y for every episode (for the wandb KDE heatmap).
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        # Sample an initial cell, either uniformly or from the precomputed empowerment distribution.
        if use_empowerment:
            init_ij = all_cells[np.random.choice(len(all_cells), p=cell_probs)]
        else:
            init_ij = all_cells[np.random.randint(len(all_cells))]

        # Sample a goal cell (depends on the dataset type).
        if FLAGS.dataset_type in ['path', 'navigate', 'explore']:
            # Sample a goal state from vertex cells.
            goal_ij = vertex_cells[np.random.randint(len(vertex_cells))]
        elif FLAGS.dataset_type == 'stitch':
            # Perform BFS to find adjacent cells.
            adj_cells = []
            adj_steps = 4  # Target distance from the initial cell.
            bfs_map = maze_map.copy()
            for i in range(bfs_map.shape[0]):
                for j in range(bfs_map.shape[1]):
                    bfs_map[i][j] = -1
            bfs_map[init_ij[0], init_ij[1]] = 0
            queue = [init_ij]
            while len(queue) > 0:
                i, j = queue.pop(0)
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < bfs_map.shape[0]
                        and 0 <= nj < bfs_map.shape[1]
                        and maze_map[ni, nj] == 0
                        and bfs_map[ni, nj] == -1
                    ):
                        bfs_map[ni][nj] = bfs_map[i][j] + 1
                        queue.append((ni, nj))
                        if bfs_map[ni][nj] == adj_steps:
                            adj_cells.append((ni, nj))

            # Sample a goal state from adjacent cells.
            goal_ij = adj_cells[np.random.randint(len(adj_cells))] if len(adj_cells) > 0 else init_ij
        else:
            raise ValueError(f'Unsupported dataset_type: {FLAGS.dataset_type}')

        ob, _ = env.reset(options=dict(task_info=dict(init_ij=init_ij, goal_ij=goal_ij)))
        start_xys.append(np.asarray(env.unwrapped.get_xy(), dtype=np.float32))

        done = False
        step = 0

        cur_subgoal_dir = None  # Current subgoal direction (only for 'explore').

        while not done:
            if FLAGS.dataset_type == 'explore':
                # Sample a random direction every 10 steps.
                if step % 10 == 0:
                    cur_subgoal_dir = np.random.randn(2)
                    cur_subgoal_dir = cur_subgoal_dir / (np.linalg.norm(cur_subgoal_dir) + 1e-6)
                subgoal_dir = cur_subgoal_dir
            else:
                # Get the oracle subgoal and compute the direction.
                subgoal_xy, _ = env.unwrapped.get_oracle_subgoal(env.unwrapped.get_xy(), env.unwrapped.cur_goal_xy)
                subgoal_dir = subgoal_xy - env.unwrapped.get_xy()
                subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)

            agent_ob = env.unwrapped.get_ob(ob_type='states')
            # Exclude the agent's position and add the subgoal direction.
            agent_ob = np.concatenate([agent_ob[2:], subgoal_dir])
            action = actor_fn(agent_ob, temperature=0)
            # Add Gaussian noise to the action.
            action = action + np.random.normal(0, FLAGS.noise, action.shape)
            action = np.clip(action, -1, 1)

            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = info['success']

            # Sample a new goal state when the current goal is reached.
            if success and FLAGS.dataset_type == 'navigate':
                goal_ij = vertex_cells[np.random.randint(len(vertex_cells))]
                env.unwrapped.set_goal(goal_ij)

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])

            ob = next_ob
            step += 1

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)

    if FLAGS.log_wandb:
        log_start_state_kde(start_xys, env)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals':
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)
