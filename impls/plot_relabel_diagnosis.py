"""Why the BC-relabelled high-level controller underperforms: relabelling fidelity.

`skill_bc_relabel_controller` builds its option MDP by labelling every H-step
dataset window with

    z*(t) = argmax_z  sum_i log pi(a_{t+i} | s_{t+i}, z)

under the FROZEN `empowerment_skill` policy, and then trains pi_hi(z | s, g) on the
option transitions ``(s_t, z*(t), s_{t+H})`` that the DATA realised. That is a valid
option model only if *executing* z*(t) from s_t actually reproduces the data window.
Nothing enforces it: the empowerment policy was never trained to reconstruct dataset
actions, and its actor is a `const_std` diagonal Gaussian, so the argmax reduces to a
nearest-mean (MSE) tiebreak among K near-identical action means. The H-step outcome
credited to a skill can therefore be nowhere near where that skill actually goes, and
the error compounds over the window.

DDS does not have this failure mode: its labels come from a VQ-VAE trajectory encoder
trained *jointly* with the decoder, so label and execution agree by construction.

This script measures the gap three ways and writes three figures into the controller
run dir:

  relabel_rollout_paths_e{E}.png
      Per eval task: the trajectory the trained controller actually walks, and -- at
      every high-level decision point -- the H-step paths the TRAINING DATA claims the
      chosen skill produces there (nearest relabelled windows carrying that label,
      anchored at the decision state). The gap between the two is exactly the model
      error the controller optimised against.

  relabel_offline_fidelity_e{E}.png
      Ground truth for the same question, without the controller in the loop: sample
      dataset windows, take their assigned label, teleport the simulator back to the
      window's first state, execute that skill for H steps, and overlay the executed
      path on the data path. Plus the mean position error vs. step index (the
      accumulation curve) against an oracle skill (the best of all K by outcome) and a
      random skill.

  relabel_labels_e{E}.png
      Is the label itself informative? Label histogram over the K skills, the
      per-step top1-top2 log-likelihood margin, the fraction of window steps whose own
      argmax equals the window label as a function of position in the window, and the
      on-policy self-consistency check (relabel the chunks the controller itself just
      executed and compare with the skill it commanded).

Both `skill_bc_relabel_controller` and `dds_controller` run dirs are accepted; the DDS
path uses `label_chunk_skills` (the VQ encoder) wherever the BC path uses
`chunk_skill_logliks`, so the two are directly comparable.

Usage (from impls/):

    python plot_relabel_diagnosis.py --controller_dir ckpts/final/empowerment_final/\
pointmaze-teleport-navigate/sd000_*/controller_awr_sweep/alpha1/OGBench/Debug/sd000_*
"""

import os

os.environ.setdefault('MUJOCO_GL', 'osmesa')  # EGL cannot open a headless display here.

import argparse
import glob
import json
import re

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from scipy.spatial import cKDTree

from agents import agents as agent_registry
from utils.datasets import Dataset, GCDataset, HGCDataset, SequenceDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import env_horizon, raise_time_limit
from utils.flax_utils import restore_agent

# Categorical slots 1/2/3/4 of the validated reference palette, assigned in fixed order
# and never cycled. Blue is always "what happened / what the data says", orange is
# always "what the assigned skill actually does", aqua is the oracle, yellow the
# random-skill control.
C_DATA = '#2a78d6'
C_ASSIGNED = '#eb6834'
C_ORACLE = '#1baf7a'
C_RANDOM = '#eda100'
INK = '#1a1a19'
INK_MUTED = '#6b6a63'
# One-hue sequential ramp (blue 250 -> 700) for time within a rollout.
TIME_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#256abf', '#184f95', '#0d366b']


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────
def latest_epoch(run_dir):
    epochs = [
        int(m.group(1))
        for path in glob.glob(os.path.join(run_dir, 'params_*.pkl'))
        if (m := re.search(r'params_(\d+)\.pkl$', os.path.basename(path)))
    ]
    if not epochs:
        raise FileNotFoundError(f'No params_*.pkl checkpoint found in {run_dir}')
    return max(epochs)


def build(run_dir, epoch, dataset_path=None):
    """Rebuild the controller from its own flags.json and restore params_{epoch}.pkl."""
    with open(os.path.join(run_dir, 'flags.json')) as f:
        saved = json.load(f)
    config = saved['agent']
    env_name = saved['env_name']
    env, train_dataset, _ = make_env_and_datasets(
        env_name,
        frame_stack=config.get('frame_stack'),
        dataset_path=dataset_path if dataset_path is not None else saved.get('dataset_path'),
    )
    dataset_class = {'GCDataset': GCDataset, 'HGCDataset': HGCDataset, 'SequenceDataset': SequenceDataset}[
        config['dataset_class']
    ]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)

    example_batch = train_dataset.sample(1)
    if config.get('discrete'):
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)
    agent = agent_registry[config['agent_name']].create(
        seed=saved.get('seed', 0),
        ex_observations=example_batch['observations'],
        ex_actions=example_batch['actions'],
        config=config,
    )
    agent = restore_agent(agent, run_dir, epoch)
    return agent, env, config, train_dataset, env_name


# ──────────────────────────────────────────────────────────────────────────────
# Environment: exact state restore + xy extraction + maze background
# ──────────────────────────────────────────────────────────────────────────────
class EnvIO:
    """Set the simulator to an arbitrary dataset observation, and read xy back out.

    PointMaze's observation IS `qpos` (2-D), so `set_xy` restores it exactly. The ant
    families observe `concat(qpos, qvel)`, the full MuJoCo state, so `set_state`
    restores those exactly too -- verified by a round-trip check in `__init__`.
    """

    def __init__(self, env):
        self.env = env
        self.base = env.unwrapped
        self.nq = int(self.base.model.nq)
        self.nv = int(self.base.model.nv)
        self.obs_dim = int(np.asarray(self.base.get_ob()).shape[0])
        self.full_state = self.obs_dim == self.nq + self.nv
        self.has_ball = hasattr(self.base, 'get_agent_ball_xy')

    def set(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        if self.full_state:
            self.base.set_state(obs[: self.nq], obs[self.nq : self.nq + self.nv])
        else:
            self.base.set_xy(obs[: self.nq])

    def ob(self):
        return np.asarray(self.base.get_ob(), dtype=np.float32)

    def xy(self, obs):
        """Agent xy from a (batch of) observation(s)."""
        return np.asarray(obs)[..., :2]

    def ball_xy(self, obs):
        """Ball xy, or None for envs without one. qpos layout puts it at qpos[-7:-5]."""
        if not self.has_ball:
            return None
        return np.asarray(obs)[..., self.nq - 7 : self.nq - 5]

    def features(self, obs):
        """The state descriptor used for nearest-neighbour lookups in the dataset."""
        if self.has_ball:
            return np.concatenate([self.xy(obs), self.ball_xy(obs)], axis=-1)
        return self.xy(obs)


class Maze:
    """Wall map / teleporter geometry for the background of every spatial panel."""

    def __init__(self, env):
        base = env.unwrapped
        self.map = getattr(base, 'maze_map', None)
        self.unit = float(getattr(base, '_maze_unit', 4.0))
        self.offx = float(getattr(base, '_offset_x', 4.0))
        self.offy = float(getattr(base, '_offset_y', 4.0))
        self.teleport = getattr(base, '_teleport_info', None)
        if self.map is None:
            self.extent = None
        else:
            rows, cols = self.map.shape
            half = self.unit / 2.0
            self.extent = (
                -self.offx - half,
                (cols - 1) * self.unit - self.offx + half,
                -self.offy - half,
                (rows - 1) * self.unit - self.offy + half,
            )

    def overlay(self, ax, alpha=0.18):
        if self.map is not None:
            rows, cols = self.map.shape
            for i in range(rows):
                for j in range(cols):
                    if self.map[i, j] == 1:
                        ax.add_patch(
                            Rectangle(
                                (j * self.unit - self.offx - self.unit / 2.0,
                                 i * self.unit - self.offy - self.unit / 2.0),
                                self.unit,
                                self.unit,
                                facecolor=INK,
                                edgecolor='none',
                                alpha=alpha,
                            )
                        )
        if self.teleport is not None:
            r = float(self.teleport.get('teleport_radius', 1.0))
            for (x, y) in self.teleport.get('teleport_in_xys', []):
                ax.add_patch(Circle((x, y), r, facecolor='none', edgecolor=INK_MUTED, lw=1.2))
            for (x, y) in self.teleport.get('teleport_out_xys', []):
                ax.add_patch(Circle((x, y), r, facecolor='none', edgecolor=INK_MUTED, lw=1.2, ls='--'))
        if self.extent is not None:
            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7, colors=INK_MUTED, length=2)
        for spine in ax.spines.values():
            spine.set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# Agent-family shims: BC-likelihood relabeller vs DDS VQ encoder
# ──────────────────────────────────────────────────────────────────────────────
class Labeller:
    """Uniform access to 'which skill explains this chunk' across controller families."""

    def __init__(self, agent):
        self.agent = agent
        self.kind = 'loglik' if hasattr(agent, 'chunk_skill_logliks') else 'encoder'
        self.num_skills = int(agent.config['num_skills'])

    def step_logliks(self, observations, actions):
        """[T, K] per-step log pi(a_t | s_t, z). None for the encoder family."""
        if self.kind != 'loglik':
            return None
        return np.asarray(jax.device_get(self.agent.chunk_skill_logliks(observations, actions)), dtype=np.float64)

    def label_chunk(self, observations, actions):
        """The single skill assigned to one [T, ...] window, exactly as training does."""
        if self.kind == 'loglik':
            return int(np.argmax(self.step_logliks(observations, actions).sum(axis=0)))
        obs_seq = jnp.asarray(observations)[None]
        act_seq = jnp.asarray(actions)[None]
        mask = jnp.ones(obs_seq.shape[:2], dtype=jnp.float32)
        return int(np.asarray(jax.device_get(self.agent.label_chunk_skills(obs_seq, act_seq, mask, self.agent.rng)))[0])


def skill_action_fn(agent):
    """action(obs, z_index, key) under the frozen low-level policy, exactly as at eval."""
    num_skills = int(agent.config['num_skills'])
    if hasattr(agent.skill_agent, '_codebook_table'):  # DDS: index -> codebook vector.
        table = np.asarray(jax.device_get(agent.skill_agent._codebook_table()))

        def vec(z):
            return jnp.asarray(table[z])
    else:  # empowerment_skill: index -> one-hot.
        eye = np.eye(num_skills, dtype=np.float32)

        def vec(z):
            return jnp.asarray(eye[z])

    def act(obs, z, key):
        a = agent.sample_actions_with_skill(observations=jnp.asarray(obs), skills=vec(z), seed=key)
        return np.clip(np.asarray(jax.device_get(a)), -1.0, 1.0)

    return act


# ──────────────────────────────────────────────────────────────────────────────
# Rollouts
# ──────────────────────────────────────────────────────────────────────────────
def rollout_controller(agent, env, task_id, seed, max_steps):
    """One eval episode of the trained controller, logging the committed skill per step.

    `max_steps` is passed explicitly because `execute_skill` needs the env's TimeLimit
    lifted, and OGBench maze envs never set `terminated` -- without a cap the loop would
    never end.
    """
    np.random.seed(seed)
    obs, info = env.reset(options=dict(task_id=task_id))
    goal = info['goal']
    state = agent.init_eval_state()
    rng = jax.random.PRNGKey(seed)

    observations, actions, skills = [np.asarray(obs, dtype=np.float32)], [], []
    done, success = False, 0.0
    while not done and len(actions) < max_steps:
        rng, key = jax.random.split(rng)
        action, state = agent.sample_actions_with_state(
            observations=jnp.asarray(obs), goals=jnp.asarray(goal), agent_state=state, seed=key, temperature=0.0
        )
        action = np.clip(np.asarray(jax.device_get(action)), -1.0, 1.0)
        skills.append(int(jax.device_get(state['skill'])))
        actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        observations.append(np.asarray(obs, dtype=np.float32))
        done = terminated or truncated
        success = float(info.get('success', success))
    return dict(
        observations=np.stack(observations),
        actions=np.stack(actions),
        skills=np.asarray(skills, dtype=np.int32),
        goal=np.asarray(goal, dtype=np.float32),
        success=success,
    )


def execute_skill(env, io, act_fn, obs0, z, horizon, key):
    """Teleport to `obs0`, then run the frozen low-level policy under skill `z` for H steps."""
    io.set(obs0)
    obs = io.ob()
    path = [obs]
    for _ in range(horizon):
        key, sub = jax.random.split(key)
        obs, _, _, _, _ = env.step(act_fn(obs, z, sub))
        obs = np.asarray(obs, dtype=np.float32)
        path.append(obs)
    return np.stack(path)


# ──────────────────────────────────────────────────────────────────────────────
# Relabelling the offline dataset (cached: it is the expensive step)
# ──────────────────────────────────────────────────────────────────────────────
def dataset_labels(agent, dataset, cache_path, force=False):
    """`dataset.chunk_skills` -- the exact labels training used. Cached to `cache_path`."""
    if not force and os.path.exists(cache_path):
        labels = np.load(cache_path)
        if labels.shape[0] == dataset.size:
            dataset.chunk_skills = labels
            print(f'[relabel] loaded cached labels from {cache_path}')
            return labels
    print(f'[relabel] labelling {dataset.size} windows (this is the training-time pass)...')
    agent.prepare_datasets([dataset])
    labels = np.asarray(dataset.chunk_skills)
    np.save(cache_path, labels)
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 -- rollouts: actual path vs. the path the data credits to the chosen skill
# ──────────────────────────────────────────────────────────────────────────────
def collect_rollouts(agent, env, tasks, episodes, seed, max_steps):
    """One dict per (task, episode); reused by both the path figure and the label figure."""
    out = []
    for task_id in tasks:
        for ep in range(episodes):
            traj = rollout_controller(agent, env, task_id, seed + 100 * task_id + ep, max_steps)
            traj['task_id'] = task_id
            out.append(traj)
        print(f'[rollout] task {task_id}: success '
              f'{np.mean([t["success"] for t in out if t["task_id"] == task_id]):.2f}')
    return out


def figure_rollouts(io, maze, dataset, labels, horizon, tasks, rollouts, out_path, title):
    """Per task: the executed trajectory, and the data windows behind each skill choice."""
    starts = valid_window_starts(dataset, horizon)
    feats = io.features(dataset.dataset['observations'][starts])
    trees = {}

    def ghosts(obs_t, z, k=6):
        """The k relabelled windows with label z whose start state is nearest obs_t."""
        if z not in trees:
            sel = np.flatnonzero(labels[starts] == z)
            trees[z] = (cKDTree(feats[sel]) if sel.size else None, starts[sel])
        tree, idxs = trees[z]
        if tree is None:
            return None, np.inf
        d, j = tree.query(io.features(obs_t)[None], k=min(k, len(idxs)))
        j = np.atleast_1d(np.asarray(j).ravel())
        d = np.atleast_1d(np.asarray(d).ravel())
        picked = idxs[j]
        paths = np.stack([dataset.dataset['observations'][i : i + horizon + 1] for i in picked])
        return paths, float(d.min())

    n = len(tasks)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 5.9 * nrows), squeeze=False)

    summary = []
    for ax, task_id in zip(axes.ravel(), tasks):
        maze.overlay(ax)
        cos_all, gap_all, ghost_dists, mag_actual, mag_believed = [], [], [], [], []
        for traj in [t for t in rollouts if t['task_id'] == task_id]:
            xy = io.xy(traj['observations'])
            gxy = io.xy(traj['goal'])
            n_steps = len(traj['actions'])

            # Trajectory, shaded by progress through the episode (one-hue ramp).
            seg = max(1, n_steps // len(TIME_RAMP))
            for s in range(len(TIME_RAMP)):
                lo, hi = s * seg, min(n_steps, (s + 1) * seg + 1)
                if hi > lo:
                    ax.plot(xy[lo:hi, 0], xy[lo:hi, 1], color=TIME_RAMP[s], lw=2.0,
                            solid_capstyle='round', zorder=3)

            for t in range(0, n_steps - horizon, horizon):
                z = int(traj['skills'][t])
                actual = xy[t : t + horizon + 1] - xy[t]
                paths, dmin = ghosts(traj['observations'][t], z)
                ghost_dists.append(dmin)
                if paths is None:
                    continue
                gh = io.xy(paths) - io.xy(paths[:, :1])
                for g in gh:
                    ax.plot(xy[t, 0] + g[:, 0], xy[t, 1] + g[:, 1],
                            color=C_ASSIGNED, lw=0.9, alpha=0.55, zorder=4)
                mean_g = gh.mean(axis=0)[-1]
                da, dg = actual[-1], mean_g
                if np.linalg.norm(da) > 1e-6 and np.linalg.norm(dg) > 1e-6:
                    cos_all.append(float(da @ dg / (np.linalg.norm(da) * np.linalg.norm(dg))))
                gap_all.append(float(np.linalg.norm(da - dg)))
                mag_actual.append(float(np.linalg.norm(da)))
                mag_believed.append(float(np.linalg.norm(gh[:, -1], axis=1).mean()))
                ax.plot([xy[t, 0]], [xy[t, 1]], 'o', ms=3.0, color=INK, zorder=6)

            ax.plot([xy[0, 0]], [xy[0, 1]], marker='*', ms=13, color=INK, zorder=7)
            ax.plot([gxy[0]], [gxy[1]], marker='X', ms=10, color=INK, zorder=7)
            if io.has_ball:
                bxy = io.ball_xy(traj['observations'])
                ax.plot(bxy[:, 0], bxy[:, 1], color=INK_MUTED, lw=1.0, ls=':', zorder=2)

        mean = lambda v: float(np.mean(v)) if len(v) else float('nan')
        succ = float(np.mean([t['success'] for t in rollouts if t['task_id'] == task_id]))
        summary.append(dict(task=task_id, success=succ, cos=mean(cos_all), gap=mean(gap_all),
                            nn_dist=mean(ghost_dists), mag_actual=mean(mag_actual),
                            mag_believed=mean(mag_believed)))
        ax.set_title(
            f'task {task_id} · success {succ:.0%}\n'
            f'per-option |Δ|: actual {mean(mag_actual):.2f} m vs believed {mean(mag_believed):.2f} m · '
            f'cos {mean(cos_all):.2f}\n'
            f'endpoint gap {mean(gap_all):.2f} m · nearest labelled window {mean(ghost_dists):.2f} m away',
            fontsize=8.5, color=INK,
        )

    for ax in axes.ravel()[n:]:
        ax.axis('off')

    for r in summary:
        print(f'[rollout] task {r["task"]}: success {r["success"]:.2f} | per-option |d| actual '
              f'{r["mag_actual"]:.3f} m vs believed {r["mag_believed"]:.3f} m | cos {r["cos"]:.2f} '
              f'| endpoint gap {r["gap"]:.3f} m | nearest labelled window {r["nn_dist"]:.2f} m')

    handles = [
        Line2D([0], [0], color=TIME_RAMP[2], lw=2.0, label='executed trajectory (early → late shading)'),
        Line2D([0], [0], color=C_ASSIGNED, lw=1.2, label='data windows carrying the chosen skill, anchored at the decision state'),
        Line2D([0], [0], color=INK, marker='o', ls='none', ms=4, label='high-level decision point'),
        Line2D([0], [0], color=INK, marker='*', ls='none', ms=10, label='start'),
        Line2D([0], [0], color=INK, marker='X', ls='none', ms=8, label='goal'),
    ]
    if io.has_ball:
        handles.append(Line2D([0], [0], color=INK_MUTED, lw=1.0, ls=':', label='ball path'))
    fig.suptitle(title, fontsize=11, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96), h_pad=3.0)
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.005))
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 -- offline fidelity: re-execute the assigned skill from the window's start
# ──────────────────────────────────────────────────────────────────────────────
def valid_window_starts(dataset, horizon):
    """Indices whose full length-(H+1) window stays inside one trajectory."""
    all_idxs = np.arange(dataset.size)
    final = dataset.terminal_locs[np.searchsorted(dataset.terminal_locs, all_idxs)]
    return all_idxs[final - all_idxs >= horizon]

def figure_offline(agent, env, io, maze, dataset, labels, horizon, num_windows, num_oracle,
                   seed, out_path, title):
    """Re-execute each window's assigned skill from the window's own first state.

    The assigned-skill curve is only meaningful against two references: an ORACLE
    (the best of all K skills for that window -- how well *any* skill could have
    explained it) and a RANDOM skill (what a label carrying no information would
    score). A label whose curve sits on the random curve is noise.
    """
    rng_np = np.random.default_rng(seed)
    act_fn = skill_action_fn(agent)
    num_skills = int(agent.config['num_skills'])
    raise_time_limit(env, 10_000_000)
    env.reset()

    starts = valid_window_starts(dataset, horizon)
    picks = rng_np.choice(starts, size=min(num_windows, len(starts)), replace=False)
    key = jax.random.PRNGKey(seed)

    data_paths, exec_paths = [], []
    for t in picks:
        obs_win = dataset.dataset['observations'][t : t + horizon + 1]
        key, sub = jax.random.split(key)
        data_paths.append(obs_win)
        exec_paths.append(execute_skill(env, io, act_fn, obs_win[0], int(labels[t]), horizon, sub))
    data_paths = np.stack(data_paths)
    exec_paths = np.stack(exec_paths)

    # Oracle and random-skill references, on a subset (K executions per window).
    oracle_picks = picks[: min(num_oracle, len(picks))]
    oracle_err, random_err, oracle_cos, random_cos = [], [], [], []
    for t in oracle_picks:
        target = io.xy(dataset.dataset['observations'][t : t + horizon + 1])
        d_target = target[-1] - target[0]
        best, best_end = None, np.inf
        rz = int(rng_np.integers(num_skills))
        for z in range(num_skills):
            key, sub = jax.random.split(key)
            p_ = io.xy(execute_skill(env, io, act_fn,
                                     dataset.dataset['observations'][t], z, horizon, sub))
            e = float(np.linalg.norm(p_[-1] - target[-1]))
            if e < best_end:
                best_end, best = e, p_
            if z == rz:
                random_err.append(np.linalg.norm(p_ - target, axis=-1))
                random_cos.append(cosine(d_target, p_[-1] - p_[0]))
        oracle_err.append(np.linalg.norm(best - target, axis=-1))
        oracle_cos.append(cosine(d_target, best[-1] - best[0]))
    oracle_err = np.stack(oracle_err) if oracle_err else None
    random_err = np.stack(random_err) if random_err else None

    err = np.linalg.norm(io.xy(exec_paths) - io.xy(data_paths), axis=-1)  # [N, H+1]
    d_data = io.xy(data_paths)[:, -1] - io.xy(data_paths)[:, 0]
    d_exec = io.xy(exec_paths)[:, -1] - io.xy(exec_paths)[:, 0]
    travel = float(np.linalg.norm(d_data, axis=1).mean())  # how far a window moves at all
    cos = np.array([cosine(a, b) for a, b in zip(d_data, d_exec)])
    cos = cos[~np.isnan(cos)]

    fig = plt.figure(figsize=(19.0, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.0, 1.0, 0.75], wspace=0.30)

    # (a) Spatial overlay.
    ax = fig.add_subplot(gs[0, 0])
    maze.overlay(ax)
    show = min(45, len(data_paths))
    for i in range(show):
        d, e = io.xy(data_paths[i]), io.xy(exec_paths[i])
        ax.plot(d[:, 0], d[:, 1], color=C_DATA, lw=1.4, alpha=0.85, solid_capstyle='round', zorder=3)
        ax.plot(e[:, 0], e[:, 1], color=C_ASSIGNED, lw=1.4, alpha=0.85, solid_capstyle='round', zorder=4)
        ax.plot([d[0, 0]], [d[0, 1]], 'o', ms=2.4, color=INK, zorder=5)
    ax.set_title(f'{show} relabelled windows: data vs. its assigned skill, executed', fontsize=9, color=INK)
    ax.legend(handles=[
        Line2D([0], [0], color=C_DATA, lw=1.6, label='data window'),
        Line2D([0], [0], color=C_ASSIGNED, lw=1.6, label='assigned skill, executed from the same state'),
    ], loc='upper center', bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=8)

    # (b) Accumulation curve.
    ax = fig.add_subplot(gs[0, 1])
    steps = np.arange(horizon + 1)
    ax.plot(steps, err.mean(axis=0), color=C_ASSIGNED, lw=2.0, label='assigned skill')
    ax.fill_between(steps, np.percentile(err, 25, axis=0), np.percentile(err, 75, axis=0),
                    color=C_ASSIGNED, alpha=0.15, lw=0)
    if oracle_err is not None:
        ax.plot(steps, oracle_err.mean(axis=0), color=C_ORACLE, lw=2.0,
                label=f'best of all {num_skills} skills (oracle)')
    if random_err is not None:
        ax.plot(steps, random_err.mean(axis=0), color=C_RANDOM, lw=2.0, ls='--', label='random skill')
    ax.axhline(travel, color=INK_MUTED, lw=1.0, ls=':')
    ax.text(horizon, travel, "the window's own travel ", va='top', ha='right', fontsize=7, color=INK_MUTED)
    ax.set_xlabel('step within the H-step window', fontsize=8, color=INK_MUTED)
    ax.set_ylabel('position error vs. the data window (m)', fontsize=8, color=INK_MUTED)
    ax.set_title("the label's error compounds over the window", fontsize=9, color=INK)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    # (c) Direction agreement, against the same random-skill reference.
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-1, 1, 41)
    ax.hist(cos, bins=bins, color=C_ASSIGNED, alpha=0.85, density=True, label='assigned skill')
    if random_cos:
        rc = np.array(random_cos)
        ax.hist(rc[~np.isnan(rc)], bins=bins, histtype='step', lw=2.0, color=C_RANDOM,
                density=True, label='random skill')
    ax.axvline(0.0, color=INK_MUTED, lw=1.0, ls='--')
    ax.set_xlabel('cos(data displacement, executed displacement)', fontsize=8, color=INK_MUTED)
    ax.set_ylabel('density over windows', fontsize=8, color=INK_MUTED)
    ax.set_title(f'direction agreement   mean cos = {cos.mean():.2f}', fontsize=9, color=INK)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    # (d) Headline: is the label better than no label at all?
    ax = fig.add_subplot(gs[0, 3])
    names, vals, colors = ['assigned'], [float(err.mean(axis=0)[-1])], [C_ASSIGNED]
    if oracle_err is not None:
        names.append('oracle')
        vals.append(float(oracle_err.mean(axis=0)[-1]))
        colors.append(C_ORACLE)
    if random_err is not None:
        names.append('random')
        vals.append(float(random_err.mean(axis=0)[-1]))
        colors.append(C_RANDOM)
    ax.bar(names, vals, color=colors, width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8, color=INK)
    ax.axhline(travel, color=INK_MUTED, lw=1.0, ls=':')
    ax.text(len(vals) - 0.5, travel, f'window travel {travel:.2f} m', va='bottom', ha='right',
            fontsize=7, color=INK_MUTED)
    headroom = (vals[-1] - vals[0]) / max(vals[-1] - (vals[1] if len(vals) > 2 else 0.0), 1e-9)
    ax.set_ylabel(f'endpoint error after H={horizon} steps (m)', fontsize=8, color=INK_MUTED)
    ax.set_title(f'how much of the achievable\nlabel quality is realised: {headroom:.0%}',
                 fontsize=9, color=INK)
    style_axes(ax)

    fig.suptitle(title, fontsize=11, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')
    print(f'[offline] endpoint error at H: assigned {vals[0]:.3f} m'
          + (f' | oracle {vals[1]:.3f} m' if len(vals) > 1 else '')
          + (f' | random {vals[-1]:.3f} m' if len(vals) > 2 else '')
          + f' | mean window travel {travel:.3f} m | mean cos {cos.mean():.2f}')
    return dict(err_mean=err.mean(axis=0),
                oracle_mean=None if oracle_err is None else oracle_err.mean(axis=0),
                random_mean=None if random_err is None else random_err.mean(axis=0),
                cos_mean=float(cos.mean()), travel=travel)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def style_axes(ax):
    ax.grid(True, color=INK_MUTED, alpha=0.15, lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=7, colors=INK_MUTED, length=2)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 -- is the label informative, and is it self-consistent on-policy?
# ──────────────────────────────────────────────────────────────────────────────
def figure_labels(agent, io, dataset, labels, horizon, rollouts, seed,
                  num_windows, out_path, title):
    lab = Labeller(agent)
    num_skills = lab.num_skills
    rng_np = np.random.default_rng(seed + 1)
    starts = valid_window_starts(dataset, horizon)
    picks = rng_np.choice(starts, size=min(num_windows, len(starts)), replace=False)

    margins, step_agree = [], []
    if lab.kind == 'loglik':
        for t in picks:
            obs_win = dataset.get_observations(np.arange(t, t + horizon))
            act_win = dataset.dataset['actions'][t : t + horizon]
            ll = lab.step_logliks(obs_win, act_win)  # [H, K]
            total = ll.sum(axis=0)
            order = np.argsort(total)[::-1]
            margins.append((total[order[0]] - total[order[1]]) / horizon)
            step_agree.append((np.argmax(ll, axis=1) == order[0]).astype(np.float64))
    margins = np.asarray(margins)
    step_agree = np.stack(step_agree) if step_agree else None

    # On-policy self-consistency: relabel the chunks the controller itself executed.
    commanded, reinferred = [], []
    for traj in rollouts:
        n_steps = len(traj['actions'])
        for t in range(0, n_steps - horizon, horizon):
            commanded.append(int(traj['skills'][t]))
            reinferred.append(
                lab.label_chunk(traj['observations'][t : t + horizon], traj['actions'][t : t + horizon])
            )
    commanded = np.asarray(commanded)
    reinferred = np.asarray(reinferred)
    agree = float((commanded == reinferred).mean()) if commanded.size else np.nan

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6))

    ax = axes[0]
    counts = np.bincount(labels, minlength=num_skills).astype(np.float64)
    probs = counts / counts.sum()
    ax.bar(np.arange(num_skills), probs, color=C_DATA, width=0.85)
    nzp = probs[probs > 0]
    ax.set_xlabel('skill index', fontsize=8, color=INK_MUTED)
    ax.set_ylabel('share of relabelled windows', fontsize=8, color=INK_MUTED)
    ax.set_title(f'label distribution\nentropy {-(nzp * np.log(nzp)).sum():.2f} / {np.log(num_skills):.2f} nats · '
                 f'coverage {(counts > 0).mean():.0%} · top skill {probs.max():.0%}',
                 fontsize=9, color=INK)
    style_axes(ax)

    ax = axes[1]
    if margins.size:
        ax.hist(margins, bins=40, color=C_DATA, alpha=0.9)
        ax.set_title(f'per-step margin of the winning label\nmedian {np.median(margins):.3f} nats/step',
                     fontsize=9, color=INK)
    else:
        ax.text(0.5, 0.5, 'n/a for the VQ-encoder labeller', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color=INK_MUTED)
        ax.set_title('per-step margin of the winning label', fontsize=9, color=INK)
    ax.set_xlabel('(top1 − top2 window log-likelihood) / H', fontsize=8, color=INK_MUTED)
    ax.set_ylabel('windows', fontsize=8, color=INK_MUTED)
    style_axes(ax)

    ax = axes[2]
    if step_agree is not None:
        m = step_agree.mean(axis=0)
        ax.plot(np.arange(horizon), m, color=C_ASSIGNED, lw=2.0, marker='o', ms=4)
        ax.axhline(1.0 / num_skills, color=INK_MUTED, lw=1.0, ls='--')
        ax.text(horizon - 1, 1.0 / num_skills, ' chance', va='bottom', ha='right',
                fontsize=7, color=INK_MUTED)
        ax.set_ylim(0, 1)
        ax.set_title(f'per-step agreement with the window label\nmean {m.mean():.2f}', fontsize=9, color=INK)
    else:
        ax.text(0.5, 0.5, 'n/a for the VQ-encoder labeller', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color=INK_MUTED)
        ax.set_title('per-step agreement with the window label', fontsize=9, color=INK)
    ax.set_xlabel('step within the window', fontsize=8, color=INK_MUTED)
    ax.set_ylabel('fraction of windows', fontsize=8, color=INK_MUTED)
    style_axes(ax)

    ax = axes[3]
    if commanded.size:
        conf = np.zeros((num_skills, num_skills))
        for c, r in zip(commanded, reinferred):
            conf[c, r] += 1
        conf = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(conf, cmap='Blues', vmin=0, vmax=1, origin='lower', aspect='auto')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7, colors=INK_MUTED)
        ax.set_xlabel('skill re-inferred from the executed chunk', fontsize=8, color=INK_MUTED)
        ax.set_ylabel('skill the controller commanded', fontsize=8, color=INK_MUTED)
        ax.set_title(f'on-policy self-consistency\nagreement {agree:.0%} (chance {1 / num_skills:.0%})',
                     fontsize=9, color=INK)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK_MUTED, length=2)

    fig.suptitle(title, fontsize=11, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')
    print(f'[labels] on-policy self-consistency {agree:.2f}, '
          f'median per-step margin {np.median(margins) if margins.size else float("nan"):.4f} nats/step')
    return dict(agreement=agree, margins=margins, step_agree=step_agree)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--controller_dir', required=True, help='trained high-level controller run dir')
    p.add_argument('--epoch', type=int, default=None)
    p.add_argument('--out_dir', default=None, help='defaults to the controller run dir')
    p.add_argument('--tasks', default=None, help='comma-separated task ids (default: all)')
    p.add_argument('--episodes', type=int, default=1, help='rollouts per task')
    p.add_argument('--num_windows', type=int, default=300, help='dataset windows re-executed')
    p.add_argument('--num_oracle', type=int, default=50, help='windows swept over all K skills')
    p.add_argument('--num_margin_windows', type=int, default=2000, help='windows scored for the label margin')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_steps', type=int, default=None, help="episode cap (default: the env's own)")
    p.add_argument('--force_relabel', action='store_true')
    p.add_argument('--figures', default='rollouts,offline,labels')
    args = p.parse_args()

    matches = sorted(glob.glob(args.controller_dir))
    assert len(matches) == 1, f'--controller_dir matched {len(matches)} dirs: {matches}'
    run_dir = matches[0].rstrip('/')
    epoch = args.epoch if args.epoch is not None else latest_epoch(run_dir)
    out_dir = args.out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)

    agent, env, config, dataset, env_name = build(run_dir, epoch)
    horizon = int(agent.config['chunk_horizon'])
    io = EnvIO(env)
    maze = Maze(env)
    # Capture the registered episode length BEFORE lifting the TimeLimit: `execute_skill`
    # needs it lifted, but the eval rollouts must still stop where training's eval did.
    max_steps = args.max_steps or env_horizon(env) or 1000
    raise_time_limit(env, 10_000_000)
    env.reset()

    task_infos = env.unwrapped.task_infos
    tasks = [int(t) for t in args.tasks.split(',')] if args.tasks else list(range(1, len(task_infos) + 1))

    labels = dataset_labels(agent, dataset, os.path.join(out_dir, f'chunk_labels_e{epoch}.npy'),
                            force=args.force_relabel)

    tag = f'{config["agent_name"]} · {env_name} · epoch {epoch} · H={horizon} · K={agent.config["num_skills"]}'
    wanted = set(args.figures.split(','))

    rollouts = None
    if {'rollouts', 'labels'} & wanted:
        rollouts = collect_rollouts(agent, env, tasks, args.episodes, args.seed, max_steps)

    if 'offline' in wanted:
        figure_offline(
            agent, env, io, maze, dataset, labels, horizon, args.num_windows, args.num_oracle, args.seed,
            os.path.join(out_dir, f'relabel_offline_fidelity_e{epoch}.png'),
            f'Does executing the assigned skill reproduce the window it was assigned to?\n{tag}',
        )
    if 'rollouts' in wanted:
        figure_rollouts(
            io, maze, dataset, labels, horizon, tasks, rollouts,
            os.path.join(out_dir, f'relabel_rollout_paths_e{epoch}.png'),
            f'What the controller walks vs. what its training data credits to the skills it picks\n{tag}',
        )
    if 'labels' in wanted:
        figure_labels(
            agent, io, dataset, labels, horizon, rollouts, args.seed,
            args.num_margin_windows,
            os.path.join(out_dir, f'relabel_labels_e{epoch}.png'),
            f'Is the chunk label informative, and does it survive its own execution?\n{tag}',
        )


if __name__ == '__main__':
    main()
