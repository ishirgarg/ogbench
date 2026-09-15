"""Step-by-step visualisation of the value-greedy skill selector, for debugging.

`eval_skill_value_policy.py` reports one number: the success rate of the
hierarchical policy that, every `--skill_horizon` env steps, reselects

    z*(s, g) = argmax_z V(s, z, g)

and executes pi(. | ., z*) until the next decision. When that number is low
(0.036 on antmaze-medium-stitch with 50 skills) the number itself says nothing
about *which* half is broken -- the selector picking a bad skill, or the
low-level policy not going where the value function says that skill goes.

(Not to be confused with `agents/skill_value_controller.py`, which amortises that
same argmax into a learned pi_hi(z | s, g). This script visualises the online
selector, and in fact rejects the controller agent, which deliberately does not
expose `skill_values`.)

This script opens up a single episode of that selector and draws, per decision:

  * the skill z* it chose;
  * the **value map** under that skill: s and z* are held fixed at the decision
    state, and V(s, z*, g) is evaluated over goals g spanning the maze, so the
    map answers "where does the value function believe this skill takes the ant
    from here?";
  * the `skill_horizon` steps the low-level policy then actually walked.

Reading the panels: the true task goal (red star) should sit on a bright region
of the map -- that is why the selector chose this skill -- and the walked segment
(white line) should head toward the map's own peak (magenta x). The two failure
modes separate cleanly:

  * peak far from the goal  ->  the *selector* is the problem: no skill's value
    map actually covers the goal, so the argmax is a coin flip among near-ties
    (check the value-margin panel in the overview figure);
  * peak near the goal but the segment walks elsewhere  ->  the *low-level
    policy* is the problem: pi(. | ., z) does not realise the transition its own
    V head predicts.

Goal candidates (`--goal_source`):

  * `dataset` (default): real observations sampled from the offline dataset,
    binned by their xy into a `--grid_res` square grid over the maze and averaged.
    V's psi tower sees the whole goal (`_extract_future(g)` -- every dimension, on a
    run that leaves `obs_indices` unset, as all the state envs here do), not just xy,
    so this maps V over the goal manifold the value function was actually fit on.
  * `grid`: the episode's own goal observation with its xy overwritten on a
    regular grid. Sharper, but every non-xy dim is held at one arbitrary pose,
    which is off-manifold for most cells.

Usage:

    python value_viz/plot_skill_value_selector.py --run_dir ckpts/empowerment/.../sd000_... --task_id 1

Writes, under `<run_dir>/value_viz/` with the basename below (`--output_prefix`
replaces the whole prefix, directory included, and its directory is created if
missing):

    skill_value_debug_e<epoch>_task<t>_h<H>.png            per-decision panels
    ..._overview.png    episode path, per-skill value table, margins, locomotion
    ..._atlas.png       one state, the top-ranked skills side by side -- with
                        `--atlas_rollout_steps > 0` (the default) each also carries
                        the short rollout pi(. | ., z) walks from that state, and the
                        first cell overlays every candidate skill's walk.
                        Omitted entirely under `--atlas_skills 0`.
    ..._stats.json      the per-decision numbers behind the figures
"""

import os

# Set before any mujoco import. Nothing here renders, but `make_env_and_datasets`
# constructs the same env training used, which initialises GL on import.
os.environ.setdefault('MUJOCO_GL', 'egl')

import sys

# This file lives in `impls/value_viz/`, but the project modules it imports
# (`agents`, `utils`, `eval_skill_policy`) sit one level up in `impls/`, which is
# not on `sys.path` when a script is run by path. Put it there, so
# `python value_viz/<script>.py` works from any cwd the way the flat layout did.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from agents import agents as agent_registry
from eval_skill_policy import eval_horizon, latest_epoch, load_agent, load_flags
from utils.evaluation import raise_time_limit
from utils.skill_graph import REQUIRED_HOOKS


# Hooks this script calls on the agent, checked before the multi-minute env and
# dataset load. `skill_set`/`skill_values`/`sample_actions_with_skill` drive the
# rollout (the same three `eval_skill_value_policy.py` requires); `REQUIRED_HOOKS`
# and `skill_values_cross` drive the value maps.
NEEDED_HOOKS = ('skill_set', 'skill_values', 'sample_actions_with_skill',
                'skill_values_cross', *REQUIRED_HOOKS)


# ── Maze geometry ────────────────────────────────────────────────────────────


class Maze:
    """The occupancy grid of an OGBench maze, in world (x, y) coordinates.

    `maze_map[i, j] == 1` is a wall; its centre is at
    (j * unit - offset_x, i * unit - offset_y), which is the convention
    `ogbench/locomaze/maze.py` places the wall geoms with and the one
    `plot_empowerment_map_antmaze.py` draws them with.
    """

    def __init__(self, env):
        base = env.unwrapped
        self.map = getattr(base, 'maze_map', None)
        if self.map is None:
            raise SystemExit(
                'env.unwrapped has no `maze_map`; this script plots V over a maze floor '
                'and only supports maze-shaped envs (antmaze / pointmaze; the '
                'teleport variants are rejected separately).'
            )
        self.unit = float(getattr(base, '_maze_unit', 4.0))
        self.offx = float(getattr(base, '_offset_x', 4.0))
        self.offy = float(getattr(base, '_offset_y', 4.0))
        rows, cols = self.map.shape
        half = self.unit / 2.0
        # The plotted extent covers the walkable cells plus the wall ring around them.
        self.extent = (
            -self.offx - half, (cols - 1) * self.unit - self.offx + half,
            -self.offy - half, (rows - 1) * self.unit - self.offy + half,
        )

    def overlay(self, ax, alpha=0.35):
        rows, cols = self.map.shape
        for i in range(rows):
            for j in range(cols):
                if self.map[i, j] == 1:
                    ax.add_patch(Rectangle(
                        (j * self.unit - self.offx - self.unit / 2.0,
                         i * self.unit - self.offy - self.unit / 2.0),
                        self.unit, self.unit,
                        facecolor='0.15', edgecolor='0.15', linewidth=0.3, alpha=alpha,
                        zorder=4,
                    ))


# ── Value maps ───────────────────────────────────────────────────────────────


def dataset_goal_candidates(train_dataset, num_goals, rng):
    """A random subset of dataset observations, used as candidate goals."""
    obs = train_dataset['observations']
    n = obs.shape[0]
    idx = rng.choice(n, size=min(num_goals, n), replace=False)
    # Sorted so the fancy-index gather walks the million-row observation array
    # forwards rather than jumping; the candidates' own order is irrelevant downstream.
    idx.sort()
    return np.asarray(obs[idx], dtype=np.float32)


def grid_goal_candidates(goal_template, maze, grid_res):
    """`goal_template` repeated over a regular xy grid, with dims 0/1 overwritten."""
    x_lo, x_hi, y_lo, y_hi = maze.extent
    xs = np.linspace(x_lo, x_hi, grid_res, dtype=np.float32)
    ys = np.linspace(y_lo, y_hi, grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    goals = np.repeat(np.asarray(goal_template, dtype=np.float32)[None, :], xx.size, axis=0)
    goals[:, 0] = xx.reshape(-1)
    goals[:, 1] = yy.reshape(-1)
    return goals


def bin_edges(maze, grid_res):
    x_lo, x_hi, y_lo, y_hi = maze.extent
    return np.linspace(x_lo, x_hi, grid_res + 1), np.linspace(y_lo, y_hi, grid_res + 1)


def bin_values(goal_xy, values, maze, grid_res):
    """Mean of `values` per spatial bin  ->  [grid_res, grid_res] with NaN holes.

    Returned in `imshow(origin='lower', extent=maze.extent)` orientation:
    `np.histogram2d(x, y)` indexes [x_bin, y_bin], and imshow wants [row=y, col=x],
    hence the transpose.
    """
    x_edges, y_edges = bin_edges(maze, grid_res)
    total, _, _ = np.histogram2d(goal_xy[:, 0], goal_xy[:, 1], bins=[x_edges, y_edges],
                                 weights=values)
    count, _, _ = np.histogram2d(goal_xy[:, 0], goal_xy[:, 1], bins=[x_edges, y_edges])
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean.T


def map_peak_xy(value_map, maze, grid_res):
    """Centre of the brightest bin of a binned value map, or None if it is empty.

    Deliberately the peak of the *drawn* map rather than of the raw candidate
    values: with dataset goals a bin averages ~10 states, and a single
    off-manifold candidate can out-score every bin mean. `|peak - goal|` is one of
    the two headline diagnostics, so it has to describe what the panel shows.
    """
    if not np.isfinite(value_map).any():
        return None
    row, col = np.unravel_index(np.nanargmax(value_map), value_map.shape)
    x_edges, y_edges = bin_edges(maze, grid_res)
    return np.array([(x_edges[col] + x_edges[col + 1]) / 2.0,
                     (y_edges[row] + y_edges[row + 1]) / 2.0], dtype=np.float32)


# ── Rollout ──────────────────────────────────────────────────────────────────


def rollout_value_selected(agent, env, config, task_id, skill_horizon, max_steps,
                           eval_temperature, eval_gaussian, candidates, seed,
                           torso_z_index=None):
    """One episode of the value-greedy hierarchical policy, fully instrumented.

    The *policy* is `utils.evaluation.evaluate_value_selected_skill`'s, decision for
    decision: the same greedy argmax over `candidates` every `skill_horizon` steps,
    the same deterministic value call, the same action post-processing. The
    differences are that it records the per-decision state and the full value vector,
    that it runs one episode rather than a batch, and that `max_steps` can cut the
    episode short (`success` is then reported as None, not 0).

    The *episode* is a matching draw, not necessarily the identical one: seeding mirrors
    the eval (an unseeded probe reset first, then `reset(seed=...)`, then the action-space
    stream, and `np.random.seed` after `load_agent` as there), but ogbench's
    initial-position noise comes from the global numpy RNG, which in the eval has also
    been advanced by every preceding task. So task 1 should line up and later tasks will
    not; `eval_skill_value_policy.py` documents the same caveat for its own cells.
    """
    skills = jnp.asarray(agent.skill_set())

    env.action_space.seed(int(seed) % (2 ** 31))
    # The eval probes the env once before its seeded episode 0, and each ogbench maze
    # reset burns a fixed run of `action_space.sample()` draws on stabilizing steps
    # (5 for ant and point, 40 for humanoid), so skipping this probe would put the whole
    # episode on a different action-space stream.
    env.reset(options=dict(task_id=task_id, render_goal=False))
    observation, info = env.reset(seed=int(seed) % (2 ** 31),
                                  options=dict(task_id=task_id, render_goal=False))
    if info.get('goal') is None:
        raise SystemExit(
            f'env.reset(task_id={task_id}) returned no `goal` in info; the value-based '
            f'skill selector needs the goal observation to score skills.'
        )
    goal = np.asarray(info['goal'], dtype=np.float32)

    rng = jax.random.PRNGKey(int(seed))
    base = env.unwrapped

    decisions = []          # one entry per skill (re)selection
    xy_traj = [np.asarray(base.get_xy(), dtype=np.float32)]
    done, step, zi = False, 0, None
    while not done and step < max_steps:
        if step % skill_horizon == 0:
            values = np.asarray(agent.skill_values(observations=jnp.asarray(observation),
                                                   goals=jnp.asarray(goal)))
            zi = int(candidates[int(values[candidates].argmax())])
            decisions.append(dict(
                step=step,
                observation=np.asarray(observation, dtype=np.float32),
                xy=np.asarray(base.get_xy(), dtype=np.float32),
                values=values,
                skill=zi,
                segment_start=len(xy_traj) - 1,
                # The full simulator state, not just the observation: the atlas replays
                # the other skills from this exact pose, and `set_state(qpos, qvel)` is
                # the only way to put the ant back (`set_xy` would keep the current
                # joint angles and velocities, which are what the skills act on).
                qpos=np.asarray(base.data.qpos, dtype=np.float64).copy(),
                qvel=np.asarray(base.data.qvel, dtype=np.float64).copy(),
            ))

        rng, key = jax.random.split(rng)
        action = np.asarray(agent.sample_actions_with_skill(
            observations=jnp.asarray(observation), skills=skills[zi],
            seed=key, temperature=eval_temperature,
        ))
        if not config.get('discrete'):
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
            action = np.clip(action, -1, 1)

        observation, _, terminated, truncated, info = env.step(action)
        observation = np.asarray(observation, dtype=np.float32)
        xy_traj.append(np.asarray(base.get_xy(), dtype=np.float32))
        done = terminated or truncated
        step += 1

    # Per-decision locomotion summary, over the segment each committed skill ran. Net
    # displacement vs. path length separates a skill that travels from one that circles
    # in place. `torso_z` (ant only, see `torso_z_index`) is the torso height: it drops
    # and stays down when the ant flips, which is the difference between "chose a bad
    # skill" and "the low-level policy stopped walking at all".
    xy_arr = np.stack(xy_traj, axis=0)
    for d in decisions:
        seg = xy_arr[d['segment_start']: d['segment_start'] + skill_horizon + 1]
        d['displacement'] = float(np.linalg.norm(seg[-1] - seg[0]))
        d['path_length'] = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
        d['torso_z'] = (None if torso_z_index is None
                        else float(d['observation'][torso_z_index]))

    return dict(
        decisions=decisions,
        xy_traj=xy_arr,
        goal=goal,
        goal_xy=np.asarray(goal[:2], dtype=np.float32),
        skill_horizon=skill_horizon,
        candidates=np.asarray(candidates),
        episode_length=step,
        # None, not 0, when --max_steps cut the episode short: the episode never got
        # the chance to succeed, and a 0 here would read as a genuine failure.
        success=(None if not done else float(info.get('success', float('nan')))),
    )


def rollout_skills_from_state(env, agent, decision, skill_ids, n_steps, config,
                              task_id, eval_temperature, eval_gaussian, seed):
    """Replay `n_steps` of pi(. | ., z) from one decision state, for each skill.

    Every skill starts from the identical simulator state -- the decision's own
    `qpos`/`qvel`, restored with `set_state` -- so the paths differ only in z. That is
    what makes them comparable, and what makes the panel a direct read on the question
    the value maps only answer indirectly: do the skills actually go anywhere
    different from here?

    Returns `{skill_id: xy path [n_steps + 1, 2]}`. Mutates the env, so call it after
    the main rollout is finished.
    """
    skills = jnp.asarray(agent.skill_set())
    base = env.unwrapped
    paths = {}
    for z in skill_ids:
        # reset() first: `set_state` does not touch the TimeLimit wrapper's elapsed-step
        # counter, so without this the replay would inherit the main episode's step count
        # and truncate early (immediately, when that episode ran to the limit). It also
        # re-draws an init pose and a goal, but `set_state` overwrites the pose on the
        # next line and pi(. | ., z) is goal-agnostic, so neither reaches the replay --
        # `task_id` is passed only because ogbench's reset rejects a None one.
        env.reset(options=dict(task_id=task_id, render_goal=False))
        base.set_state(decision['qpos'], decision['qvel'])
        # The observation recorded at the decision, not a fresh `get_ob()`: it is by
        # construction the one the wrapper chain produced, so nothing here depends on
        # which wrappers rewrite observations.
        observation = decision['observation'].copy()
        rng = jax.random.PRNGKey(int(seed) + int(z))
        xy = [np.asarray(base.get_xy(), dtype=np.float32)]
        for _ in range(n_steps):
            rng, key = jax.random.split(rng)
            action = np.asarray(agent.sample_actions_with_skill(
                observations=jnp.asarray(observation), skills=skills[int(z)],
                seed=key, temperature=eval_temperature,
            ))
            # Same post-processing as the main rollout: with --eval_gaussian set, a
            # noiseless replay would not be the policy whose decisions it explains.
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)
            observation, _, terminated, truncated, _ = env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            xy.append(np.asarray(base.get_xy(), dtype=np.float32))
            if terminated or truncated:
                break
        paths[int(z)] = np.stack(xy, axis=0)
    return paths


# ── Figures ──────────────────────────────────────────────────────────────────


def _success_label(rollout):
    return 'truncated by --max_steps' if rollout['success'] is None \
        else f'success={rollout["success"]:.0f}'


def plot_decision_panels(decisions, rollout, maze, out_path, title,
                         zoom_halfwidth=2.5, n_cols=6):
    """One panel per decision: value map under the chosen skill + what was walked.

    A skill commitment moves the ant well under a metre, which is invisible against a
    maze tens of metres across, so every panel carries a zoomed inset (a
    `zoom_halfwidth` box around the decision state) where the segment is legible.
    """
    skill_horizon = rollout['skill_horizon']
    n = len(decisions)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    x_lo, x_hi, y_lo, y_hi = maze.extent
    goal_xy = rollout['goal_xy']
    xy_traj = rollout['xy_traj']

    for ax in axes.reshape(-1):
        ax.set_visible(False)

    for panel, d in enumerate(decisions):
        ax = axes[panel // n_cols][panel % n_cols]
        ax.set_visible(True)
        vmap = d['value_map']
        finite = vmap[np.isfinite(vmap)]
        # Robust limits: a handful of far-off-manifold cells otherwise flatten the
        # whole map into one colour.
        vmin, vmax = (np.percentile(finite, [2, 100]) if finite.size else (0.0, 1.0))
        im = ax.imshow(vmap, origin='lower', extent=(x_lo, x_hi, y_lo, y_hi),
                       cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest',
                       zorder=1)
        maze.overlay(ax, alpha=0.25)

        # The `skill_horizon` steps actually walked under this skill.
        seg = xy_traj[d['segment_start']: d['segment_start'] + skill_horizon + 1]
        ax.plot(seg[:, 0], seg[:, 1], color='black', linewidth=3.0, zorder=5,
                solid_capstyle='round')
        ax.plot(seg[:, 0], seg[:, 1], color='white', linewidth=1.8, zorder=6,
                solid_capstyle='round')
        ax.scatter([d['xy'][0]], [d['xy'][1]], c='white', s=45, marker='o',
                   edgecolors='black', linewidths=1.0, zorder=7)
        if d['peak_xy'] is not None:
            ax.scatter([d['peak_xy'][0]], [d['peak_xy'][1]], c='magenta', s=90,
                       marker='x', linewidths=2.2, zorder=8)
        ax.scatter([goal_xy[0]], [goal_xy[1]], c='red', s=150, marker='*',
                   edgecolors='white', linewidths=0.7, zorder=8)

        # Zoomed inset: same value map, cropped to a box around s_t.
        # `inset_axes` defaults to zorder 5, under the peak marker and goal star the
        # parent draws at 8 -- which would render them through the inset box.
        inset = ax.inset_axes([0.635, 0.025, 0.34, 0.34], zorder=9)
        inset.imshow(vmap, origin='lower', extent=(x_lo, x_hi, y_lo, y_hi),
                     cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest',
                     zorder=1)
        inset.plot(seg[:, 0], seg[:, 1], color='black', linewidth=3.0, zorder=5,
                   solid_capstyle='round')
        inset.plot(seg[:, 0], seg[:, 1], color='white', linewidth=1.6, zorder=6,
                   solid_capstyle='round')
        # A frozen ant makes seg[-1] == seg[-2]; a zero-length FancyArrow warns and
        # draws nothing, so only annotate once there is a direction to point in.
        if np.linalg.norm(seg[-1] - seg[-2]) > 1e-6:
            inset.annotate(
                '', xy=(seg[-1, 0], seg[-1, 1]), xytext=(seg[-2, 0], seg[-2, 1]),
                arrowprops=dict(arrowstyle='-|>', color='white', linewidth=1.6,
                                shrinkA=0, shrinkB=0), zorder=7,
            )
        inset.scatter([d['xy'][0]], [d['xy'][1]], c='white', s=28, marker='o',
                      edgecolors='black', linewidths=0.8, zorder=8)
        inset.set_xlim(d['xy'][0] - zoom_halfwidth, d['xy'][0] + zoom_halfwidth)
        inset.set_ylim(d['xy'][1] - zoom_halfwidth, d['xy'][1] + zoom_halfwidth)
        inset.set_aspect('equal')
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set(edgecolor='white', linewidth=1.2)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        peak_txt = ('n/a' if d['peak_goal_dist'] is None
                    else f"{d['peak_goal_dist']:.1f}m")
        ax.set_title(
            f"t={d['step']}   z*={d['skill']}   net {d['displacement']:.2f}m "
            f"of {d['path_length']:.2f}m\n"
            f"V(g)={d['goal_value']:.3f}   gap={d['gap']:.3f}   |peak-g|={peak_txt}",
            fontsize=8,
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', linestyle='', markersize=8,
                   label='state $s_t$ at the decision'),
        plt.Line2D([0], [0], color='0.4', linewidth=2.0,
                   label=f'{skill_horizon} steps walked under $z^*$ (white)'),
        plt.Line2D([0], [0], marker='x', color='magenta', linestyle='', markersize=9,
                   markeredgewidth=2.2, label='brightest bin of the map (peak)'),
        plt.Line2D([0], [0], color='0.4', linewidth=1.2,
                   label=f'inset: same map, zoomed +/-{zoom_halfwidth:g}m around $s_t$'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markeredgecolor='white', linestyle='', markersize=14,
                   label='task goal g'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.002))
    fig.suptitle(title, fontsize=11)
    # Reserve absolute strips for the suptitle and legend: as a fraction they would
    # shrink to nothing once many rows make the figure tall. h_pad keeps a row's
    # titles off the inset of the row above (tight_layout does not measure insets).
    legend_frac = min(0.09, 0.5 / fig.get_figheight())
    title_frac = min(0.09, 0.7 / fig.get_figheight())
    fig.tight_layout(rect=(0, legend_frac, 1, 1 - title_frac), h_pad=2.5)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_overview(rollout, maze, out_path, title):
    """Episode path, the full V(s_t, ., g) table, the value margins, and locomotion.

    Covers *every* decision in the episode, not just the ones the panel figure drew.
    """
    decisions = rollout['decisions']
    skill_horizon = rollout['skill_horizon']
    steps = np.array([d['step'] for d in decisions])
    chosen = np.array([d['skill'] for d in decisions])
    values = np.stack([d['values'] for d in decisions], axis=0)          # [T, K]
    goal_xy = rollout['goal_xy']
    xy_traj = rollout['xy_traj']

    fig = plt.figure(figsize=(24, 5.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.35, 1.15, 1.15], wspace=0.30)

    # (a) The whole episode, coloured by time.
    ax = fig.add_subplot(gs[0, 0])
    maze.overlay(ax, alpha=0.35)
    t = np.arange(len(xy_traj))
    ax.scatter(xy_traj[:, 0], xy_traj[:, 1], c=t, cmap='plasma', s=3, zorder=5)
    ax.scatter([xy_traj[0, 0]], [xy_traj[0, 1]], c='black', s=70, marker='o',
               edgecolors='white', linewidths=1.0, zorder=7, label='start')
    ax.scatter([goal_xy[0]], [goal_xy[1]], c='red', s=190, marker='*',
               edgecolors='white', linewidths=0.8, zorder=7, label='goal')
    dec_xy = np.stack([d['xy'] for d in decisions], axis=0)
    ax.scatter(dec_xy[:, 0], dec_xy[:, 1], facecolors='none', edgecolors='black',
               s=26, linewidths=0.7, zorder=6, label='skill reselections')
    x_lo, x_hi, y_lo, y_hi = maze.extent
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.set_title(f"episode path ({rollout['episode_length']} steps, "
                 f"{_success_label(rollout)})", fontsize=10)

    # (b) V(s_t, z, g) for every candidate skill at every decision -- shows how
    # near-tied the argmax is. Restricted to the candidates and normalised within them:
    # rows the argmax never ranked would otherwise set the scale the near-ties are read
    # against. Normalised per row so the within-decision ordering is what is visible.
    ax = fig.add_subplot(gs[0, 1])
    cand = rollout['candidates']
    cand_vals = values[:, cand]                                       # [T, n_cand]
    row_min = cand_vals.min(axis=1, keepdims=True)
    row_max = cand_vals.max(axis=1, keepdims=True)
    normed = (cand_vals - row_min) / np.maximum(row_max - row_min, 1e-8)
    n_cand = len(cand)
    im = ax.imshow(normed.T, origin='lower', aspect='auto', cmap='magma',
                   extent=(-0.5, len(decisions) - 0.5, -0.5, n_cand - 0.5))
    # Rows are positions in `cand`, which coincide with the skill ids only when every
    # skill is a candidate; map the chosen ids through, and relabel when it is a subset.
    position = {int(z): i for i, z in enumerate(cand)}
    ax.scatter(np.arange(len(decisions)), [position[int(z)] for z in chosen], s=10,
               facecolors='none', edgecolors='cyan', linewidths=0.8)
    if n_cand != values.shape[1]:
        ax.set_yticks(np.arange(n_cand))
        ax.set_yticklabels([str(int(z)) for z in cand], fontsize=6 if n_cand > 20 else 8)
    ax.set_xlabel(f'decision index (one per reselection, {skill_horizon} env steps apart)')
    ax.set_ylabel('skill z')
    scope = ('all skills' if n_cand == values.shape[1]
             else f'{n_cand} candidate skills of {values.shape[1]}')
    ax.set_title(f'V($s_t$, z, g), min-max normalised per decision ({scope})\n'
                 f'cyan = the chosen argmax', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # (c) How decisive the argmax is, against how far the ant still is.
    ax = fig.add_subplot(gs[0, 2])
    # Both margins over the *candidate* set the argmax actually ran over, so the two
    # traces describe the same decision (they diverged when --skills was passed).
    sorted_vals = np.sort(values[:, cand], axis=1)
    gap = np.array([d['gap'] for d in decisions])
    spread = sorted_vals[:, -1] - sorted_vals[:, 0]
    ax.plot(steps, gap, color='tab:red', linewidth=1.4, label='top1 - top2 (decisiveness)')
    ax.plot(steps, spread, color='tab:orange', linewidth=1.0, alpha=0.7,
            label='top1 - worst (total spread)')
    ax.set_xlabel('env step')
    ax.set_ylabel('value margin (nats)')
    ax.legend(loc='upper left', fontsize=7)

    ax2 = ax.twinx()
    dist = np.linalg.norm(dec_xy - goal_xy[None, :], axis=1)
    ax2.plot(steps, dist, color='tab:blue', linewidth=1.4, label='||xy - goal xy||')
    peak_dist = np.array([np.nan if d['peak_goal_dist'] is None else d['peak_goal_dist']
                          for d in decisions])
    ax2.plot(steps, peak_dist, color='tab:green', linewidth=1.2, linestyle='--',
             label='||value-map peak - goal||')
    ax2.set_ylabel('distance (m)')
    ax2.legend(loc='upper right', fontsize=7)
    ax.set_title('value margin vs. distance to goal', fontsize=10)

    # (d) Is the low-level policy even walking? A committed skill that moves the ant
    # ~0 m separates "the selector chose badly" from "pi(. | ., z) has stalled" -- the
    # ant flipping shows up as the torso height collapsing and never recovering.
    ax = fig.add_subplot(gs[0, 3])
    disp = np.array([d['displacement'] for d in decisions])
    path_len = np.array([d['path_length'] for d in decisions])
    ax.plot(steps, path_len, color='0.6', linewidth=1.0,
            label='path length walked per skill')
    ax.plot(steps, disp, color='tab:purple', linewidth=1.4,
            label='net displacement per skill')
    ax.axhline(0.05, color='tab:red', linestyle=':', linewidth=1.0,
               label='0.05 m (effectively frozen)')
    ax.set_xlabel('env step')
    ax.set_ylabel(f'metres per {skill_horizon}-step commitment')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=7)

    # Only ant envs put a torso height at a known observation index; see
    # `torso_z_index` in main. Elsewhere the trace is simply absent.
    if decisions[0]['torso_z'] is not None:
        ax2 = ax.twinx()
        ax2.plot(steps, np.array([d['torso_z'] for d in decisions]),
                 color='tab:brown', linewidth=1.2, alpha=0.8)
        ax2.set_ylabel('torso height (m)', color='tab:brown')
        ax2.tick_params(axis='y', labelcolor='tab:brown')
    ax.set_title(f'low-level locomotion  ({int((disp < 0.05).sum())}/{len(disp)} '
                 f'commitments moved < 0.05 m)', fontsize=10)

    fig.suptitle(title, fontsize=11)
    # No tight_layout: the twinx pairs in (c) and (d) are not compatible with it.
    fig.subplots_adjust(left=0.035, right=0.972, bottom=0.12, top=0.82)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _paths_halfwidth(paths, state_xy, minimum=1.5, margin=1.15):
    """A zoom box around `state_xy` that contains every path, with a little air."""
    if not paths:
        return minimum
    reach = max(float(np.abs(p - state_xy[None, :]).max()) for p in paths.values())
    return max(minimum, margin * reach)


def _draw_skill_paths_panel(ax, paths, state_xy, chosen, goal_xy, maze, halfwidth):
    """All skills' short rollouts from one state, chosen highlighted, zoomed in."""
    maze.overlay(ax, alpha=0.25)
    for z, path in paths.items():
        if z == chosen:
            continue
        ax.plot(path[:, 0], path[:, 1], color='0.55', linewidth=0.9, alpha=0.85, zorder=5)
        ax.scatter([path[-1, 0]], [path[-1, 1]], c='0.35', s=8, zorder=6)
    if chosen in paths:
        path = paths[chosen]
        ax.plot(path[:, 0], path[:, 1], color='red', linewidth=2.6, zorder=8,
                solid_capstyle='round')
        # Above the state marker (zorder 10), not under it: when the chosen skill does
        # not move, its path collapses onto s and the endpoint would otherwise be
        # hidden by exactly the marker it needs to be distinguished from -- and "the
        # chosen skill went nowhere" is the case these panels exist to show.
        ax.scatter([path[-1, 0]], [path[-1, 1]], c='red', s=55, marker='o',
                   edgecolors='white', linewidths=1.0, zorder=11)
    ax.scatter([state_xy[0]], [state_xy[1]], c='white', s=70, marker='o',
               edgecolors='black', linewidths=1.2, zorder=10)

    # The goal is almost always outside this zoom box, so point at it instead.
    to_goal = goal_xy - state_xy
    norm = float(np.linalg.norm(to_goal))
    if norm > 1e-6:
        d = to_goal / norm * (0.72 * halfwidth)
        ax.annotate('', xy=(state_xy[0] + d[0], state_xy[1] + d[1]),
                    xytext=(state_xy[0], state_xy[1]),
                    arrowprops=dict(arrowstyle='-|>', color='red', linewidth=1.6,
                                    linestyle=':', alpha=0.75), zorder=7)
        ax.text(state_xy[0] + d[0], state_xy[1] + d[1], f'  goal\n  {norm:.1f}m',
                color='red', fontsize=7, ha='left', va='center', zorder=11)

    ax.set_xlim(state_xy[0] - halfwidth, state_xy[0] + halfwidth)
    ax.set_ylim(state_xy[1] - halfwidth, state_xy[1] + halfwidth)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_skill_atlas(entries, state_xy, rollout, maze, out_path, title,
                     paths=None, rollout_steps=0, n_cols=6):
    """V(s, z, g) over goals for the top-ranked skills at one fixed state.

    The decision panels only ever show the winning skill, which cannot answer the
    prior question: does *any* skill's value map put its mass on the goal? Here s is
    held fixed at one decision state and z is swept over the skills the selector
    ranked highest, so a whole row of maps that all peak in the same place --
    typically right on top of s -- says the value function does not separate skills
    by where they go, and the argmax over z is then a near-tie among all of them.

    With `paths` (from `rollout_skills_from_state`) each panel also carries the
    `rollout_steps` steps that skill actually walks from s, and the first cell
    overlays every candidate skill's walk with the chosen one in red -- so the same
    figure shows what the value function *predicts* about each skill and where the
    policy *goes* under it, side by side.
    """
    show_paths = bool(paths)
    n = len(entries) + (1 if show_paths else 0)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.9 * n_rows),
                             squeeze=False)
    x_lo, x_hi, y_lo, y_hi = maze.extent
    goal_xy = rollout['goal_xy']
    chosen_skill = next((e['skill'] for e in entries if e['chosen']), None)
    halfwidth = _paths_halfwidth(paths or {}, state_xy)

    for ax in axes.reshape(-1):
        ax.set_visible(False)

    if show_paths:
        ax = axes[0][0]
        ax.set_visible(True)
        _draw_skill_paths_panel(ax, paths, state_xy, chosen_skill, goal_xy, maze, halfwidth)
        for spine in ax.spines.values():
            spine.set(edgecolor='red', linewidth=2.0)
        # An invisible colorbar reserves the same strip the map panels spend on theirs,
        # so this cell ends up the same width as the rest of the grid.
        spacer = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax,
                              fraction=0.046, pad=0.02)
        spacer.ax.set_visible(False)
        ax.set_title(f'ALL {len(paths)} skills, {rollout_steps} steps from $s_t$\n'
                     f'chosen z={chosen_skill} in red, others grey',
                     fontsize=8.5, color='darkred')

    for i, e in enumerate(entries):
        ax = axes[(i + show_paths) // n_cols][(i + show_paths) % n_cols]
        ax.set_visible(True)
        vmap = e['value_map']
        finite = vmap[np.isfinite(vmap)]
        vmin, vmax = (np.percentile(finite, [2, 100]) if finite.size else (0.0, 1.0))
        im = ax.imshow(vmap, origin='lower', extent=(x_lo, x_hi, y_lo, y_hi),
                       cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest',
                       zorder=1)
        maze.overlay(ax, alpha=0.25)
        ax.scatter([state_xy[0]], [state_xy[1]], c='white', s=45, marker='o',
                   edgecolors='black', linewidths=1.0, zorder=7)
        if e['peak_xy'] is not None:
            ax.scatter([e['peak_xy'][0]], [e['peak_xy'][1]], c='magenta', s=90,
                       marker='x', linewidths=2.2, zorder=8)
        ax.scatter([goal_xy[0]], [goal_xy[1]], c='red', s=150, marker='*',
                   edgecolors='white', linewidths=0.7, zorder=8)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # This skill's own walk from s, inset at the shared zoom of the overlay panel
        # so the panels are directly comparable to each other and to that cell.
        net = ''
        if show_paths and e['skill'] in paths:
            path = paths[e['skill']]
            inset = ax.inset_axes([0.58, 0.02, 0.40, 0.40], zorder=9)
            inset.imshow(vmap, origin='lower', extent=(x_lo, x_hi, y_lo, y_hi),
                         cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest',
                         zorder=1)
            line_color = 'red' if e['chosen'] else 'white'
            # Black casing under the line: white vanishes over viridis's bright end and
            # red over its dark end, so neither colour survives on its own.
            inset.plot(path[:, 0], path[:, 1], color='black', linewidth=3.2, zorder=5,
                       solid_capstyle='round')
            inset.plot(path[:, 0], path[:, 1], color=line_color, linewidth=1.8, zorder=6,
                       solid_capstyle='round')
            inset.scatter([path[-1, 0]], [path[-1, 1]], c=line_color, s=22, zorder=7,
                          edgecolors='black', linewidths=0.5)
            inset.scatter([state_xy[0]], [state_xy[1]], c='white', s=26, marker='o',
                          edgecolors='black', linewidths=0.8, zorder=8)
            inset.set_xlim(state_xy[0] - halfwidth, state_xy[0] + halfwidth)
            inset.set_ylim(state_xy[1] - halfwidth, state_xy[1] + halfwidth)
            inset.set_aspect('equal')
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set(edgecolor=line_color, linewidth=1.2)
            net = f"   walk {float(np.linalg.norm(path[-1] - path[0])):.2f}m"

        peak_txt = 'n/a' if e['peak_goal_dist'] is None else f"{e['peak_goal_dist']:.1f}m"
        # `chosen` compares skill ids rather than trusting rank 0: `argmax` (the
        # rollout) takes the first maximal index while `argsort` (the ranking here)
        # takes the last, so on an exact tie the two disagree about which skill is
        # rank 1 -- and near-ties are the failure mode this whole figure is about.
        if e['chosen']:
            for spine in ax.spines.values():
                spine.set(edgecolor='red', linewidth=2.5)
        ax.set_title(f"rank {e['rank'] + 1}{' (CHOSEN)' if e['chosen'] else ''}:  "
                     f"z={e['skill']}{net}\n"
                     f"V(g)={e['goal_value']:.3f}   |peak-g|={peak_txt}", fontsize=8,
                     color='darkred' if e['chosen'] else 'black',
                     fontweight='bold' if e['chosen'] else 'normal')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=11)
    title_frac = min(0.09, 0.8 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - title_frac), h_pad=2.5)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True,
                   help='empowerment_skill run dir (holds flags.json / params_*.pkl).')
    p.add_argument('--epoch', type=int, default=None, help='Checkpoint epoch (default: latest).')
    p.add_argument('--task_id', type=int, default=1,
                   help='OGBench task id, i.e. which (start, goal) pair to run (1-indexed).')
    p.add_argument('--skill_horizon', type=int, default=10,
                   help='Env steps a chosen skill is held for (default: 10, matching '
                        'eval_skill_value_policy.py).')
    p.add_argument('--skills', type=str, default=None,
                   help='Comma-separated skill indices the selector may choose from '
                        '(default: all). Mirrors eval_skill_value_policy.py --skills; pass '
                        'the same subset the low number came from.')
    p.add_argument('--num_panels', type=int, default=24,
                   help='Number of decisions to draw a value map for (default: 24).')
    p.add_argument('--panel_stride', type=int, default=1,
                   help='Draw every Nth decision (default: 1, i.e. the first --num_panels '
                        'consecutive decisions from the start of the episode).')
    p.add_argument('--max_steps', type=int, default=None,
                   help='Env steps to roll out (default: the env horizon). Below the '
                        'horizon this truncates the episode, and `success` is then reported '
                        'as truncated rather than as a failure; above it, the TimeLimit is '
                        'lifted and the episode runs longer than the benchmark allows.')
    p.add_argument('--goal_source', type=str, default='dataset', choices=['dataset', 'grid'],
                   help="Candidate goals for the value map: real dataset states ('dataset', "
                        "default) or the episode goal with its xy overwritten on a regular "
                        "grid ('grid').")
    p.add_argument('--num_goal_states', type=int, default=30000,
                   help='Dataset states to use as candidate goals (--goal_source dataset).')
    p.add_argument('--grid_res', type=int, default=48,
                   help='Spatial bins per axis for the value map.')
    p.add_argument('--atlas_skills', type=int, default=12,
                   help='Skills to draw a value map for in the atlas figure, taken in '
                        'descending V(s, z, g) order at the --atlas_decision state '
                        '(0 disables the figure; default: 12).')
    p.add_argument('--atlas_decision', type=int, default=0,
                   help='Which decision index the atlas holds s fixed at (default: 0, the '
                        'episode start).')
    p.add_argument('--atlas_rollout_steps', type=int, default=30,
                   help='Env steps to replay under *every* candidate skill from the atlas '
                        'state, drawn per panel and overlaid in the atlas figure\'s first '
                        'cell with the chosen skill in red (0 disables; default: 30).')
    p.add_argument('--zoom_halfwidth', type=float, default=2.5,
                   help='Half-width (m) of the per-panel zoom inset around the decision '
                        'state, where the walked segment is legible (default: 2.5).')
    p.add_argument('--eval_temperature', type=float, default=0.0,
                   help='Low-level actor temperature (default: 0, as in eval).')
    p.add_argument('--eval_gaussian', type=float, default=None,
                   help='Action Gaussian noise, as in eval (default: none).')
    p.add_argument('--eval_on_cpu', type=int, default=1,
                   help='Run the agent on CPU (default: 1, matching '
                        'eval_skill_value_policy.py). Not just about speed: on GPU the '
                        'policy forward pass is not bit-reproducible, and ant dynamics '
                        'amplify a 1e-7 action difference into a different episode within '
                        'a few hundred steps, so the same --seed would not redraw the '
                        'same figure.')
    p.add_argument('--seed', type=int, default=0, help='Seed for the env and action sampling.')
    p.add_argument('--dataset_path', type=str, default=None,
                   help='Override the dataset path recorded in flags.json.')
    p.add_argument('--output_prefix', type=str, default=None,
                   help='Output path prefix (default: <run_dir>/value_viz/skill_value_debug_e<epoch>_'
                        'task<t>_h<H>).')
    args = p.parse_args()

    # Everything checkable without the env, checked here: the env build and dataset load
    # cost minutes, and the rollout that follows costs more.
    for name, lo in (('num_panels', 1), ('panel_stride', 1), ('skill_horizon', 1),
                     ('num_goal_states', 1), ('grid_res', 2)):
        if getattr(args, name) < lo:
            raise SystemExit(f'--{name} must be at least {lo}, got {getattr(args, name)}.')
    for name in ('atlas_skills', 'atlas_rollout_steps', 'atlas_decision'):
        if getattr(args, name) < 0:
            raise SystemExit(f'--{name} must be >= 0, got {getattr(args, name)}.')
    if args.zoom_halfwidth <= 0:
        raise SystemExit(f'--zoom_halfwidth must be > 0, got {args.zoom_halfwidth}.')
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit(f'--max_steps must be at least 1, got {args.max_steps}.')

    # Parsed here; the range and duplicate checks need `num_skills` and stay in main.
    args.skill_ids = None
    if args.skills is not None:
        try:
            args.skill_ids = [int(z) for z in args.skills.split(',') if z.strip()]
        except ValueError:
            raise SystemExit(f'--skills must be comma-separated integers, got "{args.skills}".')
        if not args.skill_ids:
            raise SystemExit('--skills selected no skills.')
    return args


def main():
    args = parse_args()

    run_dir = args.run_dir.rstrip('/')
    epoch = args.epoch if args.epoch is not None else latest_epoch(run_dir)
    saved = load_flags(run_dir)
    agent_name, env_name = saved['agent']['agent_name'], saved['env_name']

    # Everything checkable before the minutes spent building the env and loading the
    # dataset is checked here.
    missing = [h for h in NEEDED_HOOKS if not hasattr(agent_registry[agent_name], h)]
    if missing:
        raise SystemExit(
            f'Agent "{agent_name}" cannot be visualised here: it is missing '
            f'{", ".join("`" + h + "`" for h in missing)}. This script needs '
            f'both the value-greedy selector hooks and the cached-goal-embedding hooks the '
            f'value maps are built from (see agents/empowerment_skill.py).'
        )
    if saved['agent'].get('encoder') is not None:
        raise SystemExit(
            f'run uses encoder={saved["agent"]["encoder"]!r} (image observations). Every '
            f'figure here is indexed by the xy of a state, which pixel observations do not '
            f'expose, so there is nothing to plot against.'
        )
    if saved['agent'].get('frame_stack') is not None:
        # Same reason eval_skill_plan.py refuses these: `make_env_and_datasets` stacks
        # the env but not the dataset (stacking lives in GCDataset, which `load_agent`
        # never builds), so dataset goals and env observations would have different
        # widths -- and `grid` goals would overwrite only the oldest frame's xy.
        raise SystemExit(
            f'run uses frame_stack={saved["agent"]["frame_stack"]}, which `load_agent` does '
            f'not apply to the dataset; candidate goals would not match the observations '
            f'the env returns.'
        )

    # Both, deliberately -- they are not redundant. `jax_platform_name` sets the default
    # placement for new arrays, but the checkpoint restore still lands the agent's params
    # on GPU here, and without the `device_put` below every rollout step runs there:
    # three of these in parallel then collide over one device's preallocated pool and die
    # with RESOURCE_EXHAUSTED, on a node whose GPUs are otherwise idle.
    if args.eval_on_cpu:
        jax.config.update('jax_platform_name', 'cpu')

    agent, env, config, train_dataset = load_agent(run_dir, epoch, saved,
                                                   dataset_path=args.dataset_path)
    if args.eval_on_cpu:
        agent = jax.device_put(agent, device=jax.devices('cpu')[0])

    # ogbench's initial-position noise is drawn from the *global* numpy RNG, so it has
    # to be seeded by the caller. Seeded *after* `load_agent`, exactly as
    # eval_skill_value_policy.py does: `load_agent` samples an example batch, which
    # itself consumes global draws, so seeding before it would land the episode on a
    # different init pose than the eval's.
    np.random.seed(args.seed)

    if getattr(env.unwrapped, '_teleport_info', None) is not None:
        raise SystemExit(
            f'{env_name} has teleporters, which `step` applies as a ~20 m single-step '
            f'`set_xy` jump. That jump lands in the xy trajectory, so per-commitment '
            f'displacement and path length (and panel (d)\'s "effectively frozen" '
            f'threshold) stop meaning anything, the shared atlas zoom blows out to the '
            f'maze scale, and the destination is drawn from the global numpy RNG inside '
            f'`step` -- so the per-skill replays would no longer be matched draws. See '
            f'plot_empowerment_map_pointmaze_teleport.py for the teleport-aware drawing.'
        )
    if hasattr(env.unwrapped, 'get_agent_ball_xy'):
        raise SystemExit(
            f'{env_name} is a ball env: success is measured on the ball, while `get_xy` and '
            f'`goal[:2]` here are the agent\'s. Every panel would plot the wrong body. See '
            f'plot_empowerment_map_antsoccer.py for the two-body treatment.'
        )
    maze = Maze(env)

    # Only the ant mazes put a torso height at a fixed observation index (obs = qpos
    # then qvel, so obs[2] is the torso z). Pointmaze observations are 2-D -- obs[2]
    # would be an IndexError -- and humanoidmaze's obs[2] is an abdomen joint angle,
    # not a height. Elsewhere the locomotion panel simply omits the trace.
    torso_z_index = 2 if 'antmaze' in env_name else None

    horizon = eval_horizon(env)
    if args.max_steps is not None:
        max_steps = args.max_steps
        if horizon is not None and max_steps > horizon:
            raise_time_limit(env, max_steps)
    elif horizon is None:
        raise SystemExit(f'{env_name} registers no episode horizon; pass --max_steps.')
    else:
        max_steps = horizon

    num_skills = int(np.asarray(agent.skill_set()).shape[0])
    candidates = np.arange(num_skills)
    if args.skill_ids is not None:
        candidates = np.array(args.skill_ids)
        if candidates.min() < 0 or candidates.max() >= num_skills:
            raise SystemExit(f'--skills must be indices in [0, {num_skills}).')
        if len(set(candidates.tolist())) != candidates.size:
            raise SystemExit('--skills contains duplicate indices.')

    task_infos = (env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos')
                  else env.task_infos)
    if not 1 <= args.task_id <= len(task_infos):
        raise SystemExit(f'--task_id must be in [1, {len(task_infos)}] for {env_name}.')
    task = task_infos[args.task_id - 1]

    print(f'[{env_name}] agent={agent_name} epoch={epoch} skills={num_skills} '
          f'candidates={candidates.size} task={task["task_name"]} '
          f'start_xy={task["init_xy"]} goal_xy={task["goal_xy"]} '
          f'skill_horizon={args.skill_horizon} max_steps={max_steps}', flush=True)

    # The per-task seed formula eval_skill_value_policy.py uses, so a --seed here lands
    # on the same draw its table would (up to the global-RNG caveat in the rollout docstring).
    rollout = rollout_value_selected(
        agent=agent, env=env, config=config, task_id=args.task_id,
        skill_horizon=args.skill_horizon, max_steps=max_steps,
        eval_temperature=args.eval_temperature, eval_gaussian=args.eval_gaussian,
        candidates=candidates, torso_z_index=torso_z_index,
        seed=args.seed * 1_000_003 + (args.task_id - 1) * 1_009,
    )
    all_decisions = rollout['decisions']
    print(f'  rolled out {rollout["episode_length"]} steps, {len(all_decisions)} decisions, '
          f'{_success_label(rollout)}', flush=True)
    # Checked here rather than at the atlas, which is drawn after both other figures.
    if args.atlas_skills > 0 and args.atlas_decision >= len(all_decisions):
        raise SystemExit(
            f'--atlas_decision must be < the {len(all_decisions)} decisions this episode '
            f'made, got {args.atlas_decision}.'
        )

    # ── Candidate goals + their psi embeddings (computed once, reused per panel) ──
    rng = np.random.default_rng(args.seed)
    if args.goal_source == 'dataset':
        goal_states = dataset_goal_candidates(train_dataset, args.num_goal_states, rng)
    else:
        goal_states = grid_goal_candidates(rollout['goal'], maze, args.grid_res)
    goal_xy_all = goal_states[:, :2].copy()
    # The panel maps come from the bilinear expansion; the rollout's own argmax came
    # from `skill_values`. Confirm they agree before paying for 30k goal embeddings --
    # `skill_values_cross` embeds its own goal, so it needs nothing from them.
    probe = np.asarray(agent.skill_values_cross(
        jnp.asarray(all_decisions[0]['observation'])[None, :],
        jnp.asarray(rollout['goal'])[None, :],
    ))[0, 0]
    max_dev = float(np.abs(probe - all_decisions[0]['values']).max())
    if max_dev > 1e-3:
        raise SystemExit(
            f'skill_values_cross disagrees with skill_values by {max_dev:.3g}; the value maps '
            f'would not be the quantity the selector maximises.'
        )
    print(f'  cross-form check: max |skill_values_cross - skill_values| = {max_dev:.3g}',
          flush=True)

    print(f'  embedding {len(goal_states)} candidate goals ({args.goal_source})...', flush=True)
    goal_embeddings = agent.value_goal_embeddings(jnp.asarray(goal_states))

    def value_map_for(observation, skill):
        vals = np.asarray(agent.skill_values_from_goal_embeddings(
            jnp.asarray(observation)[None, :], goal_embeddings,
        ))[0, :, skill]                                             # [G]
        vmap = bin_values(goal_xy_all, vals, maze, args.grid_res)
        peak = map_peak_xy(vmap, maze, args.grid_res)
        dist = None if peak is None else float(np.linalg.norm(peak - rollout['goal_xy']))
        return vmap, peak, dist

    # ── Per-decision value maps ──────────────────────────────────────────────
    picked = all_decisions[::args.panel_stride][:args.num_panels]
    print(f'  computing {len(picked)} value maps...', flush=True)
    for d in picked:
        d['value_map'], d['peak_xy'], d['peak_goal_dist'] = \
            value_map_for(d['observation'], d['skill'])

    # The overview's traces cover every decision, so the peak distance and the margin
    # have to exist for all of them.
    if len(all_decisions) > len(picked):
        print(f'  computing peak distances for the remaining '
              f'{len(all_decisions) - len(picked)} decisions...', flush=True)
    for d in all_decisions:
        if 'peak_goal_dist' not in d:
            _, d['peak_xy'], d['peak_goal_dist'] = value_map_for(d['observation'], d['skill'])
        d['goal_value'] = float(d['values'][d['skill']])
        # Margin over the *candidate set*, which is what the argmax ran over. With a
        # single candidate the selector has no choice to be decisive about.
        cand_vals = np.sort(d['values'][candidates])
        d['gap'] = float(cand_vals[-1] - cand_vals[-2]) if cand_vals.size > 1 else 0.0

    # ── Draw ─────────────────────────────────────────────────────────────────
    # Figures land in a subfolder of the run dir, not loose beside the
    # checkpoints: a sweep over tasks and horizons puts a dozen files here, and
    # `latest_epoch` / the eval JSONs share the directory.
    prefix = args.output_prefix or os.path.join(
        run_dir, 'value_viz',
        f'skill_value_debug_e{epoch}_task{args.task_id}_h{args.skill_horizon}'
    )
    # Also covers an --output_prefix pointing into a directory that does not
    # exist yet, which is the usual way to collect figures from several runs.
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    common = (f'{env_name} | {os.path.basename(run_dir)} | epoch {epoch} | '
              f'{task["task_name"]}: start {tuple(task["init_xy"])} -> '
              f'goal {tuple(task["goal_xy"])} | '
              f'z* = argmax_z V(s, z, g) every {args.skill_horizon} steps')

    panels_out = f'{prefix}.png'
    plot_decision_panels(
        picked, rollout, maze, panels_out, zoom_halfwidth=args.zoom_halfwidth,
        title=(f'{common}\nvalue map = V(s_t, z*, g) over goals g '
               f'({args.goal_source} goals, {args.grid_res}x{args.grid_res} bins); '
               f'each panel fixes s_t and z*'),
    )
    print(f'Saved decision panels: {panels_out}')

    overview_out = f'{prefix}_overview.png'
    plot_overview(rollout, maze, overview_out, title=common)
    print(f'Saved overview: {overview_out}')

    if args.atlas_skills > 0:
        d0 = all_decisions[args.atlas_decision]
        n_atlas = min(args.atlas_skills, candidates.size)
        print(f'  computing the {n_atlas}-skill atlas at decision {args.atlas_decision} '
              f'(t={d0["step"]})...', flush=True)
        # One call gives every skill's values at every candidate goal, so the atlas
        # costs no more than a single decision panel.
        all_vals = np.asarray(agent.skill_values_from_goal_embeddings(
            jnp.asarray(d0['observation'])[None, :], goal_embeddings,
        ))[0]                                                       # [G, K]
        ranked = candidates[np.argsort(d0['values'][candidates])[::-1][:n_atlas]]
        entries = []
        for rank, z in enumerate(ranked):
            vmap = bin_values(goal_xy_all, all_vals[:, z], maze, args.grid_res)
            peak = map_peak_xy(vmap, maze, args.grid_res)
            entries.append(dict(
                skill=int(z), rank=rank, chosen=int(z) == d0['skill'],
                goal_value=float(d0['values'][z]), value_map=vmap, peak_xy=peak,
                peak_goal_dist=(None if peak is None
                                else float(np.linalg.norm(peak - rollout['goal_xy']))),
            ))
        # Rolled out last, because it moves the env off the episode's final state.
        paths = None
        if args.atlas_rollout_steps > 0:
            print(f'  replaying {args.atlas_rollout_steps} steps under each of the '
                  f'{candidates.size} candidate skills from t={d0["step"]}...', flush=True)
            paths = rollout_skills_from_state(
                env, agent, d0, candidates.tolist(), args.atlas_rollout_steps,
                config, task_id=args.task_id, eval_temperature=args.eval_temperature,
                eval_gaussian=args.eval_gaussian, seed=args.seed,
            )

        atlas_out = f'{prefix}_atlas.png'
        path_note = ('' if not paths else
                     f'; each panel and the first cell also show the '
                     f'{args.atlas_rollout_steps} steps pi(. | ., z) actually walks from s')
        plot_skill_atlas(
            entries, d0['xy'], rollout, maze, atlas_out, paths=paths,
            rollout_steps=args.atlas_rollout_steps,
            title=(f'{common}\natlas: s fixed at the decision at t={d0["step"]}, z swept '
                   f'over the {n_atlas} highest-valued skills -- V(s, z, g) over goals g '
                   f'({args.goal_source} goals){path_note}'),
        )
        print(f'Saved skill atlas: {atlas_out}')

    stats_out = f'{prefix}_stats.json'
    with open(stats_out, 'w') as f:
        json.dump(dict(
            run_dir=run_dir, epoch=epoch, env_name=env_name, agent_name=agent_name,
            task_id=args.task_id, task_name=task['task_name'],
            init_xy=list(task['init_xy']), goal_xy=list(task['goal_xy']),
            skill_horizon=args.skill_horizon, seed=args.seed,
            candidate_skills=candidates.tolist(),
            goal_source=args.goal_source, num_goal_candidates=int(len(goal_states)),
            episode_length=rollout['episode_length'],
            # null when --max_steps truncated the episode before it could resolve.
            success=rollout['success'],
            steps=[int(d['step']) for d in all_decisions],
            chosen_skills=[int(d['skill']) for d in all_decisions],
            goal_values=[d['goal_value'] for d in all_decisions],
            value_gaps=[d['gap'] for d in all_decisions],
            peak_goal_dist=[d['peak_goal_dist'] for d in all_decisions],
            displacement=[d['displacement'] for d in all_decisions],
            path_length=[d['path_length'] for d in all_decisions],
            torso_z=[d['torso_z'] for d in all_decisions],
        ), f, indent=2)
    print(f'Saved stats: {stats_out}')


if __name__ == '__main__':
    main()
