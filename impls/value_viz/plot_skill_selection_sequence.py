"""Decision-by-decision filmstrip of the value-greedy skill selector.

`plot_skill_value_selector.py` answers "what does one decision look like?" -- it
draws value maps, a per-skill atlas and aggregate traces, and its panels are laid
out for reading a single map closely. This script answers the neighbouring
question: **at which decision did the selector first pick the wrong skill?**

So it draws exactly one thing, in order:

    decision #0, #1, #2, ...   value map V(s_t, z*_t, g) over goals g, under the
                               skill z* that was actually chosen at that decision,
                               with the path walked so far and the segment walked next.

and nothing else per panel. The panels are a filmstrip -- the episode's own
trajectory accumulates across them (grey), so a panel shows where in the episode
it sits without cross-referencing an overview figure.

Two additions make "which one went wrong" scannable rather than something you
squint for:

  * every panel's frame is coloured by the *outcome* of its commitment -- green
    when those `skill_horizon` steps closed distance to the goal, red when they
    opened it, grey when neither (< --progress_eps m). The first red frame in a
    run of greens is the decision to look at.
  * a timeline strip under the grid plots distance-to-goal per decision with the
    same colours, over *every* decision in the episode (not just the drawn ones),
    with the drawn panels marked. Use it to pick a `--start_decision` and redraw.

Both are hindsight measures of the *committed segment*, so they conflate the two
failure modes `plot_skill_value_selector.py` separates (a bad argmax vs. a
low-level policy that does not follow its own value function). That is deliberate
-- this figure localises the step, and that script diagnoses it once you have one.

A note on reading the maps, carried over: V here is a learned discounted
state-occupancy density, not a distance-to-go, so its argmax over g (drawn as the
magenta x when --show_peak is on, off by default) sits near s by construction and
is a weak signal. What the *selector* used is the single value at the red star,
compared across skills -- that is `V(g)` and `gap` in each panel title.

Usage:

    python value_viz/plot_skill_selection_sequence.py --run_dir ckpts/empowerment/.../sd000_... --task_id 1

Writes, under `<run_dir>/value_viz/` (`--output_prefix` replaces the whole prefix,
directory included, and its directory is created if missing):

    skill_sequence_e<epoch>_task<t>_h<H>.png         the filmstrip + timeline
    skill_sequence_e<epoch>_task<t>_h<H>_stats.json  the per-decision numbers
"""

import os

# Set before any mujoco import, for the same reason plot_skill_value_selector.py does:
# `make_env_and_datasets` builds the training env, which initialises GL on import.
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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from agents import agents as agent_registry
from eval_skill_policy import eval_horizon, latest_epoch, load_agent, load_flags
from utils.evaluation import raise_time_limit
# Every piece of machinery here -- maze geometry, goal candidates, the binned value
# map, and above all the instrumented rollout -- is shared with the debug script, so
# the two figures describe the same policy rather than two re-implementations of it.
from plot_skill_value_selector import (
    NEEDED_HOOKS,
    Maze,
    _success_label,
    bin_values,
    dataset_goal_candidates,
    grid_goal_candidates,
    map_peak_xy,
    rollout_value_selected,
)


# Frame / marker colours for the hindsight outcome of a commitment.
OUTCOME_COLORS = {'closer': '#2ca02c', 'farther': '#d62728', 'flat': '#8c8c8c'}


def classify_progress(delta, eps):
    """`delta` = distance to goal at the segment's end minus at its start."""
    if delta < -eps:
        return 'closer'
    if delta > eps:
        return 'farther'
    return 'flat'


# ── Figure ───────────────────────────────────────────────────────────────────


def plot_sequence(picked, rollout, maze, out_path, title, n_cols=6,
                  zoom_halfwidth=2.5, show_peak=False, shared_scale=False,
                  progress_eps=0.05):
    """The filmstrip: one value map per drawn decision, plus the timeline strip."""
    decisions = rollout['decisions']
    skill_horizon = rollout['skill_horizon']
    goal_xy = rollout['goal_xy']
    xy_traj = rollout['xy_traj']
    x_lo, x_hi, y_lo, y_hi = maze.extent

    n = len(picked)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    # One shared colour scale makes brightness comparable across panels; per-panel
    # scales make each map's own structure legible. Neither dominates, so both exist.
    shared_lims = None
    if shared_scale:
        pool = np.concatenate([d['value_map'][np.isfinite(d['value_map'])].reshape(-1)
                               for d in picked])
        if pool.size:
            shared_lims = tuple(np.percentile(pool, [2, 100]))

    fig_h = 3.9 * n_rows + 3.3
    fig = plt.figure(figsize=(3.3 * n_cols, fig_h))
    # Laid out explicitly rather than with `tight_layout`, which cannot measure the
    # per-panel zoom insets (they live outside the gridspec) and warns about them.
    # The suptitle and the legend get absolute strips, so they do not shrink to
    # nothing once many panel rows make the figure tall.
    # Absolute inches, not fractions: the suptitle is two lines and each panel
    # carries a three-line title drawn *above* its cell, so the top strip has to hold
    # both; the bottom strip holds the timeline's x-label and the two-row legend.
    title_h, legend_h = 1.25, 1.2
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols, figure=fig,
        height_ratios=[3.9] * n_rows + [2.0],
        left=0.035, right=0.985,
        top=1.0 - title_h / fig_h, bottom=legend_h / fig_h,
        hspace=0.42, wspace=0.14,
    )

    for panel, d in enumerate(picked):
        ax = fig.add_subplot(gs[panel // n_cols, panel % n_cols])
        vmap = d['value_map']
        finite = vmap[np.isfinite(vmap)]
        if shared_lims is not None:
            vmin, vmax = shared_lims
        else:
            vmin, vmax = (np.percentile(finite, [2, 100]) if finite.size else (0.0, 1.0))
        im = ax.imshow(vmap, origin='lower', extent=(x_lo, x_hi, y_lo, y_hi),
                       cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest',
                       zorder=1)
        maze.overlay(ax, alpha=0.25)

        # The filmstrip's continuity: everything walked before this decision, faded.
        past = xy_traj[: d['segment_start'] + 1]
        if len(past) > 1:
            ax.plot(past[:, 0], past[:, 1], color='0.8', linewidth=1.6, alpha=0.95,
                    zorder=4, solid_capstyle='round')

        # The `skill_horizon` steps this commitment then walked.
        seg = xy_traj[d['segment_start']: d['segment_start'] + skill_horizon + 1]
        ax.plot(seg[:, 0], seg[:, 1], color='black', linewidth=3.0, zorder=5,
                solid_capstyle='round')
        ax.plot(seg[:, 0], seg[:, 1], color='white', linewidth=1.8, zorder=6,
                solid_capstyle='round')
        ax.scatter([d['xy'][0]], [d['xy'][1]], c='white', s=45, marker='o',
                   edgecolors='black', linewidths=1.0, zorder=7)
        if show_peak and d['peak_xy'] is not None:
            ax.scatter([d['peak_xy'][0]], [d['peak_xy'][1]], c='magenta', s=90,
                       marker='x', linewidths=2.2, zorder=8)
        ax.scatter([goal_xy[0]], [goal_xy[1]], c='red', s=150, marker='*',
                   edgecolors='white', linewidths=0.7, zorder=8)

        # A commitment moves the ant well under a metre; at maze scale that segment is
        # a dot, so it gets a zoom box. `inset_axes` defaults to zorder 5, under the
        # goal star the parent draws at 8, which would show through the inset.
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
        if len(seg) > 1 and np.linalg.norm(seg[-1] - seg[-2]) > 1e-6:
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

        # The frame is the scannable signal: colour by what the commitment achieved.
        color = OUTCOME_COLORS[d['progress_class']]
        for spine in ax.spines.values():
            spine.set(edgecolor=color, linewidth=3.0)

        ax.set_title(
            f"#{d['index']}   t={d['step']}   z*={d['skill']}\n"
            f"d(goal) {d['dist_start']:.2f} -> {d['dist_end']:.2f} m "
            f"({d['progress']:+.2f})\n"
            f"V(g)={d['goal_value']:.3f}   gap={d['gap']:.3f}",
            fontsize=8, color=color if d['progress_class'] != 'flat' else 'black',
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)

    # ── Timeline: every decision in the episode, not only the drawn ones ──────
    ax = fig.add_subplot(gs[n_rows, :])
    idx = np.array([d['index'] for d in decisions])
    dist = np.array([d['dist_start'] for d in decisions])
    ax.plot(idx, dist, color='0.5', linewidth=1.2, zorder=2)
    ax.scatter(idx, dist, s=34, zorder=3,
               c=[OUTCOME_COLORS[d['progress_class']] for d in decisions])
    # The drawn panels, so the strip doubles as an index into the grid above.
    drawn = np.array([d['index'] for d in picked])
    ax.scatter(drawn, dist[np.isin(idx, drawn)], s=150, facecolors='none',
               edgecolors='black', linewidths=1.0, zorder=4,
               label='drawn above')
    for d in picked:
        ax.annotate(f"#{d['index']}", xy=(d['index'], d['dist_start']),
                    xytext=(0, 9), textcoords='offset points', ha='center',
                    fontsize=6, color='0.25')
    # Chosen skill per decision, on a twin axis: a selector stuck on one z shows up
    # as a flat line, and a thrashing one as a scatter of noise.
    ax2 = ax.twinx()
    ax2.scatter(idx, [d['skill'] for d in decisions], s=9, color='tab:blue',
                alpha=0.6, zorder=2)
    ax2.set_ylabel('chosen skill $z^*$', fontsize=9, color='tab:blue')
    ax2.tick_params(axis='y', labelsize=8, colors='tab:blue')
    ax.set_xlabel('decision index', fontsize=9)
    ax.set_ylabel('$\\|xy_t - g_{xy}\\|$ at the decision (m)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, zorder=1)
    ax.set_title('every decision in the episode; marker colour = did that '
                 'commitment close distance to the goal?', fontsize=9)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', linestyle='', markersize=8,
                   label='state $s_t$ at the decision'),
        plt.Line2D([0], [0], color='0.4', linewidth=2.0,
                   label=f'{skill_horizon} steps walked under $z^*$ (white)'),
        plt.Line2D([0], [0], color='0.75', linewidth=1.5, label='episode path so far'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markeredgecolor='white', linestyle='', markersize=14,
                   label='task goal g'),
        plt.Line2D([0], [0], color=OUTCOME_COLORS['closer'], linewidth=3.0,
                   label='frame: commitment closed distance'),
        plt.Line2D([0], [0], color=OUTCOME_COLORS['farther'], linewidth=3.0,
                   label='frame: commitment lost distance'),
        plt.Line2D([0], [0], color=OUTCOME_COLORS['flat'], linewidth=3.0,
                   label=f'frame: no net progress (<{progress_eps:g} m)'),
    ]
    if show_peak:
        handles.insert(3, plt.Line2D([0], [0], marker='x', color='magenta',
                                     linestyle='', markersize=9, markeredgewidth=2.2,
                                     label='brightest bin of the map'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle(title, fontsize=11, y=1.0 - 0.16 / fig_h, va='top')
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
                        '(default: all). Mirrors eval_skill_value_policy.py --skills.')
    p.add_argument('--num_panels', type=int, default=18,
                   help='Decisions to draw a value map for (default: 18).')
    p.add_argument('--start_decision', type=int, default=0,
                   help='First decision index to draw (default: 0, the episode start). '
                        'Read the timeline strip, then re-run pointed at the decision '
                        'where distance to the goal started growing.')
    p.add_argument('--panel_stride', type=int, default=1,
                   help='Draw every Nth decision from --start_decision (default: 1, i.e. '
                        'consecutive decisions -- the filmstrip reading).')
    p.add_argument('--n_cols', type=int, default=6, help='Panels per row (default: 6).')
    p.add_argument('--max_steps', type=int, default=None,
                   help='Env steps to roll out (default: the env horizon). Below the '
                        'horizon this truncates the episode; above it, the TimeLimit is '
                        'lifted and the episode runs longer than the benchmark allows.')
    p.add_argument('--goal_source', type=str, default='dataset', choices=['dataset', 'grid'],
                   help="Candidate goals for the value map: real dataset states ('dataset', "
                        "default) or the episode goal with its xy overwritten on a regular "
                        "grid ('grid').")
    p.add_argument('--num_goal_states', type=int, default=30000,
                   help='Dataset states to use as candidate goals (--goal_source dataset).')
    p.add_argument('--grid_res', type=int, default=48,
                   help='Spatial bins per axis for the value map.')
    p.add_argument('--zoom_halfwidth', type=float, default=2.5,
                   help='Half-width (m) of the per-panel zoom inset around the decision '
                        'state (default: 2.5).')
    p.add_argument('--progress_eps', type=float, default=0.05,
                   help='Distance change (m) below which a commitment counts as no net '
                        'progress and its frame is drawn grey (default: 0.05).')
    p.add_argument('--show_peak', action='store_true',
                   help="Draw the map's brightest bin (magenta x). Off by default: V is an "
                        'occupancy density, so its argmax over g sits near s for every '
                        'skill and reads as a diagnostic when it is not one.')
    p.add_argument('--shared_scale', action='store_true',
                   help='One colour scale across all panels (default: per-panel robust '
                        'limits, which show each map\'s own structure).')
    p.add_argument('--eval_temperature', type=float, default=0.0,
                   help='Low-level actor temperature (default: 0, as in eval).')
    p.add_argument('--eval_gaussian', type=float, default=None,
                   help='Action Gaussian noise, as in eval (default: none).')
    p.add_argument('--eval_on_cpu', type=int, default=1,
                   help='Run the agent on CPU (default: 1, matching '
                        'eval_skill_value_policy.py). Also what makes a --seed redraw the '
                        'same figure: on GPU the policy forward pass is not '
                        'bit-reproducible, and ant dynamics amplify that within a few '
                        'hundred steps.')
    p.add_argument('--seed', type=int, default=0, help='Seed for the env and action sampling.')
    p.add_argument('--dataset_path', type=str, default=None,
                   help='Override the dataset path recorded in flags.json.')
    p.add_argument('--output_prefix', type=str, default=None,
                   help='Output path prefix (default: <run_dir>/value_viz/skill_sequence_e<epoch>_'
                        'task<t>_h<H>).')
    args = p.parse_args()

    # Everything checkable without the env, checked here: the env build and dataset load
    # cost minutes, and the rollout that follows costs more.
    for name, lo in (('num_panels', 1), ('panel_stride', 1), ('skill_horizon', 1),
                     ('num_goal_states', 1), ('grid_res', 2), ('n_cols', 1)):
        if getattr(args, name) < lo:
            raise SystemExit(f'--{name} must be at least {lo}, got {getattr(args, name)}.')
    if args.start_decision < 0:
        raise SystemExit(f'--start_decision must be >= 0, got {args.start_decision}.')
    if args.zoom_halfwidth <= 0:
        raise SystemExit(f'--zoom_halfwidth must be > 0, got {args.zoom_halfwidth}.')
    if args.progress_eps < 0:
        raise SystemExit(f'--progress_eps must be >= 0, got {args.progress_eps}.')
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
    # dataset is checked here. The rejections mirror plot_skill_value_selector.py's,
    # for the same reasons: this figure is indexed by the xy of a state throughout.
    missing = [h for h in NEEDED_HOOKS if not hasattr(agent_registry[agent_name], h)]
    if missing:
        raise SystemExit(
            f'Agent "{agent_name}" cannot be visualised here: it is missing '
            f'{", ".join("`" + h + "`" for h in missing)}. This script needs both the '
            f'value-greedy selector hooks and the cached-goal-embedding hooks the value '
            f'maps are built from (see agents/empowerment_skill.py).'
        )
    if saved['agent'].get('encoder') is not None:
        raise SystemExit(
            f'run uses encoder={saved["agent"]["encoder"]!r} (image observations). Every '
            f'panel is indexed by the xy of a state, which pixel observations do not '
            f'expose, so there is nothing to plot against.'
        )
    if saved['agent'].get('frame_stack') is not None:
        raise SystemExit(
            f'run uses frame_stack={saved["agent"]["frame_stack"]}, which `load_agent` does '
            f'not apply to the dataset; candidate goals would not match the observations '
            f'the env returns.'
        )

    # Both, deliberately: `jax_platform_name` sets the default placement for new arrays,
    # but the checkpoint restore still lands the params on GPU, and without the
    # `device_put` every rollout step would run there.
    if args.eval_on_cpu:
        jax.config.update('jax_platform_name', 'cpu')

    agent, env, config, train_dataset = load_agent(run_dir, epoch, saved,
                                                   dataset_path=args.dataset_path)
    if args.eval_on_cpu:
        agent = jax.device_put(agent, device=jax.devices('cpu')[0])

    # ogbench's initial-position noise is drawn from the *global* numpy RNG, so it has to
    # be seeded by the caller -- and after `load_agent`, exactly as
    # eval_skill_value_policy.py does, since `load_agent` samples an example batch and
    # would otherwise leave the episode on a different init pose than the eval's.
    np.random.seed(args.seed)

    if getattr(env.unwrapped, '_teleport_info', None) is not None:
        raise SystemExit(
            f'{env_name} has teleporters, which `step` applies as a ~20 m single-step '
            f'`set_xy` jump. That jump lands in the xy trajectory, so the per-commitment '
            f'distance-to-goal deltas this figure colours its frames by would not '
            f'describe anything the policy did.'
        )
    if hasattr(env.unwrapped, 'get_agent_ball_xy'):
        raise SystemExit(
            f'{env_name} is a ball env: success is measured on the ball, while `get_xy` and '
            f'`goal[:2]` here are the agent\'s. Every panel would plot the wrong body, and '
            f'the progress colouring would score the wrong distance. See '
            f'plot_empowerment_map_antsoccer.py for the two-body treatment.'
        )
    maze = Maze(env)

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

    # The per-task seed formula eval_skill_value_policy.py uses, so a --seed here lands on
    # the same draw its table would (up to the global-RNG caveat in the rollout docstring).
    rollout = rollout_value_selected(
        agent=agent, env=env, config=config, task_id=args.task_id,
        skill_horizon=args.skill_horizon, max_steps=max_steps,
        eval_temperature=args.eval_temperature, eval_gaussian=args.eval_gaussian,
        candidates=candidates, torso_z_index=None,
        seed=args.seed * 1_000_003 + (args.task_id - 1) * 1_009,
    )
    decisions = rollout['decisions']
    print(f'  rolled out {rollout["episode_length"]} steps, {len(decisions)} decisions, '
          f'{_success_label(rollout)}', flush=True)
    if args.start_decision >= len(decisions):
        raise SystemExit(
            f'--start_decision must be < the {len(decisions)} decisions this episode '
            f'made, got {args.start_decision}.'
        )

    # ── Per-decision scalars: the selector's own numbers, and the hindsight outcome ──
    xy_traj = rollout['xy_traj']
    goal_xy = rollout['goal_xy']
    for i, d in enumerate(decisions):
        d['index'] = i
        d['goal_value'] = float(d['values'][d['skill']])
        # Margin over the *candidate set*, which is what the argmax ran over. With a
        # single candidate the selector has no choice to be decisive about.
        cand_vals = np.sort(d['values'][candidates])
        d['gap'] = float(cand_vals[-1] - cand_vals[-2]) if cand_vals.size > 1 else 0.0
        seg = xy_traj[d['segment_start']: d['segment_start'] + rollout['skill_horizon'] + 1]
        d['dist_start'] = float(np.linalg.norm(seg[0] - goal_xy))
        d['dist_end'] = float(np.linalg.norm(seg[-1] - goal_xy))
        d['progress'] = d['dist_end'] - d['dist_start']
        d['progress_class'] = classify_progress(d['progress'], args.progress_eps)

    # ── Candidate goals + their psi embeddings (computed once, reused per panel) ──
    rng = np.random.default_rng(args.seed)
    if args.goal_source == 'dataset':
        goal_states = dataset_goal_candidates(train_dataset, args.num_goal_states, rng)
    else:
        goal_states = grid_goal_candidates(rollout['goal'], maze, args.grid_res)
    goal_xy_all = goal_states[:, :2].copy()
    # The panel maps come from the bilinear expansion; the rollout's own argmax came from
    # `skill_values`. Confirm they agree before paying for 30k goal embeddings.
    probe = np.asarray(agent.skill_values_cross(
        jnp.asarray(decisions[0]['observation'])[None, :],
        jnp.asarray(rollout['goal'])[None, :],
    ))[0, 0]
    max_dev = float(np.abs(probe - decisions[0]['values']).max())
    if max_dev > 1e-3:
        raise SystemExit(
            f'skill_values_cross disagrees with skill_values by {max_dev:.3g}; the value '
            f'maps would not be the quantity the selector maximises.'
        )
    print(f'  cross-form check: max |skill_values_cross - skill_values| = {max_dev:.3g}',
          flush=True)

    print(f'  embedding {len(goal_states)} candidate goals ({args.goal_source})...', flush=True)
    goal_embeddings = agent.value_goal_embeddings(jnp.asarray(goal_states))

    picked = decisions[args.start_decision::args.panel_stride][:args.num_panels]
    print(f'  computing {len(picked)} value maps '
          f'(decisions #{picked[0]["index"]}..#{picked[-1]["index"]})...', flush=True)
    for d in picked:
        vals = np.asarray(agent.skill_values_from_goal_embeddings(
            jnp.asarray(d['observation'])[None, :], goal_embeddings,
        ))[0, :, d['skill']]                                        # [G]
        d['value_map'] = bin_values(goal_xy_all, vals, maze, args.grid_res)
        d['peak_xy'] = map_peak_xy(d['value_map'], maze, args.grid_res)

    # ── Draw ─────────────────────────────────────────────────────────────────
    # Figures land in a subfolder of the run dir, not loose beside the
    # checkpoints: a sweep over tasks and horizons puts a dozen files here, and
    # `latest_epoch` / the eval JSONs share the directory.
    prefix = args.output_prefix or os.path.join(
        run_dir, 'value_viz',
        f'skill_sequence_e{epoch}_task{args.task_id}_h{args.skill_horizon}'
    )
    # Also covers an --output_prefix pointing into a directory that does not
    # exist yet, which is the usual way to collect figures from several runs.
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    out_path = f'{prefix}.png'
    plot_sequence(
        picked, rollout, maze, out_path, n_cols=args.n_cols,
        zoom_halfwidth=args.zoom_halfwidth, show_peak=args.show_peak,
        shared_scale=args.shared_scale, progress_eps=args.progress_eps,
        title=(f'{env_name} | {os.path.basename(run_dir)} | epoch {epoch} | '
               f'{task["task_name"]}: start {tuple(task["init_xy"])} -> '
               f'goal {tuple(task["goal_xy"])} | {_success_label(rollout)}\n'
               f'decision filmstrip: each panel is V(s_t, z*_t, g) over goals g '
               f'({args.goal_source} goals, {args.grid_res}x{args.grid_res} bins) under the '
               f'skill the selector chose at that decision, '
               f'z* = argmax_z V(s, z, g) every {args.skill_horizon} steps'),
    )
    print(f'Saved decision filmstrip: {out_path}')

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
            drawn_decisions=[int(d['index']) for d in picked],
            steps=[int(d['step']) for d in decisions],
            chosen_skills=[int(d['skill']) for d in decisions],
            goal_values=[d['goal_value'] for d in decisions],
            value_gaps=[d['gap'] for d in decisions],
            dist_to_goal=[d['dist_start'] for d in decisions],
            progress=[d['progress'] for d in decisions],
            progress_class=[d['progress_class'] for d in decisions],
        ), f, indent=2)
    print(f'Saved stats: {stats_out}')


if __name__ == '__main__':
    main()
