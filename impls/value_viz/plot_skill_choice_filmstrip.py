"""One image: what the selector saw, and what it picked, at every commitment.

The question this answers is "at which commitment did the selector choose the
wrong skill?", and it answers it without asking you to cross-reference three
figures. Each decision -- steps 0-H, H-2H, 2H-3H, ... -- gets a *column* of two
stacked cells:

    top     V(s_t, z*, g) over goals g: the value map under the skill actually
            chosen at that decision, on the whole maze, with the episode path so
            far (grey) and the H steps then walked (white).

    bottom  every candidate skill replayed --fan_steps steps from that exact
            simulator state, the chosen one in red and the rest in grey, zoomed.

The top cell is what the selector was looking at; the bottom cell is what the
alternatives it passed over would actually have done. Together they separate the
two failure modes at a glance, per commitment:

  * the red path goes somewhere sensible but a grey one heads at the goal
    -> the *argmax* is wrong: a better skill was available and was not picked;
  * every path, red and grey, goes to the same place (or nowhere)
    -> the *skills* are the problem: there is no decision left to get right.

Every fan cell shares one zoom half-width (the smallest box containing every
replayed path at every drawn decision), so path lengths are comparable down the
figure -- a decision where the ant stops moving reads as a collapsed fan, not as
a rescaled one. `--per_decision_zoom` reverts to a per-decision box.

Column frames are coloured by what the commitment achieved in hindsight -- green
closed distance to the goal, red opened it, grey neither (`--progress_eps`) -- so
the turn is scannable before you read a single map.

Reading the maps: V here is a learned discounted state-occupancy density, not a
distance-to-go, so its argmax over g sits near s for every skill and is not the
diagnostic it looks like (hence no peak marker). What the selector actually used
is the single value at the red star, compared across skills: `V(g)` and the
margin `gap` = top1 - top2, both in each column's header.

Usage:

    python value_viz/plot_skill_choice_filmstrip.py --run_dir ckpts/empowerment/.../sd000_... --task_id 1

Writes, under `<run_dir>/value_viz/` (`--output_prefix` replaces the whole prefix,
directory included, and its directory is created if missing):

    skill_choice_e<epoch>_task<t>_h<H>.png         the figure
    skill_choice_e<epoch>_task<t>_h<H>_stats.json  the per-decision numbers
"""

import os

# Set before any mujoco import: `make_env_and_datasets` builds the training env,
# which initialises GL on import.
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
from matplotlib.patches import Patch

from agents import agents as agent_registry
from eval_skill_policy import eval_horizon, latest_epoch, load_agent, load_flags
from utils.evaluation import raise_time_limit
# Shared with the other two figures in this folder, deliberately: the rollout, the
# maze geometry, the binned value map and the skill-fan panel are one implementation
# each, so all three figures describe the same policy rather than three re-derivations
# of it. `_draw_skill_paths_panel` in particular is what makes the chosen-skill
# highlight here identical to the atlas figure's.
from plot_skill_value_selector import (
    NEEDED_HOOKS,
    Maze,
    _draw_skill_paths_panel,
    _success_label,
    bin_values,
    dataset_goal_candidates,
    grid_goal_candidates,
    rollout_skills_from_state,
    rollout_value_selected,
)


# Frame colours for the hindsight outcome of a commitment.
OUTCOME_COLORS = {'closer': '#2ca02c', 'farther': '#d62728', 'flat': '#8c8c8c'}


def classify_progress(delta, eps):
    """`delta` = distance to goal at the segment's end minus at its start."""
    if delta < -eps:
        return 'closer'
    if delta > eps:
        return 'farther'
    return 'flat'


def shared_halfwidth(picked, minimum=1.5, margin=1.15):
    """One zoom box that contains every replayed path at every drawn decision.

    Per-decision boxes would rescale each fan to fill its cell, which is exactly
    the comparison this figure exists to make: a commitment where every skill
    barely moves has to *look* smaller than one where they fan out metres.
    """
    reach = minimum / margin
    for d in picked:
        for path in d['fan_paths'].values():
            reach = max(reach, float(np.abs(path - d['xy'][None, :]).max()))
    return max(minimum, margin * reach)


# ── Figure ───────────────────────────────────────────────────────────────────


def plot_choice_filmstrip(picked, rollout, maze, out_path, title, n_cols=6,
                          per_decision_zoom=False, progress_eps=0.05,
                          fan_steps=30, n_candidates=0):
    """The two-row-per-decision grid: value map on top, skill fan underneath."""
    skill_horizon = rollout['skill_horizon']
    goal_xy = rollout['goal_xy']
    xy_traj = rollout['xy_traj']
    x_lo, x_hi, y_lo, y_hi = maze.extent

    n = len(picked)
    n_cols = min(n_cols, n)
    n_bands = int(np.ceil(n / n_cols))
    halfwidth = None if per_decision_zoom else shared_halfwidth(picked)

    # A band is one row of value maps with its row of fans directly underneath, so a
    # column reads top-to-bottom as one decision. `hspace` is set per-band-pair below
    # by the height ratios: the fan sits tight under its map, the next band gets air.
    band_h, fan_h = 3.5, 2.9
    fig_h = n_bands * (band_h + fan_h) + 2.4
    fig = plt.figure(figsize=(3.3 * n_cols, fig_h))
    title_h, legend_h = 1.35, 1.05
    gs = gridspec.GridSpec(
        2 * n_bands, n_cols, figure=fig,
        height_ratios=[band_h, fan_h] * n_bands,
        left=0.03, right=0.985,
        top=1.0 - title_h / fig_h, bottom=legend_h / fig_h,
        hspace=0.30, wspace=0.10,
    )

    for panel, d in enumerate(picked):
        band, col = panel // n_cols, panel % n_cols
        color = OUTCOME_COLORS[d['progress_class']]

        # ── top: the value map the selector's argmax ran over ────────────────
        ax = fig.add_subplot(gs[2 * band, col])
        vmap = d['value_map']
        finite = vmap[np.isfinite(vmap)]
        # Robust limits: a handful of far-off-manifold cells otherwise flatten the
        # whole map into one colour.
        vmin, vmax = (np.percentile(finite, [2, 100]) if finite.size else (0.0, 1.0))
        im = ax.imshow(vmap, origin='lower', extent=(x_lo, x_hi, y_lo, y_hi),
                       cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest',
                       zorder=1)
        maze.overlay(ax, alpha=0.25)

        past = xy_traj[: d['segment_start'] + 1]
        if len(past) > 1:
            ax.plot(past[:, 0], past[:, 1], color='0.8', linewidth=1.6, alpha=0.95,
                    zorder=4, solid_capstyle='round')
        seg = xy_traj[d['segment_start']: d['segment_start'] + skill_horizon + 1]
        ax.plot(seg[:, 0], seg[:, 1], color='black', linewidth=3.0, zorder=5,
                solid_capstyle='round')
        ax.plot(seg[:, 0], seg[:, 1], color='white', linewidth=1.8, zorder=6,
                solid_capstyle='round')
        ax.scatter([d['xy'][0]], [d['xy'][1]], c='white', s=45, marker='o',
                   edgecolors='black', linewidths=1.0, zorder=7)
        ax.scatter([goal_xy[0]], [goal_xy[1]], c='red', s=150, marker='*',
                   edgecolors='white', linewidths=0.7, zorder=8)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set(edgecolor=color, linewidth=3.0)
        ax.set_title(
            f"steps {d['step']}-{d['step'] + skill_horizon}   $z^*$ = {d['skill']}\n"
            f"d(goal) {d['dist_start']:.2f} $\\to$ {d['dist_end']:.2f} m "
            f"({d['progress']:+.2f})\n"
            f"V(g)={d['goal_value']:.3f}   gap={d['gap']:.3f}",
            fontsize=8.5, color=(color if d['progress_class'] != 'flat' else 'black'),
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)

        # ── bottom: every candidate skill replayed from this same state ──────
        ax = fig.add_subplot(gs[2 * band + 1, col])
        hw = halfwidth
        if hw is None:
            reach = max((float(np.abs(p - d['xy'][None, :]).max())
                         for p in d['fan_paths'].values()), default=1.5)
            hw = max(1.5, 1.15 * reach)
        _draw_skill_paths_panel(ax, d['fan_paths'], d['xy'], d['skill'], goal_xy,
                                maze, hw)
        for spine in ax.spines.values():
            spine.set(edgecolor=color, linewidth=3.0)
        ax.set_xlabel(
            f"all {len(d['fan_paths'])} skills, {fan_steps} steps from $s_t$ "
            f"($z^*$={d['skill']} red)   $\\pm${hw:.1f} m",
            fontsize=7.5,
        )

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', linestyle='', markersize=8,
                   label='state $s_t$ at the decision'),
        plt.Line2D([0], [0], color='0.4', linewidth=2.0,
                   label=f'top: the {skill_horizon} steps walked under $z^*$ (white)'),
        plt.Line2D([0], [0], color='0.8', linewidth=1.8, label='top: episode path so far'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markeredgecolor='white', linestyle='', markersize=14,
                   label='task goal g'),
        plt.Line2D([0], [0], color='red', linewidth=2.6,
                   label=f'bottom: {fan_steps} steps under the chosen $z^*$'),
        plt.Line2D([0], [0], color='0.55', linewidth=1.2,
                   label=f'bottom: the other {max(n_candidates - 1, 0)} candidate skills'),
        # Drawn as empty boxes, not lines: the "lost distance" frame is the same red
        # as the chosen skill's path, and two red lines in one legend read as one thing.
        Patch(facecolor='none', edgecolor=OUTCOME_COLORS['closer'], linewidth=2.5,
              label='frame: commitment closed distance'),
        Patch(facecolor='none', edgecolor=OUTCOME_COLORS['farther'], linewidth=2.5,
              label='frame: commitment lost distance'),
        Patch(facecolor='none', edgecolor=OUTCOME_COLORS['flat'], linewidth=2.5,
              label=f'frame: no net progress (<{progress_eps:g} m)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
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
                   help='Env steps a chosen skill is held for -- the commitment each '
                        'column covers (default: 10, matching eval_skill_value_policy.py).')
    p.add_argument('--skills', type=str, default=None,
                   help='Comma-separated skill indices the selector may choose from '
                        '(default: all). These are also the skills the fan replays.')
    p.add_argument('--num_panels', type=int, default=12,
                   help='Decisions to draw (default: 12). Each costs one value map plus '
                        'one replay per candidate skill, so this is the main cost knob.')
    p.add_argument('--start_decision', type=int, default=0,
                   help='First decision index to draw (default: 0, the episode start).')
    p.add_argument('--panel_stride', type=int, default=1,
                   help='Draw every Nth decision from --start_decision (default: 1, i.e. '
                        'consecutive commitments).')
    p.add_argument('--n_cols', type=int, default=6, help='Decisions per band (default: 6).')
    p.add_argument('--fan_steps', type=int, default=30,
                   help='Env steps to replay under every candidate skill from each drawn '
                        'decision state (default: 30).')
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
    p.add_argument('--progress_eps', type=float, default=0.05,
                   help='Distance change (m) below which a commitment counts as no net '
                        'progress and its frame is drawn grey (default: 0.05).')
    p.add_argument('--per_decision_zoom', action='store_true',
                   help='Give each fan cell its own zoom box (default: one shared box, so '
                        'the fans are comparable down the figure).')
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
                   help='Output path prefix (default: <run_dir>/value_viz/skill_choice_e<epoch>_'
                        'task<t>_h<H>).')
    args = p.parse_args()

    # Everything checkable without the env, checked here: the env build and dataset load
    # cost minutes, and the rollout plus the per-decision replays cost more.
    for name, lo in (('num_panels', 1), ('panel_stride', 1), ('skill_horizon', 1),
                     ('num_goal_states', 1), ('grid_res', 2), ('n_cols', 1),
                     ('fan_steps', 1)):
        if getattr(args, name) < lo:
            raise SystemExit(f'--{name} must be at least {lo}, got {getattr(args, name)}.')
    if args.start_decision < 0:
        raise SystemExit(f'--start_decision must be >= 0, got {args.start_decision}.')
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
    # dataset is checked here. The rejections mirror plot_skill_value_selector.py's:
    # every cell of this figure is indexed by the xy of a state.
    missing = [h for h in NEEDED_HOOKS if not hasattr(agent_registry[agent_name], h)]
    if missing:
        raise SystemExit(
            f'Agent "{agent_name}" cannot be visualised here: it is missing '
            f'{", ".join("`" + h + "`" for h in missing)}. This figure needs both the '
            f'value-greedy selector hooks and the cached-goal-embedding hooks the value '
            f'maps are built from (see agents/empowerment_skill.py).'
        )
    if saved['agent'].get('encoder') is not None:
        raise SystemExit(
            f'run uses encoder={saved["agent"]["encoder"]!r} (image observations). Every '
            f'cell is indexed by the xy of a state, which pixel observations do not '
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
            f'`set_xy` jump. That jump would land inside a replayed skill path, blowing '
            f'the shared fan zoom out to the maze scale, and its destination is drawn '
            f'from the global numpy RNG inside `step` -- so the per-skill replays would '
            f'no longer be matched draws.'
        )
    if hasattr(env.unwrapped, 'get_agent_ball_xy'):
        raise SystemExit(
            f'{env_name} is a ball env: success is measured on the ball, while `get_xy` and '
            f'`goal[:2]` here are the agent\'s. Every cell would plot the wrong body. See '
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

    # ── Per-decision scalars: the selector's numbers, and the hindsight outcome ──
    xy_traj, goal_xy = rollout['xy_traj'], rollout['goal_xy']
    for i, d in enumerate(decisions):
        d['index'] = i
        d['goal_value'] = float(d['values'][d['skill']])
        # Margin over the *candidate set*, which is what the argmax ran over. With a
        # single candidate the selector has no choice to be decisive about.
        cand_vals = np.sort(d['values'][candidates])
        d['gap'] = float(cand_vals[-1] - cand_vals[-2]) if cand_vals.size > 1 else 0.0
        seg = xy_traj[d['segment_start']: d['segment_start'] + args.skill_horizon + 1]
        d['dist_start'] = float(np.linalg.norm(seg[0] - goal_xy))
        d['dist_end'] = float(np.linalg.norm(seg[-1] - goal_xy))
        d['progress'] = d['dist_end'] - d['dist_start']
        d['progress_class'] = classify_progress(d['progress'], args.progress_eps)

    picked = decisions[args.start_decision::args.panel_stride][:args.num_panels]

    # ── Candidate goals + their psi embeddings (computed once, reused per cell) ──
    rng = np.random.default_rng(args.seed)
    if args.goal_source == 'dataset':
        goal_states = dataset_goal_candidates(train_dataset, args.num_goal_states, rng)
    else:
        goal_states = grid_goal_candidates(rollout['goal'], maze, args.grid_res)
    goal_xy_all = goal_states[:, :2].copy()
    # The cell maps come from the bilinear expansion; the rollout's own argmax came from
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

    print(f'  computing {len(picked)} value maps '
          f'(decisions #{picked[0]["index"]}..#{picked[-1]["index"]})...', flush=True)
    for d in picked:
        vals = np.asarray(agent.skill_values_from_goal_embeddings(
            jnp.asarray(d['observation'])[None, :], goal_embeddings,
        ))[0, :, d['skill']]                                        # [G]
        d['value_map'] = bin_values(goal_xy_all, vals, maze, args.grid_res)

    # ── Per-decision skill fans ──────────────────────────────────────────────
    # Last, because `rollout_skills_from_state` restores the simulator to each decision
    # state in turn and so moves the env off the episode it just ran.
    print(f'  replaying {args.fan_steps} steps under each of the {candidates.size} '
          f'candidate skills, at {len(picked)} decisions '
          f'({len(picked) * candidates.size * args.fan_steps} env steps)...', flush=True)
    for d in picked:
        d['fan_paths'] = rollout_skills_from_state(
            env, agent, d, candidates.tolist(), args.fan_steps, config,
            task_id=args.task_id, eval_temperature=args.eval_temperature,
            eval_gaussian=args.eval_gaussian, seed=args.seed,
        )

    # ── Draw ─────────────────────────────────────────────────────────────────
    # Figures land in a subfolder of the run dir, not loose beside the
    # checkpoints: a sweep over tasks and horizons puts a dozen files here, and
    # `latest_epoch` / the eval JSONs share the directory.
    prefix = args.output_prefix or os.path.join(
        run_dir, 'value_viz',
        f'skill_choice_e{epoch}_task{args.task_id}_h{args.skill_horizon}'
    )
    # Also covers an --output_prefix pointing into a directory that does not
    # exist yet, which is the usual way to collect figures from several runs.
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    out_path = f'{prefix}.png'
    plot_choice_filmstrip(
        picked, rollout, maze, out_path, n_cols=args.n_cols,
        per_decision_zoom=args.per_decision_zoom, progress_eps=args.progress_eps,
        fan_steps=args.fan_steps, n_candidates=int(candidates.size),
        title=(f'{env_name} | {os.path.basename(run_dir)} | epoch {epoch} | '
               f'{task["task_name"]}: start {tuple(task["init_xy"])} -> '
               f'goal {tuple(task["goal_xy"])} | {_success_label(rollout)}\n'
               f'per commitment: (top) V(s_t, z*, g) over goals g under the chosen skill '
               f'({args.goal_source} goals, {args.grid_res}x{args.grid_res} bins), '
               f'(bottom) all {candidates.size} candidate skills replayed from s_t with '
               f'z* = argmax_z V(s, z, g) in red'),
    )
    print(f'Saved skill-choice filmstrip: {out_path}')

    stats_out = f'{prefix}_stats.json'
    with open(stats_out, 'w') as f:
        json.dump(dict(
            run_dir=run_dir, epoch=epoch, env_name=env_name, agent_name=agent_name,
            task_id=args.task_id, task_name=task['task_name'],
            init_xy=list(task['init_xy']), goal_xy=list(task['goal_xy']),
            skill_horizon=args.skill_horizon, fan_steps=args.fan_steps, seed=args.seed,
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
            # Where each skill ends up after --fan_steps from each drawn decision: the
            # numbers behind the fan cells, for when a path needs measuring not eyeballing.
            fan_endpoints={
                str(d['index']): {str(z): [float(p[-1, 0]), float(p[-1, 1])]
                                  for z, p in d['fan_paths'].items()}
                for d in picked
            },
        ), f, indent=2)
    print(f'Saved stats: {stats_out}')


if __name__ == '__main__':
    main()
