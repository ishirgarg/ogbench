"""Evaluate a skill-conditioned offline policy with a subgoal-graph high-level planner.

`eval_skill_value_policy.py` closes the hierarchical loop with a greedy selector,
`z* = argmax_z V(s, z, g)`, scoring the *task goal* directly. That asks the value
function a question it cannot answer: `V` is a gamma-discounted occupancy, so its
horizon is ~100 steps at gamma = 0.99, while the benchmark's goals sit 500-700 steps
away. Past ~3 maze cells the cost is flat, so the argmax reports per-skill bias
rather than reachability and the episode stalls.

This script replaces the selector with a planner that only ever queries `V` at short
range and recovers the long horizon by composition: calibrate the value function so
skills are comparable, build a graph over dataset states with edge cost
`-max_z Vhat(s_i, z, s_j)`, run Dijkstra to the task goal, and every
`--skill_horizon` steps execute `argmax_z Vhat(s, z, w)` for a waypoint `w` a couple
of hops along the path. See `utils/skill_graph.py`.

Everything else (env, skill count, network architecture) is read from the run's own
`flags.json`:

    python eval_skill_plan.py --run_dir ckpts/empowerment/antmaze-medium-navigate/sd000_...

`--selector` also exposes the two ablations, so a single run reports what the planner
is worth over the greedy baselines:

    raw       argmax_z V(s, z, g)              (what eval_skill_value_policy.py does)
    centered  argmax_z Vhat(s, z, g)           (calibration only, no planning)
    plan      the graph planner                (default)

On top of `skill_set` / `skill_values` / `sample_actions_with_skill`, the `plan` and
`centered` selectors need `value_goal_embeddings` and
`skill_values_from_goal_embeddings` (see `utils/skill_graph.py`).
"""
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from agents import agents as agent_registry
from eval_skill_policy import eval_horizon, latest_epoch, load_agent, load_flags
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate_planned_skill
from utils.skill_graph import REQUIRED_HOOKS, SkillGraphPlanner, SkillValueCalibrator


class GreedySelector:
    """The value-greedy baseline, with calibration optionally switched off.

    Exposes the same `select` signature as `SkillGraphPlanner` so the evaluation loop
    is identical across all three selectors.
    """

    def __init__(self, agent, calibrator=None):
        self.agent = agent
        self.calibrator = calibrator

    def select(self, observation, goal):
        values = np.asarray(
            self.agent.skill_values(observations=jnp.asarray(observation), goals=jnp.asarray(goal))
        )
        if self.calibrator is not None:
            values = values - np.asarray(self.calibrator.log_partition(np.asarray(observation)[None]))[0]
        # Constants, not measurements: this selector has no graph, so the `plan_*`
        # statistics derived from them are not comparable against `plan`'s.
        return int(values.argmax()), dict(waypoint_node=-1, direct=True, fallback=False,
                                          reachable=False, in_range=False)

    def reset(self):
        """No per-episode state; present so the eval loop can call it uniformly."""

    def summary(self):
        return dict(calibrated=self.calibrator is not None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True,
                   help='Checkpoint run dir (the folder holding flags.json / params_*.pkl).')
    p.add_argument('--epoch', type=int, default=None, help='Checkpoint epoch (default: latest).')
    p.add_argument('--selector', type=str, default='plan', choices=('plan', 'centered', 'raw'),
                   help='High-level controller: graph planner, calibrated greedy, or raw greedy.')
    p.add_argument('--eval_tasks', type=int, default=None, help='Number of tasks to evaluate (default: all).')
    p.add_argument('--eval_episodes', type=int, default=50, help='Number of episodes per task.')
    p.add_argument('--skill_horizon', type=int, default=10,
                   help='Env steps a selected skill is held for before replanning (default: 10).')
    p.add_argument('--num_nodes', type=int, default=750, help='Dataset states used as graph nodes.')
    p.add_argument('--num_reference', type=int, default=512,
                   help='Dataset states used to estimate each skill log-partition.')
    p.add_argument('--edge_quantile', type=float, default=0.02,
                   help='Quantile of the pairwise cost distribution admitted as edges. Lower is '
                        'more conservative: the metric is only trustworthy at short range.')
    p.add_argument('--max_degree', type=int, default=64,
                   help='Neighbours considered per node before the mutual and threshold filters. '
                        'Mutual filtering is aggressive, so this needs headroom.')
    p.add_argument('--no_mutual_edges', action='store_true',
                   help='Keep non-reciprocated edges (ablation). Expect shortest path to route '
                        'through the metric false positives.')
    p.add_argument('--hop_cost', type=float, default=1.0,
                   help='Per-hop penalty added to every edge weight, so a route is not cut into '
                        'many near-free steps.')
    p.add_argument('--subgoal_hops', type=int, default=2,
                   help='How many edges along the shortest path the executed waypoint sits.')
    p.add_argument('--goal_degree', type=int, default=16,
                   help='How many nodes are attached to the task goal, by nearest cost. Not '
                        'thresholded: the goal is one specific state, not a dataset state.')
    p.add_argument('--entry_degree', type=int, default=5,
                   help='How many nearest nodes the graph may be entered through. Keep small: '
                        'an argmin over many candidates finds the metric false positives.')
    p.add_argument('--stall_patience', type=int, default=2,
                   help='Consecutive decisions without cost-to-go progress before the last '
                        'executed skill is masked out. 0 disables.')
    p.add_argument('--progress_margin', type=float, default=0.1,
                   help='How much the cost-to-go must fall for a chunk to count as progress.')
    p.add_argument('--temperature', type=float, default=0.0,
                   help='Softmax temperature for the skill choice. 0 (default) is the argmax.')
    p.add_argument('--no_calibrate', action='store_true',
                   help='Skip the log-partition correction (ablation; expect the graph to fall apart).')
    p.add_argument('--eval_temperature', type=float, default=0.0, help='Actor temperature for evaluation.')
    p.add_argument('--eval_gaussian', type=float, default=None, help='Action Gaussian noise for evaluation.')
    p.add_argument('--eval_on_cpu', type=int, default=1, help='Whether to evaluate on CPU.')
    p.add_argument('--seed', type=int, default=0, help='Seed for the graph, action sampling and env RNGs.')
    p.add_argument('--dataset_path', type=str, default=None,
                   help='Override the dataset path recorded in flags.json.')
    p.add_argument('--output', type=str, default=None,
                   help='Output JSON path (default: <run_dir>/skill_plan_eval_<selector>_e<epoch>_h<horizon>.json).')
    args = p.parse_args()

    run_dir = args.run_dir.rstrip('/')
    epoch = args.epoch if args.epoch is not None else latest_epoch(run_dir)

    saved = load_flags(run_dir)
    agent_name, env_name = saved['agent']['agent_name'], saved['env_name']

    # Checked before the minutes spent building the env and loading the dataset.
    agent_class = agent_registry[agent_name]
    needed = ['skill_set', 'skill_values', 'sample_actions_with_skill']
    if args.selector != 'raw':
        needed += list(REQUIRED_HOOKS)
    missing = [h for h in needed if not hasattr(agent_class, h)]
    if missing:
        raise SystemExit(
            f'Agent "{agent_name}" cannot be evaluated with the `{args.selector}` selector: it is '
            f'missing {", ".join("`" + h + "`" for h in dict.fromkeys(missing))}. Add those hooks '
            f'to agents/{agent_name}.py (see agents/empowerment_skill.py).'
        )

    if args.eval_on_cpu:
        jax.config.update('jax_platform_name', 'cpu')

    # The env's initial-position noise is drawn from the global numpy RNG.
    np.random.seed(args.seed)

    agent, env, config, train_dataset = load_agent(run_dir, epoch, saved, dataset_path=args.dataset_path)
    if args.selector != 'raw' and config.get('frame_stack') is not None:
        # The env is frame-stacked but the dataset is not, since stacking happens in
        # `GCDataset`, which `load_agent` never builds.
        raise SystemExit(
            f'run uses frame_stack={config["frame_stack"]}, which `load_agent` does not apply to the '
            f'dataset; graph nodes would not match the observations the env returns. Only '
            f'--selector raw is safe for this run.'
        )
    skills = np.asarray(agent.skill_set())

    build_start = time.time()
    if args.selector == 'raw':
        selector = GreedySelector(agent)
    else:
        observations = np.asarray(train_dataset['observations'])
        if args.selector == 'centered':
            # Only the log-partition is needed, so skip the all-pairs cost matrix. Same
            # draw the planner's calibrator gets, so both use an identical reference set.
            picked = np.random.default_rng(args.seed).choice(
                len(observations), args.num_nodes + args.num_reference, replace=False
            )
            reference = observations[picked[args.num_nodes:]]
            selector = GreedySelector(
                agent, None if args.no_calibrate else SkillValueCalibrator(agent, reference)
            )
        else:
            selector = SkillGraphPlanner(
                agent,
                observations,
                num_nodes=args.num_nodes,
                num_reference=args.num_reference,
                edge_quantile=args.edge_quantile,
                max_degree=args.max_degree,
                mutual_edges=not args.no_mutual_edges,
                hop_cost=args.hop_cost,
                subgoal_hops=args.subgoal_hops,
                goal_degree=args.goal_degree,
                entry_degree=args.entry_degree,
                stall_patience=args.stall_patience,
                progress_margin=args.progress_margin,
                temperature=args.temperature,
                calibrate=not args.no_calibrate,
                seed=args.seed,
            )
    build_time = time.time() - build_start

    task_infos = getattr(env.unwrapped, 'task_infos', None) or env.task_infos
    num_tasks = len(task_infos)
    if args.eval_tasks is not None:
        if not 1 <= args.eval_tasks <= num_tasks:
            raise SystemExit(f'--eval_tasks must be in [1, {num_tasks}] for {env_name}, got {args.eval_tasks}.')
        num_tasks = args.eval_tasks

    per_task = []
    start = time.time()
    for task_id in range(1, num_tasks + 1):
        stats = evaluate_planned_skill(
            agent=agent,
            env=env,
            skills=skills,
            planner=selector,
            task_id=task_id,
            config=config,
            num_eval_episodes=args.eval_episodes,
            skill_horizon=args.skill_horizon,
            eval_temperature=args.eval_temperature,
            eval_gaussian=args.eval_gaussian,
            # Decorrelated per task, as in eval_skill_value_policy.py.
            seed=args.seed * 1_000_003 + task_id * 1_009,
        )
        per_task.append(stats)
        print(
            f'task {task_id}: success={stats["success"]:.3f}  switches={stats["skill_switches"]:.1f}  '
            f'direct={stats["plan_direct_frac"]:.2f}  fallback={stats["plan_fallback_frac"]:.2f}  '
            f'in_range={stats["plan_in_range_frac"]:.2f}',
            flush=True,
        )

    overall = float(np.mean([s['success'] for s in per_task]))
    report = dict(
        run_dir=run_dir,
        env_name=env_name,
        agent_name=agent_name,
        epoch=epoch,
        selector=args.selector,
        num_skills=len(skills),
        eval_episodes=args.eval_episodes,
        skill_horizon=args.skill_horizon,
        episode_horizon=eval_horizon(env),
        seed=args.seed,
        eval_temperature=args.eval_temperature,
        eval_gaussian=args.eval_gaussian,
        eval_tasks=args.eval_tasks,
        dataset_path=args.dataset_path if args.dataset_path is not None else saved.get('dataset_path'),
        overall_success=overall,
        per_task_success=[float(s['success']) for s in per_task],
        per_task_stats=[{k: (v if isinstance(v, list) else float(v)) for k, v in s.items()} for s in per_task],
        selector_summary=selector.summary(),
        build_seconds=build_time,
        eval_seconds=time.time() - start,
    )

    out_path = args.output or os.path.join(
        run_dir, f'skill_plan_eval_{args.selector}_e{epoch}_h{args.skill_horizon}.json'
    )
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\n{env_name}  epoch={epoch}  selector={args.selector}  K={len(skills)}')
    print(f'  per-task: {[f"{s:.3f}" for s in report["per_task_success"]]}')
    print(f'  overall : {overall:.3f}')
    print(f'  graph   : {selector.summary()}')
    print(f'  wrote {out_path}')


if __name__ == '__main__':
    main()
