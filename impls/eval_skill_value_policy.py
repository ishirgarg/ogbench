"""Evaluate a skill-conditioned offline policy with a value-greedy skill selector.

`eval_skill_policy.py` pins one skill for a whole episode and then takes the max
over skills *after* scoring, which upper-bounds a perfect high-level controller
but is not a policy you can deploy. This script closes that loop using the
agent's own learned value function as the high-level controller:

  * every `--skill_horizon` env steps (default 10), reselect the skill greedily,
        z* = argmax_z V(s, z, g),
    where V is the agent's learned (contrastive) value network and g is the
    task goal observation;
  * execute pi(. | ., z*) for the next `--skill_horizon` steps, then reselect.

The resulting number is the success rate of a single hierarchical policy, so
unlike the skill sweep it is directly comparable to goal-conditioned baselines.

Everything else (env, number of skills, network architecture) is read from the
run's own `flags.json`, so only the checkpoint directory is required:

    python eval_skill_value_policy.py --run_dir ckpts/empowerment/antmaze-medium-navigate/sd000_...

An agent supports this script by implementing three hooks (see
`agents/empowerment_skill.py` for the reference implementation):

    skill_set(seed=None, num_skills=None, observations=None) -> [K, skill_width]
    skill_values(observations, goals) -> [K]     # aligned with skill_set()
    sample_actions_with_skill(observations, skills, seed=None, temperature=1.0)

`skill_set` is called here with no arguments, so the skill set must be deterministic:
an agent whose skills are drawn from a seed- or state-conditioned prior would have the
selector score one draw and the policy execute another.
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import json
import time

import jax
import numpy as np

from agents import agents as agent_registry
from eval_skill_policy import eval_horizon, latest_epoch, load_agent, load_flags
from utils.evaluation import evaluate_value_selected_skill


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True,
                   help='Checkpoint run dir (the folder holding flags.json / params_*.pkl).')
    p.add_argument('--epoch', type=int, default=None, help='Checkpoint epoch (default: latest).')
    p.add_argument('--eval_tasks', type=int, default=None, help='Number of tasks to evaluate (default: all).')
    p.add_argument('--eval_episodes', type=int, default=50, help='Number of episodes per task.')
    p.add_argument('--skill_horizon', type=int, default=10,
                   help='Env steps a selected skill is held for before reselecting (default: 10).')
    p.add_argument('--eval_temperature', type=float, default=0.0, help='Actor temperature for evaluation.')
    p.add_argument('--eval_gaussian', type=float, default=None, help='Action Gaussian noise for evaluation.')
    p.add_argument('--eval_on_cpu', type=int, default=1, help='Whether to evaluate on CPU.')
    p.add_argument('--skills', type=str, default=None,
                   help='Comma-separated skill indices the selector may choose from (default: all).')
    p.add_argument('--seed', type=int, default=0,
                   help='Seed for action sampling and for the env RNGs.')
    p.add_argument('--dataset_path', type=str, default=None,
                   help='Override the dataset path recorded in flags.json (only affects agent construction).')
    p.add_argument('--output', type=str, default=None,
                   help='Output JSON path (default: <run_dir>/skill_value_eval_e<epoch>_h<horizon>.json).')
    args = p.parse_args()

    run_dir = args.run_dir.rstrip('/')
    epoch = args.epoch if args.epoch is not None else latest_epoch(run_dir)

    saved = load_flags(run_dir)
    agent_name, env_name = saved['agent']['agent_name'], saved['env_name']

    # Checked up front: building the env and loading the offline dataset takes minutes,
    # and an agent without these hooks can never be evaluated here.
    agent_class = agent_registry[agent_name]
    missing = [
        h for h in ('skill_set', 'skill_values', 'sample_actions_with_skill')
        if not hasattr(agent_class, h)
    ]
    if missing:
        raise SystemExit(
            f'Agent "{agent_name}" cannot be evaluated with a value-based skill selector: it is '
            f'missing {", ".join("`" + h + "`" for h in missing)}. Add those hooks to '
            f'agents/{agent_name}.py (see agents/empowerment_skill.py).'
        )

    agent, env, config = load_agent(run_dir, epoch, saved, dataset_path=args.dataset_path)

    if args.eval_on_cpu:
        agent = jax.device_put(agent, device=jax.devices('cpu')[0])

    np.random.seed(args.seed)

    # Called bare, exactly as `skill_values` calls it internally: the selector scores
    # `skill_set()` and this rolls out `skill_set()`, so the two cannot disagree. That
    # also means only agents with a *deterministic* skill set can be evaluated here --
    # a seed/state-conditioned prior would score one draw and execute another.
    skills = np.asarray(agent.skill_set())
    skill_ids = list(range(len(skills)))
    if args.skills is not None:
        try:
            skill_ids = [int(s) for s in args.skills.split(',') if s.strip() != '']
        except ValueError:
            raise SystemExit(f'--skills must be a comma-separated list of integers, got "{args.skills}".')
        if not skill_ids:
            raise SystemExit('--skills selected no skills.')
        if not all(0 <= z < len(skills) for z in skill_ids):
            raise SystemExit(f'--skills must be indices in [0, {len(skills)}).')
        if len(set(skill_ids)) != len(skill_ids):
            raise SystemExit('--skills contains duplicate indices.')

    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = args.eval_tasks if args.eval_tasks is not None else len(task_infos)
    if not 0 < num_tasks <= len(task_infos):
        raise SystemExit(f'--eval_tasks must be in [1, {len(task_infos)}] for {env_name}, got {num_tasks}.')
    if args.eval_episodes < 1:
        raise SystemExit(f'--eval_episodes must be at least 1, got {args.eval_episodes}.')
    if args.skill_horizon < 1:
        raise SystemExit(f'--skill_horizon must be at least 1, got {args.skill_horizon}.')
    task_names = [task_infos[t]['task_name'] for t in range(num_tasks)]

    horizon = eval_horizon(env)
    print(
        f'[{env_name}] agent={agent_name} epoch={epoch} candidate_skills={len(skill_ids)} '
        f'tasks={num_tasks} episodes/task={args.eval_episodes} skill_horizon={args.skill_horizon} '
        f'env_horizon={horizon}'
    )

    success = np.zeros(num_tasks, dtype=np.float64)
    per_task = []
    start = time.time()
    for gi in range(num_tasks):
        # A fresh but deterministic seed per task, so a full rerun of the same --seed
        # reproduces the whole table. (A single task run in isolation will not match
        # its cell here: ogbench's initial-position noise draws from the global numpy
        # RNG, which the preceding tasks advance.)
        stats = evaluate_value_selected_skill(
            agent=agent,
            env=env,
            skills=skills,
            task_id=gi + 1,
            config=config,
            num_eval_episodes=args.eval_episodes,
            skill_horizon=args.skill_horizon,
            eval_temperature=args.eval_temperature,
            eval_gaussian=args.eval_gaussian,
            seed=args.seed * 1_000_003 + gi * 1_009,
            skill_ids=skill_ids,
        )
        success[gi] = float(stats['success'])
        per_task.append(stats)
        counts = np.asarray(stats['skill_selection_counts'])
        top = int(counts.argmax())
        print(
            f'  task {gi + 1}/{num_tasks} ({task_names[gi]})  success={success[gi]:.3f}  '
            f'switches/ep={stats["skill_switches"]:.1f}  most-used skill={skill_ids[top]} '
            f'({counts[top] / max(counts.sum(), 1):.0%})   [{time.time() - start:.0f}s]',
            flush=True,
        )

    report = aggregate(success, per_task, skill_ids, args.eval_episodes)
    print_report(report, success, skill_ids, task_names, env_name, agent_name, epoch,
                 args.eval_episodes, args.skill_horizon)

    out_path = args.output or os.path.join(
        run_dir, f'skill_value_eval_e{epoch}_h{args.skill_horizon}.json'
    )
    write_report(
        out_path,
        report,
        success=success,
        skill_ids=skill_ids,
        task_names=task_names,
        per_task=per_task,
        run_dir=run_dir,
        epoch=epoch,
        env_name=env_name,
        agent_name=agent_name,
        eval_episodes=args.eval_episodes,
        skill_horizon=args.skill_horizon,
        eval_horizon=horizon,
        eval_temperature=args.eval_temperature,
        eval_gaussian=args.eval_gaussian,
        seed=args.seed,
    )
    print(f'Saved: {out_path}')


def aggregate(success, per_task, skill_ids, num_episodes):
    """Benchmark score across goals, plus how the selector used the skill set."""
    G, N = len(success), num_episodes
    overall = float(success.mean())
    # Standard error across goals (the spread the benchmark score is averaged over).
    overall_se = float(success.std(ddof=1) / np.sqrt(G)) if G > 1 else 0.0
    # Binomial SE of each task's cell, pooled -- the sampling noise inside the cells.
    cell_se = np.sqrt(success * (1 - success) / N)
    pooled_cell_se = float(np.sqrt((cell_se**2).sum()) / G)

    # Counts are positional in `skill_ids`, and comparable across tasks because the
    # skill set is built once in main() and reused for every task.
    counts = np.sum([s['skill_selection_counts'] for s in per_task], axis=0)
    total = max(int(counts.sum()), 1)
    return dict(
        overall_success=overall,
        overall_success_se=overall_se,
        overall_success_cell_se=pooled_cell_se,
        mean_skill_switches=float(np.mean([s['skill_switches'] for s in per_task])),
        mean_episode_length=float(np.mean([s['episode_length'] for s in per_task])),
        skill_selection_counts=counts.tolist(),
        skill_selection_fractions=(counts / total).tolist(),
        num_skills_used=int((counts > 0).sum()),
        num_candidate_skills=len(skill_ids),
    )


def print_report(report, success, skill_ids, task_names, env_name, agent_name, epoch,
                 num_episodes, skill_horizon):
    print('\n' + '=' * 78)
    print(
        f'{env_name} | {agent_name} | epoch {epoch} | {num_episodes} episodes/task | '
        f'skill horizon {skill_horizon}'
    )
    print('=' * 78)
    header = 'task    ' + ''.join(f'{name[:14]:>16}' for name in task_names)
    print(header)
    print('success ' + ''.join(f'{v:>16.3f}' for v in success))
    print('-' * len(header))
    print(
        f'Benchmark score (mean over goals, argmax_z V(s, z, g) every {skill_horizon} steps): '
        f'{report["overall_success"]:.4f} +/- {report["overall_success_se"]:.4f} (SE over goals)'
    )
    print(f'  within-cell binomial SE (pooled): +/- {report["overall_success_cell_se"]:.4f}')
    print(
        f'Selector: {report["num_skills_used"]}/{report["num_candidate_skills"]} skills ever chosen, '
        f'{report["mean_skill_switches"]:.1f} switches per episode, '
        f'mean episode length {report["mean_episode_length"]:.0f}'
    )
    counts, fractions = report['skill_selection_counts'], report['skill_selection_fractions']
    order = np.argsort(counts)[::-1][:5]
    print(
        '  most-selected skills: '
        + ', '.join(f'{skill_ids[i]}={fractions[i]:.1%}' for i in order if counts[i] > 0)
    )
    print('=' * 78)


def _to_py(v):
    """numpy scalars -> plain Python, so the per-task stats are JSON-serializable."""
    return v.item() if isinstance(v, np.generic) else v


def write_report(out_path, report, *, success, skill_ids, task_names, per_task, **meta):
    """Dump the result JSON with the headline accuracy as its first keys."""
    payload = dict(
        overall_success=report['overall_success'],
        overall_success_se=report['overall_success_se'],
        overall_success_cell_se=report['overall_success_cell_se'],
    )
    payload.update(meta)
    payload.update(
        task_names=task_names,
        skill_ids=skill_ids,
        success=success.tolist(),
        mean_skill_switches=report['mean_skill_switches'],
        mean_episode_length=report['mean_episode_length'],
        skill_selection_counts=report['skill_selection_counts'],
        skill_selection_fractions=report['skill_selection_fractions'],
        num_skills_used=report['num_skills_used'],
        num_candidate_skills=report['num_candidate_skills'],
        per_task=[{k: _to_py(v) for k, v in s.items()} for s in per_task],
    )
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)


if __name__ == '__main__':
    main()
