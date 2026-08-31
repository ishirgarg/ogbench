"""Skill-sweep evaluation of a pretrained skill-conditioned policy on OGBench.

Skill-conditioned agents (empowerment_skill, opal, dds, dads, ...) have no
goal-conditioned actor: their policy is pi(a | s, z) for a skill z, so the
standard OGBench eval loop (which feeds the task goal to the actor) does not
measure what they actually learned. This script instead sweeps the skill set:

  1. For every (skill z, task/goal g) pair, roll out pi(. | ., z) from g's
     initial state for `--eval_episodes` episodes and record the success rate.
     The goal never enters the policy -- it only sets the env's init state and
     the success criterion.
  2. For each goal, take the best skill:  best(g) = max_z success(z, g).
  3. Benchmark score = mean_g best(g), with the standard error across goals.

Note that step 2 picks the skill on the same episodes used to score it, so the
reported number is an optimistic (best-case-skill) estimate of what a perfect
high-level skill selector could achieve -- it is an upper bound on the
hierarchical policy, not the performance of any single deployed policy. The
"single best skill across all goals" figure is also reported for reference.

Everything else (env, number of skills, network architecture) is read from the
run's own `flags.json`, so only the checkpoint directory is required:

    python eval_skill_policy.py --run_dir ckpts/empowerment/antmaze-medium-navigate/sd000_...

An agent supports this script by implementing two hooks (see
`agents/empowerment_skill.py` for the reference implementation):

    skill_set(seed=None, num_skills=None, observations=None) -> [K, skill_width]
    sample_actions_with_skill(observations, skills, seed=None, temperature=1.0)

Agents whose policy needs per-episode history (Skill-DT's length-K Transformer
context) may replace the second hook with the stateful pair

    init_eval_state_with_skill(skill) -> agent_state
    sample_actions_with_skill_state(observations, skills, agent_state=None, seed=None,
                                    temperature=1.0) -> (action, agent_state)
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import glob
import json
import re
import time

import jax
import numpy as np

from agents import agents as agent_registry
from utils.env_utils import make_env_and_datasets
from utils.evaluation import env_horizon, evaluate_skill, raise_time_limit
from utils.flax_utils import restore_agent


def latest_epoch(run_dir):
    """Largest epoch among the run dir's params_*.pkl checkpoints."""
    epochs = [
        int(m.group(1))
        for p in glob.glob(os.path.join(run_dir, 'params_*.pkl'))
        if (m := re.search(r'params_(\d+)\.pkl$', os.path.basename(p)))
    ]
    if not epochs:
        raise FileNotFoundError(f'No params_*.pkl found in {run_dir}.')
    return max(epochs)


def load_flags(run_dir):
    with open(os.path.join(run_dir, 'flags.json')) as f:
        return json.load(f)


def eval_horizon(env):
    """The env's episode horizon -- OGBench's registered `max_episode_steps`.

    The env is built by the exact same `make_env_and_datasets` call that training
    used, so this is the standard per-env horizon (1000 for antmaze/antsoccer),
    applied by gymnasium's TimeLimit wrapper. Reported so it is verifiable, and
    shared with `utils.evaluation`, which hands it to agents that roll out over
    the whole horizon.
    """
    return env_horizon(env)


def load_agent(run_dir, epoch, saved, dataset_path=None):
    """Rebuild the agent from the run's own flags.json and load its checkpoint.

    Returns `(agent, env, config, train_dataset)`.
    """
    config = saved['agent']
    env_name = saved['env_name']

    env, train_dataset, _ = make_env_and_datasets(
        env_name,
        frame_stack=config.get('frame_stack'),
        dataset_path=dataset_path if dataset_path is not None else saved.get('dataset_path'),
    )

    example_batch = train_dataset.sample(1)
    if config.get('discrete'):
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agent_registry[config['agent_name']]
    agent = agent_class.create(
        seed=saved.get('seed', 0),
        ex_observations=example_batch['observations'],
        ex_actions=example_batch['actions'],
        config=config,
    )
    agent = restore_agent(agent, run_dir, epoch)
    return agent, env, config, train_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True,
                   help='Checkpoint run dir (the folder holding flags.json / params_*.pkl).')
    p.add_argument('--epoch', type=int, default=None, help='Checkpoint epoch (default: latest).')
    p.add_argument('--eval_tasks', type=int, default=None, help='Number of tasks to evaluate (default: all).')
    p.add_argument('--eval_episodes', type=int, default=50, help='Number of episodes per (skill, goal) pair.')
    p.add_argument('--eval_temperature', type=float, default=0.0, help='Actor temperature for evaluation.')
    p.add_argument('--eval_gaussian', type=float, default=None, help='Action Gaussian noise for evaluation.')
    p.add_argument('--eval_on_cpu', type=int, default=1, help='Whether to evaluate on CPU.')
    p.add_argument('--horizon', type=int, default=None,
                   help='Override the env episode horizon (max steps before truncation). Only ever '
                        'lengthens it (default: the env\'s registered horizon).')
    p.add_argument('--num_skills', type=int, default=None,
                   help='Number of skills to sweep. Only used by agents whose skill set is not finite '
                        '(e.g. OPAL with latent_type="continuous"); discrete skill sets ignore it.')
    p.add_argument('--skills', type=str, default=None,
                   help='Comma-separated skill indices to restrict the sweep to (default: all).')
    p.add_argument('--seed', type=int, default=0, help='Seed for skill sampling and action sampling.')
    p.add_argument('--dataset_path', type=str, default=None,
                   help='Override the dataset path recorded in flags.json (only affects agent construction).')
    p.add_argument('--output', type=str, default=None,
                   help='Output JSON path (default: <run_dir>/skill_eval_e<epoch>.json).')
    p.add_argument('--merge_shards', type=str, default=None,
                   help='Glob of per-skill shard JSONs to merge into one run-level result, instead '
                        'of running any rollouts. Use with --skills to shard a sweep across jobs.')
    args = p.parse_args()

    run_dir = args.run_dir.rstrip('/')
    epoch = args.epoch if args.epoch is not None else latest_epoch(run_dir)

    if args.merge_shards is not None:
        merge_shards(
            glob.glob(args.merge_shards),
            args.output or os.path.join(run_dir, f'skill_eval_e{epoch}.json'),
        )
        return

    saved = load_flags(run_dir)
    agent_name, env_name = saved['agent']['agent_name'], saved['env_name']

    # Checked up front: building the env and loading the offline dataset takes minutes,
    # and a non-skill-conditioned agent can never be evaluated here.
    agent_class = agent_registry[agent_name]
    has_actor = hasattr(agent_class, 'sample_actions_with_skill') or (
        hasattr(agent_class, 'init_eval_state_with_skill')
        and hasattr(agent_class, 'sample_actions_with_skill_state')
    )
    if not hasattr(agent_class, 'skill_set') or not has_actor:
        raise SystemExit(
            f'Agent "{agent_name}" is not skill-conditioned: it does not implement `skill_set` / '
            f'`sample_actions_with_skill` (or the stateful `init_eval_state_with_skill` / '
            f'`sample_actions_with_skill_state` pair). Add those hooks to agents/{agent_name}.py to '
            f'evaluate it here (see agents/empowerment_skill.py).'
        )

    agent, env, config, _ = load_agent(run_dir, epoch, saved, dataset_path=args.dataset_path)

    if args.horizon is not None:
        raise_time_limit(env, args.horizon)

    if args.eval_on_cpu:
        agent = jax.device_put(agent, device=jax.devices('cpu')[0])

    np.random.seed(args.seed)

    # Reference observation: needed only by agents whose skill set is drawn from a
    # state-conditioned prior (e.g. continuous-latent OPAL).
    ref_obs, _ = env.reset(options=dict(task_id=1, render_goal=False))
    skills = np.asarray(
        agent.skill_set(
            seed=jax.random.PRNGKey(args.seed),
            num_skills=args.num_skills,
            observations=np.asarray(ref_obs, dtype=np.float32),
        )
    )
    skill_ids = list(range(len(skills)))
    if args.skills is not None:
        skill_ids = [int(s) for s in args.skills.split(',') if s.strip() != '']
        skills = skills[skill_ids]

    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = args.eval_tasks if args.eval_tasks is not None else len(task_infos)
    task_names = [task_infos[t]['task_name'] for t in range(num_tasks)]

    K, G, N = len(skills), num_tasks, args.eval_episodes
    horizon = eval_horizon(env)
    print(
        f'[{env_name}] agent={agent_name} epoch={epoch} skills={K} tasks={G} episodes/pair={N} '
        f'horizon={horizon} -> {K * G * N} episodes total'
    )

    success = np.zeros((K, G), dtype=np.float64)
    start = time.time()
    for gi in range(G):
        for zi, skill in enumerate(skills):
            # A fresh but deterministic seed per (skill, goal) cell, so a rerun of the
            # same --seed reproduces the whole matrix.
            stats = evaluate_skill(
                agent=agent,
                env=env,
                skill=skill,
                task_id=gi + 1,
                config=config,
                num_eval_episodes=N,
                eval_temperature=args.eval_temperature,
                eval_gaussian=args.eval_gaussian,
                seed=args.seed * 1_000_003 + gi * 1_009 + zi,
            )
            success[zi, gi] = float(stats['success'])
            print(
                f'  task {gi + 1}/{G} ({task_names[gi]})  skill {skill_ids[zi]:>3}  '
                f'success={success[zi, gi]:.3f}   [{time.time() - start:.0f}s]',
                flush=True,
            )

    report = aggregate(success, skill_ids, task_names, N)
    print_report(report, success, skill_ids, task_names, env_name, agent_name, epoch, N)

    out_path = args.output or os.path.join(run_dir, f'skill_eval_e{epoch}.json')
    write_report(
        out_path,
        report,
        success=success,
        skill_ids=skill_ids,
        task_names=task_names,
        run_dir=run_dir,
        epoch=epoch,
        env_name=env_name,
        agent_name=agent_name,
        eval_episodes=N,
        eval_horizon=horizon,
        eval_temperature=args.eval_temperature,
        eval_gaussian=args.eval_gaussian,
        seed=args.seed,
    )
    print(f'Saved: {out_path}')


def aggregate(success, skill_ids, task_names, num_episodes):
    """Per-goal best skill, then the benchmark score across goals."""
    G, N = success.shape[1], num_episodes
    best_zi = success.argmax(axis=0)
    best_per_goal = success.max(axis=0)
    overall = float(best_per_goal.mean())
    # Standard error across goals (the spread the benchmark score is averaged over).
    overall_se = float(best_per_goal.std(ddof=1) / np.sqrt(G)) if G > 1 else 0.0
    # Binomial SE of each selected cell, pooled -- the sampling noise inside the cells.
    cell_se = np.sqrt(best_per_goal * (1 - best_per_goal) / N)
    pooled_cell_se = float(np.sqrt((cell_se**2).sum()) / G)

    # Reference figure: the single skill that does best when fixed across all goals.
    fixed_mean = success.mean(axis=1)
    fixed_best_zi = int(fixed_mean.argmax())

    return dict(
        overall_success=overall,
        overall_success_se=overall_se,
        overall_success_cell_se=pooled_cell_se,
        best_skill_per_goal=[skill_ids[i] for i in best_zi.tolist()],
        best_success_per_goal=best_per_goal.tolist(),
        fixed_best_skill=skill_ids[fixed_best_zi],
        fixed_best_success=float(fixed_mean[fixed_best_zi]),
        fixed_best_success_se=(
            float(success[fixed_best_zi].std(ddof=1) / np.sqrt(G)) if G > 1 else 0.0
        ),
    )


def print_report(report, success, skill_ids, task_names, env_name, agent_name, epoch, num_episodes):
    K, G = success.shape
    print('\n' + '=' * 78)
    print(f'{env_name} | {agent_name} | epoch {epoch} | {num_episodes} episodes per (skill, goal)')
    print('=' * 78)
    header = 'skill '.ljust(8) + ''.join(f'{name[:14]:>16}' for name in task_names)
    print(header)
    for zi in range(K):
        print(f'{skill_ids[zi]:<8}' + ''.join(f'{success[zi, gi]:>16.3f}' for gi in range(G)))
    print('-' * len(header))
    print('best    ' + ''.join(f'{v:>16.3f}' for v in report['best_success_per_goal']))
    print('argmax  ' + ''.join(f'{z:>16d}' for z in report['best_skill_per_goal']))
    print('=' * 78)
    print(
        f'Benchmark score (mean over goals of best-skill success): '
        f'{report["overall_success"]:.4f} +/- {report["overall_success_se"]:.4f} (SE over goals)'
    )
    print(f'  within-cell binomial SE (pooled): +/- {report["overall_success_cell_se"]:.4f}')
    print(
        f'Reference -- best single skill fixed across all goals: skill {report["fixed_best_skill"]}, '
        f'{report["fixed_best_success"]:.4f} +/- {report["fixed_best_success_se"]:.4f}'
    )


def write_report(out_path, report, *, success, skill_ids, task_names, **meta):
    """Dump the result JSON with the headline accuracy as its first keys."""
    payload = dict(
        overall_success=report['overall_success'],
        overall_success_se=report['overall_success_se'],
        overall_success_cell_se=report['overall_success_cell_se'],
    )
    payload.update(meta)
    payload.update(
        skill_ids=skill_ids,
        task_names=task_names,
        success=success.tolist(),
        best_skill_per_goal=report['best_skill_per_goal'],
        best_success_per_goal=report['best_success_per_goal'],
        fixed_best_skill=report['fixed_best_skill'],
        fixed_best_success=report['fixed_best_success'],
        fixed_best_success_se=report['fixed_best_success_se'],
    )
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)


def merge_shards(shard_paths, out_path):
    """Combine per-skill shard JSONs into one run-level result.

    Each shard holds the rows of the (skill, goal) success matrix for its own
    `--skills` subset; merging just stacks the rows back in skill order and
    re-runs the aggregation over the full skill set.
    """
    shards = []
    for p in sorted(shard_paths):
        with open(p) as f:
            shards.append(json.load(f))
    if not shards:
        raise FileNotFoundError('No shard JSONs to merge.')

    head = shards[0]
    for s in shards[1:]:
        if s['task_names'] != head['task_names']:
            raise ValueError('Shards disagree on the task list; refusing to merge.')
        if s['eval_episodes'] != head['eval_episodes']:
            raise ValueError('Shards disagree on eval_episodes; refusing to merge.')

    rows = {}
    for s in shards:
        for zi, z in enumerate(s['skill_ids']):
            rows[z] = s['success'][zi]
    skill_ids = sorted(rows)
    success = np.array([rows[z] for z in skill_ids], dtype=np.float64)

    task_names, N = head['task_names'], head['eval_episodes']
    report = aggregate(success, skill_ids, task_names, N)
    print_report(
        report, success, skill_ids, task_names,
        head['env_name'], head['agent_name'], head['epoch'], N,
    )
    write_report(
        out_path, report,
        success=success, skill_ids=skill_ids, task_names=task_names,
        run_dir=head['run_dir'], epoch=head['epoch'], env_name=head['env_name'],
        agent_name=head['agent_name'], eval_episodes=N,
        eval_horizon=head.get('eval_horizon'),
        eval_temperature=head['eval_temperature'], eval_gaussian=head['eval_gaussian'],
        seed=head['seed'],
    )
    print(f'Saved: {out_path}')
    return report


if __name__ == '__main__':
    main()
