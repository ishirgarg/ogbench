"""Post-hoc evaluation of a `skill_restricted_iql` run with eval-time overrides.

Reuses `utils.evaluation.evaluate_value_selected_skill` (greedy over the agent's
`skill_values`, held for `--skill_horizon` steps), which is exactly SR-IQL's policy
`argmax_z [Qz(s,g)[z] + alpha log c(z|s)]`, so a single trained checkpoint can be
scored at several horizons / selectors / coverage weights without retraining:

    python eval_sriql.py --run_dir <run> --skill_horizon 10
    python eval_sriql.py --run_dir <run> --selector qmu        # off-support Q(s, mu_z(s), g) ablation
    python eval_sriql.py --run_dir <run> --coverage_alpha 0

Writes <run_dir>/sriql_eval_e<epoch>_h<H>_<selector>_a<alpha>.json.
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import json
import time

import jax
import numpy as np

from eval_skill_policy import eval_horizon, latest_epoch, load_agent, load_flags
from utils.evaluation import evaluate_value_selected_skill


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--epoch', type=int, default=None)
    p.add_argument('--eval_tasks', type=int, default=None)
    p.add_argument('--eval_episodes', type=int, default=50)
    p.add_argument('--skill_horizon', type=int, default=None, help='Default: the run config\'s skill_horizon.')
    p.add_argument('--selector', type=str, default=None, choices=[None, 'qz', 'qmu'])
    p.add_argument('--coverage_alpha', type=float, default=None)
    p.add_argument('--eval_on_cpu', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()

    run_dir = args.run_dir.rstrip('/')
    epoch = args.epoch if args.epoch is not None else latest_epoch(run_dir)
    saved = load_flags(run_dir)
    assert saved['agent']['agent_name'] == 'skill_restricted_iql', saved['agent']['agent_name']

    # Eval-time overrides go into the config the agent is rebuilt from.
    cfg = saved['agent']
    if args.selector is not None:
        cfg['selector'] = args.selector
    if args.coverage_alpha is not None:
        cfg['coverage_alpha'] = float(args.coverage_alpha)
    if args.skill_horizon is not None:
        cfg['skill_horizon'] = int(args.skill_horizon)
    skill_horizon = int(cfg['skill_horizon'])

    agent, env, config, _ = load_agent(run_dir, epoch, saved)
    if args.eval_on_cpu:
        agent = jax.device_put(agent, device=jax.devices('cpu')[0])
    np.random.seed(args.seed)

    skills = np.asarray(agent.skill_set())
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = args.eval_tasks if args.eval_tasks is not None else len(task_infos)

    per_task, successes = {}, []
    t0 = time.time()
    for gi in range(num_tasks):
        stats = evaluate_value_selected_skill(
            agent=agent, env=env, skills=skills, task_id=gi + 1, config=config,
            num_eval_episodes=args.eval_episodes, skill_horizon=skill_horizon,
            seed=args.seed * 1_000_003 + gi * 1_009,
        )
        name = task_infos[gi]['task_name']
        per_task[name] = {k: (v if isinstance(v, list) else float(v)) for k, v in stats.items()}
        successes.append(float(stats['success']))
        print(f'[eval_sriql] {name}: success={stats["success"]:.3f} switches={stats["skill_switches"]:.1f} '
              f'({time.time() - t0:.0f}s)', flush=True)

    report = dict(
        overall_success=float(np.mean(successes)),
        overall_success_se=float(np.std(successes, ddof=1) / np.sqrt(len(successes))) if len(successes) > 1 else 0.0,
        run_dir=run_dir, epoch=epoch, env_name=saved['env_name'], eval_episodes=args.eval_episodes,
        skill_horizon=skill_horizon, selector=cfg['selector'], coverage_alpha=float(cfg['coverage_alpha']),
        kernel=cfg['kernel'], expectile=float(cfg['expectile']), eval_horizon=eval_horizon(env),
        kernel_sigma=float(np.asarray(agent.kernel_sigma)), seed=args.seed,
        success=successes, per_task=per_task,
    )
    out = args.output or os.path.join(
        run_dir, f'sriql_eval_e{epoch}_h{skill_horizon}_{cfg["selector"]}_a{cfg["coverage_alpha"]}.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[eval_sriql] overall_success={report["overall_success"]:.3f} -> {out}')


if __name__ == '__main__':
    main()
