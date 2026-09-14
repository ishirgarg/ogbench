"""E0: select among frozen skills with a flat goal-conditioned critic, z* = argmax_z Q(s, mu_z(s), g).

Zero training on top of two existing checkpoints:
  * a flat `gciql` run (its critic Q(s, a, g)), and
  * an `empowerment_skill` run (its K skill policies, used as functions: mu_z(s)).

Every `--skill_horizon` env steps the skill whose *first* action the flat critic scores
highest is chosen and then executed by the frozen skill policy. This queries Q at the
skill's action, which is off the data support, and scores only one step of the skill --
so it is an optimistic bound on any selector over these skills, and the question it
answers is whether such a selector can approach flat offline RL at all.

    python eval_skill_qsel.py --gciql_dir <gciql run> --skill_dir <empowerment_skill run> --skill_horizon 1

Writes <gciql_dir>/e0_qsel_e<epoch>_h<H>.json.
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
from utils.evaluation import evaluate_value_selected_skill
from utils.flax_utils import restore_agent


class QSelector:
    """Wrapper exposing the hooks `evaluate_value_selected_skill` needs."""

    def __init__(self, gciql, skill_agent, low_temperature=0.0):
        self.gciql = gciql
        self.skill_agent = skill_agent
        self.low_temperature = low_temperature
        K = int(skill_agent.config['num_skills'])

        @jax.jit
        def values(observations, goals):
            obs_b = observations[None, ...]
            goals_b = goals[None, ...]
            eye = jnp.eye(K)

            def one(k):
                dist = skill_agent.network.select('policy')(obs_b, jnp.broadcast_to(eye[k], (1, K)))
                a = jnp.clip(dist.mode(), -1.0, 1.0)
                q1, q2 = gciql.network.select('critic')(obs_b, goals_b, a)
                return jnp.minimum(q1, q2)[0]

            return jax.lax.map(one, jnp.arange(K))  # [K]

        self._values = values

    def skill_set(self):
        return self.skill_agent.skill_set()

    def skill_values(self, observations, goals):
        return self._values(jnp.asarray(observations), jnp.asarray(goals))

    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        del temperature
        return self.skill_agent.sample_actions_with_skill(observations, skills, seed=seed,
                                                          temperature=self.low_temperature)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gciql_dir', type=str, required=True)
    p.add_argument('--gciql_epoch', type=int, default=None)
    p.add_argument('--skill_dir', type=str, required=True)
    p.add_argument('--skill_epoch', type=int, default=None)
    p.add_argument('--eval_tasks', type=int, default=None)
    p.add_argument('--eval_episodes', type=int, default=50)
    p.add_argument('--skill_horizon', type=int, default=1)
    p.add_argument('--eval_on_cpu', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()

    gciql_dir = args.gciql_dir.rstrip('/')
    skill_dir = args.skill_dir.rstrip('/')
    g_epoch = args.gciql_epoch if args.gciql_epoch is not None else latest_epoch(gciql_dir)
    s_epoch = args.skill_epoch if args.skill_epoch is not None else latest_epoch(skill_dir)

    g_saved = load_flags(gciql_dir)
    s_saved = load_flags(skill_dir)
    assert g_saved['agent']['agent_name'] == 'gciql', g_saved['agent']['agent_name']
    assert s_saved['agent']['agent_name'] == 'empowerment_skill', s_saved['agent']['agent_name']
    assert g_saved['env_name'] == s_saved['env_name'], (g_saved['env_name'], s_saved['env_name'])

    gciql, env, g_config, train_dataset = load_agent(gciql_dir, g_epoch, g_saved)
    example_batch = train_dataset.sample(1)
    s_config = s_saved['agent']
    skill_agent = agent_registry['empowerment_skill'].create(
        seed=0, ex_observations=example_batch['observations'], ex_actions=example_batch['actions'], config=s_config)
    skill_agent = restore_agent(skill_agent, skill_dir, s_epoch)

    if args.eval_on_cpu:
        cpu = jax.devices('cpu')[0]
        gciql = jax.device_put(gciql, device=cpu)
        skill_agent = jax.device_put(skill_agent, device=cpu)
    np.random.seed(args.seed)

    selector = QSelector(gciql, skill_agent)
    skills = np.asarray(selector.skill_set())
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = args.eval_tasks if args.eval_tasks is not None else len(task_infos)

    per_task, successes = {}, []
    t0 = time.time()
    for gi in range(num_tasks):
        stats = evaluate_value_selected_skill(
            agent=selector, env=env, skills=skills, task_id=gi + 1, config=s_config,
            num_eval_episodes=args.eval_episodes, skill_horizon=args.skill_horizon,
            seed=args.seed * 1_000_003 + gi * 1_009,
        )
        name = task_infos[gi]['task_name']
        per_task[name] = {k: (v if isinstance(v, list) else float(v)) for k, v in stats.items()}
        successes.append(float(stats['success']))
        print(f'[e0_qsel] {name}: success={stats["success"]:.3f} switches={stats["skill_switches"]:.1f} '
              f'({time.time() - t0:.0f}s)', flush=True)

    report = dict(
        overall_success=float(np.mean(successes)),
        overall_success_se=float(np.std(successes, ddof=1) / np.sqrt(len(successes))) if len(successes) > 1 else 0.0,
        gciql_dir=gciql_dir, gciql_epoch=g_epoch, skill_dir=skill_dir, skill_epoch=s_epoch,
        env_name=g_saved['env_name'], eval_episodes=args.eval_episodes, skill_horizon=args.skill_horizon,
        eval_horizon=eval_horizon(env), seed=args.seed, success=successes, per_task=per_task,
    )
    out = args.output or os.path.join(gciql_dir, f'e0_qsel_e{g_epoch}_h{args.skill_horizon}.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[e0_qsel] overall_success={report["overall_success"]:.3f} -> {out}')


if __name__ == '__main__':
    main()
