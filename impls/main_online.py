"""Online training entry point (env interaction instead of an offline dataset).

The online sibling of `main.py`: same flags style, wandb/CSV logging, `exp/`
layout, `flags.json` and `save_agent` checkpoints, but the agent learns from
its own rollouts in the OGBench env. Two agents plug in today:

  * `agents/online_crl.py`                  -- flat online CRL (JaxGCRL's `crl`).
  * `agents/online_crl_skill_controller.py` -- CRL high-level controller over a
    frozen skill policy (JaxGCRL's `crl_skill`).

Loop (JaxGCRL structure, single env): collect `unroll_length` rows (env steps for
the flat agent, SMDP macro-steps for the controller), then run
`unroll_length * utd_ratio` gradient steps on batches from the trajectory replay
buffer, whose goals are relabelled future observations. Logging, evaluation and
checkpointing are keyed on *env steps* (`--log_interval`, `--eval_interval`,
`--save_interval`), so the two agents are comparable on the same x-axis.
Evaluation runs JaxGCRL-style random-task episodes (`utils/online_evaluation.py`).

RLPD (`--offline_dataset`): an OGBench dataset is loaded into a second replay
buffer and an exact `offline_ratio` share of every batch is drawn from it
(`utils/rlpd.py`); the controller first labels the offline windows with its
frozen skill agent. Nothing else changes (JaxGCRL `use_rlpd`).
"""

import json
import os
import random
import time
from collections import defaultdict

import jax
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.env_utils import make_example_batch
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.online_buffer import TrajectoryReplayBuffer
from utils.online_env import env_horizon, make_online_env
from utils.online_evaluation import evaluate_online, plot_skill_colored_trajectory, skill_usage_stats
from utils.online_rollout import example_transition, make_collector
from utils.rlpd import BufferSource, MixedBatchSampler, make_offline_source

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-medium-navigate-v0', 'Environment name (its registered tasks are used).')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('total_env_steps', 1000000, 'Total number of environment steps.')
flags.DEFINE_integer('episode_length', None, 'Episode horizon override (None -> the env\'s registered limit).')
flags.DEFINE_string(
    'offline_dataset', None, 'RLPD: OGBench dataset name mixed into every batch (None -> online data only).'
)
flags.DEFINE_integer('log_interval', 5000, 'Logging interval (env steps).')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval (env steps).')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval (env steps).')

flags.DEFINE_integer('eval_episodes', 20, 'Number of random-task evaluation episodes.')
flags.DEFINE_float('eval_temperature', 0, 'Actor (controller) temperature for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

config_flags.DEFINE_config_file('agent', 'agents/online_crl.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, 'wandb_run_id.txt'), 'w') as f:
        f.write(wandb.run.id)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environments (no offline data is loaded).
    config = FLAGS.agent
    if 'rollout_type' not in config:
        raise ValueError(
            f'main_online.py needs an online agent config with `rollout_type` '
            f'(agents/online_*.py); got agent_name={config.get("agent_name")!r}. Use main.py for offline agents.'
        )
    env = make_online_env(FLAGS.env_name, frame_stack=config['frame_stack'], episode_length=FLAGS.episode_length)
    eval_env = make_online_env(FLAGS.env_name, frame_stack=config['frame_stack'], episode_length=FLAGS.episode_length)
    horizon = env_horizon(env)
    assert horizon is not None, 'The env has no TimeLimit horizon; pass --episode_length.'
    rows_per_episode = horizon  # buffer rows one full episode can occupy
    if config['rollout_type'] == 'macro':
        k = int(config['skill_commitment_k'])
        assert horizon % k == 0, (
            f'episode horizon ({horizon}) must be divisible by skill_commitment_k ({k}) so macro-steps tile the '
            f'episode; pass --episode_length.'
        )
        rows_per_episode = horizon // k
    print(f'[main_online] env={FLAGS.env_name} horizon={horizon} rollout_type={config["rollout_type"]}')

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = make_example_batch(env, FLAGS.env_name)
    if config['discrete']:
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config)
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Replay buffer + collector (row layout is the collector's; capacity is the agent's).
    collector, buffer = make_collector(
        agent,
        env,
        example_batch,
        buffer_factory=lambda example: TrajectoryReplayBuffer.create(example, int(config['replay_size'])),
        seed=FLAGS.seed,
    )
    assert buffer.capacity > rows_per_episode + 1, (
        f'replay_size ({buffer.capacity}) must exceed the rows of one episode + 1 ({rows_per_episode + 1}) so a '
        f'whole trajectory (plus its final-observation marker) fits in the buffer.'
    )

    unroll_length = int(config['unroll_length'])
    updates_per_round = unroll_length * int(config['utd_ratio'])
    min_replay_size = int(config['min_replay_size'])
    batch_size = int(config['batch_size'])
    goal_discount = float(agent.config['goal_discount'])
    print(
        f'[main_online] unroll_length={unroll_length} rows -> {updates_per_round} updates/round, '
        f'batch_size={batch_size}, min_replay_size={min_replay_size}, goal_discount={goal_discount:.6f}'
    )

    # Batch sampler: the online buffer alone, or RLPD mixing with an offline buffer.
    sampler = BufferSource(buffer, discount=goal_discount)
    if FLAGS.offline_dataset is not None:
        offline_source = make_offline_source(
            FLAGS.offline_dataset,
            agent,
            example_transition(config['rollout_type'], example_batch),
            label_seed=FLAGS.seed,
        )
        sampler = MixedBatchSampler(sampler, offline_source, float(config['offline_ratio']))
        num_online, num_offline = sampler.split(batch_size)
        print(
            f'[main_online] RLPD: {num_online} online + {num_offline} offline rows per batch '
            f'(offline goal_discount={offline_source.discount:.6f}, next_offset={offline_source.next_offset})'
        )

    def run_eval(step):
        eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0]) if FLAGS.eval_on_cpu else agent
        stats, trajs, renders = evaluate_online(
            agent=eval_agent,
            env=eval_env,
            config=agent.config,
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
            eval_temperature=FLAGS.eval_temperature,
        )
        eval_metrics = {f'evaluation/{k}': v for k, v in stats.items()}

        skill_seqs = [t['skills'] for t in trajs if t['skills'] is not None]
        if skill_seqs:
            num_skills = int(agent.config['num_skills'])
            k = int(agent.config['skill_commitment_k'])
            skill_stats, counts = skill_usage_stats(skill_seqs, k, num_skills)
            eval_metrics.update({f'evaluation/{name}': v for name, v in skill_stats.items()})
            eval_metrics['evaluation/skill_usage_hist'] = wandb.Histogram(
                np_histogram=(counts, np.arange(num_skills + 1) - 0.5)
            )
            # Skill-colored xy trajectory of the first episode that has positions.
            for traj in trajs:
                if traj['xy'] is not None and traj['skills'] is not None:
                    fig = plot_skill_colored_trajectory(
                        traj['xy'], traj['skills'], num_skills, goal_xy=traj['goal_xy'],
                        title=f'skills over trajectory @ env step {step} (success={traj["success"]:.0f})',
                    )
                    eval_metrics['evaluation/skill_trajectory'] = wandb.Image(fig)
                    plt.close(fig)
                    break

        if FLAGS.video_episodes > 0 and renders:
            eval_metrics['video'] = get_wandb_video(renders=renders)
        return eval_metrics

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    last_log_step = 0

    env_steps = 0
    num_updates = 0
    rows_since_update = 0
    round_infos = defaultdict(list)  # update metrics of the current round(s), averaged at log time (as JaxGCRL)
    episode_stats = defaultdict(list)
    next_log = FLAGS.log_interval
    next_eval = 0  # Evaluate the untrained agent once (main.py evaluates at its first step).
    next_save = FLAGS.save_interval
    last_saved_step = None

    pbar = tqdm.tqdm(total=FLAGS.total_env_steps, smoothing=0.1, dynamic_ncols=True)
    while env_steps < FLAGS.total_env_steps:
        # Evaluate agent.
        if env_steps >= next_eval:
            eval_metrics = run_eval(env_steps)
            wandb.log(eval_metrics, step=env_steps)
            eval_logger.log(eval_metrics, step=env_steps)
            next_eval += FLAGS.eval_interval

        # Collect experience.
        out = collector.step(agent)
        env_steps += out['env_steps']
        rows_since_update += out['rows']
        pbar.update(out['env_steps'])
        if out['episode'] is not None:
            for name, value in out['episode'].items():
                episode_stats[name].append(value)

        # Update agent: one round of updates every `unroll_length` rows once the buffer is warm.
        if buffer.num_valid >= min_replay_size and rows_since_update >= unroll_length:
            rows_since_update = 0
            for _ in range(updates_per_round):
                batch = sampler.sample(batch_size)
                agent, update_info = agent.update(batch)
                num_updates += 1
                for name, value in update_info.items():
                    round_infos[name].append(value)

        # Log metrics. The CSV header is fixed by the first row, so wait for the first
        # update round before logging (otherwise the training/* columns would be lost).
        if env_steps >= next_log and num_updates > 0:
            train_metrics = {f'training/{k}': float(np.mean(np.asarray(v))) for k, v in round_infos.items()}
            round_infos = defaultdict(list)
            train_metrics.update({f'training/{k}': float(np.mean(v)) for k, v in episode_stats.items()})
            train_metrics['training/num_episodes'] = float(len(episode_stats['episode_length']))
            train_metrics['training/env_steps'] = env_steps
            train_metrics['training/num_updates'] = num_updates
            train_metrics['training/buffer_size'] = buffer.num_valid
            train_metrics['time/sps'] = (env_steps - last_log_step) / max(time.time() - last_time, 1e-6)
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            last_log_step = env_steps
            episode_stats = defaultdict(list)
            wandb.log(train_metrics, step=env_steps)
            train_logger.log(train_metrics, step=env_steps)
            next_log += FLAGS.log_interval
        elif env_steps >= next_log:
            next_log += FLAGS.log_interval

        # Save agent.
        if env_steps >= next_save:
            save_agent(agent, FLAGS.save_dir, env_steps)
            last_saved_step = env_steps
            next_save += FLAGS.save_interval
    pbar.close()

    # Final evaluation + checkpoint at the last env step.
    eval_metrics = run_eval(env_steps)
    wandb.log(eval_metrics, step=env_steps)
    eval_logger.log(eval_metrics, step=env_steps)
    if last_saved_step != env_steps:
        save_agent(agent, FLAGS.save_dir, env_steps)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
