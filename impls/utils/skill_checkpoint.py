"""Loading a frozen OGBench skill-conditioned agent from its run directory.

Shared version of the recipe the offline controller agents
(`skill_value_controller`, `dds_controller`, `opal_controller`) each implement
locally: rebuild the pretrained agent from the run's own `flags.json`, assert the
observation pipeline matches, restore `params_<epoch>.pkl`, never train it
afterwards. DDS checkpoints additionally get `dds_controller`'s phase check
(`restore_epoch >= skill_pretrain_steps`). The run's `env_name` is printed so a
mismatch with the online env is checkable in the log.
"""

import glob
import json
import os
import re

from utils.flax_utils import restore_agent


def latest_epoch(run_dir):
    """Return the largest epoch among params_*.pkl checkpoints in `run_dir`."""
    epochs = [
        int(m.group(1))
        for path in glob.glob(os.path.join(run_dir, 'params_*.pkl'))
        if (m := re.search(r'params_(\d+)\.pkl$', os.path.basename(path)))
    ]
    if not epochs:
        raise FileNotFoundError(f'No params_*.pkl checkpoint found in {run_dir}')
    return max(epochs)


def load_frozen_skill_agent(seed, ex_observations, ex_actions, config, agent_classes, caller='controller'):
    """Rebuild and restore the frozen skill agent named by `config['skill_checkpoint_path']`.

    Args:
        seed: Seed for the (irrelevant, restored) parameter init.
        ex_observations / ex_actions: Example batch shaping the pretrained networks.
        config: The controller config; reads `skill_checkpoint_path`,
            `skill_restore_epoch`, `num_skills`, and the observation-pipeline keys
            `encoder` / `frame_stack` / `discrete`, which must equal the checkpoint's.
        agent_classes: Mapping agent_name -> agent class for the accepted checkpoint families.
        caller: Name used in error messages.

    Returns:
        `(skill_agent, resolved)` with `resolved` holding `ckpt_path`,
        `restore_epoch`, `num_skills`, `agent_name`, `env_name`, `skill_config`.
    """
    ckpt_path = config['skill_checkpoint_path']
    if ckpt_path is None:
        raise ValueError(f'{caller} requires --agent.skill_checkpoint_path=<skill-agent run dir>.')
    ckpt_path = ckpt_path.rstrip('/')
    flags_path = os.path.join(ckpt_path, 'flags.json')
    if not os.path.exists(flags_path):
        raise FileNotFoundError(f'flags.json not found in {ckpt_path}')
    with open(flags_path) as f:
        skill_flags = json.load(f)
    skill_config = skill_flags['agent']
    agent_name = skill_config.get('agent_name')
    if agent_name not in agent_classes:
        raise ValueError(
            f'{caller}: expected one of {sorted(agent_classes)} checkpoints, got agent_name={agent_name!r} '
            f'in {flags_path}'
        )

    # The controller hands *its* observations to the pretrained network, so the two
    # observation pipelines have to agree (same checks as skill_value_controller).
    for key in ('encoder', 'frame_stack', 'discrete'):
        if config[key] != skill_config.get(key):
            expected = skill_config.get(key)
            fix = f'omit --agent.{key}' if expected is None else f'pass --agent.{key}={expected!r}'
            raise ValueError(
                f"{key}={config[key]!r} does not match the pretrained checkpoint's {key}={expected!r} "
                f'({flags_path}); {fix}.'
            )

    num_skills = int(skill_config['num_skills'])
    if config['num_skills'] is not None and int(config['num_skills']) != num_skills:
        raise ValueError(
            f"num_skills={config['num_skills']} disagrees with the pretrained checkpoint's "
            f'num_skills={num_skills} ({flags_path}); omit the flag, it is read from the checkpoint.'
        )

    restore_epoch = config['skill_restore_epoch']
    if restore_epoch is None:
        restore_epoch = latest_epoch(ckpt_path)

    extra = ''
    if agent_name == 'dds':
        # A DDS run trains the VQ-VAE / decoder for the first `skill_pretrain_steps`
        # (phase 1); only later checkpoints hold a usable skill decoder (dds_controller
        # performs the same check).
        pretrain_steps = int(skill_config.get('skill_pretrain_steps', 0))
        if int(restore_epoch) < pretrain_steps:
            raise ValueError(
                f'{caller}: skill_restore_epoch={restore_epoch} is inside the DDS pre-training phase '
                f'(skill_pretrain_steps={pretrain_steps}); pick a later epoch from {ckpt_path}.'
            )
        extra = f', sequence_length={skill_config.get("sequence_length")}'

    print(
        f'[{caller}] frozen skill policy: {ckpt_path} (epoch {restore_epoch})\n'
        f'[{caller}]   agent_name={agent_name!r}, num_skills={num_skills}{extra}, '
        f'pretrained env_name={skill_flags.get("env_name")!r} -- --env_name should match.'
    )
    skill_agent = agent_classes[agent_name].create(seed, ex_observations, ex_actions, skill_config)
    skill_agent = restore_agent(skill_agent, ckpt_path, int(restore_epoch))

    resolved = dict(
        ckpt_path=ckpt_path,
        restore_epoch=int(restore_epoch),
        num_skills=num_skills,
        agent_name=agent_name,
        env_name=skill_flags.get('env_name'),
        skill_config=skill_config,
    )
    return skill_agent, resolved
