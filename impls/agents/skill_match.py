"""
Goal-conditioned high-level skill policy via skill matching + IQL.

Given a policy π(a | s, z) pretrained with `empowerment_skill`, this agent learns
a goal-conditioned high-level policy π_hi(z | s, g) that, given a state and goal,
selects a skill z.

Training data is produced by *skill matching*: each transition (s_t, a_t) is
assigned the skill that best reproduces its action under the frozen low-level
policy,

    z*(s_t, a_t) = argmin_z || π(s_t, z) - a_t ||^2 ,

(where π(s_t, z) is the deterministic policy mode). The relabeled dataset — with
the discrete skill index as the action — is then fed to the existing
goal-conditioned IQL implementation (`agents/gciql.py`) with discrete actions.

At evaluation time the two levels are composed:  z ~ π_hi(·| s, g), then
a = π(s, z) is executed in the environment.
"""

import glob
import json
import os
import re
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from agents.empowerment_skill import EmpowermentAgent
from agents.gciql import GCIQLAgent
from utils.flax_utils import nonpytree_field, restore_agent


def _latest_epoch(run_dir):
    """Return the largest epoch among params_*.pkl checkpoints in `run_dir`."""
    ckpts = glob.glob(os.path.join(run_dir, 'params_*.pkl'))
    if not ckpts:
        raise FileNotFoundError(f'No params_*.pkl checkpoint found in {run_dir}')
    epochs = []
    for path in ckpts:
        m = re.search(r'params_(\d+)\.pkl$', os.path.basename(path))
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        raise RuntimeError(f'Could not parse checkpoint epochs in {run_dir}')
    return max(epochs)


class SkillMatchAgent(flax.struct.PyTreeNode):
    """High-level skill policy learned by skill matching + goal-conditioned IQL.

    Fields:
        rng: PRNG key (used to seed action sampling when no seed is supplied).
        iql: A `GCIQLAgent` operating over discrete skill "actions". This is the
            high-level policy π_hi(z | s, g) that is actually trained.
        skill_agent: The frozen pretrained `empowerment_skill` agent. Only its
            low-level policy π(a | s, z) is used (for relabeling and execution).
        config: Static configuration dictionary.
    """

    rng: Any
    iql: Any
    skill_agent: Any
    config: Any = nonpytree_field()

    def assign_skills(self, observations, actions):
        """Assign each transition the skill that best reproduces its action.

        z*(s, a) = argmin_z || π(s, z) - a ||^2, using the deterministic policy
        mode of the frozen low-level policy.

        Args:
            observations: [B, obs_dim] states.
            actions: [B, action_dim] (continuous) low-level actions.

        Returns:
            skills: [B] int32 array of assigned skill indices.
        """
        K = self.config['num_skills']
        batch_size = observations.shape[0]

        def pred_for_skill(z):
            z_onehot = jnp.broadcast_to(jnp.eye(K)[z], (batch_size, K))
            dist = self.skill_agent.network.select('policy')(observations, z_onehot)
            return dist.mode()  # [B, action_dim]

        preds = jax.vmap(pred_for_skill)(jnp.arange(K))      # [K, B, action_dim]
        sq_dist = jnp.sum((preds - actions[None]) ** 2, axis=-1)  # [K, B]
        return jnp.argmin(sq_dist, axis=0).astype(jnp.int32)      # [B]

    def _relabel(self, batch):
        """Return a copy of `batch` with continuous actions replaced by skills."""
        skills = self.assign_skills(batch['observations'], batch['actions'])
        relabeled = dict(batch)
        relabeled['actions'] = skills
        return relabeled

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Relabel the batch, then compute the inner IQL loss (for validation)."""
        relabeled = self._relabel(batch)
        return self.iql.total_loss(relabeled, grad_params, rng=rng)

    @jax.jit
    def update(self, batch):
        """Relabel the batch and take one IQL update step on the high-level policy."""
        new_rng, _ = jax.random.split(self.rng)
        relabeled = self._relabel(batch)
        new_iql, info = self.iql.update(relabeled)
        return self.replace(iql=new_iql, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Hierarchically sample an environment action.

        First sample a skill z ~ π_hi(·| s, g) from the high-level IQL actor,
        then return the low-level action a = π(s, z) from the frozen policy.
        """
        if seed is None:
            seed = self.rng
        hi_seed, lo_seed = jax.random.split(seed)

        # A single (unbatched) state-based obs is 1D; a single visual obs is 3D.
        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single = observations.ndim == single_obs_ndim
        if single:
            observations = observations[None, ...]
            if goals is not None:
                goals = goals[None, ...]

        K = self.config['num_skills']

        # High-level: pick a skill conditioned on (state, goal).
        skill_dist = self.iql.network.select('actor')(observations, goals, temperature=temperature)
        skills = skill_dist.sample(seed=hi_seed)             # [B] int
        skills_onehot = jnp.eye(K)[skills]                   # [B, K]

        # Low-level: execute the frozen skill-conditioned policy.
        low_dist = self.skill_agent.network.select('policy')(
            observations, skills_onehot, temperature=self.config['low_temperature']
        )
        actions = low_dist.sample(seed=lo_seed)
        if not self.skill_agent.config['discrete']:
            actions = jnp.clip(actions, -1, 1)

        if single:
            actions = actions[0]
        return actions

    # ── Skill-conditioned evaluation hook (see eval_skill_policy.py) ─────────
    #
    # Delegates to the frozen low-level skill agent: sweeping skills here bypasses
    # the high-level IQL actor entirely, which is exactly what the sweep wants.

    def skill_set(self, seed=None, num_skills=None, observations=None):
        return self.skill_agent.skill_set(seed=seed, num_skills=num_skills, observations=observations)

    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        return self.skill_agent.sample_actions_with_skill(
            observations, skills, seed=seed, temperature=self.config['low_temperature']
        )

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of (continuous) low-level actions.
            config: Configuration dictionary. Must contain `skill_checkpoint_path`
                pointing at an `empowerment_skill` run directory.
        """
        rng = jax.random.PRNGKey(seed)

        # ── Load the frozen pretrained empowerment_skill agent. ────────────────
        ckpt_path = config['skill_checkpoint_path']
        if ckpt_path is None:
            raise ValueError(
                'skill_match requires --agent.skill_checkpoint_path=<empowerment_skill run dir>.'
            )
        flags_path = os.path.join(ckpt_path, 'flags.json')
        if not os.path.exists(flags_path):
            raise FileNotFoundError(f'flags.json not found in {ckpt_path}')
        with open(flags_path) as f:
            emp_flags = json.load(f)
        emp_config = emp_flags['agent']
        assert emp_config['agent_name'] == 'empowerment_skill', (
            f"Expected an empowerment_skill checkpoint, got agent_name="
            f"{emp_config.get('agent_name')}"
        )

        skill_agent = EmpowermentAgent.create(seed, ex_observations, ex_actions, emp_config)
        restore_epoch = config['skill_restore_epoch']
        if restore_epoch is None:
            restore_epoch = _latest_epoch(ckpt_path)
        skill_agent = restore_agent(skill_agent, ckpt_path, restore_epoch)

        num_skills = int(emp_config['num_skills'])

        # ── Build the inner goal-conditioned IQL over discrete skills. ─────────
        iql_config = dict(
            agent_name='gciql',
            lr=config['lr'],
            batch_size=config['batch_size'],
            actor_hidden_dims=tuple(config['actor_hidden_dims']),
            value_hidden_dims=tuple(config['value_hidden_dims']),
            layer_norm=config['layer_norm'],
            discount=config['discount'],
            tau=config['tau'],
            expectile=config['expectile'],
            actor_loss=config['actor_loss'],
            alpha=config['alpha'],
            const_std=config['const_std'],
            discrete=True,  # The high-level "action" is a discrete skill index.
            encoder=config['encoder'],
            dataset_class=config['dataset_class'],
            value_p_curgoal=config['value_p_curgoal'],
            value_p_trajgoal=config['value_p_trajgoal'],
            value_p_randomgoal=config['value_p_randomgoal'],
            value_geom_sample=config['value_geom_sample'],
            actor_p_curgoal=config['actor_p_curgoal'],
            actor_p_trajgoal=config['actor_p_trajgoal'],
            actor_p_randomgoal=config['actor_p_randomgoal'],
            actor_geom_sample=config['actor_geom_sample'],
            gc_negative=config['gc_negative'],
            p_aug=config['p_aug'],
            frame_stack=config['frame_stack'],
        )
        ex_skill_actions = np.full((ex_observations.shape[0],), num_skills - 1, dtype=np.int32)
        iql_agent = GCIQLAgent.create(seed, ex_observations, ex_skill_actions, iql_config)

        # Store config (with the resolved num_skills) for use in the agent methods.
        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        stored_config['num_skills'] = num_skills
        stored_config['skill_restore_epoch'] = restore_epoch

        return cls(
            rng,
            iql=iql_agent,
            skill_agent=skill_agent,
            config=flax.core.FrozenDict(**stored_config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='skill_match',  # Agent name.
            # Path to a pretrained empowerment_skill run directory (contains
            # flags.json and params_*.pkl). Required.
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),
            # Epoch of the checkpoint to restore (None -> latest params_*.pkl).
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),
            # Number of skills. Auto-filled from the checkpoint's flags.json.
            num_skills=ml_collections.config_dict.placeholder(int),
            # Temperature for the frozen low-level policy at execution time
            # (0 -> deterministic).
            low_temperature=0.0,
            # Inner IQL hyperparameters.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            actor_loss='awr',  # Must be 'awr' for discrete skill actions.
            alpha=3.0,  # AWR temperature.
            const_std=True,  # Whether to use constant std for the actor.
            # discrete refers to the *environment* action space (continuous here);
            # the high-level skill action is handled discretely inside the agent.
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
            # Dataset hyperparameters (goal-conditioned).
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
