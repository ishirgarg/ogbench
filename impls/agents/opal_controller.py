"""
OPAL high-level task policy on top of a frozen, pretrained OPAL skill agent.

This is the "(2) Offline training of task policy" stage of OPAL (Ajay et al. 2021,
Fig. 2 / Sec. 4.2 / App. F), which `agents/opal.py` deliberately leaves out. Given an
OPAL run directory -- continuous (VAE, Sec. 4.1) or discrete (offline DADS, App. F) --
it turns the offline dataset into the paper's option-level dataset

    D^r_hi = { (s_0,  z,  sum_{t<c} gamma^t r_t,  s_c) },        z ~ posterior(. | tau),

and hands it, unchanged, to a standard OGBench goal-conditioned algorithm whose
action space is the skill space Z. The paper trains pi_psi(z|s) on D^r_hi with CQL; the
default here is IQL (`gciql`), and the inner algorithm is pluggable (`base_agent_name`),
exactly as in `skill_bc_relabel_controller`, whose structure this file follows.

Step 1 -- labelling with the OPAL posterior (`SequenceDataset.relabel_chunk_skills_from_windows`).
    Every start index t is labelled with ONE skill drawn from the pretrained agent's own
    posterior over its length-c window tau = (s_t, a_t, ..., s_{t+c-1}, a_{t+c-1}):

        discrete   (App. F, Eq. 71):  z ~ p(z|tau) = p_w(z) p_phi(tau|z) / sum_z' p_w(z') p_phi(tau|z'),
                                      log p_phi(tau|z) = sum_{i=1}^{c-1} log p_phi(s_{t+i} | s_{t+i-1}, z);
        continuous (Sec. 4.2):        z ~ q_phi(z|tau)   (the BiGRU encoder).

    This is what the paper does ("z_i ~ q_phi(.|tau_i)"; "we use p_{phi,w}(z|tau) to label
    the reward-labelled data"). `label_mode='mode'` takes the argmax / mean instead. It is
    NOT the BC-likelihood argmax of `skill_bc_relabel_controller`: OPAL's discrete posterior
    reads only the STATE trajectory, through the clustering model, and never the actions.

    The pass runs ONCE, before training, from `main.py` via `prepare_datasets`: the
    pretrained agent is frozen, so the labels are final.

Step 2 -- semi-MDP high level.
    Identical to `skill_bc_relabel_controller` (read its docstring for the reasoning):
    one transition per window, s_t --z--> s_{t+c}, reward R_c = sum_{i<c} gamma^i r_i with
    the goal-conditioned per-step reward from `GCDataset`, bootstrap at s_{t+c} discounted
    by the base algorithm's own `discount` (gamma_hi, per OPTION). The paper's analysis
    (Lemma 4.0.1) treats the every-c-steps MDP with discount gamma^c; pass
    `--agent.base.discount=0.904` (0.99 ** 10) for that variant. The rewrite is

        actions           <- chunk_skills        (int index, or float latent)
        next_observations <- subgoal_observations (s_{t+c})
        rewards           <- R_c
        masks             <- macro mask over the window

    so `gciql.critic_loss` computes Q(s_t, z, g) <- R_c + gamma_hi * mask * V(s_{t+c}, g).

Departures from the paper, all deliberate.
    * IQL by default instead of CQL (the paper, C.4 / F.2, uses CQL(H)); any inner
      algorithm in `BASE_AGENTS` can be selected.
    * Goal-conditioned rewards from `GCDataset` instead of a single task reward, because
      OGBench evaluates goal-reaching; the task policy is therefore pi_psi(z | s, g).
    * No low-level fine-tuning. The paper's Eq. 3 fine-tunes the continuous primitive
      pi_theta(a|s,z) by BC on the labelled windows; for the discrete path it states that
      no fine-tuning is needed since pi_theta is trained after the posterior is frozen.
      The pretrained agent is kept frozen on both paths here.
    * Continuous latents are not bounded to [-1, 1] (the prior is a unit-scale Gaussian),
      so the inner actor's DDPG+BC loss, which clips actions to [-1, 1], is refused; AWR
      is used, and the selected z is never clipped at execution.

Evaluation.
    z ~ pi_psi(. | s, g) is held for `skill_horizon` env steps (default: the checkpoint's
    `chunk_size`, i.e. "once per c steps" in Fig. 2) while the frozen OPAL decoder
    executes a ~ pi_theta(. | s, z) every step, then z is reselected. `evaluate()` picks
    this up from the `init_eval_state` / `sample_actions_with_state` pair. As in the
    sibling controllers, `--eval_temperature` reaches the high level only; the decoder
    runs at `low_temperature`.
"""

import json
import os
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from agents.opal import OPALAgent
from agents.skill_bc_relabel_controller import (
    _DATASET_CONFIG_KEYS,
    BASE_AGENTS,
    _latest_epoch,
)
from utils.flax_utils import nonpytree_field, restore_agent


def _base_config(base_agent_name):
    """The nested `agent.base` config for one inner algorithm.

    Same construction as `skill_bc_relabel_controller._base_config`; `discrete` is set
    here to a default and overwritten in `create` from the checkpoint's `latent_type`.
    """
    if base_agent_name not in BASE_AGENTS:
        raise ValueError(
            f'base_agent_name must be one of {sorted(BASE_AGENTS)}, got {base_agent_name!r}. '
            f'Select it as a config-file argument: --agent=agents/opal_controller.py:<name>'
        )
    module = __import__(f'agents.{base_agent_name}', fromlist=['get_config'])
    base = module.get_config()
    for key in _DATASET_CONFIG_KEYS:
        if key in base:
            del base[key]
    # Overwritten in `create`: True for a discrete OPAL checkpoint, False for a continuous one.
    base.discrete = True
    # AWR supports both a K-way head and an unbounded latent; DDPG+BC clips to [-1, 1].
    base.actor_loss = 'awr'
    # Per-OPTION discount (gamma_hi); see the module docstring.
    base.discount = 0.99
    return base


class OPALControllerAgent(flax.struct.PyTreeNode):
    """OPAL task policy pi_psi(z | s, g) trained on the posterior-labelled option MDP.

    Fields:
        rng: PRNG key (seeds action sampling when no seed is supplied).
        base: The inner goal-conditioned agent over skill actions; the only trained
            component. Holds pi_psi as its `actor` module.
        skill_agent: The frozen pretrained `OPALAgent`. Its posterior labels the windows
            and its decoder pi_theta(a|s,z) is executed at eval. Never updated; saved
            into every checkpoint as a plain pytree field, so `restore_agent` reloads it
            from there rather than from `skill_checkpoint_path`.
        config: Static configuration dictionary.
    """

    rng: Any
    base: Any
    skill_agent: Any
    config: Any = nonpytree_field()

    # ── Step 1: labelling with the OPAL posterior ─────────────────────────────

    @jax.jit
    def label_chunk_skills(self, observations_seq, actions_seq, seq_mask, seed):
        """One skill per window from the frozen posterior: int32 [B] or float32 [B, D]."""
        sample = self.config['label_mode'] == 'sample'
        if self.config['latent_type'] == 'discrete':
            # App. F Eq. 71: Bayes rule over the mixture. Padded steps are masked, so a
            # window cut short by its trajectory end sums fewer terms. This is
            # `OPALAgent._log_p_tau_given_z` with `lax.map` over the K skills instead of
            # `vmap`: the labelling pass pushes tens of thousands of windows at a time,
            # and vmapping would materialise K copies of every hidden activation at once.
            num_skills = int(self.config['num_skills'])
            B, C = seq_mask.shape
            prev = observations_seq[:, :-1]
            deltas = observations_seq[:, 1:] - prev
            step_mask = seq_mask[:, 1:]
            eye = jnp.eye(num_skills)

            def log_p_for_skill(skill):
                zs = jnp.broadcast_to(eye[skill], (B, C - 1, num_skills))
                dist = self.skill_agent.network.select('traj_model')(jnp.concatenate([prev, zs], axis=-1))
                return (dist.log_prob(deltas) * step_mask).sum(axis=-1)  # [B]

            log_p_tau = jax.lax.map(log_p_for_skill, jnp.arange(num_skills))  # [K, B]
            log_prior = jax.nn.log_softmax(self.skill_agent.network.select('skill_prior')())
            log_post = jax.nn.log_softmax(log_prior[:, None] + log_p_tau, axis=0).T  # [B, K]
            if sample:
                labels = jax.random.categorical(seed, log_post, axis=-1)
            else:
                labels = jnp.argmax(log_post, axis=-1)
            return labels.astype(jnp.int32)

        # Sec. 4.2: z ~ q_phi(z|tau). Same parameterisation as `OPALAgent.vae_loss`
        # (raw log_std, std = exp(0.5 * log_std)).
        skill_dim = int(self.config['skill_dim'])
        enc_out = self.skill_agent.network.select('encoder')(observations_seq, actions_seq)
        means = enc_out[..., :skill_dim]
        if not sample:
            return means
        stds = jnp.exp(0.5 * enc_out[..., skill_dim:])
        return means + stds * jax.random.normal(seed, means.shape)

    def prepare_datasets(self, datasets):
        """Run the one-time labelling pass over each dataset. Called by `main.py`."""
        name = self.config['agent_name']
        for i, dataset in enumerate(datasets):
            if not hasattr(dataset, 'relabel_chunk_skills_from_windows'):
                raise TypeError(
                    f'{name} needs a dataset that supports window labelling; set '
                    f"dataset_class='SequenceDataset' (got {type(dataset).__name__})."
                )
            stats = dataset.relabel_chunk_skills_from_windows(
                self,
                seed=int(self.config['label_seed']) + i,
                num_skills=self.config['num_skills'],
                chunk_bytes=int(self.config['label_chunk_bytes']),
            )
            counts = stats.pop('label_counts', None)
            print(
                f'[{name}] labelled {dataset.size} windows with the {self.config["latent_type"]} '
                f'OPAL posterior (c={self.config["chunk_horizon"]}, label_mode='
                f'{self.config["label_mode"]}): ' + ', '.join(f'{k}={v:.3f}' for k, v in stats.items())
            )
            if counts is not None:
                print(f'[{name}]   per-skill counts: {counts.tolist()}')

    # ── Step 2: the option MDP handed to the base algorithm ───────────────────

    def _option_batch(self, batch):
        """Rewrite a `SequenceDataset` batch into one option-level transition.

        Same rewrite as `skill_bc_relabel_controller._option_batch`: padded steps are
        dropped from R_c and never veto the bootstrap.
        """
        if 'chunk_skills' not in batch:
            raise KeyError(
                'batch is missing `chunk_skills`; the one-time labelling pass has not run. '
                '`main.py` drives it via `prepare_datasets` before training.'
            )

        horizon = int(self.config['chunk_horizon'])
        discount = self.config['discount']
        seq_mask = batch['seq_mask']  # [B, H]

        discounts = discount ** jnp.arange(horizon, dtype=jnp.float32)
        snippet_return = (batch['rewards_seq'] * seq_mask * discounts[None, :]).sum(axis=1)
        effective_masks = jnp.where(seq_mask > 0, batch['masks_seq'], 1.0)
        macro_mask = jnp.prod(effective_masks, axis=1)

        option_batch = dict(batch)
        option_batch['actions'] = batch['chunk_skills']
        option_batch['next_observations'] = batch['subgoal_observations']
        option_batch['rewards'] = snippet_return
        option_batch['masks'] = macro_mask
        return option_batch

    def _label_info(self, batch):
        """Diagnostics on the labels in this batch (collapse is invisible to the inner losses)."""
        labels = batch['chunk_skills']
        if self.config['latent_type'] == 'discrete':
            num_skills = self.config['num_skills']
            counts = jnp.bincount(labels, length=num_skills)
            probs = counts / jnp.maximum(counts.sum(), 1)
            return {
                'labels/coverage': (counts > 0).mean(),
                'labels/entropy': -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs + 1e-12), 0.0)),
                'labels/max_frac': probs.max(),
            }
        return {
            'labels/mean_norm': jnp.linalg.norm(labels, axis=-1).mean(),
            'labels/std': labels.std(axis=0).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Inner algorithm's loss on the option MDP (used for validation)."""
        option_batch = self._option_batch(batch)
        loss, info = self.base.total_loss(option_batch, grad_params, rng=rng)
        return loss, {**info, **self._label_info(batch)}

    @jax.jit
    def update(self, batch):
        """One gradient step on the high level. The pretrained OPAL agent is untouched."""
        new_rng, _ = jax.random.split(self.rng)
        option_batch = self._option_batch(batch)
        new_base, info = self.base.update(option_batch)
        return self.replace(base=new_base, rng=new_rng), {**info, **self._label_info(batch)}

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _is_single_obs(self, observations):
        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        return observations.ndim == single_obs_ndim

    def _select_skills(self, observations, goals, seed, temperature):
        """z ~ pi_psi(. | s, g): [B] int index (discrete) or [B, D] latent (continuous)."""
        dist = self.base.network.select('actor')(observations, goals, temperature=temperature)
        skills = dist.sample(seed=seed)
        if self.config['latent_type'] == 'discrete':
            return skills.astype(jnp.int32)
        return skills  # unbounded latent; deliberately not clipped

    def _skill_vectors(self, skills):
        """The skill input the OPAL decoder expects: one-hot (discrete) or z itself."""
        if self.config['latent_type'] == 'discrete':
            return jnp.eye(self.config['num_skills'])[skills]
        return skills

    def _low_level_actions(self, observations, skills, seed):
        """a ~ pi_theta(. | s, z) from the frozen OPAL decoder (clipped to [-1, 1] inside)."""
        return self.skill_agent.sample_actions_with_skill(
            observations, self._skill_vectors(skills), seed=seed, temperature=self.config['low_temperature']
        )

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Hierarchical action with no commitment: z ~ pi_psi(. | s, g), then a ~ pi_theta(. | s, z).

        `evaluate()` dispatches on the committed pair below and never lands here.
        """
        if seed is None:
            seed = self.rng
        high_seed, low_seed = jax.random.split(seed)

        single_obs = self._is_single_obs(observations)
        if single_obs:
            observations = observations[None, ...]
            goals = goals[None, ...] if goals is not None else None

        skills = self._select_skills(observations, goals, high_seed, temperature)
        actions = self._low_level_actions(observations, skills, low_seed)
        return actions[0] if single_obs else actions

    def init_eval_state(self):
        """Per-episode state: the committed skill and the step counter."""
        if self.config['latent_type'] == 'discrete':
            skill = jnp.zeros((), jnp.int32)
        else:
            skill = jnp.zeros((int(self.config['skill_dim']),), jnp.float32)
        return {'skill': skill, 'count': jnp.zeros((), jnp.int32)}

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None,
                                  temperature=1.0):
        """`sample_actions` with the skill held fixed for `skill_horizon` steps.

        Returns `(action, new_state)`; the eval harness threads `new_state` back in.
        """
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()
        high_seed, low_seed = jax.random.split(seed)

        single_obs = self._is_single_obs(observations)
        obs_b = observations[None, ...] if single_obs else observations
        goals_b = goals[None, ...] if (single_obs and goals is not None) else goals

        horizon = int(self.config['skill_horizon'])
        reselect = (agent_state['count'] % horizon) == 0

        sampled = self._select_skills(obs_b, goals_b, high_seed, temperature)
        committed = jnp.broadcast_to(agent_state['skill'], sampled.shape).astype(sampled.dtype)
        skills = jnp.where(reselect, sampled, committed)

        actions = self._low_level_actions(obs_b, skills, low_seed)

        if single_obs:
            actions = actions[0]
            new_skill = skills[0]
        else:
            new_skill = skills
        return actions, {'skill': new_skill, 'count': agent_state['count'] + 1}

    # ── Skill-conditioned evaluation hooks (see eval_skill_policy.py) ─────────
    # Delegated to the frozen OPAL agent so the skill sweep measures the pretrained
    # primitives and bypasses the task policy, as in the sibling controllers.

    def skill_set(self, seed=None, num_skills=None, observations=None):
        return self.skill_agent.skill_set(seed=seed, num_skills=num_skills, observations=observations)

    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        del temperature  # this hook reproduces the pretrained decoder's own execution
        return self.skill_agent.sample_actions_with_skill(
            observations, skills, seed=seed, temperature=self.config['low_temperature']
        )

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of low-level env actions (builds the frozen OPAL agent).
            config: Configuration dictionary. Must contain `skill_checkpoint_path`
                pointing at an `opal` run directory.
        """
        rng = jax.random.PRNGKey(seed)

        base_agent_name = config['base_agent_name']
        expected_keys = set(_base_config(base_agent_name).keys())
        if set(config['base'].keys()) != expected_keys:
            raise ValueError(
                f'agent.base does not have {base_agent_name}\'s key set (missing '
                f'{sorted(expected_keys - set(config["base"].keys()))}, extra '
                f'{sorted(set(config["base"].keys()) - expected_keys)}). Select the base '
                f'algorithm as a config-file argument: --agent=agents/opal_controller.py:{base_agent_name}'
            )

        # ── Load the frozen pretrained OPAL agent. ────────────────────────────
        ckpt_path = config['skill_checkpoint_path']
        if ckpt_path is None:
            raise ValueError('opal_controller requires --agent.skill_checkpoint_path=<opal run dir>.')
        ckpt_path = ckpt_path.rstrip('/')
        flags_path = os.path.join(ckpt_path, 'flags.json')
        if not os.path.exists(flags_path):
            raise FileNotFoundError(f'flags.json not found in {ckpt_path}')
        with open(flags_path) as f:
            opal_flags = json.load(f)
        opal_config = opal_flags['agent']
        if opal_config.get('agent_name') != 'opal':
            raise ValueError(
                f'Expected an opal checkpoint, got agent_name={opal_config.get("agent_name")!r} in {flags_path}'
            )

        # The observation pipeline and env action space must match the pretrained agent's.
        for key in ('encoder', 'frame_stack', 'discrete'):
            if config[key] != opal_config.get(key):
                expected = opal_config.get(key)
                fix = f'omit --agent.{key}' if expected is None else f'pass --agent.{key}={expected!r}'
                raise ValueError(
                    f'{key}={config[key]!r} does not match the pretrained checkpoint\'s '
                    f'{key}={expected!r} ({flags_path}); {fix}.'
                )

        latent_type = opal_config['latent_type']
        if config['latent_type'] is not None and config['latent_type'] != latent_type:
            raise ValueError(
                f'latent_type={config["latent_type"]!r} disagrees with the checkpoint\'s '
                f'{latent_type!r} ({flags_path}); omit the flag, it is read from the checkpoint.'
            )
        num_skills = int(opal_config['num_skills']) if latent_type == 'discrete' else None
        skill_dim = int(opal_config['skill_dim']) if latent_type == 'continuous' else None
        for key, value in (('num_skills', num_skills), ('skill_dim', skill_dim)):
            if config[key] is not None and value is not None and int(config[key]) != value:
                raise ValueError(
                    f'{key}={config[key]} disagrees with the checkpoint\'s {key}={value} '
                    f'({flags_path}); omit the flag, it is read from the checkpoint.'
                )

        # ── Validate the horizons. ────────────────────────────────────────────
        horizon = int(config['chunk_horizon'])
        chunk_size = int(opal_config['chunk_size'])
        if horizon != chunk_size:
            raise ValueError(
                f'chunk_horizon={horizon} must equal the checkpoint\'s chunk_size={chunk_size} '
                f'({flags_path}): the posterior and the decoder were trained on length-c windows. '
                f'Pass --agent.chunk_horizon={chunk_size} (it moves sequence_length with it).'
            )
        if int(config['sequence_length']) != horizon:
            raise ValueError(
                f'sequence_length={config["sequence_length"]} must equal chunk_horizon={horizon}; '
                f'setting either --agent.chunk_horizon or --agent.sequence_length moves both.'
            )
        if config['dataset_class'] != 'SequenceDataset':
            raise ValueError(
                f'dataset_class must be \'SequenceDataset\' (got {config["dataset_class"]!r}): '
                f'the labelling needs c-step windows.'
            )
        skill_horizon = config['skill_horizon']
        skill_horizon = horizon if skill_horizon is None else int(skill_horizon)
        if skill_horizon < 1:
            raise ValueError(f'skill_horizon must be at least 1, got {skill_horizon}.')
        if config['label_mode'] not in ('sample', 'mode'):
            raise ValueError(f'label_mode must be \'sample\' or \'mode\', got {config["label_mode"]!r}.')

        print(
            f'[opal_controller] labelling with {ckpt_path}\n'
            f'[opal_controller]   pretrained env_name={opal_flags.get("env_name")!r}, '
            f'latent_type={latent_type}, num_skills={num_skills}, skill_dim={skill_dim}, '
            f'chunk_size={chunk_size} -- --env_name must match.'
        )

        skill_agent = OPALAgent.create(seed, ex_observations, ex_actions, opal_config)
        restore_epoch = config['skill_restore_epoch']
        if restore_epoch is None:
            restore_epoch = _latest_epoch(ckpt_path)
        skill_agent = restore_agent(skill_agent, ckpt_path, restore_epoch)

        # ── Build the inner goal-conditioned agent over the skill space. ──────
        base_config = config['base'].to_dict() if hasattr(config['base'], 'to_dict') else dict(config['base'])
        base_config['batch_size'] = config['batch_size']
        base_config['encoder'] = config['encoder']
        base_config['frame_stack'] = config['frame_stack']
        base_config['discrete'] = latent_type == 'discrete'
        if latent_type == 'discrete':
            # A discrete agent reads the action-space SIZE off the example's maximum.
            ex_skill_actions = np.full((ex_observations.shape[0],), num_skills - 1, dtype=np.int32)
        else:
            if base_config.get('actor_loss') == 'ddpgbc':
                raise ValueError(
                    'base.actor_loss=\'ddpgbc\' clips actions to [-1, 1], but OPAL\'s continuous '
                    'latent is unbounded; use --agent.base.actor_loss=awr.'
                )
            ex_skill_actions = np.zeros((ex_observations.shape[0], skill_dim), dtype=np.float32)
        base_agent = BASE_AGENTS[base_agent_name].create(
            seed, ex_observations, ex_skill_actions, ml_collections.ConfigDict(base_config)
        )

        # Resolved values for the agent's own use; the run's flags.json keeps the
        # placeholders (main.py serialises FLAGS before `create`).
        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        stored_config['latent_type'] = latent_type
        stored_config['num_skills'] = num_skills
        stored_config['skill_dim'] = skill_dim
        stored_config['skill_restore_epoch'] = restore_epoch
        stored_config['skill_checkpoint_path'] = ckpt_path
        stored_config['skill_horizon'] = skill_horizon

        return cls(
            rng,
            base=base_agent,
            skill_agent=skill_agent,
            config=flax.core.FrozenDict(**stored_config),
        )


def get_config(base_agent_name='gciql'):
    """Config for the agent.

    `base_agent_name` is the config-file argument, so the inner algorithm is chosen at
    launch and its own config is nested under `agent.base`:

        --agent=agents/opal_controller.py:gciql --agent.base.expectile=0.9
        --agent=agents/opal_controller.py:crl   --agent.base.alpha=0.1
    """
    # `sequence_length` is read by SequenceDataset and `chunk_horizon` by this agent; they
    # name the same c, so they share one FieldReference. Must equal the checkpoint's
    # `chunk_size` (checked in `create`).
    chunk_horizon = ml_collections.FieldReference(10)

    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='opal_controller',  # Agent name.
            # Path to a pretrained opal run directory (holds flags.json and params_*.pkl). Required.
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),
            # Epoch of the checkpoint to restore (None -> latest params_*.pkl).
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),
            # Read from the checkpoint's flags.json (the run's own flags.json records null).
            latent_type=ml_collections.config_dict.placeholder(str),  # 'discrete' | 'continuous'
            num_skills=ml_collections.config_dict.placeholder(int),  # K (discrete)
            skill_dim=ml_collections.config_dict.placeholder(int),  # dim(Z) (continuous)
            # c: the window each label covers and the option length of the semi-MDP.
            chunk_horizon=chunk_horizon,
            # Env steps a selected skill is held for at eval (None -> chunk_horizon).
            skill_horizon=ml_collections.config_dict.placeholder(int),
            # 'sample': z ~ posterior (the paper); 'mode': its argmax / mean.
            label_mode='sample',
            label_seed=0,  # Seed of the labelling pass (val dataset uses label_seed + 1).
            # Window bytes per labelling block. The BiGRU encoder / the K-way transition
            # model keep every window's hidden activations live, so keep this modest.
            label_chunk_bytes=8 * 1024 * 1024,
            # Temperature of the frozen OPAL decoder at execution (0 -> mode).
            low_temperature=0.0,
            # Batch size (main.py sizes batches from this; forwarded to the base agent).
            batch_size=1024,
            # ── Inner goal-conditioned algorithm ─────────────────────────────
            base_agent_name=base_agent_name,
            base=_base_config(base_agent_name),
            # ── Dataset hyperparameters ──────────────────────────────────────
            # Per-ENV-step discount: weights R_c and drives GCDataset's geometric goal
            # sampling. The per-OPTION bootstrap discount is `base.discount`.
            discount=0.99,
            dataset_class='SequenceDataset',  # Required: the labelling needs windows.
            sequence_length=chunk_horizon,  # Linked to chunk_horizon; do not set apart.
            # `discrete` / `encoder` / `frame_stack` describe the ENV action space and
            # observation pipeline; they must match the pretrained checkpoint.
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            frame_stack=ml_collections.config_dict.placeholder(int),
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
        )
    )
    return config
