"""
DDS high-level policy on top of a frozen, pretrained DDS skill model.

This is the "relabel, then train the high level with IQL + AWR" stage of Discrete
Diffusion Skills (Qiao et al. 2025, arXiv:2503.20176, Sec. 4.2-4.4) as a SEPARATE,
pluggable agent. `agents/dds.py` runs the whole paper inside one job (500k steps of
VQ-VAE skill pretraining, a hard freeze, then its own built-in high level); this agent
instead takes a finished `dds` run directory, keeps ONLY its skill model (transformer
encoder, codebook, diffusion decoder), turns the offline dataset into the paper's
option-level dataset

    D_H = { (s_t,  k,  sum_{i<H} gamma^i r_{t+i},  s_{t+H}) },   k = argmin_j ||E(tau_H) - z_j||,

and hands it, unchanged, to a standard OGBench goal-conditioned algorithm whose action
space is the K codebook indices. The paper trains pi_phi(k|s) on D_H with IQL and
extracts it with AWR; the default here is `gciql` (AWR branch, discrete K-way head) with
the paper's Table 7 / A.1 hyperparameters, and the inner algorithm is pluggable
(`base_agent_name`), exactly as in `skill_bc_relabel_controller` and `opal_controller`,
whose structure this file follows.

Step 1 -- relabelling with the encoder (`SequenceDataset.relabel_chunk_skills_from_windows`).
    Every start index t is labelled with the codebook index the frozen encoder assigns to
    its length-H window tau_H = (s_t, a_t, ..., s_{t+H-1}, a_{t+H-1}) (paper Eq. 10-11,
    Sec. 4.2 "the index of the inferred skill is treated as action in D_H"):

        k*(t) = argmin_j || E(tau_H) - z_j ||.

    This is exactly `DDSAgent._assign_skill`, the label `dds.py` computes per batch; here
    it is computed once for the whole dataset because the encoder is frozen, so the
    labels are final. Padded steps past a trajectory's terminal are masked out of the
    encoder's attention and pooling by `seq_mask`, as during pretraining. It is NOT the
    BC-likelihood argmax of `skill_bc_relabel_controller`: the DDS encoder reads the
    state-action window directly and never touches the decoder.

    The pass runs ONCE, before training, from `main.py` via `prepare_datasets`.

Step 2 -- semi-MDP high level (paper Sec. 4.2-4.3, Eq. 6-9).
    Identical to the sibling controllers (read `skill_bc_relabel_controller`'s docstring
    for the reasoning): one transition per window, s_t --k--> s_{t+H}, reward
    R_H = sum_{i<H} gamma^i r_{t+i} with the goal-conditioned per-step reward from
    `GCDataset`, bootstrap at s_{t+H} discounted by the base algorithm's own `discount`
    (gamma_hi, applied once per OPTION). The rewrite is

        actions           <- chunk_skills        (int32 codebook index in [0, K))
        next_observations <- subgoal_observations (s_{t+H})
        rewards           <- R_H
        masks             <- macro mask over the window

    so `gciql.critic_loss` computes Q(s_t, k, g) <- R_H + gamma_hi * mask * V(s_{t+H}, g),
    which is the paper's Eq. 8 on D_H and `dds.high_critic_loss` verbatim. With the
    defaults both discounts are the paper's single gamma = 0.99.

Default high-level hyperparameters (paper Table 7 and A.1.3-A.1.5), applied to `gciql`:
    lr 1e-4, discount 0.99, batch 256, expectile 0.7, EMA tau 0.005, value / Q / policy
    MLPs of 2 x 256 without LayerNorm, AWR temperature alpha = 3.0 (alpha is unspecified
    in the paper; 3.0 is the IQL/AWR convention `dds.py` also uses). One knob the paper
    fixes cannot be reached without editing `gciql.py`: its MLPs use OGBench's GELU where
    the paper says ReLU. The paper also trains Q-learning for 1M steps and AWR for a
    further 500k in separate runs; `gciql` trains value, critic and actor jointly, so
    pick `--train_steps` accordingly.

Departures from the paper, all deliberate.
    * Goal-conditioned rewards from `GCDataset` instead of a single task reward, because
      OGBench evaluates goal-reaching; the high level is therefore pi_phi(k | s, g).
      Same adaptation as `dds.py` (its flag A1).
    * Every start index t yields a window (stride 1, overlapping); the paper's "divided
      into sequences of length H" reads as non-overlapping chunks. Same as `dds.py`.

Evaluation (paper Sec. 4.4).
    k ~ pi_phi(. | s, g) is held for `skill_horizon` env steps (default: the checkpoint's
    H) while the frozen diffusion decoder generates a ~ D(z_k, s) every step with the
    checkpoint's number of denoising steps, then k is reselected. `evaluate()` picks this
    up from the `init_eval_state` / `sample_actions_with_state` pair. As in the sibling
    controllers, `--eval_temperature` reaches the high level only; `low_temperature`
    reaches the decoder, where it matters only for the categorical decoder of
    discrete-action envs (DDPM sampling has no temperature).
"""

import json
import os
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from agents.dds import DDSAgent
from agents.skill_bc_relabel_controller import (
    _DATASET_CONFIG_KEYS,
    BASE_AGENTS,
    _latest_epoch,
)
from utils.flax_utils import nonpytree_field, restore_agent

# Paper Table 7 / A.1.3-A.1.5 values for the IQL high level, keyed by `gciql` config name.
_PAPER_IQL_CONFIG = dict(
    lr=1e-4,                      # Table 7: learning rate.
    expectile=0.7,                # Table 7: tau (IQL expectile).
    tau=0.005,                    # Table 7: EMA alpha (target-network rate).
    alpha=3.0,                    # AWR temperature; unspecified in the paper (see module docstring).
    actor_hidden_dims=(256, 256),  # A.1.5: 2 hidden layers of 256.
    value_hidden_dims=(256, 256),  # A.1.3 / A.1.4: 2 hidden layers of 256 (V and both Qs).
    layer_norm=False,             # A.1: plain MLPs.
)


def _base_config(base_agent_name):
    """The nested `agent.base` config for one inner algorithm.

    Same construction as `skill_bc_relabel_controller._base_config` (a discrete K-way
    action space, AWR, per-OPTION discount), plus the paper's IQL hyperparameters when
    the inner algorithm is `gciql`. Other inner algorithms keep their own defaults.
    """
    if base_agent_name not in BASE_AGENTS:
        raise ValueError(
            f'base_agent_name must be one of {sorted(BASE_AGENTS)}, got {base_agent_name!r}. '
            f'Select it as a config-file argument: --agent=agents/dds_controller.py:<name>'
        )
    module = __import__(f'agents.{base_agent_name}', fromlist=['get_config'])
    base = module.get_config()
    for key in _DATASET_CONFIG_KEYS:
        if key in base:
            del base[key]
    # The high-level action IS a codebook index, so the inner agent is always discrete.
    base.discrete = True
    # 'ddpgbc' asserts a continuous action space; AWR is the paper's extraction (Eq. 9).
    base.actor_loss = 'awr'
    # Per-OPTION discount (gamma_hi). Paper Table 7: 0.99, the same gamma as in R_H.
    base.discount = 0.99
    if base_agent_name == 'gciql':
        for key, value in _PAPER_IQL_CONFIG.items():
            base[key] = value
    return base


class DDSControllerAgent(flax.struct.PyTreeNode):
    """DDS high-level policy pi_phi(k | s, g) trained on the encoder-labelled option MDP.

    Fields:
        rng: PRNG key (seeds action sampling when no seed is supplied).
        base: The inner goal-conditioned agent over the K codebook indices; the only
            trained component. Holds pi_phi as its `actor` module.
        skill_agent: The frozen pretrained `DDSAgent`. Its encoder + codebook label the
            windows and its diffusion decoder executes skills at eval. Its own built-in
            high level (value / high_critic / high_actor) is carried along but never
            read. Never updated; saved into every checkpoint as a plain pytree field, so
            `restore_agent` reloads it from there rather than from `skill_checkpoint_path`.
        config: Static configuration dictionary.
    """

    rng: Any
    base: Any
    skill_agent: Any
    config: Any = nonpytree_field()

    # ── Step 1: relabelling with the frozen encoder ───────────────────────────

    @jax.jit
    def label_chunk_skills(self, observations_seq, actions_seq, seq_mask, seed):
        """One codebook index per window from the frozen encoder: int32 [B] (Eq. 10-11).

        Deterministic (nearest-neighbour argmin); `seed` is part of the dataset's labeller
        interface and unused.
        """
        del seed
        labels = self.skill_agent._assign_skill(observations_seq, actions_seq, seq_mask)
        return labels.astype(jnp.int32)

    def prepare_datasets(self, datasets):
        """Run the one-time relabelling pass over each dataset. Called by `main.py`."""
        name = self.config['agent_name']
        for dataset in datasets:
            if not hasattr(dataset, 'relabel_chunk_skills_from_windows'):
                raise TypeError(
                    f'{name} needs a dataset that supports window labelling; set '
                    f"dataset_class='SequenceDataset' (got {type(dataset).__name__})."
                )
            stats = dataset.relabel_chunk_skills_from_windows(
                self,
                num_skills=self.config['num_skills'],
                chunk_bytes=int(self.config['label_chunk_bytes']),
            )
            counts = stats.pop('label_counts', None)
            print(
                f'[{name}] relabelled {dataset.size} windows with the DDS encoder '
                f'(H={self.config["chunk_horizon"]}, K={self.config["num_skills"]}): '
                + ', '.join(f'{k}={v:.3f}' for k, v in stats.items())
            )
            if counts is not None:
                print(f'[{name}]   per-skill counts: {counts.tolist()}')

    # ── Step 2: the option MDP handed to the base algorithm ───────────────────

    def _option_batch(self, batch):
        """Rewrite a `SequenceDataset` batch into one option-level transition of D_H.

        Same rewrite as `skill_bc_relabel_controller._option_batch`: padded steps are
        dropped from R_H and never veto the bootstrap.
        """
        if 'chunk_skills' not in batch:
            raise KeyError(
                'batch is missing `chunk_skills`; the one-time relabelling pass has not run. '
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
        num_skills = self.config['num_skills']
        counts = jnp.bincount(batch['chunk_skills'], length=num_skills)
        probs = counts / jnp.maximum(counts.sum(), 1)
        return {
            'labels/coverage': (counts > 0).mean(),
            'labels/entropy': -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs + 1e-12), 0.0)),
            'labels/max_frac': probs.max(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Inner algorithm's loss on the option MDP (used for validation)."""
        option_batch = self._option_batch(batch)
        loss, info = self.base.total_loss(option_batch, grad_params, rng=rng)
        return loss, {**info, **self._label_info(batch)}

    @jax.jit
    def update(self, batch):
        """One gradient step on the high level. The pretrained DDS agent is untouched."""
        new_rng, _ = jax.random.split(self.rng)
        option_batch = self._option_batch(batch)
        new_base, info = self.base.update(option_batch)
        return self.replace(base=new_base, rng=new_rng), {**info, **self._label_info(batch)}

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _is_single_obs(self, observations):
        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        return observations.ndim == single_obs_ndim

    def _select_skills(self, observations, goals, seed, temperature):
        """k ~ pi_phi(. | s, g) from the inner agent's actor: [B] int32 codebook index."""
        dist = self.base.network.select('actor')(observations, goals, temperature=temperature)
        return dist.sample(seed=seed).astype(jnp.int32)

    def _low_level_actions(self, observations, skills, seed):
        """a ~ D(z_k, s) from the frozen diffusion decoder (Sec. 4.4); z_k looked up in the codebook."""
        skill_vectors = self.skill_agent._codebook_table()[skills]  # [B, D_z]
        return self.skill_agent.sample_actions_with_skill(
            observations, skill_vectors, seed=seed, temperature=self.config['low_temperature']
        )

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Hierarchical action with no commitment: k ~ pi_phi(. | s, g), then a ~ D(z_k, s).

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
        """Per-episode state: the committed codebook index and the step counter."""
        return {'skill': jnp.zeros((), jnp.int32), 'count': jnp.zeros((), jnp.int32)}

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None,
                                  temperature=1.0):
        """`sample_actions` with the skill held fixed for `skill_horizon` steps (Sec. 4.4).

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
        committed = jnp.broadcast_to(agent_state['skill'], sampled.shape)
        skills = jnp.where(reselect, sampled, committed)

        actions = self._low_level_actions(obs_b, skills, low_seed)

        if single_obs:
            actions = actions[0]
            new_skill = skills[0]
        else:
            new_skill = skills
        return actions, {'skill': new_skill.astype(jnp.int32), 'count': agent_state['count'] + 1}

    # ── Skill-conditioned evaluation hooks (see eval_skill_policy.py) ─────────
    # Delegated to the frozen DDS agent so the skill sweep measures the pretrained
    # codebook skills and bypasses the high level, as in the sibling controllers.

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
            ex_actions: Example batch of low-level env actions (builds the frozen DDS agent).
            config: Configuration dictionary. Must contain `skill_checkpoint_path`
                pointing at a `dds` run directory.
        """
        rng = jax.random.PRNGKey(seed)

        base_agent_name = config['base_agent_name']
        expected_keys = set(_base_config(base_agent_name).keys())
        if set(config['base'].keys()) != expected_keys:
            raise ValueError(
                f'agent.base does not have {base_agent_name}\'s key set (missing '
                f'{sorted(expected_keys - set(config["base"].keys()))}, extra '
                f'{sorted(set(config["base"].keys()) - expected_keys)}). Select the base '
                f'algorithm as a config-file argument: --agent=agents/dds_controller.py:{base_agent_name}'
            )

        # ── Load the frozen pretrained DDS agent. ─────────────────────────────
        ckpt_path = config['skill_checkpoint_path']
        if ckpt_path is None:
            raise ValueError('dds_controller requires --agent.skill_checkpoint_path=<dds run dir>.')
        ckpt_path = ckpt_path.rstrip('/')
        flags_path = os.path.join(ckpt_path, 'flags.json')
        if not os.path.exists(flags_path):
            raise FileNotFoundError(f'flags.json not found in {ckpt_path}')
        with open(flags_path) as f:
            dds_flags = json.load(f)
        dds_config = dds_flags['agent']
        if dds_config.get('agent_name') != 'dds':
            raise ValueError(
                f'Expected a dds checkpoint, got agent_name={dds_config.get("agent_name")!r} in {flags_path}'
            )

        # The observation pipeline and env action space must match the pretrained agent's.
        for key in ('encoder', 'frame_stack', 'discrete'):
            if config[key] != dds_config.get(key):
                expected = dds_config.get(key)
                fix = f'omit --agent.{key}' if expected is None else f'pass --agent.{key}={expected!r}'
                raise ValueError(
                    f'{key}={config[key]!r} does not match the pretrained checkpoint\'s '
                    f'{key}={expected!r} ({flags_path}); {fix}.'
                )

        num_skills = int(dds_config['num_skills'])
        skill_dim = int(dds_config['skill_dim'])
        for key, value in (('num_skills', num_skills), ('skill_dim', skill_dim)):
            if config[key] is not None and int(config[key]) != value:
                raise ValueError(
                    f'{key}={config[key]} disagrees with the checkpoint\'s {key}={value} '
                    f'({flags_path}); omit the flag, it is read from the checkpoint.'
                )

        # ── Validate the horizons. ────────────────────────────────────────────
        horizon = int(config['chunk_horizon'])
        pretrained_horizon = int(dds_config['sequence_length'])
        if horizon != pretrained_horizon:
            raise ValueError(
                f'chunk_horizon={horizon} must equal the checkpoint\'s sequence_length='
                f'{pretrained_horizon} ({flags_path}): the encoder was trained on length-H '
                f'windows (its positional encoding has exactly H slots). Pass '
                f'--agent.chunk_horizon={pretrained_horizon} (it moves sequence_length with it).'
            )
        if int(config['sequence_length']) != horizon:
            raise ValueError(
                f'sequence_length={config["sequence_length"]} must equal chunk_horizon={horizon}; '
                f'setting either --agent.chunk_horizon or --agent.sequence_length moves both.'
            )
        if config['dataset_class'] != 'SequenceDataset':
            raise ValueError(
                f'dataset_class must be \'SequenceDataset\' (got {config["dataset_class"]!r}): '
                f'the relabelling needs H-step windows.'
            )
        skill_horizon = config['skill_horizon']
        skill_horizon = horizon if skill_horizon is None else int(skill_horizon)
        if skill_horizon < 1:
            raise ValueError(f'skill_horizon must be at least 1, got {skill_horizon}.')

        restore_epoch = config['skill_restore_epoch']
        if restore_epoch is None:
            restore_epoch = _latest_epoch(ckpt_path)
        pretrain_steps = int(dds_config['skill_pretrain_steps'])
        if restore_epoch < pretrain_steps:
            raise ValueError(
                f'params_{restore_epoch}.pkl in {ckpt_path} predates the end of DDS skill '
                f'pretraining (skill_pretrain_steps={pretrain_steps}); the skill model is still '
                f'training there. Use an epoch >= {pretrain_steps}.'
            )

        print(
            f'[dds_controller] relabelling with {ckpt_path} (epoch {restore_epoch})\n'
            f'[dds_controller]   pretrained env_name={dds_flags.get("env_name")!r}, '
            f'num_skills={num_skills}, skill_dim={skill_dim}, H={pretrained_horizon}, '
            f'diffusion_steps={dds_config.get("diffusion_steps")} -- --env_name must match.'
        )

        skill_agent = DDSAgent.create(seed, ex_observations, ex_actions, dds_config)
        skill_agent = restore_agent(skill_agent, ckpt_path, restore_epoch)

        # ── Build the inner goal-conditioned agent over the K codebook indices. ─
        base_config = config['base'].to_dict() if hasattr(config['base'], 'to_dict') else dict(config['base'])
        # The inner agent never samples; the dataset keys stripped in `_base_config` are
        # supplied at the top level. `batch_size` is one of them: main.py sizes batches.
        base_config['batch_size'] = config['batch_size']
        base_config['encoder'] = config['encoder']
        base_config['frame_stack'] = config['frame_stack']
        # A discrete agent reads the action-space SIZE off the example's maximum.
        ex_skill_actions = np.full((ex_observations.shape[0],), num_skills - 1, dtype=np.int32)
        base_agent = BASE_AGENTS[base_agent_name].create(
            seed, ex_observations, ex_skill_actions, ml_collections.ConfigDict(base_config)
        )

        # Resolved values for the agent's own use; the run's flags.json keeps the
        # placeholders (main.py serialises FLAGS before `create`).
        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
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

        --agent=agents/dds_controller.py:gciql --agent.base.expectile=0.9
        --agent=agents/dds_controller.py:crl   --agent.base.alpha=0.1
    """
    # `sequence_length` is read by SequenceDataset and `chunk_horizon` by this agent; they
    # name the same H, so they share one FieldReference. Must equal the checkpoint's
    # `sequence_length` (checked in `create`).
    chunk_horizon = ml_collections.FieldReference(10)

    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='dds_controller',  # Agent name.
            # Path to a pretrained dds run directory (holds flags.json and params_*.pkl). Required.
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),
            # Epoch of the checkpoint to restore (None -> latest params_*.pkl). Must be at
            # or after the checkpoint's `skill_pretrain_steps`, i.e. a frozen skill model.
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),
            # Read from the checkpoint's flags.json (the run's own flags.json records null).
            num_skills=ml_collections.config_dict.placeholder(int),  # K (codebook size)
            skill_dim=ml_collections.config_dict.placeholder(int),  # D_z (codebook vector dim)
            # H: the window each label covers and the option length of the semi-MDP.
            chunk_horizon=chunk_horizon,
            # Env steps a selected skill is held for at eval (None -> chunk_horizon, Sec. 4.4).
            skill_horizon=ml_collections.config_dict.placeholder(int),
            # Window bytes per labelling block. The transformer encoder keeps every window's
            # token activations live (4 layers x H tokens x 256-d), so keep this modest.
            label_chunk_bytes=8 * 1024 * 1024,
            # Temperature of the frozen decoder at execution. Only the categorical decoder
            # of discrete-action envs reads it (0 -> mode); DDPM sampling ignores it.
            low_temperature=0.0,
            # Batch size (main.py sizes batches from this; forwarded to the base agent).
            # Paper Table 7: 256.
            batch_size=256,
            # ── Inner goal-conditioned algorithm ─────────────────────────────
            base_agent_name=base_agent_name,
            base=_base_config(base_agent_name),
            # ── Dataset hyperparameters ──────────────────────────────────────
            # Per-ENV-step discount: weights R_H (paper Sec. 4.2) and drives GCDataset's
            # geometric goal sampling. The per-OPTION bootstrap discount is `base.discount`.
            discount=0.99,
            dataset_class='SequenceDataset',  # Required: the relabelling needs windows.
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
