"""
Goal-conditioned high-level controller over a frozen, pretrained Skill-DT.

Skill Decision Transformer (Sudhakaran & Risi, arXiv:2301.13573, `agents/skill_dt.py`)
is reward-free: it discovers N discrete skills (a VQ-VAE codebook over states) and a
causal Transformer pi(a_t | Z, z, s) that follows whichever skill its future-skill
histogram Z points at. The paper has NO high level: it evaluates by rolling out every
skill and reporting the best (Sec. 5.2, A.5), and names a hierarchical stage that
re-uses the skills on downstream tasks as future work (Sec. 6.4). This agent is that
stage, built exactly like the sibling controllers (`skill_bc_relabel_controller`,
`opal_controller`, `dds_controller`): a finished `skill_dt` run is loaded read-only,
the offline dataset is turned into an OPTION-level dataset over the N codebook indices,

    D_H = { (s_t,  k*(t),  sum_{i<H} gamma^i r_{t+i},  s_{t+H}) },

and a standard OGBench goal-conditioned algorithm (`gciql` by default, `crl` also)
is trained on it, unchanged, as pi_hi(k | s, g).

Step 1 -- labelling every start index with a codebook skill (`prepare_datasets`).
    The frozen VQ encoder assigns every state of the dataset its skill index z_t
    (`SkillDTAgent.encode_skill_indices`, the same pass `SequenceDataset.
    relabel_skill_histograms` runs during Skill-DT training). All three label
    definitions below are read off those per-state indices, so the whole pass costs one
    encoder forward over the dataset and no window-level model call:

      label_mode='window_mode' (default)
          k*(t) = argmax_k  sum_{i<H, t+i in traj} 1[z_{t+i} = k],
          the most frequent skill over the option's own H steps. This is the skill the
          window is "in" -- the discrete analogue of DDS's encoder label -- and the label
          under which the executed option (Z's unvisited tail filled with k, below) most
          closely reproduces the window's own future-skill statistic.
      label_mode='end_state'
          k*(t) = z_{t+H}, the skill of the state the option lands in (the bootstrap
          state). Skill-DT's codes are STATE clusters (the encoder sees one state), so
          on the mazes a skill is a region and this label reads "the option led to
          region k".
      label_mode='future_hist'
          k*(t) = argmax_k Z_t, the mode of the paper's own conditioning statistic, the
          normalized skill counts over the rest of the TRAJECTORY (Sec. 4.1). It is what
          the policy was trained to condition on, but on 1000-step -navigate
          trajectories it says little about the next H steps.

    Ties break towards the lowest index. The pass runs ONCE, before training, from
    `main.py` via `prepare_datasets`: the encoder is frozen, so the labels are final.

Step 2 -- semi-MDP high level.
    Identical to the sibling controllers (read `skill_bc_relabel_controller`'s
    docstring for the reasoning): one transition per window, s_t --k--> s_{t+H},
    R_H = sum_{i<H} gamma^i r_{t+i} with the goal-conditioned per-step reward from
    `GCDataset`, bootstrap at s_{t+H} discounted by the base algorithm's own `discount`
    (gamma_hi, applied once per OPTION). The batch rewrite

        actions           <- chunk_skills        (int32 codebook index in [0, N))
        next_observations <- subgoal_observations (s_{t+H})
        rewards           <- R_H
        masks             <- macro mask over the window

    makes `gciql.critic_loss` compute Q(s_t, k, g) <- R_H + gamma_hi * mask * V(s_{t+H}, g).
    Windows start at every index (stride 1, overlapping), as in the siblings.

Evaluation.
    k ~ pi_hi(. | s, g) is drawn every `skill_horizon` env steps (default: the label
    window H) and executed by the frozen Skill-DT exactly as in the paper's Sec. A.5
    rollout (`SkillDTAgent.sample_actions_with_state`): the length-K context of states
    and re-encoded skills is kept across skill switches, and only the unvisited tail of
    the histogram, (L-1-t) * one_hot(k) over the remaining episode horizon L, is
    rebuilt for the new k. So within an option the policy sees precisely what A.5's
    per-skill sweep shows it, and the timestep embedding keeps counting env steps.
    `evaluate()` picks this up from the `init_eval_state` / `sample_actions_with_state`
    pair; `init_eval_state` takes the env horizon exactly as Skill-DT's does.

    As in the siblings, `--eval_temperature` reaches the high level only; the
    Transformer runs at `low_temperature` (0 = its deterministic mean, the paper's
    protocol). `eval_max_steps` overrides the checkpoint's histogram horizon if set
    (see `skill_dt.py`'s Z_t normalization caveat for the -stitch datasets).

    The `skill_set` / `init_eval_state_with_skill` / `sample_actions_with_skill_state`
    hooks are delegated to the frozen Skill-DT, so `eval_skill_policy.py` on a controller
    run measures the pretrained skills (the paper's max-over-skills number) and bypasses
    the high level, as in the sibling controllers.
"""

import json
import os
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from agents.skill_bc_relabel_controller import (
    BASE_AGENTS,
    _base_config,
    _latest_epoch,
)
from agents.skill_dt import SkillDTAgent
from utils.flax_utils import nonpytree_field, restore_agent

LABEL_MODES = ('window_mode', 'end_state', 'future_hist')


class SkillDTControllerAgent(flax.struct.PyTreeNode):
    """High-level policy pi_hi(k | s, g) over a frozen Skill-DT's N codebook skills.

    Fields:
        rng: PRNG key (seeds action sampling when no seed is supplied).
        base: The inner goal-conditioned agent over the N codebook indices; the only
            trained component. Holds pi_hi as its `actor` module.
        skill_agent: The frozen pretrained `SkillDTAgent` (VQ encoder + codebook +
            causal Transformer). Labels the dataset and executes skills at eval. Never
            updated; saved into every checkpoint as a plain pytree field, so
            `restore_agent` reloads it from there rather than from
            `skill_checkpoint_path`.
        config: Static configuration dictionary.
    """

    rng: Any
    base: Any
    skill_agent: Any
    config: Any = nonpytree_field()

    # ── Step 1: labelling with the frozen VQ encoder ──────────────────────────

    def encode_skill_indices(self, observations):
        """Discrete skill index of each state under the frozen encoder: [B] int32.

        This is the hook `SequenceDataset.relabel_skill_histograms` calls. `main.py`'s
        in-training re-labelling loop keys off `relabel_interval`, which this config
        deliberately lacks, so the pass runs only from `prepare_datasets`.
        """
        return self.skill_agent.encode_skill_indices(observations)

    def _labels_from_counts(self, dataset):
        """Per-start-index labels from the dataset's forward-cumulative skill counts.

        `relabel_skill_histograms` stores C[i] = sum_{j<i} one_hot(z_j) with a leading
        zero row, so the counts over any index range [a, b) are C[b] - C[a].
        """
        C = dataset.skill_cumcounts  # [size + 1, N] int32
        H = int(self.config['chunk_horizon'])
        idxs = np.arange(dataset.size)
        final = dataset.terminal_locs[np.searchsorted(dataset.terminal_locs, idxs)]
        mode = self.config['label_mode']
        if mode == 'window_mode':
            stop = np.minimum(idxs + H, final + 1)  # exclusive; clamped to the trajectory
            counts = C[stop] - C[idxs]
        elif mode == 'end_state':
            end = np.minimum(idxs + H, final)       # s_{t+H}, clamped to the terminal
            counts = C[end + 1] - C[end]
        elif mode == 'future_hist':
            counts = C[final + 1] - C[idxs]         # the paper's Z_t, unnormalized
        else:
            raise ValueError(f'label_mode must be one of {LABEL_MODES}, got {mode!r}.')
        return np.argmax(counts, axis=1).astype(np.int32)

    def prepare_datasets(self, datasets):
        """Run the one-time labelling pass over each dataset. Called by `main.py`."""
        name = self.config['agent_name']
        num_skills = int(self.config['num_skills'])
        for dataset in datasets:
            if not hasattr(dataset, 'relabel_skill_histograms') or not hasattr(dataset, 'set_chunk_skills'):
                raise TypeError(
                    f'{name} needs a dataset that supports per-state skill counts; set '
                    f"dataset_class='SequenceDataset' (got {type(dataset).__name__})."
                )
            dataset.relabel_skill_histograms(
                self, chunk_bytes=int(self.config['label_chunk_bytes']), num_skills=num_skills
            )
            stats = dataset.set_chunk_skills(self._labels_from_counts(dataset), num_skills=num_skills)
            counts = stats.pop('label_counts', None)
            print(
                f'[{name}] labelled {dataset.size} windows with the Skill-DT encoder '
                f'(H={self.config["chunk_horizon"]}, N={num_skills}, label_mode='
                f'{self.config["label_mode"]}): ' + ', '.join(f'{k}={v:.3f}' for k, v in stats.items())
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
        """One gradient step on the high level. The pretrained Skill-DT is untouched."""
        new_rng, _ = jax.random.split(self.rng)
        option_batch = self._option_batch(batch)
        new_base, info = self.base.update(option_batch)
        return self.replace(base=new_base, rng=new_rng), {**info, **self._label_info(batch)}

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _is_single_obs(self, observations):
        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        return observations.ndim == single_obs_ndim

    def _select_skills(self, observations, goals, seed, temperature):
        """k ~ pi_hi(. | s, g) from the inner agent's actor: [B] int32 codebook index."""
        dist = self.base.network.select('actor')(observations, goals, temperature=temperature)
        return dist.sample(seed=seed).astype(jnp.int32)

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Hierarchical action with no commitment and NO context: a degraded path.

        k ~ pi_hi(. | s, g), then the Transformer is run on the single triple
        [one_hot(k), codebook[k], s] -- Skill-DT's own stateless `sample_actions`, with
        the skill chosen rather than drawn. `evaluate()` dispatches on the committed,
        context-keeping pair below and never lands here; this exists only for callers
        that cannot thread per-episode state.
        """
        if seed is None:
            seed = self.rng
        high_seed, low_seed = jax.random.split(seed)

        single_obs = self._is_single_obs(observations)
        if single_obs:
            observations = observations[None, ...]
            goals = goals[None, ...] if goals is not None else None

        skills = self._select_skills(observations, goals, high_seed, temperature)
        num_skills = int(self.config['num_skills'])
        batch_size = skills.shape[0]
        z_q = self.skill_agent._codebook()[skills][:, None, :]      # [B, 1, code]
        skill_hist = jax.nn.one_hot(skills, num_skills)[:, None, :]  # [B, 1, N]
        timesteps = jnp.zeros((batch_size, 1), jnp.int32)
        dist = self.skill_agent.network.select('policy')(
            observations[:, None, ...], z_q, skill_hist, timesteps, None,
            temperature=self.config['low_temperature'],
        )
        actions = dist.sample(seed=low_seed)[:, 0]
        if not self.skill_agent.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions[0] if single_obs else actions

    def init_eval_state(self, max_steps=None):
        """Per-episode state: Skill-DT's own rollout state, pinned to skill 0 until the
        first decision at step 0 overwrites it.

        `max_steps` is the env horizon `utils.evaluation` hands over; it sizes the
        histogram tail exactly as for a plain Skill-DT rollout (`eval_max_steps` in the
        checkpoint's config wins if set).
        """
        return self.skill_agent.init_eval_state(skill=0, max_steps=max_steps)

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None,
                                  temperature=1.0):
        """Committed hierarchical action: reselect k every `skill_horizon` steps, execute
        it with the paper's Sec. A.5 rollout, keeping the Transformer context across
        skill switches. Returns `(action, new_state)`.
        """
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()
        high_seed, low_seed = jax.random.split(seed)

        if not self._is_single_obs(observations):
            raise ValueError(
                f'{self.config["agent_name"]}: the stateful rollout keeps one episode\'s context '
                f'buffers, so it needs a single observation, got shape {observations.shape}.'
            )
        obs_b = observations[None, ...]
        goals_b = goals[None, ...] if goals is not None else None

        num_skills = int(self.config['num_skills'])
        horizon = int(self.config['skill_horizon'])
        L = agent_state['tail_suffix'].shape[0] - 1  # episode horizon (static)
        reselect = (agent_state['count'] % horizon) == 0

        sampled = self._select_skills(obs_b, goals_b, high_seed, temperature)[0]
        skill = jnp.where(reselect, sampled, agent_state['skill'])
        # Only the unvisited tail of the histogram changes with the skill; the observed
        # context (states, re-encoded skills, step counter) carries over.
        tail_suffix = jnp.where(
            reselect,
            self.skill_agent._constant_skill_tail(skill, num_skills, L),
            agent_state['tail_suffix'],
        )
        state = {**agent_state, 'skill': skill.astype(jnp.int32), 'tail_suffix': tail_suffix}

        return self.skill_agent.sample_actions_with_state(
            observations, goals=None, agent_state=state, seed=low_seed,
            temperature=self.config['low_temperature'],
        )

    # ── Skill-conditioned evaluation hooks (see eval_skill_policy.py) ─────────
    # Delegated to the frozen Skill-DT so the skill sweep measures the pretrained
    # codebook skills and bypasses the high level, as in the sibling controllers.

    def skill_set(self, seed=None, num_skills=None, observations=None):
        return self.skill_agent.skill_set(seed=seed, num_skills=num_skills, observations=observations)

    def init_eval_state_with_skill(self, skill, max_steps=None):
        return self.skill_agent.init_eval_state_with_skill(skill, max_steps=max_steps)

    def sample_actions_with_skill_state(self, observations, skills, agent_state=None, seed=None,
                                        temperature=1.0):
        del temperature  # this hook reproduces the pretrained policy's own execution
        return self.skill_agent.sample_actions_with_skill_state(
            observations, skills, agent_state=agent_state, seed=seed,
            temperature=self.config['low_temperature'],
        )

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of low-level env actions (builds the frozen Skill-DT).
            config: Configuration dictionary. Must contain `skill_checkpoint_path`
                pointing at a `skill_dt` run directory.
        """
        rng = jax.random.PRNGKey(seed)

        base_agent_name = config['base_agent_name']
        expected_keys = set(_base_config(base_agent_name).keys())
        if set(config['base'].keys()) != expected_keys:
            raise ValueError(
                f'agent.base does not have {base_agent_name}\'s key set (missing '
                f'{sorted(expected_keys - set(config["base"].keys()))}, extra '
                f'{sorted(set(config["base"].keys()) - expected_keys)}). Select the base '
                f'algorithm as a config-file argument: '
                f'--agent=agents/skill_dt_controller.py:{base_agent_name}'
            )
        if config['label_mode'] not in LABEL_MODES:
            raise ValueError(f'label_mode must be one of {LABEL_MODES}, got {config["label_mode"]!r}.')

        # ── Load the frozen pretrained Skill-DT. ──────────────────────────────
        ckpt_path = config['skill_checkpoint_path']
        if ckpt_path is None:
            raise ValueError('skill_dt_controller requires --agent.skill_checkpoint_path=<skill_dt run dir>.')
        ckpt_path = ckpt_path.rstrip('/')
        flags_path = os.path.join(ckpt_path, 'flags.json')
        if not os.path.exists(flags_path):
            raise FileNotFoundError(f'flags.json not found in {ckpt_path}')
        with open(flags_path) as f:
            skill_flags = json.load(f)
        skill_config = skill_flags['agent']
        if skill_config.get('agent_name') != 'skill_dt':
            raise ValueError(
                f'Expected a skill_dt checkpoint, got agent_name={skill_config.get("agent_name")!r} '
                f'in {flags_path}'
            )

        # The observation pipeline and env action space must match the pretrained agent's.
        for key in ('encoder', 'frame_stack', 'discrete'):
            if config[key] != skill_config.get(key):
                expected = skill_config.get(key)
                fix = f'omit --agent.{key}' if expected is None else f'pass --agent.{key}={expected!r}'
                raise ValueError(
                    f'{key}={config[key]!r} does not match the pretrained checkpoint\'s '
                    f'{key}={expected!r} ({flags_path}); {fix}.'
                )

        num_skills = int(skill_config['num_skills'])
        if config['num_skills'] is not None and int(config['num_skills']) != num_skills:
            raise ValueError(
                f'num_skills={config["num_skills"]} disagrees with the checkpoint\'s num_skills='
                f'{num_skills} ({flags_path}); omit the flag, it is read from the checkpoint.'
            )

        # ── Validate the horizons. ────────────────────────────────────────────
        # Unlike DDS/OPAL, nothing ties H to the checkpoint: the encoder is per-state
        # and the Transformer context slides, so any option length is admissible.
        horizon = int(config['chunk_horizon'])
        if horizon < 1:
            raise ValueError(f'chunk_horizon must be at least 1, got {horizon}.')
        if int(config['sequence_length']) != horizon:
            raise ValueError(
                f'sequence_length={config["sequence_length"]} must equal chunk_horizon={horizon}; '
                f'setting either --agent.chunk_horizon or --agent.sequence_length moves both.'
            )
        if config['dataset_class'] != 'SequenceDataset':
            raise ValueError(
                f'dataset_class must be \'SequenceDataset\' (got {config["dataset_class"]!r}): '
                f'the option MDP needs H-step windows.'
            )
        skill_horizon = config['skill_horizon']
        skill_horizon = horizon if skill_horizon is None else int(skill_horizon)
        if skill_horizon < 1:
            raise ValueError(f'skill_horizon must be at least 1, got {skill_horizon}.')

        restore_epoch = config['skill_restore_epoch']
        if restore_epoch is None:
            restore_epoch = _latest_epoch(ckpt_path)

        eval_max_steps = config['eval_max_steps']
        if eval_max_steps is None:
            eval_max_steps = skill_config.get('eval_max_steps')

        print(
            f'[skill_dt_controller] labelling with {ckpt_path} (epoch {restore_epoch})\n'
            f'[skill_dt_controller]   pretrained env_name={skill_flags.get("env_name")!r}, '
            f'num_skills={num_skills}, context_len={skill_config.get("context_len")}, '
            f'eval_max_steps={eval_max_steps} -- --env_name must match.'
        )

        skill_agent = SkillDTAgent.create(seed, ex_observations, ex_actions, skill_config)
        skill_agent = restore_agent(skill_agent, ckpt_path, restore_epoch)
        if eval_max_steps != skill_agent.config['eval_max_steps']:
            skill_agent = skill_agent.replace(
                config=flax.core.FrozenDict({**skill_agent.config, 'eval_max_steps': eval_max_steps})
            )

        # ── Build the inner goal-conditioned agent over the N codebook indices. ─
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
        stored_config['skill_restore_epoch'] = restore_epoch
        stored_config['skill_checkpoint_path'] = ckpt_path
        stored_config['skill_horizon'] = skill_horizon
        stored_config['eval_max_steps'] = eval_max_steps

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

        --agent=agents/skill_dt_controller.py:gciql --agent.base.expectile=0.9
        --agent=agents/skill_dt_controller.py:crl   --agent.base.alpha=0.1
    """
    # `sequence_length` is read by SequenceDataset and `chunk_horizon` by this agent; they
    # name the same H, so they share one FieldReference. 20 is Skill-DT's context length K
    # (paper Table 5); nothing forces the two to agree.
    chunk_horizon = ml_collections.FieldReference(20)

    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='skill_dt_controller',  # Agent name.
            # Path to a pretrained skill_dt run directory (holds flags.json and params_*.pkl). Required.
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),
            # Epoch of the checkpoint to restore (None -> latest params_*.pkl).
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),
            # Read from the checkpoint's flags.json (the run's own flags.json records null).
            num_skills=ml_collections.config_dict.placeholder(int),  # N (codebook size)
            # H: the window each label covers and the option length of the semi-MDP.
            chunk_horizon=chunk_horizon,
            # Env steps a selected skill is held for at eval (None -> chunk_horizon).
            skill_horizon=ml_collections.config_dict.placeholder(int),
            # How a window is labelled with a codebook skill: 'window_mode' | 'end_state' |
            # 'future_hist' (module docstring, Step 1).
            label_mode='window_mode',
            # Observation bytes per encoder block in the labelling pass.
            label_chunk_bytes=64 * 1024 * 1024,
            # Episode horizon the rollout histogram is defined over (None -> the checkpoint's
            # own `eval_max_steps`, which in turn falls back to the env horizon).
            eval_max_steps=ml_collections.config_dict.placeholder(int),
            # Temperature of the frozen Transformer at execution (0 -> its deterministic
            # mean, the paper's protocol; only the sampling std / categorical logits read it).
            low_temperature=0.0,
            # Batch size (main.py sizes batches from this; forwarded to the base agent).
            batch_size=256,
            # ── Inner goal-conditioned algorithm ─────────────────────────────
            base_agent_name=base_agent_name,
            base=_base_config(base_agent_name),
            # ── Dataset hyperparameters ──────────────────────────────────────
            # Per-ENV-step discount: weights R_H and drives GCDataset's geometric goal
            # sampling. The per-OPTION bootstrap discount is `base.discount`.
            discount=0.99,
            dataset_class='SequenceDataset',  # Required: the option MDP needs windows.
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
