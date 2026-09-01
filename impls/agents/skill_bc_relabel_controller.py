"""
Goal-conditioned skill controller trained on BC-relabelled H-step chunks.

Given a low-level policy pi(a | s, z) pretrained with `empowerment_skill`, this
agent turns the offline dataset into an OPTION-level dataset and hands it,
unchanged, to a standard OGBench goal-conditioned algorithm (`gciql` or `crl`)
whose action space is the K discrete skills.

Step 1 -- chunk relabelling (`SequenceDataset.relabel_chunk_skills`).
    Every start index t is labelled with the single skill that best explains the
    whole length-H window under the frozen policy:

        z*(t) = argmax_z  sum_{i<H} log pi(a_{t+i} | s_{t+i}, z).

    `empowerment_skill`'s actor is a diagonal Gaussian with `const_std=True` and
    no tanh squash, so log pi(a | s, z) = -||pi(s, z) - a||^2 / 2 + const and the
    argmax is exactly the MSE criterion: the skill whose deterministic rollout of
    actions tracks the window's actions most closely. `dist.log_prob` is used
    rather than a literal MSE so the criterion stays correct if the pretrained
    actor ever has a learned or state-dependent std, or a tanh squash.

    This is `agents/skill_match.py`'s criterion widened from a single transition
    to a whole window: matching one action is close to unidentifiable when many
    skills agree at s, whereas agreeing for H consecutive steps is not.

    The pass runs ONCE, before training, because the low-level policy is frozen --
    the labels are exact, not stale. It is driven from `main.py` via
    `prepare_datasets`.

Step 2 -- semi-MDP high level.
    The high level acts once per H env steps, so it is trained on the OPTION MDP
    rather than the env MDP: one transition per window,

        s_t --( z*(t) )--> s_{t+H},   R_H = sum_{i<H} gamma^i r_{t+i},

    with the goal-conditioned per-step reward r_i and a bootstrap at s_{t+H}
    discounted by the base algorithm's own `discount` (gamma_hi), following
    `agents/dds.py` Eq. 8.
    Two reasons this beats relabelling each env transition with its chunk's skill
    and running the base algorithm on the 1-step MDP:

      - it is what eval executes. A 1-step Q(s, z, g) scores holding z for one
        step and then reverting to the data's behaviour, but the controller
        commits z for H steps.
      - gamma_hi is a per-OPTION discount, so the sparse goal-conditioned reward
        has to propagate across ~T/H bootstraps instead of ~T. That horizon
        shortening is the mechanism behind HIQL's gains on these environments,
        and it is lost if the option discount is written as gamma^H (0.99^10 =
        0.90, an effective horizon of ten options). Pass
        `--agent.base.discount=0.904` (i.e. discount ** chunk_horizon) to get the
        discount-consistent variant instead.

    Note the two discounts are deliberately different knobs: the top-level
    `discount` is per ENV step -- it weights R_H and drives GCDataset's geometric
    goal sampling -- while `base.discount` is gamma_hi, applied once per OPTION.

Base algorithms.
    `base_agent_name` selects the inner agent, whose full config is nested under
    `agent.base`. Pick it as a config-file argument:

        --agent=agents/skill_bc_relabel_controller.py:crl  --agent.base.alpha=0.1

    The inner agent is used verbatim -- neither `gciql.py` nor `crl.py` is
    modified -- because the option MDP is expressible entirely as a rewrite of
    the batch:

        actions           <- chunk_skills        (discrete, K skills)
        next_observations <- subgoal_observations (s_{t+H})
        rewards           <- R_H
        masks             <- macro mask over the window

    Substituting those into `gciql.critic_loss` reproduces `dds.high_critic_loss`
    exactly. `crl` needs even less: its contrastive critic is a Monte-Carlo
    classifier over `value_goals` with no TD bootstrap, so it never reads
    `next_observations`/`rewards`/`masks` at all and only the relabelled action
    matters. Its option-level discount therefore lives entirely in the goal
    sampler: raise the top-level `discount` toward `1 - (1 - gamma_hi) / H` if you
    want its future-goal distribution measured in options rather than env steps.

Evaluation.
    z ~ pi_hi(. | s, g) is held for `skill_horizon` env steps (default:
    `chunk_horizon`) while a ~ pi(. | s, z) is executed by the frozen low-level
    policy, then z is reselected. `evaluate()` picks this up automatically from
    the `init_eval_state` / `sample_actions_with_state` pair.

    As in `skill_value_controller`, `--eval_temperature` reaches the CONTROLLER,
    not the low-level actor; the latter always runs at `low_temperature`.
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

from agents.crl import CRLAgent
from agents.empowerment_skill import EmpowermentAgent
from agents.gciql import GCIQLAgent
from utils.flax_utils import nonpytree_field, restore_agent

# Inner goal-conditioned algorithms this agent has been verified against. Adding one
# means checking that the option MDP survives the rewrite in `_option_batch`: an
# algorithm that reads window-level keys, or bootstraps through something other than
# `next_observations`, needs more than a batch rewrite.
BASE_AGENTS = {
    'gciql': GCIQLAgent,
    'crl': CRLAgent,
}

# Config keys of a base algorithm that describe the DATASET rather than the algorithm.
# `main.py` builds the dataset from the top-level config, so leaving these in the
# nested config would create a second, silently ignored copy of every goal-sampling
# probability. They are stripped instead, leaving exactly one place to set them.
_DATASET_CONFIG_KEYS = (
    'dataset_class',
    'value_p_curgoal',
    'value_p_trajgoal',
    'value_p_randomgoal',
    'value_geom_sample',
    'actor_p_curgoal',
    'actor_p_trajgoal',
    'actor_p_randomgoal',
    'actor_geom_sample',
    'gc_negative',
    'p_aug',
    'frame_stack',
    'batch_size',
)


def _latest_epoch(run_dir):
    """Return the largest epoch among params_*.pkl checkpoints in `run_dir`."""
    epochs = [
        int(m.group(1))
        for path in glob.glob(os.path.join(run_dir, 'params_*.pkl'))
        if (m := re.search(r'params_(\d+)\.pkl$', os.path.basename(path)))
    ]
    if not epochs:
        raise FileNotFoundError(f'No params_*.pkl checkpoint found in {run_dir}')
    return max(epochs)


def _base_config(base_agent_name):
    """The nested `agent.base` config for one inner algorithm.

    Starts from the inner agent's own `get_config()` so its defaults and doc comments
    are inherited, then applies the three things the option MDP forces:
    a discrete K-way action space, an actor loss that supports one (DDPG+BC asserts
    continuous actions), and a `discount` that is the per-OPTION gamma_hi.
    """
    if base_agent_name not in BASE_AGENTS:
        raise ValueError(
            f'base_agent_name must be one of {sorted(BASE_AGENTS)}, got {base_agent_name!r}. '
            f'Select it as a config-file argument: '
            f'--agent=agents/skill_bc_relabel_controller.py:<name>'
        )
    module = __import__(f'agents.{base_agent_name}', fromlist=['get_config'])
    base = module.get_config()
    for key in _DATASET_CONFIG_KEYS:
        if key in base:
            del base[key]
    # The high-level action IS a skill index, so the inner agent is always discrete.
    base.discrete = True
    # 'ddpgbc' asserts a continuous action space; AWR is the discrete-action branch.
    base.actor_loss = 'awr'
    # Per-OPTION discount (gamma_hi). Read the module docstring before changing it:
    # this is deliberately NOT the per-step discount raised to chunk_horizon.
    base.discount = 0.99
    return base


class SkillBCRelabelControllerAgent(flax.struct.PyTreeNode):
    """High-level skill controller trained on BC-relabelled H-step chunks.

    Fields:
        rng: PRNG key (seeds action sampling when no seed is supplied).
        base: The inner goal-conditioned agent (`GCIQLAgent` or `CRLAgent`) over
            discrete skill actions. This is the only trained component; it holds
            pi_hi(z | s, g) as its `actor` module.
        skill_agent: The frozen pretrained `empowerment_skill` agent. Its policy
            pi(a | s, z) supplies the relabelling likelihoods and is executed at
            eval time. Never updated. It is a plain pytree field, so `save_agent`
            writes a full copy of it into every checkpoint and a later
            `restore_agent` reloads it from there rather than from
            `skill_checkpoint_path`.
        config: Static configuration dictionary.
    """

    rng: Any
    base: Any
    skill_agent: Any
    config: Any = nonpytree_field()

    # ── Step 1: chunk relabelling ─────────────────────────────────────────────

    @jax.jit
    def chunk_skill_logliks(self, observations, actions):
        """Per-step log pi(a | s, z) for every skill: [B, K].

        The dataset turns these into window sums with a prefix sum (see
        `SequenceDataset.relabel_chunk_skills`), which is why this is per-step
        rather than per-window.

        `lax.map` rather than `vmap` over the K skills: the relabelling pass uses
        blocks of up to 100k states, and vmapping would materialise K copies of
        every hidden activation at once.
        """
        num_skills = self.config['num_skills']
        batch_size = jax.tree_util.tree_leaves(observations)[0].shape[0]
        eye = jnp.eye(num_skills)

        if self.skill_agent.config['discrete']:
            targets = actions
        else:
            # A tanh-squashed actor's log_prob is +inf at |a| == 1, and OGBench action
            # arrays do contain exactly +-1 from clipping. Harmless for the Gaussian
            # actor `empowerment_skill` actually builds; cheap insurance if that changes.
            targets = jnp.clip(actions, -1.0 + 1e-6, 1.0 - 1e-6)

        def loglik_for_skill(skill):
            skills_onehot = jnp.broadcast_to(eye[skill], (batch_size, num_skills))
            dist = self.skill_agent.network.select('policy')(observations, skills_onehot)
            return dist.log_prob(targets)  # [B]

        return jax.lax.map(loglik_for_skill, jnp.arange(num_skills)).T  # [B, K]

    def prepare_datasets(self, datasets):
        """Run the one-time chunk relabelling over each dataset. Called by `main.py`.

        Not jitted and not part of training: the low-level policy is frozen, so the
        labels this writes are final.
        """
        for dataset in datasets:
            if not hasattr(dataset, 'relabel_chunk_skills'):
                raise TypeError(
                    f'{self.config["agent_name"]} needs a dataset that supports chunk '
                    f"re-labelling; set dataset_class='SequenceDataset' (got "
                    f'{type(dataset).__name__}).'
                )
            stats = dataset.relabel_chunk_skills(self)
            counts = stats.pop('label_counts')
            print(
                f'[{self.config["agent_name"]}] relabelled {dataset.size} windows '
                f'(H={self.config["chunk_horizon"]}, K={self.config["num_skills"]}): '
                + ', '.join(f'{k}={v:.3f}' for k, v in stats.items())
            )
            print(f'[{self.config["agent_name"]}]   per-skill counts: {counts.tolist()}')

    # ── Step 2: the option MDP handed to the base algorithm ───────────────────

    def _option_batch(self, batch):
        """Rewrite a `SequenceDataset` batch into one option-level transition.

        The base algorithm then computes the semi-MDP target without knowing it:
        `gciql.critic_loss` becomes
            Q(s_t, z, g) <- R_H + gamma_hi * macro_mask * V(s_{t+H}, g),
        which is `dds.high_critic_loss` (Eq. 8).

        Steps padded past the trajectory's terminal are dropped from R_H rather
        than counted, and they never veto the bootstrap -- the option is simply
        shorter there. Same treatment as `dds.py`.
        """
        if 'chunk_skills' not in batch:
            raise KeyError(
                'batch is missing `chunk_skills`; the one-time relabelling pass has not '
                'run. `main.py` drives it via `prepare_datasets` before training.'
            )

        horizon = int(self.config['chunk_horizon'])
        discount = self.config['discount']
        seq_mask = batch['seq_mask']  # [B, H]

        discounts = discount ** jnp.arange(horizon, dtype=jnp.float32)
        snippet_return = (batch['rewards_seq'] * seq_mask * discounts[None, :]).sum(axis=1)
        # Cut the bootstrap iff the goal was reached at some real step inside the window.
        effective_masks = jnp.where(seq_mask > 0, batch['masks_seq'], 1.0)
        macro_mask = jnp.prod(effective_masks, axis=1)

        option_batch = dict(batch)
        option_batch['actions'] = batch['chunk_skills']
        option_batch['next_observations'] = batch['subgoal_observations']
        option_batch['rewards'] = snippet_return
        option_batch['masks'] = macro_mask
        return option_batch

    def _label_info(self, batch):
        """Diagnostics on the skill labels in this batch.

        A relabelling that collapses onto one skill makes the high level's action
        space effectively empty while every inner loss keeps looking healthy, so
        these are worth watching alongside the base algorithm's own metrics.
        """
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
        """One gradient step on the high level. The pretrained policy is untouched."""
        new_rng, _ = jax.random.split(self.rng)
        option_batch = self._option_batch(batch)
        new_base, info = self.base.update(option_batch)
        return self.replace(base=new_base, rng=new_rng), {**info, **self._label_info(batch)}

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _low_level_actions(self, observations, skills_onehot, seed):
        """a ~ pi(. | s, z) from the frozen low-level policy."""
        # `SkillConditionedDiscreteActor` divides logits by the temperature with no
        # clamp, so a literal 0 would give all-NaN logits and an action index of -1 on
        # every step. Floor it there only: the continuous actor scales by the
        # temperature, where an exact 0 is the intended mode.
        temperature = self.config['low_temperature']
        if self.skill_agent.config['discrete']:
            temperature = max(float(temperature), 1e-6)
        dist = self.skill_agent.network.select('policy')(
            observations, skills_onehot, temperature=temperature
        )
        actions = dist.sample(seed=seed)
        if not self.skill_agent.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    def _is_single_obs(self, observations):
        """A single state-based obs is 1D; a single visual obs is 3D (HWC)."""
        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        return observations.ndim == single_obs_ndim

    def _select_skills(self, observations, goals, seed, temperature):
        """z ~ pi_hi(. | s, g) from the inner agent's actor."""
        dist = self.base.network.select('actor')(observations, goals, temperature=temperature)
        return dist.sample(seed=seed)  # [B] int

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Hierarchical action with no commitment: z ~ pi_hi(. | s, g), then a ~ pi(. | s, z).

        `evaluate()` dispatches on the committed pair below and never lands here; this
        is for other callers. At `skill_horizon == 1` the two agree.
        """
        if seed is None:
            seed = self.rng
        high_seed, low_seed = jax.random.split(seed)

        single_obs = self._is_single_obs(observations)
        if single_obs:
            observations = observations[None, ...]
            goals = goals[None, ...] if goals is not None else None

        skills = self._select_skills(observations, goals, high_seed, temperature)
        skills_onehot = jnp.eye(self.config['num_skills'])[skills]
        actions = self._low_level_actions(observations, skills_onehot, low_seed)

        return actions[0] if single_obs else actions

    # ── H-step skill commitment at eval ───────────────────────────────────────
    # `evaluate()` picks up this pair automatically and threads the state through the
    # rollout, so the executed policy matches the option MDP the high level was
    # trained on.

    def init_eval_state(self):
        """Per-episode state: the committed skill and the step counter."""
        return {'skill': jnp.zeros((), jnp.int32), 'count': jnp.zeros((), jnp.int32)}

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
        committed = jnp.broadcast_to(agent_state['skill'], sampled.shape)
        skills = jnp.where(reselect, sampled, committed)

        skills_onehot = jnp.eye(self.config['num_skills'])[skills]
        actions = self._low_level_actions(obs_b, skills_onehot, low_seed)

        if single_obs:
            actions = actions[0]
            new_skill = skills[0]
        else:
            new_skill = skills
        new_state = {'skill': new_skill.astype(jnp.int32), 'count': agent_state['count'] + 1}
        return actions, new_state

    # ── Skill-conditioned evaluation hooks (see eval_skill_policy.py) ─────────
    #
    # Delegated to the frozen agent, so the skill sweep measures the pretrained
    # low-level policy and bypasses the controller entirely -- which is what that
    # sweep is for. Same choice as `skill_match` / `skill_value_controller`.

    def skill_set(self, seed=None, num_skills=None, observations=None):
        return self.skill_agent.skill_set(seed=seed, num_skills=num_skills, observations=observations)

    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        # `temperature` is dropped for `low_temperature`: this hook reproduces the
        # pretrained policy's own execution.
        del temperature
        low_temperature = self.config['low_temperature']
        if self.skill_agent.config['discrete']:
            low_temperature = max(float(low_temperature), 1e-6)
        return self.skill_agent.sample_actions_with_skill(
            observations, skills, seed=seed, temperature=low_temperature
        )

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of low-level env actions (used only to build
                the frozen pretrained agent).
            config: Configuration dictionary. Must contain `skill_checkpoint_path`
                pointing at an `empowerment_skill` run directory.
        """
        rng = jax.random.PRNGKey(seed)

        base_agent_name = config['base_agent_name']
        if base_agent_name not in BASE_AGENTS:
            raise ValueError(
                f'base_agent_name must be one of {sorted(BASE_AGENTS)}, got '
                f'{base_agent_name!r}.'
            )
        # `get_config` builds the nested config from the config-file argument, so a run
        # launched (or resumed) without `:<name>` would carry gciql's key set while
        # `base_agent_name` says crl -- and the inner agent would be constructed from
        # the wrong knobs. Compare key sets rather than `base.agent_name`, which
        # `main.py`'s resume path restores from flags.json and would agree spuriously.
        expected_keys = set(_base_config(base_agent_name).keys())
        if set(config['base'].keys()) != expected_keys:
            raise ValueError(
                f'agent.base does not have {base_agent_name}\'s key set (missing '
                f'{sorted(expected_keys - set(config["base"].keys()))}, extra '
                f'{sorted(set(config["base"].keys()) - expected_keys)}). Select the base '
                f'algorithm as a config-file argument: '
                f'--agent=agents/skill_bc_relabel_controller.py:{base_agent_name}'
            )

        # ── Load the frozen pretrained empowerment_skill agent. ───────────────
        ckpt_path = config['skill_checkpoint_path']
        if ckpt_path is None:
            raise ValueError(
                'skill_bc_relabel_controller requires '
                '--agent.skill_checkpoint_path=<empowerment_skill run dir>.'
            )
        ckpt_path = ckpt_path.rstrip('/')
        flags_path = os.path.join(ckpt_path, 'flags.json')
        if not os.path.exists(flags_path):
            raise FileNotFoundError(f'flags.json not found in {ckpt_path}')
        with open(flags_path) as f:
            emp_flags = json.load(f)
        emp_config = emp_flags['agent']
        if emp_config.get('agent_name') != 'empowerment_skill':
            raise ValueError(
                f'Expected an empowerment_skill checkpoint, got agent_name='
                f'{emp_config.get("agent_name")!r} in {flags_path}'
            )

        # This agent's observation pipeline has to match the pretrained one:
        # `sample_actions*` reads *this* config's `encoder` to decide whether an
        # unbatched obs is 1D or 3D, then hands the obs to the *pretrained* network;
        # `frame_stack` sets the observation width that network was built for; and
        # main.py fills the example actions from *this* config's `discrete`, which is
        # what EmpowermentAgent.create infers the pretrained action_dim from.
        for key in ('encoder', 'frame_stack', 'discrete'):
            if config[key] != emp_config.get(key):
                expected = emp_config.get(key)
                fix = (f'omit --agent.{key}' if expected is None
                       else f'pass --agent.{key}={expected!r}')
                raise ValueError(
                    f'{key}={config[key]!r} does not match the pretrained checkpoint\'s '
                    f'{key}={expected!r} ({flags_path}); {fix}.'
                )

        # ── Validate the horizons. ────────────────────────────────────────────
        horizon = int(config['chunk_horizon'])
        if horizon < 1:
            raise ValueError(f'chunk_horizon must be at least 1, got {horizon}.')
        # `sequence_length` is what SequenceDataset reads, and main.py builds the dataset
        # before this runs. get_config links the two through a shared FieldReference, so
        # either flag moves both and they cannot diverge from a launch. This guards the
        # resume path, where `restore_config` writes each key from flags.json separately.
        if int(config['sequence_length']) != horizon:
            raise ValueError(
                f'sequence_length={config["sequence_length"]} must equal '
                f'chunk_horizon={horizon} -- the window the labels are fitted over is the '
                f'option the high level commits to. Setting either --agent.chunk_horizon '
                f'or --agent.sequence_length moves both; they can only disagree if a '
                f'resumed flags.json recorded them apart.'
            )
        if config['dataset_class'] != 'SequenceDataset':
            raise ValueError(
                f'dataset_class must be \'SequenceDataset\' (got '
                f'{config["dataset_class"]!r}): the relabelling needs H-step windows.'
            )
        skill_horizon = config['skill_horizon']
        skill_horizon = horizon if skill_horizon is None else int(skill_horizon)
        if skill_horizon < 1:
            raise ValueError(f'skill_horizon must be at least 1, got {skill_horizon}.')

        num_skills = int(emp_config['num_skills'])
        if config['num_skills'] is not None and int(config['num_skills']) != num_skills:
            raise ValueError(
                f'num_skills={config["num_skills"]} disagrees with the pretrained '
                f'checkpoint\'s num_skills={num_skills} ({flags_path}); omit the flag, '
                f'it is read from the checkpoint.'
            )

        # `env_name` is a main.py flag the agent never sees, so a mismatch (relabelling
        # a dataset the policy was not trained on) cannot be caught here. Print what the
        # checkpoint was trained on so the log makes it checkable.
        print(
            f'[skill_bc_relabel_controller] relabelling with {ckpt_path}\n'
            f'[skill_bc_relabel_controller]   pretrained env_name={emp_flags.get("env_name")!r}, '
            f'num_skills={num_skills} -- --env_name must match.'
        )

        skill_agent = EmpowermentAgent.create(seed, ex_observations, ex_actions, emp_config)
        restore_epoch = config['skill_restore_epoch']
        if restore_epoch is None:
            restore_epoch = _latest_epoch(ckpt_path)
        skill_agent = restore_agent(skill_agent, ckpt_path, restore_epoch)

        # ── Build the inner goal-conditioned agent over discrete skills. ──────
        base_config = config['base'].to_dict() if hasattr(config['base'], 'to_dict') else dict(config['base'])
        # The inner agent never samples; the dataset keys stripped in `_base_config` are
        # supplied at the top level. `batch_size` is one of them: main.py sizes batches.
        base_config['batch_size'] = config['batch_size']
        base_config['encoder'] = config['encoder']
        base_config['frame_stack'] = config['frame_stack']
        # `ex_actions` for a discrete agent conveys the action-space SIZE through its
        # maximum, which is how gciql/crl derive their K-way head.
        ex_skill_actions = np.full((ex_observations.shape[0],), num_skills - 1, dtype=np.int32)
        base_agent = BASE_AGENTS[base_agent_name].create(
            seed, ex_observations, ex_skill_actions, ml_collections.ConfigDict(base_config)
        )

        # Resolved values for the agent's own use. As in `skill_value_controller`, these
        # do NOT reach the run's flags.json: main.py serialises FLAGS before `create`,
        # so it keeps `num_skills: null` / `skill_restore_epoch: null`. Pass
        # --agent.skill_restore_epoch explicitly if the run should record which
        # pretrained epoch it relabelled with.
        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        stored_config['num_skills'] = num_skills
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

        --agent=agents/skill_bc_relabel_controller.py:crl
        --agent.base.alpha=0.1 --agent.base.discount=0.99
    """
    # `sequence_length` is read by SequenceDataset and `chunk_horizon` by this agent;
    # they name the same H, so they share one FieldReference and --agent.chunk_horizon
    # moves both. `create` re-checks them in case anything breaks the link.
    chunk_horizon = ml_collections.FieldReference(10)

    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='skill_bc_relabel_controller',  # Agent name.
            # Path to a pretrained empowerment_skill run directory (holds flags.json
            # and params_*.pkl). Required.
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),
            # Epoch of the checkpoint to restore (None -> latest params_*.pkl).
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),
            # Number of skills. Auto-filled from the checkpoint's flags.json (the run's
            # own flags.json still records null -- see the note in `create`).
            num_skills=ml_collections.config_dict.placeholder(int),
            # H: the window each skill label is fitted over, and the option length of
            # the semi-MDP the high level is trained on.
            chunk_horizon=chunk_horizon,
            # Env steps a selected skill is held for at eval (None -> chunk_horizon).
            # Set it apart from chunk_horizon only to probe the mismatch deliberately.
            skill_horizon=ml_collections.config_dict.placeholder(int),
            # Temperature of the frozen low-level policy at execution (0 -> mode).
            low_temperature=0.0,
            # Batch size (main.py sizes batches from this; forwarded to the base agent).
            batch_size=1024,
            # ── Inner goal-conditioned algorithm ─────────────────────────────
            base_agent_name=base_agent_name,
            base=_base_config(base_agent_name),
            # ── Dataset hyperparameters ──────────────────────────────────────
            # Per-ENV-step discount. Weights the H-step snippet return and drives
            # GCDataset's geometric goal sampling. The option-level bootstrap is
            # `base.discount` (gamma_hi) -- see the module docstring.
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
