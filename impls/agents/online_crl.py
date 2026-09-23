"""Online contrastive RL (CRL) baseline: the flat goal-conditioned agent, trained from env interaction.

This is the OGBench-side counterpart of JaxGCRL's flat `crl` agent
(`jaxgcrl/agents/crl/crl.py`), built from this repo's primitives:

  * Critic: `GCBilinearValue` (ensemble of two), Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d).
    Trained with the *same* contrastive loss as the offline `agents/crl.py`
    (in-batch dot-product logits + sigmoid binary cross-entropy); positives are
    future observations of the same trajectory drawn at sample time by
    `utils/online_buffer.TrajectoryReplayBuffer` with P(offset = j) proportional
    to discount^j over the remaining rows (JaxGCRL's `flatten_batch` distribution).
  * Actor: `GCActor` with a tanh-squashed, state-dependent-std Gaussian (as in
    `agents/sac.py`) trained with JaxGCRL's SAC-style objective
        E[ alpha * log pi(a | s, g) - Q(s, a, g) ],   Q = critic head 0 (JaxGCRL uses one critic),
    using the reparameterised sample, with the same future observation as the goal.
  * alpha: a `LogParam` auto-tuned toward target entropy -0.5 * action_dim
    (JaxGCRL's `target_entropy = -0.5 * action_size`).

There is no reward, no Bellman target and no target network: the critic is purely
contrastive, faithful to CRL. Rollouts, the replay buffer and the update schedule
live in `main_online.py` / `utils/online_rollout.py`; this file is only the learner
plus `sample_actions`, in the same shape as every other agent in `agents/`.

Empowerment-modulated entropy target (`--agent.emp_checkpoint_path=<estimator run dir>`)
------------------------------------------------------------------------------------
Optional. A frozen OFFLINE empowerment estimator (an `agents/empowerment_*.py` run;
`empowerment_skill` is the one with per-env checkpoints) supplies E(s) = I(S+; A | s)
in nats for every replay row, and the SAC-style entropy constraint becomes per-state:

    H_target(s) = target_entropy + emp_lambda * (E(s) - E_mean)

E is in nats and so is H, so `emp_lambda = 1` is the unit-consistent default: exp(E(s))
is the effective number of distinguishable futures at s, and the policy's effective
action volume exp(H) is asked to scale with it. E_mean is the dataset mean, so the
average target equals the plain run's and only its allocation across states moves.
The temperature is auto-tuned PER STATE BIN: `emp_num_bins` learnable alphas, one per
quantile bin of E over the rows the stats were computed on, each driven toward its
rows' own targets (a single scalar alpha would only match the batch average). The
entropy term stays in the actor loss only -- the contrastive critic has no Bellman
target, so nothing bootstraps it into values and there is no pull toward
high-empowerment states, which is what a reward bonus would create.

E(s) is computed ONCE per row (offline rows at load, cached on disk; online rows in
one batched call per update round, `FlatCollector.flush_empowerment`) rather than per
update: the skill estimator draws `emp_num_splus_samples` futures per skill and
state (default 64) and costs ~0.5s per 1024 states. `emp_fast_path` is an algebraically
identical rewrite of `EmpowermentSkillAgent.empowerment` (the squared distance to all
K skill embeddings expanded into one einsum, same RNG stream) that avoids materialising
the [N, K, B, d] tensor; set it False to call the checkpoint's own method.
E_mean: `emp_mean` if given, else the checkpoint's logged `training/empowerment/mean`
(its train.csv), else the mean over the calibration rows (offline rows under RLPD,
otherwise the warm-up online rows); the bin edges always come from those rows.
`emp_entropy_target=False` keeps the estimator (its E(s) still lands on every row) but
leaves the entropy constraint as plain scalar-alpha SAC; the reward bonus below is the
other consumer of E.

Exploration reward bonus (`--agent.add_explore=reward | reward-to-rlpd`)
-----------------------------------------------------------------------
Optional, independent of the entropy target (both can be on). A per-transition
exploration reward r_x(s, a, s') is learned into its OWN non-goal-conditioned critic
Q_x(s, a) (twin MLP heads + Polyak target copy, `bonus_tau`) by a hard Bellman backup

    Q_x(s, a) <- r_x(s, a, s') + bonus_discount * mask * min_i Q_x^targ(s', a'),   a' ~ pi(. | s', g),

with g the batch row's relabelled future goal (the same goal the actor loss sees). No
second entropy term is added inside the backup: the run's SAC entropy term already
lives in the actor loss (decision of 2026-09-15). The actor then maximises

    Q_crl(s, a, g) + bonus_scale * Q_x(s, a) - alpha * log pi(a | s, g).

`explore_reward` names the reward, always a function of the state s' the transition
lands in, centred with the same E_mean as the entropy target (so Q_x is a discounted
EXCESS empowerment and sits near 0 rather than at E_mean / (1 - gamma)):
  * `'empowerment'`:              r_x = E(s') - E_mean, the frozen estimator's empowerment at s'.
  * `'max_episodic_empowerment'`: r_x = max_{t' <= t+1} E(s_t') - E_mean over the transition's own
    episode s_1 .. s_{t+1}, i.e. the running max of E along the trajectory so far (rows carry
    it as `episodic_max_empowerment`; monotone in t within an episode by construction).
`s'` of a trajectory's last transition is its final observation (the buffer's marker row),
whose E and running max are filled like every other row's, so the `next_*` fields are
real everywhere.
`add_explore='reward'` fits Q_x on ONLINE rows only (offline RLPD rows are masked out of
its Bellman loss via the rows' `is_offline` flag); `'reward-to-rlpd'` also backs it up on
the RLPD rows (their E(s') is cached with the dataset). The actor's bonus term and the
CRL critic are untouched by the mode: both train on every row of the batch as before.

`explore_reward_time_frac` (default None -- NO annealing, constant `bonus_scale` for the
whole run) optionally anneals the actor's use of the bonus: set it to a float in (0, 1] and
the weight on Q_x(s, a) decays LINEARLY from `bonus_scale` at env step 0 to 0 at env step
`explore_reward_time_frac * total_env_steps`, staying 0 after (`main_online.py` threads
the current env step into every `update` call for this; `total_env_steps` is copied into
the agent's own config from `--total_env_steps` before `create`, so it is baked into the
jitted loss like every other config value). Only the ACTOR's weight on Q_x is annealed --
Q_x itself keeps fitting r_x for the whole run, so `bonus_critic/*` stays meaningful to
inspect even once the actor has stopped reading it. Rationale (2026-09-16 discussion): a
bootstrapped state bonus pulls the policy back to whatever is locally high-empowerment
under its OWN rollout distribution, i.e. a familiar "hub", not toward unvisited high-E
territory -- annealing does not fix that targeting problem, but CRL relabels goals from
any future observation, so states visited during an early, noisier, high-bonus phase stay
useful to the critic long after the bonus itself has decayed to 0.

Distilled future-max empowerment bonus (`--agent.add_explore=distill | distill-to-rlpd`)
---------------------------------------------------------------------------------------
An ALTERNATIVE to Q_x (same `add_explore` switch, so the two are mutually exclusive
ablation arms; same `bonus_scale` "alpha", same `explore_reward_time_frac` annealing).
No reward, no Bellman backup, no target network: a second network E'(s, a) (twin
non-goal-conditioned MLP heads, the Q_x architecture) is REGRESSED (MSE, both heads) onto
the Monte Carlo max of the frozen estimator's empowerment along the row's own trajectory.
`distill_target` picks which max (decision of 2026-09-19: the episode max is the default):

    'episode_max':  E'(s_t, a_t)  <-  max_k E(s_k) - E_mean,       ALL k of the trajectory (k < t too), so
                                                                   every row of a trajectory shares one target;
    'future_max':   E'(s_t, a_t)  <-  max_{k > t} E(s_k) - E_mean,  only the states after s_t;

both through the trajectory's final observation,

and the actor maximises  Q_crl(s, a, g) + bonus_scale * min_i E'_i(s, a) - alpha * log pi(a | s, g)
directly (E' at its stored params, gradient through the reparameterised action only).
`explore_reward` is not read in these modes. Targets are centred with the run's E_mean
like r_x (a constant: the actor gradient is unchanged, the regression is better conditioned).

Rows carry the target as `distill_target` plus a `distill_target_ready` flag. An online
row's target only exists once its episode has ended (`FlatCollector.flush_empowerment`
writes a reverse cummax over every episode that closed since the last flush); rows of the
episode still in progress are masked out of the E' regression until then (pure Monte
Carlo, decision of 2026-09-18) and are used by everything else as before. Offline rows'
targets are one reverse cummax per trajectory over the cached offline E values at load
time (utils/rlpd.py), so RLPD adds no per-update cost.
  * `'distill'` (online only): E' is fit on ONLINE rows only, and the actor's bonus term is
    added on ONLINE rows only (`is_offline == 0`) -- E' is never evaluated on offline
    states it was not trained on. The actor loss stays a mean over the whole batch, i.e.
    every row's loss is alpha*log pi - Q_crl - [row is online] * bonus_scale * E'.
  * `'distill-to-rlpd'` (offline + online): E' is fit on online AND RLPD rows, and the
    actor's bonus term is on every row.

`bonus_grad_diagnostics=True` (either bonus family) logs, per update, the actor-parameter
gradient norms of the three actor terms taken separately -- the CRL term -Q_crl, the
bonus term at bonus_scale = 1, the entropy term -- plus their per-sample action-gradient
norms, under `bonus_grad/*`. `bonus_grad/scale_equal_param` = ||g_crl|| / ||g_bonus|| is the
`bonus_scale` at which the two actor gradients have equal norm. Costs three extra actor
backward passes per update, so it is off by default.
Random network distillation bonus (`--agent.add_explore=reward --agent.explore_reward=rnd`)
------------------------------------------------------------------------------------------
Burda et al. 2019 (arXiv:1810.12894) as the exploration reward, through the same Q_x
machinery above (own critic, `bonus_scale`, annealing) and needing no empowerment
estimator. The paper's recipe, state-based:
  * A fixed random target f and a trained predictor f_hat, ONE architecture for both
    (`utils/networks.RNDEmbedding`: ReLU MLP trunk `rnd_hidden_dims` -> linear `rnd_rep_dim`;
    the paper's Appendix A.5 fixes only "encoder followed by dense layers" and defers the
    rest to its code), embed the LANDING state s'. r_x(s') = ||f_hat(s') - f(s')||^2, the
    squared L2 norm over the k embedding dims (the paper's Sec. 2.2 / Algorithm 1), and the
    predictor minimises that same MSE.
  * Input normalisation (paper Sec. 2.4, Table 4): s' is whitened per dimension by a RUNNING
    mean/std of the online observations and clipped to +-`rnd_obs_clip` (5); the policy
    never sees it. The stats are seeded by the warm-up rows before the first update (the
    paper initialises them from a random agent's steps; here the untrained stochastic actor's,
    decision of 2026-09-21) and updated every round with the states visited since
    (`FlatCollector.flush_rnd` -> `rnd_update_stats`), AFTER those states' rewards are
    computed, as in Algorithm 1.
  * Reward normalisation: r_x is divided by the running std of the discounted intrinsic
    RETURN (the paper's RewardForwardFilter, ret <- bonus_discount * ret + r_x, run over the
    online stream in visit order and never reset at episode ends) -- no mean subtraction.
    Both statistics live on the agent (`rnd_state`) and are checkpointed with it.
  * Online only: the predictor loss (`rnd_predictor_loss`, on a random `rnd_update_proportion`
    of the rows, the paper's `proportion_of_exp_used_for_predictor_update`) and Q_x's
    Bellman loss use the online rows alone (`add_explore='reward'` is required and
    `'reward-to-rlpd'` rejected), the running stats see online states alone, and the ACTOR
    reads Q_x on the online rows alone -- an offline RLPD state never receives a bonus.
  * Non-episodic (`rnd_nonepisodic`, the paper's choice): Q_x's backup ignores termination
    (mask 1 through a goal reach), so finishing the task is not a novelty penalty.
  * The predictor has its own Adam (`rnd_lr`, paper 1e-4; optax.multi_transform); the
    target's updates are set to zero, so it stays the random draw it was initialised as.

Combining the distilled empowerment bonus with RND (`--agent.rnd_bonus_scale=<beta>`)
--------------------------------------------------------------------------------------
`add_explore` is one switch, so the Q_x-based RND above cannot run next to the distilled
E'. `rnd_bonus_scale` (None -> off) turns on an INDEPENDENT copy of the RND machinery
instead: the same predictor / target pair, running statistics and normalised reward, fed
into its own twin critic Q_rnd(s, a) (+ Polyak target, `bonus_tau`, `bonus_discount`,
non-episodic per `rnd_nonepisodic`) by the same hard Bellman backup as Q_x, on the online
rows alone. The actor then maximises

    Q_crl(s, a, g) + bonus_scale(t) * B(s, a) + rnd_bonus_scale(t) * [row is online] * min_i Q_rnd_i(s, a) - alpha * log pi,

with B whatever `add_explore` selects (nothing, Q_x or E'). The two weights anneal
separately: `explore_reward_time_frac` only touches `bonus_scale`, `rnd_time_frac` (None ->
constant) only `rnd_bonus_scale`. It is rejected together with `explore_reward='rnd'`
(that would be RND twice); with `add_explore=None` it is a plain RND run whose bonus is
called Q_rnd instead of Q_x.
"""

import copy
import csv
import json
import os
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from jax.scipy.special import logsumexp
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent
from utils.networks import GCActor, GCBilinearValue, GCValue, LogParam, LogParamVector, RNDEmbedding, RunningMeanStd
from utils.skill_checkpoint import latest_epoch

# Offline estimator families that expose `empowerment(observations, rng) -> [B]` (nats).
EMPOWERMENT_ESTIMATOR_AGENTS = (
    'empowerment_skill',
    'empowerment_crl',
    'empowerment_crl_flowbc',
    'empowerment_dads',
    'empowerment_dv',
)

Q_BONUS_MODES = ('reward', 'reward-to-rlpd')  # Bellman critic Q_x over an exploration reward
DISTILL_MODES = ('distill', 'distill-to-rlpd')  # regressed future-max empowerment E'(s, a)
DISTILL_TARGETS = ('episode_max', 'future_max')  # E' regression target: max of E over the whole trajectory | after s_t
ADD_EXPLORE_MODES = Q_BONUS_MODES + DISTILL_MODES  # plus None / 'none' -> off
EXPLORE_REWARDS = ('empowerment', 'max_episodic_empowerment', 'rnd')  # per-transition exploration rewards
_UNSET = object()  # `bonus_scale_at`: "use the config value" (None is a legitimate frac)


def _read_last_metric(csv_path, column):
    """Last logged value of `column` in a run's train.csv, or None if the file/column is missing."""
    if not os.path.exists(csv_path):
        return None
    value = None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cell = row.get(column)
            if cell not in (None, ''):
                value = float(cell)
    return value


def load_empowerment_estimator(seed, ex_observations, ex_actions, config, caller='online_crl'):
    """Rebuild and restore the frozen offline empowerment estimator named by `config['emp_checkpoint_path']`.

    Same recipe as `utils.skill_checkpoint.load_frozen_skill_agent` (rebuild from the run's
    own flags.json, check the observation pipeline, restore `params_<epoch>.pkl`), for the
    estimator families above. For `empowerment_skill` the checkpoint's `num_splus_samples`
    (1 during its training) is replaced by `config['emp_num_splus_samples']`.

    Returns `(estimator, resolved)`; `resolved` holds `ckpt_path`, `restore_epoch`,
    `agent_name`, `env_name` and `mean_metric` (the run's last logged
    `training/empowerment/mean`, or None).
    """
    from agents import agents as agent_classes  # local: agents/__init__ imports this module

    ckpt_path = config['emp_checkpoint_path'].rstrip('/')
    flags_path = os.path.join(ckpt_path, 'flags.json')
    if not os.path.exists(flags_path):
        raise FileNotFoundError(f'{caller}: flags.json not found in {ckpt_path}')
    with open(flags_path) as f:
        emp_flags = json.load(f)
    emp_config = dict(emp_flags['agent'])
    agent_name = emp_config.get('agent_name')
    if agent_name not in EMPOWERMENT_ESTIMATOR_AGENTS:
        raise ValueError(
            f'{caller}: emp_checkpoint_path must be one of {EMPOWERMENT_ESTIMATOR_AGENTS} runs, got '
            f'agent_name={agent_name!r} in {flags_path}'
        )
    for key in ('encoder', 'frame_stack', 'discrete'):
        if config[key] != emp_config.get(key):
            expected = emp_config.get(key)
            fix = f'omit --agent.{key}' if expected is None else f'pass --agent.{key}={expected!r}'
            raise ValueError(
                f"{caller}: {key}={config[key]!r} does not match the estimator checkpoint's {key}={expected!r} "
                f'({flags_path}); {fix}.'
            )
    if agent_name == 'empowerment_skill':
        emp_config['num_splus_samples'] = int(config['emp_num_splus_samples'])
    restore_epoch = config['emp_restore_epoch']
    if restore_epoch is None:
        restore_epoch = latest_epoch(ckpt_path)
    estimator = agent_classes[agent_name].create(seed, ex_observations, ex_actions, emp_config)
    estimator = restore_agent(estimator, ckpt_path, int(restore_epoch))
    mean_metric = _read_last_metric(os.path.join(ckpt_path, 'train.csv'), 'training/empowerment/mean')
    print(
        f'[{caller}] frozen empowerment estimator: {ckpt_path} (epoch {restore_epoch})\n'
        f'[{caller}]   agent_name={agent_name!r}, pretrained env_name={emp_flags.get("env_name")!r} '
        f'-- --env_name should be its online sibling; logged training/empowerment/mean='
        f'{"n/a" if mean_metric is None else f"{mean_metric:.4f}"}'
        + (f', num_splus_samples={emp_config["num_splus_samples"]}' if agent_name == 'empowerment_skill' else '')
    )
    resolved = dict(
        ckpt_path=ckpt_path,
        restore_epoch=int(restore_epoch),
        agent_name=agent_name,
        env_name=emp_flags.get('env_name'),
        mean_metric=mean_metric,
    )
    return estimator, resolved


def skill_empowerment_fast(estimator, observations, rng):
    """`EmpowermentSkillAgent.empowerment` with the same RNG stream and an einsum instead of a broadcast.

    The original scores every future sample psi against all K skill embeddings as
    -||phi_z' - psi||^2 / d, which materialises a [N, K, B, d] difference tensor per skill.
    Expanding the square, ||phi||^2 - 2 phi.psi + ||psi||^2, turns that into a [K, N, B]
    einsum over d; the positive term keeps the direct form. Results agree to float
    rounding (checked in tests against the checkpoint's own method).
    """
    K = int(estimator.config['num_skills'])
    N = int(estimator.config['num_splus_samples'])
    d = int(estimator.config['value_latent_dim'])
    log_K = jnp.log(K)

    act_rng = jax.random.fold_in(rng, 5)
    rng, sample_rng = jax.random.split(rng)
    skill_rngs = jax.random.split(sample_rng, K)

    phi_all = estimator._v_phi_all_skills(observations, use_target=False, policy_params=None, rng=act_rng)  # [K,B,d]
    phi_sq = jnp.sum(phi_all**2, axis=-1)  # [K, B]

    def per_skill(carry, xs):
        phi_z, skill_rng = xs  # [B, d]
        noise = jax.random.normal(skill_rng, (N, *phi_z.shape))
        psi = phi_z[None] + noise * jnp.sqrt(d / 2.0)  # [N, B, d]
        psi_sq = jnp.sum(psi**2, axis=-1)  # [N, B]
        cross = jnp.einsum('kbd,nbd->knb', phi_all, psi)  # [K, N, B]
        log_v_all = -(phi_sq[:, None, :] - 2.0 * cross + psi_sq[None]) / d  # [K, N, B]
        log_v = -jnp.sum((phi_z[None] - psi) ** 2, axis=-1) / d  # [N, B]
        log_denom = logsumexp(log_v_all, axis=0) - log_K  # [N, B]
        return carry, (log_v - log_denom).mean(axis=0)  # [B]

    _, emp_per_skill = jax.lax.scan(per_skill, None, (phi_all, skill_rngs))  # [K, B]
    return emp_per_skill.mean(axis=0)


class OnlineCRLAgent(flax.struct.PyTreeNode):
    """Flat online CRL agent (contrastive critic + entropy-regularised actor)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    emp_agent: Any = None  # frozen offline empowerment estimator (None -> plain online CRL)
    emp_stats: Any = None  # dict(mean=[], edges=[emp_num_bins - 1]) once `with_empowerment_stats` ran
    rnd_state: Any = None  # dict(obs_rms=RunningMeanStd, ret_rms=RunningMeanStd, ret_carry=[]) when explore_reward='rnd'

    # ── Empowerment estimator ─────────────────────────────────────────────────

    @property
    def uses_empowerment(self):
        return self.emp_agent is not None

    @property
    def uses_entropy_target(self):
        """Per-state entropy target from E(s) (needs the estimator and `emp_entropy_target`)."""
        return self.uses_empowerment and bool(self.config['emp_entropy_target'])

    @property
    def uses_explore_bonus(self):
        """Some exploration bonus is in the actor loss: Q_x or E' (any `add_explore` mode) and/or the independent Q_rnd."""
        return self.config['add_explore'] is not None or self.uses_rnd_bonus

    @property
    def uses_q_bonus(self):
        """Exploration reward bonus critic Q_x is on (`add_explore` is 'reward' or 'reward-to-rlpd')."""
        return self.config['add_explore'] in Q_BONUS_MODES

    @property
    def uses_distill_bonus(self):
        """Distilled future-max empowerment E'(s, a) is on (`add_explore` is 'distill' or 'distill-to-rlpd')."""
        return self.config['add_explore'] in DISTILL_MODES

    @property
    def uses_rnd_reward(self):
        """Q_x's reward is random network distillation (`add_explore='reward'` with `explore_reward='rnd'`)."""
        return self.uses_q_bonus and self.config['explore_reward'] == 'rnd'

    @property
    def uses_rnd_bonus(self):
        """The independent RND bonus critic Q_rnd is on (`rnd_bonus_scale` set), next to whatever `add_explore` is."""
        return self.config['rnd_bonus_scale'] is not None

    @property
    def uses_rnd(self):
        """An RND predictor / target pair and the running RND statistics exist (either RND path)."""
        return self.uses_rnd_reward or self.uses_rnd_bonus

    @jax.jit
    def empowerment(self, observations, seed):
        """E(s) in nats for a batch of observations, from the frozen estimator."""
        if self.config['emp_agent_name'] == 'empowerment_skill' and self.config['emp_fast_path']:
            return skill_empowerment_fast(self.emp_agent, observations, seed)
        return self.emp_agent.empowerment(observations, seed)

    def empowerment_np(self, observations, seed, chunk_size=1024):
        """`empowerment` over a large array, chunked (one RNG key per chunk), as a numpy array."""
        observations = np.asarray(observations)
        out = np.empty((len(observations),), dtype=np.float32)
        for start in range(0, len(observations), chunk_size):
            seed, key = jax.random.split(seed)
            chunk = observations[start : start + chunk_size]
            out[start : start + len(chunk)] = np.asarray(self.empowerment(jnp.asarray(chunk), key))
        return out

    def with_empowerment_stats(self, values, mean=None):
        """Finalise E_mean and the quantile bin edges from the calibration rows' values (numpy, [n]).

        `mean` overrides the rows' mean (the checkpoint metric or `emp_mean`). Must run before
        the first update; `main_online.py` does it once the calibration rows exist.
        """
        assert self.uses_empowerment, 'with_empowerment_stats: this agent has no empowerment estimator'
        values = np.asarray(values, dtype=np.float32)
        assert values.ndim == 1 and len(values) > 0 and np.all(np.isfinite(values)), (
            'with_empowerment_stats: calibration values must be a non-empty finite 1-D array'
        )
        num_bins = int(self.config['emp_num_bins'])
        edges = np.quantile(values, np.arange(1, num_bins) / num_bins).astype(np.float32)
        mean = float(values.mean()) if mean is None else float(mean)
        stats = dict(mean=jnp.asarray(mean, dtype=jnp.float32), edges=jnp.asarray(edges))
        config = dict(self.config)
        config['emp_stats_ready'] = True
        config['emp_mean_used'] = mean
        config['emp_bin_edges'] = tuple(float(e) for e in edges)
        return self.replace(emp_stats=stats, config=flax.core.FrozenDict(**config))

    # ── Losses ────────────────────────────────────────────────────────────────

    def contrastive_loss(self, batch, grad_params):
        """In-batch contrastive critic loss; identical in form to `agents/crl.py`."""
        batch_size = batch['observations'].shape[0]

        v, phi, psi = self.network.select('critic')(
            batch['observations'],
            batch['value_goals'],
            actions=batch['actions'],
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        v = jnp.exp(v)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }

    # ── Exploration reward bonus ──────────────────────────────────────────────

    def _check_emp_stats_ready(self):
        if not self.config['emp_stats_ready']:
            raise RuntimeError(
                'online_crl: empowerment stats (E_mean, bin edges) are not finalised; '
                'main_online.py must call agent.with_empowerment_stats before the first update.'
            )

    def explore_reward(self, batch):
        """Per-transition exploration reward r_x(s, a, s') for every row of the batch, [B].

        The one hook a new bonus type has to fill in. Both existing rewards are centred
        with the E_mean the entropy target also uses and read the landing state s':
        `'empowerment'` E(s') (rows carry `next_empowerment`), `'max_episodic_empowerment'`
        the running max of E over the episode through s' (`next_episodic_max_empowerment`).
        """
        kind = self.config['explore_reward']
        if kind == 'empowerment':
            self._check_emp_stats_ready()
            return batch['next_empowerment'] - self.emp_stats['mean']
        if kind == 'max_episodic_empowerment':
            self._check_emp_stats_ready()
            return batch['next_episodic_max_empowerment'] - self.emp_stats['mean']
        if kind == 'rnd':
            return self.rnd_reward(batch)
        raise ValueError(f"online_crl: unknown explore_reward {kind!r}")

    def _q_critic_loss(self, batch, grad_params, rng, module, reward, weight, masks):
        """Hard Bellman regression of the non-goal-conditioned twin critic `module`(s, a) onto
        reward + bonus_discount * masks * min_i target_`module`_i(s', a'), a' ~ pi(.|s', g), on the rows with `weight` > 0.
        Shared by Q_x (`bonus_critic`) and the independent RND critic (`rnd_critic`)."""
        next_dist = self.network.select('actor')(batch['next_observations'], batch['actor_goals'])
        next_actions = next_dist.sample(seed=rng)
        next_qs = self.network.select(f'target_{module}')(batch['next_observations'], actions=next_actions)
        next_q = jnp.min(next_qs, axis=0)  # twin-min, as SAC
        target_q = reward + self.config['bonus_discount'] * masks * next_q
        # No entropy term here: the run's SAC entropy term is already in the actor loss.

        qs = self.network.select(module)(batch['observations'], actions=batch['actions'], params=grad_params)
        sq_err = jnp.square(qs - target_q[None])  # [2, B]
        num_rows = jnp.maximum(weight.sum(), 1.0)
        loss = (sq_err * weight[None]).sum() / (num_rows * qs.shape[0])
        return loss, {
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
            'reward_mean': reward.mean(),
            'reward_min': reward.min(),
            'reward_max': reward.max(),
            'target_q_mean': target_q.mean(),
            'frac_rows_fit': weight.mean(),
        }

    def bonus_critic_loss(self, batch, grad_params, rng):
        """Hard Bellman regression of Q_x(s, a) onto r_x + gamma * mask * min Q_x^targ(s', a'), a' ~ pi(.|s', g).

        `add_explore='reward'` fits only the online rows (`is_offline == 0`); `'reward-to-rlpd'`
        fits every row. The CRL critic and the actor are not affected by this mask.
        """
        reward = self.explore_reward(batch)  # [B]
        if self.uses_rnd_reward and self.config['rnd_nonepisodic']:
            masks = jnp.ones_like(batch['masks'])  # paper: the intrinsic return runs through episode ends
        else:
            masks = batch['masks']
        if self.config['add_explore'] == 'reward':
            weight = 1.0 - batch['is_offline']
        else:
            weight = jnp.ones_like(reward)
        loss, info = self._q_critic_loss(batch, grad_params, rng, 'bonus_critic', reward, weight, masks)
        return loss, {'bonus_critic_loss': loss, **info}

    def rnd_critic_loss(self, batch, grad_params, rng):
        """Independent RND bonus critic (`rnd_bonus_scale`): Q_rnd(s, a) backed up on the ONLINE rows only onto the
        normalised RND reward, through episode ends when `rnd_nonepisodic` (the paper's choice)."""
        reward = self.rnd_reward(batch)  # [B]
        masks = jnp.ones_like(batch['masks']) if self.config['rnd_nonepisodic'] else batch['masks']
        weight = 1.0 - batch['is_offline']
        loss, info = self._q_critic_loss(batch, grad_params, rng, 'rnd_critic', reward, weight, masks)
        return loss, {'rnd_critic_loss': loss, **info}

    def distill_loss(self, batch, grad_params):
        """MSE regression of E'(s, a) (both heads) onto the row's Monte Carlo target (`distill_target`: episode or future max of E) - E_mean.

        Only rows whose target exists are fit (`distill_target_ready`: offline rows always, online
        rows once their episode has ended); `add_explore='distill'` additionally drops the
        offline RLPD rows. Not-ready rows hold a NaN target, which the `where` keeps out.
        """
        self._check_emp_stats_ready()
        weight = batch['distill_target_ready']
        if self.config['add_explore'] == 'distill':
            weight = weight * (1.0 - batch['is_offline'])
        target = jnp.where(weight > 0, batch['distill_target'] - self.emp_stats['mean'], 0.0)  # [B]

        es = self.network.select('distill_critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        sq_err = jnp.square(es - target[None])  # [2, B]
        num_rows = jnp.maximum(weight.sum(), 1.0)
        distill_loss = (sq_err * weight[None]).sum() / (num_rows * es.shape[0])

        target_mean = (target * weight).sum() / num_rows
        target_var = (jnp.square(target - target_mean) * weight).sum() / num_rows
        online = 1.0 - batch['is_offline']
        return distill_loss, {
            'distill_loss': distill_loss,
            # 1 - MSE / Var(target): how much of the target's spread E' explains on the fitted rows.
            'explained_variance': 1.0 - distill_loss / jnp.maximum(target_var, 1e-8),
            'e_mean': (es.mean(axis=0) * weight).sum() / num_rows,
            'e_max': jnp.where(weight > 0, es.max(axis=0), -jnp.inf).max(),
            'e_min': jnp.where(weight > 0, es.min(axis=0), jnp.inf).min(),
            'target_mean': target_mean,
            'target_std': jnp.sqrt(target_var),
            'target_min': jnp.where(weight > 0, target, jnp.inf).min(),
            'target_max': jnp.where(weight > 0, target, -jnp.inf).max(),
            'frac_rows_fit': weight.mean(),
            'frac_online_rows_ready': (batch['distill_target_ready'] * online).sum() / jnp.maximum(online.sum(), 1.0),
        }

    def bonus_value(self, batch, actions):
        """The actor's exploration bonus at `actions`, before `bonus_scale`: `(value [B], row weight [B])`.

        Q_x modes: twin-min Q_x(s, a) on every row. Distill modes: twin-min E'(s, a), on every
        row for 'distill-to-rlpd' and on online rows only for 'distill'. Both networks are read
        at their stored params, so the gradient reaches the actor through `actions` only.
        """
        module = 'distill_critic' if self.uses_distill_bonus else 'bonus_critic'
        values = self.network.select(module)(batch['observations'], actions=actions)
        value = jnp.min(values, axis=0)
        if self.config['add_explore'] == 'distill' or self.uses_rnd_reward:
            # 'distill': E' is never evaluated on offline states; RND reward: Q_x is fit on online rows only.
            weight = 1.0 - batch['is_offline']
        else:
            weight = jnp.ones_like(value)
        return value, weight

    def rnd_bonus_value(self, batch, actions):
        """The independent RND bonus at `actions`, before `rnd_bonus_scale`: `(twin-min Q_rnd(s, a) [B], online-row weight [B])`."""
        values = self.network.select('rnd_critic')(batch['observations'], actions=actions)
        return jnp.min(values, axis=0), 1.0 - batch['is_offline']

    def bonus_grad_stats(self, batch, rng):
        """Gradient norms of the actor's three loss terms taken separately (`bonus_grad_diagnostics`).

        Terms, each a mean over the batch like the actor loss itself: CRL -Q_crl(s, a_pi, g), the
        bonus -w * B(s, a_pi) at bonus_scale = 1 (B = Q_x or E', w its row weight), and the entropy
        term alpha * log pi. `scale_equal_param` is the bonus_scale at which the bonus term's
        actor-parameter gradient has the CRL term's norm; `scale_equal_action` the same from the
        per-sample action gradients dQ/da, dB/da (mean norm over the rows the bonus is on).
        """
        actor_params = self.network.params['modules_actor']
        if self.uses_entropy_target:
            alpha = self.network.select('alpha')()[jnp.searchsorted(self.emp_stats['edges'], batch['empowerment'])]
        else:
            alpha = self.network.select('alpha')()

        def terms(params):
            dist = self.network.select('actor')(
                batch['observations'], batch['actor_goals'], params={'modules_actor': params}
            )
            actions, log_probs = dist.sample_and_log_prob(seed=rng)
            q = self.network.select('critic')(batch['observations'], batch['actor_goals'], actions=actions)[0]
            value, weight = self.bonus_value(batch, actions)
            return -q.mean(), -(weight * value).mean(), (alpha * log_probs).mean()

        grads = [jax.grad(lambda p, i=i: terms(p)[i])(actor_params) for i in range(3)]
        flat = [jnp.concatenate([g.ravel() for g in jax.tree_util.tree_leaves(grad)]) for grad in grads]
        norms = [jnp.linalg.norm(f) for f in flat]

        def cosine(a, b):
            return jnp.dot(a, b) / jnp.maximum(jnp.linalg.norm(a) * jnp.linalg.norm(b), 1e-20)

        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'])
        actions = jax.lax.stop_gradient(dist.sample(seed=rng))
        dq_da = jax.grad(
            lambda a: self.network.select('critic')(batch['observations'], batch['actor_goals'], actions=a)[0].sum()
        )(actions)
        db_da = jax.grad(lambda a: self.bonus_value(batch, a)[0].sum())(actions)
        _, weight = self.bonus_value(batch, actions)
        num_rows = jnp.maximum(weight.sum(), 1.0)
        dq_norm = (jnp.linalg.norm(dq_da, axis=-1) * weight).sum() / num_rows
        db_norm = (jnp.linalg.norm(db_da, axis=-1) * weight).sum() / num_rows
        return {
            'param_norm_crl': norms[0],
            'param_norm_bonus': norms[1],
            'param_norm_entropy': norms[2],
            'scale_equal_param': norms[0] / jnp.maximum(norms[1], 1e-20),
            'cos_crl_bonus': cosine(flat[0], flat[1]),
            'action_norm_crl': dq_norm,
            'action_norm_bonus': db_norm,
            'scale_equal_action': dq_norm / jnp.maximum(db_norm, 1e-20),
        }
    # ── Random network distillation (explore_reward='rnd') ────────────────────

    def rnd_normalize(self, observations):
        """RND input: observations whitened by the running obs stats, clipped to +-rnd_obs_clip (paper: 5)."""
        return self.rnd_state['obs_rms'].normalize(observations)

    def rnd_reward(self, batch):
        """Normalised RND intrinsic reward at the rows' landing states: raw / running std of the intrinsic return (not centred)."""
        ret_rms = self.rnd_state['ret_rms']
        return self.rnd_raw_reward(batch['next_observations']) / jnp.sqrt(ret_rms.var + ret_rms.eps)

    def rnd_raw_reward(self, next_observations):
        """Un-normalised intrinsic reward ||f_hat(s') - f(s')||^2 at the landing states, [B] (paper Algorithm 1)."""
        x = self.rnd_normalize(next_observations)
        target = self.network.select('rnd_target')(x)
        pred = self.network.select('rnd_predictor')(x)
        return jnp.sum(jnp.square(pred - target), axis=-1)

    def rnd_predictor_loss(self, batch, grad_params, rng):
        """Predictor regression onto the fixed target on the ONLINE rows of the batch (the paper's aux loss).

        A random `rnd_update_proportion` of those rows contributes (the paper's
        `proportion_of_exp_used_for_predictor_update`, 1 in its single-env runs); offline RLPD
        rows never do, so the predictor only ever fits states the agent itself visited.
        """
        x = self.rnd_normalize(batch['next_observations'])
        target = self.network.select('rnd_target')(x)  # stored params: fixed, no gradient
        pred = self.network.select('rnd_predictor')(x, params=grad_params)
        per_row = jnp.sum(jnp.square(pred - target), axis=-1)  # [B], ||f_hat - f||^2 as the paper's MSE
        keep = jax.random.uniform(rng, per_row.shape) < self.config['rnd_update_proportion']
        online = 1.0 - batch['is_offline']
        weight = online * keep.astype(jnp.float32)
        loss = (per_row * weight).sum() / jnp.maximum(weight.sum(), 1.0)

        offline = batch['is_offline']
        ret_rms = self.rnd_state['ret_rms']
        return loss, {
            'predictor_loss': loss,
            'int_rew_raw_online_mean': (per_row * online).sum() / jnp.maximum(online.sum(), 1.0),
            'int_rew_raw_offline_mean': (per_row * offline).sum() / jnp.maximum(offline.sum(), 1.0),
            'ret_std': jnp.sqrt(ret_rms.var + ret_rms.eps),
            'feat_var': jnp.mean(jnp.var(target, axis=0)),  # the paper's feat_var / max_feat diagnostics
            'max_feat': jnp.max(jnp.abs(target)),
            'frac_rows_fit': weight.mean(),
        }

    @jax.jit
    def rnd_update_stats(self, observations, is_next, valid):
        """Advance the RND running statistics on a chunk of newly visited states, in visit order.

        `observations` [P, ...] are the states as visited, `valid` [P] masks padding and
        `is_next` [P] marks the ones that are some transition's landing state s' (an
        episode's reset state is not). The paper's per-rollout bookkeeping (Algorithm 1):
          1. r_x(s') is computed for the `is_next` states with the current predictor and the
             obs stats as they stand (a rollout's rewards precede its stats update);
          2. the reward-forward filter ret <- bonus_discount * ret + r_x runs over them in
             order (never reset at episode ends; `ret_carry` continues across calls) and the
             returns' running std -- the intrinsic reward's normaliser -- absorbs its values;
          3. the obs running mean/std then absorbs every valid state.
        Returns `(agent, info)`.
        """
        state = self.rnd_state
        raw = self.rnd_raw_reward(observations)  # [P], with the stats before this chunk
        obs_rms = state['obs_rms'].update_masked(observations, valid)
        emit = jnp.asarray(valid, jnp.float32) * jnp.asarray(is_next, jnp.float32)
        gamma = self.config['bonus_discount']

        def filter_step(ret, xs):
            r, e = xs
            ret = jnp.where(e > 0, gamma * ret + r, ret)
            return ret, ret

        ret_carry, rets = jax.lax.scan(filter_step, state['ret_carry'], (raw, emit))
        ret_rms = state['ret_rms'].update_masked(rets, emit)
        new_state = dict(obs_rms=obs_rms, ret_rms=ret_rms, ret_carry=ret_carry)
        info = {
            'int_rew_raw_mean': (raw * emit).sum() / jnp.maximum(emit.sum(), 1.0),
            'ret_std': jnp.sqrt(ret_rms.var + ret_rms.eps),
            'ret_carry': ret_carry,
            'obs_count': obs_rms.count,
            'obs_std_mean': jnp.mean(jnp.sqrt(obs_rms.var + obs_rms.eps)),
        }
        return self.replace(rnd_state=new_state), info

    def bonus_scale_at(self, env_steps, scale0=None, frac=_UNSET):
        """Actor's current weight on Q_x / E'(s, a): constant `bonus_scale` unless
        `explore_reward_time_frac` is set, in which case it linearly decays to 0 by env step
        `explore_reward_time_frac * total_env_steps`, held at 0 after. `env_steps` is a traced
        scalar (the env step count at the start of this update round); `total_env_steps` and
        `explore_reward_time_frac` are static config, baked in at trace time.
        `scale0` / `frac` override the two config values (the independent RND bonus passes
        `rnd_bonus_scale` / `rnd_time_frac`, so the two terms anneal separately).
        """
        scale0 = float(self.config['bonus_scale'] if scale0 is None else scale0)
        frac = self.config['explore_reward_time_frac'] if frac is _UNSET else frac
        total = self.config['total_env_steps']
        if frac is None:
            # Default: no annealing schedule at all, regardless of whether total_env_steps
            # happens to be known.
            return jnp.asarray(scale0, dtype=jnp.float32)
        frac = float(frac)
        if total is None:
            # A schedule was requested but there is no time budget to schedule against, e.g.
            # the agent was built without main_online.py setting total_env_steps -- fall back
            # to the constant rather than error, so the bonus is still usable standalone.
            return jnp.asarray(scale0, dtype=jnp.float32)
        if frac <= 0.0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        assert env_steps is not None, (
            'bonus_scale_at: total_env_steps is set (the schedule is wired up) but env_steps was '
            "not passed to update()/total_loss()/actor_loss(); main_online.py threads it in "
            'automatically whenever agent.uses_explore_bonus, so this usually means the agent was '
            'called directly. Pass env_steps explicitly, or leave total_env_steps unset for the '
            'constant bonus_scale (no decay).'
        )
        decay_denom = frac * float(total)
        progress = jnp.asarray(env_steps, dtype=jnp.float32) / decay_denom
        return scale0 * jnp.clip(1.0 - progress, 0.0, 1.0)

    def actor_loss(self, batch, grad_params, rng, env_steps=None):
        """SAC-style actor + alpha losses (JaxGCRL `update_actor_and_alpha`), goal = the sampled future state."""
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)

        # Critic at its stored params (no `params=grad_params`): the actor loss has zero
        # gradient w.r.t. the critic, so the gradient reaches the actor only through the
        # reparameterised action, as in JaxGCRL. (All losses share one Adam step here;
        # their parameter dependences are disjoint, so this equals separate updates.)
        # JaxGCRL's actor reads a single contrastive critic: use head 0 of the ensemble
        # (both heads are still trained by the contrastive loss, as in agents/crl.py).
        qs = self.network.select('critic')(batch['observations'], batch['actor_goals'], actions=actions)
        q = qs[0]

        entropy = -jax.lax.stop_gradient(log_probs)  # per-sample, [B]
        info = {}
        if self.uses_entropy_target:
            # Per-state entropy target from the row's offline empowerment; temperature per E bin.
            self._check_emp_stats_ready()
            emp = batch['empowerment']  # [B], nats
            bins = jnp.searchsorted(self.emp_stats['edges'], emp)  # [B] in [0, emp_num_bins)
            target_entropy = self.config['target_entropy'] + self.config['emp_lambda'] * (emp - self.emp_stats['mean'])
            alpha = self.network.select('alpha')()[bins]
            alpha_param = self.network.select('alpha')(params=grad_params)[bins]

            num_bins = int(self.config['emp_num_bins'])
            one_hot = jax.nn.one_hot(bins, num_bins)  # [B, K]
            counts = one_hot.sum(axis=0)  # [K]
            safe = jnp.maximum(counts, 1.0)
            alpha_bins = self.network.select('alpha')()
            entropy_bins = (one_hot * entropy[:, None]).sum(axis=0) / safe
            target_bins = (one_hot * target_entropy[:, None]).sum(axis=0) / safe
            info.update(
                {
                    'emp_mean': emp.mean(),
                    'emp_min': emp.min(),
                    'emp_max': emp.max(),
                    'emp_mean_used': self.emp_stats['mean'],
                }
            )
            for i in range(num_bins):
                info[f'alpha_bin{i}'] = alpha_bins[i]
                info[f'entropy_bin{i}'] = entropy_bins[i]
                info[f'target_entropy_bin{i}'] = target_bins[i]
                info[f'frac_bin{i}'] = counts[i] / entropy.shape[0]
        else:
            alpha = self.network.select('alpha')()
            alpha_param = self.network.select('alpha')(params=grad_params)
            target_entropy = jnp.full_like(entropy, self.config['target_entropy'])

        if self.uses_explore_bonus:
            # Exploration bonus: + bonus_scale(t) * B(s, a_pi), B = Q_x or the distilled E' at its stored
            # params (twin-min), on the rows `bonus_value` weights in (every row, or online rows only
            # for 'distill'). bonus_scale(t) is the env-step-annealed weight (bonus_scale_at); with
            # env_steps=None (no schedule wired up) it is just the constant bonus_scale.
            q_bonus, bonus_weight = self.bonus_value(batch, actions)
            scale = self.bonus_scale_at(env_steps)
            q_total = q + scale * bonus_weight * q_bonus
            info['bonus_scale_effective'] = scale
            info['q_bonus_pi_mean'] = (bonus_weight * q_bonus).sum() / jnp.maximum(bonus_weight.sum(), 1.0)
            info['bonus_term_mean'] = (scale * bonus_weight * q_bonus).mean()
            info['bonus_frac_rows'] = bonus_weight.mean()
        else:
            q_total = q
        if self.uses_rnd_bonus:
            # Independent RND bonus: + rnd_bonus_scale(t) * min_i Q_rnd_i(s, a_pi) on the online rows only (Q_rnd
            # is fit on them alone), on top of whatever `add_explore` added above; its own schedule (rnd_time_frac).
            q_rnd, rnd_weight = self.rnd_bonus_value(batch, actions)
            rnd_scale = self.bonus_scale_at(
                env_steps, scale0=self.config['rnd_bonus_scale'], frac=self.config['rnd_time_frac']
            )
            q_total = q_total + rnd_scale * rnd_weight * q_rnd
            info['rnd_bonus_scale_effective'] = rnd_scale
            info['q_rnd_pi_mean'] = (rnd_weight * q_rnd).sum() / jnp.maximum(rnd_weight.sum(), 1.0)
            info['rnd_bonus_term_mean'] = (rnd_scale * rnd_weight * q_rnd).mean()

        actor_loss = (alpha * log_probs - q_total).mean()
        # Entropy temperature: alpha(s) * (H(s) - H_target(s)), H from the stop-gradient sample. With a scalar
        # alpha and constant target this equals the JaxGCRL form alpha * (mean H - H_target).
        alpha_loss = (alpha_param * (entropy - target_entropy)).mean()

        total_loss = actor_loss + alpha_loss
        info.update(
            {
                'total_loss': total_loss,
                'actor_loss': actor_loss,
                'alpha_loss': alpha_loss,
                'alpha': jnp.mean(alpha),
                'entropy': entropy.mean(),
                'target_entropy': jnp.mean(target_entropy),
                'q_pi_mean': q.mean(),
                'std': dist._distribution.stddev().mean(),
            }
        )
        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, env_steps=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, bonus_rng, diag_rng = jax.random.split(rng, 4)
        rnd_rng, rnd_critic_rng = jax.random.split(rng)  # `rng` is not read again below

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, env_steps=env_steps)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        if self.uses_q_bonus:
            # Disjoint parameters again (bonus_critic only), so one shared Adam step equals a separate update.
            bonus_loss, bonus_info = self.bonus_critic_loss(batch, grad_params, bonus_rng)
            for k, v in bonus_info.items():
                info[f'bonus_critic/{k}'] = v
            loss = loss + bonus_loss
        if self.uses_distill_bonus:
            # Disjoint parameters once more (distill_critic only).
            distill_loss, distill_info = self.distill_loss(batch, grad_params)
            for k, v in distill_info.items():
                info[f'distill/{k}'] = v
            loss = loss + distill_loss
        if self.uses_rnd:
            # rnd_predictor params only; the target's are held fixed by its optimizer.
            rnd_loss, rnd_info = self.rnd_predictor_loss(batch, grad_params, rnd_rng)
            for k, v in rnd_info.items():
                info[f'rnd/{k}'] = v
            loss = loss + rnd_loss
        if self.uses_rnd_bonus:
            # Disjoint parameters (rnd_critic only): the independent RND bonus critic.
            rnd_critic_loss, rnd_critic_info = self.rnd_critic_loss(batch, grad_params, rnd_critic_rng)
            for k, v in rnd_critic_info.items():
                info[f'rnd_critic/{k}'] = v
            loss = loss + rnd_critic_loss
        if self.config['add_explore'] is not None and self.config['bonus_grad_diagnostics']:
            for k, v in self.bonus_grad_stats(batch, diag_rng).items():
                info[f'bonus_grad/{k}'] = jax.lax.stop_gradient(v)
        return loss, info

    def target_update(self, network, module_name):
        """Polyak-average `module_name`'s params into `target_<module_name>` (in place, as agents/sac.py)."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['bonus_tau'] + tp * (1 - self.config['bonus_tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, env_steps=None):
        """Update the agent and return a new agent with information dictionary.

        `env_steps` (the env step count at the start of this round, from `main_online.py`'s
        loop) drives the exploration-bonus annealing (`bonus_scale_at`); leave it None for
        plain flat CRL, or when `explore_reward_time_frac` is left at its default (no annealing).
        """
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, env_steps=env_steps)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if self.uses_q_bonus:
            self.target_update(new_network, 'bonus_critic')
        if self.uses_rnd_bonus:
            self.target_update(new_network, 'rnd_critic')
        return self.replace(network=new_network, rng=new_rng), info

    # ── Acting ────────────────────────────────────────────────────────────────

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor (temperature=0 gives the tanh(mean) action)."""
        if seed is None:
            seed = self.rng
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations (also used as example goals).
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if config['discrete']:
            raise NotImplementedError('online_crl supports continuous action spaces only.')

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Define networks.
        critic_def = GCBilinearValue(
            hidden_dims=tuple(config['value_hidden_dims']),
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
            state_encoder=encoders.get('critic_state'),
            goal_encoder=encoders.get('critic_goal'),
        )
        actor_def = GCActor(
            hidden_dims=tuple(config['actor_hidden_dims']),
            action_dim=action_dim,
            log_std_min=-5,
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
            gc_encoder=encoders.get('actor'),
        )
        # Exploration reward bonus mode (None -> off).
        add_explore = config['add_explore']
        if add_explore in (None, 'none', ''):
            add_explore = None
        elif add_explore not in ADD_EXPLORE_MODES:
            raise ValueError(f'online_crl: add_explore must be one of {ADD_EXPLORE_MODES} or none, got {add_explore!r}')
        uses_rnd_reward = add_explore in Q_BONUS_MODES and config['explore_reward'] == 'rnd'
        uses_rnd_bonus = config['rnd_bonus_scale'] is not None
        uses_rnd = uses_rnd_reward or uses_rnd_bonus
        if uses_rnd_bonus and uses_rnd_reward:
            raise ValueError(
                "online_crl: rnd_bonus_scale (the independent RND critic Q_rnd) and explore_reward='rnd' (RND through "
                'Q_x) would be RND twice; use one of them.'
            )
        if uses_rnd and config['encoder'] is not None:
            raise NotImplementedError('online_crl: RND is implemented for state observations only.')
        if add_explore is not None:
            if add_explore in Q_BONUS_MODES and config['explore_reward'] not in EXPLORE_REWARDS:
                raise ValueError(
                    f"online_crl: explore_reward must be one of {EXPLORE_REWARDS}, got {config['explore_reward']!r}"
                )
            if add_explore in DISTILL_MODES and config['distill_target'] not in DISTILL_TARGETS:
                raise ValueError(
                    f"online_crl: distill_target must be one of {DISTILL_TARGETS}, got {config['distill_target']!r}"
                )
            if uses_rnd_reward:
                pass  # online-only by construction (add_explore == 'reward'); no estimator needed
            elif config['emp_checkpoint_path'] is None:  # the empowerment bonuses are functions of E(s)
                raise ValueError(
                    f'online_crl: add_explore={add_explore!r} needs --agent.emp_checkpoint_path '
                    '(the frozen estimator that supplies E(s)).'
                )

        # Frozen offline empowerment estimator (optional): per-bin entropy temperatures when
        # `emp_entropy_target`, and/or the empowerment exploration reward.
        emp_agent = None
        emp_resolved = None
        if config['emp_checkpoint_path'] is not None:
            assert int(config['emp_num_bins']) >= 1, 'emp_num_bins must be >= 1'
            emp_agent, emp_resolved = load_empowerment_estimator(seed, ex_observations, ex_actions, config)
        if emp_agent is not None and config['emp_entropy_target']:
            alpha_def = LogParamVector(size=int(config['emp_num_bins']))
        else:
            alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            alpha=(alpha_def, ()),
        )
        if add_explore is not None:
            # Non-goal-conditioned twin MLP over (s, a): Q_x for the exploration reward (+ its Polyak
            # target copy), or the distilled future-max empowerment E' (plain regression, no target copy).
            bonus_encoder = None
            if config['encoder'] is not None:
                bonus_encoder = GCEncoder(state_encoder=encoder_modules[config['encoder']]())
            bonus_critic_def = GCValue(
                hidden_dims=tuple(config['value_hidden_dims']),
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=bonus_encoder,
            )
            if add_explore in DISTILL_MODES:
                network_info['distill_critic'] = (bonus_critic_def, (ex_observations, None, ex_actions))
            else:
                network_info['bonus_critic'] = (bonus_critic_def, (ex_observations, None, ex_actions))
                network_info['target_bonus_critic'] = (
                    copy.deepcopy(bonus_critic_def),
                    (ex_observations, None, ex_actions),
                )
        if uses_rnd_bonus:
            # The independent RND bonus critic Q_rnd (+ Polyak target), the Q_x architecture; state-based only.
            rnd_critic_def = GCValue(
                hidden_dims=tuple(config['value_hidden_dims']),
                layer_norm=config['layer_norm'],
                ensemble=True,
            )
            network_info['rnd_critic'] = (rnd_critic_def, (ex_observations, None, ex_actions))
            network_info['target_rnd_critic'] = (copy.deepcopy(rnd_critic_def), (ex_observations, None, ex_actions))
        if uses_rnd:
            # Fixed random target f and trained predictor f_hat, same architecture, over the whitened landing state.
            rnd_kwargs = dict(hidden_dims=tuple(config['rnd_hidden_dims']), rep_dim=int(config['rnd_rep_dim']))
            network_info['rnd_target'] = (RNDEmbedding(**rnd_kwargs), (ex_observations,))
            network_info['rnd_predictor'] = (RNDEmbedding(**rnd_kwargs), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if uses_rnd:
            # The predictor trains with its own Adam (paper: 1e-4); the target never moves.
            rnd_labels = {'modules_rnd_predictor': 'rnd_predictor', 'modules_rnd_target': 'rnd_target'}
            network_tx = optax.multi_transform(
                {
                    'agent': optax.adam(learning_rate=config['lr']),
                    'rnd_predictor': optax.adam(learning_rate=config['rnd_lr']),
                    'rnd_target': optax.set_to_zero(),
                },
                lambda params: {k: rnd_labels.get(k, 'agent') for k in params},
            )
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        if add_explore in Q_BONUS_MODES:
            network_params['modules_target_bonus_critic'] = network_params['modules_bonus_critic']
        if uses_rnd_bonus:
            network_params['modules_target_rnd_critic'] = network_params['modules_rnd_critic']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        # Future-goal sampling discount for the replay buffer (per env-step row).
        stored_config['goal_discount'] = float(config['discount'])
        stored_config['emp_stats_ready'] = False
        stored_config['emp_agent_name'] = None
        stored_config['emp_mean_metric'] = None
        stored_config['add_explore'] = add_explore
        if config['bonus_discount'] is None:
            stored_config['bonus_discount'] = float(config['discount'])
        if emp_resolved is not None:
            stored_config['emp_checkpoint_path'] = emp_resolved['ckpt_path']
            stored_config['emp_restore_epoch'] = emp_resolved['restore_epoch']
            stored_config['emp_agent_name'] = emp_resolved['agent_name']
            stored_config['emp_env_name'] = emp_resolved['env_name']
            stored_config['emp_mean_metric'] = emp_resolved['mean_metric']
            if config['emp_entropy_target']:
                print(
                    f"[online_crl] empowerment entropy target: H_target(s) = {stored_config['target_entropy']:.3f} + "
                    f"{float(config['emp_lambda'])} * (E(s) - E_mean), {int(config['emp_num_bins'])} alpha bins"
                )
            else:
                print('[online_crl] emp_entropy_target=False: scalar alpha, constant target entropy')
        if add_explore is not None:
            rows = 'online rows only' if add_explore == 'reward' else 'online + RLPD rows'
            formula = {
                'empowerment': "E(s') - E_mean",
                'max_episodic_empowerment': "max_{t' <= t+1} E(s_t') - E_mean (episode running max)",
                'rnd': "||f_hat(s') - f(s')||^2 / running std of the intrinsic return (RND)",
            }[config['explore_reward']]
            frac = config['explore_reward_time_frac']
            total = config['total_env_steps']
            if frac is None:
                schedule = 'no annealing (explore_reward_time_frac unset) -- constant bonus_scale'
            elif total is None:
                schedule = f'explore_reward_time_frac={float(frac)} but total_env_steps unknown -- constant bonus_scale'
            elif float(frac) <= 0.0:
                schedule = 'explore_reward_time_frac<=0 -- bonus off from step 0'
            else:
                frac = float(frac)
                schedule = f'linearly decayed to 0 by env step {int(round(frac * total))} (explore_reward_time_frac={frac})'
            if add_explore in DISTILL_MODES:
                rows = 'online rows only' if add_explore == 'distill' else 'online + RLPD rows'
                print(
                    f"[online_crl] distilled empowerment bonus: add_explore={add_explore}, E'(s_t, a_t) regressed "
                    f"onto {'max_k' if config['distill_target'] == 'episode_max' else 'max_{k>t}'} E(s_k) - E_mean "
                    f"(distill_target={config['distill_target']}) over closed trajectories ({rows}), actor += bonus_scale(t) * "
                    f"E'(s, a) on {rows}, bonus_scale(0)={float(config['bonus_scale'])}, {schedule}"
                )
            else:
                print(
                    f"[online_crl] exploration bonus: add_explore={add_explore} (Q_x fit on {rows}), "
                    f"explore_reward={config['explore_reward']} (r_x = {formula}), actor += "
                    f"bonus_scale(t) * Q_x(s, a), bonus_scale(0)={float(config['bonus_scale'])}, {schedule}, "
                    f"bonus_discount={stored_config['bonus_discount']}, bonus_tau={float(config['bonus_tau'])}"
                )

        if uses_rnd_bonus:
            rfrac = config['rnd_time_frac']
            total = config['total_env_steps']
            if rfrac is None or total is None:
                rschedule = 'constant rnd_bonus_scale (rnd_time_frac unset)' if rfrac is None else 'total_env_steps unknown -- constant'
            elif float(rfrac) <= 0.0:
                rschedule = 'rnd_time_frac<=0 -- RND bonus off from step 0'
            else:
                rschedule = f'linearly decayed to 0 by env step {int(round(float(rfrac) * total))} (rnd_time_frac={float(rfrac)})'
            print(
                f"[online_crl] independent RND bonus: Q_rnd(s, a) fit on online rows only, actor += rnd_bonus_scale(t) * "
                f"Q_rnd(s, a) on online rows next to add_explore={add_explore}, rnd_bonus_scale(0)="
                f"{float(config['rnd_bonus_scale'])}, {rschedule}, bonus_discount={stored_config['bonus_discount']}, "
                f"bonus_tau={float(config['bonus_tau'])}"
            )

        rnd_state = None
        if uses_rnd:
            obs_shape = tuple(ex_observations.shape[1:])
            rnd_state = dict(
                obs_rms=RunningMeanStd(
                    mean=jnp.zeros(obs_shape, jnp.float32),
                    var=jnp.ones(obs_shape, jnp.float32),
                    count=jnp.float32(0.0),
                    clip_max=float(config['rnd_obs_clip']),
                ),
                ret_rms=RunningMeanStd(mean=jnp.float32(0.0), var=jnp.float32(1.0), count=jnp.float32(0.0)),
                ret_carry=jnp.float32(0.0),
            )
            print(
                f"[online_crl] RND: target and predictor both ReLU MLP {tuple(config['rnd_hidden_dims'])} -> "
                f"{int(config['rnd_rep_dim'])}-d, input whitened by running obs stats and clipped to "
                f"+-{float(config['rnd_obs_clip'])}, reward ||f_hat - f||^2 / running std of the discounted intrinsic return "
                f"(gamma={stored_config['bonus_discount']}), predictor Adam lr={float(config['rnd_lr'])} on a "
                f"{float(config['rnd_update_proportion'])} share of the online rows, "
                f"{'non-episodic' if config['rnd_nonepisodic'] else 'episodic'} Q_x, actor bonus on online rows only"
            )

        return cls(
            rng, network=network, config=flax.core.FrozenDict(**stored_config), emp_agent=emp_agent, rnd_state=rnd_state
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='online_crl',  # Agent name.
            rollout_type='flat',  # Experience collector (see utils/online_rollout.py).
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor (future-goal sampling: P(offset=j) ~ discount^j).
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None -> -mult * dim(A)).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy (JaxGCRL: 0.5).
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            # Empowerment-modulated entropy target (None -> off; see the module docstring).
            emp_checkpoint_path=ml_collections.config_dict.placeholder(str),  # Frozen offline estimator run dir.
            emp_restore_epoch=ml_collections.config_dict.placeholder(int),  # Estimator epoch (None -> latest).
            emp_num_splus_samples=64,  # Future samples per skill and state for the skill estimator's E(s).
            emp_lambda=1.0,  # Nats of target entropy per nat of (E(s) - E_mean); 1 is unit-consistent.
            emp_num_bins=8,  # Quantile bins of E, each with its own auto-tuned alpha.
            emp_mean=ml_collections.config_dict.placeholder(float),  # E_mean override (None -> metric, else rows).
            emp_fast_path=True,  # Einsum rewrite of the skill estimator (False -> the checkpoint's own method).
            emp_entropy_target=True,  # Per-state entropy target from E(s) (above); False -> scalar alpha, E unused there.
            # Exploration reward bonus (None -> off; see the module docstring).
            add_explore=ml_collections.config_dict.placeholder(str),  # 'reward' (Q_x on online rows) | 'reward-to-rlpd'
            # | 'distill' (regressed future-max E'(s, a), online rows only) | 'distill-to-rlpd' (online + RLPD rows).
            explore_reward='empowerment',  # Bonus reward r_x: 'empowerment' (E(s') - E_mean) | 'max_episodic_empowerment'.
            bonus_scale=1.0,  # Actor weight on Q_x(s, a) / E'(s, a) next to the CRL critic ("alpha" of the bonus).
            distill_target='episode_max',  # E' target: 'episode_max' (max of E over the WHOLE trajectory, one value per
            # trajectory) | 'future_max' (max over the states after s_t only). Distill modes only.
            bonus_grad_diagnostics=False,  # Log per-term actor gradient norms under bonus_grad/* (3 extra backward passes).
            bonus_discount=ml_collections.config_dict.placeholder(float),  # Bellman discount of Q_x (None -> discount).
            bonus_tau=0.005,  # Polyak rate of Q_x's target copy (SAC default).
            explore_reward_time_frac=ml_collections.config_dict.placeholder(float),  # None (default) ->
            # NO annealing, constant bonus_scale. Set to a fraction in (0, 1] to linearly decay
            # bonus_scale to 0 by that fraction of total_env_steps, held at 0 after (<=0 -> bonus
            # off from the start); needs total_env_steps (set by main_online.py from --total_env_steps).
            total_env_steps=ml_collections.config_dict.placeholder(int),  # Set by main_online.py; None -> no decay.
            # Random network distillation (explore_reward='rnd'; module docstring). Defaults follow the paper
            # where its text is explicit (Sec. 2.4, Tables 4-5) and its code where the text is silent (net sizes).
            rnd_hidden_dims=(512, 512),  # ReLU MLP trunk shared by target and predictor (paper: "encoder + dense layers").
            rnd_rep_dim=512,  # Embedding size k (the paper's code uses 512; its text leaves k open).
            rnd_lr=1e-4,  # The predictor's own Adam learning rate (paper Table 5: 1e-4); the target is never updated.
            rnd_update_proportion=1.0,  # Share of the online rows in the predictor loss (paper: 1 single-env, 0.25 at 128 envs).
            rnd_obs_clip=5.0,  # Clip of the whitened RND input (paper: [-5, 5]).
            rnd_nonepisodic=True,  # Q_x bootstraps through episode ends (paper: the intrinsic reward is non-episodic).
            # Independent RND bonus critic Q_rnd, combinable with any add_explore (module docstring). None -> off.
            rnd_bonus_scale=ml_collections.config_dict.placeholder(float),  # Actor weight on Q_rnd(s, a) (online rows).
            rnd_time_frac=ml_collections.config_dict.placeholder(float),  # Its own linear decay fraction (None -> constant).
            # Online schedule (consumed by main_online.py).
            unroll_length=50,  # Env steps collected between update rounds (JaxGCRL unroll_length).
            utd_ratio=1,  # Gradient steps per env step; each round runs unroll_length * utd_ratio updates.
            min_replay_size=1000,  # Transitions collected before the first update.
            replay_size=1000000,  # Replay buffer capacity in transitions.
            offline_ratio=0.5,  # RLPD (--offline_dataset): fraction of every batch drawn from the offline buffer.
            # Observation pipeline.
            discrete=False,  # Whether the action space is discrete (unsupported here).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None for state-based).
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
