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
"""

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
from utils.networks import GCActor, GCBilinearValue, LogParam, LogParamVector
from utils.skill_checkpoint import latest_epoch

# Offline estimator families that expose `empowerment(observations, rng) -> [B]` (nats).
EMPOWERMENT_ESTIMATOR_AGENTS = (
    'empowerment_skill',
    'empowerment_crl',
    'empowerment_crl_flowbc',
    'empowerment_dads',
    'empowerment_dv',
)


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

    # ── Empowerment estimator ─────────────────────────────────────────────────

    @property
    def uses_empowerment(self):
        return self.emp_agent is not None

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

    def actor_loss(self, batch, grad_params, rng):
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
        if self.uses_empowerment:
            # Per-state entropy target from the row's offline empowerment; temperature per E bin.
            if not self.config['emp_stats_ready']:
                raise RuntimeError(
                    'online_crl: empowerment stats (E_mean, bin edges) are not finalised; '
                    'main_online.py must call agent.with_empowerment_stats before the first update.'
                )
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

        actor_loss = (alpha * log_probs - q).mean()
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
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
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
        # Frozen offline empowerment estimator (optional) -> per-bin entropy temperatures.
        emp_agent = None
        emp_resolved = None
        if config['emp_checkpoint_path'] is not None:
            assert int(config['emp_num_bins']) >= 1, 'emp_num_bins must be >= 1'
            emp_agent, emp_resolved = load_empowerment_estimator(seed, ex_observations, ex_actions, config)
            alpha_def = LogParamVector(size=int(config['emp_num_bins']))
        else:
            alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        # Future-goal sampling discount for the replay buffer (per env-step row).
        stored_config['goal_discount'] = float(config['discount'])
        stored_config['emp_stats_ready'] = False
        stored_config['emp_agent_name'] = None
        stored_config['emp_mean_metric'] = None
        if emp_resolved is not None:
            stored_config['emp_checkpoint_path'] = emp_resolved['ckpt_path']
            stored_config['emp_restore_epoch'] = emp_resolved['restore_epoch']
            stored_config['emp_agent_name'] = emp_resolved['agent_name']
            stored_config['emp_env_name'] = emp_resolved['env_name']
            stored_config['emp_mean_metric'] = emp_resolved['mean_metric']
            print(
                f"[online_crl] empowerment entropy target: H_target(s) = {stored_config['target_entropy']:.3f} + "
                f"{float(config['emp_lambda'])} * (E(s) - E_mean), {int(config['emp_num_bins'])} alpha bins"
            )

        return cls(rng, network=network, config=flax.core.FrozenDict(**stored_config), emp_agent=emp_agent)


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
