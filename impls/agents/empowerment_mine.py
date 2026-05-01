"""
Offline MINE-based empowerment agent for OGBench.

Estimates E(s) = I(S'_proj ; A | s) using the Donsker–Varadhan lower bound
of mutual information (MINE; Belghazi et al. 2018). The agent is a port of
JaxGCRL's online MINE empowerment exploration bonus
(`jaxgcrl/agents/go_explore/online_mine_empowerment.py`) adapted to OGBench's
offline setting:

  - World model p_w(s' | s, a) = N(s + μ_w, diag σ²_w)  (DADS-style; trained
    by Gaussian NLL on Δs over the **full** state).
  - Statistics network T_φ(s, a, proj(s')) → R, trained by maximising
        E_{(s,a,s')~data}[T] − log E_{(s,a*,s'_marg) ~ π × p_w}[exp T] .
    Joint samples are real offline transitions; marginal samples come from
    two independent draws of the actor (one for the action coordinate a*,
    one for the dynamics input a**, which is then pushed through the world
    model to produce s'_marg).
  - Stochastic actor π_θ(a | s), trained to maximise the MINE per-sample
    score with SAC-style entropy regularisation (auto-tuned α via
    LogParam, target entropy −0.5 · action_dim by default), plus a BC
    constraint on the offline action distribution.

`obs_indices` projects the next state onto a goal-relevant subspace before
scoring with T (e.g. xy in antmaze). The world model still predicts the
full state — only the input to T is projected — so empowerment is
I(proj(S'); A | s) under the policy.

Polyak target networks (`target_t`, `target_world_model`) provide stable
inputs to non-stationarity-sensitive callers: the actor sees a slow-moving
score and the MINE T loss sees a stationary marginal distribution. Targets
are soft-updated with rate `tau` after every gradient step, matching the
empowerment_action / DADS conventions.
"""

import copy
from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from jax.scipy.special import logsumexp

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, LogParam, MLP, TransformedWithMode, default_init


# ── Helpers ───────────────────────────────────────────────────────────────────


def _gaussian_log_prob_diag(x, mean, log_std):
    """Sum of per-dim log N(x; mean, diag(exp(2*log_std)))."""
    var = jnp.exp(2.0 * log_std)
    log_prob_per_dim = -0.5 * (
        jnp.log(2.0 * jnp.pi) + 2.0 * log_std + (x - mean) ** 2 / var
    )
    return log_prob_per_dim.sum(axis=-1)


# ── Network modules ───────────────────────────────────────────────────────────


class WorldModel(nn.Module):
    """p_w(s' | s, a) = N(s + μ(s,a), diag(σ²(s,a))) on the full state space.

    Identical to DADSAgent.WorldModel — copied here so this agent has no
    cross-agent imports.
    """

    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    def setup(self):
        self.trunk = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_head = nn.Dense(self.state_dim, kernel_init=default_init())
        self.log_std_head = nn.Dense(self.state_dim, kernel_init=default_init())

    def __call__(self, observations, actions):
        h = self.trunk(jnp.concatenate([observations, actions], axis=-1))
        mean = self.mean_head(h)
        log_std = jnp.clip(self.log_std_head(h), self.log_std_min, self.log_std_max)
        return mean, log_std


class StatisticsT(nn.Module):
    """MINE statistics network T(s, a, s'_proj) → R."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, actions, next_observations_proj):
        x = jnp.concatenate([observations, actions, next_observations_proj], axis=-1)
        out = MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)(x)
        return out.squeeze(-1)


class EmpowermentMineActor(GCActor):
    """Continuous Gaussian-tanh actor π(a|s); ignores goals."""

    def __call__(self, observations, goals=None, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, None)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        elif self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_stds
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        dist = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            dist = TransformedWithMode(dist, distrax.Block(distrax.Tanh(), ndims=1))
        return dist


# ── Agent ────────────────────────────────────────────────────────────────────


class EmpowermentMineAgent(flax.struct.PyTreeNode):
    """Offline MINE empowerment agent.

    Network layout (ModuleDict keys):
      world_model : p_w(s'|s,a)  — Gaussian over Δs, trained by NLL on data
      t           : T(s, a, proj(s')) — MINE statistics network
      policy      : π(a|s)       — Gaussian-tanh actor
      alpha       : SAC entropy temperature (LogParam)
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _proj(self, states):
        """Project states onto the obs_indices subspace (identity if None)."""
        obs_indices = self.config.get('obs_indices', None)
        if obs_indices is not None:
            return states[..., jnp.array(obs_indices)]
        return states

    def _world_model_sample(self, observations, actions, eps_rng,
                             *, wm_module='world_model', params=None):
        """Reparameterised sample s' = s + μ_w + σ_w · ε."""
        mean, log_std = self.network.select(wm_module)(
            observations, actions, params=params
        )
        eps = jax.random.normal(eps_rng, mean.shape)
        return observations + mean + jnp.exp(log_std) * eps

    def _marginal_t_samples(self, observations, rng, *,
                             t_module='t', wm_module='world_model',
                             t_params=None, wm_params=None, policy_params=None):
        """Score (s, a*, proj(s'_marg)) with T, where (a*, s'_marg) is a sample
        from the marginal π(a|s) p(s'|s).

        Two independent action draws — a* for the action coordinate of the
        marginal, a** for the world-model input so s'_marg = s + Δ_w(s, a**).
        Independence is required for the marginal to factorise as
        π(a|s) p(s'|s) (matching JaxGCRL, online_mine_empowerment.py:182-195).

        `t_module` / `wm_module` select which T (current vs target) and which
        world model (current vs target) is used.
        """
        a_star_key, a_dyn_key, eps_key = jax.random.split(rng, 3)
        dist_star = self.network.select('policy')(observations, params=policy_params)
        a_star = dist_star.sample(seed=a_star_key)
        dist_dyn = self.network.select('policy')(observations, params=policy_params)
        a_dyn = dist_dyn.sample(seed=a_dyn_key)
        s_marg = self._world_model_sample(
            observations, a_dyn, eps_key, wm_module=wm_module, params=wm_params,
        )
        t_marg = self.network.select(t_module)(
            observations, a_star, self._proj(s_marg), params=t_params,
        )
        return t_marg

    # ── Losses ───────────────────────────────────────────────────────────────

    def world_model_loss(self, batch, grad_params):
        """NLL of Δs = s' − s under p_w(·|s, a)."""
        obs = batch['observations']
        next_obs = batch['next_observations']
        actions = batch['actions']
        delta = next_obs - obs

        mean, log_std = self.network.select('world_model')(obs, actions, params=grad_params)
        log_prob = _gaussian_log_prob_diag(delta, mean, log_std)
        loss = -log_prob.mean()
        return loss, {
            'world_model_loss': loss,
            'world_model_mean_abs_err': jnp.mean(jnp.abs(delta - mean)),
            'world_model_log_std_mean': log_std.mean(),
        }

    def mine_t_loss(self, batch, grad_params, rng):
        """Negated Donsker–Varadhan bound. Gradient flows through T only.

        Joint samples come from the offline batch (s, a, proj(s')); marginal
        samples come from π × p_w with the actor and world model frozen.
        """
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        t_joint = self.network.select('t')(
            obs, actions, self._proj(next_obs), params=grad_params,
        )
        # Marginal: gradient flows through T only. Use the **target** world
        # model so the marginal distribution T sees stays stationary across
        # consecutive updates (the analogue of SAC's target-bootstrap).
        t_marg = self._marginal_t_samples(
            obs, rng,
            t_module='t', wm_module='target_world_model',
            t_params=grad_params,
        )

        n = jnp.asarray(t_marg.shape[0], dtype=t_marg.dtype)
        log_partition = logsumexp(t_marg) - jnp.log(n)
        bound = jnp.mean(t_joint) - log_partition
        loss = -bound
        return loss, {
            'mine_loss': loss,
            'mine_lower_bound': bound,
            'mine_t_joint_mean': jnp.mean(t_joint),
            'mine_t_marg_mean': jnp.mean(t_marg),
            'mine_log_partition': log_partition,
        }

    def actor_alpha_loss(self, batch, grad_params, rng):
        """SAC actor + dual-descent α. Gradient flows through policy and α only.

        - Reparam-sample a ~ π(·|s) and s' ~ p_w(·|s,a); score with T (frozen)
          to get the per-sample DV contribution; the policy maximises this
          minus α · log π(a|s).
        - α is updated by the standard SAC dual loss
          −E[α · stop_grad(log π + target_entropy)] (matches dads.py:378-382).
        """
        obs = batch['observations']
        a_key, dyn_key, marg_key = jax.random.split(rng, 3)

        # Reparameterised action sample with gradient flow through policy.
        dist = self.network.select('policy')(obs, params=grad_params)
        actions = dist.sample(seed=a_key)
        log_probs = dist.log_prob(actions)

        # Reparameterised next-state sample from the **target** world model
        # (frozen via no params=, and target rather than current so the
        # actor's reparam grad sees a slow-moving dynamics).
        s_next = self._world_model_sample(
            obs, actions, dyn_key, wm_module='target_world_model',
        )

        # T score from the **target** T network — stable improvement signal,
        # SAC-style. Gradient still flows to the policy through the
        # reparameterised actions and s_next.
        t_score = self.network.select('target_t')(obs, actions, self._proj(s_next))

        # log_partition over a fresh marginal batch using the target
        # networks (frozen for grad and slow-moving for stability).
        log_partition = jax.lax.stop_gradient(
            logsumexp(
                self._marginal_t_samples(
                    obs, marg_key,
                    t_module='target_t', wm_module='target_world_model',
                )
            )
            - jnp.log(obs.shape[0])
        )
        score = t_score - log_partition

        alpha_no_grad = self.network.select('alpha')()
        actor_loss = (alpha_no_grad * log_probs - score).mean()

        # α loss: gradient through α only; log_probs detached.
        alpha = self.network.select('alpha')(params=grad_params)
        target_entropy = self.config['target_entropy']
        alpha_loss = -(alpha * jax.lax.stop_gradient(log_probs + target_entropy)).mean()

        total = actor_loss + alpha_loss
        return total, {
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': -log_probs.mean(),
            'score_mean': score.mean(),
            't_score_mean': t_score.mean(),
        }

    def bc_loss(self, batch, grad_params):
        """−log π(a_data | s)."""
        dist = self.network.select('policy')(batch['observations'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        loss = -log_prob.mean()
        return loss, {
            'bc_loss': loss,
            'bc_log_prob_mean': log_prob.mean(),
            'bc_log_prob_min': log_prob.min(),
            'bc_log_prob_max': log_prob.max(),
        }

    # ── Empowerment as a state function ──────────────────────────────────────

    def empowerment(self, observations, rng):
        """E(s) ≈ E_{a~π(·|s), s'~p_w(·|s,a)} [T(s, a, proj(s')) − log_Z],

        where log_Z = log E_marg[exp T] is estimated from a fresh batch of
        marginal samples. Pure state function — no gradient, no policy
        update happens here. Mirrors the convention used by
        empowerment_action.empowerment / dads.empowerment so the existing
        plot_empowerment_map_*.py scripts work unchanged.
        """
        M = self.config['num_a_samples']
        action_rng, dyn_rng, marg_rng = jax.random.split(rng, 3)

        action_seeds = jax.random.split(action_rng, M)
        dyn_seeds = jax.random.split(dyn_rng, M)

        def score_for_sample(a_seed, d_seed):
            dist = self.network.select('policy')(observations)
            a = dist.sample(seed=a_seed)
            s_next = self._world_model_sample(
                observations, a, d_seed, wm_module='target_world_model',
            )
            return self.network.select('target_t')(observations, a, self._proj(s_next))

        t_per_sample = jax.vmap(score_for_sample)(action_seeds, dyn_seeds)  # [M, B]

        log_partition = (
            logsumexp(
                self._marginal_t_samples(
                    observations, marg_rng,
                    t_module='target_t', wm_module='target_world_model',
                )
            )
            - jnp.log(observations.shape[0])
        )
        return t_per_sample.mean(axis=0) - log_partition  # [B]

    # ── Combined loss ────────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        wm_rng, t_rng, actor_rng, emp_rng = jax.random.split(rng, 4)
        info = {}

        wm_loss, wm_info = self.world_model_loss(batch, grad_params)
        info.update({f'world_model/{k}': v for k, v in wm_info.items()})

        t_loss, t_info = self.mine_t_loss(batch, grad_params, t_rng)
        info.update({f't/{k}': v for k, v in t_info.items()})

        actor_loss, actor_info = self.actor_alpha_loss(batch, grad_params, actor_rng)
        info.update({f'actor/{k}': v for k, v in actor_info.items()})

        bc_loss_val, bc_info = self.bc_loss(batch, grad_params)
        info.update({f'bc/{k}': v for k, v in bc_info.items()})

        # State-function empowerment metric (no gradient).
        emp = jax.lax.stop_gradient(self.empowerment(batch['observations'], emp_rng))
        info['empowerment/mean'] = emp.mean()
        info['empowerment/min'] = emp.min()
        info['empowerment/max'] = emp.max()

        bc_alpha = jnp.asarray(self.config['bc_alpha'], dtype=jnp.float32)
        info['bc/alpha'] = bc_alpha
        total = wm_loss + t_loss + actor_loss + bc_alpha * bc_loss_val
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )

        # Polyak soft-update of target networks.
        tau = self.config['tau']
        new_target_t = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            new_network.params['modules_t'],
            new_network.params['modules_target_t'],
        )
        new_target_wm = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            new_network.params['modules_world_model'],
            new_network.params['modules_target_world_model'],
        )
        new_params = {
            **new_network.params,
            'modules_target_t': new_target_t,
            'modules_target_world_model': new_target_wm,
        }
        new_network = new_network.replace(params=new_params)
        return self.replace(network=new_network, rng=new_rng), info

    # ── Evaluation ───────────────────────────────────────────────────────────

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions from π. Goals accepted for API compat and ignored."""
        if seed is None:
            seed = self.rng

        single_obs = observations.ndim == 1
        if single_obs:
            observations = observations[None, :]

        dist = self.network.select('policy')(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1.0, 1.0)
        if single_obs:
            actions = actions[0]
        return actions

    # ── Constructor ──────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        state_dim = ex_observations.shape[-1]

        config = dict(config)
        config['state_dim'] = state_dim
        if config.get('target_entropy', None) is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # obs_indices projection (identity when None)
        obs_indices = config.get('obs_indices', None)
        if obs_indices is not None:
            ex_next_proj = ex_observations[:, jnp.array(obs_indices)]
        else:
            ex_next_proj = ex_observations

        hidden_dims = config['hidden_dims']
        wm_hidden_dims = config.get('world_model_hidden_dims') or hidden_dims
        t_hidden_dims = config.get('t_hidden_dims') or hidden_dims
        actor_hidden_dims = config.get('actor_hidden_dims') or hidden_dims
        layer_norm = config['layer_norm']

        # Encoders for image obs (mirrors empowerment_action / DADS)
        encoders = {}
        if config.get('encoder') is not None:
            enc = encoder_modules[config['encoder']]
            encoders = {k: GCEncoder(concat_encoder=enc()) for k in ('policy',)}

        world_model_def = WorldModel(
            hidden_dims=wm_hidden_dims,
            state_dim=state_dim,
            layer_norm=layer_norm,
            log_std_min=config['world_model_log_std_min'],
            log_std_max=config['world_model_log_std_max'],
        )
        target_world_model_def = copy.deepcopy(world_model_def)
        t_def = StatisticsT(
            hidden_dims=t_hidden_dims,
            layer_norm=layer_norm,
        )
        target_t_def = copy.deepcopy(t_def)
        policy_def = EmpowermentMineActor(
            hidden_dims=actor_hidden_dims,
            action_dim=action_dim,
            log_std_min=-5,
            log_std_max=2,
            tanh_squash=config['tanh_squash'],
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
            gc_encoder=encoders.get('policy'),
        )
        alpha_def = LogParam(init_value=config['init_alpha'])

        network_info = dict(
            world_model=(world_model_def, (ex_observations, ex_actions)),
            target_world_model=(target_world_model_def, (ex_observations, ex_actions)),
            t=(t_def, (ex_observations, ex_actions, ex_next_proj)),
            target_t=(target_t_def, (ex_observations, ex_actions, ex_next_proj)),
            policy=(policy_def, (ex_observations,)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        # Initialise targets as exact copies of their online counterparts so
        # the very first update sees identical predictions.
        network_params['modules_target_world_model'] = network_params['modules_world_model']
        network_params['modules_target_t'] = network_params['modules_t']

        network = TrainState.create(network_def, network_params, tx=network_tx)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ────────────────────────────────────────────────────────────────────


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='empowerment_mine',
        # ── Optimisation ─────────────────────────────────────────────────────
        lr=3e-4,
        batch_size=1024,
        # ── Architecture ─────────────────────────────────────────────────────
        hidden_dims=(512, 512, 512, 512),
        world_model_hidden_dims=ml_collections.config_dict.placeholder(tuple),
        t_hidden_dims=ml_collections.config_dict.placeholder(tuple),
        actor_hidden_dims=ml_collections.config_dict.placeholder(tuple),
        layer_norm=True,
        # ── World model ──────────────────────────────────────────────────────
        world_model_log_std_min=-5.0,
        world_model_log_std_max=2.0,
        # ── Subspace projection (e.g. (0, 1) for xy in antmaze) ──────────────
        obs_indices=ml_collections.config_dict.placeholder(tuple),
        # ── Target networks (Polyak averaging) ───────────────────────────────
        tau=0.005,
        # ── SAC entropy temperature ──────────────────────────────────────────
        init_alpha=1.0,
        target_entropy=ml_collections.config_dict.placeholder(float),
        target_entropy_multiplier=0.5,
        # ── Actor head ───────────────────────────────────────────────────────
        tanh_squash=False,
        actor_fc_scale=0.01,
        # ── BC constraint ────────────────────────────────────────────────────
        bc_alpha=0.1,
        # ── Empowerment metric (MC) ─────────────────────────────────────────
        num_a_samples=32,
        # ── Compatibility with main.py / GCDataset (fields unused by the
        #    agent itself, but consumed by GCDataset wiring). ────────────────
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='GCDataset',
        value_p_curgoal=0.0,
        value_p_trajgoal=0.0,
        value_p_randomgoal=1.0,
        value_geom_sample=True,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=True,
        p_aug=0.0,
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))
