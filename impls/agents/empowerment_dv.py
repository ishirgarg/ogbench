"""Offline action-level empowerment agent (Donsker-Varadhan with flow-matching negatives).

Variant of agents/empowerment_crl_flowbc.py. Same flow-matching BC policy and
dynamics model, same scalar critic; the ONLY differences are (a) the negatives
are full (action, next-state) pairs, decorrelated by a DERANGEMENT, and (b) the
critic optimizes the Donsker-Varadhan (DV) lower bound instead of InfoNCE.

Estimates effective empowerment I(s+; a | s) under the behavior policy, where
s+ is the discounted future state (GCDataset value_goals with geometric
sampling), via the DV bound. Per transition (s, a, s+):

    J = T(s+, a, s)  -  log( (1/N) * sum_{i=1..N} exp( T(s+_j(i), a_i, s) ) )

with N negatives built by sampling a_i ~ pi_bc(.|s), s+_i ~ f_dyn(.|s, a_i),
then applying a cyclic-shift derangement j(i) = (i + 1) mod N so that no a_i
keeps its OWN generated next-state. Each deranged pair (a_i, s+_j) is a sample
from the product of marginals pi_bc x f_dyn at s (action and next-state
independent given s). The critic does gradient ASCENT on J (minimizes -J); the
BC and dynamics flows are FROZEN through the negatives (stop-gradient on the
sampled a_i, s+_i).

Components (all trained by update()):
  - pi_bc(a|s): rectified-flow BC policy (sampler only; provides negative
    actions, query-time joints, and sample_actions).
  - f_dyn(s+|s,a): rectified-flow model of the discounted future (negative
    next-states and query-time joints at arbitrary states).
  - T(s+, a, s): MLP critic. Trained only after `pretrain_steps` so its
    negatives come from an already-converged BC/dyn pair. bc/dyn KEEP training
    in phase 2, so the negative distribution keeps (slowly) improving.

Caveats (see empowerment/crl_empowerment.py for the full discussion):
  - Negatives come from LEARNED pi_bc x f_dyn, so BC/dynamics error inflates the
    estimate; it lower-bounds I plus a non-negative KL term, never deflates it.
  - The DV bound is unbounded above (unlike InfoNCE's log(N+1) cap); with finite
    N the log-mean-exp is a biased estimate of the true normalizer.
  - `empowerment(s)` draws joints from pi_bc x f_dyn, i.e. it is MODEL
    empowerment (MI of the learned models), sign-indeterminate vs the truth.
  - GCDataset's geometric goal sampling CLAMPS overshoots to the trajectory's
    final state (repo convention), piling some future mass on terminal states.

This agent has no task actor; sample_actions returns pi_bc samples.
"""

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


class VelocityField(nn.Module):
    """Rectified-flow velocity field v(x_t, t, cond)."""

    hidden_dims: Sequence[int]
    out_dim: int
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x_t, t, cond):
        h = jnp.concatenate([x_t, t, cond], axis=-1)
        return MLP((*self.hidden_dims, self.out_dim), activate_final=False, layer_norm=self.layer_norm)(h)


class CriticT(nn.Module):
    """Scalar critic T(s+, a, s)."""

    hidden_dims: Sequence[int]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, s_plus, a, s):
        h = jnp.concatenate([s_plus, a, s], axis=-1)
        out = MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)(h)
        return out.squeeze(-1)


class EmpowermentDVAgent(flax.struct.PyTreeNode):
    """Offline Donsker-Varadhan empowerment estimator with flow-matching negatives."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # --------------------------------------------------------------- flows --
    def _flow_loss(self, module_name, x, cond, rng, grad_params):
        """Rectified-flow MSE: regress v(x_t, t, cond) onto (x - x_0)."""
        k0, k1 = jax.random.split(rng)
        x0 = jax.random.normal(k0, x.shape)
        t = jax.random.uniform(k1, (*x.shape[:-1], 1))
        x_t = (1 - t) * x0 + t * x
        v = self.network.select(module_name)(x_t, t, cond, params=grad_params)
        return jnp.mean(jnp.square(v - (x - x0)))

    def _flow_sample(self, module_name, cond, rng, out_dim, clip=None, params=None):
        """Euler-integrate the flow from x_0 ~ N(0, I). cond: [..., C]."""
        x = jax.random.normal(rng, (*cond.shape[:-1], out_dim))
        n = self.config['flow_steps']
        dt = 1.0 / n

        def step(i, x):
            t = jnp.full((*cond.shape[:-1], 1), i * dt)
            v = self.network.select(module_name)(x, t, cond, params=params)
            return x + dt * v

        x = jax.lax.fori_loop(0, n, step, x)
        if clip is not None:
            x = jnp.clip(x, -clip, clip)
        return x

    def sample_bc(self, observations, rng):
        """Actions from pi_bc(.|s), clipped to the action box."""
        return self._flow_sample('bc_flow', observations, rng, self.config['action_dim'], clip=1.0)

    def sample_dyn(self, observations, actions, rng):
        """Discounted futures from f_dyn(.|s, a)."""
        cond = jnp.concatenate([observations, actions], axis=-1)
        return self._flow_sample('dyn_flow', cond, rng, self.config['obs_dim'])

    # -------------------------------------------------------------- critic --
    def _neg_term(self, s, rng, grad_params):
        """DV log-mean-exp over N deranged product-of-marginals negatives at s.

        a_i ~ pi_bc(.|s), s+_i ~ f_dyn(.|s, a_i), then cyclic-shift derangement
        pairs a_i with s+_{(i+1) mod N}; both flows are stop-gradient'd. Returns
        logsumexp_i T(s+_j(i), a_i, s) - log(N), shape [B]."""
        B = s.shape[0]
        N = self.config['num_negatives']
        D = s.shape[-1]
        k_a, k_sp = jax.random.split(rng)
        s_rep = jnp.broadcast_to(s[:, None, :], (B, N, D))
        a_neg = jax.lax.stop_gradient(
            self._flow_sample('bc_flow', s_rep, k_a, self.config['action_dim'], clip=1.0)
        )  # [B, N, A]
        cond = jnp.concatenate([s_rep, a_neg], axis=-1)
        sp_neg = jax.lax.stop_gradient(
            self._flow_sample('dyn_flow', cond, k_sp, self.config['obs_dim'])
        )  # [B, N, D]
        # Derangement: pair a_i with s+_{(i+1) mod N} (roll by -1 along N). Valid
        # (no fixed points) for N >= 2, unlike a uniformly random permutation.
        sp_deranged = jnp.roll(sp_neg, shift=-1, axis=1)
        neg_logits = self.network.select('critic')(sp_deranged, a_neg, s_rep, params=grad_params)  # [B, N]
        return jax.nn.logsumexp(neg_logits, axis=1) - jnp.log(N)  # [B]

    def _dv_values(self, s, a, sp, rng, grad_params):
        """Per-element DV objective J = T(s+, a, s) - logmeanexp(negatives)."""
        pos_logit = self.network.select('critic')(sp, a, s, params=grad_params)  # [B]
        neg_term = self._neg_term(s, rng, grad_params)  # [B]
        return pos_logit - neg_term

    # -------------------------------------------------------------- losses --
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        k_bc, k_dyn, k_neg = jax.random.split(rng, 3)

        s = batch['observations']
        a = batch['actions']
        sp = batch['value_goals']  # geometric discounted future (GCDataset)

        bc_loss = self._flow_loss('bc_flow', a, s, k_bc, grad_params)
        dyn_loss = self._flow_loss('dyn_flow', sp, jnp.concatenate([s, a], axis=-1), k_dyn, grad_params)
        info['bc_loss'] = bc_loss
        info['dyn_loss'] = dyn_loss

        # Critic trains only once pi_bc/f_dyn negatives are converged. The
        # negative sampling (B*N flow integrations for both flows) dominates the
        # cost, so the whole branch is skipped via lax.cond during pretrain.
        in_critic = self.network.step >= self.config['pretrain_steps']

        def critic_branch(_):
            J = self._dv_values(s, a, sp, k_neg, grad_params)  # [B]
            return -jnp.mean(J)  # gradient ascent on J

        critic_loss = jax.lax.cond(in_critic, critic_branch, lambda _: jnp.float32(0.0), None)
        info['critic_loss'] = critic_loss
        info['mi_estimate'] = jnp.where(in_critic, -critic_loss, 0.0)  # DV bound (0 in pretrain)
        info['phase/in_critic'] = in_critic.astype(jnp.float32)
        loss = bc_loss + dyn_loss + critic_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )
        return self.replace(network=new_network, rng=new_rng), info

    # -------------------------------------------------- estimation for eval --
    @jax.jit
    def empowerment(self, observations, rng=None):
        """Model empowerment (DV) estimate at arbitrary states. [B, D] -> [B] nats.

        E(s) = (1/K) sum_k T(s+_k, a_k, s)  -  logmeanexp_i T(s+_j(i), a_i, s),
        with K positive joints (a_k ~ pi_bc, s+_k ~ f_dyn kept paired) and N
        deranged product-of-marginals negatives (same construction as training).
        """
        rng = rng if rng is not None else self.rng
        B = observations.shape[0]
        D = observations.shape[-1]
        K = self.config['est_num_joints']
        k_a, k_sp, k_neg = jax.random.split(rng, 3)

        s_rep = jnp.broadcast_to(observations[:, None, :], (B, K, D))  # [B, K, D]
        a_pos = self.sample_bc(s_rep, k_a)
        sp_pos = self.sample_dyn(s_rep, a_pos, k_sp)
        pos = self.network.select('critic')(sp_pos, a_pos, s_rep, params=None)  # [B, K]
        pos_term = pos.mean(axis=1)  # [B]
        neg_term = self._neg_term(observations, k_neg, None)  # [B]
        return pos_term - neg_term

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """pi_bc samples (this agent's only policy is the BC model)."""
        seed = seed if seed is not None else self.rng
        single = observations.ndim == 1
        if single:
            observations = observations[None]
        actions = self.sample_bc(observations, seed)
        return actions[0] if single else actions

    # ---------------------------------------------------------- constructor --
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        config = dict(config)
        assert config.get('encoder') is None, 'empowerment_dv is state-based only (no encoder support)'
        config['action_dim'] = ex_actions.shape[-1]
        config['obs_dim'] = ex_observations.shape[-1]

        bc_def = VelocityField(config['flow_hidden_dims'], config['action_dim'], config['layer_norm'])
        dyn_def = VelocityField(config['flow_hidden_dims'], config['obs_dim'], config['layer_norm'])
        critic_def = CriticT(config['critic_hidden_dims'], config['layer_norm'])

        ex_t = jnp.zeros((*ex_observations.shape[:-1], 1))
        ex_sa = jnp.concatenate([ex_observations, ex_actions], axis=-1)
        network_info = dict(
            bc_flow=(bc_def, (ex_actions, ex_t, ex_observations)),
            dyn_flow=(dyn_def, (ex_observations, ex_t, ex_sa)),
            critic=(critic_def, (ex_observations, ex_actions, ex_observations)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='empowerment_dv',
        # Matched to agents/empowerment_crl_flowbc.py for a like-for-like
        # comparison (same lr, batch size, trunk architecture, layer norm).
        lr=3e-4,
        batch_size=256,
        flow_hidden_dims=(512, 512, 512),
        critic_hidden_dims=(512, 512, 512),
        layer_norm=True,
        flow_steps=8,           # Euler steps for flow sampling
        num_negatives=7,        # DV N: deranged product-of-marginals negatives
        pretrain_steps=100000,   # bc/dyn-only steps before the critic trains
        est_num_joints=16,       # K positive joints per state in empowerment()
        # main.py / GCDataset wiring: value_goals = geometric discounted future.
        discount=0.99,
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='GCDataset',
        value_p_curgoal=0.0,
        value_p_trajgoal=1.0,
        value_p_randomgoal=0.0,
        value_geom_sample=True,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=False,
        p_aug=0.0,
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))
