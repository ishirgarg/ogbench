"""
Offline Empowerment CRL agent for OGBench.

Estimates the conditional mutual information

    E(s) = I(S+; A | s) = E_{(s,a,s+) ~ D}[ log p(s+|s,a) / p(s+|s) ]

via two Contrastive RL critics that share their s+ encoder ψ.

Critics
───────
Both critics use in-batch InfoNCE / sigmoid-binary-cross-entropy with the
same parameterisation as `agents/crl.py`:

  Action critic:    f1(s, a, s+) = φ(s, a)·ψ(s+) / √d
  No-action critic: f2(s,    s+) = φ'(s) ·ψ(s+) / √d

With (s, a, s+) drawn from the data and value_goals=traj_goal (s+ is a future
state in the same trajectory), the InfoNCE optimum gives

  f1(s,a,s+) ≈ log p(s+|s,a) − log p(s+)   (up to a constant)
  f2(s,  s+) ≈ log p(s+|s)   − log p(s+)   (up to the same constant)

so that the difference is exactly the conditional pointwise MI:

  f1 − f2 ≈ log p(s+|s,a) − log p(s+|s) = (φ(s,a) − φ'(s))·ψ(s+) / √d.

ψ is shared between the two critics; φ and φ' are not.

Distillation
────────────
After `crl_pretrain_steps` optimiser steps the CRL critic is frozen and a
small MLP E_θ(s) is regressed by MSE onto the per-sample target

  T(s, a, s+) = (φ(s, a) − φ'(s)) · ψ(s+) / √d

with sg on (φ, φ', ψ). Because the conditional expectation minimises MSE,

  E_θ(s) → E_{(a, s+) ~ p(a, s+ | s)} [T(s, a, s+)] = I(S+; A | s).

Triples (s, a, s+) come from the offline dataset with s+ = value_goals
(traj_goal).

Phase schedule (DADS-style):
  step <  crl_pretrain_steps : only contrastive_loss is active
  step >= crl_pretrain_steps : only distill_loss is active; CRL params frozen
                                by snapshot+revert, defending against any
                                Adam momentum carry-over.

This agent is a state-empowerment estimator — it has no actor.
`sample_actions` returns uniform random actions for API compatibility with
the rest of OGBench (eval scripts, plotters).
"""

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


# ── Network modules ──────────────────────────────────────────────────────────


class EmpowermentCRLCritic(nn.Module):
    """Two CRL critics that share their ψ(s+) encoder.

    Encoders:
      φ(s, a) — action-conditioned state encoder
      φ'(s)  — state-only encoder
      ψ(s+)  — future-state encoder (shared by both critics)

    Calling the module returns the three embeddings; the agent forms the
    bilinear logit matrices and contrastive losses externally so the
    in-batch negative structure exactly matches `agents/crl.py`.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        self.phi = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.phi_prime = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.psi = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def _encode_state(self, observations):
        if self.state_encoder is not None:
            return self.state_encoder(observations)
        return observations

    def _encode_goal(self, goals):
        if self.goal_encoder is not None:
            return self.goal_encoder(goals)
        return goals

    def phi_apply(self, observations, actions):
        s = self._encode_state(observations)
        return self.phi(jnp.concatenate([s, actions], axis=-1))

    def phi_prime_apply(self, observations):
        s = self._encode_state(observations)
        return self.phi_prime(s)

    def psi_apply(self, future_states):
        g = self._encode_goal(future_states)
        return self.psi(g)

    def __call__(self, observations, actions, future_states):
        return (
            self.phi_apply(observations, actions),
            self.phi_prime_apply(observations),
            self.psi_apply(future_states),
        )


class EmpowermentNet(nn.Module):
    """Scalar empowerment estimator E_θ(s)."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    state_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        out = MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(observations)
        return out.squeeze(-1)


# ── Agent ────────────────────────────────────────────────────────────────────


class EmpowermentCRLAgent(flax.struct.PyTreeNode):
    """Offline empowerment agent built on dual CRL critics + distillation."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── Loss components ─────────────────────────────────────────────────────

    def contrastive_loss(self, batch, grad_params):
        """Two InfoNCE/sigmoid critics with shared ψ; in-batch negatives.

        Same numerical recipe as `agents/crl.py`: sigmoid binary cross-entropy
        with a [B, B] logit matrix and identity labels — random batch elements
        approximate the marginal p(s+).
        """
        batch_size = batch['observations'].shape[0]

        phi, phi_prime, psi = self.network.select('critic')(
            batch['observations'],
            batch['actions'],
            batch['value_goals'],
            params=grad_params,
        )
        d = phi.shape[-1]
        scale = jnp.sqrt(d)

        # logits[i, j] = encoder(s_i, ·) · ψ(s+_j) / √d
        logits_action = jnp.einsum('ik,jk->ij', phi, psi) / scale
        logits_state = jnp.einsum('ik,jk->ij', phi_prime, psi) / scale

        I = jnp.eye(batch_size)
        loss_action = optax.sigmoid_binary_cross_entropy(
            logits=logits_action, labels=I
        ).mean()
        loss_state = optax.sigmoid_binary_cross_entropy(
            logits=logits_state, labels=I
        ).mean()
        loss = loss_action + loss_state

        def diag_stats(logits):
            pos = jnp.sum(logits * I) / jnp.sum(I)
            neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
            cat_acc = jnp.mean(
                jnp.argmax(logits, axis=1) == jnp.arange(batch_size)
            )
            bin_acc = jnp.mean((logits > 0) == I)
            return pos, neg, cat_acc, bin_acc

        a_pos, a_neg, a_cat, a_bin = diag_stats(logits_action)
        s_pos, s_neg, s_cat, s_bin = diag_stats(logits_state)

        return loss, {
            'contrastive_loss': loss,
            'loss_action': loss_action,
            'loss_state': loss_state,
            'logits_action_pos': a_pos,
            'logits_action_neg': a_neg,
            'cat_acc_action': a_cat,
            'bin_acc_action': a_bin,
            'logits_state_pos': s_pos,
            'logits_state_neg': s_neg,
            'cat_acc_state': s_cat,
            'bin_acc_state': s_bin,
        }

    def distill_loss(self, batch, grad_params):
        """MSE distillation of E_θ(s) onto the per-sample CMI integrand.

        Target is computed from the (frozen) CRL critic; gradient flows only
        into the empowerment network via `grad_params`.
        """
        # CRL critic forward pass — note the absence of `params=`, which uses
        # the live params; we then stop_gradient to prevent any flow back into
        # φ/φ'/ψ when this loss is non-zero (i.e., during distillation).
        phi, phi_prime, psi = self.network.select('critic')(
            batch['observations'],
            batch['actions'],
            batch['value_goals'],
        )
        phi = jax.lax.stop_gradient(phi)
        phi_prime = jax.lax.stop_gradient(phi_prime)
        psi = jax.lax.stop_gradient(psi)

        d = phi.shape[-1]
        target = jnp.sum((phi - phi_prime) * psi, axis=-1) / jnp.sqrt(d)

        pred = self.network.select('empowerment_net')(
            batch['observations'], params=grad_params,
        )
        loss = jnp.mean((pred - target) ** 2)
        return loss, {
            'distill_loss': loss,
            'target_mean': target.mean(),
            'target_min': target.min(),
            'target_max': target.max(),
            'pred_mean': pred.mean(),
            'pred_min': pred.min(),
            'pred_max': pred.max(),
        }

    # ── Empowerment as a state function ─────────────────────────────────────

    def empowerment(self, observations):
        """E_θ(s). Untrained until distillation begins."""
        return self.network.select('empowerment_net')(observations)

    # ── Combined loss ───────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}

        contrastive_loss, c_info = self.contrastive_loss(batch, grad_params)
        info.update({f'crl/{k}': v for k, v in c_info.items()})

        distill_loss, d_info = self.distill_loss(batch, grad_params)
        info.update({f'distill/{k}': v for k, v in d_info.items()})

        # Two-phase schedule. During pretraining, only the contrastive loss
        # contributes; during distillation, only the distill loss contributes.
        # The CRL params are additionally hard-frozen via snapshot+revert in
        # `update`, defending against Adam momentum carry-over.
        in_distill = self.network.step >= self.config['crl_pretrain_steps']
        in_distill_f = in_distill.astype(jnp.float32)
        crl_active = 1.0 - in_distill_f
        info['phase/in_distill'] = in_distill_f

        emp = jax.lax.stop_gradient(self.empowerment(batch['observations']))
        info['empowerment/mean'] = emp.mean()
        info['empowerment/min'] = emp.min()
        info['empowerment/max'] = emp.max()

        total = crl_active * contrastive_loss + in_distill_f * distill_loss
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        # Snapshot CRL params BEFORE the optimiser step. After distillation
        # begins we revert them, robustly freezing the critic even though
        # the loss multiplier alone would already zero out its gradient
        # (Adam's accumulated moments could otherwise still nudge it).
        old_critic = self.network.params['modules_critic']
        in_distill = self.network.step >= self.config['crl_pretrain_steps']

        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )

        frozen_critic = jax.tree_util.tree_map(
            lambda old, new: jnp.where(in_distill, old, new),
            old_critic,
            new_network.params['modules_critic'],
        )
        new_params = {**new_network.params, 'modules_critic': frozen_critic}
        new_network = new_network.replace(params=new_params)
        return self.replace(network=new_network, rng=new_rng), info

    # ── Evaluation stub (no actor) ──────────────────────────────────────────

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Return uniform random actions in [-1, 1]. This agent has no policy."""
        if seed is None:
            seed = self.rng

        single_obs = observations.ndim == 1
        if single_obs:
            observations = observations[None, :]

        batch_shape = observations.shape[:-1]
        action_dim = self.config['action_dim']
        actions = jax.random.uniform(
            seed, batch_shape + (action_dim,), minval=-1.0, maxval=1.0,
        )
        if single_obs:
            actions = actions[0]
        return actions

    # ── Constructor ─────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        config = dict(config)
        config['action_dim'] = action_dim

        # Encoders (image obs). The contrastive critic needs separate state
        # and goal encoders (s+ is encoded by ψ); the empowerment network only
        # sees the current state.
        encoders = {}
        if config.get('encoder') is not None:
            enc = encoder_modules[config['encoder']]
            encoders['critic_state'] = GCEncoder(concat_encoder=enc())
            encoders['critic_goal'] = GCEncoder(concat_encoder=enc())
            encoders['empowerment'] = GCEncoder(concat_encoder=enc())

        critic_def = EmpowermentCRLCritic(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            state_encoder=encoders.get('critic_state'),
            goal_encoder=encoders.get('critic_goal'),
        )
        empowerment_def = EmpowermentNet(
            hidden_dims=config['empowerment_hidden_dims'],
            layer_norm=config['layer_norm'],
            state_encoder=encoders.get('empowerment'),
        )

        ex_goals = ex_observations
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions, ex_goals)),
            empowerment_net=(empowerment_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ───────────────────────────────────────────────────────────────────


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='empowerment_crl',
        # NOTE: shared constants matched to agents/empowerment_skill.py for a
        # like-for-like empowerment comparison (previous defaults per line).
        # ── Optimisation ─────────────────────────────────────────────────────
        lr=1e-3,
        batch_size=512,                       # Matched (was 512).
        # ── CRL critic (φ, φ', ψ) ───────────────────────────────────────────
        value_hidden_dims=(256, 256),     # Matched (was 4x512).
        latent_dim=256,                        # Matched to value_latent_dim (was 512).
        layer_norm=True,
        # ── Empowerment distillation network E_θ(s) ─────────────────────────
        empowerment_hidden_dims=(256, 256),  # Matched (was 4x512).
        # ── Two-phase schedule ──────────────────────────────────────────────
        crl_pretrain_steps=500_000,
        # ── Compatibility with main.py / GCDataset ──────────────────────────
        discount=0.99,
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='GCDataset',
        # value_goals = traj_goal 100% (positives = future state in trajectory)
        value_p_curgoal=0.0,
        value_p_trajgoal=1.0,
        value_p_randomgoal=0.0,
        value_geom_sample=True,
        # actor_goals = traj_goal 100% (kept consistent with the spec, even
        # though this agent has no actor; main.py / GCDataset still wire them)
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=False,
        p_aug=0.0,
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))
