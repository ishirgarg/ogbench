"""
Offline empowerment agent
"""

from typing import Any, Optional, Sequence
import copy

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
from utils.networks import GCActor, GCDiscreteActor, MLP, TransformedWithMode


# ── Numerics helpers ──────────────────────────────────────────────────────────


def log1mexp(x):
    """Stable log(1 − exp(x)) for x ≤ 0.

    Branch on x vs log(½) to avoid catastrophic cancellation:
      x < log½  →  log1p(−exp(x))     exp(x) is small; log1p is safe
      x ≥ log½  →  log(−expm1(x))     expm1 is accurate near 0
    """
    log_half = -0.6931471805599453
    return jnp.where(
        x < log_half,
        jnp.log1p(-jnp.exp(x)),
        jnp.log(-jnp.expm1(x)),
    )


def log_diff_exp(log_total, log_part):
    """Stable log(exp(log_total) − exp(log_part)), requires log_part ≤ log_total.

    Identity:  log(A − B) = log_total + log1mexp(log_part − log_total)
    where  log_part − log_total ≤ 0  keeps log1mexp in its valid domain.
    """
    return log_total + log1mexp(log_part - log_total)

def clipped_linexp_loss(target, pred, gamma, t=5.0):
    """Clipped linexp loss in log space; regresses e^-pred to gamma * e^-target."""
    target, t = jax.lax.stop_gradient(target), jax.lax.stop_gradient(t)

    p0 = target + t
    value_p0 = gamma * jnp.exp(t) - (target + t)
    slope_p0 = gamma * jnp.exp(t) - 1.0

    true_loss = gamma * jnp.exp(pred - target) - pred
    linear_loss = value_p0 + slope_p0 * (pred - p0)

    loss = jnp.where(
        pred - target < t,
        true_loss,
        linear_loss,
    )

    return loss.mean()



# ── Network modules ───────────────────────────────────────────────────────────


class EmpowermentQNetwork(nn.Module):
    """Q^z(s⁺ | s, a): φ(s, a, z) · ψ(s⁺) bilinear structure."""

    hidden_dims: Sequence[int]
    action_dim: int
    num_skills: int
    latent_dim: int = 128
    layer_norm: bool = True
    discrete: bool = False
    gc_encoder: Optional[nn.Module] = None
    shared_psi: Optional[nn.Module] = None

    def setup(self):
        self.phi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        if self.shared_psi is None:
            self.psi_net = MLP(
                (*self.hidden_dims, self.latent_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )

    def phi(self, observations, actions, skills):
        obs = self.gc_encoder(observations, None) if self.gc_encoder else observations
        acts = jnp.eye(self.action_dim)[actions] if self.discrete else actions
        return self.phi_net(jnp.concatenate([obs, acts, skills], axis=-1))

    def psi(self, future_states):
        if self.shared_psi is not None:
            return self.shared_psi(future_states)
        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states
        return self.psi_net(future)

    def __call__(self, observations, actions, skills, future_states):
        phi_emb = self.phi(observations, actions, skills)
        psi_emb = self.psi(future_states)
        diff = phi_emb - psi_emb
        l2_sq = jnp.sum(diff ** 2, axis=-1)
        # log p(ψ | φ) for ψ ~ N(φ, (d/2)·I):  −½d·log(πd) − ‖φ−ψ‖²/d
        return -l2_sq / self.latent_dim


class EmpowermentVNetwork(nn.Module):
    """V^z(s⁺ | s): φ(s, z) · ψ(s⁺) — independent of actions.

    Only instantiated when separate_qv=True; otherwise V is derived from Q.
    The `action_dim` field is accepted for API symmetry but is unused.
    """

    hidden_dims: Sequence[int]
    action_dim: int          # unused; kept for symmetric construction
    num_skills: int
    latent_dim: int = 128
    layer_norm: bool = True
    discrete: bool = False   # unused; kept for symmetric construction
    gc_encoder: Optional[nn.Module] = None
    shared_psi: Optional[nn.Module] = None

    def setup(self):
        self.phi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        if self.shared_psi is None:
            self.psi_net = MLP(
                (*self.hidden_dims, self.latent_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )

    def phi(self, observations, skills):
        obs = self.gc_encoder(observations, None) if self.gc_encoder else observations
        return self.phi_net(jnp.concatenate([obs, skills], axis=-1))

    def psi(self, future_states):
        if self.shared_psi is not None:
            return self.shared_psi(future_states)
        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states
        return self.psi_net(future)

    def __call__(self, observations, skills, future_states):
        phi_emb = self.phi(observations, skills)
        psi_emb = self.psi(future_states)
        diff = phi_emb - psi_emb
        l2_sq = jnp.sum(diff ** 2, axis=-1)
        return -l2_sq / self.latent_dim


class SharedPsiEncoder(nn.Module):
    """Shared ψ(s⁺): optional GC encoder on s⁺ + one ψ MLP."""

    hidden_dims: Sequence[int]
    latent_dim: int = 128
    layer_norm: bool = True
    gc_encoder: Optional[nn.Module] = None

    def setup(self):
        self.psi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, future_states):
        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states
        return self.psi_net(future)


class SkillConditionedActor(GCActor):
    """Continuous actor conditioned on a skill one-hot rather than a goal."""

    def __call__(self, observations, skills, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = jnp.concatenate([self.gc_encoder(observations, None), skills], axis=-1)
        else:
            inputs = jnp.concatenate([observations, skills], axis=-1)
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        elif self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_stds
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            dist = TransformedWithMode(dist, distrax.Block(distrax.Tanh(), ndims=1))
        return dist


class SkillConditionedDiscreteActor(GCDiscreteActor):
    """Discrete actor conditioned on a skill one-hot rather than a goal."""

    def __call__(self, observations, skills, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = jnp.concatenate([self.gc_encoder(observations, None), skills], axis=-1)
        else:
            inputs = jnp.concatenate([observations, skills], axis=-1)
        logits = self.logit_net(self.actor_net(inputs))
        return distrax.Categorical(logits=logits / temperature)


# ── Agent ─────────────────────────────────────────────────────────────────────


class EmpowermentAgent(flax.struct.PyTreeNode):
    """Offline empowerment agent (Myers 2025).

    Learns a value network modelling the discounted future-state occupancy
    measure and a skill-conditioned policy π(a | s, z) that maximises
    empowerment.

    Network layout
    ──────────────
    separate_qv=False (default)
        ModuleDict keys:  q, target_q, policy
        V is derived from Q by evaluating Q at the policy's action:
            V^z(s⁺ | s) ≡ Q^z(s⁺ | s, π(s, z))

    separate_qv=True
        ModuleDict keys:  q, v, target_q, target_v, policy
        Q and V are independent networks with separate parameters.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── Basic helpers ─────────────────────────────────────────────────────────

    def _extract_future(self, states):
        """Slice states to the goal-relevant subspace fed into ψ(s⁺)."""
        obs_indices = self.config.get('obs_indices', None)
        if obs_indices is not None:
            return states[..., jnp.array(obs_indices)]
        return states

    def _sample_skills(self, rng, batch_size):
        """Sample a batch of skill indices and their one-hot encodings."""
        K = self.config['num_skills']
        skills = jax.random.randint(rng, (batch_size,), 0, K)
        return skills, jnp.eye(K)[skills]

    def _policy_actions(self, observations, skills_onehot, params, rng=None,
                        num_noise_samples=None):
        """Policy actions (mode / argmax), optionally through a noisy actuation channel.

        With `stochastic_policy_actions` the emitted action is perturbed by a
        fixed-scale Gaussian, a = π(s,z) + η, η ~ N(0, σ_a² I), and clipped back
        to the valid range.  σ_a is a constant (`action_noise_std`), *not* the
        actor's learned log_std — a learnable scale could be driven to zero,
        which would collapse the channel back to the deterministic case.

        The noise is what makes the skill occupancies genuinely overlapping
        measures rather than atoms; without it the empowerment objective is
        invariant to how far apart the skills actually travel.  Requires an rng;
        callers that pass rng=None stay deterministic.

        `num_noise_samples=M` draws M independent η per state around the *same*
        deterministic mode (one actor forward pass), returning [M, batch, A]
        instead of [batch, A].  Only meaningful with the channel on.
        """
        dist = self.network.select('policy')(observations, skills_onehot, params=params)
        if self.config['discrete']:
            # A Gaussian actuation channel is not meaningful on action indices.
            return dist.probs.argmax(axis=-1)
        actions = dist.mode()
        if self.config.get('stochastic_policy_actions', False) and rng is not None:
            shape = (actions.shape if num_noise_samples is None
                     else (num_noise_samples, *actions.shape))
            noise = jax.random.normal(rng, shape) * self.config['action_noise_std']
            actions = jnp.clip(actions + noise, -1.0, 1.0)
        return actions

    def _logits_from_embeddings(self, phi_emb, psi_emb, latent_dim):
        """log p(ψ | φ) for ψ ~ N(φ, (d/2)·I).

        Args:
            phi_emb:    [..., d]
            psi_emb:    [..., d]
            latent_dim: int  (d)
        Returns:
            log_q: [...]
        """
        diff = phi_emb - psi_emb
        l2_sq = jnp.sum(diff ** 2, axis=-1)
        return -l2_sq / latent_dim

    def _v_phi(self, observations, skills_onehot, *, use_target: bool,
                policy_params=None, rng=None):
        """φ_V embedding for a single skill batch  →  [batch, d].

        Args:
            observations:  [batch, obs_dim]
            skills_onehot: [batch, K]
            use_target:    use the target network
            policy_params: params for the policy network;
                           None = frozen policy (no grad);
                           grad_params = gradient flows through policy.
            rng:           key for the noisy actuation channel (None = off).
        """
        if self.config['separate_qv']:
            key = 'target_v' if use_target else 'v'
            net = self.network.model_def.modules[key]
            vars_ = jax.lax.stop_gradient(
                {'params': self.network.params[f'modules_{key}']}
            )
            return net.apply(vars_, observations, skills_onehot, method=net.phi)
        else:
            key = 'target_q' if use_target else 'q'
            net = self.network.model_def.modules[key]
            vars_ = jax.lax.stop_gradient(
                {'params': self.network.params[f'modules_{key}']}
            )
            actions = self._policy_actions(observations, skills_onehot,
                                           params=policy_params, rng=rng)
            return net.apply(vars_, observations, actions, skills_onehot,
                             method=net.phi)

    def _v_phi_all_skills(self, observations, *, use_target: bool,
                           policy_params=None, rng=None):
        """φ_V embeddings for *all* K skills  →  [K, batch, d].

        Args: same as _v_phi.  Internally vmaps over the skill index.  Each
        skill gets its own actuation-noise key so the perturbations are
        independent across skills rather than shared.
        """
        K = self.config['num_skills']
        batch_size = observations.shape[0]
        skills_onehot = jnp.eye(K)
        # vmap needs a concrete array of keys even when the channel is off.
        act_rngs = jax.random.split(
            rng if rng is not None else jax.random.PRNGKey(0), K
        )

        if self.config['separate_qv']:
            key = 'target_v' if use_target else 'v'
            net = self.network.model_def.modules[key]
            vars_ = jax.lax.stop_gradient(
                {'params': self.network.params[f'modules_{key}']}
            )

            def phi_for_skill(z_onehot, act_rng):
                # V is action-free in separate_qv mode; act_rng is unused here.
                del act_rng
                z_batch = jnp.repeat(z_onehot[None, :], batch_size, axis=0)
                return net.apply(vars_, observations, z_batch, method=net.phi)
        else:
            key = 'target_q' if use_target else 'q'
            net = self.network.model_def.modules[key]
            vars_ = jax.lax.stop_gradient(
                {'params': self.network.params[f'modules_{key}']}
            )

            def phi_for_skill(z_onehot, act_rng):
                z_batch = jnp.repeat(z_onehot[None, :], batch_size, axis=0)
                actions = self._policy_actions(observations, z_batch,
                                               params=policy_params,
                                               rng=act_rng if rng is not None else None)
                return net.apply(vars_, observations, actions, z_batch,
                                 method=net.phi)

        return jax.vmap(phi_for_skill)(skills_onehot, act_rngs)   # [K, batch, d]

    def _q_phi(self, observations, actions, skills_onehot, *, use_target: bool):
        """φ_Q embedding  →  [batch, d].

        Network params are stop_grad'd; gradient enters through `actions`.
        """
        key = 'target_q' if use_target else 'q'
        net = self.network.model_def.modules[key]
        vars_ = jax.lax.stop_gradient(
            {'params': self.network.params[f'modules_{key}']}
        )
        return net.apply(vars_, observations, actions, skills_onehot,
                         method=net.phi)

    # ── Core value computations ───────────────────────────────────────────────

    def compute_q_logits(self, observations, actions, skills_onehot,
                          future_states, params=None):
        """log Q^z(s⁺ | s, a) — online Q network, params differentiable."""
        future_extracted = self._extract_future(future_states)
        return self.network.select('q')(
            observations, actions, skills_onehot, future_extracted, params=params
        )

    def compute_q_logits_target(self, observations, actions, skills_onehot,
                                 future_states):
        """log Q^z(s⁺ | s, a) — target Q network (frozen)."""
        future_extracted = self._extract_future(future_states)
        return self.network.select('target_q')(
            observations, actions, skills_onehot, future_extracted, params=None
        )

    def compute_v_logits(self, observations, skills_onehot, future_states,
                          params=None, policy_params=None, rng=None):
        """log V^z(s⁺ | s) — online network, params differentiable.

        Combined mode: evaluates Q(s, π(s,z), z, s⁺) with the given `params`
            applied to the Q network; policy is frozen (policy_params=None in
            v_loss because we are not optimising the policy here).
        Separate mode: evaluates the V network directly with `params`.
        """
        future_extracted = self._extract_future(future_states)
        if self.config['separate_qv']:
            return self.network.select('v')(
                observations, skills_onehot, future_extracted, params=params
            )
        else:
            actions = self._policy_actions(observations, skills_onehot,
                                           params=policy_params, rng=rng)
            return self.network.select('q')(
                observations, actions, skills_onehot, future_extracted,
                params=params
            )

    def compute_v_logits_target(self, observations, skills_onehot, future_states,
                                 policy_params=None, rng=None):
        """log V^z(s⁺ | s) — target network (frozen).

        Combined mode: evaluates target_Q(s, π(s,z), z, s⁺).
        Separate mode: evaluates target_V(s, z, s⁺).
        """
        future_extracted = self._extract_future(future_states)
        if self.config['separate_qv']:
            return self.network.select('target_v')(
                observations, skills_onehot, future_extracted, params=None
            )
        else:
            actions = self._policy_actions(observations, skills_onehot,
                                           params=policy_params, rng=rng)
            return self.network.select('target_q')(
                observations, actions, skills_onehot, future_extracted,
                params=None
            )

    def empowerment(self, observations, rng):
        """Monte-Carlo estimate of I(Z; S⁺ | s) for each observation."""
        batch_size = observations.shape[0]
        K = self.config['num_skills']
        num_samples = self.config['num_splus_samples']
        d = self.config['value_latent_dim']
        log_K = jnp.log(K)

        act_rng = jax.random.fold_in(rng, 5)
        rng, sample_rng = jax.random.split(rng)
        skill_rngs = jax.random.split(sample_rng, K)

        # φ_V for all skills: [K, batch, d]  (policy frozen, no grad needed)
        phi_all = self._v_phi_all_skills(
            observations, use_target=False, policy_params=None, rng=act_rng
        )

        def empowerment_for_skill(phi_z, skill_rng):
            noise = jax.random.normal(skill_rng, (num_samples, *phi_z.shape))
            psi_samples = phi_z[None] + noise * jnp.sqrt(d / 2.0)

            def contribution(psi_splus):
                # log V^z(psi | s):         [batch]
                # log V^{z'}(psi | s) ∀z':  [K, batch]
                log_v = self._logits_from_embeddings(phi_z, psi_splus, d)
                log_v_all = self._logits_from_embeddings(phi_all, psi_splus, d)
                log_denom = logsumexp(log_v_all, axis=0) - log_K
                return log_v - log_denom

            contributions = jax.vmap(contribution)(psi_samples)  # [N, batch]
            return contributions.mean(axis=0)

        emp_per_skill = jax.vmap(
            empowerment_for_skill, in_axes=(0, 0)
        )(phi_all, skill_rngs)          # [K, batch]
        return emp_per_skill.mean(axis=0)

    # ── Losses ────────────────────────────────────────────────────────────────
    #
    # No branching on separate_qv inside any loss.  All mode-specific
    # dispatching is handled by the V modulation interface above.

    def q_loss(self, batch, grad_params, skills_onehot, rng=None):
        """Q loss.

        separate_qv=True:
            Two terms:
              1) Q(s⁺ | s, a, z) = γ · V(s⁺ | s', z)
              2) (optional; enabled when use_self_q_loss)
                 Q(s | s, a, z) = (1-γ) + γ · V(s | s', z)
        separate_qv=False (shared Q/V):
            Apply two Bellman-style Q losses:
              1) Q(s⁺ | s, a, z) = γ · Q(s⁺ | s', π(s',z), z)
              2) (optional; enabled when use_self_q_loss)
                 Q(s' | s, a, z) = (1-γ) + γ · Q(s' | s', π(s',z), z)
        """
        future = batch['value_goals']

        actions = batch['actions']
        if (self.config.get('stochastic_policy_actions', False)
                and self.config.get('perturb_q_loss_actions', True)
                and not self.config['discrete'] and rng is not None):
            data_noise = (
                jax.random.normal(jax.random.fold_in(rng, 7), actions.shape)
                * self.config['action_noise_std']
            )
            actions = jnp.clip(actions + data_noise, -1.0, 1.0)

        log_q = self.compute_q_logits(
            batch['observations'], actions,
            skills_onehot, future, params=grad_params
        )

        if self.config['separate_qv']:
            # 1) Future-state loss: Q(s+ | s,a,z) = discount * V(s+ | s', z)
            log_v_next_future = self.compute_v_logits_target(
                batch['next_observations'], skills_onehot, future
            )
            loss_future = clipped_linexp_loss(
                target=-log_v_next_future,
                pred=-log_q,
                gamma=self.config['discount'],
            )

            loss = loss_future
            metrics = {
                'q_loss': loss,
                'q_loss_future': loss_future,
                'q_log_mean': log_q.mean(),
                'v_next_future_log_mean': log_v_next_future.mean(),
            }

            # 2) Optional self loss: Q(s | s,a,z) = (1-γ) + γ · V(s | s', z)
            if self.config.get('use_self_q_loss', True):
                log_q_current = self.compute_q_logits(
                    batch['observations'], actions,
                    skills_onehot, batch['observations'], params=grad_params
                )
                log_v_next_current = self.compute_v_logits_target(
                    batch['next_observations'], skills_onehot, batch['observations']
                )
                log_current_target = jnp.logaddexp(
                    jnp.log(1.0 - self.config['discount']),
                    jnp.log(self.config['discount']) + log_v_next_current,
                )
                loss_current = clipped_linexp_loss(
                    target=-log_current_target,
                    pred=-log_q_current,
                    gamma=1.0,
                )
                loss = loss + loss_current
                metrics.update({
                    'q_loss': loss,
                    'q_loss_current': loss_current,
                    'q_log_current_mean': log_q_current.mean(),
                    'v_next_current_log_mean': log_v_next_current.mean(),
                })

            return loss, metrics

        # Shared Q/V mode: no V loss; Q carries both Bellman terms.
        # One action sample shared by both Bellman terms — they are the same
        # V(·|s') evaluated at two different s⁺.
        actions_next = self._policy_actions(
            batch['next_observations'], skills_onehot, params=None, rng=rng
        )
        # 1) Q(s+ | s,a) = gamma * Q(s+ | s', pi(s',z), z)
        log_q_next_future = self.compute_q_logits_target(
            batch['next_observations'], actions_next, skills_onehot, future
        )
        loss_future = clipped_linexp_loss(
            target=-log_q_next_future,
            pred=-log_q,
            gamma=self.config['discount'],
        )

        loss = loss_future
        metrics = {
            'q_loss': loss,
            'q_loss_future': loss_future,
            'q_log_mean': log_q.mean(),
            'q_log_next_future_mean': log_q_next_future.mean(),
        }

        # 2) Optional self loss: Q(s' | s,a) = (1-γ) + γ · Q(s' | s', π(s',z), z)
        if self.config.get('use_self_q_loss', True):
            log_q_current = self.compute_q_logits(
                batch['observations'], actions,
                skills_onehot, batch['next_observations'], params=grad_params
            )
            log_q_next_current = self.compute_q_logits_target(
                batch['next_observations'], actions_next,
                skills_onehot, batch['next_observations']
            )
            log_current_target = jnp.logaddexp(
                jnp.log(1.0 - self.config['discount']),
                jnp.log(self.config['discount']) + log_q_next_current,
            )
            loss_current = clipped_linexp_loss(
                target=-log_current_target,
                pred=-log_q_current,
                gamma=1.0,
            )
            loss = loss + loss_current
            metrics.update({
                'q_loss': loss,
                'q_loss_current': loss_current,
                'q_log_current_mean': log_q_current.mean(),
                'q_log_next_current_mean': log_q_next_current.mean(),
            })

        return loss, metrics

    def v_loss(self, batch, grad_params, skills_onehot, rng=None):
        """L_V — Bellman backup for the occupancy V (eqs. 16-17).


        In combined mode `grad_params` differentiates through the Q network
        (since V≡Q(π)); the policy is frozen (policy_params=None).
        In separate mode `grad_params` differentiates through the V network.
        """
        if not self.config['separate_qv']:
            # In shared Q/V mode, Q loss already includes both Bellman terms.
            zero = jnp.array(0.0, dtype=batch['observations'].dtype)
            return zero, {
                'v_loss': zero,
                'v_loss_future': zero,
                'v_loss_current': zero,
            }

        future = batch['value_goals']

        # Regress V(s+ | s, z) onto Q(s+ | s, π(s,z), z) (no discount, no self V loss).
        # This is where V becomes the occupancy of the *noisy* policy: with the
        # actuation channel on, the single-sample target makes V ≈ E_η[Q(s, π+η)].
        actions_pi = self._policy_actions(
            batch['observations'], skills_onehot, params=None, rng=rng
        )
        log_q_pi = self.compute_q_logits_target(
            batch['observations'], actions_pi, skills_onehot, future
        )
        log_v = self.compute_v_logits(
            batch['observations'], skills_onehot, future,
            params=grad_params, policy_params=None
        )
        loss_future = clipped_linexp_loss(
            target=-log_q_pi,
            pred=-log_v,
            gamma=1.0,
        )

        loss = loss_future
        metrics = {
            'v_loss': loss,
            'v_loss_future': loss_future,
            'v_log_mean': log_v.mean(),
            'q_pi_log_mean': log_q_pi.mean(),
            'v_max': jnp.exp(log_v).max(),
            'v_min': jnp.exp(log_v).min(),
        }

        return loss, metrics

    def bc_loss(self, batch, grad_params, skills_onehot):
        """Behavioral cloning loss: −log π(a_expert | s, z)."""
        dist = self.network.select('policy')(
            batch['observations'], skills_onehot, params=grad_params
        )
        log_prob = dist.log_prob(batch['actions'])
        loss = -log_prob.mean()
        return loss, {
            'bc_loss': loss,
            'bc_log_prob_mean': log_prob.mean(),
            'bc_log_prob_max': log_prob.max(),
            'bc_log_prob_min': log_prob.min(),
        }

    def policy_loss(self, batch, grad_params, skills, skills_onehot, rng=None):
        """Policy loss via empowerment gradient.

        Unified IS scheme.  For each sample slot m we have a skill z'_m and
        a ψ_m ~ V_{z'_m}(·|s); the per-sample reward is

            r_m = (M_{z'_m}/V_{z'_m}) · log(M_{z'_m}/m̄_m)

        with M_{z'} = Q^z(ψ|s,a) if z'=z else V_{z'}(ψ|s), and
        m̄ = (Q^z + Σ_{z''≠z} V_{z''}) / K.  The IS weight collapses to
        Q^z/V^z when z'=z and to 1 when z'≠z.

        sample_z=True:   M = N MC samples of (z' ~ Unif(K), ψ).
        sample_z=False:  M = N·K — each z' ∈ [K] used N times analytically
                         (one ψ ~ V_{z'} per slot); aggregation divides by K
                         so both modes scale as N · E_{z'}E_ψ[r].

        Gradient flow:  V's are stop_grad'd (frozen V net + explicit sg on
        log_v_z_prime and log_c_sg).  Policy grad flows only through log_q,
        which appears as itself and inside log_m_bar.
        """
        batch_size = batch['observations'].shape[0]
        K = self.config['num_skills']
        N = self.config['num_splus_samples']
        d = self.config['value_latent_dim']
        log_K = jnp.log(K)

        # Must use a key derived from total_loss's rng, NOT self.rng:
        # update() splits self.rng the same way, so re-splitting it here made
        # sample_rng collide with total_loss's rng — z' sampling reproduced the
        # skills draw exactly (z' == z in slot 0 every step).
        rng = rng if rng is not None else self.rng
        rng, sample_rng = jax.random.split(rng)
        act_rng_v = jax.random.fold_in(rng, 3)
        act_rng_q = jax.random.fold_in(rng, 4)

        # use_target=False routes the policy loss through the online Q/V
        # networks (no target bootstrap on the policy gradient).
        use_target = not self.config['no_target_q_for_policy']

        sample_z = self.config['sample_z']

        # phi_all: V at all K skills.  V is frozen w.r.t. the policy in both
        # modes — every V reference downstream is stop_grad'd anyway.
        phi_all = self._v_phi_all_skills(
            batch['observations'], use_target=use_target, policy_params=None,
            rng=act_rng_v,
        )  # [K, batch, d]

        # ── Build z' for each of M sample slots ─────────────────────────────
        if sample_z:
            rng_zp, sample_rng = jax.random.split(sample_rng)
            z_prime = jax.random.randint(
                rng_zp, (N, batch_size), 0, K,
            )  # [N, batch]
        else:
            # Enumerate z' ∈ [K] exactly once per (N, batch).
            z_prime = jnp.broadcast_to(
                jnp.arange(K)[None, :, None], (N, K, batch_size),
            ).reshape(N * K, batch_size)  # [N·K, batch]

        M = z_prime.shape[0]

        # ── Sample ψ ~ V_{z'}(·|s) ──────────────────────────────────────────
        batch_idx = jnp.arange(batch_size)[None, :]
        phi_z_prime_v = phi_all[z_prime, batch_idx]  # [M, batch, d]
        psi = (
            phi_z_prime_v
            + jax.random.normal(sample_rng, (M, batch_size, d))
            * jnp.sqrt(d / 2.0)
        )  # [M, batch, d]

        # ── log Q^z(ψ | s, π(s,z)) ──────────────────────────────────────────
        # Reparameterized: the noise is an additive constant w.r.t. grad_params,
        # so the policy gradient still flows through the action mean.  Each of
        # the M sample slots gets its own independent η draw so the MC average
        # over slots also averages over the actuation noise, rather than
        # conditioning every slot on one shared η.  The mode is computed once;
        # only the noise carries the extra M axis.
        if self.config.get('stochastic_policy_actions', False):
            policy_actions = self._policy_actions(
                batch['observations'], skills_onehot, params=grad_params,
                rng=act_rng_q, num_noise_samples=M,
            )  # [M, batch, action_dim]
            # vmap over the M axis: obs/skills stay [batch, ...] and φ_Q
            # concatenates on the last axis, so map rather than broadcast.
            phi_z_q = jax.vmap(
                lambda a: self._q_phi(
                    batch['observations'], a, skills_onehot,
                    use_target=use_target,
                )
            )(policy_actions)  # [M, batch, d]
        else:
            # Channel off: all slots share the deterministic action; evaluate
            # the policy and φ_Q once.
            policy_actions = self._policy_actions(
                batch['observations'], skills_onehot, params=grad_params,
            )
            phi_z_q = self._q_phi(
                batch['observations'], policy_actions, skills_onehot,
                use_target=use_target,
            )[None]  # [1, batch, d] — broadcasts over M
        log_q = self._logits_from_embeddings(phi_z_q, psi, d)  # [M, batch]

        # ── log V_{z''}(ψ) for all z''; pick out V^z (assigned skill) ───────
        log_v_all = self._logits_from_embeddings(phi_all[None], psi[:, None], d)  # [M, K, batch]
        log_v_z = log_v_all[:, skills, jnp.arange(batch_size)]                    # [M, batch]
        log_v_all_lse = logsumexp(log_v_all, axis=1)                              # [M, batch]
        log_c_sg = jax.lax.stop_gradient(
            log_diff_exp(log_v_all_lse, log_v_z)
        )  # log Σ_{z''≠z} V_{z''}:  [M, batch]

        # ── m̄ = (Q^z + Σ_{z''≠z} V_{z''}) / K ──────────────────────────────
        log_m_bar = (
            logsumexp(jnp.stack([log_q, log_c_sg], axis=0), axis=0) - log_K
        )  # [M, batch]

        # ── log V_{z'}(ψ): identical to log_v_all[m, z'_m, b], so just index
        # into the tensor we already computed.  Frozen V → stop_grad.
        log_v_z_prime_sg = jax.lax.stop_gradient(
            log_v_all[jnp.arange(M)[:, None], z_prime, jnp.arange(batch_size)[None, :]]
        )  # [M, batch]

        # log M_{z'}: log_q if z'=z (grad alive), else sg(log V_{z'}).
        z_eq_zp = (z_prime == skills[None, :])  # [M, batch]
        log_m_z_prime = jnp.where(z_eq_zp, log_q, log_v_z_prime_sg)

        # IS weight w = M_{z'}/V_{z'}:  Q^z/V^z when z'=z, else 1.
        w = jnp.exp(log_m_z_prime - log_v_z_prime_sg)            # [M, batch]

        reward = w * (log_m_z_prime - log_m_bar)                 # [M, batch]

        # ── Aggregate ───────────────────────────────────────────────────────
        # sample_z=True:   sum over N MC samples           → N · E_{z'}E_ψ[r]
        # sample_z=False:  sum over N·K, divide by K       → N · E_{z'}E_ψ[r]
        if sample_z:
            e_delta = reward.sum(axis=0)
        else:
            e_delta = (reward / K).sum(axis=0)

        loss = -e_delta.mean()

        return loss, {
            'policy_loss': loss,
            'e_delta_mean': e_delta.mean(),
            'e_delta_max': e_delta.max(),
            'e_delta_min': e_delta.min(),
        }

    # ── Training ──────────────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        skills_rng, empowerment_rng = jax.random.split(rng, 2)
        # fold_in rather than widening the split above, so the pre-existing
        # rng streams (and hence the default-config results) are unchanged.
        # Constants must be >= 2: split(rng, 2)[i] == fold_in(rng, i), so
        # small constants would collide with skills_rng/empowerment_rng.
        q_act_rng, v_act_rng = jax.random.fold_in(rng, 101), jax.random.fold_in(rng, 102)
        policy_rng = jax.random.fold_in(rng, 103)
        batch_size = batch['observations'].shape[0]
        skills, skills_onehot = self._sample_skills(skills_rng, batch_size)
        info = {}

        bc_loss, bc_info = self.bc_loss(batch, grad_params, skills_onehot)
        info.update({f'bc/{k}': v for k, v in bc_info.items()})

        # only_bc: skip the empowerment machinery entirely (Q/V/policy losses,
        # empowerment MC estimate) and train the skill-conditioned policy with
        # pure behavioral cloning.
        if self.config.get('only_bc', False):
            info['total_loss'] = bc_loss
            return bc_loss, info

        q_loss, q_info = self.q_loss(batch, grad_params, skills_onehot, rng=q_act_rng)
        info.update({f'q/{k}': v for k, v in q_info.items()})

        v_loss, v_info = self.v_loss(batch, grad_params, skills_onehot, rng=v_act_rng)
        info.update({f'v/{k}': v for k, v in v_info.items()})

        pi_loss, pi_info = self.policy_loss(batch, grad_params, skills, skills_onehot,
                                            rng=policy_rng)
        info.update({f'policy/{k}': v for k, v in pi_info.items()})

        # empowerment() is only consumed at log steps, so skip the MC estimate
        # otherwise. step is the iteration count during update (== 0 mod
        # log_interval at log iterations); validation total_loss runs *after*
        # update has incremented step, so it sits at == 1 mod log_interval.
        log_interval = self.config['log_interval']
        step_mod = self.network.step % log_interval
        should_compute_emp = (step_mod == 0) | (step_mod == 1)
        zero = jnp.zeros((), dtype=jnp.float32)

        def _compute_emp_metrics():
            emp = jax.lax.stop_gradient(
                self.empowerment(batch['observations'], rng=empowerment_rng)
            )
            return emp.mean(), emp.min(), emp.max()

        emp_mean, emp_min, emp_max = jax.lax.cond(
            should_compute_emp,
            _compute_emp_metrics,
            lambda: (zero, zero, zero),
        )
        info['empowerment/mean'] = emp_mean
        info['empowerment/min']  = emp_min
        info['empowerment/max']  = emp_max

        base_alpha = self.config.get('bc_alpha', 0.0)
        if self.config.get('anneal_alpha', False):
            # Exponential decay with half-life 100k optimizer steps.
            step = jnp.asarray(self.network.step, dtype=jnp.float32)
            alpha = base_alpha * jnp.power(0.5, step / 100000.0)
        else:
            alpha = jnp.asarray(base_alpha, dtype=jnp.float32)
        info['bc/alpha'] = alpha

        total = q_loss + v_loss + pi_loss + alpha * bc_loss
        info['total_loss'] = total
        
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )

        # Always soft-update target Q.
        new_target_q = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            new_network.params['modules_q'],
            new_network.params['modules_target_q'],
        )
        new_params = {**new_network.params, 'modules_target_q': new_target_q}

        # Soft-update target V only when it is a separate network.
        if self.config['separate_qv']:
            new_target_v = jax.tree_util.tree_map(
                lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
                new_network.params['modules_v'],
                new_network.params['modules_target_v'],
            )
            new_params['modules_target_v'] = new_target_v

        new_network = new_network.replace(params=new_params)
        return self.replace(network=new_network, rng=new_rng), info

    # ── Evaluation ────────────────────────────────────────────────────────────

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions. Goals are mapped deterministically to skills."""
        if seed is None:
            seed = self.rng

        # A single (unbatched) state-based obs is 1D; a single visual obs is 3D
        # (HWC). Treat both as the unbatched case and prepend a batch dim.
        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        if single_obs:
            observations = observations[None, ...]
            if goals is not None:
                goals = goals[None, ...]

        batch_size = observations.shape[0]

        if goals is not None:
            goal_future = self._extract_future(goals)
            goal_flat   = goal_future.reshape(batch_size, -1).astype(jnp.int32)
            goal_hash   = jnp.sum(goal_flat, axis=-1)
            skills = (jnp.abs(goal_hash) % self.config['num_skills']).astype(jnp.int32)
        else:
            skills = jax.random.randint(
                seed, (batch_size,), 0, self.config['num_skills']
            )

        skills_onehot = jnp.eye(self.config['num_skills'])[skills]
        dist = self.network.select('policy')(
            observations, skills_onehot, temperature=temperature
        )
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)

        if single_obs:
            actions = actions[0]
        return actions

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim   = ex_actions.max() + 1 if config['discrete'] else ex_actions.shape[-1]
        num_skills   = config.get('num_skills', 10)
        hidden_dims  = config.get('hidden_dims', (512, 512))
        separate_qv  = config.get('separate_qv', False)

        # Note that target and main networks need their own encoder each (even though its the same frozen encoder)
        encoders = {}
        target_encoders = {}
        if config.get('encoder') is not None:
            enc  = encoder_modules[config['encoder']]
            keys = ('q', 'v', 'policy') if separate_qv else ('q', 'policy')
            encoders = {k: GCEncoder(state_encoder=enc()) for k in keys}
            if separate_qv:
                target_encoders = {k: GCEncoder(state_encoder=enc()) for k in ('q', 'v')}

        # Shared kwargs for both Q and V network constructors.
        value_kwargs = dict(
            hidden_dims=config.get('value_hidden_dims', None) or hidden_dims,
            action_dim=action_dim,
            num_skills=num_skills,
            latent_dim=config.get('value_latent_dim', 128),
            layer_norm=config.get('layer_norm', True),
            discrete=config['discrete'],
        )

        actor_cls    = (SkillConditionedDiscreteActor if config['discrete']
                        else SkillConditionedActor)
        actor_kwargs = dict(
            hidden_dims=config.get('actor_hidden_dims', (512, 512, 512)),
            action_dim=action_dim,
            gc_encoder=encoders.get('policy'),
        )
        if not config['discrete']:
            actor_kwargs.update(
                state_dependent_std=False,
                const_std=config.get('const_std', True),
            )
        policy_def = actor_cls(**actor_kwargs)

        batch_size  = ex_observations.shape[0]
        ex_skills   = jnp.eye(num_skills)[jnp.arange(batch_size) % num_skills]
        obs_indices = config.get('obs_indices', None)
        ex_future   = (
            ex_observations[:, jnp.array(obs_indices)]
            if obs_indices is not None
            else ex_observations
        )

        if separate_qv:
            # Independent Q and V networks with shared ψ(s+) encoder.
            psi_encoder = None
            target_psi_encoder = None
            if config.get('encoder') is not None:
                enc = encoder_modules[config['encoder']]
                psi_encoder = GCEncoder(state_encoder=enc())
                target_psi_encoder = GCEncoder(state_encoder=enc())

            shared_psi_def = SharedPsiEncoder(
                hidden_dims=value_kwargs['hidden_dims'],
                latent_dim=value_kwargs['latent_dim'],
                layer_norm=value_kwargs['layer_norm'],
                gc_encoder=psi_encoder,
            )
            target_shared_psi_def = SharedPsiEncoder(
                hidden_dims=value_kwargs['hidden_dims'],
                latent_dim=value_kwargs['latent_dim'],
                layer_norm=value_kwargs['layer_norm'],
                gc_encoder=target_psi_encoder,
            )

            q_def = EmpowermentQNetwork(
                **value_kwargs,
                gc_encoder=encoders.get('q'),
                shared_psi=shared_psi_def,
            )
            v_def = EmpowermentVNetwork(
                **value_kwargs,
                gc_encoder=encoders.get('v'),
                shared_psi=shared_psi_def,
            )
            target_q_def = EmpowermentQNetwork(
                **value_kwargs,
                gc_encoder=target_encoders.get('q'),
                shared_psi=target_shared_psi_def,
            )
            target_v_def = EmpowermentVNetwork(
                **value_kwargs,
                gc_encoder=target_encoders.get('v'),
                shared_psi=target_shared_psi_def,
            )

            network_def = ModuleDict(dict(
                q=q_def, v=v_def,
                target_q=target_q_def, target_v=target_v_def,
                policy=policy_def,
            ))
            network_params = network_def.init(
                init_rng,
                q=(ex_observations, ex_actions, ex_skills, ex_future),
                v=(ex_observations, ex_skills, ex_future),
                target_q=(ex_observations, ex_actions, ex_skills, ex_future),
                target_v=(ex_observations, ex_skills, ex_future),
                policy=(ex_observations, ex_skills),
            )['params']
            network_params['modules_target_q'] = network_params['modules_q']
            network_params['modules_target_v'] = network_params['modules_v']
        else:
            # Single Q network; V is derived at runtime via the policy.
            q_def = EmpowermentQNetwork(**value_kwargs, gc_encoder=encoders.get('q'))
            target_q_def = copy.deepcopy(q_def)
            network_def = ModuleDict(dict(
                q=q_def, target_q=target_q_def, policy=policy_def,
            ))
            network_params = network_def.init(
                init_rng,
                q=(ex_observations, ex_actions, ex_skills, ex_future),
                target_q=(ex_observations, ex_actions, ex_skills, ex_future),
                policy=(ex_observations, ex_skills),
            )['params']
            network_params['modules_target_q'] = network_params['modules_q']

        network = TrainState.create(
            network_def, network_params,
            tx=optax.adam(config.get('lr', 3e-4)),
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ────────────────────────────────────────────────────────────────────


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='empowerment_skill',
        lr=3e-4,
        batch_size=1024,
        hidden_dims=(512, 512, 512),
        value_hidden_dims=ml_collections.config_dict.placeholder(tuple),
        value_latent_dim=256,
        actor_hidden_dims=(512, 512, 512),
        layer_norm=True,
        discount=0.99,
        tau=0.005,
        num_skills=15,
        num_splus_samples=1,
        obs_indices=ml_collections.config_dict.placeholder(tuple),
        bc_alpha=0.0,
        anneal_alpha=False,
        only_bc=False,  # If True, skip the Q/V/policy/empowerment losses entirely and train the policy with pure BC.
        # ── Architecture flag ───────────────────────────────────────────────
        # False (default): V derived from Q via the policy (single network).
        # True:            independent Q and V networks with separate targets.
        separate_qv=True,
        # Self-loss flags
        use_self_v_loss=True,  # Whether to add loss regressing V(s | s, z) onto 1 - discount; this is used ONLY  in the Q representing NEXT state occupancy formulation
        use_self_q_loss=True,  # Whether to add the "self" Q loss terms; this is used ONLY  in the Q representing CURRENT state occupancy formulation
        # Policy-loss flags
        no_target_q_for_policy=True,  # Use main Q/V networks (not target) when computing the policy loss.
        sample_z=True,                # Sample one z per batch element for the policy loss instead of summing analytically over all Z.
        # ── Noisy actuation channel ─────────────────────────────────────────
        # When enabled, a = π(s,z) + η, ignored when discrete=True.
        stochastic_policy_actions=False,
        action_noise_std=0.1,
        # Also perturb *dataset* actions fed into Q in q_loss with the same σ,
        perturb_q_loss_actions=True,
        # Log gating
        log_interval=5000,            # Skip empowerment() inside total_loss except on steps that match this interval.
        # ───────────────────────────────────────────────────────────────────
        discrete=False,
        const_std=True,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='GCDataset',
        value_p_curgoal=0.0,
        value_p_trajgoal=0.0,
        value_p_randomgoal=1.0,
        value_geom_sample=False,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=True,
        p_aug=0.0,
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))