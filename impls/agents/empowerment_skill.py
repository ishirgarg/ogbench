"""
Offline empowerment agent (Myers 2025).

Set  config['separate_qv'] = True   to use independent Q and V networks.
Set  config['separate_qv'] = False  (default) to derive V from Q via the
policy, sharing a single value network (φ(s,a,z) == φ_V(s,z) when a=π(s,z)).

All branching on `separate_qv` is isolated to the V modulation interface:
    _v_phi              – φ_V(s, z)  for a single skill batch
    _v_phi_all_skills   – φ_V(s, z') for every skill  [K, batch, d]
    compute_v_logits        – log V^z(s⁺ | s)   (online network)
    compute_v_logits_target – log V^z(s⁺ | s)   (target network)

Every loss function calls these helpers without any case analysis.
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

    def setup(self):
        self.phi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
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
        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states
        return self.psi_net(future)

    def __call__(self, observations, actions, skills, future_states):
        phi_emb = self.phi(observations, actions, skills)
        psi_emb = self.psi(future_states)
        diff = phi_emb - psi_emb
        l2_sq = jnp.sum(diff ** 2, axis=-1)
        # log p(ψ | φ) for ψ ~ N(φ, (d/2)·I):  −½d·log(πd) − ‖φ−ψ‖²/d
        norm = -0.5 * self.latent_dim * jnp.log(jnp.pi * self.latent_dim)
        return norm - l2_sq / self.latent_dim


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

    def setup(self):
        self.phi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.psi_net = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def phi(self, observations, skills):
        obs = self.gc_encoder(observations, None) if self.gc_encoder else observations
        return self.phi_net(jnp.concatenate([obs, skills], axis=-1))

    def psi(self, future_states):
        future = self.gc_encoder(future_states, None) if self.gc_encoder else future_states
        return self.psi_net(future)

    def __call__(self, observations, skills, future_states):
        phi_emb = self.phi(observations, skills)
        psi_emb = self.psi(future_states)
        diff = phi_emb - psi_emb
        l2_sq = jnp.sum(diff ** 2, axis=-1)
        norm = -0.5 * self.latent_dim * jnp.log(jnp.pi * self.latent_dim)
        return norm - l2_sq / self.latent_dim


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

    def _policy_actions(self, observations, skills_onehot, params):
        """Deterministic policy actions (mode / argmax)."""
        dist = self.network.select('policy')(observations, skills_onehot, params=params)
        if self.config['discrete']:
            return dist.probs.argmax(axis=-1)
        return dist.mode()

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
        norm = -0.5 * latent_dim * jnp.log(jnp.pi * latent_dim)
        return norm - l2_sq / latent_dim

    def _v_phi(self, observations, skills_onehot, *, use_target: bool,
                policy_params=None):
        """φ_V embedding for a single skill batch  →  [batch, d].

        Args:
            observations:  [batch, obs_dim]
            skills_onehot: [batch, K]
            use_target:    use the target network
            policy_params: params for the policy network;
                           None = frozen policy (no grad);
                           grad_params = gradient flows through policy.
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
                                           params=policy_params)
            return net.apply(vars_, observations, actions, skills_onehot,
                             method=net.phi)

    def _v_phi_all_skills(self, observations, *, use_target: bool,
                           policy_params=None):
        """φ_V embeddings for *all* K skills  →  [K, batch, d].

        Args: same as _v_phi.  Internally vmaps over the skill index.
        """
        K = self.config['num_skills']
        batch_size = observations.shape[0]
        skills_onehot = jnp.eye(K)

        if self.config['separate_qv']:
            key = 'target_v' if use_target else 'v'
            net = self.network.model_def.modules[key]
            vars_ = jax.lax.stop_gradient(
                {'params': self.network.params[f'modules_{key}']}
            )

            def phi_for_skill(z_onehot):
                z_batch = jnp.repeat(z_onehot[None, :], batch_size, axis=0)
                return net.apply(vars_, observations, z_batch, method=net.phi)
        else:
            key = 'target_q' if use_target else 'q'
            net = self.network.model_def.modules[key]
            vars_ = jax.lax.stop_gradient(
                {'params': self.network.params[f'modules_{key}']}
            )

            def phi_for_skill(z_onehot):
                z_batch = jnp.repeat(z_onehot[None, :], batch_size, axis=0)
                actions = self._policy_actions(observations, z_batch,
                                               params=policy_params)
                return net.apply(vars_, observations, actions, z_batch,
                                 method=net.phi)

        return jax.vmap(phi_for_skill)(skills_onehot)   # [K, batch, d]

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
                          params=None, policy_params=None):
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
                                           params=policy_params)
            return self.network.select('q')(
                observations, actions, skills_onehot, future_extracted,
                params=params
            )

    def compute_v_logits_target(self, observations, skills_onehot, future_states,
                                 policy_params=None):
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
                                           params=policy_params)
            return self.network.select('target_q')(
                observations, actions, skills_onehot, future_extracted,
                params=None
            )

    def empowerment(self, observations, rng):
        """Monte-Carlo estimate of I(A; S⁺ | s) for each observation."""
        batch_size = observations.shape[0]
        K = self.config['num_skills']
        num_samples = self.config['num_splus_samples']
        d = self.config['value_latent_dim']
        log_K = jnp.log(K)

        rng, sample_rng = jax.random.split(rng)
        skill_rngs = jax.random.split(sample_rng, K)

        # φ_V for all skills: [K, batch, d]  (policy frozen, no grad needed)
        phi_all = self._v_phi_all_skills(
            observations, use_target=False, policy_params=None
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

    def q_loss(self, batch, grad_params, skills_onehot):
        """L_Q = −V^z(s⁺|s) ▷ log Q^z(s⁺|s,a)  (eq. 15)."""
        future = batch['value_goals']

        # Target V from next_observations for the Bellman target (frozen).
        log_v = self.compute_v_logits_target(
            batch['next_observations'], skills_onehot, future
        )
        # Online Q being optimised.
        log_q = self.compute_q_logits(
            batch['observations'], batch['actions'],
            skills_onehot, future, params=grad_params
        )

        v = jnp.exp(log_v)
        one_minus_v = -jnp.expm1(log_v)   # 1 − exp(log_v), precise near 0

        loss = -(
            jax.lax.stop_gradient(v) * log_q
            + jax.lax.stop_gradient(one_minus_v) * log1mexp(log_q)
        ).mean()

        return loss, {
            'q_loss': loss,
            'q_log_mean': log_q.mean(),
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def v_loss(self, batch, grad_params, skills_onehot):
        """L_V — Bellman backup for the occupancy V (eqs. 16-17).

        In combined mode `grad_params` differentiates through the Q network
        (since V≡Q(π)); the policy is frozen (policy_params=None).
        In separate mode `grad_params` differentiates through the V network.
        """
        future = batch['value_goals']

        # Policy is frozen in the Bellman backup (policy_params=None).
        actions_next = self._policy_actions(
            batch['observations'], skills_onehot, params=None
        )

        # Target Q for the backup — frozen.
        log_q_next = self.compute_q_logits_target(
            batch['observations'], actions_next, skills_onehot, future
        )
        # Online V being optimised; policy frozen.
        log_v = self.compute_v_logits(
            batch['observations'], skills_onehot, future,
            params=grad_params, policy_params=None
        )
        v = jnp.exp(log_v)

        log_discount = jnp.log(self.config['discount'])
        discount_exp_q = jnp.exp(log_discount + log_q_next)
        # Use expm1 for numerical precision when γQ is very small.
        one_minus_discount_exp_q = -jnp.expm1(log_discount + log_q_next)

        loss = -(
            jax.lax.stop_gradient(discount_exp_q) * log_v
            + jax.lax.stop_gradient(one_minus_discount_exp_q) * log1mexp(log_v)
        ).mean()

        return loss, {
            'v_loss': loss,
            'v_log_mean': log_v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

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

    def policy_loss(self, batch, grad_params, skills, skills_onehot):
        """Policy loss via empowerment gradient — fully in log-space.

        Gradient flow
        ─────────────
        Combined (separate_qv=False):
            φ_z_v == φ_z_q (same Q network/policy).  Grad flows through
            policy_actions in *both* phi_z_v (psi sampling base) and phi_z_q
            (log_q), and through phi_all for all skills.

        Separate (separate_qv=True):
            φ_z_v comes from the frozen V network (no policy actions) — psi
            sampling has no grad.  φ_z_q comes from the frozen Q network via
            policy_actions(grad_params) — grad flows only through log_q.
            phi_all comes from the frozen V network — no grad.

        All Q/V network *parameters* are stop_grad'd in both modes; grad
        enters only via the policy output (`policy_params=grad_params`).
        """
        batch_size = batch['observations'].shape[0]
        K = self.config['num_skills']
        N = self.config['num_splus_samples']
        d = self.config['value_latent_dim']
        log_K = jnp.log(K)

        rng, sample_rng = jax.random.split(self.rng)

      
        phi_z_v = self._v_phi(
            batch['observations'], skills_onehot,
            use_target=True, policy_params=grad_params
        )  # [batch, d]

        # ── Sample ψ ~ N(φ_z_v, d/2 · I) ─────────────────────────────────
        psi = (
            phi_z_v[None]
            + jax.random.normal(sample_rng, (N, *phi_z_v.shape)) * jnp.sqrt(d / 2.0)
        )  # [N, batch, d]

        policy_actions = self._policy_actions(
            batch['observations'], skills_onehot, params=grad_params
        )
        phi_z_q = self._q_phi(
            batch['observations'], policy_actions, skills_onehot, use_target=True
        )  # [batch, d]

        # log Q^z(ψ | s, π(s,z)):  [N, batch]
        log_q = self._logits_from_embeddings(phi_z_q[None], psi, d)

     
        phi_all = self._v_phi_all_skills(
            batch['observations'], use_target=True, policy_params=grad_params
        )  # [K, batch, d]

        # log V^{z'}(ψ | s) for all z':  [N, K, batch]
        log_v_all = self._logits_from_embeddings(phi_all[None], psi[:, None], d)

        # log V^z for the assigned skill:  [N, batch]
        log_v_z = log_v_all[:, skills, jnp.arange(batch_size)]

     
        log_v_all_lse = logsumexp(log_v_all, axis=1)   # [N, batch]
        log_c_sg = jax.lax.stop_gradient(
            log_diff_exp(log_v_all_lse, log_v_z)
        )  # log(Σ V^{z'} − V^z),  [N, batch]

        # Grad of log_m_bar flows through log_q only (log_c_sg is stop_grad).
        log_m_bar = (
            logsumexp(jnp.stack([log_q, log_c_sg], axis=0), axis=0) - log_K
        )  # [N, batch]

        # ── q_term ────────────────────────────────────────────────────────
        log_q_over_v_z = log_q - jax.lax.stop_gradient(log_v_z)   # [N, batch]
        q_term = jnp.exp(log_q_over_v_z) * (log_q - log_m_bar)    # [N, batch]

        # ── v_others_term (all stop_grad'd — V networks are frozen) ───────
        log_v_ratio = log_v_all - jax.lax.stop_gradient(log_v_z[:, None, :])
        v_ratio = jnp.exp(log_v_ratio)   # [N, K, batch]

        # Σ_{z'} (v_{z'}/v_z) · log v_{z'}
        v_log_v_sum = jax.lax.stop_gradient(
            (v_ratio * log_v_all).sum(axis=1)
        )  # [N, batch]

        # (Σ_{z'} v_{z'}/v_z) · log m̄  — note: grad of log_m_bar is alive
        v_all_sum_weighted = jax.lax.stop_gradient(
            jnp.exp(log_v_all_lse - log_v_z)
        )  # [N, batch]
        v_log_m_sum = v_all_sum_weighted * log_m_bar   # [N, batch]

        # v_z/v_z = 1 analytically; last_term = log v_z − log m̄
        last_term = jax.lax.stop_gradient(log_v_z) - log_m_bar    # [N, batch]

        v_others_term = (v_log_v_sum - v_log_m_sum) - last_term   # [N, batch]

        # ── Aggregate over N samples then over batch ───────────────────────
        e_delta = ((q_term + v_others_term) / K).sum(axis=0)       # [batch]
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
        batch_size = batch['observations'].shape[0]
        skills, skills_onehot = self._sample_skills(skills_rng, batch_size)
        info = {}

        q_loss, q_info = self.q_loss(batch, grad_params, skills_onehot)
        jax.lax.cond(
            jnp.isnan(q_loss),
            lambda: jax.debug.print("⚠️ NaN in q_loss: {x}", x=q_loss, ordered=True),
            lambda: None,
        )
        info.update({f'q/{k}': v for k, v in q_info.items()})

        v_loss, v_info = self.v_loss(batch, grad_params, skills_onehot)
        jax.lax.cond(
            jnp.isnan(v_loss),
            lambda: jax.debug.print("⚠️ NaN in v_loss: {x}", x=v_loss, ordered=True),
            lambda: None,
        )
        info.update({f'v/{k}': v for k, v in v_info.items()})

        pi_loss, pi_info = self.policy_loss(batch, grad_params, skills, skills_onehot)
        jax.lax.cond(
            jnp.isnan(pi_loss),
            lambda: jax.debug.print("⚠️ NaN in pi_loss: {x}", x=pi_loss, ordered=True),
            lambda: None,
        )
        info.update({f'policy/{k}': v for k, v in pi_info.items()})

        bc_loss, bc_info = self.bc_loss(batch, grad_params, skills_onehot)
        jax.lax.cond(
            jnp.isnan(bc_loss),
            lambda: jax.debug.print("⚠️ NaN in bc_loss: {x}", x=bc_loss, ordered=True),
            lambda: None,
        )
        info.update({f'bc/{k}': v for k, v in bc_info.items()})

        empowerment_vals = jax.lax.stop_gradient(
            self.empowerment(batch['observations'], rng=empowerment_rng)
        )
        info['empowerment/mean'] = empowerment_vals.mean()
        info['empowerment/min']  = empowerment_vals.min()
        info['empowerment/max']  = empowerment_vals.max()

        alpha = self.config.get('bc_alpha', 0.0)
        total = q_loss + v_loss + pi_loss + alpha * bc_loss
        jax.lax.cond(
            jnp.isnan(total),
            lambda: jax.debug.print("⚠️ NaN in total_loss: {x}", x=total, ordered=True),
            lambda: None,
        )
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

        single_obs = observations.ndim == 1
        if single_obs:
            observations = observations[None, :]
        if goals is not None and goals.ndim == 1:
            goals = goals[None, :]

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

        # Optional visual encoders.
        encoders = {}
        if config.get('encoder') is not None:
            enc  = encoder_modules[config['encoder']]
            keys = ('q', 'v', 'policy') if separate_qv else ('q', 'policy')
            encoders = {k: GCEncoder(concat_encoder=enc()) for k in keys}

        # Shared kwargs for both Q and V network constructors.
        value_kwargs = dict(
            hidden_dims=config.get('value_hidden_dims', None) or hidden_dims,
            action_dim=action_dim,
            num_skills=num_skills,
            latent_dim=config.get('value_latent_dim', 128),
            layer_norm=config.get('layer_norm', True),
            discrete=config['discrete'],
        )

        q_def        = EmpowermentQNetwork(**value_kwargs, gc_encoder=encoders.get('q'))
        target_q_def = copy.deepcopy(q_def)

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
            # Independent Q and V networks, each with their own target copy.
            v_def        = EmpowermentVNetwork(**value_kwargs, gc_encoder=encoders.get('v'))
            target_v_def = copy.deepcopy(v_def)

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
        value_latent_dim=128,
        actor_hidden_dims=(512, 512, 512),
        layer_norm=True,
        discount=0.99,
        tau=0.005,
        num_skills=5,
        num_splus_samples=8,
        obs_indices=ml_collections.config_dict.placeholder(tuple),
        bc_alpha=0.0,
        # ── Architecture flag ───────────────────────────────────────────────
        # False (default): V derived from Q via the policy (single network).
        # True:            independent Q and V networks with separate targets.
        separate_qv=False,
        # ───────────────────────────────────────────────────────────────────
        discrete=False,
        const_std=True,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='GCDataset',
        value_p_curgoal=0.2,
        value_p_trajgoal=0.5,
        value_p_randomgoal=0.3,
        value_geom_sample=True,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=True,
        p_aug=0.0,
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))