"""
Offline DADS (Dynamics-Aware Discovery of Skills) agent for OGBench.

Reference: https://arxiv.org/abs/1907.01657

Adaptation to the offline setting
─────────────────────────────────
Standard DADS is on-policy: it rolls out the agent in the environment, and
trains (a) a skill dynamics model q_phi(s' | s, z), (b) a SAC policy with the
intrinsic reward r = log q(s'|s,z) - log[(1/K) Σ_{z'} q(s'|s,z')].

Here we cannot interact with the env.  We instead:
  1. Train a *world model*  p_w(s' | s, a)  on the offline OGBench dataset
     by maximum likelihood (Gaussian over Δs = s' − s with learned diagonal
     covariance) for `model_warmup_steps` optimizer steps, then FREEZE it.
  2. With the world model frozen, run online DADS *inside the world model*:
       - sample a batch of initial states from the offline data,
       - sample one skill per state (one-hot),
       - roll out H steps using a_t = π(s_t, z),  s_{t+1} ~ p_w(s_t, a_t),
       - train q_phi by NLL on the synthetic transitions,
       - train SAC π and Q on the synthetic transitions with the DADS
         intrinsic reward.
     Because the policy is rolled out from many randomly drawn offline
     states, the empowerment estimate is well-defined across the offline
     state distribution — i.e., empowerment is a state function evaluable
     anywhere on the offline support.

Empowerment as a state function
───────────────────────────────
DADS estimates the variational lower bound

    I(s'; z | s)  =  E_{z, s' ~ q(·|s,z)} [ log q(s'|s,z) − log (1/K) Σ_{z'} q(s'|s,z') ].

Because q_phi(s'|s,z) is a closed-form Gaussian, we can evaluate this at any
state s by Monte-Carlo sampling s' from q_phi(·|s,z) for each skill z and
averaging.  No environment access or policy rollout is required at evaluation
time — empowerment(s) depends only on the (trained) skill dynamics network.
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

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, MLP, TransformedWithMode, LogParam, default_init


# ── Helpers ───────────────────────────────────────────────────────────────────


def _gaussian_log_prob_diag(x, mean, log_std):
    """Sum of per-dim log N(x; mean, diag(exp(2*log_std)))."""
    var = jnp.exp(2.0 * log_std)
    log_prob_per_dim = -0.5 * (
        jnp.log(2.0 * jnp.pi) + 2.0 * log_std + (x - mean) ** 2 / var
    )
    return log_prob_per_dim.sum(axis=-1)


def _gaussian_log_prob_identity(x, mean):
    """Sum of per-dim log N(x; mean, I)."""
    return -0.5 * jnp.sum((x - mean) ** 2 + jnp.log(2.0 * jnp.pi), axis=-1)


# ── Network modules ──────────────────────────────────────────────────────────


class WorldModel(nn.Module):
    """p_w(s' | s, a) = N(s + μ(s,a), diag(σ²(s,a))) on the full state space."""

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


class SkillDynamics(nn.Module):
    """q_phi(s' | s, z) = N(s_proj + μ(s, z), I) on the goal-relevant subspace.

    Covariance is fixed to identity (DADS convention).
    Output dimension matches the size of `obs_indices`, i.e., the goal subspace.
    """

    hidden_dims: Sequence[int]
    output_dim: int
    layer_norm: bool = True

    def setup(self):
        self.trunk = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_head = nn.Dense(self.output_dim, kernel_init=default_init())

    def __call__(self, observations, skills):
        h = self.trunk(jnp.concatenate([observations, skills], axis=-1))
        return self.mean_head(h)


class SkillCritic(nn.Module):
    """Skill-conditioned SAC critic Q(s, a, z), ensembled over `num_critics`."""

    hidden_dims: Sequence[int]
    num_critics: int = 2
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, actions, skills):
        x = jnp.concatenate([observations, actions, skills], axis=-1)
        qs = []
        for _ in range(self.num_critics):
            q = MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)(x)
            qs.append(q.squeeze(-1))
        return jnp.stack(qs, axis=0)  # [num_critics, batch]


class SkillActor(GCActor):
    """Continuous Gaussian-tanh actor conditioned on a skill one-hot."""

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
        dist = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            dist = TransformedWithMode(dist, distrax.Block(distrax.Tanh(), ndims=1))
        return dist


# ── Agent ────────────────────────────────────────────────────────────────────


class DADSAgent(flax.struct.PyTreeNode):
    """Offline DADS agent.

    Network layout (ModuleDict keys):
      world_model     : p_w(s' | s, a)
      skill_dynamics  : q_phi(s' | s, z)  (DADS skill dynamics)
      actor           : π(a | s, z)
      critic          : Q(s, a, z)  (ensemble)
      target_critic   : target Q for SAC bootstrap
      alpha           : SAC entropy temperature (LogParam)
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── Basic helpers ────────────────────────────────────────────────────────

    def _proj(self, states):
        """Project full state onto the goal-relevant subspace used by q_phi."""
        obs_indices = self.config.get('obs_indices', None)
        if obs_indices is not None:
            return states[..., jnp.array(obs_indices)]
        return states

    def _proj_dim(self):
        """Output dimensionality of q_phi (size of goal subspace)."""
        obs_indices = self.config.get('obs_indices', None)
        if obs_indices is not None:
            return len(obs_indices)
        return self.config['state_dim']

    def _sample_skills(self, rng, batch_size):
        """Sample one-hot skills uniformly."""
        K = self.config['num_skills']
        idx = jax.random.randint(rng, (batch_size,), 0, K)
        return idx, jnp.eye(K)[idx]

    # ── World model (offline-data) loss ──────────────────────────────────────

    def world_model_loss(self, batch, grad_params):
        """NLL of Δs = s' − s under p_w(·|s,a) on offline transitions."""
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

    # ── Model rollout (used inside DADS branch) ──────────────────────────────

    def rollout_in_model(self, init_obs, skills_onehot, rng):
        """Roll out the policy in the (frozen) world model for H steps.

        No gradients flow through this rollout — it serves as a non-
        differentiable simulator producing synthetic (s, a, z, s') tuples.

        Args:
            init_obs:      [B, state_dim]
            skills_onehot: [B, K]   (one skill per env, fixed for the rollout)
            rng:           PRNGKey

        Returns:
            states      : [H, B, state_dim]
            actions     : [H, B, action_dim]
            next_states : [H, B, state_dim]
            skills_seq  : [H, B, K]
        """
        H = self.config['rollout_length']

        def step(carry, _):
            obs, key = carry
            key, action_key, dyn_key = jax.random.split(key, 3)

            dist = self.network.select('actor')(obs, skills_onehot)
            action = dist.sample(seed=action_key)
            action = jnp.clip(action, -1.0, 1.0)

            mean, log_std = self.network.select('world_model')(obs, action)
            noise = jax.random.normal(dyn_key, mean.shape)
            delta = mean + jnp.exp(log_std) * noise
            next_obs = obs + delta

            return (next_obs, key), (obs, action, next_obs)

        (_, _), (states, actions, next_states) = jax.lax.scan(
            step, (init_obs, rng), None, length=H
        )
        skills_seq = jnp.broadcast_to(skills_onehot[None], (H,) + skills_onehot.shape)
        # All quantities returned have no gradient w.r.t. params (they came from
        # the world model and policy without grad_params); we still wrap in
        # stop_gradient to be defensive against accidental grad flow.
        return jax.lax.stop_gradient((states, actions, next_states, skills_seq))

    # ── Skill dynamics & DADS reward ─────────────────────────────────────────

    def _skill_dynamics_apply(self, observations, skills, params=None):
        return self.network.select('skill_dynamics')(observations, skills, params=params)

    def _delta_proj(self, observations, next_observations):
        """Δ on the goal subspace (the target for q_phi)."""
        return self._proj(next_observations) - self._proj(observations)

    def skill_dynamics_loss(self, observations, next_observations, skills_onehot, grad_params):
        """NLL of Δproj under q_phi(·|s,z) = N(μ_phi, I)."""
        target = self._delta_proj(observations, next_observations)
        mean = self._skill_dynamics_apply(observations, skills_onehot, params=grad_params)

        # Batch-normalise inputs/targets following the DADS paper convention to
        # stabilise the dynamics regression across very-different scale features.
        target_mean = jnp.mean(target, axis=0, keepdims=True)
        target_std = jnp.std(target, axis=0, keepdims=True) + 1e-6
        target_n = (target - target_mean) / target_std
        mean_n = (mean - target_mean) / target_std

        log_prob = _gaussian_log_prob_identity(target_n, mean_n)
        loss = -log_prob.mean()
        return loss, {
            'skill_dynamics_loss': loss,
            'skill_dynamics_mean_abs_err': jnp.mean(jnp.abs(target - mean)),
        }

    def dads_intrinsic_reward(self, observations, next_observations, skills_idx):
        """r(s, z, s') = log q(s'|s,z) − log [(1/K) Σ_{z'} q(s'|s,z')].

        Uses the (frozen) skill dynamics network (no grad).  Inputs/targets are
        batch-normalised for numerical parity with the dynamics loss.
        """
        K = self.config['num_skills']
        B = observations.shape[0]

        target = self._delta_proj(observations, next_observations)  # [B, dproj]

        # Batch normalisation for parity with skill_dynamics_loss
        target_mean = jnp.mean(target, axis=0, keepdims=True)
        target_std = jnp.std(target, axis=0, keepdims=True) + 1e-6
        target_n = (target - target_mean) / target_std

        # log q(s'|s, z) for the actually-sampled skill (no grad)
        skills_onehot = jnp.eye(K)[skills_idx]
        mean_z = jax.lax.stop_gradient(
            self._skill_dynamics_apply(observations, skills_onehot)
        )
        mean_z_n = (mean_z - target_mean) / target_std
        log_q_z = _gaussian_log_prob_identity(target_n, mean_z_n)  # [B]

        # log q(s'|s, z') for ALL skills z' — vmap over the skill index
        all_skills = jnp.eye(K)  # [K, K]

        def log_q_for_skill(z_oh):
            z_batch = jnp.broadcast_to(z_oh[None], (B, K))
            mean_zp = jax.lax.stop_gradient(
                self._skill_dynamics_apply(observations, z_batch)
            )
            mean_zp_n = (mean_zp - target_mean) / target_std
            return _gaussian_log_prob_identity(target_n, mean_zp_n)

        log_q_all = jax.vmap(log_q_for_skill)(all_skills)  # [K, B]
        log_marginal = jax.scipy.special.logsumexp(log_q_all, axis=0) - jnp.log(K)
        return log_q_z - log_marginal  # [B]

    # ── SAC losses (skill-conditioned, on synthetic transitions) ─────────────

    def sac_critic_loss(self, observations, actions, rewards, next_observations,
                         skills_onehot, grad_params, rng):
        """SAC critic loss with the DADS intrinsic reward.

        We do not have explicit episode terminals from the model rollout;
        we treat all transitions as non-terminal (mask = 1).
        """
        next_dist = self.network.select('actor')(next_observations, skills_onehot)
        next_actions = next_dist.sample(seed=rng)
        next_log_probs = next_dist.log_prob(next_actions)
        next_actions = jnp.clip(next_actions, -1.0, 1.0)

        next_qs = self.network.select('target_critic')(
            next_observations, next_actions, skills_onehot
        )  # [num_critics, B]
        next_q = jnp.min(next_qs, axis=0)

        alpha = self.network.select('alpha')()
        target = rewards + self.config['discount'] * (
            next_q - alpha * next_log_probs
        )
        target = jax.lax.stop_gradient(target)

        qs = self.network.select('critic')(
            observations, actions, skills_onehot, params=grad_params
        )  # [num_critics, B]
        loss = jnp.mean((qs - target[None]) ** 2)
        return loss, {
            'critic_loss': loss,
            'q_mean': qs.mean(),
            'q_min': qs.min(),
            'q_max': qs.max(),
            'target_q_mean': target.mean(),
            'reward_mean': rewards.mean(),
        }

    def sac_actor_loss(self, observations, skills_onehot, grad_params, rng):
        """SAC actor loss + auto-tuned entropy temperature loss."""
        dist = self.network.select('actor')(observations, skills_onehot, params=grad_params)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)

        qs = self.network.select('critic')(observations, actions, skills_onehot)
        q = jnp.min(qs, axis=0)

        alpha_no_grad = self.network.select('alpha')()
        actor_loss = (alpha_no_grad * log_probs - q).mean()

        # Alpha (entropy) loss
        alpha = self.network.select('alpha')(params=grad_params)
        target_entropy = self.config['target_entropy']
        alpha_loss = -(alpha * jax.lax.stop_gradient(log_probs + target_entropy)).mean()

        total = actor_loss + alpha_loss
        return total, {
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': -log_probs.mean(),
        }

    def bc_loss(self, batch, skills_onehot, grad_params):
        """BC constraint on the DADS policy: −log π(a_expert | s, z).

        Pulls π toward the offline action distribution at each state, under the
        sampled skill.  Same recipe as in `empowerment_skill.py`: skills are
        sampled uniformly, and the expert action at each (s, a) pair is treated
        as in-distribution under any skill.  Stabilises the policy and keeps
        sampled actions on the data manifold while DADS still drives diversity.
        """
        dist = self.network.select('actor')(
            batch['observations'], skills_onehot, params=grad_params
        )
        log_prob = dist.log_prob(batch['actions'])
        loss = -log_prob.mean()
        return loss, {
            'bc_loss': loss,
            'bc_log_prob_mean': log_prob.mean(),
            'bc_log_prob_min': log_prob.min(),
            'bc_log_prob_max': log_prob.max(),
        }

    # ── Empowerment (state-only function) ────────────────────────────────────

    def empowerment(self, observations, rng):
        """E(s) ≈ (1/K) Σ_z E_{s'~q(·|s,z)} [log q(s'|s,z) − log (1/K) Σ_{z'} q(s'|s,z')].

        Pure function of the state — depends only on the trained skill
        dynamics network, not on the policy or the world model.
        """
        K = self.config['num_skills']
        N = self.config['num_splus_samples']
        B = observations.shape[0]
        dproj = self._proj_dim()

        # Means of q(·|s, z) for every skill: [K, B, dproj]
        all_skills = jnp.eye(K)

        def mean_for_skill(z_oh):
            z_batch = jnp.broadcast_to(z_oh[None], (B, K))
            return self._skill_dynamics_apply(observations, z_batch)

        mu_all = jax.vmap(mean_for_skill)(all_skills)  # [K, B, dproj]

        rng, sample_rng = jax.random.split(rng)
        skill_rngs = jax.random.split(sample_rng, K)

        def emp_for_skill(mu_z, skill_rng):
            # Sample s' ~ N(mu_z, I) for this skill
            noise = jax.random.normal(skill_rng, (N, B, dproj))
            samples = mu_z[None] + noise  # [N, B, dproj]

            def contrib(s_sample):
                # log q(s_sample | s, z)   for this skill   shape [B]
                log_q_z = _gaussian_log_prob_identity(s_sample, mu_z)
                # log q(s_sample | s, z')  for all skills   shape [K, B]
                log_q_all = jax.vmap(
                    lambda mu: _gaussian_log_prob_identity(s_sample, mu)
                )(mu_all)
                log_marg = jax.scipy.special.logsumexp(log_q_all, axis=0) - jnp.log(K)
                return log_q_z - log_marg

            contribs = jax.vmap(contrib)(samples)  # [N, B]
            return contribs.mean(axis=0)

        emp_per_skill = jax.vmap(emp_for_skill, in_axes=(0, 0))(mu_all, skill_rngs)
        return emp_per_skill.mean(axis=0)  # [B]

    # ── Total loss ───────────────────────────────────────────────────────────

    def _dads_branch(self, batch, grad_params, rng):
        """Compute DADS losses on a batch of synthetic (model-rollout) transitions.

        Sequence:
          1. Sample H-step rollouts in the (frozen) world model from offline
             initial states with skills sampled per-rollout.
          2. Flatten T*B transitions and compute skill dynamics, critic, actor
             losses.
        """
        rollout_rng, skill_rng, sd_rng, critic_rng, actor_rng = jax.random.split(rng, 5)

        B = batch['observations'].shape[0]

        # Sample a skill per starting state (kept fixed throughout rollout)
        skills_idx, skills_onehot = self._sample_skills(skill_rng, B)

        states, actions, next_states, skills_seq = self.rollout_in_model(
            batch['observations'], skills_onehot, rollout_rng
        )
        # Flatten H, B → (H*B)
        flat_states = states.reshape((-1,) + states.shape[2:])
        flat_actions = actions.reshape((-1,) + actions.shape[2:])
        flat_next_states = next_states.reshape((-1,) + next_states.shape[2:])
        flat_skills = skills_seq.reshape((-1,) + skills_seq.shape[2:])
        flat_skills_idx = jnp.tile(skills_idx, states.shape[0])

        # 1) Skill dynamics NLL (gradient through skill_dynamics module only)
        sd_loss, sd_info = self.skill_dynamics_loss(
            flat_states, flat_next_states, flat_skills, grad_params
        )

        # 2) Intrinsic reward (no grad; uses frozen skill_dynamics)
        intrinsic_r = self.config['reward_scale'] * self.dads_intrinsic_reward(
            flat_states, flat_next_states, flat_skills_idx
        )

        # 3) SAC critic loss
        critic_loss, critic_info = self.sac_critic_loss(
            flat_states, flat_actions, intrinsic_r, flat_next_states,
            flat_skills, grad_params, critic_rng
        )

        # 4) SAC actor + alpha loss
        actor_loss, actor_info = self.sac_actor_loss(
            flat_states, flat_skills, grad_params, actor_rng
        )

        info = {}
        info.update({f'skill_dyn/{k}': v for k, v in sd_info.items()})
        info.update({f'critic/{k}': v for k, v in critic_info.items()})
        info.update({f'actor/{k}': v for k, v in actor_info.items()})
        info['dads/intrinsic_reward_mean'] = intrinsic_r.mean()
        info['dads/intrinsic_reward_min'] = intrinsic_r.min()
        info['dads/intrinsic_reward_max'] = intrinsic_r.max()
        return sd_loss + critic_loss + actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        wm_rng, dads_rng, bc_skill_rng, emp_rng = jax.random.split(rng, 4)

        wm_loss, wm_info = self.world_model_loss(batch, grad_params)
        info = {f'world_model/{k}': v for k, v in wm_info.items()}

        # Two-phase schedule:
        #   step <  model_warmup_steps:  only world-model + BC losses train.
        #   step >= model_warmup_steps:  world model is FROZEN; DADS losses
        #                                (skill dynamics, SAC, BC) train.
        in_dads = self.network.step >= self.config['model_warmup_steps']
        in_dads_f = in_dads.astype(jnp.float32)
        wm_active = 1.0 - in_dads_f

        dads_loss, dads_info = self._dads_branch(batch, grad_params, dads_rng)

        # BC constraint on the policy, computed on the offline batch with a
        # freshly sampled skill per (s, a).
        B = batch['observations'].shape[0]
        _, bc_skills_onehot = self._sample_skills(bc_skill_rng, B)
        bc_loss_val, bc_info = self.bc_loss(batch, bc_skills_onehot, grad_params)
        info.update({f'bc/{k}': v for k, v in bc_info.items()})

        # Optional exponential annealing of the BC coefficient (matches
        # empowerment_skill.py: half-life of 100k optimiser steps).
        base_alpha = self.config.get('bc_alpha', 0.0)
        if self.config.get('anneal_alpha', False):
            step = jnp.asarray(self.network.step, dtype=jnp.float32)
            bc_alpha = base_alpha * jnp.power(0.5, step / 100_000.0)
        else:
            bc_alpha = jnp.asarray(base_alpha, dtype=jnp.float32)
        info['bc/alpha'] = bc_alpha

        # Empowerment metric (no grad) — only meaningful once skill dynamics
        # has been trained, but always cheap to compute.
        emp = jax.lax.stop_gradient(
            self.empowerment(batch['observations'], emp_rng)
        )
        info['empowerment/mean'] = emp.mean()
        info['empowerment/min'] = emp.min()
        info['empowerment/max'] = emp.max()

        for k, v in dads_info.items():
            info[k] = v
        info['dads/active'] = in_dads_f

        total = wm_active * wm_loss + in_dads_f * dads_loss + bc_alpha * bc_loss_val
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        # Snapshot world-model params BEFORE the update.  Once past warmup we
        # restore them, which freezes the world model robustly even though
        # Adam's momentum on those params from warmup would otherwise keep
        # nudging them after their gradient becomes zero.
        old_wm = self.network.params['modules_world_model']
        in_dads = self.network.step >= self.config['model_warmup_steps']

        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )

        # Soft update target critic
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            new_network.params['modules_critic'],
            new_network.params['modules_target_critic'],
        )

        # Freeze world model after warmup by reverting its params.
        frozen_wm = jax.tree_util.tree_map(
            lambda old, new: jnp.where(in_dads, old, new),
            old_wm,
            new_network.params['modules_world_model'],
        )

        new_params = {
            **new_network.params,
            'modules_target_critic': new_target,
            'modules_world_model': frozen_wm,
        }
        new_network = new_network.replace(params=new_params)
        return self.replace(network=new_network, rng=new_rng), info

    # ── Evaluation ───────────────────────────────────────────────────────────

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions.  Goals (if provided) are mapped deterministically to
        skill indices via a hash so the same goal always maps to the same skill.
        """
        if seed is None:
            seed = self.rng

        single_obs = observations.ndim == 1
        if single_obs:
            observations = observations[None, :]
        if goals is not None and goals.ndim == 1:
            goals = goals[None, :]

        B = observations.shape[0]
        K = self.config['num_skills']

        if goals is not None:
            goal_proj = self._proj(goals)
            goal_flat = goal_proj.reshape(B, -1).astype(jnp.int32)
            goal_hash = jnp.sum(goal_flat, axis=-1)
            skills_idx = (jnp.abs(goal_hash) % K).astype(jnp.int32)
        else:
            skills_idx = jax.random.randint(seed, (B,), 0, K)
        skills_onehot = jnp.eye(K)[skills_idx]

        dist = self.network.select('actor')(observations, skills_onehot, temperature=temperature)
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
        num_skills = config['num_skills']

        # Track state_dim in config for downstream methods.
        config = dict(config)
        config['state_dim'] = state_dim
        if config.get('target_entropy', None) is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # Goal-subspace dimension for q_phi
        obs_indices = config.get('obs_indices', None)
        proj_dim = len(obs_indices) if obs_indices is not None else state_dim

        hidden_dims = config['hidden_dims']
        value_hidden_dims = config.get('value_hidden_dims') or hidden_dims
        actor_hidden_dims = config.get('actor_hidden_dims') or hidden_dims
        layer_norm = config['layer_norm']

        world_model_def = WorldModel(
            hidden_dims=hidden_dims,
            state_dim=state_dim,
            layer_norm=layer_norm,
        )
        skill_dynamics_def = SkillDynamics(
            hidden_dims=hidden_dims,
            output_dim=proj_dim,
            layer_norm=layer_norm,
        )
        actor_def = SkillActor(
            hidden_dims=actor_hidden_dims,
            action_dim=action_dim,
            log_std_min=-5,
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )
        critic_def = SkillCritic(
            hidden_dims=value_hidden_dims,
            num_critics=config['num_critics'],
            layer_norm=layer_norm,
        )
        target_critic_def = copy.deepcopy(critic_def)
        alpha_def = LogParam(init_value=config['init_alpha'])

        ex_skills = jnp.eye(num_skills)[jnp.arange(ex_observations.shape[0]) % num_skills]

        network_info = dict(
            world_model=(world_model_def, (ex_observations, ex_actions)),
            skill_dynamics=(skill_dynamics_def, (ex_observations, ex_skills)),
            actor=(actor_def, (ex_observations, ex_skills)),
            critic=(critic_def, (ex_observations, ex_actions, ex_skills)),
            target_critic=(target_critic_def, (ex_observations, ex_actions, ex_skills)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_critic'] = network_params['modules_critic']

        network = TrainState.create(network_def, network_params, tx=network_tx)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ───────────────────────────────────────────────────────────────────


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='dads',
        # ── Optimisation ─────────────────────────────────────────────────────
        lr=3e-4,
        batch_size=512,
        # ── Architecture ─────────────────────────────────────────────────────
        hidden_dims=(512, 512, 512),
        value_hidden_dims=ml_collections.config_dict.placeholder(tuple),
        actor_hidden_dims=ml_collections.config_dict.placeholder(tuple),
        layer_norm=True,
        # ── DADS ─────────────────────────────────────────────────────────────
        num_skills=8,
        num_splus_samples=64,             # MC samples for empowerment estimate
        reward_scale=1.0,
        rollout_length=50,                # H, model-rollout horizon
        model_warmup_steps=500_000,       # train world model only for this many steps; frozen after
        # Goal-subspace projection: which obs dims define skills (e.g. (0, 1) for
        # x-y in ant maze).  None = use full state for skill dynamics output.
        obs_indices=ml_collections.config_dict.placeholder(tuple),
        # ── BC constraint on the policy ──────────────────────────────────────
        bc_alpha=0.1,                     # coefficient on −log π(a_expert | s, z)
        anneal_alpha=False,               # exponentially decay bc_alpha (half-life 100k)
        # ── SAC ──────────────────────────────────────────────────────────────
        discount=0.99,
        tau=0.005,
        num_critics=2,
        init_alpha=1.0,
        target_entropy=ml_collections.config_dict.placeholder(float),
        target_entropy_multiplier=0.5,
        tanh_squash=True,
        state_dependent_std=True,
        actor_fc_scale=0.01,
        # ── Compatibility with OGBench main.py / GCDataset ───────────────────
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='GCDataset',
        # The DADS agent does not use these goal-relabelling fields, but
        # GCDataset asserts they sum to 1 and produces value/actor goals which
        # we simply ignore.
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
