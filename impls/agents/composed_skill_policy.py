"""Composed (non-frozen) hierarchical policy for offline goal-conditioned RL.

Every other `*_controller` agent in this directory trains a HIGH LEVEL on top of a
HARD-FROZEN low level, over an option/semi-MDP built by relabelling length-H windows
with a skill label. This agent does neither. There is no skill horizon, no window, no
relabelling pass, and the low level is trained: the two levels are simply composed into
one flat policy over primitive actions,

    pi(a | s, g)  =  sum_{k=1..K}  pi_hi(k | s, g) * pi_lo(a | s, z_k),            (1)

and that mixture is what offline RL improves. At every environment step the high level
draws a skill k ~ pi_hi(.|s,g) and the low level emits a ~ pi_lo(.|s,z_k); the skill is
redrawn on the next step. Eq. 1 is the exact marginal of that procedure.

Why the discrete skill space makes this clean
---------------------------------------------
All the pretrained skill families here (`empowerment_skill`, `dds`, `opal` with
`latent_type='discrete'`) have a FINITE skill set of size K: the skill is an index,
fed to the low level either as a one-hot (empowerment, opal-discrete) or as the
VQ codebook vector z_k (dds). So the sum in Eq. 1 is a real, exactly computable sum
over K terms -- no Gumbel-softmax, no straight-through estimator, no REINFORCE. Two
consequences drive the whole implementation:

  * The composed log-likelihood is exact:

        log pi(a|s,g) = logsumexp_k [ log pi_hi(k|s,g) + log pi_lo(a|s,z_k) ],     (2)

    one low-level forward pass per skill (vmapped over K). Its gradient reaches BOTH
    levels, weighted by the posterior responsibility

        r_k(s,a,g) = softmax_k [ log pi_hi(k|s,g) + log pi_lo(a|s,z_k) ].          (3)

    Eq. 3 is exactly the "which skill explains this transition" question that the
    frozen-low-level controllers answer in a one-shot offline relabelling pass
    (`skill_bc_relabel_controller.chunk_skill_logliks` and friends). Here it falls out
    of the gradient for free, per batch, and -- crucially -- it stays CORRECT as the low
    level moves. That is the main reason a precomputed label table is not merely
    unnecessary but wrong once pi_lo is trainable: the labels would go stale.

  * The Q-value of the composed policy is also an exact K-term sum:

        E_{k~pi_hi} Q(s, a_k, g),   a_k = mode pi_lo(.|s,z_k),                     (4)

    so the DDPG+BC branch differentiates through the categorical probabilities
    analytically (high level) and through a_k (low level).

The critic
----------
Because the policy is flat, the critic must be flat too: Q(s, a, g) and V(s, g) over
PRIMITIVE actions on the ordinary per-step `GCDataset`. A skill-level critic Q(s, k, g)
-- what every sibling controller uses -- cannot work here: it has no functional
dependence on the low-level parameters at all, so it can produce no gradient for
pi_lo. Hence the critic is an off-the-shelf goal-conditioned agent (`gciql` or `crl`,
selected with `--agent=agents/composed_skill_policy.py:<name>`) trained exactly as it
would be standalone, and read through the two accessors both agents share verbatim:
`network.select('value')(s, g)` and `network.select('critic')(s, g, a)`.

That base agent also keeps training its own FLAT actor. It is never used for control;
it costs one small MLP per step and gives you the flat single-level baseline logged
side by side (`actor/*`) with the composed one (`composed/*`) in the same run.

The two actor losses (`composed_actor_loss`)
--------------------------------------------
  'awr' (default)   L = -E[ min(exp(alpha*A), 100) * log pi(a|s,g) ],  A = Q(s,a,g)-V(s,g),
                    with log pi from Eq. 2. Advantage-weighted maximum likelihood of a
                    mixture-of-experts policy: good transitions pull up the responsible
                    skill's probability AND that skill's action likelihood.
  'ddpgbc'          L = -E_k[Q(s,a_k,g)] / |.|  -  alpha * log pi(a|s,g),  Eq. 4 + Eq. 2
                    as the BC anchor. Continuous actions only.

The AWR branch never differentiates through Q, so it is the safe default offline; the
DDPG+BC branch is stronger when the critic is trustworthy.

Separate learning rates
-----------------------
The high level is fresh and needs a normal learning rate; the low level is pretrained
and must not be destroyed, so it gets its own optimizer at `low_lr` (default 3e-5, i.e.
lr/10). They are two `TrainState`s -- `high` and `skill_agent.network` -- updated from
ONE `jax.value_and_grad` over the pair, so the composition is differentiated jointly
and only the step sizes differ. `low_lr=0.0` recovers the frozen-low-level baseline
exactly (Adam with lr 0 is a no-op), which is the ablation this agent exists to beat.

The pretrained agent's own optimizer is discarded and replaced at `create` time: `dds`
in particular ships a phased `optax.multi_transform` that HARD-FREEZES the decoder after
`skill_pretrain_steps`, which would silently zero every update here.

Only the low-level action module receives gradient; the rest of the pretrained agent's
modules (its critics, its own high level) sit in the same parameter tree, get exactly
zero gradient from this loss, and therefore never move.

Supported low levels (`skill_agent_name` is read from the checkpoint's flags.json)
---------------------------------------------------------------------------------
  empowerment_skill   pi_lo = `policy`(s, one_hot(k))            exact Gaussian density.
  opal (discrete)     pi_lo = `decoder`(concat[s, one_hot(k)])   exact Gaussian density.
  dds (discrete env)  pi_lo = `decoder`(s, z_k)                  exact categorical density.
  dds (continuous)    the decoder is a 5-step DDPM epsilon-network with NO tractable
                      density. Eq. 2 then uses the standard diffusion surrogate
                      log pi_lo(a|s,z) ~ -c * E_{t,eps} ||eps_psi(a_t,t,s,z) - eps||^2
                      (a scaled negative ELBO, up to an additive constant that is the
                      same for every k and so cancels inside the softmax of Eq. 3).
                      `c` is `diffusion_logprob_scale` and it MATTERS: it sets how peaked
                      the responsibilities are. Opt in with `low_logprob='diffusion'`.
                      The DDPG+BC branch unrolls the sampler, which is differentiable but
                      `diffusion_steps` times more expensive.
  opal (continuous)   rejected: the latent is continuous, so Eq. 1 is an integral, not a
                      sum.
  skill_dt            rejected: its policy is a causal Transformer over a context window
                      plus a future-skill histogram, so pi_lo(a|s,z) is not a function of
                      (s, z) alone and the per-step composition is not even well defined.

Cost. One training step costs K low-level forward passes on the batch (`jax.vmap` over
the skill axis). At K=15 and batch 1024 that is fine; if K is large, set
`composed_batch_size` to subsample the rows used by the composed actor loss -- the
critic still sees the full batch.
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
import optax

from agents.crl import CRLAgent
from agents.dds import DDSAgent, diffusion_schedule
from agents.empowerment_skill import EmpowermentAgent
from agents.gciql import GCIQLAgent
from agents.opal import OPALAgent
from agents.skill_bc_relabel_controller import _DATASET_CONFIG_KEYS, _latest_epoch
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent
from utils.networks import GCDiscreteActor

# Flat goal-conditioned algorithms that can serve as the critic. Both expose the two
# accessors this agent needs with identical signatures:
#   network.select('value')(observations, goals)              -> V  [B]
#   network.select('critic')(observations, goals, actions)    -> (Q1, Q2)
BASE_AGENTS = {
    'gciql': GCIQLAgent,
    'crl': CRLAgent,
}

# Pretrained low-level families this agent knows how to compose with.
SKILL_AGENTS = {
    'empowerment_skill': EmpowermentAgent,
    'opal': OPALAgent,
    'dds': DDSAgent,
}


def _base_config(base_agent_name):
    """The nested `agent.base` config for the flat critic."""
    if base_agent_name not in BASE_AGENTS:
        raise ValueError(
            f'base_agent_name must be one of {sorted(BASE_AGENTS)}, got {base_agent_name!r}. '
            f'Select it as a config-file argument: '
            f'--agent=agents/composed_skill_policy.py:<name>'
        )
    module = __import__(f'agents.{base_agent_name}', fromlist=['get_config'])
    base = module.get_config()
    for key in _DATASET_CONFIG_KEYS:
        if key in base:
            del base[key]
    # The critic's action space IS the environment's, unlike every sibling controller, whose
    # inner agent is discrete over the K skills. `create` overwrites this from the top-level
    # `discrete` once the environment is known; the value here is only the config-file default.
    base.discrete = False
    # `crl` only builds its V network when actor_loss='awr', and the composed AWR loss
    # needs V. Keeping the flat baseline's loss aligned with the composed one also makes
    # the two directly comparable in the logs. Override with --agent.base.actor_loss.
    base.actor_loss = 'awr'
    return base


class ComposedSkillPolicyAgent(flax.struct.PyTreeNode):
    """Flat mixture policy pi(a|s,g) = sum_k pi_hi(k|s,g) pi_lo(a|s,z_k), both trained.

    Fields:
        rng: PRNG key.
        base: Flat goal-conditioned agent over PRIMITIVE actions (`gciql` or `crl`).
            Supplies Q(s,a,g) and V(s,g); its own flat actor is trained but unused.
        high: TrainState holding the high level pi_hi(k|s,g), a K-way `GCDiscreteActor`.
            Its own Adam at `lr`.
        skill_agent: The pretrained low-level agent. Unlike every sibling controller this
            is TRAINED: its `network` TrainState carries a fresh Adam at `low_lr`, and
            gradients from the composed actor loss flow into its action module.
        config: Static configuration dictionary.
    """

    rng: Any
    base: Any
    high: Any
    skill_agent: Any
    config: Any = nonpytree_field()

    # ── Low-level adapters ────────────────────────────────────────────────────
    #
    # Everything family-specific lives in these four helpers. `low_params` is either
    # None (stop gradient, use stored params) or the traced parameter tree of
    # `skill_agent.network`, exactly as `TrainState.__call__` expects.

    def _skill_vectors(self, low_params=None):
        """The K conditioning vectors {z_k} handed to the low level. [K, D].

        One-hots for `empowerment_skill` / `opal`-discrete; the live VQ codebook for
        `dds` (so the codes themselves are fine-tuned along with the decoder).
        """
        num_skills = int(self.config['num_skills'])
        if self.config['skill_agent_name'] == 'dds':
            params = self.skill_agent.network.params if low_params is None else low_params
            return params['modules_codebook']['codebook']  # [K, D_z]
        return jnp.eye(num_skills)

    def _low_dist(self, observations, skills, low_params=None, temperature=1.0):
        """pi_lo(. | s, z) as a distrax distribution. Raises for DDS + continuous actions."""
        name = self.config['skill_agent_name']
        net = self.skill_agent.network
        if name == 'empowerment_skill':
            return net.select('policy')(observations, skills, temperature=temperature, params=low_params)
        if name == 'opal':
            inputs = jnp.concatenate([observations, skills], axis=-1)
            return net.select('decoder')(inputs, temperature, params=low_params)
        if name == 'dds':
            if not self.config['discrete']:
                raise ValueError(
                    "DDS's continuous decoder is a diffusion epsilon-network and has no "
                    "closed-form density; this path is only for discrete-action envs."
                )
            return net.select('decoder')(observations, skills, temperature=temperature, params=low_params)
        raise ValueError(f'Unsupported skill_agent_name: {name}')

    def _low_log_prob(self, observations, skills, actions, low_params=None, rng=None):
        """log pi_lo(a | s, z), exact or (DDS+continuous) the diffusion surrogate. [B]."""
        if self.config['low_logprob'] == 'diffusion':
            return -self.config['diffusion_logprob_scale'] * self._diffusion_eps_mse(
                observations, skills, actions, low_params, rng
            )
        return self._low_dist(observations, skills, low_params).log_prob(actions)

    def _diffusion_eps_mse(self, observations, skills, actions, low_params, rng):
        """E_{t,eps} ||eps_psi(a_t, t, s, z) - eps||^2 for the DDS decoder. [B].

        One shared (t, eps) draw across the K skills of a row -- a common random number,
        so the K surrogate scores compared inside Eq. 3 are differences of the same
        noise realisation rather than K independent ones.
        """
        T = int(self.skill_agent.config['diffusion_steps'])
        _, _, alpha_bar, _ = diffusion_schedule(
            T, self.skill_agent.config['beta_min'], self.skill_agent.config['beta_max']
        )
        rng_t, rng_noise = jax.random.split(rng)
        tt = jax.random.randint(rng_t, (actions.shape[0],), 0, T)
        ab = alpha_bar[tt][:, None]
        noise = jax.random.normal(rng_noise, actions.shape)
        x_t = jnp.sqrt(ab) * actions + jnp.sqrt(1.0 - ab) * noise
        times = tt.astype(jnp.float32) / T
        pred_noise = self.skill_agent.network.select('decoder')(
            x_t, times, observations, skills, params=low_params
        )
        return jnp.sum((pred_noise - noise) ** 2, axis=-1)

    def _low_action(self, observations, skills, low_params=None, rng=None, temperature=None):
        """A single action from pi_lo(.|s,z): the mode for density families, an unrolled
        DDPM sample for DDS + continuous actions (differentiable in `low_params`)."""
        if self.config['low_logprob'] == 'diffusion':
            actions = self._ddpm_sample(observations, skills, rng, low_params)
            return jnp.clip(actions, -1, 1)
        temperature = self.config['low_temperature'] if temperature is None else temperature
        dist = self._low_dist(observations, skills, low_params, temperature=temperature)
        if self.config['discrete']:
            return dist.sample(seed=rng)
        return jnp.clip(dist.mode() if rng is None else dist.sample(seed=rng), -1, 1)

    def _ddpm_sample(self, observations, skills, rng, low_params=None):
        """`DDSAgent._ddpm_sample`, re-expressed so gradients reach `low_params`."""
        T = int(self.skill_agent.config['diffusion_steps'])
        betas, alphas, alpha_bar, _ = diffusion_schedule(
            T, self.skill_agent.config['beta_min'], self.skill_agent.config['beta_max']
        )
        batch_shape = skills.shape[:-1]
        rng, noise_rng = jax.random.split(rng)
        x = jax.random.normal(noise_rng, (*batch_shape, int(self.skill_agent.config['action_dim'])))
        for t in reversed(range(T)):
            times = jnp.full(batch_shape, t / T)
            eps = self.skill_agent.network.select('decoder')(
                x, times, observations, skills, params=low_params
            )
            mean = (x - (betas[t] / jnp.sqrt(1.0 - alpha_bar[t])) * eps) / jnp.sqrt(alphas[t])
            if t > 0:
                rng, z_rng = jax.random.split(rng)
                x = mean + jnp.sqrt(betas[t]) * jax.random.normal(z_rng, x.shape)
            else:
                x = mean
        return x

    # ── The composed policy ───────────────────────────────────────────────────

    def _high_dist(self, observations, goals, high_params=None, temperature=1.0):
        """pi_hi(. | s, g): a Categorical over the K skills."""
        return self.high.select('high_actor')(observations, goals, temperature=temperature, params=high_params)

    def _per_skill_log_probs(self, observations, actions, low_params, rng):
        """log pi_lo(a | s, z_k) for every skill. [B, K].

        `jax.vmap` over the skill axis: K forward passes of the low level on the whole
        batch, sharing one rng so the DDS surrogate uses common random numbers.
        """
        skill_vectors = self._skill_vectors(low_params)  # [K, D]
        batch_size = actions.shape[0]

        def one_skill(z_k):
            skills = jnp.broadcast_to(z_k, (batch_size, z_k.shape[-1]))
            return self._low_log_prob(observations, skills, actions, low_params, rng)

        return jax.vmap(one_skill)(skill_vectors).T  # [K, B] -> [B, K]

    def composed_log_prob(self, observations, goals, actions, high_params, low_params, rng):
        """Eq. 2, plus the responsibilities of Eq. 3. Returns (log_pi [B], info)."""
        log_low = self._per_skill_log_probs(observations, actions, low_params, rng)  # [B, K]
        log_high = jax.nn.log_softmax(self._high_dist(observations, goals, high_params).logits, axis=-1)
        joint = log_high + log_low                                                   # [B, K]
        log_pi = jax.nn.logsumexp(joint, axis=-1)                                    # [B]

        # Diagnostics that replace the relabelling statistics of the frozen controllers:
        # `resp_*` describe the posterior over skills given the DATA action (Eq. 3),
        # `high_*` the high level's prior over skills.
        resp = jax.nn.softmax(jax.lax.stop_gradient(joint), axis=-1)
        resp_mean = resp.mean(axis=0)
        high_probs = jnp.exp(jax.lax.stop_gradient(log_high)).mean(axis=0)
        info = {
            'log_pi': log_pi.mean(),
            'log_low_best': log_low.max(axis=-1).mean(),
            'resp_entropy': -(resp * jnp.log(resp + 1e-8)).sum(axis=-1).mean(),
            'resp_max_frac': resp.max(axis=-1).mean(),
            'resp_coverage': (resp_mean > 1e-3).sum(),
            'high_entropy': -(high_probs * jnp.log(high_probs + 1e-8)).sum(),
            'high_max_frac': high_probs.max(),
        }
        return log_pi, info

    # ── Losses ────────────────────────────────────────────────────────────────

    def _q(self, observations, goals, actions):
        """min(Q1, Q2)(s, a, g) from the base agent, never differentiated."""
        q1, q2 = self.base.network.select('critic')(observations, goals, actions)
        return jnp.minimum(q1, q2)

    def _v(self, observations, goals):
        """V(s, g) from the base agent, never differentiated."""
        return self.base.network.select('value')(observations, goals)

    def _actor_batch(self, batch, rng):
        """Optionally subsample the rows the composed actor loss runs on (cost is K x)."""
        sub = self.config['composed_batch_size']
        if sub is None or int(sub) >= batch['observations'].shape[0]:
            return batch
        idxs = jax.random.choice(rng, batch['observations'].shape[0], (int(sub),), replace=False)
        return jax.tree_util.tree_map(lambda x: x[idxs], batch)

    def composed_actor_loss(self, batch, high_params, low_params, rng):
        """AWR or DDPG+BC on the composed policy of Eq. 1."""
        sub_rng, lp_rng, act_rng = jax.random.split(rng, 3)
        batch = self._actor_batch(batch, sub_rng)
        observations = batch['observations']
        goals = batch['actor_goals']
        alpha = self.config['alpha']

        log_pi, info = self.composed_log_prob(
            observations, goals, batch['actions'], high_params, low_params, lp_rng
        )

        if self.config['composed_actor_loss'] == 'awr':
            adv = self._q(observations, goals, batch['actions']) - self._v(observations, goals)
            exp_a = jnp.minimum(jnp.exp(adv * alpha), 100.0)
            actor_loss = -(exp_a * log_pi).mean()
            info.update(actor_loss=actor_loss, adv=adv.mean(), exp_a=exp_a.mean())
            return actor_loss, info

        if self.config['composed_actor_loss'] == 'ddpgbc':
            assert not self.config['discrete'], 'DDPG+BC requires continuous actions.'
            skill_vectors = self._skill_vectors(low_params)
            batch_size = observations.shape[0]

            def q_of_skill(z_k):
                skills = jnp.broadcast_to(z_k, (batch_size, z_k.shape[-1]))
                a_k = self._low_action(observations, skills, low_params, rng=act_rng)
                return self._q(observations, goals, a_k)

            # Eq. 4: the expectation over the discrete skill is exact, so the gradient
            # reaches pi_hi through the probabilities and pi_lo through a_k.
            q_per_skill = jax.vmap(q_of_skill)(skill_vectors).T                  # [B, K]
            probs = self._high_dist(observations, goals, high_params).probs      # [B, K]
            q = (probs * q_per_skill).sum(axis=-1)                               # [B]

            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            bc_loss = -(alpha * log_pi).mean()
            actor_loss = q_loss + bc_loss
            info.update(
                actor_loss=actor_loss,
                q_loss=q_loss,
                bc_loss=bc_loss,
                q_mean=q.mean(),
                q_best_skill=q_per_skill.max(axis=-1).mean(),
            )
            return actor_loss, info

        raise ValueError(f'Unsupported composed_actor_loss: {self.config["composed_actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Validation-path loss (`main.py`): the base agent's loss plus the composed one.

        `grad_params` is accepted for interface compatibility and ignored; the real
        training step uses two parameter trees and lives in `update`.
        """
        del grad_params
        rng = rng if rng is not None else self.rng
        base_rng, actor_rng = jax.random.split(rng)
        base_loss, info = self.base.total_loss(batch, None, rng=base_rng)
        actor_loss, actor_info = self.composed_actor_loss(
            batch, self.high.params, self.skill_agent.network.params, actor_rng
        )
        info = dict(info)
        info.update({f'composed/{k}': v for k, v in actor_info.items()})
        return base_loss + actor_loss, info

    @jax.jit
    def update(self, batch):
        """One step: the flat critic, then one joint gradient over (pi_hi, pi_lo).

        The two levels are differentiated together from a single loss but stepped by two
        optimizers, which is what gives the low level its own learning rate.
        """
        new_rng, actor_rng = jax.random.split(self.rng)

        # 1. Flat critic (and the base agent's own unused flat actor).
        new_base, info = self.base.update(batch)
        info = dict(info)

        # 2. Composed actor, jointly over both parameter trees.
        def loss_fn(params):
            return self.composed_actor_loss(batch, params['high'], params['low'], actor_rng)

        params = {'high': self.high.params, 'low': self.skill_agent.network.params}
        (_, actor_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        new_high = self.high.apply_gradients(grads=grads['high'])
        new_low = self.skill_agent.network.apply_gradients(grads=grads['low'])
        new_skill_agent = self.skill_agent.replace(network=new_low)

        actor_info['grad_norm_high'] = optax.global_norm(grads['high'])
        actor_info['grad_norm_low'] = optax.global_norm(grads['low'])
        info.update({f'composed/{k}': v for k, v in actor_info.items()})

        return self.replace(base=new_base, high=new_high, skill_agent=new_skill_agent, rng=new_rng), info

    # ── Evaluation ────────────────────────────────────────────────────────────
    #
    # The composed policy is flat and stateless, so there is deliberately no
    # `init_eval_state` / `sample_actions_with_state` pair: `utils.evaluation.evaluate`
    # takes the plain `sample_actions` path and the skill is redrawn every step.

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """a ~ pi(.|s,g): draw k ~ pi_hi(.|s,g), then a ~ pi_lo(.|s,z_k). One env step."""
        seed = self.rng if seed is None else seed
        high_seed, low_seed = jax.random.split(seed)

        single_obs = observations.ndim == (3 if self.config['encoder'] is not None else 1)
        obs = observations[None] if single_obs else observations
        goals_b = None if goals is None else (goals[None] if single_obs else goals)

        skill_idxs = self._high_dist(obs, goals_b, temperature=temperature).sample(seed=high_seed)
        skills = self._skill_vectors()[skill_idxs]
        actions = self._low_action(obs, skills, rng=low_seed)

        if single_obs:
            actions = actions[0]
        return actions

    # Skill-conditioned evaluation hooks (`eval_skill_policy.py`), now reporting the
    # FINE-TUNED low level rather than the pretrained one.
    def skill_set(self, seed=None, num_skills=None, observations=None):
        del seed, num_skills, observations
        return self._skill_vectors()

    @jax.jit
    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        seed = self.rng if seed is None else seed
        single_obs = observations.ndim == (3 if self.config['encoder'] is not None else 1)
        obs = observations[None] if single_obs else observations
        skills = skills[None, ...] if skills.ndim == 1 else skills
        skills = jnp.broadcast_to(skills, (obs.shape[0], skills.shape[-1]))
        actions = self._low_action(obs, skills, rng=seed, temperature=temperature)
        return actions[0] if single_obs else actions

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions (PRIMITIVE env actions).
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, high_rng = jax.random.split(rng, 2)

        base_agent_name = config['base_agent_name']
        if base_agent_name not in BASE_AGENTS:
            raise ValueError(f'base_agent_name must be one of {sorted(BASE_AGENTS)}, got {base_agent_name!r}.')
        if set(config['base'].keys()) != set(_base_config(base_agent_name).keys()):
            raise ValueError(
                f"The nested 'base' config does not match {base_agent_name!r}. Select the inner "
                f'algorithm as a config-file argument: '
                f'--agent=agents/composed_skill_policy.py:{base_agent_name}'
            )

        # ── The pretrained low level ──────────────────────────────────────────
        ckpt_path = config['skill_checkpoint_path']
        if not ckpt_path:
            raise ValueError('composed_skill_policy requires --agent.skill_checkpoint_path=<run dir>.')
        with open(os.path.join(ckpt_path, 'flags.json')) as f:
            skill_flags = json.load(f)
        skill_config = skill_flags['agent']
        skill_agent_name = skill_config['agent_name']
        if skill_agent_name not in SKILL_AGENTS:
            raise ValueError(
                f'{skill_agent_name!r} is not a supported low level. Supported: '
                f'{sorted(SKILL_AGENTS)}. `skill_dt` is excluded on purpose: its policy is a '
                f'Transformer over a context window and a future-skill histogram, so '
                f'pi_lo(a|s,z) is not a function of (s, z) alone and the per-step composition '
                f'is not well defined.'
            )
        if skill_agent_name == 'opal' and skill_config['latent_type'] != 'discrete':
            raise ValueError(
                "OPAL with latent_type='continuous' has no finite skill set, so the composed "
                'policy is an integral rather than the K-term sum this agent evaluates exactly. '
                "Use an OPAL run trained with latent_type='discrete'."
            )
        for key in ('encoder', 'frame_stack', 'discrete'):
            if skill_config.get(key) != config[key]:
                raise ValueError(
                    f'{key}={config[key]!r} does not match the pretrained low level '
                    f'({skill_config.get(key)!r}).'
                )

        num_skills = int(skill_config['num_skills'])
        if config['num_skills'] is not None and int(config['num_skills']) != num_skills:
            raise ValueError(
                f"num_skills={config['num_skills']} does not match the checkpoint's {num_skills}."
            )

        skill_agent = SKILL_AGENTS[skill_agent_name].create(
            seed, ex_observations, ex_actions, ml_collections.ConfigDict(skill_config)
        )
        restore_epoch = config['skill_restore_epoch'] or _latest_epoch(ckpt_path)
        skill_agent = restore_agent(skill_agent, ckpt_path, restore_epoch)

        # Give the low level its OWN optimizer at `low_lr`. The pretrained one is
        # discarded deliberately: `dds` ships a phased optimizer that hard-freezes the
        # decoder after pretraining and would zero every update made here.
        low_net = skill_agent.network
        low_net = TrainState.create(
            low_net.model_def, low_net.params, tx=optax.adam(learning_rate=config['low_lr'])
        )
        skill_agent = skill_agent.replace(network=low_net)

        low_logprob = config['low_logprob']
        if low_logprob is None:
            low_logprob = 'diffusion' if (skill_agent_name == 'dds' and not config['discrete']) else 'exact'
        if low_logprob == 'exact' and skill_agent_name == 'dds' and not config['discrete']:
            raise ValueError(
                "DDS's continuous decoder is a diffusion epsilon-network with no closed-form "
                "density. Set --agent.low_logprob=diffusion to use the scaled negative-ELBO "
                'surrogate (and tune --agent.diffusion_logprob_scale).'
            )

        # ── The high level ────────────────────────────────────────────────────
        encoders = dict()
        if config['encoder'] is not None:
            encoders['high_actor'] = GCEncoder(concat_encoder=encoder_modules[config['encoder']]())
        high_actor_def = GCDiscreteActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=num_skills,
            gc_encoder=encoders.get('high_actor'),
        )
        high_def = ModuleDict(dict(high_actor=high_actor_def))
        high_params = high_def.init(high_rng, high_actor=(ex_observations, ex_observations))['params']
        high = TrainState.create(high_def, high_params, tx=optax.adam(learning_rate=config['lr']))

        # ── The flat critic ───────────────────────────────────────────────────
        raw_base = config['base']
        base_config = ml_collections.ConfigDict(raw_base.to_dict() if hasattr(raw_base, 'to_dict') else dict(raw_base))
        # The critic acts on PRIMITIVE actions, so its action space is the environment's.
        base_config['discrete'] = config['discrete']
        base_config['batch_size'] = config['batch_size']
        base_config['encoder'] = config['encoder']
        base_config['frame_stack'] = config['frame_stack']
        base_agent = BASE_AGENTS[base_agent_name].create(seed, ex_observations, ex_actions, base_config)

        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        stored_config['num_skills'] = num_skills
        stored_config['skill_agent_name'] = skill_agent_name
        stored_config['skill_restore_epoch'] = int(restore_epoch)
        stored_config['skill_checkpoint_path'] = ckpt_path
        stored_config['low_logprob'] = low_logprob

        return cls(
            rng,
            base=base_agent,
            high=high,
            skill_agent=skill_agent,
            config=flax.core.FrozenDict(**stored_config),
        )


def get_config(base_agent_name='gciql'):
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='composed_skill_policy',
            lr=3e-4,  # Learning rate of the HIGH level pi_hi(k|s,g).
            low_lr=3e-5,  # Learning rate of the LOW level pi_lo(a|s,z). 0.0 == frozen baseline.
            batch_size=1024,  # Batch size (the critic always sees all of it).
            composed_batch_size=ml_collections.config_dict.placeholder(int),  # Rows for the composed
            # actor loss (None = all). Each row costs K low-level forward passes.
            actor_hidden_dims=(512, 512, 512),  # High-level actor hidden dimensions.
            composed_actor_loss='awr',  # Composed actor loss ('awr' or 'ddpgbc').
            alpha=3.0,  # AWR temperature, or the BC coefficient in DDPG+BC.
            low_temperature=0.0,  # Sampling temperature of the low level at eval (0 = mode).
            discrete=False,  # Whether the ENVIRONMENT's action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
            # Pretrained low level.
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),  # Required: run dir.
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),  # None = latest.
            skill_agent_name=ml_collections.config_dict.placeholder(str),  # Read from flags.json.
            num_skills=ml_collections.config_dict.placeholder(int),  # Read from flags.json.
            low_logprob=ml_collections.config_dict.placeholder(str),  # 'exact' | 'diffusion'
            # (None = 'diffusion' for continuous-action DDS, 'exact' otherwise).
            diffusion_logprob_scale=1.0,  # c in log pi_lo ~ -c * ||eps - eps_psi||^2 (DDS only).
            # Flat critic.
            base_agent_name=base_agent_name,
            base=_base_config(base_agent_name),
            # Dataset hyperparameters. Flat per-step transitions: no windows, no horizon.
            dataset_class='GCDataset',
            discount=0.99,  # Per-step discount (also drives GCDataset geometric goal sampling).
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
        )
    )
    return config
