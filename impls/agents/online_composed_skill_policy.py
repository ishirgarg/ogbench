"""Online composed (non-frozen) hierarchical policy: pi_hi picks a skill, pi_lo acts, both train.

The online sibling of `agents/composed_skill_policy.py` (which does the same thing offline),
and the non-frozen sibling of `agents/online_crl_skill_controller.py`. There is NO skill
horizon and no semi-MDP: the high level redraws a skill at every environment step and the two
levels are simply composed into one flat policy over primitive actions,

    pi(a | s, g)  =  sum_{k=1..K}  pi_hi(k | s, g) * pi_lo(a | s, z_k),              (1)

which is what online RL improves. Rows in the replay buffer are env steps (`rollout_type='flat'`,
the `FlatCollector`), so RLPD (`--offline_dataset`) needs no window labelling: offline rows are
(s_t, a_t) exactly as for `online_crl`.

Why the low level can be trained at all
---------------------------------------
Every `*_controller` agent here freezes pi_lo because its critic is Q(s, z, g) -- a function of
the skill INDEX, with no functional dependence on the low-level parameters, hence no gradient for
them. Here the critic is flat, Q(s, a, g) over primitive actions (the contrastive CRL critic of
`agents/online_crl.py`, unchanged), so the actor loss differentiates through the action pi_lo
emits and reaches its parameters.

The actor loss, and why the skill is enumerated rather than sampled
-------------------------------------------------------------------
Acting draws an integer k ~ pi_hi(.|s,g) and then a ~ pi_lo(.|s,z_k). The action is
reparameterisable, so d/d(theta_lo) flows through it by ordinary backprop; the SKILL is not.
Nothing connects pi_hi's logits to the integer k continuously (dk/dlogits = 0 almost everywhere),
so autodiff through a sampled skill returns exactly zero gradient for the high level -- the usual
reason discrete latents need REINFORCE or a Gumbel relaxation.

They are not needed here, because K is finite: the expectation over k is a K-term sum, and a sum
IS differentiable. With the SAC objective of `online_crl` (alpha * log pi - Q) applied to the
skill,

    J = E_s sum_k pi_hi(k | s, g) * [ alpha * log pi_hi(k | s, g) - Q(s, a_k, g) ],  (2)
    a_k = the action pi_lo(. | s, z_k) emits (its mode at low_temperature=0).        (3)

`pi_hi(k|s,g)` appears as a plain multiplicative factor, so d/d(theta_hi) is backprop through the
softmax -- exact, unbiased, zero variance -- and d/d(theta_lo) is backprop through each a_k.
Eq. 2 costs K low-level forward passes and K critic evaluations per row (`actor_batch_size`
subsamples the rows if that ever matters; the critic loss always sees the full batch).

Eq. 2 is deliberately the same objective the frozen controller already optimises
(`online_crl_skill_controller.actor_loss`), with its skill-level Q(s, z, g) replaced by
Q(s, a_k(z), g). With `low_lr=0.0` (Adam at lr 0 is a no-op) this agent therefore reduces to that
controller at skill_commitment_k=1, which is the ablation it exists to beat.

`composed_grad_method`: two estimators of the same gradient, and one relaxation
-------------------------------------------------------------------------------
Splitting Eq. 2 into its two halves,

    J = -alpha * H(pi_hi)  -  E_{k~pi_hi}[ Q(s, a_k, g) ],                           (5)

the entropy half is analytic under every method, so the flag dispatches ONLY the second half:

  'enumerate' (default)  The sum above, evaluated over all K skills. Exact, unbiased, zero
                         variance; K low-level forward passes and K critic evaluations per row.

  'reinforce'            One skill k ~ pi_hi(.|s,g) is sampled per row and handed to the low
                         level, which emits a single action a = pi_lo(.|s,z_k). That action is
                         reparameterised, so the LOW level takes its ordinary pathwise gradient,

                             L_lo = -Q(s, a, g),                                      (6)

                         while the HIGH level -- which has no path to an integer k -- takes a
                         score-function gradient with the low level's own loss as a NEGATIVE
                         reward, R = -L_lo = Q(s, a, g):

                             L_hi = -sg[R - b] * log pi_hi(k | s, g).                 (7)

                         Eqs. 6 and 7 depend on disjoint parameters (everything in Eq. 7 but
                         log pi_hi is stop-gradiented), so summing them double-counts nothing.
                         Unbiased for the same Eq. 5, since grad E_k[f] = E_k[f grad log pi_k],
                         at 1 low-level pass per row instead of K -- paid for in variance.
                         `reinforce_baseline='batch'` subtracts the leave-one-out batch mean of
                         Q (exactly unbiased, and it strips the arbitrary global scale a
                         contrastive critic's Q carries); 'none' subtracts nothing. A per-STATE
                         baseline would need all K critic evaluations, i.e. exactly the cost
                         this method exists to avoid.

  'softmax'              NOT an estimator of Eq. 5 but a relaxation of the POLICY: the low level
                         is conditioned on the full probability vector p = softmax(logits(s,g))
                         instead of a one-hot,

                             pi(a | s, g) = pi_lo(a | s, p(s, g)),                    (8)

                         so the composed policy is a single low-level pass whose skill input is
                         continuous, and BOTH levels take an ordinary pathwise gradient of
                         -Q(s, a, g) through a = pi_lo(s, p): the high level through p (backprop
                         through the softmax), the low level through a. Exact for Eq. 8, zero
                         variance, 1 low-level pass per row -- but pi_lo was pretrained on
                         one-hots only, so interior points of the simplex are off-distribution
                         for it until it adapts (low_lr > 0), and Eq. 8 is a different policy
                         class from Eq. 1 (it can express a = pi_lo(s, 0.5 z_1 + 0.5 z_2), which
                         no mixture of one-hot skills can). The entropy term is kept, on the same
                         categorical, and acts as a pull of p toward uniform. Requires
                         skill_commitment_k == 1 (asserted): with a horizon the vector p would
                         have to be frozen across steps, which nothing here implements.

Acting is IDENTICAL under 'enumerate' and 'reinforce': `sample_actions` draws one k and feeds
one pure one-hot to the low level; the flag changes only how the gradient of Eq. 5 is estimated,
never the policy. Under 'softmax' acting follows Eq. 8: the low level is fed
p = softmax(logits / temperature) (temperature=0, the evaluator's, makes p the argmax one-hot, so
eval is a plain committed skill). NOTE this makes the collector's policy DETERMINISTIC at the
default low_temperature=0 -- the categorical is never sampled, so no exploration noise enters
anywhere; set low_temperature > 0 if you want any. The recorded eval "skill" is argmax p.
`actor/objective` logs Eq. 5 (Eq. 8's -Q for 'softmax') on a common scale.

`learned_action_std` (softmax mode only): SAC-style action noise from the high level
--------------------------------------------------------------------------------------
Under 'softmax' nothing is ever sampled, so at low_temperature=0 the collector is deterministic.
With `learned_action_std=True` the high level grows a second head, a state-and-goal-conditioned
log-std `action_log_std(s, g)` (clipped to [action_log_std_min, action_log_std_max], the range
online_crl uses), and the composed action becomes the tanh-squashed Gaussian of online_crl/SAC
centred on the low level's action:

    a = tanh( atanh(clip(mu_lo(s, p(s,g)))) + exp(log_std_hi(s,g)) * eps ),  eps ~ N(0, I).  (9)

At zero noise a = clip(mu_lo), i.e. the deterministic softmax policy EXACTLY (to the 1e-5
clip margin), so the pretrained mean mapping is preserved; the atanh is straight-through in the
backward pass (value atanh(clip(mu)), gradient 1), so its 1/(1-mu^2) blow-up near the box never
reaches the low level. The constant unit std the pretrained low level shipped with is simply not
used (low_temperature must be 0). Eq. 9 is reparameterised, so the same pathwise gradient of -Q
reaches p, mu_lo and now log_std_hi.

The entropy is that of the SQUASHED distribution, estimated as online_crl does (closed-form
Gaussian part + the sampled tanh log-det, sum_i log(1 - a_i^2)). This matters: the pre-squash
Gaussian entropy is unbounded in the std, so an alpha term on it drives log_std into its clip
where the gradient is zero and it can never come back (observed: std 1 -> 5 in five updates
against a fresh critic). The squashed entropy peaks at a finite std (the uniform on the box, A
log 2) and falls beyond it, so the bonus itself stops the runaway. A second alpha,
`action_alpha`, is tuned toward `action_target_entropy` (default -0.5 * A, online_crl's target).
The categorical entropy term on p and its alpha stay exactly as they are, so
`target_entropy_frac` is now a REGULARISER on how soft p is rather than the exploration knob.
Acting samples Eq. 9 at the high level's temperature (the collector's 1 scales the std, the
evaluator's 0 zeroes it and takes the argmax one-hot).

Entropy
-------
One alpha, auto-tuned on the CATEGORICAL entropy H(pi_hi) toward
`target_entropy_frac * log(num_skills)` -- identical to the controller, and reachable by
construction for frac <= 1. There is no action-level entropy term: exploration at the action level
is whatever pi_lo's own sampling provides, and `low_temperature=0.0` (the controller's default,
i.e. pi_lo acts at its mode) makes the composed policy deterministic given k, so ALL exploration
comes from the categorical. This also sidesteps a term that would be vacuous for the checkpoints
here anyway -- `empowerment_skill` runs with const_std=True, whose entropy is a constant with no
gradient.

Separate learning rates
-----------------------
The high level is fresh and takes the normal `lr`; the low level is pretrained and must not be
destroyed, so it gets its own optimizer at `low_lr` (default 3e-5 = lr/10). They are two
`TrainState`s -- `network` (critic + high level + alpha) and `skill_agent.network` -- updated from
ONE `jax.value_and_grad` over the pair, so the composition is differentiated jointly and only the
step sizes differ. The pretrained agent's own optimizer is discarded and replaced at `create`
time (its Adam state is stale, and some families ship phased optimizers that freeze modules).
Only the low level's action module receives gradient; its other modules sit in the same parameter
tree, get exactly zero gradient, and therefore never move.

Supported low levels (`skill_agent_name` is read from the checkpoint's flags.json)
---------------------------------------------------------------------------------
  empowerment_skill   pi_lo = `policy`(s, one_hot(k))            Gaussian, differentiable mode.
  opal (discrete)     pi_lo = `decoder`(concat[s, one_hot(k)])   Gaussian, differentiable mode.
  dds                 refused: its continuous decoder is a 5-step DDPM epsilon-network, so a_k
                      only exists as a differentiable unroll of the sampler (5x the forward cost
                      per skill). `composed_skill_policy` does that offline; add the same
                      `_ddpm_sample` here if you want it online.
  opal (continuous)   refused: a continuous latent makes Eq. 1 an integral, not a K-term sum.
  skill_dt            refused: its policy is a causal Transformer over a context window plus a
                      future-skill histogram, so pi_lo(a|s,z) is not a function of (s, z) alone
                      and the per-step composition is not even well defined.
"""

import json
import os
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.empowerment_skill import EmpowermentAgent
from agents.opal import OPALAgent
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCBilinearValue, GCDiscreteActor, LogParam, default_init
from utils.skill_checkpoint import load_frozen_skill_agent

# How the expectation over the discrete skill is turned into a gradient (see `actor_loss`).
COMPOSED_GRAD_METHODS = ('enumerate', 'reinforce', 'softmax')
REINFORCE_BASELINES = ('batch', 'none')

# Pretrained families with a finite skill set AND a low level that is a plain function of
# (s, z) with a differentiable action. See the module docstring for the ones that are refused.
SKILL_AGENT_CLASSES = dict(empowerment_skill=EmpowermentAgent, opal=OPALAgent)

# Rejected families, with the reason (raised before the checkpoint is rebuilt, so the message
# explains the modelling obstacle rather than a missing dict key).
UNSUPPORTED_SKILL_AGENTS = {
    'dds': (
        "DDS's continuous decoder is a 5-step DDPM epsilon-network, so a_k = pi_lo(.|s,z_k) exists only "
        'as a differentiable unroll of the sampler (5x the forward cost per skill, 250 decoder passes '
        'per row at K=50). agents/composed_skill_policy.py implements exactly that offline '
        '(`_ddpm_sample`); port it here if you want DDS online.'
    ),
    'skill_dt': (
        "Skill-DT's policy is a causal Transformer over a K-step context plus a future-skill histogram, "
        'so pi_lo(a | s, z) is not a function of (s, z) alone and the per-step composition of Eq. 1 is '
        'not well defined. Use agents/online_crl_skill_controller.py, which executes it as an option.'
    ),
}


class GCActionLogStd(nn.Module):
    """State-and-goal-conditioned per-dimension action log-std head (Eq. 9): an MLP over (s, g).

    Mirrors `GCDiscreteActor`'s input handling (optional `gc_encoder` for pixels, otherwise a
    plain concat) and the small-scale final init `GCActor` uses for its std head, so the noise
    starts near exp(0) = 1 and moves from there.
    """

    hidden_dims: Any
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    final_fc_init_scale: float = 1e-2
    gc_encoder: Any = None
    activations: Any = nn.gelu

    def setup(self):
        self.trunk = MLP(self.hidden_dims, activate_final=True, activations=self.activations)
        self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(self, observations, goals=None, goal_encoded=False):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        log_stds = self.log_std_net(self.trunk(inputs))
        return jnp.clip(log_stds, self.log_std_min, self.log_std_max)


class OnlineComposedSkillPolicyAgent(flax.struct.PyTreeNode):
    """Flat online agent whose policy is pi(a|s,g) = sum_k pi_hi(k|s,g) pi_lo(a|s,z_k), both trained.

    Fields:
        rng: PRNG key.
        network: `TrainState` over a `ModuleDict` with `critic` (contrastive `GCBilinearValue` over
            PRIMITIVE actions), `actor` (the high level pi_hi, a K-way `GCDiscreteActor`) and
            `alpha` (`LogParam`). One Adam at `lr`.
        skill_agent: The pretrained low level. Unlike every `*_controller`, this is TRAINED: its
            `network` TrainState carries a fresh Adam at `low_lr` and receives gradient from the
            actor loss through a_k.
        config: Static configuration dictionary.
    """

    rng: Any
    network: Any
    skill_agent: Any
    config: Any = nonpytree_field()

    # ── The low level ─────────────────────────────────────────────────────────
    #
    # `low_params` is either None (use the stored params, no gradient) or the traced parameter
    # tree of `skill_agent.network`, exactly as `TrainState.__call__` expects.

    def _skill_vectors(self):
        """The K conditioning vectors {z_k} handed to the low level: the one-hots. [K, K]."""
        return jnp.eye(int(self.config['num_skills']))

    def _low_dist(self, observations, skills, low_params=None, temperature=1.0):
        """pi_lo(. | s, z) as a distrax distribution."""
        name = self.config['skill_agent_name']
        net = self.skill_agent.network
        if name == 'empowerment_skill':
            return net.select('policy')(observations, skills, temperature=temperature, params=low_params)
        if name == 'opal':
            inputs = jnp.concatenate([observations, skills], axis=-1)
            return net.select('decoder')(inputs, temperature, params=low_params)
        raise ValueError(f'Unsupported skill_agent_name: {name}')

    def _low_action(self, observations, skills, low_params=None, rng=None):
        """The action pi_lo(.|s,z) emits, differentiable in `low_params` and clipped to the env's box.

        At `low_temperature=0` (the default, and what every frozen controller does) this is the
        distribution's mode, i.e. the mean network's output -- a smooth function of the low-level
        parameters, which is what carries the deterministic policy gradient of Eq. 2/3.
        """
        temperature = float(self.config['low_temperature'])
        if temperature == 0.0:
            # Build the distribution at temperature 1 and take its mode rather than handing a
            # zero scale to distrax; identical action, no degenerate distribution.
            actions = self._low_dist(observations, skills, low_params, temperature=1.0).mode()
        else:
            actions = self._low_dist(observations, skills, low_params, temperature=temperature).sample(seed=rng)
        return jnp.clip(actions, -1.0, 1.0)

    # Margin keeping atanh finite: the deterministic action is clip(mu) to within this.
    _SQUASH_EPS = 1e-5

    def _noisy_low_action(self, observations, goals, skills, low_params, rng, temperature=1.0, grad_params=None):
        """Eq. 9: tanh(atanh(clip(mu_lo(s, z))) + temperature * exp(log_std_hi(s, g)) * eps).

        `learned_action_std` only. Returns (action, entropy_estimate [B]) where the estimate is
        the squashed distribution's -log pi(a) with the Gaussian part in closed form:
        sum_i log_std_i + A/2 (1 + log 2 pi) + sum_i log(1 - a_i^2). Its expectation is the
        squashed entropy and its reparameterised gradient is that entropy's gradient.
        """
        mean = self._low_dist(observations, skills, low_params, temperature=1.0).mode()
        mean = jnp.clip(mean, -1.0 + self._SQUASH_EPS, 1.0 - self._SQUASH_EPS)
        # Straight-through atanh: forward value atanh(mean), backward gradient 1. Together with
        # tanh's (1 - a^2) on the way out, the low level sees d a / d mu_lo = 1 - a^2 <= 1,
        # i.e. the soft version of clip's gradient, never the 1/(1 - mu^2) of atanh.
        pre = mean + jax.lax.stop_gradient(jnp.arctanh(mean) - mean)
        log_std = self.network.select('action_log_std')(observations, goals, params=grad_params)
        eps = jax.random.normal(rng, mean.shape)
        u = pre + temperature * jnp.exp(log_std) * eps
        actions = jnp.tanh(u)
        action_dim = mean.shape[-1]
        # log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u)), the numerically stable form.
        log_det = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
        gauss_entropy = jnp.sum(log_std, axis=-1) + 0.5 * action_dim * (1.0 + jnp.log(2.0 * jnp.pi))
        entropy_est = gauss_entropy + jnp.sum(log_det, axis=-1)
        return actions, entropy_est, log_std

    def _per_skill_actions(self, observations, low_params, rng):
        """a_k for every skill: [K, B, action_dim]. `jax.vmap` over the skill axis."""
        skill_vectors = self._skill_vectors()
        batch_size = observations.shape[0]

        def one_skill(z_k):
            skills = jnp.broadcast_to(z_k, (batch_size, z_k.shape[-1]))
            # One shared rng across the K skills: common random numbers, so the K candidate
            # actions compared inside Eq. 2 differ by the skill, not by the noise draw.
            return self._low_action(observations, skills, low_params, rng=rng)

        return jax.vmap(one_skill)(skill_vectors)

    # ── Losses ────────────────────────────────────────────────────────────────

    def contrastive_loss(self, batch, grad_params):
        """In-batch contrastive critic loss over (s, a) vs. future goals; verbatim `agents/online_crl.py`."""
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
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

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

    def _actor_batch(self, batch, rng):
        """Optionally subsample the rows the actor loss runs on (its cost is K x the critic's)."""
        sub = self.config['actor_batch_size']
        if sub is None or int(sub) >= batch['observations'].shape[0]:
            return batch
        idxs = jax.random.choice(rng, batch['observations'].shape[0], (int(sub),), replace=False)
        return jax.tree_util.tree_map(lambda x: x[idxs], batch)

    def actor_loss(self, batch, grad_params, low_params, rng):
        """Eq. 2, split so the two `composed_grad_method`s differ in exactly one term.

        Writing Eq. 2 out,

            J = E_s sum_k pi_hi(k|s,g) * [ alpha * log pi_hi(k|s,g) - Q(s, a_k, g) ]
              = -alpha * H(pi_hi)  -  E_{k~pi_hi}[ Q(s, a_k, g) ],                   (5)

        the entropy term is analytic and identical under both methods (the full logits are
        in hand either way), so ONLY the second term -- how the Q signal reaches the two
        levels -- is dispatched. Both branches are unbiased estimators of the gradient of
        the SAME objective, Eq. 5, and `actor/objective` reports it on a common scale.
        """
        sub_rng, act_rng = jax.random.split(rng)
        batch = self._actor_batch(batch, sub_rng)
        observations = batch['observations']
        goals = batch['actor_goals']

        dist = self.network.select('actor')(observations, goals, params=grad_params)
        log_pi = jax.nn.log_softmax(dist.logits, axis=-1)  # (B, K)
        pi = jnp.exp(log_pi)
        entropy = -jnp.sum(pi * log_pi, axis=-1)  # (B,)  H(pi_hi)

        # alpha is read at its STORED value: the actor loss never trains it, `alpha_loss` does.
        alpha = self.network.select('alpha')()

        method = self.config['composed_grad_method']
        if method == 'enumerate':
            q_loss, info = self._q_loss_enumerate(observations, goals, pi, alpha, log_pi, low_params, act_rng)
        elif method == 'reinforce':
            q_loss, info = self._q_loss_reinforce(observations, goals, log_pi, alpha, low_params, act_rng)
        else:
            q_loss, info = self._q_loss_softmax(observations, goals, pi, low_params, act_rng, grad_params)

        # -alpha * H(pi_hi): the entropy half of Eq. 5. Gradient reaches only the high level.
        entropy_loss = -(alpha * entropy).mean()
        actor_loss = q_loss + entropy_loss

        alpha_param = self.network.select('alpha')(params=grad_params)
        entropy_sg = jax.lax.stop_gradient(entropy).mean()
        target_entropy = self.config['target_entropy']
        alpha_loss = alpha_param * (entropy_sg - target_entropy)

        if self.config['learned_action_std']:
            # SAC on the action Gaussian of Eq. 9: -action_alpha * H_a, its own alpha tuned
            # toward action_target_entropy. The categorical term above is untouched.
            action_entropy = info.pop('action_entropy_rows')  # (B,), carries gradient to the std head
            action_alpha = self.network.select('action_alpha')()
            action_alpha_param = self.network.select('action_alpha')(params=grad_params)
            action_entropy_loss = -(action_alpha * action_entropy).mean()
            action_entropy_sg = jax.lax.stop_gradient(action_entropy).mean()
            action_alpha_loss = action_alpha_param * (action_entropy_sg - self.config['action_target_entropy'])
            actor_loss = actor_loss + action_entropy_loss
            alpha_loss = alpha_loss + action_alpha_loss
            info.update(
                action_entropy=action_entropy_sg,
                action_entropy_loss=action_entropy_loss,
                action_alpha=action_alpha,
                action_alpha_loss=action_alpha_loss,
                action_target_entropy=self.config['action_target_entropy'],
            )

        total_loss = actor_loss + alpha_loss
        info.update(
            total_loss=total_loss,
            actor_loss=actor_loss,
            entropy_loss=entropy_loss,
            alpha_loss=alpha_loss,
            alpha=alpha,
            entropy=entropy_sg,
            target_entropy=target_entropy,
            pi_max_mean=jnp.mean(jnp.max(pi, axis=-1)),
        )
        # Eq. 5 itself, on the same scale under every method (exact under 'enumerate', the
        # one-sample estimate under 'reinforce', Eq. 8's -Q under 'softmax'), so learning curves
        # are comparable.
        info['objective'] = jax.lax.stop_gradient(info.pop('q_term') - (alpha * entropy).mean())
        return total_loss, info

    def _low_action_stats(self, actions):
        """Diagnostics shared by both branches. `jnp.clip` zeroes the gradient of a saturated
        action dimension, so watch `low_action_sat_frac`: a low level whose mode sits outside
        the box cannot be improved in those dimensions."""
        raw = jax.lax.stop_gradient(actions)
        return {
            'low_action_abs_mean': jnp.abs(raw).mean(),
            'low_action_sat_frac': jnp.mean((jnp.abs(raw) >= 1.0 - 1e-6).astype(jnp.float32)),
        }

    def _q_loss_enumerate(self, observations, goals, pi, alpha, log_pi, low_params, rng):
        """-E_{k~pi_hi}[Q(s, a_k, g)] as an EXACT K-term sum (the default).

        The skill is never sampled: every k is enumerated, so the expectation is a plain
        differentiable sum. d/d(theta_hi) is backprop through the softmax weights (exact,
        unbiased, zero variance) and d/d(theta_lo) is backprop through each a_k, weighted by
        the responsibility pi_hi(k|s,g). Costs K low-level forward passes and K critic
        evaluations per row.
        """
        del alpha
        num_skills = int(self.config['num_skills'])

        # a_k for every skill, then Q(s, a_k, g). The critic is read at its STORED parameters
        # (no `params=grad_params`), so the actor loss has no gradient w.r.t. the critic -- but
        # the gradient w.r.t. the ACTIONS still flows, which is what reaches the low level.
        actions = self._per_skill_actions(observations, low_params, rng)  # (K, B, A)
        obs_b = jnp.broadcast_to(observations[None], (num_skills, *observations.shape))
        # The goals keep a leading axis of 1 rather than K: psi(g) does not depend on the skill, so
        # this evaluates the critic's goal tower ONCE and lets it broadcast against phi's K rows,
        # halving the actor loss's critic cost. (phi: (E, K, B, d); psi: (E, 1, B, d).)
        goals_b = goals[None]
        qs = self.network.select('critic')(obs_b, goals_b, actions=actions)  # (E, K, B)
        q = qs[0].T  # (B, K); head 0, as in online_crl / the frozen controller.

        q_loss = -jnp.sum(pi * q, axis=-1).mean()
        info = {
            'q_term': q_loss,
            'q_pi_mean': jnp.sum(pi * q, axis=-1).mean(),
            'q_max_skill_mean': q.max(axis=-1).mean(),
            'q_skill_spread': (q.max(axis=-1) - q.min(axis=-1)).mean(),
        }
        info.update(self._low_action_stats(actions))
        return q_loss, info

    def _q_loss_reinforce(self, observations, goals, log_pi, alpha, low_params, rng):
        """-E_{k~pi_hi}[Q(s, a_k, g)] from ONE sampled skill per row: REINFORCE up, pathwise down.

        A single k ~ pi_hi(.|s,g) is drawn and handed to the low level, which emits one action
        a = pi_lo(.|s,z_k). That action is reparameterised, so the LOW level takes its ordinary
        pathwise gradient, exactly as in the enumerate branch:

            L_lo = -Q(s, a, g),            d/d(theta_lo) flows through a.                  (6)

        The HIGH level cannot: k is an integer, so there is no path from the logits to it. It
        gets a score-function (REINFORCE) gradient instead, with the low level's own loss as a
        negative reward -- R = -L_lo = Q(s, a, g):

            L_hi = -sg[ R - b ] * log pi_hi(k | s, g),                                     (7)

        whose gradient is -E[(R - b) grad log pi_hi], i.e. gradient ASCENT on the reward. Eqs. 6
        and 7 have disjoint parameter dependences (everything in Eq. 7 that is not log pi_hi is
        stop-gradiented), so summing them double-counts nothing: each is the correct estimator
        of its own half of d/d(theta) of -E_k[Q].

        Unbiased for the same objective as 'enumerate', because
        grad E_k[f(k)] = E_k[f(k) grad log pi_k], but with variance where the enumerate branch
        has none -- that is the whole trade, and it buys 1 low-level forward pass per row
        instead of K.

        `b` is `reinforce_baseline`: 'batch' is the LEAVE-ONE-OUT batch mean of Q (exactly
        unbiased, and it removes the arbitrary global scale a contrastive critic's Q carries),
        'none' is b = 0. A per-STATE baseline (E_k[Q] under pi_hi) is deliberately not offered:
        computing it needs all K critic evaluations, which is precisely the cost this method
        exists to avoid.
        """
        skill_rng, act_rng = jax.random.split(rng)
        batch_size = observations.shape[0]

        # Sample the skill from the SAME categorical the loss differentiates (stop-gradient is
        # implicit -- an integer carries no gradient), then act with that one pure one-hot.
        skills = jax.random.categorical(skill_rng, jax.lax.stop_gradient(log_pi), axis=-1)  # (B,)
        log_pi_k = jnp.take_along_axis(log_pi, skills[:, None], axis=-1)[:, 0]  # (B,)
        z = self._skill_vectors()[skills]  # (B, K) one-hots

        actions = self._low_action(observations, z, low_params, rng=act_rng)  # (B, A)
        q = self.network.select('critic')(observations, goals, actions=actions)[0]  # (B,)

        # Eq. 6: the low level's ordinary pathwise loss. Gradient flows through `actions`.
        low_loss = -q.mean()

        # Eq. 7: REINFORCE for the high level, reward = -low_loss = Q.
        reward = jax.lax.stop_gradient(q)
        if self.config['reinforce_baseline'] == 'batch':
            # Leave-one-out mean: b_i = (sum_j R_j - R_i) / (B - 1), independent of R_i and so
            # exactly unbiased (a plain batch mean would carry an O(1/B) bias).
            baseline = (reward.sum() - reward) / jnp.maximum(batch_size - 1, 1)
        else:
            baseline = jnp.zeros_like(reward)
        advantage = reward - baseline
        high_loss = -(advantage * log_pi_k).mean()

        skill_fracs = jnp.bincount(skills, length=int(self.config['num_skills'])) / batch_size

        q_loss = low_loss + high_loss
        info = {
            # The objective's Q term, NOT the surrogate: `q_loss` above mixes a score-function
            # surrogate (whose value is meaningless) with Eq. 6, so report the estimand.
            'q_term': jax.lax.stop_gradient(low_loss),
            'q_pi_mean': reward.mean(),
            'reinforce_loss': high_loss,
            'low_loss': low_loss,
            'adv_mean': advantage.mean(),
            'adv_std': advantage.std(),
            'log_pi_sampled': log_pi_k.mean(),
            # Empirical skill diversity of the batch's draws (the enumerate branch has no
            # sampled skills, so this key is specific to this method).
            'skill_sample_entropy': -(skill_fracs * jnp.log(skill_fracs + 1e-8)).sum(),
        }
        info.update(self._low_action_stats(actions))
        return q_loss, info

    def _q_loss_softmax(self, observations, goals, pi, low_params, rng, grad_params=None):
        """-Q(s, pi_lo(s, p), g) with p = pi_hi(.|s,g) fed to the low level as a VECTOR (Eq. 8).

        No skill is sampled and none is enumerated: the probability vector itself is the
        low level's conditioning input, in place of the one-hot z_k. The action is then a
        smooth function of BOTH parameter trees, so a single pathwise gradient reaches the
        high level (through the softmax, via p) and the low level (through a). One low-level
        forward pass and one critic evaluation per row.

        This is a relaxation of the policy, not an estimator of Eq. 5's gradient: it
        optimises Eq. 8, whose value -Q(s, a(p), g) is reported as `q_term` so the objective
        curve stays on the same scale as the other two methods.
        """
        # `pi` carries the gradient to the high level; nothing is stop-gradiented here.
        info = {}
        if self.config['learned_action_std']:
            # Eq. 9: the high level's std head adds reparameterised noise to the low level's mean.
            actions, entropy_rows, log_std = self._noisy_low_action(
                observations, goals, pi, low_params, rng, grad_params=grad_params
            )
            info['action_entropy_rows'] = entropy_rows
            info['action_std_mean'] = jnp.exp(jax.lax.stop_gradient(log_std)).mean()
            info['action_log_std_mean'] = jax.lax.stop_gradient(log_std).mean()
        else:
            actions = self._low_action(observations, pi, low_params, rng=rng)  # (B, A)
        q = self.network.select('critic')(observations, goals, actions=actions)[0]  # (B,)

        q_loss = -q.mean()
        info.update({
            'q_term': q_loss,
            'q_pi_mean': q.mean(),
            # How far the conditioning vector sits from the one-hot vertices pi_lo was trained
            # on: 1 at a vertex, 1/K at the centroid (`pi_max_mean` in actor_loss is the same
            # number; the L2 norm is the complementary view).
            'skill_vec_l2_mean': jnp.linalg.norm(jax.lax.stop_gradient(pi), axis=-1).mean(),
        })
        info.update(self._low_action_stats(actions))
        return q_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, low_params, rng=None):
        """Critic loss (full batch) + composed actor loss (over both parameter trees)."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, low_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """One step: a single joint gradient over (critic + pi_hi + alpha, pi_lo), two optimizers.

        The two parameter trees are differentiated together from one loss and stepped separately,
        which is what gives the low level its own learning rate. `low_lr=0.0` makes the second
        step a no-op (Adam at lr 0), recovering the frozen-low-level baseline exactly.
        """
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params['main'], params['low'], rng=rng)

        params = {'main': self.network.params, 'low': self.skill_agent.network.params}
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        new_network = self.network.apply_gradients(grads=grads['main'])
        new_low = self.skill_agent.network.apply_gradients(grads=grads['low'])

        info = dict(info)
        info['actor/grad_norm_high'] = optax.global_norm(grads['main']['modules_actor'])
        if self.config['learned_action_std']:
            info['actor/grad_norm_action_log_std'] = optax.global_norm(grads['main']['modules_action_log_std'])
        info['actor/grad_norm_low'] = optax.global_norm(grads['low'])
        info['critic/grad_norm'] = optax.global_norm(grads['main']['modules_critic'])

        return (
            self.replace(
                network=new_network,
                skill_agent=self.skill_agent.replace(network=new_low),
                rng=new_rng,
            ),
            info,
        )

    # ── Acting ────────────────────────────────────────────────────────────────

    def _single_obs(self, observations):
        return observations.ndim == (3 if self.config['encoder'] is not None else 1)

    @jax.jit
    def sample_skills(self, observations, goals, seed=None, temperature=1.0):
        """k ~ pi_hi(. | s, g) (temperature=0 -> argmax). Accepts a single observation or a batch."""
        if seed is None:
            seed = self.rng
        if goals is None:
            raise ValueError('online_composed_skill_policy needs a goal: pi_hi(k | s, g) is goal-conditioned.')
        single = self._single_obs(observations)
        obs_b = observations[None, ...] if single else observations
        goals_b = goals[None, ...] if single else goals
        skills = self.network.select('actor')(obs_b, goals_b, temperature=temperature).sample(seed=seed)
        skills = skills.astype(jnp.int32)
        return skills[0] if single else skills

    def _skill_conditioning(self, obs_b, goals_b, seed, temperature):
        """The vector handed to the low level for a batch, plus the skill index to RECORD for it.

        'enumerate' / 'reinforce': k ~ pi_hi(.|s,g) at `temperature`, returned as its one-hot (Eq. 1).
        'softmax': p = softmax(logits / temperature) itself (Eq. 8); the recorded index is argmax p.
        At temperature=0 both give the argmax one-hot, so evaluation is identical across methods.
        """
        if self.config['composed_grad_method'] == 'softmax':
            probs = self.network.select('actor')(obs_b, goals_b, temperature=temperature).probs  # (B, K)
            return probs, jnp.argmax(probs, axis=-1).astype(jnp.int32)
        skill_idxs = self.network.select('actor')(obs_b, goals_b, temperature=temperature).sample(seed=seed)
        skill_idxs = skill_idxs.astype(jnp.int32)
        return self._skill_vectors()[skill_idxs], skill_idxs

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """a ~ pi(.|s,g): Eq. 1 (draw k, act with pi_lo(.|s,z_k)) or Eq. 8 under 'softmax'. One env step.

        `temperature` is the HIGH level's (the collector explores at 1, the evaluator commits at 0);
        the low level always acts at `low_temperature`, as it does under a frozen controller.
        """
        if seed is None:
            seed = self.rng
        if goals is None:
            raise ValueError('online_composed_skill_policy needs a goal: pi_hi(k | s, g) is goal-conditioned.')
        high_seed, low_seed = jax.random.split(seed)
        single = self._single_obs(observations)
        obs_b = observations[None, ...] if single else observations
        goals_b = goals[None, ...] if single else goals

        skills, _ = self._skill_conditioning(obs_b, goals_b, high_seed, temperature)
        actions = self._act_low(obs_b, goals_b, skills, low_seed, temperature)
        return actions[0] if single else actions

    def _act_low(self, obs_b, goals_b, skills, rng, temperature):
        """The low level's action for acting: Eq. 9 with the std scaled by `temperature` when
        `learned_action_std`, else the plain (low_temperature) action."""
        if self.config['learned_action_std']:
            return self._noisy_low_action(obs_b, goals_b, skills, None, rng, temperature=temperature)[0]
        return self._low_action(obs_b, skills, rng=rng)

    # ── Eval hooks (contract used by utils/online_evaluation.py) ──────────────
    #
    # The composed policy is flat and stateless -- the skill is redrawn every step -- so this pair
    # exists only so the evaluator can RECORD which skill acted at each step (the skill-usage
    # histogram and the skill-colored trajectory plot main_online.py logs). `skill_commitment_k`
    # is pinned to 1 in the config for the same reason: `skill_usage_stats` strides by it.

    def init_eval_state(self, max_steps=None):
        del max_steps
        return {'skill': jnp.zeros((), jnp.int32), 'count': jnp.zeros((), jnp.int32)}

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None, temperature=1.0):
        """`sample_actions`, additionally reporting the skill it drew (single-observation eval)."""
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()
        if goals is None:
            raise ValueError('online_composed_skill_policy needs a goal: pi_hi(k | s, g) is goal-conditioned.')
        high_seed, low_seed = jax.random.split(seed)
        single = self._single_obs(observations)
        obs_b = observations[None, ...] if single else observations
        goals_b = goals[None, ...] if single else goals

        skills, skill_idxs = self._skill_conditioning(obs_b, goals_b, high_seed, temperature)
        actions = self._act_low(obs_b, goals_b, skills, low_seed, temperature)
        actions = actions[0] if single else actions
        skill = skill_idxs[0] if single else skill_idxs
        return actions, {'skill': skill, 'count': agent_state['count'] + 1}

    # ── Skill-conditioned evaluation hooks (see eval_skill_policy.py) ─────────
    #
    # These report the FINE-TUNED low level, not the pretrained checkpoint it started from.

    def skill_set(self, seed=None, num_skills=None, observations=None):
        del seed, num_skills, observations
        return self._skill_vectors()

    @jax.jit
    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        del temperature  # Reproduce the composed policy's own execution, at low_temperature.
        if seed is None:
            seed = self.rng
        single = self._single_obs(observations)
        obs_b = observations[None, ...] if single else observations
        skills = skills[None, ...] if skills.ndim == 1 else skills
        skills = jnp.broadcast_to(skills, (obs_b.shape[0], skills.shape[-1]))
        actions = self._low_action(obs_b, skills, rng=seed)
        return actions[0] if single else actions

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations (also the example goals).
            ex_actions: Example batch of PRIMITIVE env actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        if config['discrete']:
            raise NotImplementedError(
                'online_composed_skill_policy supports continuous action spaces only (both supported low '
                'levels are Gaussian and the actor loss differentiates through the emitted action).'
            )
        if float(config['low_lr']) < 0.0:
            raise ValueError(f"low_lr must be >= 0 (0 == frozen low level), got {config['low_lr']}.")
        if config['composed_grad_method'] not in COMPOSED_GRAD_METHODS:
            raise ValueError(
                f"composed_grad_method must be one of {sorted(COMPOSED_GRAD_METHODS)}, got "
                f"{config['composed_grad_method']!r}."
            )
        if config['reinforce_baseline'] not in REINFORCE_BASELINES:
            raise ValueError(
                f"reinforce_baseline must be one of {sorted(REINFORCE_BASELINES)}, got "
                f"{config['reinforce_baseline']!r}."
            )
        if config['learned_action_std']:
            if config['composed_grad_method'] != 'softmax':
                raise ValueError(
                    "learned_action_std=True requires composed_grad_method='softmax': under 'enumerate' the "
                    'K-component mixture has no closed-form action entropy, and under \'reinforce\' the sampled '
                    f"skill already carries the exploration. Got {config['composed_grad_method']!r}."
                )
            if float(config['low_temperature']) != 0.0:
                raise ValueError(
                    "learned_action_std=True replaces the low level's own std with the high level's head, so "
                    f"low_temperature must be 0 (got {config['low_temperature']})."
                )
        if config['composed_grad_method'] == 'softmax' and int(config['skill_commitment_k']) != 1:
            raise ValueError(
                f"composed_grad_method='softmax' requires skill_commitment_k == 1 (got "
                f"{config['skill_commitment_k']}): the low level is conditioned on the probability vector "
                f'p(s, g) of the CURRENT step, and holding it fixed across a skill horizon is not implemented.'
            )

        # ── The pretrained low level (TRAINED here, unlike every *_controller) ──
        _reject_unsupported_family(config['skill_checkpoint_path'])
        skill_agent, resolved = load_frozen_skill_agent(
            seed, ex_observations, ex_actions, config, SKILL_AGENT_CLASSES, caller='online_composed_skill_policy'
        )
        num_skills = resolved['num_skills']
        if resolved['agent_name'] == 'opal' and resolved['skill_config'].get('latent_type') != 'discrete':
            raise ValueError(
                f'[online_composed_skill_policy] the opal checkpoint {resolved["ckpt_path"]} has '
                f'latent_type={resolved["skill_config"].get("latent_type")!r}. Eq. 1 is a sum over a FINITE skill '
                f'set, which only the discrete (App. F) OPAL path has; a continuous latent makes it an integral. '
                f'Use a latent_type=discrete run.'
            )

        # Give the low level its OWN optimizer at `low_lr`. The pretrained optimizer is discarded
        # deliberately: its Adam state belongs to a different objective, and some families ship
        # phased optimizers that hard-freeze modules and would silently zero every update here.
        low_net = skill_agent.network
        low_net = TrainState.create(
            low_net.model_def, low_net.params, tx=optax.adam(learning_rate=float(config['low_lr']))
        )
        skill_agent = skill_agent.replace(network=low_net)
        print(
            f'[online_composed_skill_policy] the low level is TRAINED, not frozen: fresh Adam at '
            f"low_lr={float(config['low_lr'])} (lr={float(config['lr'])}); low_lr=0.0 reproduces the frozen "
            f'baseline at skill_commitment_k=1.'
        )

        # ── Trainable flat critic + high level + alpha ──────────────────────────
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        ex_goals = ex_observations

        # The critic is FLAT: Q(s, a, g) over primitive actions. This is the whole reason the low
        # level can be trained -- a skill-level Q(s, z, g) has no dependence on its parameters.
        critic_def = GCBilinearValue(
            hidden_dims=tuple(config['value_hidden_dims']),
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
            state_encoder=encoders.get('critic_state'),
            goal_encoder=encoders.get('critic_goal'),
        )
        actor_def = GCDiscreteActor(
            hidden_dims=tuple(config['actor_hidden_dims']),
            action_dim=num_skills,
            gc_encoder=encoders.get('actor'),
        )
        alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            alpha=(alpha_def, ()),
        )
        if config['learned_action_std']:
            # Eq. 9: the high level's second head (per-dimension action log-std over (s, g)) and
            # the second entropy temperature. Both live in `network`, i.e. train at `lr`.
            if config['encoder'] is not None:
                encoders['action_log_std'] = GCEncoder(concat_encoder=encoder_modules[config['encoder']]())
            log_std_def = GCActionLogStd(
                hidden_dims=tuple(config['actor_hidden_dims']),
                action_dim=ex_actions.shape[-1],
                log_std_min=float(config['action_log_std_min']),
                log_std_max=float(config['action_log_std_max']),
                gc_encoder=encoders.get('action_log_std'),
            )
            network_info['action_log_std'] = (log_std_def, (ex_observations, ex_goals))
            network_info['action_alpha'] = (LogParam(), ())
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))

        # Resolved values for the agent's own use (main_online.py serialises FLAGS before create,
        # so these do not reach flags.json unless passed explicitly).
        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        stored_config['num_skills'] = num_skills
        stored_config['skill_restore_epoch'] = resolved['restore_epoch']
        stored_config['skill_checkpoint_path'] = resolved['ckpt_path']
        stored_config['skill_agent_name'] = resolved['agent_name']
        stored_config['target_entropy'] = float(config['target_entropy_frac']) * float(np.log(num_skills))
        if config['learned_action_std']:
            action_dim = int(ex_actions.shape[-1])
            stored_config['action_target_entropy'] = (
                -0.5 * action_dim if config['action_target_entropy'] is None else float(config['action_target_entropy'])
            )
            print(
                f'[online_composed_skill_policy] learned_action_std=True: the high level has an action log-std head '
                f'(clipped to [{float(config["action_log_std_min"])}, {float(config["action_log_std_max"])}]) that '
                f"replaces the low level's constant std; action_target_entropy = "
                f"{stored_config['action_target_entropy']:.3f} (action_dim={action_dim}), its own alpha. The "
                f'categorical entropy target below is now a regulariser on p, not the exploration knob.'
            )
        # Rows are env steps, so the buffer's future-goal discount is the per-step one.
        stored_config['goal_discount'] = float(config['discount'])
        method = config['composed_grad_method']
        method_desc = {
            'enumerate': 'an EXACT gradient from the K-term sum over skills.',
            'reinforce': "a REINFORCE gradient from one sampled skill, rewarded by the low level's own loss.",
            'softmax': (
                'a pathwise gradient through softmax(logits), which is fed to the low level AS THE SKILL '
                'VECTOR (no one-hot, no sampling; acting does the same, so the collector is deterministic at '
                f"low_temperature={float(config['low_temperature'])})."
            ),
        }[method]
        print(
            f'[online_composed_skill_policy] composed_grad_method={method!r}'
            + (f" (reinforce_baseline={config['reinforce_baseline']!r})" if method == 'reinforce' else '')
            + f': the high level gets {method_desc}'
        )
        print(
            f"[online_composed_skill_policy] target_entropy = target_entropy_frac * log(num_skills) = "
            f"{float(config['target_entropy_frac']):.3f} * {float(np.log(num_skills)):.3f} = "
            f"{stored_config['target_entropy']:.3f} (num_skills={num_skills}); the skill is redrawn every env "
            f'step (no skill horizon).'
        )

        return cls(rng, network=network, skill_agent=skill_agent, config=flax.core.FrozenDict(**stored_config))


def _reject_unsupported_family(ckpt_path):
    """Raise the modelling reason (not a KeyError) for a checkpoint family Eq. 1 cannot compose."""
    if not ckpt_path:
        return  # load_frozen_skill_agent raises the missing-flag error.
    flags_path = os.path.join(str(ckpt_path).rstrip('/'), 'flags.json')
    if not os.path.exists(flags_path):
        return
    with open(flags_path) as f:
        agent_name = json.load(f).get('agent', {}).get('agent_name')
    if agent_name in UNSUPPORTED_SKILL_AGENTS:
        raise ValueError(f'[online_composed_skill_policy] {ckpt_path}: {UNSUPPORTED_SKILL_AGENTS[agent_name]}')


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='online_composed_skill_policy',  # Agent name.
            rollout_type='flat',  # Experience collector: one env step per row (no SMDP).
            lr=3e-4,  # Learning rate of the critic, the high level pi_hi and alpha.
            low_lr=3e-5,  # Learning rate of the LOW level pi_lo(a|s,z). 0.0 == frozen baseline.
            batch_size=1024,  # Batch size (the critic always sees all of it).
            actor_batch_size=ml_collections.config_dict.placeholder(int),  # Rows for the K-term actor
            # loss (None = all). Each row costs K low-level forward passes and K critic evaluations.
            actor_hidden_dims=(512, 512, 512),  # High-level actor hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Contrastive critic hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount per env step (future-goal sampling: P(offset=j) ~ discount^j).
            target_entropy_frac=0.5,  # H_target = frac * log(num_skills) (frac <= 1 -> always reachable).
            # How the Q signal reaches the two levels (both optimise the SAME objective, Eq. 5):
            #   'enumerate' -- sum over all K skills. Exact, zero-variance, K low-level forward
            #                  passes and K critic evaluations per row.
            #   'reinforce' -- sample ONE skill per row. The low level takes its ordinary
            #                  pathwise gradient through the emitted action; the high level takes
            #                  a score-function gradient with the low level's loss as a negative
            #                  reward. 1 low-level pass per row, but the high-level gradient now
            #                  carries variance.
            #   'softmax'   -- relaxation, not an estimator: softmax(logits) is handed to the low
            #                  level AS ITS SKILL VECTOR, so one pathwise gradient reaches both
            #                  levels. Changes the policy (acting feeds the same vector; eval at
            #                  temperature 0 is still the argmax one-hot). Needs skill_commitment_k=1.
            composed_grad_method='enumerate',
            # REINFORCE only: baseline subtracted from the reward. 'batch' is the leave-one-out
            # batch mean of Q (exactly unbiased); 'none' is b = 0. Ignored by 'enumerate'.
            reinforce_baseline='batch',
            low_temperature=0.0,  # Sampling temperature of the low level (0 -> its mode, as when frozen).
            # 'softmax' only: give the high level an action log-std head and sample
            # a = clip(mu_lo(s, p) + exp(log_std_hi(s, g)) * eps) (Eq. 9), with a second alpha tuned on
            # the action Gaussian's entropy. Otherwise the softmax collector is deterministic.
            learned_action_std=False,
            action_target_entropy=ml_collections.config_dict.placeholder(float),  # None -> -0.5 * action_dim.
            action_log_std_min=-5.0,  # Clip range of the action log-std head (online_crl's).
            action_log_std_max=2.0,
            # Pretrained low level (trained here at low_lr).
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),  # Required: skill run dir.
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),  # Pretrained epoch (None -> latest).
            num_skills=ml_collections.config_dict.placeholder(int),  # Read from the checkpoint; asserted if set.
            skill_agent_name=ml_collections.config_dict.placeholder(str),  # Read from the checkpoint's flags.json.
            # Always 1: the skill is redrawn every env step. Present because main_online.py's
            # skill-usage logging strides the recorded skill sequence by it.
            skill_commitment_k=1,
            # Online schedule (consumed by main_online.py); units are env-step rows.
            #
            # These are LITERALLY the values in `agents/online_crl_skill_controller.py` -- not
            # rescaled by its skill_commitment_k. A row is one high-level DECISION in both agents
            # (a k-step option there, a single env step here), so utd_ratio=1 means "one gradient
            # step per policy decision" and replay_size means "50000 decisions of history" for
            # both. That is the sense in which the two runs share a schedule; note it does NOT
            # make them equal per ENV STEP, which is the x-axis main_online.py logs on:
            #
            #                        controller (k=10)      here
            #   updates / env step   0.1                    1.0     (1M vs 100k updates at 1M steps)
            #   first update at      10000 env steps        1000
            #   buffer holds         500000 env steps       50000
            #
            unroll_length=50,  # Rows collected between update rounds (controller: 50).
            utd_ratio=1.0,  # Gradient steps per row (controller: 1). Float, so fractions are allowed.
            min_replay_size=1000,  # Rows collected before the first update (controller: 1000).
            replay_size=50000,  # Replay buffer capacity in rows (controller: 50000).
            offline_ratio=0.5,  # RLPD (--offline_dataset): fraction of every batch drawn from the offline buffer.
            # Observation pipeline (must match the skill checkpoint).
            discrete=False,  # Whether the action space is discrete (unsupported here).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None for state-based).
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
