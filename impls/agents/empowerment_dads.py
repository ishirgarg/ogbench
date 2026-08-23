"""Offline skill-level empowerment agent (DADS/DIAYN-style, MI-maximizing labels).

Instead of discovering skills by likelihood EM under a step-dynamics model and
then measuring the MI of that fixed labeling (the DADS/OPAL-Appendix-F
recipe; see empowerment/run_dads.py), this agent chooses the labels themselves
to maximize the Barber-Agakov bracket directly:

    max_{alpha, q}  E_{s0,s+} E_{z ~ alpha(.|s0,s+)} [ log q(s+|s0,z)
                        - logsumexp_z'(log m(z'|s0) + log q(s+|s0,z')) ]

with, per (s0, s+) pair drawn straight from the dataset -- s+ is a discounted
future state of the same trajectory, i.e. GCDataset's `value_goals` with
value_geom_sample=True and value_p_trajgoal=1.0, so s+ = s_{t+Delta} with
Delta ~ Geometric(1-gamma) clamped at the trajectory end. No windows, no
SequenceDataset, no masking:
  - alpha(z|s0,s+): soft label assignment from an encoder that sees only the
    endpoint pair (s0 and the delta s+ - s0, for the same s+ that the channel
    is scored on) -> softmax over k skills. Conditioning on the pair rather
    than on a whole window is what makes this a true empowerment estimate: a
    window encoder could hand z information about intermediate states that
    never surfaces in s+, inflating the bracket with content the channel
    q(s+|s0,z) cannot be credited for. The expectation over z is computed
    exactly (the bracket is evaluated for all z), so gradients flow to the
    encoder without sampling tricks.
  - q(s+|s0,z): MDN channel on the delta target s+ - s0. Trained on the
    bracket itself: the numerator rewards fit, the denominator rewards
    separation between skills.
  - m(z|s0): classifier fit by cross-entropy toward alpha, i.e. toward the
    actual skill mix at s0 under the current labeling. m is stop-gradiented
    inside the bracket: letting the bracket's gradient shape m would push it
    away from the true marginal and invalidate the bound. Fitting m to the
    mix is part of the bound's correctness, not a regularizer.

No BA routine is needed: the max over labelings/usage that BA supplies for a
fixed channel is built into the training objective, and at the optimum the
assignment tends toward the capacity-achieving structure. Full collapse (all
mass on one label) drives the bracket to zero, but partial collapse (dead
skills that never receive gradient) is a stable local optimum; `kl_coef` > 0
adds KL(Unif[k] || batch skill usage) to the loss (gradient into the encoder),
the Lagrangian form of DADS's fixed uniform skill prior, to keep all k skills
in use. It does not enter the reported bound.

Note on interpretation: because the labels are optimized to maximize the very
bound being reported, `mi_estimate` is a prescriptive quantity ("the most
information-carrying way to carve this data into k modes"), upper-tilted
relative to a likelihood-EM estimate (empowerment/run_dads.py).

This agent has no actor; sample_actions returns uniform random actions.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from empowerment.dads_empowerment import MDN, mdn_log_prob, mdn_sample
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


class EmpowermentDADSAgent(flax.struct.PyTreeNode):
    """Offline skill-level empowerment via MI-maximizing label assignment."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # -------------------------------------------------------------- q-utils --
    def _q_ll_allz(self, s, x_delta, params=None):
        """log q(x | s, z) for every z. [R, D] -> [k, R]."""
        k = self.config['num_skills']
        eye = jnp.eye(k)

        def per_z(zoh):
            cond = jnp.concatenate([s, jnp.tile(zoh, (s.shape[0], 1))], axis=-1)
            return mdn_log_prob(*self.network.select('q')(cond, params=params), x_delta)

        return jax.vmap(per_z)(eye)

    def _bracket_allz(self, s, x_delta, params=None):
        """b_z = log q(x|s,z) - logsumexp_z'(log m(z'|s) + log q(x|s,z')) for
        every z: [k, B]. m is stop-gradiented (see module docstring)."""
        lq_all = self._q_ll_allz(s, x_delta, params=params)  # [k, B]
        logm = jax.lax.stop_gradient(
            jax.nn.log_softmax(self.network.select('clf')(s, params=params), axis=-1)
        )  # [B, k]
        log_mix = jax.nn.logsumexp(logm.T + lq_all, axis=0)  # [B]
        return lq_all - log_mix[None]

    def _bracket(self, s, z_onehot, x_delta, params=None):
        """b for a specific z per row (used by empowerment())."""
        b_all = self._bracket_allz(s, x_delta, params=params)  # [k, B]
        return (z_onehot.T * b_all).sum(0)

    # -------------------------------------------------------------- losses --
    @jax.jit
    def total_loss(self, batch, grad_params):
        info = {}

        s0 = batch['observations']  # [B, D]
        # Discounted future state of the same trajectory (GCDataset value_goals
        # with geometric sampling); the channel is trained on the delta.
        target = batch['value_goals'] - s0

        # Label assignment: soft posterior over z from (s0, s+) only -- the same
        # pair the channel is scored on, so z cannot smuggle in extra content.
        enc_in = jnp.concatenate([s0, target], axis=-1)
        logits = self.network.select('enc')(enc_in, params=grad_params)  # [B, k]
        alpha = jax.nn.softmax(logits, axis=-1)

        # The objective: expected bracket under alpha, exact over z. Gradients
        # flow to the encoder (through alpha) and to q (through the bracket).
        b_all = self._bracket_allz(s0, target, params=grad_params)  # [k, B]
        bracket_loss = -(alpha * b_all.T).sum(-1).mean()
        info['bracket_loss'] = bracket_loss

        # m(z|s0): cross-entropy toward the current assignment mix.
        logm = jax.nn.log_softmax(self.network.select('clf')(s0, params=grad_params), axis=-1)
        alpha_sg = jax.lax.stop_gradient(alpha)
        clf_loss = -(alpha_sg * logm).sum(-1).mean()
        info['clf_loss'] = clf_loss

        # KL(Unif[k] || batch usage): pushes the labeling toward uniform skill usage.
        k = self.config['num_skills']
        usage = alpha.mean(0)  # [k]
        kl_usage = -jnp.log(k) - jnp.log(usage + 1e-8).mean()
        info['kl_usage'] = kl_usage

        loss = bracket_loss + clf_loss + self.config.get('kl_coef', 0.0) * kl_usage

        # Diagnostics.
        info['mi_estimate'] = -jax.lax.stop_gradient(bracket_loss)
        usage = jax.lax.stop_gradient(usage)
        info['usage_min'] = usage.min()
        info['usage_max'] = usage.max()
        return loss, info

    @jax.jit
    def update(self, batch):
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p)
        )
        return self.replace(network=new_network), info

    # -------------------------------------------------- estimation for eval --
    @jax.jit
    def empowerment(self, observations, rng=None):
        """Model-empowerment estimate at arbitrary states. [B, D] -> [B] nats.

        Per state: M paired draws z ~ m(.|s), s+ ~ q(.|s,z); mean bracket.
        """
        rng = rng if rng is not None else self.rng
        B = observations.shape[0]
        M = self.config['est_num_joints']
        k = self.config['num_skills']
        eye = jnp.eye(k)
        k_z, k_x = jax.random.split(rng)

        s = jnp.repeat(observations, M, axis=0)  # [B*M, D]
        logm = jax.nn.log_softmax(self.network.select('clf')(s), axis=-1)
        z = jax.random.categorical(k_z, logm, axis=-1)  # [B*M]
        cond = jnp.concatenate([s, eye[z]], axis=-1)
        heads = self.network.select('q')(cond)
        keys = jax.random.split(k_x, s.shape[0])
        deltas = jax.vmap(mdn_sample)(keys, *heads)  # [B*M, D]
        b = self._bracket(s, eye[z], deltas)
        return b.reshape(B, M).mean(axis=1)

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Uniform random actions in [-1, 1]; this agent has no policy."""
        seed = seed if seed is not None else self.rng
        single = observations.ndim == 1
        if single:
            observations = observations[None]
        actions = jax.random.uniform(
            seed, (*observations.shape[:-1], self.config['action_dim']), minval=-1.0, maxval=1.0
        )
        return actions[0] if single else actions

    # ---------------------------------------------------------- constructor --
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        config = dict(config)
        assert config.get('encoder') is None, 'empowerment_dads is state-based only (no encoder support)'
        config['action_dim'] = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]
        config['obs_dim'] = obs_dim
        k = config['num_skills']

        enc_def = MLP((*config['enc_hidden_dims'], k), activate_final=False,
                      layer_norm=config['layer_norm'])
        q_def = MDN(obs_dim, config['num_components'], tuple(config['mdn_hidden_dims']),
                    layer_norm=config['layer_norm'])
        clf_def = MLP((*config['clf_hidden_dims'], k), activate_final=False,
                      layer_norm=config['layer_norm'])

        ex_enc_in = jnp.zeros((*ex_observations.shape[:-1], obs_dim * 2))
        ex_cond = jnp.zeros((*ex_observations.shape[:-1], obs_dim + k))
        network_info = dict(
            enc=(enc_def, (ex_enc_in,)),
            q=(q_def, (ex_cond,)),
            clf=(clf_def, (ex_observations,)),
        )
        networks = {n: v[0] for n, v in network_info.items()}
        network_args = {n: v[1] for n, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='empowerment_dads',
        # Matched to the other empowerment agents for a like-for-like
        # comparison (same lr, batch size, MDN/classifier trunks, skill count,
        # discount).
        lr=3e-4,
        batch_size=1024,
        num_skills=15,           # k
        num_components=8,        # MDN mixture components
        mdn_hidden_dims=(512, 512, 512),
        clf_hidden_dims=(512, 512, 512),
        enc_hidden_dims=(512, 512, 512),
        layer_norm=True,
        est_num_joints=16,       # M draws per state in empowerment()
        kl_coef=0.0,             # weight on KL(Unif[k] || skill usage); 0 disables
        # main.py / GCDataset wiring. `value_goals` IS the channel target s+:
        # geometric future state from the same trajectory, so p_trajgoal=1.0
        # and geom_sample=True are load-bearing, not defaults. The actor_ keys
        # are only there to satisfy GCDataset's __post_init__ asserts.
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
