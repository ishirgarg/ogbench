"""Offline skill-level empowerment agent (DADS/DIAYN-style, MI-maximizing labels).

Companion to agents/empowerment_opal_dads.py with the EM machinery removed:
instead of discovering skills by likelihood EM under a step-dynamics model and
then measuring the MI of that fixed labeling (with Blahut-Arimoto as a
separate capacity routine), this agent chooses the labels themselves to
maximize the Barber-Agakov bracket directly:

    max_{alpha, q}  E_w E_{z ~ alpha(.|w)} [ log q(s+|s0,z)
                        - logsumexp_z'(log m(z'|s0) + log q(s+|s0,z')) ]

with, per length-K window w (SequenceDataset):
  - alpha(z|w): soft label assignment from a window encoder (s0 + flattened
    deltas -> softmax over k skills). The expectation over z is computed
    exactly (the bracket is evaluated for all z), so gradients flow to the
    encoder without sampling tricks.
  - q(s+|s0,z): MDN channel on delta targets s_Delta - s0 with Delta ~
    truncated Geometric(1-gamma) on {1..K-1} (exact categorical). Trained on
    the bracket itself: the numerator rewards fit, the denominator rewards
    separation between skills.
  - m(z|s0): classifier fit by cross-entropy toward alpha, i.e. toward the
    actual skill mix at s0 under the current labeling. m is stop-gradiented
    inside the bracket: letting the bracket's gradient shape m would push it
    away from the true marginal and invalidate the bound. Fitting m to the
    mix is part of the bound's correctness, not a regularizer.

No BA routine is needed: the max over labelings/usage that BA supplies for a
fixed channel is built into the training objective, and at the optimum the
assignment tends toward the capacity-achieving structure. Collapse (all mass
on one label) is not a fixed point worth reaching under this objective -- it
drives the bracket to zero -- so no auxiliary entropy terms are added.

Note on interpretation: because the labels are optimized to maximize the very
bound being reported, `mi_estimate` is a prescriptive quantity ("the most
information-carrying way to carve this data into k modes"), upper-tilted
relative to the likelihood-EM estimate of empowerment_opal_dads.

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

    def _sample_trunc_geom(self, rng, shape):
        """Delta ~ Geometric(1-gamma) truncated to {1..K-1} (exact categorical)."""
        K = self.config['sequence_length']
        gamma = self.config['discount']
        logits = jnp.arange(K - 1) * jnp.log(gamma)  # delta = 1..K-1
        return 1 + jax.random.categorical(rng, logits, shape=shape)

    # -------------------------------------------------------------- losses --
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        k_d, _ = jax.random.split(rng)

        obs_seq = batch['observations_seq']  # [B, K, D]
        seq_mask = batch['seq_mask']  # [B, K]
        B, K, D = obs_seq.shape
        # Only complete windows count (a window crossing a trajectory end has
        # a clamped tail and would corrupt the truncated-discounted target).
        w = seq_mask[:, -1]
        w_norm = w / jnp.clip(w.sum(), 1.0, None)

        s0 = obs_seq[:, 0]
        bidx = jnp.arange(B)

        # Label assignment: soft posterior over z from the window encoder.
        enc_in = jnp.concatenate([s0, (obs_seq[:, 1:] - s0[:, None]).reshape(B, -1)], axis=-1)
        logits = self.network.select('enc')(enc_in, params=grad_params)  # [B, k]
        alpha = jax.nn.softmax(logits, axis=-1)

        # Discounted channel target.
        d_off = self._sample_trunc_geom(k_d, (B,))
        target = obs_seq[bidx, d_off] - s0

        # The objective: expected bracket under alpha, exact over z. Gradients
        # flow to the encoder (through alpha) and to q (through the bracket).
        b_all = self._bracket_allz(s0, target, params=grad_params)  # [k, B]
        bracket_loss = -(w_norm * (alpha * b_all.T).sum(-1)).sum()
        info['bracket_loss'] = bracket_loss

        # m(z|s0): cross-entropy toward the current assignment mix.
        logm = jax.nn.log_softmax(self.network.select('clf')(s0, params=grad_params), axis=-1)
        alpha_sg = jax.lax.stop_gradient(alpha)
        clf_loss = (w_norm * -(alpha_sg * logm).sum(-1)).sum()
        info['clf_loss'] = clf_loss

        loss = bracket_loss + clf_loss

        # Diagnostics.
        info['mi_estimate'] = -jax.lax.stop_gradient(bracket_loss)
        usage = (w[:, None] * alpha_sg).sum(0) / jnp.clip(w.sum(), 1.0, None)  # [k]
        info['usage_min'] = usage.min()
        info['usage_max'] = usage.max()
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
        K = config['sequence_length']

        enc_def = MLP((*config['enc_hidden_dims'], k), activate_final=False,
                      layer_norm=config['layer_norm'])
        q_def = MDN(obs_dim, config['num_components'], tuple(config['mdn_hidden_dims']),
                    layer_norm=config['layer_norm'])
        clf_def = MLP((*config['clf_hidden_dims'], k), activate_final=False,
                      layer_norm=config['layer_norm'])

        ex_enc_in = jnp.zeros((*ex_observations.shape[:-1], obs_dim * K))
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
        # Matched to agents/empowerment_opal_dads.py for a like-for-like
        # comparison (same lr, batch size, MDN/classifier trunks, skill count,
        # window length, discount).
        lr=3e-4,
        batch_size=1024,
        num_skills=15,           # k
        num_components=8,        # MDN mixture components
        mdn_hidden_dims=(512, 512, 512),
        clf_hidden_dims=(512, 512, 512),
        enc_hidden_dims=(512, 512, 512),
        layer_norm=True,
        est_num_joints=16,       # M draws per state in empowerment()
        # main.py / SequenceDataset wiring. The value_/actor_ goal keys below
        # are required by GCDataset's __post_init__ asserts (SequenceDataset
        # subclasses it) but this agent never reads the sampled goals.
        sequence_length=15,      # K (chunk length)
        discount=0.99,
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='SequenceDataset',
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
