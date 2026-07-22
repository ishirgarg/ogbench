"""Offline skill-level empowerment agent (DADS-style, stochastic EM).

main.py-integrated version of Algorithms 2+3 in empowerment/dads_empowerment.py:
estimates the K-step-truncated discounted effective empowerment I(s+; z | s)
with z a discrete skill in {0..k-1} labeling length-K behavior windows
(SequenceDataset), via the exact-z-sum Barber-Agakov bracket

    b = log q(s+|s,z) - logsumexp_z'(log m(z'|s) + log q(s+|s,z')).

Instead of the standalone runner's batch EM over a fixed chunk table, this
agent runs STOCHASTIC EM inside update(): each step computes the exact Bayes
posterior post(z | window) for the sampled windows under the current step
dynamics d(s_{t+1}|s_t,z) and prior p(z), samples z ~ post (responsibility
weighting in expectation), takes an NLL gradient step on d, and EMA-updates
p(z). After `em_steps`, d and p(z) are frozen and the phase-2 nets train from
the (now fixed) on-the-fly posteriors:
  - q(s+|s0,z): MDN channel, target s_Delta - s0 with Delta ~ truncated
    Geometric(1-gamma) on {1..K-1} (exact, via categorical inverse-CDF).
  - m(z|s0): soft-label classifier on the window posteriors (the BEHAVIOR
    skill mix at s -- NOT uniform; uniform would change the measured quantity).

All MDNs model DELTAS (x - s), a unit-Jacobian reparameterization.

Caveats (full discussion in empowerment/dads_empowerment.py):
  - The bracket lower-bounds I only up to +KL(p(z|s) || m): channel (q) error
    biases the estimate DOWN, skill-marginal (m) error biases it UP.
  - `empowerment(s)` samples (z, s+) from m x q, i.e. MODEL empowerment.
  - No collapse-guard reseeding here (pure-jit paradigm); monitor p_z in the
    logs. The honest MI ceiling is H(p_z) <= log k. The standalone runner
    (empowerment/run_dads.py) implements exact table EM with a bounded-revival
    guard; use it for reference numbers.
  - Windows here are sampled at arbitrary offsets (SequenceDataset) rather
    than the runner's non-overlapping grid; incomplete windows (crossing a
    trajectory end) are masked out entirely.

Algorithm 3 (Blahut-Arimoto capacity of the learned channel at a state) is
available post-hoc via `make_ba_model(agent)`, which adapts this agent to
empowerment.ba_capacity.capacity_ba.

This agent has no actor; sample_actions returns uniform random actions.
"""

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from empowerment.dads_empowerment import MDN, mdn_log_prob, mdn_sample
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


class EmpowermentOPALDADSAgent(flax.struct.PyTreeNode):
    """Offline skill-level empowerment estimator (stochastic EM + BA bracket)."""

    rng: Any
    network: Any
    p_z: Any  # [k] categorical skill prior (EMA of batch posterior means)
    config: Any = nonpytree_field()

    # --------------------------------------------------------------- E-step --
    def _dyn_ll_allz(self, s, target, params=None):
        """log d(target | s, z) for every z. s, target: [R, D] -> [k, R]."""
        k = self.config['num_skills']
        eye = jnp.eye(k)

        def per_z(zoh):
            cond = jnp.concatenate([s, jnp.tile(zoh, (s.shape[0], 1))], axis=-1)
            return mdn_log_prob(*self.network.select('dyn')(cond, params=params), target)

        return jax.vmap(per_z)(eye)

    def _posteriors(self, obs_seq, seq_mask):
        """Exact Bayes posteriors over z for each window (no gradients).

        obs_seq: [B, K, D]; seq_mask: [B, K]. Only in-trajectory transitions
        (mask[t+1] == 1) contribute to the score.
        Returns post [B, k].
        """
        B, K, D = obs_seq.shape
        k = self.config['num_skills']
        st = obs_seq[:, :-1].reshape(-1, D)
        delta = (obs_seq[:, 1:] - obs_seq[:, :-1]).reshape(-1, D)
        ll = self._dyn_ll_allz(st, delta)  # [k, B*(K-1)]
        ll = ll.reshape(k, B, K - 1) * seq_mask[None, :, 1:]  # mask padded transitions
        score = ll.sum(-1).T + jnp.log(jnp.clip(self.p_z, 1e-12, None))[None]  # [B, k]
        return jax.lax.stop_gradient(jax.nn.softmax(score, axis=-1))

    # -------------------------------------------------------------- q-utils --
    def _q_ll_allz(self, s, x_delta, params=None):
        """log q(x | s, z) for every z. [R, D] -> [k, R]."""
        k = self.config['num_skills']
        eye = jnp.eye(k)

        def per_z(zoh):
            cond = jnp.concatenate([s, jnp.tile(zoh, (s.shape[0], 1))], axis=-1)
            return mdn_log_prob(*self.network.select('q')(cond, params=params), x_delta)

        return jax.vmap(per_z)(eye)

    def _bracket(self, s, z_onehot, x_delta, params=None):
        """b = log q(x|s,z) - logsumexp_z'(log m(z'|s) + log q(x|s,z'))."""
        cond = jnp.concatenate([s, z_onehot], axis=-1)
        lq_sel = mdn_log_prob(*self.network.select('q')(cond, params=params), x_delta)
        logm = jax.nn.log_softmax(self.network.select('clf')(s, params=params), axis=-1)  # [B, k]
        lq_all = self._q_ll_allz(s, x_delta, params=params)  # [k, B]
        log_mix = jax.nn.logsumexp(logm.T + lq_all, axis=0)
        return lq_sel - log_mix

    def _sample_trunc_geom(self, rng, shape):
        """Delta ~ Geometric(1-gamma) truncated to {1..K-1} (exact categorical)."""
        K = self.config['sequence_length']
        gamma = self.config['discount']
        logits = jnp.arange(K - 1) * jnp.log(gamma)  # delta = 1..K-1
        return 1 + jax.random.categorical(rng, logits, shape=shape)

    # -------------------------------------------------------------- losses --
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Public loss (main.py validation logs every info entry, so the
        [k]-shaped EMA aux from _loss_with_aux is stripped here)."""
        loss, info = self._loss_with_aux(batch, grad_params, rng=rng)
        info = dict(info)
        info.pop('aux_batch_pz')
        return loss, info

    def _loss_with_aux(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        k_z, k_t, k_d, k_zq = jax.random.split(rng, 4)

        obs_seq = batch['observations_seq']  # [B, K, D]
        seq_mask = batch['seq_mask']  # [B, K]
        B, K, D = obs_seq.shape
        k = self.config['num_skills']
        eye = jnp.eye(k)
        # Only complete windows count (a window crossing a trajectory end has
        # a clamped tail and would corrupt the truncated-discounted target).
        w = seq_mask[:, -1]
        w_norm = w / jnp.clip(w.sum(), 1.0, None)

        post = self._posteriors(obs_seq, seq_mask)  # [B, k]

        # --- M-step on the step dynamics d (one random transition per window).
        z_dyn = jax.random.categorical(k_z, jnp.log(jnp.clip(post, 1e-12, None)), axis=-1)  # [B]
        t = jax.random.randint(k_t, (B,), 0, K - 1)
        bidx = jnp.arange(B)
        st = obs_seq[bidx, t]
        delta = obs_seq[bidx, t + 1] - st
        cond = jnp.concatenate([st, eye[z_dyn]], axis=-1)
        dyn_nll = -mdn_log_prob(*self.network.select('dyn')(cond, params=grad_params), delta)
        dyn_loss = (w_norm * dyn_nll).sum()
        info['dyn_loss'] = dyn_loss

        # --- Phase-2: discounted channel q and skill marginal m. The whole
        # branch (including the bracket diagnostic) is skipped via lax.cond
        # during EM rather than merely zero-weighted.
        in_phase2 = self.network.step >= self.config['em_steps']
        s0 = obs_seq[:, 0]

        def phase2_branch(_):
            d_off = self._sample_trunc_geom(k_d, (B,))
            target = obs_seq[bidx, d_off] - s0
            z_q = jax.random.categorical(k_zq, jnp.log(jnp.clip(post, 1e-12, None)), axis=-1)
            q_cond = jnp.concatenate([s0, eye[z_q]], axis=-1)
            q_nll = -mdn_log_prob(*self.network.select('q')(q_cond, params=grad_params), target)
            q_loss = (w_norm * q_nll).sum()
            logm = jax.nn.log_softmax(self.network.select('clf')(s0, params=grad_params), axis=-1)
            clf_loss = (w_norm * -(post * logm).sum(-1)).sum()
            # Running effective-empowerment estimate on the (paired) batch draws.
            b = self._bracket(s0, eye[z_q], target, params=None)
            mi = (w_norm * b).sum()
            return q_loss, clf_loss, mi

        zero = jnp.float32(0.0)
        q_loss, clf_loss, mi = jax.lax.cond(
            in_phase2, phase2_branch, lambda _: (zero, zero, zero), None
        )
        info['q_loss'] = q_loss
        info['clf_loss'] = clf_loss
        info['mi_estimate'] = mi  # 0 during EM
        info['mi_cap_log_k'] = jnp.log(float(k))

        in_phase2_f = in_phase2.astype(jnp.float32)
        info['phase/in_phase2'] = in_phase2_f
        loss = (1.0 - in_phase2_f) * dyn_loss + q_loss + clf_loss

        # Diagnostics.
        info['p_z_min'] = self.p_z.min()
        info['p_z_max'] = self.p_z.max()
        info['p_z_entropy'] = -(self.p_z * jnp.log(jnp.clip(self.p_z, 1e-12, None))).sum()
        info['post_entropy'] = (w_norm * -(post * jnp.log(jnp.clip(post, 1e-12, None))).sum(-1)).sum()
        # Batch posterior mean for update()'s p_z EMA (popped there; avoids a
        # second E-step). Weighted by complete windows only.
        info['aux_batch_pz'] = (w[:, None] * post).sum(0) / jnp.clip(w.sum(), 1.0, None)
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        # Snapshot dyn params: hard-frozen after em_steps (defends against Adam
        # momentum carry-over, same trick as agents/empowerment_crl.py).
        old_dyn = self.network.params['modules_dyn']
        in_phase2 = self.network.step >= self.config['em_steps']

        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self._loss_with_aux(batch, p, rng=rng)
        )
        frozen_dyn = jax.tree_util.tree_map(
            lambda old, new: jnp.where(in_phase2, old, new),
            old_dyn,
            new_network.params['modules_dyn'],
        )
        new_network = new_network.replace(params={**new_network.params, 'modules_dyn': frozen_dyn})

        # Prior EMA from the batch posterior mean computed inside total_loss
        # (complete windows only); frozen together with dyn in phase 2.
        batch_pz = info.pop('aux_batch_pz')
        tau = self.config['p_z_ema']
        new_p_z = jnp.where(in_phase2, self.p_z, tau * self.p_z + (1.0 - tau) * batch_pz)
        new_p_z = new_p_z / new_p_z.sum()

        return self.replace(network=new_network, rng=new_rng, p_z=new_p_z), info

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
        assert config.get('encoder') is None, 'empowerment_opal_dads is state-based only (no encoder support)'
        config['action_dim'] = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]
        config['obs_dim'] = obs_dim
        k = config['num_skills']

        dyn_def = MDN(obs_dim, config['num_components'], tuple(config['mdn_hidden_dims']),
                      layer_norm=config['layer_norm'])
        q_def = MDN(obs_dim, config['num_components'], tuple(config['mdn_hidden_dims']),
                    layer_norm=config['layer_norm'])
        clf_def = MLP((*config['clf_hidden_dims'], k), activate_final=False,
                      layer_norm=config['layer_norm'])

        ex_cond = jnp.zeros((*ex_observations.shape[:-1], obs_dim + k))
        network_info = dict(
            dyn=(dyn_def, (ex_cond,)),
            q=(q_def, (ex_cond,)),
            clf=(clf_def, (ex_observations,)),
        )
        networks = {n: v[0] for n, v in network_info.items()}
        network_args = {n: v[1] for n, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        p_z = jnp.full((k,), 1.0 / k)
        return cls(rng, network=network, p_z=p_z, config=flax.core.FrozenDict(**config))


# ------------------------------------------------------------- BA adapter --


class _BAModel:
    """Duck-type adapter: exposes a trained agent's channel q to
    empowerment.ba_capacity.capacity_ba (Algorithm 3).

    The agent's params are baked into the jit closures at construction time:
    build the adapter AFTER training, and rebuild it if the agent object is
    ever replaced (it does not track further updates).
    """

    def __init__(self, agent):
        self._agent = agent
        self.k = agent.config['num_skills']
        self.eye = np.eye(self.k, dtype=np.float32)
        self.q_state = {'params': None}  # params live in the agent's network

        @partial(jax.jit, static_argnames=('n',))
        def q_sample(_params, cond, key, n):
            rep = jnp.repeat(cond, n, axis=0)
            heads = agent.network.select('q')(rep)
            keys = jax.random.split(key, rep.shape[0])
            out = jax.vmap(mdn_sample)(keys, *heads)
            return out.reshape(cond.shape[0], n, -1)

        @jax.jit
        def q_ll_allz(_params, s, target):
            return agent._q_ll_allz(s, target)

        self._q_sample = q_sample
        self._q_ll_allz = q_ll_allz


def make_ba_model(agent):
    """Adapt a trained EmpowermentOPALDADSAgent for capacity_ba:

        from empowerment.ba_capacity import capacity_ba
        C, w, iters = capacity_ba(make_ba_model(agent), s, key)
    """
    return _BAModel(agent)


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='empowerment_opal_dads',
        # Matched to agents/empowerment_skill.py for a like-for-like comparison
        # (same lr, batch size, trunk architecture, layer norm, skill count).
        lr=3e-4,
        batch_size=1024,
        num_skills=15,           # k; MI cap = log k (empowerment_skill uses 15)
        num_components=8,        # MDN mixture components
        mdn_hidden_dims=(512, 512, 512),
        clf_hidden_dims=(512, 512, 512),
        layer_norm=True,
        em_steps=100000,         # stochastic-EM steps before phase 2
        p_z_ema=0.999,           # prior EMA rate during EM
        est_num_joints=16,       # M draws per state in empowerment()
        # main.py / SequenceDataset wiring. The value_/actor_ goal keys below
        # are required by GCDataset's __post_init__ asserts (SequenceDataset
        # subclasses it) but this agent never reads the sampled goals.
        sequence_length=10,      # K (chunk length)
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
