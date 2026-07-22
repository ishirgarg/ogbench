"""CRL effective empowerment (action-level InfoNCE) for offline OGBench datasets.

Estimates I(s+; a | s) where s+ is the discounted future state (Delta ~
Geometric(1 - GAMMA), same trajectory), via an InfoNCE lower bound with a
learned critic T(s+, a, s). Negatives are i.i.d. samples from the behavior
policy pi_beta(.|s) at the SAME state s, modeled with flow-matching BC.

Components:
  - pi_bc(a|s): rectified-flow BC policy (sampler only).
  - f_dyn(s+|s,a): rectified-flow model of the discounted future (sampler only).
  - T(s+, a, s): MLP critic, scalar output.

All logs are natural logs; estimates are in nats.

Caveats on the "lower bound" claim (know what the numbers mean):
  - Negatives come from the LEARNED pi_bc, not the true pi_beta. The estimator
    therefore lower-bounds I(s+; a | s) + E_s[KL(pi_beta(.|s) || pi_bc(.|s))]:
    BC model error inflates the estimate (cuts upward), it does not deflate it.
    The strict lower-bound guarantee holds only in the limit pi_bc = pi_beta.
    (Clipping BC samples to the action box creates boundary atoms matching
    saturated dataset actions; residual mass mismatch at the boundary folds
    into the same KL term.)
  - Arbitrary-state mode draws joints from pi_bc x f_dyn, so it lower-bounds
    the MI of the LEARNED models ("model empowerment"), which can sit on
    either side of the true empowerment (an under-dispersed f_dyn pushes it
    above the truth).
  - The trajectory-truncated geometric future makes the estimand the
    occurrence-weighted truncated discounted empowerment; the uniform fallback
    fires only within a few steps of trajectory ends, where it nearly
    coincides with the truncated geometric law.
"""

import functools
import pickle
from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from empowerment.common import sample_discounted_future_idxs


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """Multi-layer perceptron (matches repo idiom)."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
        return x


class VelocityField(nn.Module):
    """Rectified-flow velocity field v(x_t, t, cond)."""

    hidden_dims: Sequence[int]
    out_dim: int

    @nn.compact
    def __call__(self, x_t, t, cond):
        # t: [..., 1]
        h = jnp.concatenate([x_t, t, cond], axis=-1)
        h = MLP(self.hidden_dims, activate_final=True)(h)
        return nn.Dense(self.out_dim, kernel_init=default_init())(h)


class Critic(nn.Module):
    """Scalar critic T(s+, a, s)."""

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, s_plus, a, s):
        h = jnp.concatenate([s_plus, a, s], axis=-1)
        h = MLP(self.hidden_dims, activate_final=True)(h)
        return nn.Dense(1, kernel_init=default_init())(h).squeeze(-1)


def flow_matching_loss(apply_fn, params, x, cond, rng):
    """Rectified-flow MSE loss: regress v(x_t, t, cond) onto (x - x_0)."""
    k0, k1 = jax.random.split(rng)
    x0 = jax.random.normal(k0, x.shape)
    t = jax.random.uniform(k1, (*x.shape[:-1], 1))
    x_t = (1 - t) * x0 + t * x
    v = apply_fn(params, x_t, t, cond)
    return jnp.mean(jnp.square(v - (x - x0)))


def flow_sample(apply_fn, params, cond, rng, out_dim, num_steps=16, clip=None):
    """Euler-integrate the flow from x_0 ~ N(0, I). cond: [..., C]."""
    x = jax.random.normal(rng, (*cond.shape[:-1], out_dim))
    dt = 1.0 / num_steps

    def step(i, x):
        t = jnp.full((*cond.shape[:-1], 1), i * dt)
        return x + dt * apply_fn(params, x, t, cond)

    x = jax.lax.fori_loop(0, num_steps, step, x)
    if clip is not None:
        x = jnp.clip(x, -clip, clip)
    return x


class CRLEmpowerment:
    """Bundles pi_bc, f_dyn, and the InfoNCE critic T with jitted train/eval fns."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        seed=0,
        hidden_dims=(512, 512, 512),
        lr=3e-4,
        num_negatives=63,
        num_flow_steps=16,
        action_clip=1.0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_negatives = num_negatives
        self.num_flow_steps = num_flow_steps
        self.action_clip = action_clip
        self.rng = jax.random.PRNGKey(seed)

        self.bc_model = VelocityField(hidden_dims, act_dim)
        self.dyn_model = VelocityField(hidden_dims, obs_dim)
        self.critic_model = Critic(hidden_dims)

        k_bc, k_dyn, k_T, self.rng = jax.random.split(self.rng, 4)
        s = jnp.zeros((1, obs_dim))
        a = jnp.zeros((1, act_dim))
        t1 = jnp.zeros((1, 1))
        tx = optax.adam(lr)
        self.bc = train_state.TrainState.create(
            apply_fn=self.bc_model.apply, params=self.bc_model.init(k_bc, a, t1, s), tx=tx
        )
        self.dyn = train_state.TrainState.create(
            apply_fn=self.dyn_model.apply,
            params=self.dyn_model.init(k_dyn, s, t1, jnp.concatenate([s, a], -1)),
            tx=tx,
        )
        self.critic = train_state.TrainState.create(
            apply_fn=self.critic_model.apply, params=self.critic_model.init(k_T, s, a, s), tx=tx
        )
        self._build_jitted()

    # ------------------------------------------------------------------ jit --
    def _build_jitted(self):
        bc_apply = self.bc_model.apply
        dyn_apply = self.dyn_model.apply
        critic_apply = self.critic_model.apply
        N = self.num_negatives
        num_flow_steps = self.num_flow_steps
        act_dim = self.act_dim
        obs_dim = self.obs_dim
        clip = self.action_clip

        @jax.jit
        def bc_step(state, s, a, rng):
            loss, grads = jax.value_and_grad(
                lambda p: flow_matching_loss(bc_apply, p, a, s, rng)
            )(state.params)
            return state.apply_gradients(grads=grads), loss

        @jax.jit
        def dyn_step(state, s, a, sp, rng):
            cond = jnp.concatenate([s, a], -1)
            loss, grads = jax.value_and_grad(
                lambda p: flow_matching_loss(dyn_apply, p, sp, cond, rng)
            )(state.params)
            return state.apply_gradients(grads=grads), loss

        @jax.jit
        def sample_bc(bc_params, s, rng):
            return flow_sample(bc_apply, bc_params, s, rng, act_dim, num_flow_steps, clip=clip)

        @jax.jit
        def sample_dyn(dyn_params, s, a, rng):
            cond = jnp.concatenate([s, a], -1)
            return flow_sample(dyn_apply, dyn_params, cond, rng, obs_dim, num_flow_steps)

        def _b_values(critic_params, bc_params, s, a, sp, rng):
            """Per-element InfoNCE bound b = log(N+1) + log_softmax(logits)[0].

            Each element's N negatives are sampled from pi_bc conditioned on
            that element's OWN s (negatives are never shared across elements).
            """
            B = s.shape[0]
            s_rep = jnp.broadcast_to(s[:, None, :], (B, N, s.shape[-1]))  # [B, N, D]
            neg = flow_sample(bc_apply, bc_params, s_rep, rng, act_dim, num_flow_steps, clip=clip)
            pos_logit = critic_apply(critic_params, sp, a, s)  # [B]
            neg_logits = critic_apply(
                critic_params,
                jnp.broadcast_to(sp[:, None, :], (B, N, sp.shape[-1])),
                neg,
                s_rep,
            )  # [B, N]
            logits = jnp.concatenate([pos_logit[:, None], neg_logits], axis=1)  # [B, N+1]
            return jnp.log(N + 1.0) + jax.nn.log_softmax(logits, axis=-1)[:, 0]

        @jax.jit
        def critic_step(state, bc_params, s, a, sp, rng):
            def loss_fn(p):
                b = _b_values(p, bc_params, s, a, sp, rng)
                return jnp.log(N + 1.0) - jnp.mean(b)  # = mean(-log_softmax[0])

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss

        @jax.jit
        def b_values(critic_params, bc_params, s, a, sp, rng):
            return _b_values(critic_params, bc_params, s, a, sp, rng)

        self._bc_step = bc_step
        self._dyn_step = dyn_step
        self._sample_bc = sample_bc
        self._sample_dyn = sample_dyn
        self._critic_step = critic_step
        self._b_values = b_values

    def _next_rng(self):
        self.rng, k = jax.random.split(self.rng)
        return k

    # ------------------------------------------------------- training steps --
    def bc_train_step(self, s, a):
        self.bc, loss = self._bc_step(self.bc, s, a, self._next_rng())
        return float(loss)

    def dyn_train_step(self, s, a, sp):
        self.dyn, loss = self._dyn_step(self.dyn, s, a, sp, self._next_rng())
        return float(loss)

    def critic_train_step(self, s, a, sp):
        self.critic, loss = self._critic_step(
            self.critic, self.bc.params, s, a, sp, self._next_rng()
        )
        return float(loss)

    def critic_eval_loss(self, s, a, sp):
        """InfoNCE loss (no grad) on a batch, with fresh negatives."""
        b = self._b_values(self.critic.params, self.bc.params, s, a, sp, self._next_rng())
        return float(np.log(self.num_negatives + 1.0) - np.mean(np.asarray(b)))

    # ---------------------------------------------------------- sampling api --
    def sample_bc(self, s):
        return np.asarray(self._sample_bc(self.bc.params, jnp.asarray(s), self._next_rng()))

    def sample_dyn(self, s, a):
        return np.asarray(
            self._sample_dyn(self.dyn.params, jnp.asarray(s), jnp.asarray(a), self._next_rng())
        )

    def compute_b(self, s, a, sp):
        """Per-element InfoNCE bound b for jointly sampled (s, a, s+) triples."""
        return np.asarray(
            self._b_values(
                self.critic.params,
                self.bc.params,
                jnp.asarray(s),
                jnp.asarray(a),
                jnp.asarray(sp),
                self._next_rng(),
            )
        )

    # -------------------------------------------------------------- estimates --
    def estimate_crl(self, s=None, data=None, t=None, M=64, np_rng=None):
        """Estimate empowerment at a single state. Returns (mean_nats, stderr).

        Dataset mode (data + t given): joints are the real action a_t paired
        with M fresh geometric s+ draws from t's trajectory. Note this
        conditions on the SINGLE recorded action a_t: unbiased for the bound
        only in expectation over a_t ~ pi_beta(.|s), and the returned stderr
        covers s+/negative noise but NOT the action draw (0 degrees of freedom
        in a). Per-state values can exceed the per-state MI; aggregates over
        many states remain valid.
        Arbitrary-state mode (s given): a_j ~ pi_bc(.|s), s+_j ~ f_dyn(s, a_j)
        sampled FROM that a_j (pairing preserved). This estimates MODEL
        empowerment (MI of the learned pi_bc x f_dyn); see module docstring.
        """
        if t is not None:
            assert data is not None
            if np_rng is None:
                np_rng = np.random.default_rng()
            idxs = np.full(M, t, dtype=np.int64)
            sp_idxs = sample_discounted_future_idxs(data, idxs, np_rng)
            s_batch = data.observations[idxs]
            a_batch = data.actions[idxs]
            sp_batch = data.observations[sp_idxs]
        else:
            s = np.asarray(s, dtype=np.float32).reshape(1, -1)
            s_batch = np.repeat(s, M, axis=0)
            a_batch = self.sample_bc(s_batch)
            sp_batch = self.sample_dyn(s_batch, a_batch)
        b = self.compute_b(s_batch, a_batch, sp_batch)
        return float(np.mean(b)), float(np.std(b) / np.sqrt(M))

    def estimate_global(self, data, num=50000, batch_size=1000, np_rng=None, shuffle=False):
        """Mean InfoNCE bound over `num` random dataset triples.

        shuffle ablations (pairing broken -> result should collapse to <= ~0;
        the exact null is slightly NEGATIVE by Jensen, not exactly 0):
          - 'bc': replace each positive action with a fresh pi_bc(.|s) sample
            at the element's OWN s. The positive becomes exchangeable with the
            negatives, so this is the surgical null for the bound (and also
            directly probes pi_bc/pi_beta mismatch).
          - 'permute' (or True): cyclically shift actions across the batch (a
            derangement: no element keeps its own action). Also breaks the
            a<->s link, so strongly state-dependent pi_beta can push this
            further negative for reasons unrelated to future information.
        Returns (mean_nats, stderr).
        """
        if np_rng is None:
            np_rng = np.random.default_rng()
        if shuffle is True:
            shuffle = 'permute'
        bs = []
        for _ in range(int(np.ceil(num / batch_size))):
            idxs = data.random_nonfinal_idxs(batch_size, np_rng)
            sp_idxs = sample_discounted_future_idxs(data, idxs, np_rng)
            s = data.observations[idxs]
            a = data.actions[idxs]
            if shuffle == 'permute':
                # Cycle actions along a random ordering: position perm[j] gets
                # perm[j-1]'s action, so no element keeps its own (derangement).
                perm = np_rng.permutation(batch_size)
                shuffled = np.empty_like(a)
                shuffled[perm] = a[np.roll(perm, 1)]
                a = shuffled
            elif shuffle == 'bc':
                a = self.sample_bc(s)
            elif shuffle:
                raise ValueError(f'unknown shuffle mode: {shuffle!r}')
            bs.append(self.compute_b(s, a, data.observations[sp_idxs]))
        b = np.concatenate(bs)[:num]
        return float(np.mean(b)), float(np.std(b) / np.sqrt(len(b)))

    # ------------------------------------------------------------ save/load --
    # NOTE: save/load round-trips network params ONLY (no optimizer state or
    # RNG key). Restoring is exact for inference; resuming training restarts
    # Adam moments from zero, so expect a transient loss bump.
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(
                {
                    'bc': jax.device_get(self.bc.params),
                    'dyn': jax.device_get(self.dyn.params),
                    'critic': jax.device_get(self.critic.params),
                },
                f,
            )

    def load(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.bc = self.bc.replace(params=params['bc'])
        self.dyn = self.dyn.replace(params=params['dyn'])
        self.critic = self.critic.replace(params=params['critic'])


def sample_bc_batch(data, batch_size, np_rng):
    """(s, a) batch over ALL flat indices (final states' actions are valid pairs)."""
    idxs = np_rng.integers(0, data.size, size=batch_size)
    return data.observations[idxs], data.actions[idxs]


def sample_triple_batch(data, batch_size, np_rng):
    """Jointly sampled (s, a, s+) triples via the geometric future sampler."""
    idxs = data.random_nonfinal_idxs(batch_size, np_rng)
    sp_idxs = sample_discounted_future_idxs(data, idxs, np_rng)
    return data.observations[idxs], data.actions[idxs], data.observations[sp_idxs]
