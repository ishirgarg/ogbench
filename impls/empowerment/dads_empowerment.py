"""Algorithm 2: DADS/OPAL-style effective empowerment with discrete skills.

Phase 1: EM skill discovery over non-overlapping length-K chunks
    (categorical prior p(z) + a single per-skill-conditioned MDN step
    dynamics d(s_{t+1} | s_t, z)).
Phase 2: skill-marginal classifier m(z|s) (soft-label cross-entropy on the
    EM posteriors) and a SECOND MDN for the chunk-truncated discounted
    channel q(s+ | s, z).
Phase 3: Barber-Agakov estimator with exact k-term z-sums:
    b = log q(s+|s,z) - logsumexp_z'(log m(z'|s) + log q(s+|s,z')).

All densities use natural log (nats); gamma = 0.99.

Bias characterization (know what the numbers mean). With r(z|x,s) defined by
r ~ m(z|s) q(x|s,z), the bracket satisfies
    E_joint[b] = I(s+; z | s) - E_x[KL(p(z|x,s) || r(z|x,s))]
                              + KL(p(z|s) || m(.|s)).
So channel-model (q) error only LOWERS the estimate (classic Barber-Agakov
direction), but skill-marginal (m) error RAISES it: the estimate is a lower
bound on I only up to a +KL(p(z|s) || m) inflation term, and pointwise
b <= -log m(z|s), which can exceed log k wherever m(z|s) < 1/k. Report the
m-misfit KL alongside per-state estimates (run_dads does).

Per-state interpretation: with continuous states, s a.s. identifies its chunk,
under which exact conditioning z and s+ are independent given s and the
literal per-state conditional MI degenerates to 0. Nonzero per-state values
are meaningful as the MI of the function-approximator-SMOOTHED channel (q and
m pooling over chunks with nearby start states) -- the same implicit
coarse-graining DADS relies on.

Parameterization note: both MDNs model the DELTA x - s rather than x itself.
This is a unit-Jacobian reparameterization, so every reported quantity (all
are differences of log densities at a fixed (s, x)) is identical to the
raw-state parameterization, but the regression is far better conditioned.
The runner additionally standardizes observations once at load time; a
uniform affine rescaling likewise cancels in all log-density differences.
"""

import pickle
from functools import partial
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from empowerment.common import GAMMA, sample_geometric_offsets

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def default_init(scale=1.0):
    """Default kernel initializer (matches impls/utils/networks.py)."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """Multi-layer perceptron (matches impls/utils/networks.py idiom)."""

    hidden_dims: Sequence[int]
    activate_final: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.gelu(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class MDN(nn.Module):
    """Diagonal-Gaussian mixture density network.

    Heads: mixture logits, means, log-stds clamped to [LOG_STD_MIN, LOG_STD_MAX].
    """

    out_dim: int
    num_components: int = 8
    hidden_dims: Sequence[int] = (256, 256)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, cond):
        h = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(cond)
        C, D = self.num_components, self.out_dim
        logits = nn.Dense(C, kernel_init=default_init())(h)
        means = nn.Dense(C * D, kernel_init=default_init(1e-2))(h)
        means = means.reshape(h.shape[:-1] + (C, D))
        log_stds = nn.Dense(C * D, kernel_init=default_init(1e-2))(h)
        log_stds = jnp.clip(log_stds.reshape(h.shape[:-1] + (C, D)), LOG_STD_MIN, LOG_STD_MAX)
        return logits, means, log_stds


def mdn_log_prob(logits, means, log_stds, x):
    """Exact mixture log-density, evaluated in log space with logsumexp."""
    log_w = jax.nn.log_softmax(logits, axis=-1)
    diff = (x[..., None, :] - means) / jnp.exp(log_stds)
    comp_lp = -0.5 * (diff**2 + 2.0 * log_stds + jnp.log(2.0 * jnp.pi)).sum(-1)
    return jax.nn.logsumexp(log_w + comp_lp, axis=-1)


def mdn_sample(key, logits, means, log_stds):
    """Ancestral sample from one MDN head triple (unbatched or batched)."""
    k1, k2 = jax.random.split(key)
    comp = jax.random.categorical(k1, logits)
    mu = jnp.take_along_axis(means, comp[..., None, None], axis=-2)[..., 0, :]
    std = jnp.exp(jnp.take_along_axis(log_stds, comp[..., None, None], axis=-2)[..., 0, :])
    return mu + std * jax.random.normal(k2, mu.shape)


def np_logsumexp(x, axis):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def kmeans(x, k, rng, iters=100):
    """Plain numpy Lloyd's k-means (sklearn is unavailable in this env)."""
    n = len(x)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    assign = np.full(n, -1, dtype=np.int64)
    for _ in range(iters):
        d2 = ((x[:, None, :] - centers[None]) ** 2).sum(-1)
        new = d2.argmin(1)
        if (new == assign).all():
            break
        assign = new
        for j in range(k):
            m = assign == j
            centers[j] = x[m].mean(0) if m.any() else x[rng.integers(n)]
    return assign, centers


def make_chunks(data, K):
    """Non-overlapping length-K state chunks per trajectory; short remainders dropped.

    Returns [Nc, K, obs_dim] float32.
    """
    out = []
    for s, e in zip(data.initial_locs, data.terminal_locs):
        n = (e - s + 1) // K
        if n > 0:
            out.append(data.observations[s : s + n * K].reshape(n, K, -1))
    return np.concatenate(out, axis=0)


def sample_categorical_rows(p, rng):
    """Vectorized per-row categorical sampling; p is [B, k] (rows sum to 1)."""
    u = rng.random((len(p), 1))
    return np.minimum((p.cumsum(1) < u).sum(1), p.shape[1] - 1)


class DADS:
    """Holds all networks/params and provides EM, phase-2 training, and estimation."""

    def __init__(self, obs_dim, k, K, seed=0, num_components=8, hidden_dims=(256, 256), lr=3e-4):
        self.obs_dim, self.k, self.K = obs_dim, k, K
        self.num_components, self.hidden_dims, self.lr = num_components, tuple(hidden_dims), lr
        self.eye = np.eye(k, dtype=np.float32)
        self.p_z = np.full(k, 1.0 / k)
        # Revival bookkeeping for the collapse guard: a skill that dies again
        # after `max_revive_attempts` reseeds is declared unsupported by the
        # data and left dead (endless forced revival just injects label churn).
        self.revive_attempts = np.zeros(k, dtype=np.int64)
        self.max_revive_attempts = 3

        self.dyn = MDN(obs_dim, num_components, self.hidden_dims)  # phase-1 step dynamics d
        self.qnet = MDN(obs_dim, num_components, self.hidden_dims)  # phase-2 discounted channel q
        self.clf = MLP(self.hidden_dims + (k,))  # phase-2 skill marginal m(z|s) (logits)

        key = jax.random.PRNGKey(seed)
        kd, kq, kc = jax.random.split(key, 3)
        cond = jnp.zeros((1, obs_dim + k))
        s_in = jnp.zeros((1, obs_dim))
        self.tx = optax.adam(lr)
        self.dyn_state = self._init_state(self.dyn, kd, cond)
        self.q_state = self._init_state(self.qnet, kq, cond)
        self.clf_state = self._init_state(self.clf, kc, s_in)
        self._build_jits()

    def _init_state(self, module, key, sample_in):
        params = module.init(key, sample_in)['params']
        return {'params': params, 'opt': self.tx.init(params)}

    def _build_jits(self):
        dyn, qnet, clf, tx, k = self.dyn, self.qnet, self.clf, self.tx, self.k
        jeye = jnp.eye(k, dtype=jnp.float32)

        def mdn_step(module):
            @jax.jit
            def step(state, cond, target):
                def loss_fn(p):
                    return -mdn_log_prob(*module.apply({'params': p}, cond), target).mean()

                loss, grads = jax.value_and_grad(loss_fn)(state['params'])
                updates, opt = tx.update(grads, state['opt'], state['params'])
                return {'params': optax.apply_updates(state['params'], updates), 'opt': opt}, loss

            return step

        self._dyn_step = mdn_step(dyn)
        self._q_step = mdn_step(qnet)

        @jax.jit
        def clf_step(state, s, soft_labels):
            def loss_fn(p):
                logp = jax.nn.log_softmax(clf.apply({'params': p}, s), axis=-1)
                return -(soft_labels * logp).sum(-1).mean()

            loss, grads = jax.value_and_grad(loss_fn)(state['params'])
            updates, opt = tx.update(grads, state['opt'], state['params'])
            return {'params': optax.apply_updates(state['params'], updates), 'opt': opt}, loss

        self._clf_step = clf_step

        def all_z_ll(module):
            @jax.jit
            def f(params, s, target):
                def per_z(zoh):
                    cond = jnp.concatenate([s, jnp.tile(zoh, (s.shape[0], 1))], axis=-1)
                    return mdn_log_prob(*module.apply({'params': params}, cond), target)

                return jax.vmap(per_z)(jeye)  # [k, B]

            return f

        self._dyn_ll_allz = all_z_ll(dyn)
        self._q_ll_allz = all_z_ll(qnet)

        @jax.jit
        def bracket(q_params, clf_params, s, zoh, x_delta):
            cond = jnp.concatenate([s, zoh], axis=-1)
            lq_sel = mdn_log_prob(*qnet.apply({'params': q_params}, cond), x_delta)
            logm = jax.nn.log_softmax(clf.apply({'params': clf_params}, s), axis=-1)  # [B, k]

            def per_z(ze):
                c = jnp.concatenate([s, jnp.tile(ze, (s.shape[0], 1))], axis=-1)
                return mdn_log_prob(*qnet.apply({'params': q_params}, c), x_delta)

            lq_all = jax.vmap(per_z)(jeye)  # [k, B]
            log_mix = jax.nn.logsumexp(logm.T + lq_all, axis=0)
            return lq_sel - log_mix

        self._bracket = bracket

        @jax.jit
        def q_heads(q_params, cond):
            return qnet.apply({'params': q_params}, cond)

        self._q_heads = q_heads

        @jax.jit
        def clf_probs(clf_params, s):
            return jax.nn.softmax(clf.apply({'params': clf_params}, s), axis=-1)

        self._clf_probs = clf_probs

        @partial(jax.jit, static_argnames=('n',))
        def q_sample(q_params, cond, key, n):
            """n samples (deltas) from q for each row of cond [R, cd] -> [R, n, D]."""
            rep = jnp.repeat(cond, n, axis=0)
            heads = qnet.apply({'params': q_params}, rep)
            keys = jax.random.split(key, rep.shape[0])
            out = jax.vmap(mdn_sample)(keys, *heads)
            return out.reshape(cond.shape[0], n, -1)

        self._q_sample = q_sample

    # ------------------------------------------------------------------ Phase 1

    def e_step(self, chunks, block=1 << 16):
        """Exact Bayes posteriors over skills for every chunk.

        Returns (post [Nc, k], avg_chunk_ll (marginal), chunk_ll [Nc]).
        """
        Nc, K, D = chunks.shape
        st = np.ascontiguousarray(chunks[:, :-1].reshape(-1, D))
        target = np.ascontiguousarray(chunks[:, 1:].reshape(-1, D)) - st
        rows = st.shape[0]
        ll = np.empty((self.k, rows), np.float32)
        for lo in range(0, rows, block):
            hi = min(rows, lo + block)
            pad = block - (hi - lo)
            s_b = np.concatenate([st[lo:hi], np.zeros((pad, D), np.float32)]) if pad else st[lo:hi]
            t_b = np.concatenate([target[lo:hi], np.zeros((pad, D), np.float32)]) if pad else target[lo:hi]
            out = np.asarray(self._dyn_ll_allz(self.dyn_state['params'], s_b, t_b))
            ll[:, lo:hi] = out[:, : hi - lo]
        score = ll.reshape(self.k, Nc, K - 1).sum(-1).T + np.log(self.p_z)[None]  # [Nc, k]
        chunk_ll = np_logsumexp(score, axis=1)
        post = np.exp(score - chunk_ll[:, None])
        return post, float(chunk_ll.mean()), chunk_ll

    def update_prior_and_reseed(self, post, chunk_ll):
        """p(z) <- mean posterior; skills with p(z) < 0.5/k get reseeded with the
        worst-likelihood chunks (hard-assigned in post).

        Seed mass is 1/k per dead skill -- deliberately ABOVE the 0.5/k revival
        threshold (a seed below the threshold can never directly resurrect a
        skill). Hard-assigning rows transiently removes soft mass from healthy
        skills; callers must follow reseed -> M-step with a closing E-step so
        the stored posteriors are exact under the final dynamics.

        Each skill gets at most `max_revive_attempts` reseeds; a skill that
        keeps dying is declared unsupported by the data and left dead (the
        effective skill count is then < k; H(p_z) is the honest MI ceiling).
        Returns (post, reseeded_skills, given_up_skills).
        """
        p_z = post.mean(0)
        dead = np.nonzero(p_z < 0.5 / self.k)[0]
        revive = [int(z) for z in dead if self.revive_attempts[z] < self.max_revive_attempts]
        given_up = [int(z) for z in dead if self.revive_attempts[z] >= self.max_revive_attempts]
        if revive:
            n_seed = max(64, len(post) // self.k)
            worst = np.argsort(chunk_ll)  # ascending: worst first
            ptr = 0
            for z in revive:
                idxs = worst[ptr : ptr + n_seed]
                ptr += n_seed
                post[idxs] = 0.0
                post[idxs, z] = 1.0
                self.revive_attempts[z] += 1
            p_z = post.mean(0)
        self.p_z = p_z
        return post, revive, given_up

    def m_step(self, chunks, post, steps, batch, rng):
        """Refit d with weighted NLL via posterior sampling of z."""
        Nc, K, _ = chunks.shape
        losses = []
        for _ in range(steps):
            i = rng.integers(0, Nc, batch)
            t = rng.integers(0, K - 1, batch)
            z = sample_categorical_rows(post[i], rng)
            st = chunks[i, t]
            delta = chunks[i, t + 1] - st
            cond = np.concatenate([st, self.eye[z]], axis=1)
            self.dyn_state, loss = self._dyn_step(self.dyn_state, cond, delta)
            losses.append(float(loss))
        return float(np.mean(losses[-50:]))

    # ------------------------------------------------------------------ Phase 2

    def train_phase2(self, chunks, post, q_steps, clf_steps, batch, rng, gamma=GAMMA):
        """Train the discounted channel q(s+|s,z) and the skill marginal m(z|s)."""
        Nc, K, _ = chunks.shape
        maxoff = np.full(batch, K - 1, dtype=np.int64)
        q_losses, clf_losses = [], []
        for _ in range(q_steps):
            i = rng.integers(0, Nc, batch)
            z = sample_categorical_rows(post[i], rng)
            d = sample_geometric_offsets(rng, maxoff, gamma=gamma)
            s0 = chunks[i, 0]
            target = chunks[i, d] - s0
            cond = np.concatenate([s0, self.eye[z]], axis=1)
            self.q_state, loss = self._q_step(self.q_state, cond, target)
            q_losses.append(float(loss))
        for _ in range(clf_steps):
            i = rng.integers(0, Nc, batch)
            self.clf_state, loss = self._clf_step(self.clf_state, chunks[i, 0], post[i])
            clf_losses.append(float(loss))
        return float(np.mean(q_losses[-50:])), float(np.mean(clf_losses[-50:]))

    # ------------------------------------------------------------------ Phase 3

    def bracket_np(self, s, z, x, block=1 << 14):
        """b = log q(x|s,z) - log_mix(x, s) for numpy batches (x is s+, raw)."""
        B, D = s.shape
        x_delta = x - s
        zoh = self.eye[z]
        out = np.empty(B, np.float32)
        for lo in range(0, B, block):
            hi = min(B, lo + block)
            pad = block - (hi - lo)

            def p(a):
                return np.concatenate([a[lo:hi], np.zeros((pad,) + a.shape[1:], np.float32)]) if pad else a[lo:hi]

            b = np.asarray(
                self._bracket(self.q_state['params'], self.clf_state['params'], p(s), p(zoh), p(x_delta))
            )
            out[lo:hi] = b[: hi - lo]
        return out

    def estimate_dads(self, chunks, post, idx, rng, M=64, gamma=GAMMA):
        """Effective-empowerment estimate at the dataset state where chunk idx starts.

        Jointly samples (z ~ post[idx], Delta ~ trunc-geometric, s+ = chunk[idx, Delta]).
        Returns (mean, stderr) in nats.
        """
        K = chunks.shape[1]
        z = sample_categorical_rows(np.tile(post[idx], (M, 1)), rng)
        d = sample_geometric_offsets(rng, np.full(M, K - 1, dtype=np.int64), gamma=gamma)
        s = np.tile(chunks[idx, 0], (M, 1))
        x = chunks[idx, d]
        b = self.bracket_np(s, z, x)
        return float(b.mean()), float(b.std(ddof=1) / np.sqrt(M))

    def estimate_dads_model(self, s, key, M=64):
        """Model-based estimate at an arbitrary state s: z ~ m(.|s), s+ ~ q(.|s,z) (paired)."""
        s = np.asarray(s, np.float32).reshape(1, -1)
        m = np.asarray(self._clf_probs(self.clf_state['params'], s))[0]
        k_np, k_sample = jax.random.split(key)
        rng = np.random.default_rng(int(jax.random.randint(k_np, (), 0, 2**31 - 1)))
        z = sample_categorical_rows(np.tile(m, (M, 1)), rng)
        cond = np.concatenate([np.tile(s, (M, 1)), self.eye[z]], axis=1)
        deltas = np.asarray(self._q_sample(self.q_state['params'], jnp.asarray(cond), k_sample, 1))[:, 0]
        sb = np.tile(s, (M, 1))
        b = self.bracket_np(sb, z, sb + deltas)
        return float(b.mean()), float(b.std(ddof=1) / np.sqrt(M))

    def global_estimate(self, chunks, post, rng, n=50_000, shuffle=False, gamma=GAMMA):
        """Global mean of the Barber-Agakov bracket over random chunk draws.

        shuffle=True breaks the (z, s+) pairing (mandatory validation; must
        give <= 0): for this density bracket the shuffled null is a NEGATIVE
        Jensen gap, and it is strongly negative when the per-skill channels are
        well separated (~0 only if they overlap heavily). s+ is drawn from a
        DIFFERENT random chunk than z's chunk -- z comes
        from chunk i's posterior while the (s, s+) pair comes from chunk j. The
        (s, s+) pair stays together so s+ remains in-distribution for the state
        being queried; only the skill pairing is broken.
        """
        Nc, K, _ = chunks.shape
        i = rng.integers(0, Nc, n)
        z = sample_categorical_rows(post[i], rng)
        d = sample_geometric_offsets(rng, np.full(n, K - 1, dtype=np.int64), gamma=gamma)
        src = rng.integers(0, Nc, n) if shuffle else i
        s = chunks[src, 0]
        x = chunks[src, d]
        b = self.bracket_np(s, z, x)
        return float(b.mean()), float(b.std(ddof=1) / np.sqrt(n))

    # ------------------------------------------------------------------ I/O

    def save(self, path):
        blob = {
            'config': dict(
                obs_dim=self.obs_dim,
                k=self.k,
                K=self.K,
                num_components=self.num_components,
                hidden_dims=self.hidden_dims,
                lr=self.lr,
            ),
            'p_z': self.p_z,
            'revive_attempts': self.revive_attempts,
            'dyn_params': jax.device_get(self.dyn_state['params']),
            'q_params': jax.device_get(self.q_state['params']),
            'clf_params': jax.device_get(self.clf_state['params']),
        }
        with open(path, 'wb') as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path, seed=0):
        with open(path, 'rb') as f:
            blob = pickle.load(f)
        model = cls(seed=seed, **blob['config'])
        model.p_z = blob['p_z']
        model.revive_attempts = blob.get('revive_attempts', np.zeros(blob['config']['k'], dtype=np.int64))
        for name, key in (('dyn_state', 'dyn_params'), ('q_state', 'q_params'), ('clf_state', 'clf_params')):
            state = getattr(model, name)
            state['params'] = jax.device_put(blob[key])
            state['opt'] = model.tx.init(state['params'])
        return model
