"""Algorithm 3: Blahut-Arimoto channel capacity of the Phase-2 channel q(s+|s,z).

Per-state fixed-point iteration over the skill input distribution w(z); no
training. Uses common random numbers: one pre-sampled [k, n_mc] bank of
channel outputs per (state, restart), with the full log-density tensor
LQ[z', z, m] = log q(S[z][m] | s, z') computed once in a single batched MDN
pass and reused across every BA iteration.

Capacity is reported in nats and is upper-bounded by log k.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _logsumexp(x, axis):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def capacity_ba(model, s, key, n_mc=128, tol=1e-4, max_iters=200):
    """Blahut-Arimoto capacity of q(.|s, z) over z in {0..k-1} at state s.

    Args:
        model: a trained dads_empowerment.DADS (phase 2 complete).
        s: [obs_dim] state (same normalization as training data).
        key: jax PRNG key for the sample bank.
        n_mc: Monte Carlo samples per skill.
    Returns:
        (C, w, iters): capacity in nats, the maximizing w(z), iterations used.
    """
    k = model.k
    s = np.asarray(s, np.float32).reshape(1, -1)
    cond = np.concatenate([np.tile(s, (k, 1)), model.eye], axis=1)  # [k, cd]

    # 1. Pre-sample the bank S[z] (deltas; q models s+ - s, unit Jacobian) and
    #    compute LQ[z', z, m] in one batched pass. Reused for all iterations.
    samples = model._q_sample(model.q_state['params'], jnp.asarray(cond), key, n_mc)  # [k, n_mc, D]
    flat = jnp.reshape(samples, (k * n_mc, -1))
    s_rep = jnp.tile(jnp.asarray(s), (k * n_mc, 1))
    lq = np.asarray(model._q_ll_allz(model.q_state['params'], s_rep, flat))  # [k(z'), k*n_mc]
    LQ = lq.reshape(k, k, n_mc).astype(np.float64)  # [z', z, m]

    # 2. Start from uniform w (BA converges from any interior point).
    logw = np.full(k, -np.log(k))
    diag = LQ[np.arange(k), np.arange(k)]  # [k, n_mc]

    C_prev, C, it = -np.inf, 0.0, 0
    for it in range(1, max_iters + 1):
        mix = _logsumexp(logw[:, None, None] + LQ, axis=0)  # [k, n_mc]
        D = (diag - mix).mean(axis=1)  # [k]
        C = float((np.exp(logw) * D).sum())
        if abs(C - C_prev) < tol:
            break
        C_prev = C
        logw = logw + D
        logw = logw - _logsumexp(logw[None], axis=1)
    return C, np.exp(logw), it


def capacity_ba_with_spread(model, s, key, n_restarts=3, **kwargs):
    """Rerun BA with fresh sample banks; returns (mean C, spread (max-min), Cs, w of first run)."""
    keys = jax.random.split(key, n_restarts)
    Cs, w0 = [], None
    for i in range(n_restarts):
        C, w, _ = capacity_ba(model, s, keys[i], **kwargs)
        Cs.append(C)
        if w0 is None:
            w0 = w
    Cs = np.array(Cs)
    return float(Cs.mean()), float(Cs.max() - Cs.min()), Cs, w0
