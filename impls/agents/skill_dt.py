"""Skill Decision Transformer (Skill-DT).

Re-implementation of "Skill Decision Transformer", Shyam Sudhakaran & Sebastian
Risi (arXiv:2301.13573), as an OGBench offline-RL agent. The authors' reference
repository (github.com/shyamsn97/skill-dt) still reads "code to be published
soon", so the manuscript is the sole authority; every section/table reference
below points at it.

================================================================================
METHOD (paper Sec. 4)
================================================================================
  * VQ-VAE skill encoder (Sec. 4.2). An MLP encodes each state to a continuous
    embedding zhat_t, quantized to the nearest codebook entry
        z = argmin_n ||zhat - z_n||^2 .
    Skills are therefore DISCRETE: one index per codebook entry.

  * Future-skill distribution / "skill histogram" (Sec. 4.1). The conditioning
    statistic is the normalized count of skills used from t to the END OF THE
    TRAJECTORY,
        Z_t = normalize( sum_{t'=t}^{T} one_hot(z_{t'}) ) ,
    exactly the `generate_histogram` reverse-cumulative-sum in Sec. A.5.

  * Hindsight skill re-labelling (Sec. 4.1.1, Alg. 1). Because the encoder keeps
    moving, `Z` is recomputed over the WHOLE dataset every training ITERATION --
    Alg. 1's outer loop, whose body is {re-label, sample, J gradient updates}.
    So the re-label period in gradient steps is J, and Table 5's only candidate
    for J is "updates between rollouts" = 50, which is `relabel_interval`'s
    default. Z_t is therefore up to J-1 steps stale, exactly as in Alg. 1. This
    is also what makes the trajectory-end histogram computable from
    fixed-length minibatches, and the paper calls it "required to ensure
    stability in action predictions".

  * Causal Transformer policy (Sec. 4.2). Token order per step is
        [ Z_t , z_t , s_t ]
    and ahat_t is read from the s_t position, giving
        pi(a_t | Z_{t-K}, z_{t-K}, s_{t-K}, ..., Z_t, z_t, s_t) .
    (Sec. 4.1's displayed equation stops the conditioning set at s_{t-1}, but
    Alg. 1 reads ahat_{t:t+K} out of f_theta(..., Z_{t+K}, z_{t+K}, s_{t+K});
    Alg. 1 is what is implemented.) Actions are NOT fed back in as tokens
    (Sec. 4.1). States and histograms get a learned linear token embedding, a
    learned TIMESTEP embedding, and DT's input LayerNorm; the skill embedding
    z_t gets none of the three -- no projection, no timestep embedding, and no
    LayerNorm, since LayerNorm is affine-invariant and would erase the codebook
    vector's magnitude ("we don't tokenize our skill embeddings ... we don't
    lose important skill embedding information", Sec. 4.2). Hence
    code_dim == embed_dim.

  * Objective (Sec. 4.3, Alg. 1):
        L_{theta,phi} = (1/K) sum_t (a_t - ahat_t)^2 + VQLOSS_phi(z, zhat),
        VQLOSS(z, zhat) = MSE(z, zhat)   (Eq. 1),
    with the codebook itself maintained by an exponential moving average
    ("we optimize this loss using an exponential moving average", Sec. 4.2), so
    the gradient part of VQLOSS is the commitment term. The two terms are added
    with no coefficient, so both are reduced the same way -- torch's
    `F.mse_loss`, a mean over valid steps and over the feature axis; reducing
    one over its feature axis and summing the other would silently reweight
    VQLOSS by 1/code_dim.

  * Evaluation (Sec. 5.2 and the `evaluate_skill_dt` pseudocode in Sec. A.5).
    Skill-DT is reward-free, so a rollout is run for EACH discrete skill and the
    best is reported. During a rollout the agent keeps a length-K context
    buffer, RE-ENCODES the state it actually observes at every step (overwriting
    that step's entry in the skill id buffer), and rebuilds the histogram over
    the remaining horizon, whose unvisited tail still holds the target skill.
    See `sample_actions_with_state`.

  * Target-trajectory reconstruction / SMM (Sec. 6.2, Alg. 2). The same rollout
    loop, but the unvisited tail of the histogram holds a TARGET TRAJECTORY's
    encoded skills instead of one repeated skill. See
    `init_eval_state_from_trajectory`.

Training is purely OFFLINE and REWARD-FREE: no rewards, returns, or goals enter
the objective.

================================================================================
HYPERPARAMETERS -- paper Table 5 (Sec. A.2), reproduced exactly
================================================================================
  layers 4 | attention heads 4 | embedding dim 256 | context length K 20 |
  dropout 0.0 (so no dropout modules exist here) | batch size 256 |
  updates between rollouts 50 (-> `relabel_interval`) | lr 1e-4 |
  gradient norm 0.25.
  Codebook size is per-environment in the paper (Table 1): 10 for
  walker2d-medium, halfcheetah-medium and ant-medium; 32 for hopper-medium,
  both -medium-replay tasks and both antmaze-umaze tasks; 64 for both
  antmaze-medium tasks. The default here is 32, the antmaze-umaze value.

================================================================================
CHOICES THE PAPER DOES NOT PIN DOWN, AND OGBENCH-FORCED DEVIATIONS
================================================================================
(a) Not stated by the paper; taken from the original Decision Transformer,
    whose architecture Sec. 4.2 says Skill-DT "shares" (Table 5 is otherwise
    identical to min-decision-transformer's defaults):
      - AdamW with weight decay 1e-4 and a linear LR warmup over 10k steps.
        Weight decay is masked off the EMA codebook accumulators, which carry no
        gradient and are overwritten by the EMA update.
      - tanh on the continuous action head (`use_action_tanh`).
      - the input LayerNorm (`embed_ln`), here over the tokenized inputs only.
(b) Not stated by the paper; conventional defaults:
      - A straight-through estimator on the quantized skill token. The paper
        never names it, but Sec. 4.3 requires the action MSE to reach the
        encoder, and argmin quantization has no gradient without one.
      - The EMA rule is van den Oord et al.'s (Appendix A.1; decay 0.99,
        Laplace eps 1e-5). Neither decay nor eps is stated. Note Sec. 4.2's
        pointer -- "we optimize this loss using an exponential moving average,
        as detailed in Lai et al. 2022" -- is a broken citation: Lai et al.
        (arXiv:2202.01987, Robust VQ-VAE, since WITHDRAWN for code bugs) never
        mentions an EMA, and updates its codebook by the ordinary gradient loss
        ||sg[zhat]-z||^2 + beta*||zhat-sg[z]||^2 that EMA exists to replace; its
        actual subject is dual inlier/outlier codebooks for image outlier
        robustness. So van den Oord's is the only EMA the sentence can denote.
      - `vq_hidden_dims` (256, 256) and `layer_norm` for the encoder MLP.
      - `vq_beta` = 1.0, so the objective is literally Eq. 1's unweighted
        MSE(z, zhat); the van den Oord commitment default would be 0.25.
(b2) Deliberate departures from the LITERAL text, with reasons:
      - Alg. 1 samples one minibatch OUTSIDE its `j = 1..J` inner loop, i.e. J
        gradient steps on the same batch. A fresh minibatch is drawn per step
        here; 50 consecutive updates on one batch would overfit it, and Alg. 1
        also places "sample timesteps uniformly" outside the inner loop, so the
        pseudocode reads as loose rather than literal.
      - Sec. 4.3 samples the window start "uniformly for each trajectory in the
        batch"; `SequenceDataset` samples uniformly over states, which
        over-weights long trajectories. The two coincide on OGBench's
        equal-length datasets, and changing the shared sampler would alter
        every other agent that uses it.
(c) Forced by OGBench:
      - DISCRETE-ACTION CROSS-ENTROPY. The paper's action term is MSE for
        continuous control. OGBench's discrete-action envs use cross-entropy on
        a categorical head (MSE on action indices is meaningless). Continuous
        envs use the paper's MSE exactly.
      - Continuous actions are wrapped in a constant-std MultivariateNormalDiag
        for the `sample_actions` interface; the loss is pure MSE on the mean, so
        the wrapper does not change the objective.
      - The action loss divides by the number of VALID (masked) steps rather
        than a literal 1/K, so steps padded past a trajectory boundary do not
        dilute the mean. Identical to 1/K on full windows.
      - `sample_actions` (OGBench's stateless per-step actor) runs a degenerate
        length-1 context. The paper's protocol is the stateful K-window rollout
        in `sample_actions_with_state`, which `utils.evaluation.evaluate` picks
        up automatically; `sample_actions` exists only for callers that cannot
        thread per-episode state.
      - Sec. A.5's `t < context_len` branch slices `state_buffer[t : t+K]` --
        a forward slice of a zero buffer -- and reads `actions[t]`. That is a
        typo; the intended (and DT-standard) behaviour is a front-padded window
        whose last valid position is the current step, which is what is
        implemented here, with padded positions masked out of attention.
      - Z_t normalizes over the remaining TRAJECTORY at training and over the
        remaining EPISODE HORIZON at eval (Sec. A.5's `max_steps`). In D4RL
        those coincide; in OGBench they do not for datasets whose trajectories
        are shorter than the env horizon (the -stitch slices are 51-201 steps
        against a 1000-step horizon), which puts the eval statistic off the
        training distribution. `eval_max_steps` defaults to None (follow the
        env); set it to the dataset's trajectory length to restore the paper's
        correspondence on those datasets -- an explicit value always wins over
        the env horizon.
      - `init_eval_state()` with no skill draws one uniformly per episode, so
        main.py's in-training `evaluation/success` is E_z[success] over random
        skills and is NOT comparable to the paper's Table 1. Skill-DT has no
        goal-conditioned inference at all; the paper's number is the
        max-over-skills sweep, driven here by `skill_set` +
        `sample_actions_with_skill_state` (eval_skill_policy.py).

The length-K context is supplied by `SequenceDataset` (`utils/datasets.py`,
registered in `main.py`), which also carries the re-labelled `skill_hist_seq`
and the absolute `timesteps_seq`. Select it via `dataset_class='SequenceDataset'`.
"""

from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP

# Parameters holding the VQ-VAE's EMA accumulators. They receive no gradient and
# are overwritten by the EMA update, so AdamW's decoupled weight decay must not
# touch them.
EMA_PARAM_NAMES = ('cluster_size', 'embed_avg')


# ── Skill helpers ─────────────────────────────────────────────────────────────


def ema_codebook(cluster_size, embed_avg, eps=1e-5):
    """Codebook read out of its EMA accumulators, e_i = m_i / N_i.

    `N_i` is Laplace-smoothed exactly as in van den Oord et al.'s EMA VQ-VAE, so
    a code that no data currently maps to does not blow up.
    """
    num_skills = cluster_size.shape[0]
    n = cluster_size.sum()
    cluster = (cluster_size + eps) / (n + num_skills * eps) * n
    return embed_avg / cluster[:, None]


def future_skill_histogram(onehot, mask):
    """Window-bounded fallback for Z_t; used only when re-labelling is disabled.

    Z_t = normalize( sum_{t'>=t, valid} one_hot(z_{t'}) ) over the CONTEXT WINDOW
    rather than to the trajectory end. The paper's statistic runs to the end of
    the trajectory (Sec. 4.1) and is supplied by `SequenceDataset` as
    `skill_hist_seq` after hindsight re-labelling; this truncated version exists
    only for `relabel_interval=0` runs.

    Args:
        onehot: [B, T, N] one-hot skill indices.
        mask:   [B, T] 1.0 for valid steps, 0.0 for padding past trajectory end.
    Returns:
        [B, T, N] histogram; each row sums to 1 over valid futures.
    """
    onehot = onehot * mask[..., None]
    # Reverse cumulative sum along time so position t aggregates t' >= t.
    rev = jnp.flip(onehot, axis=1)
    fwd = jnp.flip(jnp.cumsum(rev, axis=1), axis=1)  # [B, T, N]
    counts = fwd.sum(axis=-1, keepdims=True)
    return fwd / jnp.maximum(counts, 1.0)


# ── Causal Transformer (added for this repo; the codebase is otherwise MLP-only) ─


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (GPT-style), JIT-friendly.

    `key_mask` ([B, L], 1.0 = real token) additionally blocks padded tokens.
    Self-attention is always permitted so that a fully-padded query row cannot
    produce a NaN softmax that would then contaminate valid positions.
    """

    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x, key_mask=None):
        B, L, D = x.shape
        H = self.num_heads
        hd = D // H

        qkv = nn.Dense(3 * D)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(t):
            return t.reshape(B, L, H, hd).transpose(0, 2, 1, 3)  # [B, H, L, hd]

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        att = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(hd)
        allowed = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None]  # [1, 1, L, L]
        if key_mask is not None:
            allowed = allowed & (key_mask > 0)[:, None, None, :]
        # Guarantee at least one live key per row (the diagonal).
        allowed = allowed | jnp.eye(L, dtype=bool)[None, None]
        att = jnp.where(allowed, att, -jnp.inf)
        att = jax.nn.softmax(att, axis=-1)

        out = jnp.einsum('bhqk,bhkd->bhqd', att, v)  # [B, H, L, hd]
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return nn.Dense(D)(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block."""

    embed_dim: int
    num_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, key_mask=None):
        x = x + CausalSelfAttention(self.embed_dim, self.num_heads)(nn.LayerNorm()(x), key_mask)
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.mlp_ratio * self.embed_dim)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.embed_dim)(h)
        return x + h


# ── Network modules ───────────────────────────────────────────────────────────


class SkillVQ(nn.Module):
    """VQ-VAE skill encoder + discrete EMA codebook (paper Sec. 4.1-4.2).

    Encodes each state to a continuous embedding zhat_t and quantizes it to the
    nearest codebook entry. Returns the continuous encoding, the quantized
    embedding, and the discrete skill indices.
    """

    num_skills: int
    code_dim: int
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    gc_encoder: Optional[nn.Module] = None
    eps: float = 1e-5

    def setup(self):
        self.encoder_mlp = MLP(
            (*self.hidden_dims, self.code_dim),
            activate_final=False, layer_norm=self.layer_norm,
        )
        # EMA accumulators: cluster size N_i and embedding sum m_i. The codebook
        # is derived on read as e_i = m_i / N_i (Laplace-smoothed).
        self.cluster_size = self.param('cluster_size', nn.initializers.ones, (self.num_skills,))
        self.embed_avg = self.param(
            'embed_avg', nn.initializers.normal(1.0), (self.num_skills, self.code_dim)
        )

    def codebook(self):
        return ema_codebook(self.cluster_size, self.embed_avg, self.eps)  # [N, code_dim]

    def encode(self, observations):
        x = observations
        if self.gc_encoder is not None:
            x = jax.vmap(lambda s: self.gc_encoder(s, None), in_axes=1, out_axes=1)(x)
        return self.encoder_mlp(x)  # [B, T, code]

    def __call__(self, observations):
        z_e = self.encode(observations)
        cb = jax.lax.stop_gradient(self.codebook())  # frozen w.r.t. gradients (EMA only)

        # z = argmin_n ||zhat - z_n||^2  (paper Sec. 4.2), expanded as
        # ||zhat||^2 - 2 zhat.z_n + ||z_n||^2 so the distances cost one [..., N]
        # matmul. The literal `(z_e[..., None, :] - cb) ** 2` form materializes a
        # [B, T, N, code] intermediate -- 2 GB for a 65536-state re-labelling
        # chunk -- which XLA spends minutes trying to fuse.
        dist = (
            (z_e ** 2).sum(-1, keepdims=True)
            - 2.0 * jnp.einsum('...d,nd->...n', z_e, cb)
            + (cb ** 2).sum(-1)
        )                                               # [B, T, N]
        indices = jnp.argmin(dist, axis=-1)             # [B, T]
        z_q = cb[indices]                               # [B, T, code]
        return z_e, z_q, indices


class SkillDTPolicy(nn.Module):
    """Skill-conditioned causal Transformer policy (paper Sec. 4.2).

    Per timestep the input token triple is ordered [Z_t, z_t, s_t]
    (future-skill histogram, quantized skill embedding, state). The action
    ahat_t is read out from the hidden state at the s_t position.
    """

    action_dim: int
    num_skills: int
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_timestep: int = 4096
    discrete: bool = False
    const_std: bool = True
    use_action_tanh: bool = True
    gc_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations, skill_embeddings, skill_hist, timesteps, mask=None, temperature=1.0):
        B, T = observations.shape[0], observations.shape[1]

        s_in = observations
        if self.gc_encoder is not None:
            s_in = jax.vmap(lambda s: self.gc_encoder(s, None), in_axes=1, out_axes=1)(s_in)

        # Per-token embeddings.
        s_tok = nn.Dense(self.embed_dim)(s_in)
        z_tok = skill_embeddings
        h_tok = nn.Dense(self.embed_dim)(skill_hist)

        # Learned ABSOLUTE-timestep embedding, as in the original DT. The paper
        # adds it to the state and skill-distribution (histogram) tokens, but NOT
        # to the raw skill embedding token z_t: that one is fed straight through
        # with no projection ("we don't tokenize our skill embeddings", Sec. 4.2)
        # and still picks up temporal information through attention.
        t_emb = nn.Embed(self.max_timestep, self.embed_dim)(
            jnp.clip(timesteps, 0, self.max_timestep - 1)
        )  # [B, T, D]

        # DT's single input LayerNorm over the *tokenized* inputs. It is not
        # applied to z_t, which Sec. 4.2 deliberately leaves un-tokenized "so
        # that we don't lose important skill embedding information": LayerNorm
        # is affine-invariant, so it would discard the codebook vector's mean and
        # norm outright. (Each block's own pre-LN still normalizes z_t, so the
        # magnitude is largely lost anyway; skipping embed_ln is the reading that
        # matches the paper's stated intent, not a guarantee of scale.)
        embed_ln = nn.LayerNorm()
        s_tok = embed_ln(s_tok + t_emb)
        h_tok = embed_ln(h_tok + t_emb)

        # Interleave into [Z_0, z_0, s_0, Z_1, z_1, s_1, ...] -> [B, 3T, D].
        x = jnp.stack([h_tok, z_tok, s_tok], axis=2).reshape(B, T * 3, self.embed_dim)
        # A padded step hides all three of its tokens.
        token_mask = None
        if mask is not None:
            token_mask = jnp.repeat(mask, 3, axis=1)  # [B, 3T]

        for _ in range(self.num_layers):
            x = TransformerBlock(self.embed_dim, self.num_heads)(x, token_mask)
        x = nn.LayerNorm()(x)

        # Action prediction from the s_t positions (index 3t + 2).
        s_hidden = x[:, 2::3, :]  # [B, T, D]

        if self.discrete:
            logits = nn.Dense(self.action_dim)(s_hidden)
            return distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))
        else:
            means = nn.Dense(self.action_dim)(s_hidden)
            if self.use_action_tanh:
                means = jnp.tanh(means)
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))
            log_stds = jnp.clip(log_stds, -5.0, 2.0)
            return distrax.MultivariateNormalDiag(
                loc=means, scale_diag=jnp.exp(log_stds) * temperature
            )


# ── Agent ─────────────────────────────────────────────────────────────────────


class SkillDTAgent(flax.struct.PyTreeNode):
    """Skill Decision Transformer agent (offline, reward-free, discrete skills)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── Batch normalization to sequences ──────────────────────────────────────

    @staticmethod
    def _as_seq_obs(observations):
        """Ensure observations are [B, T, obs_dim]. Adds T=1 for flat batches."""
        if observations.ndim == 2:
            return observations[:, None, :]
        return observations

    def _as_seq_actions(self, actions):
        """Ensure actions are [B, T, ...] (discrete: [B, T]; continuous: [B, T, A])."""
        if self.config['discrete']:
            return actions[:, None] if actions.ndim == 1 else actions
        return actions[:, None, :] if actions.ndim == 2 else actions

    # ── Losses ────────────────────────────────────────────────────────────────

    def vq_loss(self, z_e, z_q, mask):
        """VQLOSS(z, zhat) = MSE(z, zhat) (Eq. 1).

        The codebook side of that MSE is handled by the EMA update rather than by
        gradients (Sec. 4.2), so what is left to differentiate is the commitment
        term beta * ||zhat - sg[z]||^2 with beta = `vq_beta`.
        """
        sg = jax.lax.stop_gradient
        m = mask[..., None]
        denom = jnp.maximum(m.sum() * z_e.shape[-1], 1.0)
        commit_loss = (((z_e - sg(z_q)) ** 2) * m).sum() / denom
        loss = self.config['vq_beta'] * commit_loss
        return loss, {
            'vq_loss': loss,
            'vq_commit_loss': commit_loss,
        }

    def action_loss(self, dist, actions, mask):
        """(1/K) sum_t (a_t - ahat_t)^2 for continuous; CE for OGBench's discrete envs.

        The mean runs over valid steps AND action dimensions, i.e. torch's
        `F.mse_loss` -- the same reduction `vq_loss` uses for MSE(z, zhat). The
        two terms of Eq. 1 are added with no coefficient, so they must be scaled
        alike; reducing one over the feature axis and summing the other would
        silently reweight VQLOSS by 1/code_dim.
        """
        if self.config['discrete']:
            nll = -dist.log_prob(actions)  # [B, T]
            loss = (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            metrics = {'action_loss': loss, 'action_nll': loss}
        else:
            pred = dist.mode()  # deterministic mean
            se = (pred - actions) ** 2  # [B, T, A]
            denom = jnp.maximum(mask.sum() * se.shape[-1], 1.0)
            loss = (se * mask[..., None]).sum() / denom
            metrics = {'action_loss': loss, 'action_mse': loss}
        return loss, metrics

    def _seq_batch(self, batch):
        """Pull the real length-K window from the SequenceDataset batch.

        Falls back to a length-1 window if the seq keys are absent (only reached
        by the shape-probing example batch).
        """
        if 'observations_seq' in batch:
            observations = batch['observations_seq']
            actions = batch['actions_seq']
        else:
            observations = self._as_seq_obs(batch['observations'])
            actions = self._as_seq_actions(batch['actions'])
        B, T = observations.shape[0], observations.shape[1]
        mask = batch.get('seq_mask', None)
        # The validity mask must be float32 (an integer mask overflows the
        # loss-denominator reductions for uint8 image observations).
        if mask is None:
            mask = jnp.ones((B, T), dtype=jnp.float32)
        else:
            mask = mask.astype(jnp.float32)
            if mask.ndim == 1:
                # A per-sample mask still has to cover every step of the window;
                # [B, 1] would broadcast into the loss but not weight step 1..T-1.
                mask = jnp.broadcast_to(mask[:, None], (B, T))
        timesteps = batch.get('timesteps_seq', None)
        if timesteps is None:
            timesteps = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))
        else:
            timesteps = timesteps.astype(jnp.int32)
        return observations, actions, mask, timesteps

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        observations, actions, mask, timesteps = self._seq_batch(batch)

        # 1) VQ-VAE skill encoding + quantization.
        z_e, z_q, indices = self.network.select('vq')(observations, params=grad_params)

        # Straight-through estimator: gradients of the action loss flow to the
        # encoder, while the quantized value is used in the forward pass.
        z_q_st = z_e + jax.lax.stop_gradient(z_q - z_e)

        # 2) Future-skill histogram Z_t (the conditioning statistic, Sec. 4.1).
        #    Hindsight re-labelling (Sec. 4.1.1) precomputes it to the trajectory
        #    end over the whole dataset, so it arrives in the batch.
        onehot = jax.nn.one_hot(indices, self.config['num_skills'])  # [B, T, N]
        if 'skill_hist_seq' in batch:
            skill_hist = jax.lax.stop_gradient(batch['skill_hist_seq'].astype(jnp.float32))
        elif self.config['relabel_interval'] > 0:
            raise KeyError(
                'skill_dt: `skill_hist_seq` is missing from the batch, but relabel_interval='
                f"{self.config['relabel_interval']} > 0 requires the hindsight-re-labelled "
                'trajectory-end histogram (paper Sec. 4.1.1). Use '
                "dataset_class='SequenceDataset' and run through main.py, which calls "
                '`SequenceDataset.relabel_skill_histograms` before each iteration.'
            )
        else:
            skill_hist = jax.lax.stop_gradient(future_skill_histogram(onehot, mask))

        # 3) Skill-conditioned causal Transformer policy.
        dist = self.network.select('policy')(
            observations, z_q_st, skill_hist, timesteps, mask, params=grad_params
        )

        # Losses.
        vq_loss, vq_info = self.vq_loss(z_e, z_q, mask)
        info.update({f'vq/{k}': v for k, v in vq_info.items()})

        act_loss, act_info = self.action_loss(dist, actions, mask)
        info.update({f'policy/{k}': v for k, v in act_info.items()})

        # Codebook usage diagnostics.
        usage = (onehot * mask[..., None]).sum((0, 1))  # [N]
        usage = usage / jnp.maximum(usage.sum(), 1.0)
        info['vq/codebook_perplexity'] = jnp.exp(-(usage * jnp.log(usage + 1e-10)).sum())

        total = act_loss + vq_loss
        info['total_loss'] = total
        return total, info

    def _ema_codebook(self, network, batch):
        """EMA codebook update (Sec. 4.2: VQLOSS "optimized using an EMA").

        Accumulates over the encodings the loss was computed from -- i.e. the
        PRE-gradient encoder (`self.network`) -- and writes the result into the
        post-gradient params, as in van den Oord et al.'s EMA VQ-VAE.
        """
        beta = self.config['vq_decay']
        N = self.config['num_skills']
        code_dim = self.config['code_dim']

        observations, _, mask, _ = self._seq_batch(batch)

        z_e, _, indices = self.network.select('vq')(observations)
        z_e = jax.lax.stop_gradient(z_e)

        onehot = jax.nn.one_hot(indices, N) * mask[..., None]  # [B, T, N], padded->0
        onehot_flat = onehot.reshape(-1, N)                    # [B*T, N]
        z_flat = z_e.reshape(-1, code_dim)                     # [B*T, code]

        n_i = onehot_flat.sum(0)        # [N] cluster counts in this batch
        dw = onehot_flat.T @ z_flat     # [N, code] summed encodings per code

        vq_params = network.params['modules_vq']
        new_cluster = beta * vq_params['cluster_size'] + (1.0 - beta) * n_i
        new_embed_avg = beta * vq_params['embed_avg'] + (1.0 - beta) * dw

        new_vq = {**vq_params, 'cluster_size': new_cluster, 'embed_avg': new_embed_avg}
        new_params = {**network.params, 'modules_vq': new_vq}
        return network.replace(params=new_params)

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )
        # EMA codebook update after the gradient step.
        new_network = self._ema_codebook(new_network, batch)
        return self.replace(network=new_network, rng=new_rng), info

    # ── Hindsight skill re-labelling hook (paper Sec. 4.1.1, Alg. 1) ───────────

    @jax.jit
    def encode_skill_indices(self, observations):
        """Discrete skill index of each state. [B, ...] -> [B] int32.

        Called by `SequenceDataset.relabel_skill_histograms` every
        `relabel_interval` updates to rebuild the trajectory-end histograms.
        """
        _, _, indices = self.network.select('vq')(observations[:, None, ...])
        return indices[:, 0]

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _codebook(self):
        """Codebook read straight out of the stored EMA accumulators."""
        vq_params = self.network.params['modules_vq']
        return ema_codebook(vq_params['cluster_size'], vq_params['embed_avg'])

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Stateless single-step actor: a DEGRADED length-1 context.

        The paper's inference procedure is the stateful K-window rollout in
        `sample_actions_with_state`, which `utils.evaluation.evaluate` prefers
        automatically. This entry point exists only for callers that cannot
        thread per-episode state; it feeds a single [Z_t, z_t, s_t] triple with
        Z_t collapsed to the one-hot of a uniformly drawn skill, and `goals` is
        ignored (Skill-DT has no goal-conditioned inference).
        """
        if seed is None:
            seed = self.rng

        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        if single_obs:
            observations = observations[None, ...]

        batch_size = observations.shape[0]
        N = self.config['num_skills']
        skill_seed, action_seed = jax.random.split(seed)
        skills = jax.random.randint(skill_seed, (batch_size,), 0, N)

        z_q = self._codebook()[skills][:, None, :]                 # [B, 1, code]
        skill_hist = jax.nn.one_hot(skills, N)[:, None, :]         # [B, 1, N]
        timesteps = jnp.zeros((batch_size, 1), jnp.int32)

        obs_seq = observations[:, None, ...]                       # [B, 1, obs]
        dist = self.network.select('policy')(
            obs_seq, z_q, skill_hist, timesteps, None, temperature=temperature
        )
        actions = dist.sample(seed=action_seed)[:, 0]              # [B, ...]
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)

        if single_obs:
            actions = actions[0]
        return actions

    def _horizon(self, max_steps):
        """Episode horizon the rollout histogram is defined over (Sec. A.5's `max_steps`).

        An explicit `eval_max_steps` in the config always wins; otherwise the
        env's own `max_episode_steps`, handed over by `utils.evaluation`, is used.
        1000 is the last-resort fallback for an env with no registered horizon.
        """
        configured = self.config['eval_max_steps']
        if configured is not None:
            return int(configured)
        return int(max_steps) if max_steps is not None else 1000

    @staticmethod
    def _constant_skill_tail(skill, num_skills, horizon):
        """[L+1, N] reverse-cumulative counts of one skill repeated over the horizon.

        Row j is the count over steps [j, L-1], i.e. `(L - j) * one_hot(skill)`.
        This is A.5's `one_hot_skill_ids`, which starts out filled with the
        target skill at every step of the episode.
        """
        remaining = (horizon - jnp.arange(horizon + 1)).astype(jnp.float32)  # [L+1]
        return remaining[:, None] * jax.nn.one_hot(skill, num_skills)[None, :]

    def init_eval_state(self, skill=None, max_steps=None):
        """Per-episode rollout state for the paper's Sec. A.5 procedure.

        Holds the length-K context buffers plus `tail_suffix`, the [L+1, N]
        reverse-cumulative skill counts of whatever the rollout is conditioned
        towards; row j covers steps [j, L-1]. For A.5's skill sweep that is one
        skill repeated over the whole horizon; `init_eval_state_from_trajectory`
        fills it from a target trajectory instead (Alg. 2).

        `skill=None` stores the sentinel -1, which makes the first
        `sample_actions_with_state` call draw a skill uniformly, build its tail
        and freeze both for the episode -- Skill-DT is unsupervised, so there is
        nothing to pick a skill from. The paper's real protocol sweeps every
        skill; use `init_eval_state_with_skill`.
        """
        K = int(self.config['context_len'])
        N = int(self.config['num_skills'])
        obs_shape = tuple(self.config['obs_shape'])
        horizon = self._horizon(max_steps)
        if skill is None:
            tail_suffix = jnp.zeros((horizon + 1, N), jnp.float32)  # filled in at step 0
        else:
            tail_suffix = self._constant_skill_tail(int(skill), N, horizon)
        return dict(
            skill=jnp.asarray(-1 if skill is None else int(skill), jnp.int32),
            count=jnp.zeros((), jnp.int32),
            tail_suffix=tail_suffix,
            obs_buf=jnp.zeros((K,) + obs_shape, jnp.float32),
            idx_buf=jnp.zeros((K,), jnp.int32),
            valid=jnp.zeros((K,), jnp.float32),
        )

    def init_eval_state_with_skill(self, skill, max_steps=None):
        """`init_eval_state` pinned to one skill of `skill_set()` (a one-hot row).

        Host-side only: it reads `skill` as a concrete index, so it must be
        called outside `jax.jit` (as `utils.evaluation` does, once per episode).
        """
        skill = jnp.asarray(skill)
        index = int(jnp.argmax(skill)) if skill.ndim > 0 else int(skill)
        return self.init_eval_state(skill=index, max_steps=max_steps)

    def init_eval_state_from_trajectory(self, target_observations, max_steps=None):
        """Rollout state for Alg. 2: reconstruct a TARGET TRAJECTORY (Sec. 6.2 SMM).

        Alg. 2 encodes the target trajectory's states, turns the resulting skill
        indices into histograms, and rolls out under them:

            2. (z_0,...,z_T), (zindex_0,...,zindex_T) = E_phi(s_0^target,...)
            3. (Z_0,...,Z_T) = histogram(zindex_0,...,zindex_T)
            4. tauhat ~ pi(a | Z_0, z_0, s_0, ...)

        Step 4 runs A.5's loop, so the only difference from the skill sweep is
        what the unvisited tail of the histogram holds: the target trajectory's
        own future skills rather than one repeated skill.

        A target shorter than the horizon contributes nothing past its end (the
        agent has run out of trajectory to match); a longer one is truncated.

        Args:
            target_observations: [T, *obs_shape] states of the target trajectory.
            max_steps: episode horizon; see `_horizon`.
        """
        N = int(self.config['num_skills'])
        horizon = self._horizon(max_steps)
        indices = self.encode_skill_indices(jnp.asarray(target_observations, jnp.float32))
        onehot = jax.nn.one_hot(indices, N)[:horizon]                       # [T', N]
        pad = horizon - onehot.shape[0]
        if pad > 0:
            onehot = jnp.concatenate([onehot, jnp.zeros((pad, N), jnp.float32)], axis=0)
        rev = jnp.flip(jnp.cumsum(jnp.flip(onehot, axis=0), axis=0), axis=0)  # [L, N]
        tail_suffix = jnp.concatenate([rev, jnp.zeros((1, N), jnp.float32)], axis=0)
        state = self.init_eval_state(skill=int(indices[0]), max_steps=max_steps)
        return {**state, 'tail_suffix': tail_suffix}

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None, temperature=1.0):
        """Paper Sec. A.5 rollout: K-window context with re-encoded skills.

        At step t the observed state is encoded to its own skill index, which
        overwrites entry t of the skill-id buffer; the remaining horizon still
        holds the target skill, so the histogram

            Z_j = normalize( observed skills over [j, t]
                             + tail_suffix[t + 1] )

        keeps steering the policy towards the target while reflecting where the
        rollout actually is ("even though the policy is completely conditioned to
        follow a single skill, it may end up reaching states that are classified
        under another"). `tail_suffix[t+1]` is the count over the unvisited steps
        t+1..L-1, which is `(L-1-t) * one_hot(target)` for A.5's skill sweep and
        the target trajectory's own future skills for Alg. 2. `goals` is ignored:
        Skill-DT is goal-agnostic.

        Returns `(action, new_state)`.
        """
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()

        obs_ndim = len(tuple(self.config['obs_shape']))
        if observations.ndim != obs_ndim:
            raise ValueError(
                f'skill_dt: the stateful rollout keeps one episode\'s context buffers, so it needs a '
                f'single observation of shape {tuple(self.config["obs_shape"])}, got shape '
                f'{observations.shape}.'
            )

        N = self.config['num_skills']
        K = int(self.config['context_len'])
        L = agent_state['tail_suffix'].shape[0] - 1  # episode horizon (static)
        skill_seed, action_seed = jax.random.split(seed)

        # Resolve (and then freeze) the target skill and the tail it implies.
        unset = agent_state['skill'] < 0
        skill = jnp.where(unset, jax.random.randint(skill_seed, (), 0, N), agent_state['skill'])
        tail_suffix = jnp.where(
            unset, self._constant_skill_tail(skill, N, L), agent_state['tail_suffix']
        )
        t = agent_state['count']

        # Slide the context buffers and append the current step.
        obs_buf = jnp.concatenate([agent_state['obs_buf'][1:], observations[None].astype(jnp.float32)], axis=0)
        valid = jnp.concatenate([agent_state['valid'][1:], jnp.ones((1,), jnp.float32)], axis=0)

        # Re-encode the state we actually landed in (Sec. A.5).
        _, _, cur_index = self.network.select('vq')(observations[None, None, ...])
        idx_buf = jnp.concatenate([agent_state['idx_buf'][1:], cur_index[:, 0]], axis=0)  # [K]

        # Future-skill histogram over [j, eval_max_steps - 1] for each window slot.
        onehot = jax.nn.one_hot(idx_buf, N) * valid[:, None]              # [K, N]
        suffix = jnp.flip(jnp.cumsum(jnp.flip(onehot, axis=0), axis=0), axis=0)  # [K, N]
        tail = tail_suffix[jnp.minimum(t + 1, L)]                          # [N]
        counts = suffix + tail[None, :]                                   # [K, N]
        skill_hist = counts / jnp.maximum(counts.sum(-1, keepdims=True), 1e-8)

        z_q = self._codebook()[idx_buf]                                   # [K, code]
        # Absolute timesteps of the window; the newest slot is step t.
        timesteps = jnp.clip(t - (K - 1) + jnp.arange(K, dtype=jnp.int32), 0)

        dist = self.network.select('policy')(
            obs_buf[None], z_q[None], skill_hist[None], timesteps[None], valid[None],
            temperature=temperature,
        )
        actions = dist.sample(seed=action_seed)[0, -1]  # newest (current) step
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)

        new_state = dict(
            skill=skill, count=t + 1, tail_suffix=tail_suffix,
            obs_buf=obs_buf, idx_buf=idx_buf, valid=valid,
        )
        return actions, new_state

    # ── Skill-conditioned evaluation hooks (see eval_skill_policy.py) ──────────

    def skill_set(self, seed=None, num_skills=None, observations=None):
        """The learned skill set: the `num_skills` one-hot codebook selectors.

        Sweeping this set and keeping the best rollout IS the paper's evaluation
        protocol (Sec. 5.2, A.5). `seed`/`num_skills`/`observations` are accepted
        for interface compatibility and ignored -- the skill set is finite and
        fully determined by the codebook.
        """
        return jnp.eye(int(self.config['num_skills']))

    def sample_actions_with_skill_state(
        self, observations, skills, agent_state=None, seed=None, temperature=1.0
    ):
        """`sample_actions_with_state` pinned to a fixed skill of `skill_set()`.

        `skills` selects the target skill only when `agent_state` is None; once a
        state exists the skill is baked into it (that is what pins it for the
        episode), so passing a state built for one skill together with a
        different `skills` follows the state, not `skills`.
        """
        if agent_state is None:
            # No `max_steps` here, so the horizon falls back to `eval_max_steps`;
            # `utils.evaluation` builds the state with the env's own horizon.
            agent_state = self.init_eval_state_with_skill(skills)
        return self.sample_actions_with_state(
            observations, goals=None, agent_state=agent_state, seed=seed, temperature=temperature
        )

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # Normalize the example batch to a sequence so module shapes initialize
        # correctly whether the dataset yields flat transitions or K-windows.
        # Insert a length-1 time axis so the policy/VQ modules initialize with
        # the correct time dimension (the encoder is applied per timestep).
        if config.get('encoder') is not None:
            ex_obs = ex_observations[:, None, ...]
        else:
            ex_obs = cls._as_seq_obs(ex_observations)
        if config['discrete']:
            action_dim = int(ex_actions.max()) + 1
        else:
            action_dim = ex_actions.shape[-1]

        num_skills = config['num_skills']
        code_dim = config['code_dim']

        # The skill token is fed to the Transformer without a Dense projection,
        # so the skill embedding dimension must equal the Transformer width.
        assert code_dim == config['embed_dim'], (
            f'Skill-DT feeds the skill embedding directly as a token, so '
            f"code_dim ({code_dim}) must equal embed_dim ({config['embed_dim']})."
        )
        assert int(config['sequence_length']) == int(config['context_len']), (
            f"sequence_length ({config['sequence_length']}) is the SequenceDataset window and must "
            f"equal the Transformer context length K ({config['context_len']})."
        )

        # Optional visual encoders (applied per timestep). Paper is state-based.
        vq_encoder = None
        policy_encoder = None
        if config.get('encoder') is not None:
            enc = encoder_modules[config['encoder']]
            vq_encoder = GCEncoder(state_encoder=enc())
            policy_encoder = GCEncoder(state_encoder=enc())

        vq_def = SkillVQ(
            num_skills=num_skills,
            code_dim=code_dim,
            hidden_dims=config['vq_hidden_dims'],
            layer_norm=config['layer_norm'],
            gc_encoder=vq_encoder,
        )
        policy_def = SkillDTPolicy(
            action_dim=action_dim,
            num_skills=num_skills,
            embed_dim=config['embed_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            max_timestep=config['max_timestep'],
            discrete=config['discrete'],
            const_std=config['const_std'],
            use_action_tanh=config['use_action_tanh'],
            gc_encoder=policy_encoder,
        )

        B, T = ex_obs.shape[0], ex_obs.shape[1]
        ex_skill_emb = jnp.zeros((B, T, code_dim))
        ex_hist = jnp.zeros((B, T, num_skills))
        ex_timesteps = jnp.zeros((B, T), jnp.int32)
        ex_mask = jnp.ones((B, T), jnp.float32)

        network_def = ModuleDict(dict(vq=vq_def, policy=policy_def))
        network_params = network_def.init(
            init_rng,
            vq=(ex_obs,),
            policy=(ex_obs, ex_skill_emb, ex_hist, ex_timesteps, ex_mask),
        )['params']

        # AdamW + linear warmup + global-norm clipping. The paper gives lr 1e-4
        # and gradient norm 0.25 (Table 5) but not the optimizer; these are the
        # original DT's, whose architecture Sec. 4.2 says Skill-DT shares.
        warmup = int(config['warmup_steps'])
        if warmup > 0:
            lr = optax.linear_schedule(
                init_value=config['lr'] / warmup, end_value=config['lr'], transition_steps=warmup
            )
        else:
            lr = config['lr']

        def weight_decay_mask(params):
            flat = flax.traverse_util.flatten_dict(params)
            return flax.traverse_util.unflatten_dict(
                {path: path[-1] not in EMA_PARAM_NAMES for path in flat}
            )

        tx = optax.chain(
            optax.clip_by_global_norm(config['grad_clip']),
            optax.adamw(
                learning_rate=lr,
                weight_decay=config['weight_decay'],
                mask=weight_decay_mask,
            ),
        )
        network = TrainState.create(network_def, network_params, tx=tx)

        # Per-step observation shape, needed to allocate the rollout context
        # buffers in `init_eval_state`. Derived here rather than configured.
        config = dict(config)
        config['obs_shape'] = tuple(ex_obs.shape[2:])
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ────────────────────────────────────────────────────────────────────


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='skill_dt',
        # Optimization. lr / batch_size / grad_clip are paper Table 5; weight
        # decay and warmup are the original DT's (paper does not state them).
        lr=1e-4,
        batch_size=256,
        grad_clip=0.25,
        weight_decay=1e-4,
        warmup_steps=10000,
        # Hindsight skill re-labelling (Sec. 4.1.1, Alg. 1). Every this many
        # gradient steps the whole dataset is re-encoded and the trajectory-end
        # histograms Z_t are rebuilt; 50 is Alg. 1's J, i.e. Table 5's "updates
        # between rollouts". Measured on an idle A6000, antmaze-medium-navigate-v0
        # (1.0M states, 32 skills): 119 ms per re-label against a 23 ms training
        # step, i.e. ~10% overhead at 50 (~0.7 h over a 1M-step run); device peak
        # stays under 0.7 GB even at 64 skills. Raise it to trade histogram
        # freshness for speed; 0 disables re-labelling entirely and
        # falls back to a window-bounded histogram (NOT the paper's statistic).
        relabel_interval=50,
        # VQ-VAE skill codebook.
        num_skills=32,            # paper Table 1: 10 / 32 / 64 depending on env.
        code_dim=256,             # skill embedding dim (MUST equal embed_dim).
        vq_hidden_dims=(256, 256),
        vq_beta=1.0,              # VQLOSS = MSE(z, zhat), Eq. 1 (unweighted).
        vq_decay=0.99,            # EMA codebook decay (Sec. 4.2 EMA variant).
        layer_norm=True,
        # Causal Transformer policy (paper Table 5; dropout 0.0 -> no dropout).
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        context_len=20,           # K
        # Size of the learned timestep-embedding table. 4096 is the original DT's
        # `max_ep_len` and clears every OGBench horizon (the longest is
        # humanoidmaze-giant's 4000); a smaller table would silently clip
        # distinct timesteps onto one row. Costs 4096 * embed_dim params (~4 MB).
        max_timestep=4096,
        # Episode horizon the rollout histogram is defined over (Sec. A.5's
        # `max_steps`). None follows the env's registered `max_episode_steps`;
        # set an int to override (see the Z_t normalization caveat in the header).
        eval_max_steps=ml_collections.config_dict.placeholder(int),
        const_std=True,
        use_action_tanh=True,     # original DT's continuous action head.
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='SequenceDataset',
        sequence_length=20,       # K: must equal context_len.
        value_p_curgoal=0.0,
        value_p_trajgoal=1.0,
        value_p_randomgoal=0.0,
        value_geom_sample=False,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=True,
        p_aug=0.0,
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))
