"""QueST: Self-Supervised Skill Abstractions for Learning Continuous Control.

Faithful OGBench re-implementation of QueST (Mete et al., NeurIPS 2024,
arXiv:2407.15840). The method has two stages, both trained purely offline:

  Stage I  (Sec. 4.1):  A chunked action autoencoder. An encoder maps an action
      chunk a_{t:t+T-1} to n = T/F latent vectors, which are discretized with
      Finite Scalar Quantization (FSQ; Mentzer et al. 2023). A transformer
      decoder reconstructs the chunk. Trained with L1 reconstruction only
      (Eq. 4). FSQ needs NO codebook / commitment loss.

  Stage II (Sec. 4.2):  An autoregressive (causal Transformer) prior over the
      discrete skill tokens, conditioned on the (goal-)state. Trained with
      token-level cross-entropy / NLL (Eq. 6).

This file mirrors the public agent interface used across OGBench
(`impls/agents/empowerment_skill.py`, `gcbc.py`): a `flax.struct.PyTreeNode`
with `create`, jitted `update` / `total_loss`, `sample_actions`, and a
module-level `get_config()`.

================================================================================
ASSUMPTIONS & DEVIATIONS FROM THE PAPER (arXiv:2407.15840)
================================================================================
Read this first. Everything below is either an assumption forced by the OGBench
interface, or a point where this code is NOT an exact match to the paper. Each
line gives a one-line reason + paper section. `quest_NOTES.md` expands on these.

A. ASSUMPTIONS FORCED BY THE OGBENCH INTERFACE
  A1. Execution horizon = 1 (replan every step). Paper executes T_a = 8 actions
      open-loop before replanning (Sec. 4.3, Table 4), but OGBench's eval loop
      calls `sample_actions` every env step and is stateless, so we decode the
      chunk and return only the first action a_t. More compute at eval; more
      conservative behavior.
  A2. Conditioning e = goal observation, not a CLIP / learned task embedding.
      Paper uses CLIP (LIBERO) or a learned task embedding (MetaWorld) for e
      (Sec. 4.2); OGBench is goal-conditioned, so e is the goal observation
      (`actor_goals`), optionally encoded by a visual `GCEncoder`. Toggle with
      `goal_conditioned`.
  A3. Conditioning is injected as a single prefix token that plays the role of
      the learnable start token <s> at position 0 (Eq. 5). Paper conditions on a
      short observation history (h = 1) plus e; we fold the current observation
      (+ goal) into one MLP-produced prefix token.
  A4. Two-stage schedule is realized by step-gating inside one `update`
      (`stage1_steps`) rather than two separate training scripts (Sec. 5). The
      two losses are gradient-decoupled (prior targets are stop-grad integer
      tokens), so this is equivalent; the split point is our hyperparameter.
  A5. Few-shot decoder finetuning (Eq. 7, Table 7) is NOT implemented — it is a
      transfer/adaptation step outside the offline pretraining objective.

B. NON-EXACT MATCHES (with reason)
  B1. No attention/embedding dropout. Paper Table 4 lists attention dropout 0.1
      for the prior; OGBench networks do not thread a train/eval flag or dropout
      rng through agents, so we omit it (regularization only; affects training
      dynamics, not architecture).
  B2. Decoder interleaves self- then cross-attention within every block (paper:
      "alternate masked self-attention and cross-attention layers", Sec. 4.1) —
      same operator set, packaged per-block rather than as strictly alternating
      standalone layers.
  B3. Minor underspecified details follow standard GPT/ViT practice: pre-LN,
      MLP ratio 4, GELU (prior/decoder MLPs). Capacities (dims/layers/heads/
      kernels/strides, FSQ levels, T, F, n, vocab) match Tables 3-4 exactly.

Matched to the official repo (github.com/pairlab/QueST) after audit: the encoder
now uses FIXED SINUSOIDAL positional embeddings (official Summer(PositionalEncoding1D))
and GroupNorm(4)+Mish conv blocks (official Conv1dBlock); the decoder adds no
positional embedding to the code memory tokens. Decoder query inputs and prior
token positions are fixed sinusoidal, as in the paper/official code.

See `quest_NOTES.md` for per-component paper citations and full detail.
"""

from typing import Any, Optional, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


# ── Finite Scalar Quantization (paper Sec. 3.2, Eq. 1-2) ───────────────────────
#
# FSQ (Mentzer et al. 2023, arXiv:2309.15505) quantizes each latent channel to a
# fixed small number of levels with a bounded, straight-through rounding op. The
# implicit codebook has size prod(levels) and needs no learned embeddings and no
# auxiliary VQ losses — that is precisely why QueST's Stage I objective is a pure
# reconstruction loss.


def _round_ste(z):
    """Round with a straight-through estimator: forward=round, backward=identity."""
    return z + jax.lax.stop_gradient(jnp.round(z) - z)


def fsq_bound(z, levels, eps=1e-3):
    """Bound z into the open interval covering `levels` quantization cells.

    f(z) = tanh(z + shift) * half_l - offset   (Mentzer et al., Eq. 4 of FSQ).
    """
    levels = levels.astype(z.dtype)
    half_l = (levels - 1.0) * (1.0 - eps) / 2.0
    offset = jnp.where(levels % 2 == 0, 0.5, 0.0)
    shift = jnp.arctanh(offset / half_l)
    return jnp.tanh(z + shift) * half_l - offset


def fsq_quantize(z, levels):
    """Quantize z (straight-through) and renormalize codes to ~[-1, 1].

    This is the differentiable path used for reconstruction. Gradients flow
    through `_round_ste` unchanged (QueST Sec. 3.2 / Eq. 2).
    """
    quantized = _round_ste(fsq_bound(z, levels))
    half_width = (levels // 2).astype(z.dtype)
    return quantized / half_width


def _fsq_basis(levels):
    """Mixed-radix place values for flattening per-channel codes to an index."""
    levels = jnp.asarray(levels)
    ones = jnp.ones((1,), dtype=jnp.int32)
    return jnp.concatenate([ones, jnp.cumprod(levels[:-1]).astype(jnp.int32)])


def fsq_codes_to_indices(codes, levels):
    """Map renormalized codes [..., d] -> integer token ids [...] in [0, prod(levels))."""
    half_width = (levels // 2)
    zhat = jnp.round(codes * half_width + half_width)  # per-channel 0..L_i-1
    basis = _fsq_basis(levels)
    return (zhat.astype(jnp.int32) * basis).sum(axis=-1)


def fsq_indices_to_codes(indices, levels):
    """Inverse map: integer token ids [...] -> renormalized codes [..., d]."""
    basis = _fsq_basis(levels)
    codes_non_centered = (indices[..., None] // basis) % levels  # 0..L_i-1
    half_width = (levels // 2).astype(jnp.float32)
    return (codes_non_centered.astype(jnp.float32) - half_width) / half_width


# ── Attention building blocks (real Flax attention) ────────────────────────────


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention with an optional additive mask.

    Supports both self-attention (q_in is kv_in) and cross-attention.
    `mask` is a boolean array broadcastable to [B, n_heads, Tq, Tk]; positions
    that are False are forbidden (set to -inf before softmax).
    """

    dim: int
    num_heads: int

    @nn.compact
    def __call__(self, q_in, kv_in, mask=None):
        B, Tq, _ = q_in.shape
        Tk = kv_in.shape[1]
        h, hd = self.num_heads, self.dim // self.num_heads

        q = nn.Dense(self.dim, name='q_proj')(q_in).reshape(B, Tq, h, hd)
        k = nn.Dense(self.dim, name='k_proj')(kv_in).reshape(B, Tk, h, hd)
        v = nn.Dense(self.dim, name='v_proj')(kv_in).reshape(B, Tk, h, hd)

        attn = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(hd)
        if mask is not None:
            attn = jnp.where(mask, attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum('bhqk,bkhd->bqhd', attn, v).reshape(B, Tq, self.dim)
        return nn.Dense(self.dim, name='out_proj')(out)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: (self-attn) -> [optional cross-attn] -> MLP."""

    dim: int
    num_heads: int
    mlp_ratio: int = 4
    cross: bool = False

    @nn.compact
    def __call__(self, x, context=None, self_mask=None, cross_mask=None):
        x = x + MultiHeadAttention(self.dim, self.num_heads, name='self_attn')(
            nn.LayerNorm()(x), nn.LayerNorm()(x), self_mask
        )
        if self.cross:
            xq = nn.LayerNorm()(x)
            ctx = nn.LayerNorm()(context)
            x = x + MultiHeadAttention(self.dim, self.num_heads, name='cross_attn')(
                xq, ctx, cross_mask
            )
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.dim * self.mlp_ratio)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.dim)(y)
        return x + y


def _causal_mask(length):
    """Lower-triangular [length, length] boolean self-attention mask."""
    return jnp.tril(jnp.ones((length, length), dtype=bool))


def sinusoidal_embedding(length, dim):
    """Fixed (non-learned) sinusoidal positional embeddings [length, dim].

    Standard Vaswani et al. (2017) construction. QueST uses *fixed sinusoidal*
    positional embeddings for the decoder query inputs and the prior's skill
    tokens (paper Sec. 4.1-4.2), so we use this rather than learned `nn.param`s
    at those two sites to match the paper exactly.
    """
    pos = jnp.arange(length)[:, None]
    idx = jnp.arange(dim)[None, :]
    angle_rates = 1.0 / jnp.power(10000.0, (2 * (idx // 2)) / dim)
    angles = pos * angle_rates
    return jnp.where(idx % 2 == 0, jnp.sin(angles), jnp.cos(angles))


# ── Stage I: chunked action autoencoder (paper Sec. 4.1) ───────────────────────


class CausalConv1d(nn.Module):
    """Left-padded (causal) strided 1-D convolution over the time axis."""

    features: int
    kernel_size: int
    stride: int

    @nn.compact
    def __call__(self, x):
        pad = self.kernel_size - 1
        x = jnp.pad(x, ((0, 0), (pad, 0), (0, 0)))
        return nn.Conv(
            self.features, (self.kernel_size,), strides=(self.stride,), padding='VALID'
        )(x)


class ActionEncoder(nn.Module):
    """phi_theta: action chunk [B, T, A] -> latent tokens [B, n, fsq_dim].

    Causal strided convolutions downsample time by F, followed by masked
    (causal) self-attention (QueST Sec. 4.1; Table 3). The final linear layer
    projects to the FSQ latent dimension d = len(levels).
    """

    dim: int
    fsq_dim: int
    conv_kernels: Sequence[int]
    conv_strides: Sequence[int]
    num_layers: int
    num_heads: int

    @nn.compact
    def __call__(self, action_chunk):
        x = action_chunk
        for k, s in zip(self.conv_kernels, self.conv_strides):
            x = CausalConv1d(self.dim, k, s)(x)
            # Official Conv1dBlock (skill_vae.py): GroupNorm(4) + Mish.
            x = nn.GroupNorm(num_groups=4)(x)
            x = x * jnp.tanh(jax.nn.softplus(x))  # Mish
        n = x.shape[1]
        # Fixed sinusoidal positional embeddings (the official encoder uses
        # Summer(PositionalEncoding1D)), not a learned parameter.
        x = x + sinusoidal_embedding(n, self.dim)[None]
        mask = _causal_mask(n)[None, None]
        for _ in range(self.num_layers):
            x = TransformerBlock(self.dim, self.num_heads)(x, self_mask=mask)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.fsq_dim)(x)  # [B, n, fsq_dim]


class ActionDecoder(nn.Module):
    """psi_theta: latent codes [B, n, fsq_dim] -> reconstructed chunk [B, T, A].

    Fixed sinusoidal per-timestep positional query inputs cross-attend to ALL n
    skill tokens; the query self-attention is causal (masked). This matches the
    paper: "the decoder cross attends between fixed sinusoidal positional
    embedding inputs and the skill tokens", "attending to all codes while
    maintaining causality" (QueST Sec. 4.1; Table 3).
    """

    dim: int
    horizon: int
    num_tokens: int
    action_dim: int
    num_layers: int
    num_heads: int

    @nn.compact
    def __call__(self, codes):
        B = codes.shape[0]

        # Official decoder adds NO positional embedding to the code memory tokens
        # (only the sinusoidal query inputs carry positions).
        skill = nn.Dense(self.dim)(codes)

        # Fixed sinusoidal positional embeddings are the decoder query inputs
        # (paper Sec. 4.1), not learned parameters.
        queries = sinusoidal_embedding(self.horizon, self.dim)
        x = jnp.broadcast_to(queries[None], (B, self.horizon, self.dim))

        self_mask = _causal_mask(self.horizon)[None, None]
        # The decoder cross-attends to ALL skill codes (cross_mask=None); causality
        # is maintained by the masked self-attention over query positions, not by
        # masking the cross-attention (paper Sec. 4.1).
        cross_mask = None

        for _ in range(self.num_layers):
            x = TransformerBlock(self.dim, self.num_heads, cross=True)(
                x, context=skill, self_mask=self_mask, cross_mask=cross_mask
            )
        x = nn.LayerNorm()(x)
        return nn.Dense(self.action_dim)(x)  # [B, T, A]


# ── Stage II: autoregressive skill prior (paper Sec. 4.2) ──────────────────────


class SkillPrior(nn.Module):
    """pi_phi: causal Transformer over discrete skill tokens.

    Factorizes pi(Z | state, goal) = prod_i pi(z_i | <s>, z_{<i}, state, goal)
    (QueST Eq. 5). A conditioning token built from the (goal-)state observation
    plays the role of the learnable start token <s> at position 0; teacher-forced
    token embeddings fill positions 1..n-1. The head predicts a distribution over
    the full vocabulary at every position. Trained with NLL / cross-entropy
    (Eq. 6).
    """

    vocab_size: int
    num_tokens: int
    dim: int
    num_layers: int
    num_heads: int
    gc_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations, goals, tokens):
        B = tokens.shape[0]

        # Conditioning embedding from state (+ goal) -> the <s>/prefix token.
        if self.gc_encoder is not None:
            cond_in = self.gc_encoder(observations, goals)
        elif goals is not None:
            cond_in = jnp.concatenate([observations, goals], axis=-1)
        else:
            cond_in = observations
        cond = MLP((self.dim, self.dim), activate_final=True)(cond_in)  # [B, dim]

        tok_embed = nn.Embed(self.vocab_size, self.dim, name='tok_embed')
        # Input sequence: [cond, e(z_0), ..., e(z_{n-2})], length n.
        prev = tok_embed(tokens[:, : self.num_tokens - 1])  # [B, n-1, dim]
        x = jnp.concatenate([cond[:, None, :], prev], axis=1)  # [B, n, dim]
        # Fixed sinusoidal positional embeddings over the token sequence
        # (paper Sec. 4.2: "sinusoidal positional embeddings"), not learned.
        x = x + sinusoidal_embedding(self.num_tokens, self.dim)[None]

        mask = _causal_mask(self.num_tokens)[None, None]
        for _ in range(self.num_layers):
            x = TransformerBlock(self.dim, self.num_heads)(x, self_mask=mask)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.vocab_size, name='head')(x)  # [B, n, vocab]


# ── Agent ──────────────────────────────────────────────────────────────────────


class QueSTAgent(flax.struct.PyTreeNode):
    """QueST offline skill-abstraction agent (discrete FSQ skill tokens)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── helpers ────────────────────────────────────────────────────────────────

    @property
    def _levels(self):
        return jnp.asarray(self.config['fsq_levels'])

    def _get_chunk(self, batch):
        """Return (action chunk [B, T, A], step mask [B, T]).

        Consumes the real sub-trajectory action sequence from `SequenceDataset`
        (`batch['actions_seq']`, [B, T, A]); `batch['seq_mask']` ([B, T]) marks
        real vs padded (past-terminal) steps and must gate any per-timestep loss.
        """
        chunk = batch['actions_seq']  # [B, T, A]
        mask = batch.get('seq_mask')
        if mask is None:
            mask = jnp.ones(chunk.shape[:2], dtype=chunk.dtype)
        return chunk, mask

    def _encode_indices(self, chunk, grad_params=None):
        """Encode + FSQ-quantize a chunk to integer token ids [B, n] (stop-grad)."""
        z = self.network.select('encoder')(chunk, params=grad_params)
        codes = fsq_quantize(z, self._levels)
        indices = fsq_codes_to_indices(jax.lax.stop_gradient(codes), self._levels)
        return z, codes, indices

    # ── losses ───────────────────────────────────────────────────────────────

    def reconstruction_loss(self, batch, grad_params):
        """Stage I L1 reconstruction loss (paper Eq. 4)."""
        chunk, mask = self._get_chunk(batch)
        z, codes, indices = self._encode_indices(chunk, grad_params)
        recon = self.network.select('decoder')(codes, params=grad_params)
        # Mask out padded (past-terminal) timesteps; average over real
        # (step, action-dim) elements only.
        step_mask = mask[..., None]  # [B, T, 1]
        l1_elem = jnp.abs(recon - chunk) * step_mask
        denom = jnp.maximum(step_mask.sum() * chunk.shape[-1], 1.0)
        l1 = l1_elem.sum() / denom
        # Codebook usage diagnostic (fraction of unique tokens in the batch).
        usage = jnp.unique(indices, size=indices.size, fill_value=-1)
        usage = (usage >= 0).sum() / self.config['vocab_size']
        return l1, {
            'recon_l1': l1,
            'code_usage': usage,
            'z_abs_mean': jnp.abs(z).mean(),
        }

    def prior_loss(self, batch, grad_params):
        """Stage II token cross-entropy / NLL (paper Eq. 5-6)."""
        chunk, _ = self._get_chunk(batch)
        # Encoder params are frozen w.r.t. the prior gradient (tokens are integer
        # and stop-gradient'd); pass None so no grad flows into the encoder here.
        _, _, indices = self._encode_indices(chunk, grad_params=None)
        goals = batch.get('actor_goals') if self.config['goal_conditioned'] else None
        logits = self.network.select('prior')(
            batch['observations'], goals, indices, params=grad_params
        )
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, indices)  # [B, n]
        ce = ce.mean()
        acc = (logits.argmax(axis=-1) == indices).mean()
        return ce, {
            'prior_ce': ce,
            'prior_token_acc': acc,
            'prior_perplexity': jnp.exp(ce),
        }

    # ── training ───────────────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        recon_loss, recon_info = self.reconstruction_loss(batch, grad_params)
        info.update({f'ae/{k}': v for k, v in recon_info.items()})

        prior_loss, prior_info = self.prior_loss(batch, grad_params)
        info.update({f'prior/{k}': v for k, v in prior_info.items()})

        # Two-stage gating (offline). Before `stage1_steps` only the autoencoder
        # trains; afterwards the autoencoder is frozen (zero weight -> zero grad)
        # and only the prior trains. `joint_training=True` trains both at once.
        step = jnp.asarray(self.network.step, dtype=jnp.float32)
        if self.config['joint_training']:
            w_recon = jnp.array(1.0)
            w_prior = jnp.array(self.config['prior_weight'])
        else:
            in_stage1 = step < self.config['stage1_steps']
            w_recon = jnp.where(in_stage1, 1.0, 0.0)
            w_prior = jnp.where(in_stage1, 0.0, self.config['prior_weight'])
        info['ae/weight'] = w_recon
        info['prior/weight'] = w_prior

        total = w_recon * recon_loss + w_prior * prior_loss
        info['total_loss'] = total
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )
        return self.replace(network=new_network, rng=new_rng), info

    # ── evaluation / inference (paper Sec. 4.3) ────────────────────────────────

    def _sample_tokens(self, observations, goals, rng, temperature):
        """Autoregressively sample skill tokens [B, n] from the prior.

        Top-k filtering (default k=5) then temperature sampling; temperature==0
        falls back to greedy argmax (used by OGBench eval, which sets temp=0).
        """
        B = observations.shape[0]
        n = self.config['num_tokens']
        top_k = self.config['top_k']

        def sample_one(logits, key):
            kth = jax.lax.top_k(logits, top_k)[0][:, -1:]  # [B, 1]
            logits = jnp.where(logits < kth, jnp.finfo(logits.dtype).min, logits)
            greedy = logits.argmax(axis=-1)
            scaled = logits / jnp.maximum(temperature, 1e-8)
            sampled = jax.random.categorical(key, scaled, axis=-1)
            return jnp.where(temperature == 0, greedy, sampled)

        def body(i, carry):
            tokens, key = carry
            key, sub = jax.random.split(key)
            logits = self.network.select('prior')(observations, goals, tokens)
            next_tok = sample_one(logits[:, i, :], sub)
            tokens = tokens.at[:, i].set(next_tok)
            return tokens, key

        tokens0 = jnp.zeros((B, n), dtype=jnp.int32)
        tokens, _ = jax.lax.fori_loop(0, n, body, (tokens0, rng))
        return tokens

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample skill tokens, decode to an action chunk, return the first action.

        OGBench's evaluation loop calls `sample_actions` every environment step,
        so we replan each step and return a_t (open-loop horizon 1) rather than
        executing T_a actions before replanning (deviation; see quest_NOTES.md).
        """
        if seed is None:
            seed = self.rng

        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        if single_obs:
            observations = observations[None, ...]
            if goals is not None:
                goals = goals[None, ...]

        cond_goals = goals if self.config['goal_conditioned'] else None
        tokens = self._sample_tokens(observations, cond_goals, seed, temperature)
        codes = fsq_indices_to_codes(tokens, self._levels)
        chunk = self.network.select('decoder')(codes)  # [B, T, A]
        actions = jnp.clip(chunk[:, 0, :], -1, 1)

        if single_obs:
            actions = actions[0]
        return actions

    # ── constructor ────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        assert not config['discrete'], 'QueST targets continuous control (Sec. 4).'
        action_dim = ex_actions.shape[-1]
        T = config['horizon_length']
        F = config['downsample_factor']
        assert T % F == 0, 'horizon_length must be divisible by downsample_factor.'
        assert config['sequence_length'] == T, (
            'sequence_length must equal horizon_length so actions_seq spans the chunk.'
        )
        num_tokens = T // F
        levels = tuple(config['fsq_levels'])
        vocab_size = int(np.prod(levels))
        config = dict(config)
        config['num_tokens'] = num_tokens
        config['vocab_size'] = vocab_size

        # Optional visual encoder for the prior's conditioning input.
        prior_encoder = None
        if config.get('encoder') is not None:
            enc = encoder_modules[config['encoder']]
            prior_encoder = GCEncoder(concat_encoder=enc())

        encoder_def = ActionEncoder(
            dim=config['ae_dim'],
            fsq_dim=len(levels),
            conv_kernels=tuple(config['conv_kernels']),
            conv_strides=tuple(config['conv_strides']),
            num_layers=config['enc_layers'],
            num_heads=config['enc_heads'],
        )
        decoder_def = ActionDecoder(
            dim=config['ae_dim'],
            horizon=T,
            num_tokens=num_tokens,
            action_dim=action_dim,
            num_layers=config['dec_layers'],
            num_heads=config['dec_heads'],
        )
        prior_def = SkillPrior(
            vocab_size=vocab_size,
            num_tokens=num_tokens,
            dim=config['prior_dim'],
            num_layers=config['prior_layers'],
            num_heads=config['prior_heads'],
            gc_encoder=prior_encoder,
        )

        ex_chunk = jnp.zeros((ex_observations.shape[0], T, action_dim))
        ex_codes = jnp.zeros((ex_observations.shape[0], num_tokens, len(levels)))
        ex_tokens = jnp.zeros((ex_observations.shape[0], num_tokens), dtype=jnp.int32)
        ex_goals = ex_observations if config['goal_conditioned'] else None

        network_def = ModuleDict(dict(
            encoder=encoder_def, decoder=decoder_def, prior=prior_def,
        ))
        network_params = network_def.init(
            init_rng,
            encoder=(ex_chunk,),
            decoder=(ex_codes,),
            prior=(ex_observations, ex_goals, ex_tokens),
        )['params']

        network = TrainState.create(
            network_def, network_params, tx=optax.adam(config['lr'])
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ──────────────────────────────────────────────────────────────────────


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='quest',
        lr=1e-4,
        batch_size=1024,
        # ── Stage I: chunked action autoencoder (paper Table 3) ──────────────
        horizon_length=32,        # action chunk length T.
        downsample_factor=4,      # F; num_tokens n = T / F = 8.
        fsq_levels=(8, 5, 5, 5),  # FSQ levels; implicit vocab = prod = 1000.
        ae_dim=256,               # autoencoder hidden dim.
        conv_kernels=(5, 3, 3),   # encoder causal-conv kernel sizes.
        conv_strides=(2, 2, 1),   # encoder causal-conv strides (downsample x4).
        enc_layers=2,             # encoder self-attention layers.
        enc_heads=4,
        dec_layers=4,             # decoder transformer layers.
        dec_heads=4,
        # ── Stage II: autoregressive skill prior (paper Table 4) ─────────────
        prior_dim=384,
        prior_layers=6,
        prior_heads=6,
        top_k=5,                  # top-k token sampling at inference.
        prior_weight=1.0,         # weight on the prior CE term.
        # ── Two-stage offline schedule ───────────────────────────────────────
        stage1_steps=500000,      # steps of AE-only training before prior-only.
        joint_training=False,     # if True, train AE + prior simultaneously.
        goal_conditioned=True,    # condition the prior on the goal observation.
        # ── Standard OGBench plumbing ────────────────────────────────────────
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='SequenceDataset',
        sequence_length=32,       # = horizon_length T; window length for SequenceDataset.
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
