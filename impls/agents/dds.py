"""Discrete Diffusion Skills (DDS) agent.

================================================================================
ASSUMPTIONS & DEVIATIONS FROM THE PAPER (arXiv:2503.20176)
--------------------------------------------------------------------------------
Read this first. This is the authoritative summary; `dds_NOTES.md` expands each
item with the equation-by-equation mapping. The architecture, diffusion decoder,
semi-MDP snippet return + single-gamma bootstrap, separate optimizers + hard
phased freeze, all hyperparameters, and H-step eval commitment now match the
paper. Everything below is either an IRREDUCIBLE consequence of the OGBench
setting or a genuinely paper-UNSPECIFIED detail (with the cited-method
convention named).

(a) IRREDUCIBLE ADAPTATIONS FORCED BY THE OGBench SETTING (kept, documented)
  A1. Goal-conditioning. The paper trains an UNCONDITIONAL IQL on a relabeled
      semi-MDP. OGBench has no separate task reward, so the value/critic/actor
      are goal-conditioned and trained with the GCDataset goal sampler — exactly
      how OGBench's reference `gciql`/`hiql` instantiate IQL/HIQL. (Sec. 3.3.)
  A2. Per-step reward = OGBench's goal-conditioned reward. The snippet return
      Sigma_{i<H} gamma^i r_i (Eq. 8, faithful) sums the GC reward at each window
      step w.r.t. the SAME value goal (computed by `SequenceDataset` exactly as
      `GCDataset` computes a single-transition reward). There is no other reward
      signal in OGBench. This is the standard OGBench instantiation, not a
      DDS-specific shortcut.

(b) GENUINELY PAPER-UNSPECIFIED DETAILS (cited-method convention used)
  B1. Discrete-action envs. The diffusion decoder is defined for continuous
      actions (the paper's AntMaze/Kitchen). For discrete-action OGBench envs we
      substitute a skill-conditioned categorical BC decoder (masked
      cross-entropy); the VQ skill machinery and high level are identical. The
      paper does not cover discrete action spaces.
  B2. AWR temperature alpha. Table 7 does not list the AWR alpha; we use the
      IQL/AWR convention alpha = 3.0 (`high_alpha`).
  B3. Unspecified activations + dropout. All paper-SPECIFIED activations are now
      matched: the encoder MLP head (ReLU, A.1.1) and the value/Q/policy MLPs
      (ReLU, A.1.3-A.1.5). Only the diffusion-decoder block and transformer-FFN
      activations are genuinely unspecified by the paper; those use OGBench's GELU.
      The paper's 0.1 dropout (Tables 4-5) is omitted (deterministic), since
      OGBench's ModuleDict apply path does not thread a dropout rng — a minor
      regularization detail with no effect on the objective. The value/Q MLPs also
      use LayerNorm (OGBench's IQL/gciql default); the paper does not mention
      LayerNorm for these nets — another unspecified normalization detail.
  B4. Phase-2 budget. The paper trains skill 500k steps, then Q-learning 1M +
      AWR 500k as SEPARATE runs. We honor the 500k skill phase, then train the
      whole high level for OGBench's remaining single-run budget (default 500k).
      This is a compute-budget choice, not a method change.
  B5. Codebook init. The paper says "standard vector quantization (Van Den Oord
      et al. 2017)" and nothing more; we use that paper's init, uniform in
      [-1/K, 1/K]. (Runs before 2026-09-01 used a unit-normal init and, together
      with a mean- rather than sum-reduced reconstruction loss, collapsed to a
      single active code within 5k steps; see `VQCodebook` and `skill_loss`.)
================================================================================

Faithful OGBench re-implementation of

    "Offline Reinforcement Learning with Discrete Diffusion Skills"
    (DDS), arXiv:2503.20176.

The method has three coupled components, all trained purely offline from the
OGBench dataset:

  1. A VQ-VAE skill model (Sec. 3.2 / Eq. 14):
       - a *transformer sub-sequence encoder* E(tau_H) -> embedding (Table 4:
         4 layers, 8 heads, 256-d, learnable positional encoding, masked average
         pooling, 2-layer MLP head) over the real length-H window
         {s_i, a_i}_{i=t..t+H-1} (the `SequenceDataset` observations_seq /
         actions_seq with seq_mask),
       - a *discrete skill codebook* {z_k}_{k=1..K} with nearest-neighbour
         (argmin) assignment and the straight-through estimator,
       - a *diffusion decoder* (Table 5: input projection -> 4 residual blocks
         each with a 4x-expand-then-compress LayerNorm-MLP + sinusoidal time
         embedding -> output projection) that reconstructs the whole H-step
         ACTION BLOCK D(z, s_i) -> a_i by epsilon-prediction with the VP
         beta-schedule (categorical decoder for discrete-action envs).
     Losses: action-block reconstruction (seq_mask applied) + codebook loss
             ||sg[E]-z||^2 + beta * commitment loss ||sg[z]-E||^2  (beta = 0.25).

  2. A high-level semi-MDP IQL value/critic (Sec. 3.3, Eq. 6-9): a skill-indexed
     critic Q(s, k, g) over the K codes regressed toward the semi-MDP target
     R_H + gamma * V(s_{t+H}, g) with the H-step discounted snippet return
     R_H = Sigma_{i=0}^{H-1} gamma^i r_i and a SINGLE-gamma bootstrap (Eq. 8),
     and an implicit value V(s, g) trained by expectile regression toward the
     (target) critic L2^tau(Q_target(s, k*, g) - V(s, g)) (Eq. 6). Canonical IQL.

  3. A high-level policy that *selects discrete codes* (Sec. 3.3): a categorical
     actor pi_h(k | s, g) over the K codebook indices, trained with
     advantage-weighted regression (AWR) using the skill-indexed advantage
     exp(alpha * (Q(s, k, g) - V(s, g))) toward the VQ-assigned skill index.

Training is a true two-phase procedure within one run (the paper's
relabel-then-train): SEPARATE Adam optimizers (skill lr=5e-5, IQL lr=1e-4) via
`optax.multi_transform`. For the first `skill_pretrain_steps` only the skill
VQ-VAE updates (the IQL optimizer is hard-zeroed); afterwards the skill modules
are HARD-frozen (zero updates AND zero Adam momentum) and only the high-level
value/critic/policy train on the now-fixed codes/labels.

See `dds_NOTES.md` for an equation-by-equation mapping.
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
from utils.networks import GCDiscreteActor, GCDiscreteCritic, GCValue, MLP


# ── Diffusion helpers ─────────────────────────────────────────────────────────


def timestep_embedding(timesteps, dim, max_period=10000.0):
    """Sinusoidal time embedding (Table 5: size-16 sinusoidal time embedding)."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / jnp.maximum(half, 1))
    args = timesteps[..., None].astype(jnp.float32) * freqs
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[..., :1])], axis=-1)
    return emb


def diffusion_schedule(num_steps, beta_min=0.1, beta_max=10.0):
    """VP (variance-preserving) beta-schedule (Song et al. 2020).

    Matches the paper's diffusion schedule (Table 5: VP schedule with
    beta_min=0.1, beta_max=10). The continuous VP SDE marginal
    alpha_bar(t) = exp(-1/2 * int_0^t beta(s) ds) with linear
    beta(t) = beta_min + t*(beta_max - beta_min), t in [0, 1], is evaluated on
    the `num_steps` denoising grid and the per-step betas are recovered from the
    cumulative alpha_bar ratio (clipped for a valid ancestral sampler).

    Returns per-step (betas, alphas, alpha_bar, alpha_bar_prev), each length
    `num_steps`.
    """
    t = jnp.arange(num_steps + 1, dtype=jnp.float32) / num_steps  # t in [0, 1]
    integral = beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2  # int_0^t beta
    abar_full = jnp.exp(-0.5 * integral)  # length T+1, abar_full[0] = 1.
    alpha_bar = abar_full[1:]             # cumulative alpha at step t (t = 0..T-1)
    alpha_bar_prev = abar_full[:-1]       # cumulative alpha at step t-1
    betas = jnp.clip(1.0 - alpha_bar / alpha_bar_prev, 1e-5, 0.999)
    alphas = 1.0 - betas
    return betas, alphas, alpha_bar, alpha_bar_prev


# ── Network modules ───────────────────────────────────────────────────────────


def _encode_sequence(state_encoder, obs_seq):
    """Apply a (visual) state encoder per timestep of a [B, T, *obs] window.

    Folds the time axis into the batch axis, encodes, then unfolds. Returns the
    raw window unchanged when no encoder is given (vector observations).
    """
    if state_encoder is None:
        return obs_seq
    b, t = obs_seq.shape[0], obs_seq.shape[1]
    flat = obs_seq.reshape((b * t,) + obs_seq.shape[2:])
    enc = state_encoder(flat)
    return enc.reshape((b, t) + enc.shape[1:])


class TransformerEncoderBlock(nn.Module):
    """Pre-LN transformer encoder block (Table 4: 8 heads, 256-d, 4x MLP).

    Masked self-attention over the token stream followed by a position-wise MLP,
    both with residual connections. `mask` ([B, T] boolean) excludes padded
    (past-terminal) window steps from attention.
    """

    dim: int
    num_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, mask):
        attn_mask = mask[:, None, None, :]  # [B, 1, 1, T] -> broadcast over heads/queries
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.dim)(
            h, h, mask=attn_mask
        )
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.dim * self.mlp_ratio)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.dim)(h)
        return x + h


class TrajectoryEncoder(nn.Module):
    """Transformer sub-sequence encoder E(tau_H) -> e in R^{D_z} (Sec. 3.2, Table 4).

    Consumes the real length-H sub-trajectory window
    tau_H = {s_i, a_i}_{i=t..t+H-1} supplied by `SequenceDataset`
    (`observations_seq` [B, T, *obs], `actions_seq` [B, T, *act]). Architecture
    (Table 4): per-step concat [enc(s_i), a_i] -> linear input embedding ->
    learnable positional encoding -> `num_layers` transformer blocks (`num_heads`
    heads, `hidden_dim`-d) -> masked average pooling over time -> 2-layer MLP
    projection to D_z. `seq_mask` excludes padded steps from both attention and
    the pool, so the skill embedding only sees genuine in-trajectory tokens.
    """

    hidden_dim: int
    num_layers: int
    num_heads: int
    skill_dim: int
    state_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations_seq, actions_seq, seq_mask):
        obs = _encode_sequence(self.state_encoder, observations_seq)  # [B, T, d]
        x = jnp.concatenate([obs, actions_seq], axis=-1)              # [B, T, d + act]
        x = nn.Dense(self.hidden_dim)(x)                             # token embedding
        T = x.shape[1]
        pos = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (1, T, self.hidden_dim))
        x = x + pos
        bool_mask = seq_mask > 0
        for _ in range(self.num_layers):
            x = TransformerEncoderBlock(self.hidden_dim, self.num_heads)(x, bool_mask)
        x = nn.LayerNorm()(x)
        # Masked average pooling (Table 4 'adaptive avg pooling'); padded steps excluded.
        m = seq_mask[..., None]
        pooled = (x * m).sum(axis=1) / jnp.maximum(m.sum(axis=1), 1e-6)  # [B, hidden]
        # 2-layer MLP head with ReLU (paper A.1.1: "two-layer MLP (with ReLU activation)").
        h = nn.relu(nn.Dense(self.hidden_dim)(pooled))
        e = nn.Dense(self.skill_dim)(h)                                 # [B, D_z]
        return e


class VQCodebook(nn.Module):
    """Discrete skill codebook with nearest-neighbour quantization (Eq. 10-14).

    Holds {z_k}_{k=1..K}, z_k in R^{D_z}.  Forward pass returns the
    straight-through quantized embedding, the assigned indices, and the two VQ
    loss terms (per-sample). Codebook lookup for sampling reads the raw param.
    """

    num_codes: int
    code_dim: int

    @nn.compact
    def __call__(self, e):
        # Standard VQ-VAE init (van den Oord et al. 2017): codes uniform in [-1/K, 1/K],
        # i.e. all near the origin. A unit-normal init in D_z=128 dims gave codes of
        # norm ~11 that were far apart; the untrained encoder mapped nearly every window
        # onto the one code most aligned with its output, and the codes that received
        # no assignment at step 0 never got a gradient (codebook loss reaches only the
        # assigned code), so every pretraining run collapsed to a single active code.
        bound = 1.0 / self.num_codes

        def codebook_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype, -bound, bound)

        codebook = self.param('codebook', codebook_init, (self.num_codes, self.code_dim))
        # Nearest-neighbour assignment:  k = argmin_j ||E(tau) - z_j||  (Eq. 10/11).
        dists = jnp.sum((e[:, None, :] - codebook[None, :, :]) ** 2, axis=-1)  # [B, K]
        indices = jnp.argmin(dists, axis=-1)                                   # [B]
        z_q = codebook[indices]                                                # [B, D_z]
        # VQ losses (Eq. 14). sg = stop-gradient.
        codebook_loss = jnp.sum((jax.lax.stop_gradient(e) - z_q) ** 2, axis=-1)     # ||sg[E]-z||^2
        commitment_loss = jnp.sum((e - jax.lax.stop_gradient(z_q)) ** 2, axis=-1)   # ||sg[z]-E||^2
        # Straight-through estimator: gradient of z_q_st w.r.t. e is identity.
        z_q_st = e + jax.lax.stop_gradient(z_q - e)
        return z_q_st, indices, codebook_loss, commitment_loss


class ResidualDenoiseBlock(nn.Module):
    """Residual denoiser block (Table 5: linear expand 4x, compress back, LayerNorm).

    Pre-LayerNorm residual MLP that expands the hidden width by `expand` and
    compresses it back, matching the paper's 'Dense Fully Connected Layer Blocks'.
    """

    dim: int
    expand: int = 4

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.dim * self.expand)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.dim)(h)
        return x + h


class DiffusionActionDecoder(nn.Module):
    """Low-level diffusion decoder D(z, s) -> a (Sec. 3.2 / Table 5).

    Noise-prediction (epsilon) network conditioned on (noisy action x_t, time t,
    state s, skill z). Structure (Table 5): sinusoidal time embedding -> input
    projection -> `num_blocks` residual blocks (each 4x-expand-then-compress
    LayerNorm-MLP) -> output projection.
    """

    hidden_dim: int
    num_blocks: int
    action_dim: int
    expand: int = 4
    time_dim: int = 16  # Paper uses a size-16 sinusoidal time embedding (Table 5).
    state_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, noisy_actions, times, observations, skills):
        t_emb = timestep_embedding(times, self.time_dim)
        obs = self.state_encoder(observations) if self.state_encoder is not None else observations
        x = jnp.concatenate([noisy_actions, t_emb, obs, skills], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)  # input projection
        for _ in range(self.num_blocks):
            x = ResidualDenoiseBlock(self.hidden_dim, self.expand)(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.action_dim)(x)  # output projection


class SkillDiscreteDecoder(nn.Module):
    """Low-level decoder for *discrete*-action environments.

    The diffusion decoder is defined for continuous actions (AntMaze/Kitchen).
    For discrete-action OGBench envs we substitute a skill-conditioned
    categorical behaviour-cloning decoder (see flag block B1). It reconstructs
    a ~ p(a | s, z).
    """

    hidden_dims: Sequence[int]
    action_dim: int
    state_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations, skills, temperature=1.0):
        obs = self.state_encoder(observations) if self.state_encoder is not None else observations
        x = jnp.concatenate([obs, skills], axis=-1)
        h = MLP(self.hidden_dims, activate_final=True)(x)
        logits = nn.Dense(self.action_dim)(h)
        return distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))


# ── Separate phased optimizer (Sec. 4 training: relabel-then-train) ────────────


def make_phased_adam(lr, threshold, train_in_phase1):
    """Adam that is HARD-frozen outside its phase (zero updates AND zero momentum).

    `optax.multi_transform` routes the skill modules to a phase-1 Adam and the
    IQL modules to a phase-2 Adam. Each carries its own step counter; the step is
    `count + 1` to mirror `TrainState.step` (which starts at 1), so the freeze
    flips on exactly the same step as the loss-weight gate in `total_loss`.

    When inactive the transform (a) zeroes the parameter updates and (b) zeroes
    the wrapped Adam's first/second moments, so a frozen module's parameters do
    not drift at all (skill-param max-change == 0 in phase 2).
    """
    base = optax.adam(lr)

    def init_fn(params):
        return (jnp.zeros((), jnp.int32), base.init(params))

    def update_fn(updates, state, params=None):
        count, inner = state
        step = count + 1  # mirror TrainState.step (1-indexed)
        in_phase1 = step < threshold
        active = in_phase1 if train_in_phase1 else jnp.logical_not(in_phase1)
        upd_active, inner_active = base.update(updates, inner, params)
        # Inactive: emit zero updates AND leave the inner Adam state UNTOUCHED, so a
        # frozen optimizer never advances its count/moments. On (re)activation it
        # therefore behaves as a fresh Adam with correct early-step bias correction,
        # avoiding the cold-moments-but-huge-count transient at the phase boundary
        # (the IQL optimizer otherwise enters phase 2 with count~=skill_pretrain_steps
        # and disabled bias correction).
        new_updates = jax.tree_util.tree_map(
            lambda u: jnp.where(active, u, jnp.zeros_like(u)), upd_active
        )
        new_inner = jax.tree_util.tree_map(
            lambda a, old: jnp.where(active, a, old), inner_active, inner
        )
        return new_updates, (count + 1, new_inner)

    return optax.GradientTransformation(init_fn, update_fn)


# ── Agent ─────────────────────────────────────────────────────────────────────

_SKILL_MODULES = ('modules_encoder', 'modules_codebook', 'modules_decoder')


class DDSAgent(flax.struct.PyTreeNode):
    """Discrete Diffusion Skills (DDS) offline agent (arXiv:2503.20176).

    Module layout (ModuleDict):
        encoder             transformer sub-sequence encoder E(tau_H) (Table 4)
        codebook            discrete VQ skill codebook {z_k}
        decoder             low-level diffusion (or categorical) action-block decoder
        value               high-level goal-conditioned IQL value V(s, g)
        high_critic         skill-indexed semi-MDP critic Q(s, k, g) over K codes
        target_high_critic  target critic network (IQL Eq. 6 / AWR)
        high_actor          high-level categorical policy pi_h(k | s, g) over K codes
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── Small helpers ─────────────────────────────────────────────────────────

    def _action_vec(self, actions):
        """Encoder-side action representation (one-hot for discrete actions).

        Works for both single actions [B] / [B, act] and action sequences
        [B, T] / [B, T, act].
        """
        if self.config['discrete']:
            # Robust to a trailing integer-column axis: (..., 1) -> (...) so one_hot
            # yields (..., action_dim) rather than (..., 1, action_dim).
            if actions.shape and actions.shape[-1] == 1:
                actions = jnp.squeeze(actions, axis=-1)
            return jax.nn.one_hot(actions, self.config['action_dim'])
        return actions

    def _codebook_table(self):
        """Raw codebook matrix [K, D_z] for lookup at sampling time."""
        return self.network.params['modules_codebook']['codebook']

    def _assign_skill(self, observations_seq, actions_seq, seq_mask, params=None):
        """Return the VQ-assigned skill indices for a length-H window (argmin).

        Reads the real sub-trajectory window (observations_seq/actions_seq +
        seq_mask), matching the encoder.
        """
        e = self.network.select('encoder')(
            observations_seq, self._action_vec(actions_seq), seq_mask, params=params
        )
        _, indices, _, _ = self.network.select('codebook')(e, params=params)
        return indices

    # ── VQ-VAE skill loss (Sec. 3.2, Eq. 14) ──────────────────────────────────

    def skill_loss(self, batch, grad_params, rng):
        """L_skill = recon + ||sg[E]-z||^2 + beta*||sg[z]-E||^2  (Eq. 13-14).

        The encoder consumes the real length-H window and the decoder
        reconstructs the whole H-step ACTION BLOCK D(z, s_i)->a_i (one denoising
        target / categorical target per in-window step), with `seq_mask` applied
        so padded past-terminal steps contribute zero loss. The skill phase uses
        its own (smaller) batch (`skill_batch_size`, paper Table 6: 128); we slice
        the leading rows of the shared batch so the skill and IQL phases use the
        paper's 128 / 256 batch sizes within OGBench's single-batch pipeline.
        """
        skill_bs = min(int(self.config['skill_batch_size']), batch['seq_mask'].shape[0])
        obs_seq = batch['observations_seq'][:skill_bs]   # [b, T, *obs]
        act_seq = batch['actions_seq'][:skill_bs]        # [b, T, *act] or [b, T]
        seq_mask = batch['seq_mask'][:skill_bs]          # [b, T]
        b, t_len = seq_mask.shape

        # Encode the window -> continuous embedding -> quantized skill (shared z).
        e = self.network.select('encoder')(obs_seq, self._action_vec(act_seq), seq_mask, params=grad_params)
        z_q_st, indices, codebook_loss, commitment_loss = self.network.select('codebook')(e, params=grad_params)

        beta = self.config['commitment_beta']

        # Flatten the (b, T) window into a batch of per-step decode problems; the
        # decoder is conditioned on the per-step state s_i and the shared skill z.
        obs_flat = obs_seq.reshape((b * t_len,) + obs_seq.shape[2:])
        mask_flat = seq_mask.reshape(b * t_len)
        # Eq. 13 reduction: the reconstruction error is a SUM over the H window steps
        # (and, inside ||.||^2, over action dims) per training example, then averaged
        # over the batch like the VQ terms. Averaging over steps and action dims instead
        # left the reconstruction gradient ~H*|A| (80x on ant) weaker than the paper's
        # relative to the codebook/commitment sums over the D_z=128 latent dims, so the
        # encoder was pulled onto its assigned code far harder than it was pushed to
        # encode anything, which drove codebook collapse.
        denom = jnp.asarray(b, jnp.float32)
        # Broadcast the shared skill across the window: [b, D_z] -> [b*T, D_z].
        skills_flat = jnp.broadcast_to(z_q_st[:, None, :], (b, t_len, z_q_st.shape[-1])).reshape(b * t_len, -1)

        if self.config['discrete']:
            # Categorical reconstruction of each action: -log p(a_i | s_i, z), summed
            # over the window steps per example (the discrete analogue of Eq. 13).
            act_flat = act_seq.reshape(b * t_len)
            dist = self.network.select('decoder')(obs_flat, skills_flat, params=grad_params)
            nll = -dist.log_prob(act_flat)                       # [b*T]
            recon_loss = (nll * mask_flat).sum() / denom
        else:
            # Diffusion reconstruction (epsilon-prediction MSE) per window step.
            act_flat = act_seq.reshape((b * t_len,) + act_seq.shape[2:])
            T = self.config['diffusion_steps']
            betas, alphas, alpha_bar, alpha_bar_prev = diffusion_schedule(
                T, self.config['beta_min'], self.config['beta_max']
            )
            rng_t, rng_noise = jax.random.split(rng)
            tt = jax.random.randint(rng_t, (b * t_len,), 0, T)
            ab = alpha_bar[tt][:, None]
            noise = jax.random.normal(rng_noise, act_flat.shape)
            x_t = jnp.sqrt(ab) * act_flat + jnp.sqrt(1.0 - ab) * noise
            times = tt.astype(jnp.float32) / T
            pred_noise = self.network.select('decoder')(x_t, times, obs_flat, skills_flat, params=grad_params)
            per_step = jnp.sum((pred_noise - noise) ** 2, axis=-1)  # [b*T]  ||eps - eps_psi||^2 (Eq. 5/14)
            recon_loss = (per_step * mask_flat).sum() / denom

        codebook_loss = codebook_loss.mean()
        commitment_loss = commitment_loss.mean()
        loss = recon_loss + codebook_loss + beta * commitment_loss

        # Codebook usage / perplexity (codebook collapse is the main VQ-skill
        # failure mode, so log both the entropy-based perplexity in [1, K] and the
        # directly-interpretable count/fraction of codes actually used this batch).
        K = self.config['num_skills']
        onehot = jax.nn.one_hot(indices, K)
        avg_probs = onehot.mean(axis=0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))
        active_codes = (avg_probs > 0).sum()          # # of the K codes used (collapse -> small)
        code_usage = active_codes / K                  # fraction in [0, 1]

        return loss, {
            'skill_loss': loss,
            'recon_loss': recon_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
            'active_codes': active_codes,
            'code_usage': code_usage,
        }

    @staticmethod
    def _expectile_loss(diff, expectile):
        """Asymmetric L2 (expectile) loss, IQL Eq. 6."""
        weight = jnp.where(diff >= 0, expectile, 1 - expectile)
        return weight * diff ** 2

    # ── High-level IQL value loss V(s, g) (Sec. 3.3, Eq. 6) ────────────────────

    def value_loss(self, batch, grad_params):
        """IQL implicit value (Eq. 6): V regressed toward the target critic.

        V(s, g) is trained to be the tau-expectile of the skill-indexed target
        critic Q_target(s, k*, g) over the dataset code k*:
            L_V = E[ L2^tau( Q_target(s, k*, g) - V(s, g) ) ].
        """
        k_star = self._assign_skill(
            batch['observations_seq'], batch['actions_seq'], batch['seq_mask'], params=None
        )
        q1, q2 = self.network.select('target_high_critic')(batch['observations'], batch['value_goals'], k_star)
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self._expectile_loss(q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    # ── Skill-indexed semi-MDP critic Q(s, k, g) (Sec. 3.3, Eq. 8) ────────────

    def high_critic_loss(self, batch, grad_params):
        """Skill-indexed semi-MDP IQL critic Q(s, k, g) (Eq. 8).

        Faithful Eq. 8 macro-transition target with the H-step discounted snippet
        return and a SINGLE-gamma bootstrap:
            target = Sigma_{i=0}^{H-1} gamma^i r_i  +  gamma * mask_H * V(s_{t+H}, g),
        where r_i / mask_H are the goal-conditioned per-window reward/mask supplied
        by `SequenceDataset` (`rewards_seq` / `masks_seq`, computed exactly as
        `GCDataset` does for a single transition), s_{t+H} = `subgoal_observations`,
        and k* is the frozen VQ skill label. The macro mask cuts the bootstrap
        once the goal is reached inside the window.
        """
        observations = batch['observations']
        goals = batch['value_goals']
        gamma = self.config['discount']
        H = int(self.config['sequence_length'])

        seq_mask = batch['seq_mask']        # [B, H]
        rewards_seq = batch['rewards_seq']  # [B, H]
        masks_seq = batch['masks_seq']      # [B, H]

        # H-step discounted snippet return Sigma_{i<H} gamma^i r_i (padded steps zeroed).
        discounts = gamma ** jnp.arange(H, dtype=jnp.float32)          # [H]
        snippet_return = (rewards_seq * seq_mask * discounts[None, :]).sum(axis=1)  # [B]
        # Macro mask: bootstrap only if the goal was never reached inside the window.
        eff_mask = jnp.where(seq_mask > 0, masks_seq, 1.0)
        macro_mask = jnp.prod(eff_mask, axis=1)                        # [B]

        # Frozen VQ skill label k* for this window.
        k_star = self._assign_skill(
            batch['observations_seq'], batch['actions_seq'], batch['seq_mask'], params=None
        )

        # Single-gamma bootstrap on the macro-transition (Eq. 8) to V(s_{t+H}, g).
        next_v = self.network.select('value')(batch['subgoal_observations'], goals)
        target = snippet_return + gamma * macro_mask * next_v
        target = jax.lax.stop_gradient(target)

        q1, q2 = self.network.select('high_critic')(observations, goals, k_star, params=grad_params)
        critic_loss = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()

        return critic_loss, {
            'high_critic_loss': critic_loss,
            'q_mean': ((q1 + q2) / 2).mean(),
            'q_target_mean': target.mean(),
            'snippet_return_mean': snippet_return.mean(),
        }

    # ── High-level discrete-code policy loss (Sec. 3.3, Eq. 9, AWR) ───────────

    def high_actor_loss(self, batch, grad_params):
        """AWR over discrete skill codes (Eq. 9).

        The weight uses the skill-indexed advantage A = Q(s, k*, g) - V(s, g),
        weight min(exp(alpha * A), 100). The BC target is the (frozen)
        VQ-assigned code k* for the executed sub-trajectory.
        """
        observations = batch['observations']
        goals = batch['actor_goals']

        # Target discrete code k* from the (frozen) VQ skill model.
        target_skills = self._assign_skill(
            batch['observations_seq'], batch['actions_seq'], batch['seq_mask'], params=None
        )

        # Skill-indexed advantage A = Q(s, k*, g) - V(s, g) (constant w.r.t. actor
        # params: critic/value read stored params, not grad_params).
        q1, q2 = self.network.select('high_critic')(observations, goals, target_skills)
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(observations, goals)
        adv = q - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(observations, goals, params=grad_params)
        log_prob = dist.log_prob(target_skills)

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'high_actor_loss': actor_loss,
            'adv': adv.mean(),
            'exp_a': exp_a.mean(),
            'bc_log_prob': log_prob.mean(),
            'entropy': dist.entropy().mean(),
        }

    # ── Training ──────────────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        info = {}

        skill_rng, _ = jax.random.split(rng)
        skill_loss, skill_info = self.skill_loss(batch, grad_params, skill_rng)
        info.update({f'skill/{k}': v for k, v in skill_info.items()})

        value_loss, value_info = self.value_loss(batch, grad_params)
        info.update({f'value/{k}': v for k, v in value_info.items()})

        high_critic_loss, high_critic_info = self.high_critic_loss(batch, grad_params)
        info.update({f'high_critic/{k}': v for k, v in high_critic_info.items()})

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        info.update({f'high_actor/{k}': v for k, v in high_actor_info.items()})

        # Two-phase, step-gated schedule (the paper's relabel-then-train). For the
        # first `skill_pretrain_steps` only the skill VQ-VAE loss has weight (and
        # only the skill optimizer is active); afterwards the skill loss weight is
        # 0 and the high level trains on the now-stationary codes/labels. The
        # optimizer freeze (make_phased_adam) flips on the SAME step, so the skill
        # modules receive zero update AND zero momentum in phase 2.
        step = jnp.asarray(self.network.step)
        in_phase1 = step < self.config['skill_pretrain_steps']
        skill_w = jnp.where(in_phase1, 1.0, 0.0)
        high_w = jnp.where(in_phase1, 0.0, 1.0)
        info['schedule/phase1'] = skill_w

        loss = skill_w * skill_loss + high_w * (value_loss + high_critic_loss + high_actor_loss)
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=lambda p: self.total_loss(batch, p, rng=rng))

        # Soft-update the target critic network (IQL Eq. 6 / AWR target).
        new_target_critic = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            new_network.params['modules_high_critic'],
            new_network.params['modules_target_high_critic'],
        )
        new_params = {**new_network.params, 'modules_target_high_critic': new_target_critic}
        new_network = new_network.replace(params=new_params)
        return self.replace(network=new_network, rng=new_rng), info

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _ddpm_sample(self, observations, skills, rng):
        """Ancestral DDPM sampling of an action conditioned on (s, z)."""
        T = self.config['diffusion_steps']
        betas, alphas, alpha_bar, alpha_bar_prev = diffusion_schedule(
            T, self.config['beta_min'], self.config['beta_max']
        )
        # Leading batch dims come from the skill tensor (robust to image obs).
        batch_shape = skills.shape[:-1]
        rng, noise_rng = jax.random.split(rng)
        x = jax.random.normal(noise_rng, (*batch_shape, self.config['action_dim']))
        for t in reversed(range(T)):
            times = jnp.full(batch_shape, t / T)
            eps = self.network.select('decoder')(x, times, observations, skills)
            a_t = alphas[t]
            ab_t = alpha_bar[t]
            b_t = betas[t]
            mean = (x - (b_t / jnp.sqrt(1.0 - ab_t)) * eps) / jnp.sqrt(a_t)
            if t > 0:
                rng, z_rng = jax.random.split(rng)
                x = mean + jnp.sqrt(b_t) * jax.random.normal(z_rng, x.shape)
            else:
                x = mean
        return x

    def _decode(self, observations, skills, seed, temperature):
        """Low-level decode a ~ D(z, s): DDPM (continuous) / categorical (discrete)."""
        if self.config['discrete']:
            low_dist = self.network.select('decoder')(observations, skills, temperature=temperature)
            return low_dist.sample(seed=seed)
        actions = self._ddpm_sample(observations, skills, seed)
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Stateless hierarchical action sampling (re-selects a code every step).

        1. High-level policy selects a discrete code k ~ pi_h(k | s, g).
        2. Look up z_k from the codebook.
        3. Low-level decoder generates a ~ D(z_k, s).

        Kept for direct callers; the OGBench eval harness instead uses the
        H-step-committed path (`sample_actions_with_state`).
        """
        if seed is None:
            seed = self.rng

        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        if single_obs:
            observations = observations[None, ...]
            if goals is not None:
                goals = goals[None, ...]

        high_seed, low_seed = jax.random.split(seed)
        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        skill_idx = high_dist.sample(seed=high_seed)
        skills = self._codebook_table()[skill_idx]
        actions = self._decode(observations, skills, low_seed, temperature)

        if single_obs:
            actions = actions[0]
        return actions

    # ── H-step skill commitment at eval (Sec. 4.4: skill held fixed for H steps) ─

    def init_eval_state(self):
        """Per-episode state for H-step skill commitment: committed code + step count."""
        return {'skill': jnp.zeros((), jnp.int32), 'count': jnp.zeros((), jnp.int32)}

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None, temperature=1.0):
        """H-step-committed hierarchical sampling (paper's execution, Sec. 4.4).

        The high-level policy selects a code only every H steps; the code is held
        fixed (carried in `agent_state['skill']`) and the low-level decoder
        produces a_i = D(z, s_i) at every step. Returns (action, new_state); the
        eval harness threads `new_state` back in. Designed for single-env eval.
        """
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()

        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        obs_b = observations[None, ...] if single_obs else observations
        goals_b = goals[None, ...] if (single_obs and goals is not None) else goals

        high_seed, low_seed = jax.random.split(seed)
        H = int(self.config['sequence_length'])
        reselect = (agent_state['count'] % H) == 0

        high_dist = self.network.select('high_actor')(obs_b, goals_b, temperature=temperature)
        sampled = high_dist.sample(seed=high_seed)                      # [B]
        committed = jnp.broadcast_to(agent_state['skill'], sampled.shape)
        skill_idx = jnp.where(reselect, sampled, committed)            # hold for H steps

        skills = self._codebook_table()[skill_idx]                     # [B, D_z]
        actions = self._decode(obs_b, skills, low_seed, temperature)

        if single_obs:
            actions = actions[0]
            new_skill = skill_idx[0]
        else:
            new_skill = skill_idx
        new_state = {'skill': new_skill.astype(jnp.int32), 'count': agent_state['count'] + 1}
        return actions, new_state

    # ── Skill-conditioned evaluation hook (see eval_skill_policy.py) ──────────

    def skill_set(self, seed=None, num_skills=None, observations=None):
        """Candidate skills to sweep at eval time: the VQ codebook. [K, D_z]."""
        return jnp.asarray(self._codebook_table())

    @jax.jit
    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        """Act under a *fixed* codebook skill: a ~ D(z, s), bypassing the high-level actor."""
        if seed is None:
            seed = self.rng

        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        obs_b = observations[None, ...] if single_obs else observations

        skills = skills[None, ...] if skills.ndim == 1 else skills
        skills = jnp.broadcast_to(skills, (obs_b.shape[0], skills.shape[-1]))

        actions = self._decode(obs_b, skills, seed, temperature)
        if single_obs:
            actions = actions[0]
        return actions

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        if config['discrete']:
            action_dim = int(ex_actions.max() + 1)
        else:
            action_dim = ex_actions.shape[-1]

        config = dict(config)
        config['action_dim'] = action_dim
        num_skills = config['num_skills']
        skill_dim = config['skill_dim']

        ex_goals = ex_observations
        batch_size = ex_observations.shape[0]
        ex_skills = jnp.zeros((batch_size, skill_dim))
        ex_e = jnp.zeros((batch_size, skill_dim))
        ex_times = jnp.zeros((batch_size,))
        ex_noisy = jnp.zeros((batch_size, action_dim))
        ex_skill_idx = jnp.zeros((batch_size,), dtype=jnp.int32)  # example discrete code

        # Example length-T sub-trajectory window (SequenceDataset feeds the real one).
        # The snippet horizon H (=subgoal_steps) must match the actual window length
        # T=sequence_length, else a sweep that changes only one silently
        # desynchronizes the H-step return from the subgoal distance.
        assert int(config['sequence_length']) == int(config['subgoal_steps']), (
            "DDS requires sequence_length == subgoal_steps (H); got "
            f"sequence_length={config['sequence_length']}, subgoal_steps={config['subgoal_steps']}."
        )
        T = int(config['sequence_length'])
        ex_obs_seq = jnp.broadcast_to(
            ex_observations[:, None], (batch_size, T) + ex_observations.shape[1:]
        )
        ex_seq_mask = jnp.ones((batch_size, T))
        if config['discrete']:
            ex_act_seq_vec = jnp.zeros((batch_size, T, action_dim))
        else:
            ex_act_seq_vec = jnp.broadcast_to(ex_actions[:, None], (batch_size, T, action_dim))

        # Visual encoders (optional). Each network gets its own instance.
        def make_state_encoder():
            if config['encoder'] is None:
                return None
            return encoder_modules[config['encoder']]()

        encoder_def = TrajectoryEncoder(
            hidden_dim=config['encoder_dim'],
            num_layers=config['encoder_layers'],
            num_heads=config['encoder_heads'],
            skill_dim=skill_dim,
            state_encoder=make_state_encoder(),
        )
        codebook_def = VQCodebook(num_codes=num_skills, code_dim=skill_dim)

        if config['discrete']:
            decoder_def = SkillDiscreteDecoder(
                hidden_dims=config['decoder_hidden_dims'],
                action_dim=action_dim,
                state_encoder=make_state_encoder(),
            )
        else:
            decoder_def = DiffusionActionDecoder(
                hidden_dim=config['decoder_dim'],
                num_blocks=config['decoder_blocks'],
                action_dim=action_dim,
                expand=config['decoder_expand'],
                time_dim=config['time_dim'],
                state_encoder=make_state_encoder(),
            )

        # High-level goal-conditioned value, skill-indexed critic, and policy.
        value_encoder = None
        high_critic_encoder = None
        target_high_critic_encoder = None
        high_actor_encoder = None
        if config['encoder'] is not None:
            value_encoder = GCEncoder(concat_encoder=encoder_modules[config['encoder']]())
            high_critic_encoder = GCEncoder(concat_encoder=encoder_modules[config['encoder']]())
            target_high_critic_encoder = GCEncoder(concat_encoder=encoder_modules[config['encoder']]())
            high_actor_encoder = GCEncoder(concat_encoder=encoder_modules[config['encoder']]())

        # IQL implicit value V(s, g): a single (non-ensemble) head (Eq. 6).
        # Paper A.1.3-A.1.5: the value/Q/policy MLPs use ReLU activations.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=value_encoder,
            activations=nn.relu,
        )
        # Skill-indexed critic Q(s, k, g) over the K codes (ensemble of 2).
        high_critic_def = GCDiscreteCritic(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            action_dim=num_skills,
            gc_encoder=high_critic_encoder,
            activations=nn.relu,
        )
        target_high_critic_def = GCDiscreteCritic(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            action_dim=num_skills,
            gc_encoder=target_high_critic_encoder,
            activations=nn.relu,
        )
        high_actor_def = GCDiscreteActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=num_skills,
            gc_encoder=high_actor_encoder,
            activations=nn.relu,
        )

        if config['discrete']:
            decoder_args = (ex_observations, ex_skills)
        else:
            decoder_args = (ex_noisy, ex_times, ex_observations, ex_skills)

        network_info = dict(
            encoder=(encoder_def, (ex_obs_seq, ex_act_seq_vec, ex_seq_mask)),
            codebook=(codebook_def, (ex_e,)),
            decoder=(decoder_def, decoder_args),
            value=(value_def, (ex_observations, ex_goals)),
            high_critic=(high_critic_def, (ex_observations, ex_goals, ex_skill_idx)),
            target_high_critic=(target_high_critic_def, (ex_observations, ex_goals, ex_skill_idx)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        # Separate phased optimizers (paper Tables 6-7): skill lr=5e-5, IQL lr=1e-4.
        # Skill modules are HARD-frozen (zero update + zero momentum) after the gate.
        threshold = int(config['skill_pretrain_steps'])

        def label_fn(params):
            labels = {}
            for k in params:
                if k in _SKILL_MODULES:
                    labels[k] = 'skill'
                elif k == 'modules_target_high_critic':
                    labels[k] = 'frozen'  # updated only by the manual soft-update
                else:
                    labels[k] = 'iql'
            return labels

        network_tx = optax.multi_transform(
            {
                'skill': make_phased_adam(config['skill_lr'], threshold, train_in_phase1=True),
                'iql': make_phased_adam(config['iql_lr'], threshold, train_in_phase1=False),
                'frozen': optax.set_to_zero(),
            },
            label_fn,
        )

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        network.params['modules_target_high_critic'] = network.params['modules_high_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ────────────────────────────────────────────────────────────────────


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # ── Agent ───────────────────────────────────────────────────────
            agent_name='dds',                # Agent name.
            # NOTE: shared constants below are matched to
            # agents/empowerment_skill.py for a like-for-like empowerment
            # comparison (paper defaults noted per line).
            batch_size=256,                  # IQL/high-level batch size (paper Table 7).
            skill_batch_size=128,            # Skill VQ-VAE batch size (paper Table 6).
            skill_lr=3e-4,                   # Matched (paper Table 6: 5e-5).
            iql_lr=3e-4,                     # Matched (paper Table 7: 1e-4).
            # ── VQ skill model (Sec. 3.2, Eq. 14) ──────────────────────────
            num_skills=15,                   # Matched (paper default 16; swept 4-32).
            skill_dim=128,                   # Skill / codebook latent dim D_z (paper Table 6).
            commitment_beta=0.25,            # VQ commitment coefficient beta (paper Eq. 14).
            subgoal_steps=10,                # Skill horizon H (paper default 10).
            # ── Transformer sub-sequence encoder (paper Table 4) ───────────
            encoder_dim=256,                 # Transformer hidden dim (Table 4).
            encoder_layers=4,                # Transformer layers (Table 4).
            encoder_heads=8,                 # Attention heads (Table 4).
            # ── Diffusion decoder (paper Table 5) ──────────────────────────
            decoder_dim=256,                 # Denoiser hidden dim (Table 5).
            decoder_blocks=4,                # Residual denoiser blocks (Table 5).
            decoder_expand=4,                # Per-block expand factor (Table 5: 4x expand-compress).
            decoder_hidden_dims=(256, 256),  # Categorical decoder MLP (discrete-action envs only).
            diffusion_steps=5,               # Denoising steps (paper: 5 inference steps).
            time_dim=16,                     # Sinusoidal time-embedding dim (paper Table 5).
            beta_min=0.1,                    # VP schedule beta_min (paper Table 5).
            beta_max=10.0,                   # VP schedule beta_max (paper Table 5).
            # ── High-level IQL value + AWR policy (Sec. 3.3) ────────────────
            value_hidden_dims=(512, 512, 512),  # Matched (paper: 2x256).
            actor_hidden_dims=(512, 512, 512),  # Matched (paper: 2x256).
            layer_norm=True,                 # Layer normalization.
            discount=0.99,                   # Discount factor (paper Table 7).
            tau=0.005,                       # Target value soft-update / EMA (paper Table 7).
            expectile=0.7,                   # IQL expectile (paper tau_IQL = 0.7).
            high_alpha=3.0,                  # AWR temperature (paper-unspecified -> IQL/AWR default).
            # ── Two-phase, step-gated training schedule ─────────────────────
            skill_pretrain_steps=500000,     # Train skill VQ-VAE alone for this many steps (paper Table 6),
                                             # then HARD-FREEZE it and train the high level.
            # ── Misc ────────────────────────────────────────────────────────
            discrete=False,                  # Discrete action space?
            const_std=True,                  # (kept for interface symmetry)
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
            # ── Dataset: SequenceDataset feeds the real length-H window ──────
            # (observations_seq/actions_seq/seq_mask + rewards_seq/masks_seq/
            #  subgoal_observations); sequence_length = H = subgoal_steps.
            dataset_class='SequenceDataset',
            sequence_length=10,              # Window length T = skill horizon H (== subgoal_steps).
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
