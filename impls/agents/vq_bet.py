"""VQ-BeT: Behavior Generation with Latent Actions (Lee et al., ICML 2024 spotlight).

Faithful OGBench re-implementation of VQ-BeT (arXiv:2403.03181). See the
companion notes ``vq_bet_NOTES.md`` for an equation-by-equation cross-reference
to the paper.

╔══════════════════════════════════════════════════════════════════════════════╗
║ ASSUMPTIONS & DEVIATIONS FROM THE PAPER (arXiv:2403.03181)                    ║
║ Authoritative summary; read this first.                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ (A) ASSUMPTIONS WE MUST MAKE                                                  ║
║  A1. Chunk length n == act_window_size, and sequence_length == n (asserted   ║
║      in create). Real chunks a_{t:t+n} come from SequenceDataset's           ║
║      actions_seq/seq_mask. Forced by the dataset wiring. (§3.2)              ║
║  A2. Observation history h = 1. OGBench yields a single obs, so the GPT      ║
║      token sequence is [goal, obs] (length 2). The paper uses an obs window  ║
║      (h up to 100, Table 13). Forced by OGBench's batch. (§3.4)             ║
║  A3. N_q == 2 (one primary + one secondary code). The policy's two code      ║
║      heads hard-assume this; asserted in create. Paper: "N_q := 2". (§3.2)  ║
║  A4. Loss-weight / hyperparameter choices the paper leaves task-dependent:   ║
║      secondary weight β (Table 13: 0.1-0.6; default 0.5), focal γ = 2 (Lin   ║
║      et al. 2017; the paper does not pin a value), codebook size k = 16 and  ║
║      latent dim within the paper's ranges. λ_commit = 1 is exact (§3.2).    ║
║  A5. Primary-code focal weight = 1 and offset weight = 1, matching Eq. 4/7   ║
║      literally (the official repo instead tunes these to 5 / 1e3; both are   ║
║      exposed as config). (§3.3, Eq. 4, 7)                                    ║
║ (B) NOT AN EXACT MATCH                                                        ║
║  B1. EMA codebook: stored as zero-grad accumulators (cluster_size, embed_avg)║
║      overwritten post-step by the EMA rule, not as a gradient param. Hence   ║
║      Eq. 3's codebook term ‖SG[φ]−e‖² is realized BY the EMA update, not    ║
║      added to the gradient loss (kept only as a diagnostic). Equivalent to   ║
║      the paper's EMA codebook. (§3.2)                                        ║
║  B2. Single-loop, step-gated two stages inside one jitted update() (not two  ║
║      separate scripts): stage-gated optimizers hard-freeze the tokenizer in  ║
║      stage 2 (zero updates + no momentum drift). Boundary fixed by           ║
║      vqvae_pretrain_steps rather than a manual relaunch. (§3.2→§3.3)        ║
║  B3. Offset head ζ_offset(o_t) emits ONE chunk-sized offset (Eq. 6 literal); ║
║      the official code emits a per-(group,code) offset tensor and gathers.   ║
║  B4. seq_mask zeroes post-terminal padded steps in the L1 recon (Eq. 2) and  ║
║      offset (Eq. 6) — an OGBench-specific addition, not in the paper.        ║
║  B5. Hierarchical secondary prediction done as a 2-pass forward conditioned  ║
║      on the primary one-hot (faithful to Fig. 2); the official repo uses a   ║
║      single flattened code head. (§3.3)                                      ║
║  B6. No dropout in the minGPT blocks (JIT/RNG simplicity; offline BC).       ║
║  B7. Continuous actions only (assert discrete == False). (§1)               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Method summary (paper §3)
─────────────────────────
VQ-BeT tokenizes short *action chunks* into DISCRETE latent codes with a
hierarchical Residual-VQ (RVQ) autoencoder, then trains a minGPT-style causal
Transformer to predict, per step, the discrete codes (focal classification
loss) plus a continuous offset (L1). Training is OFFLINE, reward-free, and
two-stage:

  Stage 1 (paper §3.2, Eq. 1-3): train the Residual VQ-VAE (encoder φ, decoder
    ψ, N_q=2 codebooks) on action chunks with L1 reconstruction + commitment
    (λ_commit=1); the codebooks themselves are updated by EMA (paper §3.2).
  Stage 2 (paper §3.3, Eq. 4-7): freeze the tokenizer; train the Transformer
    with focal code loss (Eq. 4) over the primary + secondary codes and an
    L1 offset loss (Eq. 6). Total: L_VQ-BeT = L_code + L_offset (Eq. 7).

Terminology: here a "skill" ≈ a tokenized short action chunk, i.e. the discrete
RVQ code(s) that index a chunk of actions. The codes are discrete; there is no
reward and no environment interaction during training.

Interface mirrors ``empowerment_skill.py`` / ``gcbc.py``: a
``flax.struct.PyTreeNode`` agent with ``@classmethod create``, ``@jax.jit
update``, ``@jax.jit total_loss``, ``sample_actions(observations, goals, seed,
temperature)`` and a module-level ``get_config()``.
"""

from typing import Any, NamedTuple, Optional, Sequence

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


# ── Loss helpers ───────────────────────────────────────────────────────────────


def focal_loss(logits, targets, gamma):
    """Multiclass focal loss (Lin et al., 2017), used for code prediction (Eq. 4).

    L_focal = mean[ -(1 - p_t)^gamma * log p_t ],  p_t = softmax(logits)[target].
    gamma=0 recovers standard cross-entropy.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_p_t = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    p_t = jnp.exp(log_p_t)
    return jnp.mean(-((1.0 - p_t) ** gamma) * log_p_t)


# ── Residual VQ-VAE (paper §2.3, §3.2) ─────────────────────────────────────────


class ResidualVQ(nn.Module):
    """Residual VQ-VAE tokenizer over action chunks (paper §3.2, Eq. 1-3).

    Encoder φ maps a (flattened) action chunk a_{t:t+n} to a latent z = φ(a).
    The latent is quantized by N_q codebooks: the first codebook quantizes z
    (the *primary* code, Eq. 1), and each subsequent codebook quantizes the
    running residual (the *secondary* codes, paper §2.3). The decoder ψ
    reconstructs the chunk from the summed codebook vectors z_q(x)=Σ_i z_q^i.

    Codebooks are maintained by EMA (decay ~0.99), per residual layer, as in
    paper §3.2 ("moving averages rather than direct gradient updates"). The
    codebook is NOT a gradient parameter; instead we store the EMA accumulators
    (cluster_size N, embed_avg m) as params (they receive zero gradient — all
    codebook reads are stop-gradient'd) and overwrite them with the EMA rule
    after the optax step (see ``VQBeTAgent._ema_codebook``, the standard van den
    Oord EMA codebook). The codebook e = m / N is derived on read with Laplace
    smoothing. The encoder still commits via the straight-through estimator with
    λ_commit=1.
    """

    n_groups: int           # N_q residual layers (paper: N_q = 2)
    codebook_size: int      # codes per layer (k)
    latent_dim: int         # codebook / encoder embedding dim
    action_chunk_dim: int   # n * action_dim (flattened chunk size)
    enc_hidden_dims: Sequence[int]
    commitment_weight: float = 1.0   # λ_commit (paper §3.2: λ_commit := 1)
    ema_decay: float = 0.99          # EMA codebook decay (paper §3.2)
    ema_eps: float = 1e-5            # Laplace smoothing for the EMA codebook

    def setup(self):
        self.encoder = MLP((*self.enc_hidden_dims, self.latent_dim), activate_final=False)
        self.decoder = MLP((*self.enc_hidden_dims, self.action_chunk_dim), activate_final=False)
        # EMA accumulators (per residual layer) replace the gradient-trained
        # codebook param. e_i = embed_avg_i / cluster_size_i.
        self.cluster_size = self.param(
            'cluster_size', nn.initializers.ones, (self.n_groups, self.codebook_size)
        )
        self.embed_avg = self.param(
            'embed_avg', nn.initializers.normal(1.0),
            (self.n_groups, self.codebook_size, self.latent_dim),
        )

    def codebook(self):
        """Derive the per-layer codebook e_i = m_i / N_i (Laplace-smoothed)."""
        n = self.cluster_size.sum(axis=-1, keepdims=True)  # [N_q, 1]
        cluster = (self.cluster_size + self.ema_eps) / (n + self.codebook_size * self.ema_eps) * n
        return self.embed_avg / cluster[..., None]  # [N_q, k, d]

    def encode(self, actions):
        return self.encoder(actions)

    def decode(self, embeddings):
        return self.decoder(embeddings)

    def quantize(self, z):
        """Residual quantization of z (paper §2.3).

        Returns:
            quantized_st: straight-through quantized latent (grad → encoder).
            indices: [batch, N_q] chosen code indices per layer.
            codebook_loss: ‖sg[z] − e‖²  (diagnostic only; codebook is EMA-trained).
            commitment_loss: ‖z − sg[e]‖²  (Eq. 3 commitment term, before λ).
            residual_inputs: [batch, N_q, d] the residual fed into each layer,
                used by the EMA codebook update.
        """
        # Quantize against the EMA codebook (stop-gradient).
        cb = jax.lax.stop_gradient(self.codebook())  # [N_q, k, d]
        residual = z
        quantized = jnp.zeros_like(z)
        indices = []
        residual_inputs = []
        for i in range(self.n_groups):
            cb_i = cb[i]  # [k, d]
            residual_inputs.append(residual)
            # Nearest-neighbor lookup (Eq. 1): argmin_j ‖residual − e_j‖².
            dist = jnp.sum((residual[:, None, :] - cb_i[None, :, :]) ** 2, axis=-1)  # [B, k]
            idx = jnp.argmin(dist, axis=-1)  # [B]
            q = cb_i[idx]  # [B, d]
            quantized = quantized + q
            residual = residual - q  # residual passed to next layer (paper §2.3)
            indices.append(idx)
        indices = jnp.stack(indices, axis=-1)            # [B, N_q]
        residual_inputs = jnp.stack(residual_inputs, axis=1)  # [B, N_q, d]

        # `quantized` is built purely from the stop-gradient codebook, so it is
        # already detached; codebook_loss has zero gradient and is kept only as a
        # quantization-error diagnostic (the codebook is trained by EMA, not here).
        codebook_loss = jnp.mean(jnp.sum((jax.lax.stop_gradient(z) - quantized) ** 2, axis=-1))
        commitment_loss = jnp.mean(jnp.sum((z - jax.lax.stop_gradient(quantized)) ** 2, axis=-1))

        # Straight-through estimator: decoder sees `quantized`, gradient flows to z.
        quantized_st = z + jax.lax.stop_gradient(quantized - z)
        return quantized_st, indices, codebook_loss, commitment_loss, residual_inputs

    def decode_codes(self, codes):
        """Decode an action chunk from discrete code indices (paper Eq. 5).

        codes: [batch, N_q] integer indices. Returns ψ(Σ_i e_{codes_i}).
        """
        cb = jax.lax.stop_gradient(self.codebook())
        emb = jnp.zeros((codes.shape[0], self.latent_dim))
        for i in range(self.n_groups):
            emb = emb + cb[i][codes[:, i]]
        return self.decoder(emb)

    def __call__(self, actions):
        """Full RVQ-VAE forward for stage-1 training.

        Returns: reconstruction, indices, codebook_loss, commitment_loss,
        residual_inputs.
        """
        z = self.encode(actions)
        quantized_st, indices, cb_loss, commit_loss, residual_inputs = self.quantize(z)
        recon = self.decode(quantized_st)
        return recon, indices, cb_loss, commit_loss, residual_inputs


# ── minGPT-style Transformer (paper §3.3) ──────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """minGPT causal multi-head self-attention."""

    n_embd: int
    n_head: int

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        head_dim = C // self.n_head
        qkv = nn.Dense(3 * C)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(z):
            return z.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        att = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)  # [B, H, T, T]
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        att = jnp.where(causal_mask[None, None, :, :], att, -1e9)
        att = jax.nn.softmax(att, axis=-1)
        y = att @ v  # [B, H, T, head_dim]
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return nn.Dense(C)(y)


class GPTBlock(nn.Module):
    """minGPT transformer block: pre-LN attention + MLP residual streams."""

    n_embd: int
    n_head: int

    @nn.compact
    def __call__(self, x):
        x = x + CausalSelfAttention(self.n_embd, self.n_head)(nn.LayerNorm()(x))
        h = nn.LayerNorm()(x)
        h = nn.Dense(4 * self.n_embd)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.n_embd)(h)
        return x + h


class VQBeTPolicy(nn.Module):
    """minGPT predicting discrete codes (focal) + continuous offset (L1).

    Inputs are a short observation sequence; in OGBench we condition on
    [goal_token, observation_token] (paper §3.4 conditional formulation
    π: O^h × O^g → A^n). The Transformer feature at the observation position
    feeds three heads (paper §3.3):

      head1 (code predictor, primary): logits over k codes for the 1st codebook.
      head2 (code predictor, secondary): logits over k codes for the 2nd
        codebook, conditioned on the (one-hot) primary code — the *hierarchical
        code prediction* of Fig. 2.
      offset head ζ_offset(o_t): continuous residual of size n*action_dim (Eq. 6).

    The two code heads assume N_q == 2 (one primary, one secondary). The agent
    asserts ``vqvae_groups == 2`` in ``create`` so this hard-coded head structure
    stays consistent with the tokenizer.
    """

    n_embd: int
    n_head: int
    n_layer: int
    codebook_size: int
    action_chunk_dim: int
    head_hidden_dims: Sequence[int]
    gc_encoder: Optional[nn.Module] = None  # optional visual encoder

    def setup(self):
        self.obs_proj = nn.Dense(self.n_embd)
        self.goal_proj = nn.Dense(self.n_embd)
        # Positional embedding for the length-2 [goal, obs] token sequence.
        self.pos_emb = self.param('pos_emb', nn.initializers.normal(0.02), (2, self.n_embd))
        self.blocks = [GPTBlock(self.n_embd, self.n_head) for _ in range(self.n_layer)]
        self.ln_f = nn.LayerNorm()
        self.head1 = MLP((*self.head_hidden_dims, self.codebook_size), activate_final=False)
        self.head2 = MLP((*self.head_hidden_dims, self.codebook_size), activate_final=False)
        self.offset_head = MLP((*self.head_hidden_dims, self.action_chunk_dim), activate_final=False)

    def _encode(self, x):
        return self.gc_encoder(x, None) if self.gc_encoder is not None else x

    def __call__(self, observations, goals, primary_onehot):
        """Returns (primary_logits, secondary_logits, offset).

        primary_onehot: [batch, k] one-hot of the primary code conditioning the
            secondary head (ground-truth code during training; sampled code at
            inference). Note primary_logits and offset do NOT depend on it.
        """
        obs_tok = self.obs_proj(self._encode(observations))    # [B, n_embd]
        goal_tok = self.goal_proj(self._encode(goals))         # [B, n_embd]
        x = jnp.stack([goal_tok, obs_tok], axis=1)             # [B, 2, n_embd]
        x = x + self.pos_emb[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        feat = x[:, -1, :]  # observation position

        primary_logits = self.head1(feat)
        secondary_logits = self.head2(jnp.concatenate([feat, primary_onehot], axis=-1))
        offset = self.offset_head(feat)
        return primary_logits, secondary_logits, offset


# ── Two-stage optimizer (paper §3.2→§3.3 stage boundary) ───────────────────────


class _GatedState(NamedTuple):
    count: jnp.ndarray
    inner: Any


def _stage_gated(inner, freeze_step, active_in_stage1):
    """Wrap an optax transform so it updates params in ONE stage only.

    A single shared Adam would keep nudging the RVQ params during stage 2 from
    stage-1 momentum, so the tokenizer would never be truly frozen. This wrapper
    tracks the update count and, when inactive, emits EXACTLY-zero updates AND
    freezes the inner optimizer state (so no momentum drift). Composed via
    ``optax.multi_transform`` (below) this gives the paper's hard stage boundary:
    the RVQ/encoder/decoder receive ZERO updates in stage 2, and the policy
    receives ZERO updates in stage 1. The count starts at 1 to match
    ``TrainState.step`` (which starts at 1), so the freeze boundary lines up
    exactly with the loss-gating in ``total_loss``.
    """

    def init_fn(params):
        return _GatedState(count=jnp.asarray(1, jnp.int32), inner=inner.init(params))

    def update_fn(updates, state, params=None):
        new_updates, new_inner = inner.update(updates, state.inner, params)
        in_stage1 = state.count < freeze_step
        active = in_stage1 if active_in_stage1 else jnp.logical_not(in_stage1)
        updates_out = jax.tree_util.tree_map(
            lambda u: jnp.where(active, u, jnp.zeros_like(u)), new_updates
        )
        inner_out = jax.tree_util.tree_map(
            lambda n, o: jnp.where(active, n, o), new_inner, state.inner
        )
        return updates_out, _GatedState(count=state.count + 1, inner=inner_out)

    return optax.GradientTransformation(init_fn, update_fn)


def make_vqbet_optimizer(config):
    """Two separate, stage-gated Adam optimizers routed by ``multi_transform``.

    The RVQ params (``modules_rvq``) train ONLY in stage 1; the policy params
    (``modules_policy``) train ONLY in stage 2. This is the faithful realization
    of the paper's frozen-tokenizer-in-stage-2 requirement inside OGBench's
    single jitted update loop.
    """
    freeze_step = config['vqvae_pretrain_steps']
    rvq_tx = _stage_gated(optax.adam(config['lr']), freeze_step, active_in_stage1=True)
    policy_tx = _stage_gated(optax.adam(config['lr']), freeze_step, active_in_stage1=False)

    def label_params(params):
        flat = {
            k: jax.tree_util.tree_map(lambda _: ('rvq' if 'rvq' in k else 'policy'), v)
            for k, v in params.items()
        }
        return type(params)(flat) if isinstance(params, flax.core.FrozenDict) else flat

    return optax.multi_transform({'rvq': rvq_tx, 'policy': policy_tx}, label_params)


# ── Agent ──────────────────────────────────────────────────────────────────────


class VQBeTAgent(flax.struct.PyTreeNode):
    """VQ-BeT offline behavior-generation agent (Lee et al., 2024).

    Network layout (ModuleDict): {'rvq': ResidualVQ, 'policy': VQBeTPolicy}.
    Two-stage training is realized inside a single OGBench update loop by gating
    the loss on the optimizer step (see ``total_loss``) AND by stage-gating two
    separate optimizers so the tokenizer is hard-frozen in stage 2.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _action_chunk(self, batch):
        """Return (flattened action chunk [B, n*action_dim], seq_mask [B, n]).

        Uses the REAL action chunk a_{t:t+n} from ``SequenceDataset``
        (``actions_seq`` is [B, T, action_dim] with T = n = ``act_window_size``).
        ``seq_mask`` is 1.0 for in-trajectory steps and 0.0 for steps padded past
        the terminal, and is applied to the L1 reconstruction (Eq. 2) and offset
        (Eq. 6) losses.
        """
        actions_seq = batch['actions_seq']  # [B, T, action_dim]
        flat = actions_seq.reshape(actions_seq.shape[0], -1)  # [B, T*action_dim]
        return flat, batch['seq_mask']  # mask: [B, T]

    def _masked_chunk_l1(self, target_flat, pred_flat, mask):
        """Mean L1 over the chunk, masking padded (post-terminal) steps.

        target_flat / pred_flat: [B, T*action_dim]; mask: [B, T].
        """
        B, T = mask.shape
        diff = jnp.abs(target_flat - pred_flat).reshape(B, T, -1)  # [B, T, action_dim]
        act_dim = diff.shape[-1]
        num = jnp.sum(diff * mask[:, :, None])
        den = jnp.sum(mask) * act_dim + 1e-8
        return num / den

    def _rvq_decode_codes(self, codes):
        """Decode action chunk from code indices using the FROZEN tokenizer."""
        rvq = self.network.model_def.modules['rvq']
        params = jax.lax.stop_gradient(self.network.params['modules_rvq'])
        return rvq.apply({'params': params}, codes, method=rvq.decode_codes)

    # ── Losses ──────────────────────────────────────────────────────────────────

    def rvq_loss(self, batch, grad_params):
        """Stage-1 Residual VQ-VAE loss (paper §3.2, Eq. 2-3)."""
        actions, mask = self._action_chunk(batch)
        recon, indices, cb_loss, commit_loss, _ = self.network.select('rvq')(actions, params=grad_params)

        # Masked L1 reconstruction so padded steps don't count (Eq. 2).
        recon_loss = self._masked_chunk_l1(actions, recon, mask)
        # EMA-trained codebook → only recon + commitment in the gradient loss
        # (the codebook term of Eq. 3 is handled by the EMA update).
        loss = recon_loss + self.config['commitment_weight'] * commit_loss

        metrics = {
            'rvq_loss': loss,
            'recon_loss': recon_loss,
            'codebook_loss': cb_loss,
            'commitment_loss': commit_loss,
            'code_usage_primary': jnp.sum(
                jnp.bincount(indices[:, 0], length=self.config['vqvae_n_embed']) > 0
            ),
        }
        return loss, metrics

    def transformer_loss(self, batch, grad_params):
        """Stage-2 Transformer loss: focal code loss + offset L1 (Eq. 4-7)."""
        actions, mask = self._action_chunk(batch)
        observations = batch['observations']
        goals = batch['actor_goals']

        # Tokenize ground-truth actions with the FROZEN tokenizer (params=None →
        # stored params, no gradient). Targets are detached labels (paper §3.3).
        _, target_codes, _, _, _ = self.network.select('rvq')(actions, params=None)
        primary_target = target_codes[:, 0]
        secondary_target = target_codes[:, 1]
        primary_onehot = jax.nn.one_hot(primary_target, self.config['vqvae_n_embed'])

        primary_logits, secondary_logits, offset = self.network.select('policy')(
            observations, goals, primary_onehot, params=grad_params
        )

        # Code loss (Eq. 4): focal on primary + β·focal on secondary.
        gamma = self.config['focal_gamma']
        loss_primary = focal_loss(primary_logits, primary_target, gamma)
        loss_secondary = focal_loss(secondary_logits, secondary_target, gamma)
        code_loss = (
            self.config['primary_code_weight'] * loss_primary
            + self.config['secondary_code_weight'] * loss_secondary
        )

        # Offset loss (Eq. 6): L1 between the action chunk and (decoded predicted
        # codes + offset). ⌊a⌋ uses the predicted (argmax) codes through the
        # frozen decoder ψ (Eq. 5); argmax is non-differentiable so the code
        # heads receive gradient only from the focal loss, the offset head only
        # from this L1 term — matching the paper's separate heads/objectives.
        pred_primary = jnp.argmax(primary_logits, axis=-1)
        pred_secondary = jnp.argmax(secondary_logits, axis=-1)
        pred_codes = jnp.stack([pred_primary, pred_secondary], axis=-1)
        floor_a = self._rvq_decode_codes(pred_codes)
        # Masked offset L1 so padded steps don't count (Eq. 6).
        offset_loss = self._masked_chunk_l1(actions, floor_a + offset, mask)

        loss = code_loss + self.config['offset_loss_weight'] * offset_loss  # Eq. 7

        metrics = {
            'transformer_loss': loss,
            'code_loss': code_loss,
            'focal_primary': loss_primary,
            'focal_secondary': loss_secondary,
            'offset_loss': offset_loss,
            'primary_acc': jnp.mean((pred_primary == primary_target).astype(jnp.float32)),
            'secondary_acc': jnp.mean((pred_secondary == secondary_target).astype(jnp.float32)),
        }
        return loss, metrics

    # ── Training ────────────────────────────────────────────────────────────────

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Two-stage gated objective.

        Stage 1 (step < vqvae_pretrain_steps): optimize only L_RVQ.
        Stage 2 (step ≥ vqvae_pretrain_steps): optimize only L_VQ-BeT, with the
        tokenizer frozen (its targets/decoder are stop-gradient'd, L_RVQ is masked
        out, AND the RVQ optimizer is stage-gated so its params get ZERO updates).
        See NOTES for the exact two-stage emulation.
        """
        info = {}
        rvq_loss, rvq_info = self.rvq_loss(batch, grad_params)
        info.update({f'rvq/{k}': v for k, v in rvq_info.items()})

        tf_loss, tf_info = self.transformer_loss(batch, grad_params)
        info.update({f'transformer/{k}': v for k, v in tf_info.items()})

        stage1 = (self.network.step < self.config['vqvae_pretrain_steps']).astype(jnp.float32)
        info['stage1_active'] = stage1
        total = stage1 * rvq_loss + (1.0 - stage1) * tf_loss
        info['total_loss'] = total
        return total, info

    def _ema_codebook(self, network, batch, stage1):
        """EMA update of the per-layer RVQ codebooks (paper §3.2).

        Standard van den Oord EMA codebook. For each residual layer i:
            N_i ← β N_i + (1−β) n_i
            m_i ← β m_i + (1−β) Σ_{j: code=i} residual_i(j)
        where n_i counts assignments and residual_i is the input fed to layer i.
        Codebook e_i = m_i / N_i is derived on read. Runs ONLY during stage 1
        (stage1 gate), so the tokenizer is fully frozen in stage 2.
        """
        beta = self.config['vqvae_ema_decay']
        k = self.config['vqvae_n_embed']

        actions, _ = self._action_chunk(batch)
        # Forward with the freshly-updated params (no gradient needed).
        _, indices, _, _, residual_inputs = network.select('rvq')(actions)
        indices = jax.lax.stop_gradient(indices)
        residual_inputs = jax.lax.stop_gradient(residual_inputs)  # [B, N_q, d]

        onehot = jax.nn.one_hot(indices, k)              # [B, N_q, k]
        n_i = onehot.sum(0)                              # [N_q, k]
        dw = jnp.einsum('bgk,bgd->gkd', onehot, residual_inputs)  # [N_q, k, d]

        rvq_params = network.params['modules_rvq']
        new_cluster = beta * rvq_params['cluster_size'] + (1.0 - beta) * n_i
        new_embed_avg = beta * rvq_params['embed_avg'] + (1.0 - beta) * dw

        # Only update during stage 1; stage 2 leaves the tokenizer untouched.
        new_cluster = jnp.where(stage1, new_cluster, rvq_params['cluster_size'])
        new_embed_avg = jnp.where(stage1, new_embed_avg, rvq_params['embed_avg'])

        new_rvq = {**rvq_params, 'cluster_size': new_cluster, 'embed_avg': new_embed_avg}
        new_params = {**network.params, 'modules_rvq': new_rvq}
        return network.replace(params=new_params)

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        # Stage flag from the pre-update step (matches the loss/optimizer gating).
        stage1 = self.network.step < self.config['vqvae_pretrain_steps']
        new_network, info = self.network.apply_loss_fn(
            loss_fn=lambda p: self.total_loss(batch, p, rng=rng)
        )
        # EMA codebook update *after* the gradient step, stage-1 only (no gradient).
        new_network = self._ema_codebook(new_network, batch, stage1)
        return self.replace(network=new_network, rng=new_rng), info

    # ── Evaluation ────────────────────────────────────────────────────────────────

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample an action via hierarchical code prediction + offset (paper §3.3).

        Predict the primary code, then the secondary code conditioned on it,
        decode Σ_i e_{code_i} through ψ, add the offset, and return the first
        action of the chunk.
        """
        if seed is None:
            seed = self.rng

        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        if single_obs:
            observations = observations[None, ...]
            if goals is not None:
                goals = goals[None, ...]
        if goals is None:
            goals = observations  # unconditional fallback (self-goal token)

        batch_size = observations.shape[0]
        k = self.config['vqvae_n_embed']
        temp = jnp.maximum(temperature, 1e-6)

        seed, s1, s2 = jax.random.split(seed, 3)

        # Pass 1: primary code + offset (independent of the conditioning one-hot).
        dummy_onehot = jnp.zeros((batch_size, k))
        primary_logits, _, offset = self.network.select('policy')(observations, goals, dummy_onehot)
        primary = distrax.Categorical(logits=primary_logits / temp).sample(seed=s1)

        # Pass 2: secondary code conditioned on the sampled primary code.
        primary_onehot = jax.nn.one_hot(primary, k)
        _, secondary_logits, _ = self.network.select('policy')(observations, goals, primary_onehot)
        secondary = distrax.Categorical(logits=secondary_logits / temp).sample(seed=s2)

        codes = jnp.stack([primary, secondary], axis=-1)
        floor_a = self._rvq_decode_codes(codes)  # [B, n*action_dim]
        chunk = floor_a + offset                 # add continuous offset (Eq. 6)

        # Take the first action of the chunk: [B, n, action_dim] → [B, action_dim].
        chunk = chunk.reshape(batch_size, self.config['act_window_size'], -1)
        actions = chunk[:, 0, :]
        actions = jnp.clip(actions, -1, 1)

        if single_obs:
            actions = actions[0]
        return actions

    # ── Constructor ────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # VQ-BeT tokenizes CONTINUOUS actions; discrete action spaces unsupported.
        assert not config['discrete'], 'VQ-BeT operates on continuous actions only.'

        # The policy's two code heads (primary + one secondary) hard-assume
        # N_q == 2; assert it with a clear message.
        assert config['vqvae_groups'] == 2, (
            f"VQ-BeT's policy code heads are specialized to N_q (vqvae_groups) == 2 "
            f"(one primary + one secondary code); got vqvae_groups={config['vqvae_groups']}. "
            f'Either set vqvae_groups=2 or generalize VQBeTPolicy to emit N_q hierarchical heads.'
        )
        # The RVQ chunk length must match the dataset window.
        assert config['sequence_length'] == config['act_window_size'], (
            f"SequenceDataset window (sequence_length={config['sequence_length']}) must equal the "
            f"action chunk length (act_window_size={config['act_window_size']})."
        )

        action_dim = ex_actions.shape[-1]
        # Chunk dim = n * action_dim (n = act_window_size). Real chunks come from
        # SequenceDataset's `actions_seq`; the example below is just a shape
        # placeholder for parameter init.
        act_window_size = config['act_window_size']
        action_chunk_dim = act_window_size * action_dim
        batch_size = ex_observations.shape[0]
        ex_actions_chunk = jnp.zeros((batch_size, action_chunk_dim))
        ex_goals = ex_observations

        # Optional visual encoder for the Transformer tokens.
        encoders = {}
        if config.get('encoder') is not None:
            enc = encoder_modules[config['encoder']]
            encoders['policy'] = GCEncoder(state_encoder=enc())

        rvq_def = ResidualVQ(
            n_groups=config['vqvae_groups'],
            codebook_size=config['vqvae_n_embed'],
            latent_dim=config['vqvae_latent_dim'],
            action_chunk_dim=action_chunk_dim,
            enc_hidden_dims=config['vqvae_hidden_dims'],
            commitment_weight=config['commitment_weight'],
            ema_decay=config['vqvae_ema_decay'],
        )
        policy_def = VQBeTPolicy(
            n_embd=config['gpt_n_embd'],
            n_head=config['gpt_n_head'],
            n_layer=config['gpt_n_layer'],
            codebook_size=config['vqvae_n_embed'],
            action_chunk_dim=action_chunk_dim,
            head_hidden_dims=config['head_hidden_dims'],
            gc_encoder=encoders.get('policy'),
        )

        ex_primary_onehot = jnp.zeros((batch_size, config['vqvae_n_embed']))
        network_def = ModuleDict(dict(rvq=rvq_def, policy=policy_def))
        network_params = network_def.init(
            init_rng,
            rvq=(ex_actions_chunk,),
            policy=(ex_observations, ex_goals, ex_primary_onehot),
        )['params']

        # Stage-gated two-optimizer setup: RVQ trains in stage 1 only, policy in
        # stage 2 only (instead of one shared Adam).
        network = TrainState.create(
            network_def, network_params, tx=make_vqbet_optimizer(config)
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ──────────────────────────────────────────────────────────────────────


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='vq_bet',
        lr=1e-4,                         # Adam LR (official VQ-BeT uses ~1e-4).
        batch_size=1024,                 # OGBench pipeline default (official uses smaller).
        discrete=False,                  # VQ-BeT is continuous-action only.

        # ── Residual VQ-VAE (paper §3.2) ────────────────────────────────────────
        vqvae_groups=2,                  # N_q residual layers (paper: N_q := 2).
        vqvae_n_embed=16,                # codes per codebook k (paper: 8-16).
        vqvae_latent_dim=128,            # encoder/codebook embedding dim.
        vqvae_hidden_dims=(128, 128),    # encoder/decoder MLP hidden dims.
        commitment_weight=1.0,           # λ_commit (paper §3.2: := 1).
        vqvae_ema_decay=0.99,            # EMA codebook decay (paper §3.2).
        act_window_size=5,               # action chunk length n (paper: 1-6; real chunks via SequenceDataset).

        # ── minGPT Transformer (paper §3.3) ─────────────────────────────────────
        gpt_n_layer=6,                   # transformer blocks (official minGPT).
        gpt_n_head=6,
        gpt_n_embd=120,
        head_hidden_dims=(256,),         # code/offset head MLP hidden dims.

        # ── Stage-2 objective (Eq. 4, 6, 7) ─────────────────────────────────────
        focal_gamma=2.0,                 # focal loss γ (Lin et al., 2017; paper does not pin a value).
        # Defaults follow the paper equations LITERALLY: Eq. 4 puts weight 1 on the
        # primary focal term and β on the secondary; Eq. 7 puts weight 1 on L_offset.
        # The official repo instead tunes these (primary 5, offset 1e3) for
        # normalized actions; both remain configurable here.
        primary_code_weight=1.0,         # weight of primary focal loss (Eq. 4: := 1; official 5).
        secondary_code_weight=0.5,       # β for secondary focal loss (Eq. 4; paper Table 13: 0.1-0.6, task-dependent).
        offset_loss_weight=1.0,          # weight of offset L1 (Eq. 7: := 1; official 1e3).

        # ── Two-stage gating ────────────────────────────────────────────────────
        vqvae_pretrain_steps=100000,     # stage-1 length; afterwards train Transformer.

        # ── Dataset (goal-conditioned BC sampling + action-chunk windows) ───────
        encoder=ml_collections.config_dict.placeholder(str),
        # SequenceDataset supplies actions_seq/seq_mask (real multi-step chunks).
        dataset_class='SequenceDataset',
        sequence_length=5,               # MUST equal act_window_size (asserted in create).
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
