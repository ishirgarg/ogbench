"""Skill Decision Transformer (Skill-DT).

Faithful re-implementation of "Skill Decision Transformer" (Shyam et al.,
arXiv:2301.13573) as an OGBench offline-RL agent.

================================================================================
ASSUMPTIONS & DEVIATIONS FROM THE PAPER (arXiv:2301.13573)
================================================================================
Read this first. The body below this header and `skill_dt_NOTES.md` give the
full component->paper mapping. Everything that is NOT a byte-exact match to the
paper is listed here with a one-line reason + section reference.

(a) Assumptions that MUST be made for the OGBench interface
  1. Future-skill histogram is WINDOW-BOUNDED. The paper's prior is
     Z_t = normalize(sum_{t'>=t}^{T} onehot(z_{t'})) to the trajectory end
     (Sec. 4.1). Training uses fixed length-K windows, so the tail past the
     window is unreachable; the sum is truncated to the masked window. Reason:
     fixed-length minibatches cannot see the true trajectory end.
  2. MEMORYLESS, GOAL-CONDITIONED inference. OGBench's eval loop calls
     `sample_actions` per step with only the current observation (no history)
     and is goal-conditioned, whereas the paper rolls the Transformer over a
     full K-window and selects the best skill (Sec. 4). So eval runs a length-1
     context and picks the discrete skill by a deterministic goal hash (uniform
     if no goal). Reason: stateless + goal-conditioned eval API.
  3. DISCRETE-ACTION CROSS-ENTROPY vs continuous MSE. The paper's action term is
     MSE for continuous control (Eq. for L_{theta,phi}, Sec. 4.3). OGBench
     discrete-action envs use cross-entropy on a categorical head (MSE on action
     indices is meaningless). Continuous envs use the paper's MSE exactly.

(b) Other deviations (not exact matches)
  4. EMA codebook, not gradient codebook. Paper writes VQLOSS = MSE(z, zhat)
     (Eq. 1) "optimized using an exponential moving average" (Sec. 4.2). The
     codebook is held as EMA accumulators (zero-gradient, decay 0.99), so the
     gradient term reduces to the commitment loss beta*||zhat - sg[z]||^2 with a
     straight-through estimator. Decay/beta are unspecified in the paper.
  5. Unspecified sizes set by convention: code_dim (= embed_dim, because the
     skill embedding is fed as a token with no projection, Sec. 4.2),
     vq_hidden_dims, and vq_beta=0.25 (van den Oord VQ-VAE default). Paper does
     not state these (Sec. A.2).
  6. Continuous actions are wrapped in a constant-std MultivariateNormalDiag for
     the `sample_actions` interface; the loss is pure MSE on the mean, so the
     wrapper does not change the objective (Sec. 4.3).
  7. Action loss is divided by the number of VALID (masked) steps rather than a
     literal 1/K, so padded steps past a trajectory boundary do not dilute the
     mean. Equivalent to 1/K on full windows (Sec. 4.3).
  8. Optimizer is Adam (paper does not state the optimizer; Sec. A.2). All other
     hyperparameters match Table 5 exactly: K=20, 4 layers, 4 heads, dim 256,
     lr 1e-4, grad-norm clip 0.25, batch 256, dropout 0.0.

EXACT matches to the paper (verified): argmin codebook assignment
z=argmin_n||zhat-z_n||^2; straight-through estimator; future-skill histogram
prior; token order per step [Z_t, z_t, s_t] with action read from the s_t
position; skill embedding NOT tokenized AND no timestep embedding added to it
(timestep embeddings go only on state + histogram tokens, Sec. 4.2); causal
4-layer/4-head/256-dim Transformer; (1/K)-style action reconstruction + VQLOSS.
================================================================================

Method (paper section references in skill_dt_NOTES.md):
  * A VQ-VAE skill encoder maps each state to a continuous embedding zhat_t and
    quantizes it to the nearest codebook entry  z_t = z_{argmin_n ||zhat_t - z_n||^2}
    (Sec. 4.1-4.2). Skills are therefore DISCRETE (one index per codebook entry).
  * A "skill prior" Z_t = normalize(sum_{t'>=t} one_hot(z_{t'})) is the future
    skill histogram (Sec. 4.1), fed as a conditioning token. With fixed-length
    training windows the sum is taken over the masked FUTURE WITHIN THE WINDOW
    (a window-bounded approximation of the paper's trajectory-end sum).
  * A causal Transformer policy autoregressively predicts actions from the token
    sequence [Z_t, z_t, s_t] over a context window of length K (Sec. 4.2).
  * Objective (Sec. 4.3, Alg. 1):
        L = (1/K) sum_t (a_t - ahat_t)^2  +  VQLOSS(z, zhat)
    i.e. MSE action reconstruction + VQ-VAE codebook/commitment regularization.
    The codebook itself is maintained by an EMA update (decay 0.99), so the
    gradient VQLOSS reduces to the commitment term.

Training is purely OFFLINE and REWARD-FREE: no rewards, returns, or goals enter
the objective. Skills are discovered from the data via the VQ codebook.

The real length-K context is supplied by the shared `SequenceDataset`
(`utils/datasets.py`, registered in `main.py`): it augments the GCDataset batch
with `observations_seq`/`actions_seq`/`seq_mask` windows of length
`sequence_length = K`. Select it via `dataset_class='SequenceDataset'`.
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


# ── Sequence helpers ──────────────────────────────────────────────────────────


def future_skill_histogram(onehot, mask):
    """Normalized future-skill histogram Z_t (paper Sec. 4.1).

    Z_t = normalize( sum_{t'>=t, valid} one_hot(z_{t'}) ), the distribution over
    skills used from step t to the end of the (masked) context window. This is a
    window-bounded approximation of the paper's sum to the trajectory end: with
    fixed-length windows the true tail is unreachable, so the future is truncated
    to the window and padded steps are masked out.

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
    """Multi-head causal self-attention (GPT-style), JIT-friendly."""

    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        B, L, D = x.shape
        H = self.num_heads
        hd = D // H

        qkv = nn.Dense(3 * D)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(t):
            return t.reshape(B, L, H, hd).transpose(0, 2, 1, 3)  # [B, H, L, hd]

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        att = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(hd)
        causal = jnp.tril(jnp.ones((L, L), dtype=bool))
        att = jnp.where(causal, att, -jnp.inf)
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
    def __call__(self, x):
        x = x + CausalSelfAttention(self.embed_dim, self.num_heads)(nn.LayerNorm()(x))
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
        # EMA accumulators: cluster_size N_i and embed_avg m_i. Codebook is
        # derived on read as e_i = m_i / N_i (Laplace-smoothed).
        self.cluster_size = self.param('cluster_size', nn.initializers.ones, (self.num_skills,))
        self.embed_avg = self.param(
            'embed_avg', nn.initializers.normal(1.0), (self.num_skills, self.code_dim)
        )

    def codebook(self):
        n = self.cluster_size.sum()
        cluster = (self.cluster_size + self.eps) / (n + self.num_skills * self.eps) * n
        return self.embed_avg / cluster[:, None]  # [N, code_dim]

    def encode(self, observations):
        x = observations
        if self.gc_encoder is not None:
            x = jax.vmap(lambda s: self.gc_encoder(s, None), in_axes=1, out_axes=1)(x)
        return self.encoder_mlp(x)  # [B, T, code]

    def __call__(self, observations):
        z_e = self.encode(observations)
        cb = jax.lax.stop_gradient(self.codebook())  # frozen w.r.t. gradients (EMA only)

        # z = argmin_n ||zhat - z_n||^2  (paper Eq. quantization, Sec. 4.2).
        dist = ((z_e[..., None, :] - cb) ** 2).sum(-1)  # [B, T, N]
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
    context_len: int = 20
    discrete: bool = False
    const_std: bool = True
    gc_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations, skill_embeddings, skill_hist, temperature=1.0):
        B, T = observations.shape[0], observations.shape[1]

        s_in = observations
        if self.gc_encoder is not None:
            s_in = jax.vmap(lambda s: self.gc_encoder(s, None), in_axes=1, out_axes=1)(s_in)

        # Per-token embeddings.
        s_tok = nn.Dense(self.embed_dim)(s_in)
        z_tok = skill_embeddings
        h_tok = nn.Dense(self.embed_dim)(skill_hist)

        # Learned per-timestep positional (timestep) embedding (DT-style). The
        # paper adds timestep embeddings to the state and skill-distribution
        # (histogram) tokens, but NOT to the raw skill embedding token z_t
        # ("we don't tokenize our skill embeddings ... we want to ensure that we
        # don't lose important skill embedding information", Sec. 4.2). So z_tok
        # is fed directly with no projection and no timestep embedding.
        pos_emb = self.param('pos_emb', nn.initializers.normal(0.02),
                             (self.context_len, self.embed_dim))[:T]
        s_tok = s_tok + pos_emb
        h_tok = h_tok + pos_emb

        # Interleave into [Z_0, z_0, s_0, Z_1, z_1, s_1, ...] -> [B, 3T, D].
        tokens = jnp.stack([h_tok, z_tok, s_tok], axis=2).reshape(B, T * 3, self.embed_dim)

        x = nn.LayerNorm()(tokens)
        for _ in range(self.num_layers):
            x = TransformerBlock(self.embed_dim, self.num_heads)(x)
        x = nn.LayerNorm()(x)

        # Action prediction from the s_t positions (index 3t + 2).
        s_hidden = x[:, 2::3, :]  # [B, T, D]

        if self.discrete:
            logits = nn.Dense(self.action_dim)(s_hidden)
            return distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))
        else:
            means = nn.Dense(self.action_dim)(s_hidden)
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
        """VQ-VAE commitment loss beta * ||zhat - sg[z]||^2 (the codebook term is
        handled by the EMA update, not by gradients)."""
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
        """(1/K) sum_t (a_t - ahat_t)^2 for continuous; CE for discrete (NOTES)."""
        if self.config['discrete']:
            # OGBench discrete envs: paper's MSE is replaced by cross-entropy.
            nll = -dist.log_prob(actions)  # [B, T]
            loss = (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            metrics = {'action_loss': loss, 'action_nll': loss}
        else:
            pred = dist.mode()  # deterministic mean
            se = ((pred - actions) ** 2).sum(-1)  # [B, T]
            loss = (se * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            metrics = {'action_loss': loss, 'action_mse': loss}
        return loss, metrics

    def _seq_batch(self, batch):
        """Pull the real length-K window from the SequenceDataset batch.

        Falls back to a length-1 window if the seq keys are absent.
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
                mask = mask[:, None]
        return observations, actions, mask

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        observations, actions, mask = self._seq_batch(batch)
        B, T = observations.shape[0], observations.shape[1]

        # 1) VQ-VAE skill encoding + quantization.
        z_e, z_q, indices = self.network.select('vq')(observations, params=grad_params)

        # Straight-through estimator: gradients of the action loss flow to the
        # encoder, while the quantized value is used in the forward pass.
        z_q_st = z_e + jax.lax.stop_gradient(z_q - z_e)

        # 2) Future-skill histogram Z_t (the "skill prior" conditioning token).
        onehot = jax.nn.one_hot(indices, self.config['num_skills'])  # [B, T, N]
        skill_hist = jax.lax.stop_gradient(future_skill_histogram(onehot, mask))

        # 3) Skill-conditioned causal Transformer policy.
        dist = self.network.select('policy')(
            observations, z_q_st, skill_hist, params=grad_params
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

        total = act_loss + self.config['vq_coef'] * vq_loss
        info['total_loss'] = total
        return total, info

    def _ema_codebook(self, network, batch):
        beta = self.config['vq_decay']
        N = self.config['num_skills']
        code_dim = self.config['code_dim']

        observations, _, mask = self._seq_batch(batch)

        # Forward the encoder with the freshly-updated params (no gradient).
        z_e, _, indices = network.select('vq')(observations)
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

    # ── Evaluation ────────────────────────────────────────────────────────────

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions for OGBench's memoryless, goal-conditioned evaluation.

        OGBench's evaluation loop calls this per environment step with only the
        current observation (no maintained history), so the causal Transformer
        is run on a length-1 context [Z_t, z_t, s_t] (documented inference
        simplification; paper rolls out over the full K-window). The skill index
        is selected deterministically from the goal (mirroring the discrete
        skill-conditioned convention in empowerment_skill.py), or sampled
        uniformly when no goal is provided.
        """
        if seed is None:
            seed = self.rng

        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        single_obs = observations.ndim == single_obs_ndim
        if single_obs:
            observations = observations[None, ...]
            if goals is not None:
                goals = goals[None, ...]

        batch_size = observations.shape[0]
        N = self.config['num_skills']

        # Map goal -> discrete skill (deterministic hash), else sample uniformly.
        if goals is not None:
            goal_flat = goals.reshape(batch_size, -1).astype(jnp.int32)
            goal_hash = jnp.sum(goal_flat, axis=-1)
            skills = (jnp.abs(goal_hash) % N).astype(jnp.int32)
        else:
            skills = jax.random.randint(seed, (batch_size,), 0, N)

        # Derive the codebook from the EMA accumulators e_i = m_i / N_i
        # (Laplace-smoothed), matching SkillVQ.codebook().
        vq_params = self.network.params['modules_vq']
        cluster_size = vq_params['cluster_size']                    # [N]
        embed_avg = vq_params['embed_avg']                         # [N, code]
        eps = 1e-5
        n = cluster_size.sum()
        cluster = (cluster_size + eps) / (n + N * eps) * n
        codebook = embed_avg / cluster[:, None]                    # [N, code]
        z_q = codebook[skills][:, None, :]                         # [B, 1, code]
        skill_hist = jax.nn.one_hot(skills, N)[:, None, :]         # [B, 1, N]

        obs_seq = observations[:, None, ...]                       # [B, 1, obs]
        dist = self.network.select('policy')(
            obs_seq, z_q, skill_hist, temperature=temperature
        )
        actions = dist.sample(seed=seed)[:, 0]                     # [B, ...]
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)

        if single_obs:
            actions = actions[0]
        return actions

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
            ex_act = ex_actions[:, None] if ex_actions.ndim == 1 else ex_actions
        else:
            action_dim = ex_actions.shape[-1]
            ex_act = ex_actions[:, None, :] if ex_actions.ndim == 2 else ex_actions
        del ex_act  # only used to derive action_dim

        num_skills = config['num_skills']
        code_dim = config['code_dim']

        # The skill token is fed to the Transformer without a Dense projection,
        # so the skill embedding dimension must equal the Transformer width.
        assert code_dim == config['embed_dim'], (
            f'Skill-DT feeds the skill embedding directly as a token, so '
            f"code_dim ({code_dim}) must equal embed_dim ({config['embed_dim']})."
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
            context_len=config['context_len'],
            discrete=config['discrete'],
            const_std=config['const_std'],
            gc_encoder=policy_encoder,
        )

        T = ex_obs.shape[1]
        ex_skill_emb = jnp.zeros((ex_obs.shape[0], T, code_dim))
        ex_hist = jnp.zeros((ex_obs.shape[0], T, num_skills))

        network_def = ModuleDict(dict(vq=vq_def, policy=policy_def))
        network_params = network_def.init(
            init_rng,
            vq=(ex_obs,),
            policy=(ex_obs, ex_skill_emb, ex_hist),
        )['params']

        # Adam + global-norm gradient clipping (paper grad-norm 0.25, Sec. A.2).
        tx = optax.chain(
            optax.clip_by_global_norm(config['grad_clip']),
            optax.adam(config['lr']),
        )
        network = TrainState.create(network_def, network_params, tx=tx)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ── Config ────────────────────────────────────────────────────────────────────


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='skill_dt',
        # Optimization (paper Sec. A.2, Table 5).
        lr=1e-4,
        batch_size=256,
        grad_clip=0.25,
        # VQ-VAE skill codebook.
        num_skills=16,            # codebook size (paper sweeps 5-64).
        code_dim=256,             # skill embedding dimension (MUST equal embed_dim).
        vq_hidden_dims=(256, 256),
        vq_beta=0.25,             # commitment coefficient (van den Oord VQ-VAE).
        vq_coef=1.0,              # weight of VQLOSS in the total objective.
        vq_decay=0.99,            # EMA codebook decay (paper EMA variant).
        layer_norm=True,
        # Causal Transformer policy (paper Table 5).
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        context_len=20,
        const_std=True,
        discrete=False,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class='SequenceDataset',
        sequence_length=20,       # K: matches context_len; SequenceDataset reads this.
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
