# Skill Decision Transformer (Skill-DT) — Implementation Notes

Re-implementation of **"Skill Decision Transformer"** (Shyam Sudhakaran & Sebastian
Risi, arXiv:2301.13573) as the OGBench agent `agents/skill_dt.py`
(`agent_name='skill_dt'`). Training is **purely offline** and **reward-free**;
skills are **discrete** (a VQ-VAE codebook). Section/equation references below are
to the arXiv paper. A reviewer with the PDF can cross-check each component.

The official code repo (github.com/shyamsn97/skill-dt) was unpublished at the time
of writing ("Code to be published soon!"), so this implementation follows the
paper text. Where the paper is compact, the standard VQ-VAE / Decision-Transformer
mechanism it references is used and flagged below.

---

## Component → paper mapping

### 1. VQ-VAE skill encoder + discrete codebook  (Sec. 4.1–4.2)
`SkillVQ` (module key `vq`).
- Encoder MLP maps each state `s_t` to a continuous embedding `ẑ_t ∈ R^code_dim`.
- Codebook `{z_n}_{n=1..N}` is a learned parameter `codebook : [N, code_dim]`.
- Quantization (paper Sec. 4.2): `z_t = z_{argmin_n ||ẑ_t − z_n||²}`. Implemented
  as `indices = argmin_n ||ẑ − z_n||²`, `z_q = codebook[indices]`.
- **Discrete-skill mechanism**: the skill is the integer codebook index. There are
  exactly `N = num_skills` skills (paper sweeps **5–64**; default **16**).

### 2. VQ loss + EMA codebook  (Sec. 4.2, Eq. 1)
`SkillDTAgent.vq_loss` + `SkillVQ` (EMA accumulators) + `SkillDTAgent._ema_codebook`.
- Paper states `VQLOSS(z, ẑ) = MSE(z, ẑ)` (Eq. 1), references VQ-VAE, and uses an
  **EMA codebook**. We now implement the EMA variant (decay `vq_decay=0.99`),
  the standard van den Oord EMA VQ-VAE codebook.
- The codebook is no longer a gradient-trained `param`. `SkillVQ` holds EMA
  accumulators `cluster_size` (Nᵢ) and `embed_avg` (mᵢ) as params and derives the
  codebook on read as `eᵢ = mᵢ / Nᵢ` (Laplace-smoothed). All codebook reads are
  `stop_gradient`'d ⇒ the accumulators get **zero gradient**.
- After every optax step, `_ema_codebook` overwrites the accumulators:
  `Nᵢ ← β Nᵢ + (1−β) nᵢ`, `mᵢ ← β mᵢ + (1−β) Σ_{j: q(ẑⱼ)=i} ẑⱼ`, with padded
  steps masked out (window flattened over `(B, T)`).
- Consequently the **gradient** VQLOSS reduces to the commitment term only:
  `commit = β·||ẑ − sg[z]||²` (the codebook term is handled by EMA, not grads).
- Straight-through estimator (`z_q_st = ẑ + sg[z_q − ẑ]`) lets the action-MSE
  gradient reach the encoder while the policy consumes the quantized embedding.
- `β_commit = vq_beta = 0.25` (van den Oord default; paper does not state it).

### 3. Skill prior = future-skill histogram  (Sec. 4.1)
`future_skill_histogram`.
- `Z_t = normalize( Σ_{t'≥t, valid} one_hot(z_{t'}) )` — the normalized histogram of
  skill indices used from `t` to the end of the (masked) context **window**.
- **Window-bounded approximation.** With fixed-length training windows we cannot
  reach the true trajectory end, so the future sum is truncated to the window and
  padded steps are masked. This is the faithful-as-possible fixed-batch version of
  the paper's trajectory-end sum (documented deviation). The reverse-cumsum
  realizes the `t'≥t` aggregation; the mask keeps the denominator over real steps.
- Built from discrete `indices` (stop-grad: it is a conditioning signal, not
  differentiated), then fed to the Transformer as one of the three tokens.

### 4. Skill-conditioned causal Transformer policy  (Sec. 4.2)
`SkillDTPolicy` (module key `policy`) + `CausalSelfAttention` / `TransformerBlock`.
- Per-timestep token triple, in paper order **`[Z_t, z_t, s_t]`** (future-skill
  histogram, quantized skill embedding, state), interleaved into a length-`3K`
  sequence with a learned per-timestep positional embedding.
- **Skill not tokenized.** The paper says "we don't tokenize our skill
  embeddings", so the quantized skill `z_t` is fed **directly** as its token (no
  Dense projection) and, per Sec. 4.2, **no timestep embedding is added to it**
  (timestep embeddings go only on the state and histogram tokens). This requires
  `code_dim == embed_dim` (asserted in `create`); the state and histogram tokens
  are still projected and get the timestep embedding.
- **Encoder per timestep.** Optional visual encoders are
  applied per timestep via `jax.vmap` over the time axis (in `SkillVQ.encode` and
  `SkillDTPolicy`), instead of feeding a `[B, T, …]` tensor into an encoder that
  expects `[B, …]`. `create` also inserts a length-1 time axis for image example
  obs so module shapes initialize correctly. State-based default is unaffected.
- GPT-style pre-LN blocks, causal mask `tril`. Action `âₜ` is read from the hidden
  state at the `s_t` position (index `3t+2`), exactly the DT readout convention.
- The repo was MLP-only; the attention blocks were added here in Flax and are
  JIT-traceable (static shapes, `einsum` attention, `jnp.tril` mask).

### 5. Total objective  (Sec. 4.3, Alg. 1)
`SkillDTAgent.total_loss` + `_seq_batch`.
- `L = (1/K) Σ_t (a_t − âₜ)²  +  vq_coef · VQLOSS(z, ẑ)`.
- The real **length-K window** is now read from the shared `SequenceDataset`
  keys `observations_seq` `[B, K, …]`, `actions_seq` `[B, K, …]`, and `seq_mask`
  `[B, K]` (built in `_seq_batch`). Previously the agent consumed single GCDataset
  transitions, which collapsed the context to **K=1** (no causal window, trivial
  histogram). Padded steps are masked everywhere they are reduced.
- Continuous actions: MSE on the predicted mean (`dist.mode()`), masked by
  `seq_mask`. Discrete actions: cross-entropy (see Deviations).
- `vq_coef = 1.0` (paper writes the two terms with unit weight in Alg. 1).

### 6. Inference  (Sec. 4, rollout)
`SkillDTAgent.sample_actions`.
- Paper rolls the Transformer autoregressively over a K-window and selects the
  best skill by evaluating all skills. OGBench's evaluation loop is **memoryless**
  (calls `sample_actions` per step with only the current observation) and
  **goal-conditioned**, so:
  - the Transformer is run on a **length-1 context** `[Z_t, z_t, s_t]`;
  - the discrete skill is chosen by a deterministic hash of the goal (mirroring
    `empowerment_skill.py`), or sampled uniformly when no goal is given;
  - `z_t` = `codebook[skill]`, `Z_t = one_hot(skill)`.

### Sub-trajectory sampling  (context length K, Sec. 4.2)
The bespoke `SkillDTDataset` was **removed**. Length-K windows now come from the
shared, tested `SequenceDataset` (`utils/datasets.py`, already registered in
`main.py`), selected via `dataset_class='SequenceDataset'` with
`sequence_length = K`. It augments the standard GCDataset batch with
`observations_seq` / `actions_seq` / `seq_mask` (windows clamped at trajectory
boundaries, padded steps masked). No `main.py` edit is required.

---

## Hyperparameters (`get_config`)

| Param | Default | Paper (Sec. A.2, Table 5) |
|---|---|---|
| `num_layers` | 4 | 4 |
| `num_heads` | 4 | 4 |
| `embed_dim` | 256 | 256 |
| `context_len` (K) | 20 | 20 |
| `lr` | 1e-4 | 1e-4 |
| `batch_size` | 256 | 256 |
| `grad_clip` (global norm) | 0.25 | 0.25 |
| dropout | none (0.0) | 0.0 |
| `num_skills` (codebook N) | 16 | 5–64 (task-dependent) |
| `code_dim` | 256 | not stated (set = embed_dim) |
| `vq_hidden_dims` | (256, 256) | not stated |
| `vq_beta` (commitment) | 0.25 | not stated (VQ-VAE default) |
| `vq_coef` | 1.0 | 1.0 (unit weight, Alg. 1) |
| `vq_decay` (EMA) | 0.99 | EMA codebook (paper EMA variant) |
| `dataset_class` | `SequenceDataset` | — (supplies length-K windows) |
| `sequence_length` (K) | 20 | = context_len |

---

## How to run

1. Register the agent (human step): add to `agents/__init__.py`
   ```python
   from agents.skill_dt import SkillDTAgent
   agents = dict(..., skill_dt=SkillDTAgent)
   ```
2. **Faithful K=20 training** (default, no `main.py` edit): the config already sets
   `dataset_class='SequenceDataset'` and `sequence_length=20`, both registered/
   tested in `main.py` and `utils/datasets.py`. Just run the agent — it trains at
   the paper's context length K=20 out of the box.
3. To change K, set `--agent.sequence_length=K --agent.context_len=K` (keep them
   equal: `sequence_length` sizes the data window, `context_len` sizes the
   Transformer positional embedding).

---

## Deviations vs. the paper (scrutinize these)

1. **VQ loss form + EMA codebook.** Paper writes `VQLOSS = MSE(z, ẑ)` (Eq. 1) and
   uses an EMA codebook. We implement the **EMA codebook** (decay 0.99): the
   codebook is derived from EMA accumulators (zero-gradient, updated after the
   optax step), so the gradient loss reduces to the commitment term
   `β·||ẑ − sg[z]||²` with straight-through. This is the standard van den Oord
   EMA VQ pattern.
2. **Discrete-action loss.** Paper's MSE action term targets continuous control
   (D4RL MuJoCo). For OGBench discrete-action environments we substitute
   **cross-entropy** on a categorical action head (MSE on action indices is
   meaningless). Continuous environments use the paper's MSE exactly.
3. **Memoryless, goal-conditioned inference.** OGBench evaluates per-step without
   feeding history, so inference uses a **length-1 context** and a goal→skill hash
   rather than full-window autoregression with exhaustive best-skill selection
   (Sec. 4). This matches the convention in `empowerment_skill.py`.
4. **Context source.** Faithful K=20 uses the shared, tested `SequenceDataset`
   (default `dataset_class`) — no bespoke dataset, no `main.py` edit. (Previously
   the agent consumed single GCDataset transitions and degenerated to K=1.)
8. **Future-skill histogram is window-bounded.** With fixed-length windows the sum
   `Z_t = normalize(Σ_{t'≥t, valid} onehot(z_{t'}))` runs to the window end, not the
   true trajectory end — a faithful-as-possible fixed-batch approximation of the
   paper's trajectory-end sum.
5. **Action distribution wrapper.** Continuous actions are emitted as a
   `MultivariateNormalDiag` with constant std for interface compatibility with
   OGBench's `sample_actions`; the training loss is pure MSE on the mean, so this
   wrapper does not change the objective.
6. **Unspecified sizes.** `code_dim`, `vq_hidden_dims`, and `vq_beta` are not given
   in the paper; defaults chosen to match the Transformer width / VQ-VAE norms.
7. **Reward-free / offline.** No rewards, returns-to-go, or value learning are used
   anywhere — consistent with the paper's reward-free state-marginal-matching
   framing (Sec. 1, 3). The goal inputs from `GCDataset` are ignored by the loss.
