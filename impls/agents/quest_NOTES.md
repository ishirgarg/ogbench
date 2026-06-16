# QueST agent — implementation notes

Re-implementation of **QueST: Self-Supervised Skill Abstractions for Learning
Continuous Control** (Mete, Xue, Wilcox, Chen, Garg; NeurIPS 2024;
arXiv:2407.15840). Official code: github.com/pairlab/QueST.

File: `impls/agents/quest.py` (`agent_name='quest'`). Public interface mirrors
`empowerment_skill.py` / `gcbc.py`: `flax.struct.PyTreeNode`, `@classmethod
create`, jitted `update` / `total_loss`, `sample_actions(observations, goals,
seed, temperature)`, module-level `get_config()`.

A reviewer with the PDF can cross-check each component below. Equation / section
/ table numbers refer to the arXiv v3 HTML.

---

## 1. Finite Scalar Quantization (FSQ) — paper Sec. 3.2, Eq. 1–2

QueST discretizes skills with FSQ (Mentzer et al. 2023, arXiv:2309.15505)
instead of VQ. Each of the `d = len(levels)` latent channels is bounded and
rounded to `levels[i]` integer values; the implicit codebook has size
`prod(levels)`.

Implemented as pure functions in `quest.py`:

- `fsq_bound(z, levels)` — `f(z) = tanh(z + shift)·half_l − offset`
  (FSQ paper Eq. 4); `offset = 0.5` for even levels, `0` for odd.
- `fsq_quantize(z, levels)` — `round_ste(bound(z)) / (levels//2)`. The rounding
  uses a **straight-through estimator** (`_round_ste`: forward `round`, backward
  identity). This is the differentiable path used for reconstruction.
- `fsq_codes_to_indices` / `fsq_indices_to_codes` — mixed-radix
  (de)serialization between per-channel codes and a single integer token id in
  `[0, prod(levels))`, used by the prior.

**Levels `[8,5,5,5]` → vocab `8·5·5·5 = 1000`** (paper: "effective vocab ≈1000",
Tables 3–4). Verified: round-trip `codes→indices→codes` is exact (err 0.0),
indices land in `[0, 1000)`.

**No codebook / commitment loss.** FSQ's fixed grid needs no auxiliary VQ terms,
which is exactly why Stage I is reconstruction-only (paper Sec. 3.2, Sec. 4.1).

---

## 2. Stage I — chunked action autoencoder — paper Sec. 4.1, Table 3

- **Input:** action chunk `a_{t:t+T-1}`, shape `[B, T, A]`, `T = 32`. These are
  **real sub-trajectory windows** supplied by `SequenceDataset` as
  `batch['actions_seq']` (see §7.1), with `batch['seq_mask']` ([B, T], 1.0 = real
  in-trajectory step, 0.0 = padded past terminal). `_get_chunk` returns the
  `(chunk, mask)` pair and the L1 loss masks padded steps.
- **Encoder `ActionEncoder` (φ_θ):** 3 causal strided 1-D convolutions
  (kernels `[5,3,3]`, strides `[2,2,1]`, dim 256) downsample time by `F = 4` →
  `n = T/F = 8` tokens, followed by `enc_layers = 2` causal self-attention
  blocks (4 heads), then a linear projection to the FSQ dim `d = 4`. Causality
  via left-padded convolutions (`CausalConv1d`) + lower-triangular attention
  mask (paper: "causal strided convolutions … masked self-attention").
- **Quantization:** `(z¹,…,zⁿ) = FSQ(φ_θ(a_{t:t+T-1}))` (Eq. 2).
- **Decoder `ActionDecoder` (ψ_θ):** `dec_layers = 4` transformer blocks
  (dim 256, 4 heads). **Fixed sinusoidal** per-timestep positional **query**
  inputs (length `T`) self-attend **causally** (lower-triangular self-mask) and
  **cross-attend to ALL `n` skill tokens** (no cross-attention mask). This
  matches the paper: "the decoder cross attends between fixed sinusoidal
  positional embedding inputs and the skill tokens", "attending to all codes
  while maintaining causality" (paper Sec. 4.1). Causality is enforced purely by
  the masked query self-attention, not by masking the cross-attention. A final
  linear layer outputs `[B, T, A]`.
- **Loss — Eq. 4:** masked L1 reconstruction `‖ψ_θ(FSQ(φ_θ(a))) − a‖₁` only
  (`reconstruction_loss`). The per-timestep L1 is multiplied by `seq_mask` and
  averaged over real `(step, action-dim)` elements, so padded post-terminal steps
  contribute zero gradient. Verified: overfits a fixed batch (recon_l1
  1.26 → 0.37 in 60 steps), confirming gradients flow through the FSQ STE.

---

## 3. Stage II — autoregressive skill prior — paper Sec. 4.2, Table 4

- **Module `SkillPrior` (π_φ):** GPT-style **causal Transformer**
  (`prior_dim = 384`, `prior_layers = 6`, `prior_heads = 6`) over the discrete
  skill tokens. Real multi-head attention implemented from scratch
  (`MultiHeadAttention`, `TransformerBlock`) with a lower-triangular mask.
- **Factorization — Eq. 5:**
  `π_φ(Z | state, e) = Π_i π_φ(z_i | <s>, z_{<i}, state, e)`. A conditioning
  token built from the (goal-)state observation via a 2-layer MLP plays the role
  of the learnable start token `<s>` at sequence position 0; teacher-forced token
  embeddings (`nn.Embed(1000, 384)`) fill positions `1..n-1`. **Fixed
  sinusoidal** positional embeddings (`sinusoidal_embedding`, paper Sec. 4.2)
  are added on the token sequence. The head predicts a full `vocab = 1000`-way
  distribution at every position. (Paper Table 4 also lists attention dropout
  0.1; omitted here because OGBench agents do not thread a train/eval flag or
  dropout rng — see §7.7.)
- **Conditioning `e`:** in QueST `e` is a task embedding (CLIP for LIBERO,
  learned for MetaWorld). In the OGBench **goal-conditioned** setting we map
  `e` → the goal observation (`actor_goals`), concatenated with the current
  observation (and passed through the visual `GCEncoder` when `encoder` is set).
  Toggle with `goal_conditioned` (default True).
- **Loss — Eq. 6:** token-level NLL / cross-entropy
  `−log π_φ(Z_t | state, e)` via
  `optax.softmax_cross_entropy_with_integer_labels` (`prior_loss`). Verified:
  overfits to CE ≈ 0.002, token accuracy 1.0.
- **Gradient isolation:** the prior's targets and inputs are integer token ids
  (`stop_gradient` + rounding), so prior gradients never reach the encoder —
  matching the paper's frozen-codebook Stage II.

---

## 4. Inference — paper Sec. 4.3

`sample_actions` (`_sample_tokens`): sample `n` tokens autoregressively from the
prior with **top-k = 5** filtering then temperature sampling (paper Table 4:
`top-k = 5`, `τ = 1`); `temperature == 0` → greedy argmax (used by OGBench eval,
which passes `eval_temperature = 0`). Tokens → FSQ codes → decoder → action
chunk `â_{t:t+T-1}`; the first action `â_t` is returned, clipped to `[-1, 1]`.
The autoregressive loop is a `jax.lax.fori_loop` (O(n²), n = 8) and fully JIT-able.

---

## 5. Two-stage offline training inside one OGBench `update`

The paper trains Stage I then Stage II sequentially. OGBench drives a single
`agent.update(batch)` per step, so `total_loss` gates the two losses by the
optimizer step (`stage1_steps`, default 500k of 1M):

- `step < stage1_steps`: weight `recon = 1`, `prior = 0` → only the autoencoder
  trains (prior params get zero gradient).
- `step ≥ stage1_steps`: weight `recon = 0`, `prior = 1` → autoencoder frozen
  (zero gradient), only the prior trains on the now-fixed token targets.

Set `joint_training = True` to optimize both simultaneously (the two losses do
not share gradients, so this is a valid alternative). Verified: gating flips at
`stage1_steps`.

---

## 6. Key hyperparameters (`get_config`)

| Param | Default | Paper source |
|---|---|---|
| `horizon_length` T | 32 | Sec. 4.1 / Table 3 |
| `downsample_factor` F | 4 → `num_tokens` n = 8 | Table 3 |
| `fsq_levels` | (8,5,5,5) → vocab 1000 | Sec. 3.2 / Tables 3–4 |
| `ae_dim` | 256 | Table 3 |
| `conv_kernels` / `conv_strides` | (5,3,3) / (2,2,1) | Table 3 |
| `enc_layers` / `enc_heads` | 2 / 4 | Table 3 |
| `dec_layers` / `dec_heads` | 4 / 4 | Table 3 |
| `prior_dim` | 384 | Table 4 |
| `prior_layers` / `prior_heads` | 6 / 6 | Table 4 |
| `top_k` | 5 | Table 4 |
| `lr` | 1e-4 | Sec. 5 / appendix |
| `stage1_steps` | 500000 | (schedule, our choice) |
| `batch_size` | 1024 | OGBench convention |

---

## 7. Deviations a reviewer should scrutinize

1. **Dataset / action chunks — FIXED.** Stage I now consumes **real** T-step
   action windows. The agent selects `dataset_class='SequenceDataset'` with
   `sequence_length = horizon_length = 32` (`create` asserts the two agree).
   `SequenceDataset` (registered in `main.py`) augments the standard `GCDataset`
   batch with `observations_seq` `[B, T, *obs]`, `actions_seq` `[B, T, A]`, and
   `seq_mask` `[B, T]` — contiguous windows sampled from the SAME trajectory with
   trajectory-boundary clamping (its window starts at the base transition index,
   so `actions_seq[:, 0]` equals the base single-step action). `_get_chunk`
   returns `(batch['actions_seq'], batch['seq_mask'])`; the L1 reconstruction
   masks padded post-terminal steps so they contribute no gradient. The earlier
   tiling placeholder (broadcasting one `[B, A]` action across `T`) is **gone** —
   Stage I now sees genuine `a_{t:t+T-1}` sub-trajectories.

2. **Execution horizon.** The paper executes `T_a = 8` actions open-loop, then
   replans (Sec. 4.3). OGBench's eval loop calls `sample_actions` every env step
   and is stateless, so we **replan every step and return only `â_t`**
   (open-loop horizon 1). More conservative; slightly more compute at eval.

3. **Two-stage schedule via step-gating** rather than two separate training
   scripts (Sec. 5). Equivalent in gradient flow (the losses are decoupled), but
   the split point `stage1_steps` is a hyperparameter we introduced.

4. **Conditioning `e` = goal observation.** QueST uses CLIP/learned task
   embeddings; we substitute the OGBench goal observation (Sec. 4.2). For the
   non-goal-conditioned variant set `goal_conditioned=False` (prior conditions on
   state only).

5. **Few-shot decoder finetuning (Eq. 7)** is **not implemented** — it is a
   transfer/adaptation step outside the offline pretraining objective requested
   here; documented for completeness only.

6. **Minor architecture specifics** (exact LayerNorm placement = pre-LN, MLP
   ratio 4, GELU) follow standard GPT/ViT practice where the paper/Tables are
   underspecified; capacities (dims, heads, layers, kernels, strides) match
   Tables 3–4 exactly. **Positional embeddings — FIXED to match the paper:** the
   decoder query inputs and the prior's token sequence now use **fixed
   sinusoidal** embeddings (`sinusoidal_embedding`) exactly as the paper states
   (Sec. 4.1–4.2). The encoder's positional embedding remains a learned param
   because the paper does not specify the encoder's positional scheme.

7. **No attention/embedding dropout.** Paper Table 4 lists attention dropout 0.1
   for the prior. OGBench agents do not thread a train/eval flag or a dropout
   rng through `update` / `sample_actions`, so dropout is omitted. This only
   affects regularization/training dynamics, not the architecture or objective.

---

## 8. Validation performed (no heavy training)

CPU smoke tests (see commit/PR description): FSQ exact round-trip and index
range; `create` + jitted `update` on `SequenceDataset`-style batches carrying
`observations_seq` / `actions_seq` / `seq_mask` (with randomly truncated masks);
both stages exercised (recon-only then prior-only) with finite losses;
`sample_actions` single/batch, bounded to `[-1,1]`, deterministic at `τ=0`;
validation path `total_loss(batch, grad_params=None)`; stage-gating flip;
**AE overfit** (recon_l1 1.26→0.37) and **prior overfit** (CE→0.002, acc 1.0);
visual `encoder='impala_small'` build + update + sample on image observations.
Post-fix smoke test (`/Users/ishirgarg/anaconda3/bin/python`): both jitted
`update` stages and `sample_actions` run crash-free with finite losses on a
synthetic batch containing real `actions_seq` and a partial `seq_mask`.
