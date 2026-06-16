# VQ-BeT (`vq_bet.py`) — implementation notes

Re-implementation of **VQ-BeT: Behavior Generation with Latent Actions**
(Lee, Wang, Etukuru, Kim, Shafiullah, Pinto; ICML 2024 spotlight;
arXiv:2403.03181; PMLR 235:26991–27008). Official code:
`github.com/jayLEE0301/vq_bet_official`.

This document cites the exact paper section/equation per component and lists
every deviation, so a reviewer with the PDF can cross-check. Equation numbers
below refer to the arXiv v2 PDF.

"Skill" in this codebase ≈ a **tokenized short action chunk**: the discrete RVQ
code(s) indexing a chunk of actions. Codes are DISCRETE; training is OFFLINE and
reward-free (paper §2.1: "behavior datasets crucially do not contain any reward
information").

---

## 1. Method overview (paper §3, Fig. 2)

Two stages:
- **Stage 1 — Action (chunk) discretization via Residual VQ** (§3.2): train a
  Residual VQ-VAE (encoder φ, decoder ψ, `N_q` EMA codebooks) on action chunks
  `a_{t:t+n}` (`n = act_window_size`, real chunks from `SequenceDataset`).
- **Stage 2 — VQ-BeT learning** (§3.3): freeze the tokenizer; a minGPT predicts
  the discrete codes (focal classification) + a continuous offset (L1).

**How the two stages are realized in one OGBench update loop.** A single jitted
`agent.update(batch)` is called repeatedly. We (a) step-gate the *loss*
(`total_loss`): for `step < vqvae_pretrain_steps` only `L_RVQ` flows, afterwards
only `L_VQ-BeT`; and (b) stage-gate two *separate* optimizers via
`optax.multi_transform` (`make_vqbet_optimizer`): the RVQ params (`modules_rvq`)
train ONLY in stage 1 and the policy params (`modules_policy`) ONLY in stage 2.
The gate emits exactly-zero updates AND freezes the inner Adam state when
inactive, so there is **no momentum drift** — the tokenizer is hard-frozen in
stage 2 (verified: RVQ param max-change across a stage-2 step is `0.0`).

---

## 2. Residual VQ-VAE — `ResidualVQ` (paper §2.3, §3.2)

| Component | Paper | Implementation |
|---|---|---|
| Nearest-neighbor code lookup | Eq. 1: `z_q = e_c, c = argmin_j ‖x − e_j‖₂` | `quantize()`: `argmin` of squared L2 over codebook `i`. |
| Residual cascade | §2.3: residual `x − Σ_{i} z_q^i` passed to next layer; `z_q(x)=Σ_{i=1}^{N_q} z_q^i` | `residual = residual − q` per layer; `quantized = Σ q`. |
| Number of layers `N_q` | §3.2: "it was sufficient to use `N_q := 2`" | `vqvae_groups = 2`. |
| Action chunk | §3.2: tokenize `a_{t:t+n}` | `_action_chunk` flattens `batch['actions_seq']` `[B,n,act_dim]` → `[B, n·act_dim]` (real chunks from `SequenceDataset`, `n = act_window_size`). |
| Reconstruction loss | Eq. 2: `L_Recon = ‖a_{t:t+n} − ψ(z_q(φ(a)))‖₁` | `_masked_chunk_l1`: mean `|a − recon|` over valid steps only, weighted by `seq_mask` (padded post-terminal steps excluded). |
| RVQ objective | Eq. 3: `L_RVQ = L_Recon + ‖SG[φ(a)] − e‖₂² + λ_commit‖φ(a) − SG[e]‖₂²` | `recon_loss + commitment_weight·commitment_loss`; the `‖SG[φ]−e‖²` codebook term is realized by the **EMA codebook update** (paper §3.2), not by gradient. `codebook_loss` is still computed as a diagnostic metric. |
| Codebook update rule | §3.2: **EMA** of codebook entries (decay ~0.99), per residual layer | `_ema_codebook` (standard van den Oord EMA codebook): stores EMA accumulators `cluster_size [N_q,k]`, `embed_avg [N_q,k,d]` as zero-grad params; updates `N_i ← βN_i+(1−β)n_i`, `m_i ← βm_i+(1−β)Σ residual_i` after the optax step, stage-1 only. Codebook `e_i = m_i/N_i` (Laplace-smoothed) derived on read, stop-gradient'd. |
| Commitment weight | §3.2: `λ_commit := 1` | `commitment_weight = 1.0`. |
| Primary / secondary codes | §3.2: layer-1 code = *primary*, remaining = *secondary* | `indices[:,0]` primary, `indices[:,1]` secondary. |
| Straight-through estimator | implicit (standard VQ-VAE, Van Den Oord 2017) | `quantized_st = z + SG(quantized − z)`. |

**Codebook update rule (now faithful).** Paper §3.2 updates codebook embeddings
with **EMA** ("moving averages rather than direct gradient updates"). This is now
implemented exactly (`ResidualVQ` EMA accumulators + `VQBeTAgent._ema_codebook`,
decay `vqvae_ema_decay = 0.99`, per residual layer). The encoder still commits
via the straight-through estimator with `λ_commit = 1`; the codebook itself
receives zero gradient (all reads are stop-gradient'd) and is overwritten by the
EMA rule, run only during stage 1 so the tokenizer is fully frozen in stage 2.

---

## 3. minGPT Transformer + heads — `VQBeTPolicy` (paper §3.3, §3.4)

- **Causal self-attention / blocks** (`CausalSelfAttention`, `GPTBlock`):
  real pre-LN minGPT blocks (multi-head attention + 4× GELU MLP + residual),
  causal mask. Defaults `gpt_n_layer=6, gpt_n_head=6, gpt_n_embd=120`
  (official minGPT sizing).
- **Conditioning** (§3.4 conditional formulation `π: O^h × O^g → A^n`):
  token sequence `[goal_token, obs_token]` with a learned positional embedding;
  the Transformer feature at the observation position drives the heads.
- **Code prediction (focal)** — Eq. 4:
  `L_code = L_focal(ζ_code^{i=1}(o_t)) + β·L_focal(ζ_code^{i>1}(o_t))`.
  `head1` → primary logits; `head2(concat(feat, onehot(primary)))` → secondary
  logits (the *hierarchical code prediction* of Fig. 2). `focal_loss` is
  multiclass focal (Lin et al. 2017) with `focal_gamma = 2.0`.
- **Offset head** — Eq. 5/6:
  `⌊a_{t:t+n}⌋ = ψ(Σ_{j,i} e_j^i·𝟙[ζ_code^i=j])`,
  `L_offset = |a_{t:t+n} − (⌊a⌋ + ζ_offset(o_t))|₁`.
  `offset_head(feat)` → continuous offset of size `n·action_dim`; `⌊a⌋` is the
  **frozen** decode of the **predicted (argmax)** codes (`_rvq_decode_codes`,
  `params` stop-gradient'd). `argmax` is non-differentiable, so code heads get
  gradient only from focal loss and the offset head only from the L1 — matching
  the paper's separate heads/objectives.
- **Total stage-2 loss** — Eq. 7: `L_VQ-BeT = L_code + L_offset`
  (`offset_loss_weight` multiplies L_offset; see deviations).

### Inference (`sample_actions`, paper §3.3 + Fig. 2 bottom-right)
1. Predict primary logits + offset (pass 1).
2. Sample primary code; condition `head2` on its one-hot → secondary logits
   (pass 2); sample secondary code.
3. `⌊a⌋ = ψ(e_primary + e_secondary)`; `action_chunk = ⌊a⌋ + offset` (Eq. 6).
4. Return the first action of the chunk, clipped to `[-1, 1]`.
Temperature scales the code logits (eval `temperature=0` → ≈ argmax via a 1e-6
floor, mirroring `GCDiscreteActor`).

---

## 4. Hyperparameters (`get_config`)

| Key | Default | Source |
|---|---|---|
| `vqvae_groups` (`N_q`) | 2 | §3.2 (`N_q := 2`). |
| `vqvae_n_embed` (`k`) | 16 | task spec "8–16 codes/layer"; official default 32 (env-dependent). |
| `vqvae_latent_dim` | 128 | official EncoderMLP width. |
| `vqvae_hidden_dims` | (128,128) | official EncoderMLP. |
| `commitment_weight` (`λ_commit`) | 1.0 | §3.2. |
| `act_window_size` (`n`) | 5 | chunk length; paper uses 1–6 (Table 13). |
| `gpt_n_layer/head/embd` | 6/6/120 | official minGPT. |
| `focal_gamma` | 2.0 | official default (Lin et al. 2017). |
| `primary_code_weight` | 1.0 | Eq. 4 primary term := 1 (official code uses 5; see deviations). |
| `secondary_code_weight` (`β`) | 0.5 | Eq. 4 secondary; paper Table 13 task-dependent 0.1–0.6. |
| `offset_loss_weight` | 1.0 | Eq. 7 := 1 (official uses `1e3`; see deviations). |
| `vqvae_pretrain_steps` | 100000 | stage-1 length (see deviations). |
| `lr` | 1e-4 | official ≈1e-4. |

---

## 5. Deviations to scrutinize

The following are faithful to the paper and called out so a reviewer can confirm
the exact realization; items 5–9 are genuine, unavoidable deviations.

1. **EMA codebook updates (faithful).** The codebook is EMA-updated per residual
   layer (decay 0.99), matching paper §3.2, instead of a gradient on the Eq. 3
   codebook term. See the "Codebook update rule" row in §2.

2. **Hard tokenizer freeze in stage 2 (faithful).** In addition to the loss
   step-gating (`L_RVQ` only in stage 1, `L_VQ-BeT` only in stage 2), the
   optimizer is `optax.multi_transform` over two **stage-gated** Adams
   (`make_vqbet_optimizer` + `_stage_gated`): `modules_rvq` updates ONLY in
   stage 1, `modules_policy` ONLY in stage 2. When inactive, a stage emits
   exactly-zero updates *and* freezes the inner Adam moments, so there is **no
   momentum drift** — a true freeze (verified: RVQ param max-change across a
   stage-2 step is `0.0`). The gate count starts at 1 to align with
   `TrainState.step`, so the optimizer boundary matches the loss-gate boundary
   at `vqvae_pretrain_steps`. The predicted-code floor-decode (`_rvq_decode_codes`)
   uses a stop-gradient'd, frozen tokenizer. Residual caveat: the boundary is
   fixed by `vqvae_pretrain_steps` rather than a manual two-phase launch, but the
   parameter freeze itself is exact.

3. **Real multi-step action chunks (faithful).** Training uses `SequenceDataset`
   (`dataset_class='SequenceDataset'`, `sequence_length = act_window_size`,
   default `act_window_size = 5`). `_action_chunk` reads `batch['actions_seq']`
   `[B, n, action_dim]` and flattens to `[B, n·action_dim]`; `seq_mask` masks
   padded post-terminal steps in BOTH the L1 reconstruction (Eq. 2) and the
   offset L1 (Eq. 6) via `_masked_chunk_l1`. `action_chunk_dim = act_window_size ·
   action_dim`. `create` asserts `sequence_length == act_window_size`. (The chunk
   *latent* is per-chunk, so the EMA codebook clusters the whole-chunk encoding;
   the mask applies to the per-step L1 terms.)

4. **`N_q == 2` assertion (faithful).** The policy's two code heads (one primary
   + one secondary) hard-assume `N_q = 2`. `create` asserts `vqvae_groups == 2`
   with a clear message pointing at `VQBeTPolicy` for generalization. The RVQ
   tokenizer itself is `N_q`-generic.

5. **Observation history `h = 1`.** OGBench provides a single observation (no
   `o_{t-h:t}` history). The Transformer sequence is `[goal, obs]` (length 2),
   which still yields non-trivial causal attention (obs attends to goal). The
   paper uses an observation window (`obs_window_size` 10 in the official code).
   Longer histories would require a windowed dataset.

6. **Offset head parameterization.** Paper Eq. 6 writes `ζ_offset(o_t)` (a
   function of the observation), which we implement directly: `offset_head(feat)`
   → `n·action_dim`. The official code emits a richer offset tensor indexed by
   the chosen `(group, code)` and gathers per sampled code. Our form matches the
   equation literally; the official form has more capacity.

7. **Loss weights `primary_code_weight` / `offset_loss_weight`.** Paper Eq. 4/7
   put weight 1 on the primary focal term and weight 1 on `L_offset`; the
   official code multiplies the primary focal by 5 and `L_offset` by `1e3`
   (tuned for normalized actions). We default to the paper's Eq. 4/7 weights
   (1.0 / 1.0) and expose both as config so the official scaling is reproducible.
   `secondary_code_weight` (β) is genuinely task-dependent in the paper
   (Table 13: 0.1–0.6); default 0.5.

8. **Continuous actions only.** VQ-BeT tokenizes continuous action vectors; the
   constructor asserts `discrete == False`. Discrete-action OGBench tasks are not
   supported by this method (paper §1: actions are "continuous-valued vectors").

9. **No dropout.** Dropout is omitted from the minGPT blocks for JIT/RNG
   simplicity (offline BC); add `nn.Dropout` if regularization is needed.

---

## 6. What is faithful (quick checklist)

- Residual VQ with `N_q=2`, nearest-neighbor codes, **EMA codebooks** (decay
  0.99, per layer) + commitment loss (Eq. 3, `λ_commit=1`). ✓
- Real multi-step action chunks `a_{t:t+n}` (`SequenceDataset`, `n=act_window_size`)
  with `seq_mask` applied to the L1 recon (Eq. 2) and offset (Eq. 6). ✓
- Discrete primary + secondary codes; hierarchical (secondary conditioned on
  primary). ✓
- Real minGPT causal-attention Transformer. ✓
- Focal classification loss for code prediction (Eq. 4, `γ=2`, weight `β`). ✓
- Continuous offset head with L1 loss (Eq. 6); total `L_code + L_offset` (Eq. 7).
  ✓
- Offline, reward-free, two-stage with a **hard-frozen tokenizer in stage 2**
  (stage-gated optimizers: zero updates + no momentum drift on `modules_rvq`). ✓
- Inference: predict primary → secondary code, decode summed embeddings, add
  offset. ✓
