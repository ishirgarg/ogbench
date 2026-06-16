# DDS agent — implementation notes

Faithful OGBench re-implementation of **"Offline Reinforcement Learning with
Discrete Diffusion Skills" (DDS), arXiv:2503.20176**. There is no official code
release; "faithful" means matching the paper's specified method, and using the
conventions of the cited methods for genuinely paper-unspecified details.

File: `impls/agents/dds.py` (agent_name = `dds`). Public interface:
`create / update / total_loss / sample_actions / sample_actions_with_state /
init_eval_state / get_config`.

Section/equation/table numbers refer to the DDS paper (arXiv HTML rendering).

---

## 1. Component → paper mapping

### 1.1 Transformer sub-sequence encoder `TrajectoryEncoder` (Sec. 3.2, Table 4)
- Paper: a transformer encoder maps a length-`H` state–action sub-sequence
  `τ_H` to a continuous embedding `E(τ_H) ∈ R^{D_z}`. Table 4: **4 layers, 8
  heads, 256-d**, dropout 0.1; flow = concat state-action → input embedding →
  positional encoding → 4 transformer layers → adaptive avg pooling → 2-layer
  MLP projection → `D_z`.
- Here: **exactly that architecture** over the real length-`H` window from
  `SequenceDataset` (`observations_seq`/`actions_seq`). Per-step concat
  `[enc(s_i), a_i]` → linear token embedding → learnable positional encoding →
  `encoder_layers` pre-LN `TransformerEncoderBlock`s (`encoder_heads` heads,
  `encoder_dim`-d, 4× MLP) → masked average pooling (padded steps excluded via
  `seq_mask`) → 2-layer MLP head → `D_z`. Dropout is omitted (deterministic);
  see B3.

### 1.2 Discrete skill codebook `VQCodebook` (Sec. 3.2, Eq. 10–11, 14)
- Codebook `{z_k}_{k=1..K}`. **Exact** nearest-neighbour assignment
  `k = argmin_j ‖E(τ_H) − z_j‖` and **exact** straight-through estimator
  `z_q_st = e + sg(z_q − e)`. Implemented verbatim.

### 1.3 VQ loss terms (Sec. 3.2, Eq. 14)
- Codebook loss `‖sg[E]−z‖²`, commitment loss `β‖sg[z]−E‖²` with **β = 0.25**,
  plus the diffusion reconstruction of the **whole `H`-step action block**
  (`seq_mask` applied). `L_skill = recon + codebook + β·commitment`.

### 1.4 Diffusion decoder `DiffusionActionDecoder` (Sec. 3.2, Table 5)
- Paper Table 5: sinusoidal time embedding (size 16) → input projection → **4
  residual blocks, each a 4×-expand-then-compress LayerNorm-MLP** → output
  projection; **VP β-schedule** (`β_min=0.1`, `β_max=10`); **5** diffusion steps.
- Here: **exactly that** denoiser (`decoder_blocks` × `ResidualDenoiseBlock`,
  each `LayerNorm → Dense(4·dim) → GELU → Dense(dim) → +residual`), ε-prediction
  MSE per window step, forward noising + ancestral DDPM sampling
  (`_ddpm_sample`) over `diffusion_steps`, on the VP schedule. The shared skill
  `z` and per-step state `s_i` condition every denoise.
- For **discrete-action** envs the diffusion decoder does not apply; we
  substitute `SkillDiscreteDecoder` (categorical BC, masked cross-entropy). See
  B1.

### 1.5 High-level value + skill-indexed critic `value_loss` / `high_critic_loss` (Sec. 3.3, Eq. 6, 8)
- `value_loss` (**Eq. 6**): implicit value `V(s,g)` (single `GCValue` head) =
  τ-expectile of the **target** critic `Q_target(s, k*, g)`, `τ_IQL = 0.7`.
- `high_critic_loss` (**Eq. 8**): skill-indexed `Q(s,k,g)` (`GCDiscreteCritic`,
  ensemble of 2 over the `K` codes) regressed toward the **macro-transition
  target with the H-step discounted snippet return and a SINGLE-γ bootstrap**:
  `target = Σ_{i=0}^{H-1} γ^i r_i + γ · mask_H · V(s_{t+H}, g)`. The per-window
  rewards `r_i` / masks come from `SequenceDataset` (`rewards_seq` / `masks_seq`,
  computed exactly as `GCDataset` does for a single transition); `s_{t+H}` is
  `subgoal_observations`; `mask_H` cuts the bootstrap once the goal is reached
  inside the window. This is the paper's Eq. 8 (single γ, not γ^H).

### 1.6 High-level discrete-code policy `high_actor_loss` (Sec. 3.3, Eq. 9)
- `GCDiscreteActor` over the `K` codes, AWR with the skill-indexed advantage
  `A = Q(s, k*, g) − V(s, g)`, weight `min(exp(α·A), 100)`; BC target = `k*`.

### 1.7 Test-time execution — H-step skill commitment `sample_actions_with_state` (Sec. 4.4)
- Paper: the agent **selects a skill every `H` steps** and the diffusion
  decoder produces `a_i = D(z, s_i)` each step with `z` held.
- Here: `init_eval_state` + `sample_actions_with_state` thread a per-episode
  `{skill, count}` through OGBench's eval loop (an **additive** hook in
  `utils/evaluation.py`). The high-level policy re-selects only when
  `count % H == 0`; otherwise the committed code is held and the low-level
  decoder runs each step. Verified: the code holds for exactly `H` steps. Agents
  without the hook keep the original stateless per-step path unchanged.
  `sample_actions` (stateless, per-step) is kept for direct callers.

---

## 2. Hyperparameters (`get_config`, paper defaults)

| Config key             | Default      | Paper reference                          |
|------------------------|--------------|------------------------------------------|
| `num_skills` (K)       | 16           | codebook size K = 16 (Table 6)           |
| `skill_dim` (D_z)      | 128          | latent dim D_z = 128 (Table 6)           |
| `commitment_beta` (β)  | 0.25         | commitment coefficient (Eq. 14 / Table 6)|
| `subgoal_steps` (H)    | 10           | horizon H = 10 (Table 6)                 |
| `encoder_dim/layers/heads` | 256/4/8  | transformer encoder (Table 4)            |
| `decoder_dim/blocks/expand`| 256/4/4  | residual denoiser (Table 5)              |
| `diffusion_steps`      | 5            | 5 denoising steps (Table 5)              |
| `time_dim`             | 16           | sinusoidal time embedding (Table 5)      |
| `beta_min` / `beta_max`| 0.1 / 10.0   | VP schedule (Table 5)                    |
| `value/actor_hidden_dims` | (256,256) | 2×256 ReLU/MLP heads (paper)             |
| `skill_lr` / `iql_lr`  | 5e-5 / 1e-4  | Tables 6 / 7                             |
| `skill_batch_size` / `batch_size` | 128 / 256 | Tables 6 / 7                  |
| `discount` (γ)         | 0.99         | Table 7                                  |
| `expectile` (τ_IQL)    | 0.7          | Table 7                                  |
| `tau` (EMA)            | 0.005        | Table 7                                  |
| `high_alpha` (AWR α)   | 3.0          | unspecified → IQL/AWR default (B2)       |
| `skill_pretrain_steps` | 500000       | skill phase length (Table 6) → hard freeze|

`sequence_length` (window length) **must equal** `subgoal_steps` = `H`
(asserted in `create`).

---

## 3. Training — separate optimizers, hard phased freeze (Sec. 4)

The paper trains in phases: skill VQ-VAE (500k steps) → relabel → IQL Q-learning
(1M) + AWR (500k), as separate runs with separate optimizers (skill lr 5e-5,
IQL lr 1e-4). We implement this as a **true two-phase procedure within one run**:

- `optax.multi_transform` routes the skill modules
  (`encoder`/`codebook`/`decoder`) to a skill Adam (lr 5e-5) and the IQL modules
  (`value`/`high_critic`/`high_actor`) to an IQL Adam (lr 1e-4); the
  `target_high_critic` is `set_to_zero` (updated only by the manual soft-update).
- `make_phased_adam` HARD-freezes the inactive phase: when inactive it zeroes
  both the parameter updates **and** the Adam first/second moments, so a frozen
  module's parameters do not drift at all. The freeze flips on exactly the same
  step as the loss-weight gate in `total_loss` (both use the 1-indexed
  `TrainState.step`).
- Phase 1 (`step < skill_pretrain_steps`): only the skill loss has weight and
  only the skill optimizer is active. Phase 2: skill loss weight = 0, skill
  optimizer frozen; the high level trains on the now-fixed codes/labels (the
  high-level losses read `k*` from stored params, so no gradient ever reaches the
  skill model). **Verified: skill-param max-change == 0 across phase-2 steps.**
- Skill phase uses `skill_batch_size` (128) by slicing the leading rows of the
  shared 256-row batch; the IQL phase uses the full 256. This realizes the
  paper's 128/256 batch sizes inside OGBench's single-batch pipeline without
  touching `main.py`.

---

## 4. Irreducible adaptations / paper-unspecified details (kept, documented)

- **A1 Goal-conditioning (irreducible).** The paper's IQL is unconditional on a
  relabeled task-reward semi-MDP. OGBench has no separate task reward, so the
  value/critic/actor are goal-conditioned with the GCDataset goal sampler —
  exactly the standard OGBench instantiation used by reference `gciql`/`hiql`.
- **A2 Per-step reward = OGBench GC reward (irreducible).** The snippet return
  `Σ_{i<H} γ^i r_i` sums OGBench's goal-conditioned per-step reward w.r.t. the
  same value goal (there is no other reward signal). The relabel/return/bootstrap
  structure (Eq. 8) is otherwise the paper's.
- **B1 Discrete action spaces (paper-unspecified).** The paper covers only
  continuous actions; we add a categorical skill-conditioned decoder for
  discrete-action OGBench envs. VQ + high level identical.
- **B2 AWR α (paper-unspecified).** Table 7 omits α; we use the IQL/AWR
  convention α = 3.0.
- **B3 Activation / dropout (paper-unspecified / minor).** Tables 4–5 give block
  structure but not the activation (we use OGBench GELU); the 0.1 dropout is
  omitted (deterministic) because OGBench's ModuleDict apply path does not thread
  a dropout rng — a regularization detail with no effect on the objective.
- **B4 Phase-2 budget (compute budget, not method).** We honor the 500k skill
  phase, then train the whole high level for OGBench's remaining single-run
  budget (default 500k), rather than the paper's separate 1M Q + 500k AWR runs.

---

## 5. Additive OGBench infrastructure changes

- `utils/datasets.py`: `GCDataset.sample` gains an opt-in `return_goal_idxs`
  (default False, fully backward compatible). `SequenceDataset` additionally
  emits `rewards_seq` / `masks_seq` (per-window GC reward/mask vs. the same value
  goal) and `subgoal_observations` (`s_{t+H}`), used to form the H-step snippet
  return + single-γ bootstrap. Existing fields/agents (Skill-DT, QueST, VQ-BeT)
  are unaffected (verified).
- `utils/evaluation.py`: opt-in `init_eval_state` / `sample_actions_with_state`
  hook for H-step skill commitment; agents without the hook are unchanged.
- `main.py`: unchanged (per-phase batch sizing is handled inside the agent).
