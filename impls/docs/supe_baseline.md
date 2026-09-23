# SUPE baseline (arXiv:2410.18076) on our online envs

Status: implemented 2026-09-22 on branch `claude/supe-baseline-integration-4d17b9` (forked from the
distilled-bonus branch `claude/empowerment-bonus-network-crl-daeab5`, so every distill run's code, envs
and launchers are on this branch too). Smoke-tested locally; no cluster runs submitted yet.

## What SUPE is, in our terms

SUPE = frozen **OPAL trajectory skills** (a VAE `q(z|s_{1:H},a_{1:H})`, `p(z|s_1)`, `pi(a|s,z)`,
`H = 4`) + an online **SAC high level** over the latent (in tanh space) trained with **RLPD** on the
offline dataset, whose windows are pseudo-labelled optimistically: reward = the dataset minimum
(`ds_minr`) and mask = a learned termination model, plus an **RND** bonus on offline and online
`(s, z)` pairs. Warm-up skills come from the OPAL prior. Everything else is RLPD-flavoured SAC
(10 critics with LayerNorm, random `num_min_qs` heads in the target, no entropy backup).

## Two stages, the first run once

| stage | script | what it produces |
|---|---|---|
| offline OPAL, once per dataset | `scripts/slurm/submit_supe_opal_pretrain.sh` (-> `run_supe_opal_pretrain.sbatch`) | `ckpts/final/supe_opal/<dataset>/OGBench/Debug/sd000_*` in the MAIN checkout: `agents/opal.py`, `latent_type=continuous`, chunk 4, kl 0.1 (0.2 cube), 1M steps |
| online SUPE, per cell x seed x arm | `scripts/slurm/submit_supe_online_seeds.sh` (-> `run_supe_online_seed.sbatch`, cells in `supe_cells.sh`) | `exp/supe_slurm/<cell>/<tag>/OGBench/Debug/sd<seed>_*` in the MAIN checkout (eval.csv / train.csv / wandb, `agent_name=supe`) |

The online launcher finds the dataset's OPAL run automatically and refuses a mismatch in
`latent_type`, `chunk_size` (must equal `K`) or dataset. Nothing is retrained online: the
checkpoint is loaded frozen exactly like the other controllers (`utils/skill_checkpoint.py`).

Cells (`supe_cells.sh`) are the exact (online env, RLPD dataset, episode length, estimator run)
quadruples of the flat-CRL and distilled-bonus sweeps, with SUPE's per-env `num_min_qs` (1 mazes /
antsoccer, 2 cube) and discount (0.995 cube / antsoccer, 0.99 mazes): `amz_center amz_corner amz_all
pmt_center pmt_sparse pmt_corner pmt_all asoc_center asoc_corner asoc_cmg asoc_fmg cube_sgl cube_mg
cube_dbl`.

## Combining with the empowerment distillation bonus

One flag. `run_supe_online_seed.sbatch <cell> <seed> distill|distill-to-rlpd <bonus_scale> [anneal_frac]`
or `submit_supe_online_seeds.sh` with `MODES="none distill-to-rlpd" ALPHAS="10 30" ANNEAL_FRAC=0.5`.
Inside `agents/supe.py` this is the same recipe as `agents/online_crl.py`: a twin-head `E'(s, u)`
regressed onto the trajectory-max empowerment `distill_target` (same estimator run, same cached offline
E values, same annealing `bonus_scale_at`), added to the actor loss as
`alpha log pi - (mean_q Q + bonus_scale(t) * w * E'(s, u))`. Tags mirror the flat sweeps
(`supe_rlpd_ed10`, `supe_rlpd_ann0.5_edrlpd30`, `..._fut`).

## Where the pieces live

* `agents/supe.py` -- the agent (SAC + reward model + RND + optional distill), config in `get_config`.
  Own Adam per parameter group; `update()` = `critic_updates_per_update` scanned critic steps + 1
  actor/alpha step, called `utd_ratio` times per macro row (default k: one per env step).
* `utils/online_rollout.py` -- `MacroCollector` now stores a float latent when the agent has
  `example_skill()`, applies `reward_shift`, passes `env_steps` to `sample_skills` (prior warm-up),
  and writes the same empowerment / distill / `is_offline` fields as flat rows.
* `utils/rlpd.py` -- `make_offline_macro_source` takes `[size, D]` float labels and the extra fields;
  `full_window_mask` keeps only windows inside one trajectory (SUPE `ChunkDataset`).
* `main_online.py` -- macro horizon check after agent creation (k comes from the checkpoint).

## Conventions and deliberate deviations

* **Rewards.** SUPE trains on -1/0 rewards. Our envs give 0/1; the agent applies `reward_shift=-1`
  per env step inside the macro reward, so the env, the success metric and the other baselines are
  unchanged. The `min` relabel is then `-sum_{i<H} gamma_low^i`. This matters: with the learned
  termination mask, offline transitions that reach the goal stop accumulating -1s -- that is the
  optimism of SUPE's relabelling and it vanishes under 0/1 rewards.
* **Budget = the flat CRL baseline's (default).** One update per env step (`utd_ratio = k` per macro
  row), each 1 critic + 1 actor step of 1024 rows, exactly `online_crl`'s batch_size=1024 / utd_ratio=1.
  The paper's schedule (4 calls x 20 critic steps x 256 rows per 4-step macro step, 20x the budget)
  is `UTD_RATIO=4 CRITIC_UPDATES=20 MINIBATCH=256` on the launcher and is not used by default.
* **RND predictor** trains on online rows of a replay minibatch once per update (SUPE: one step on the
  single newest transition per macro step). Reward-model heads train inside the critic scan on each
  minibatch's online rows (same count as SUPE's separate pass).
* **Skill horizon** `K = 4` (SUPE) by default; our other controllers use 10. `CHUNK=10` on the
  pretrain submitter + `K=10` on the online launcher gives the k=10 comparison (tag `_k10`).
* **Low level samples** `a ~ pi(a|s,z)` (`low_temperature=1`, SUPE's behaviour in train and eval);
  our OPAL controller decodes the mode.
