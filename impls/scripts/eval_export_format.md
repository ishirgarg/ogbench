# Eval-curve export format

A machine-independent, self-describing layout for **evaluation success curves**, so that
runs produced on different machines (dgx6, rnn, Savio, BRC) can be pooled, diffed and
plotted without anyone re-reading `eval.csv` files or re-deriving what a run tag meant.

Reference implementation: `impls/scripts/export_eval_tree.py`.
An export produced by it lives on dgx6 at `~/Desktop/ogbench_eval_curves`.

---

## 1. Directory layout

```
<export root>/
    README.md                              a copy of this document
    <online env>/                          e.g. antmaze-medium-corner-sparse-online-v0
        steps.npy                          int64 (n_evals,)
        <offline dataset>/                 e.g. antmaze-medium-navigate-v0
            <method>/                      crl | empowerment_distill | rnd
                <config>.npy               float64 (n_seeds, n_evals)
                manifest.json
```

Four levels, in this order, each one a directory: **online env → offline dataset →
method → config**. The config is a *file*, not a directory: one `.npy` holding all of
that config's seeds.

### Level 1 — online env

The full registered gym id of the online task, including the `-online-v0` suffix, exactly
as passed to `main_online.py --env_name`. Example:
`pointmaze-teleport-corner-all-squares-online-v0`.

Use the id, not a nickname: `amz_sparse` and `pmt_all` are sweep-script shorthand and do
not survive contact with another machine.

### Level 2 — offline dataset

The OGBench dataset mixed into training (RLPD's offline buffer), exactly as passed to
`--offline_dataset`, including the `-v0` suffix. This is the level that distinguishes the
`navigate` / `stitch` / `explore` / `noisy` variants of the same task, so **it is never
collapsed even when only one variant exists**.

Examples: `antmaze-medium-navigate-v0`, `antmaze-medium-stitch-v0`,
`antmaze-medium-explore-v0`, `cube-single-noisy-v0`.

A run with no offline dataset (pure online, no RLPD) uses the literal directory name
`none`.

### Level 3 — method

One of a **closed set**, so that a consumer can iterate methods without string matching:

| directory | meaning |
|---|---|
| `crl` | plain online CRL (+ RLPD if an offline dataset is present), no exploration bonus |
| `empowerment_distill` | distilled empowerment bonus: `add_explore=distill` or `distill-to-rlpd` |
| `rnd` | RND exploration bonus |

Every method directory is created **even when it holds no runs** — an empty method
directory with only a `manifest.json` (`"configs": {}`) means *"this cell was exported and
this method has no runs here"*, which is different from a missing directory meaning *"this
cell was never exported"*. Consumers rely on that distinction; do not prune empties.

Adding a method means adding a row to the table above, not inventing a directory name
locally.

### Level 4 — config

`<config>.npy`, where the config name spells out the sweep hyperparameters in a fixed
grammar. Lowercase, `_` between fields, no spaces:

```
crl                   rlpd_no_bonus                        the no-bonus reference run
empowerment_distill   <mode>_alpha<A>[_anneal<F>]          mode = distill | distill-to-rlpd
                                                           A    = bonus_scale, e.g. 10, 30
                                                           F    = explore_reward_time_frac
rnd                   alpha<A>[_anneal<F>]                 same convention, RND's own scale
```

Examples: `distill_alpha10.npy`, `distill-to-rlpd_alpha30_anneal0.5.npy`,
`rlpd_no_bonus.npy`.

Omit a field when the run does not use it — a constant-alpha run has no `_anneal` suffix
rather than `_anneal0`. If a sweep varies a hyperparameter not covered above, append it in
the same style (`_k50`, `_bc0.001`) and record it in the manifest; the manifest is the
authority, the filename is the human-readable handle.

---

## 2. Array semantics

**`<config>.npy`** — `float64`, shape `(n_seeds, n_evals)`, the value of
`evaluation/success` (the fraction of eval episodes that reached the goal, so in `[0, 1]`).

- **Rows are seeds, in ascending seed order** (`0, 1, 2, 3, 4` for a 5-seed sweep). The
  actual seed values are listed in the manifest; do not assume they start at 0.
- **Columns are eval points, aligned to `steps.npy`** — the same column index means the
  same env step in every array under that online env.
- No aggregation is stored. Mean, s.e., smoothing and tail-averaging are the consumer's
  job, computed from the raw seeds.

**`steps.npy`** — `int64`, shape `(n_evals,)`, env steps at which evals were run, ascending.
Column 0 is typically the step-0 eval of the untrained policy; drop it when plotting if
you do not want it.

It lives at the **online-env level** because every run of a cell shares one grid. The
exporter asserts this and fails rather than emitting ragged arrays.

### The two hard invariants

> **1. Alignment.** Every `.npy` under an online env has the same `n_evals` as that env's
> `steps.npy`, and column `j` of any array corresponds to `steps[j]`.

A partially finished run therefore **cannot** be exported into a finished cell's tree.
Export a cell only when its runs have all reached the same eval count, or export the
common prefix for every run in the cell (and say so in the manifest's `total_env_steps`).

> **2. Completeness.** A config is exported only when **all of its seeds have finished** --
> every seed of the sweep is present, and each has run to `total_env_steps`.

A config with 3 of 5 seeds done, or with 5 seeds that are all still mid-run, is **not
written at all**. It is not written as a short array, not written with the missing seeds
padded or dropped, and not written with a "partial" marker.

The reason is that a partial config is worse than a missing one: seeds finish in a
different order than they were launched, so an early snapshot is a biased sample of the
seed population (the fast seeds are often the ones that collapsed early), and any mean or
error bar taken from it silently misrepresents the config. A consumer that finds a config
present is entitled to assume it is final and comparable to every other config in the
tree.

Incomplete configs are simply skipped, and the producer says which ones it withheld and
why. A cell with a mix of finished and unfinished configs therefore exports the finished
ones and is re-exported later; re-exporting a cell overwrites it in place, so the tree
converges as runs land.

---

## 3. `manifest.json`

One per method directory. It makes each array self-describing, so a config name never has
to be reverse-engineered:

```json
{
  "online_env": "pointmaze-teleport-corner-all-squares-online-v0",
  "offline_dataset": "pointmaze-teleport-navigate-v0",
  "method": "empowerment_distill",
  "metric": "evaluation/success",
  "axes": "(seed, eval step)",
  "steps": "../../steps.npy",
  "configs": {
    "distill_alpha10_anneal0.5": {
      "run_tag": "rlpd_noent_ann0.5_ed10",
      "seeds": [0, 1, 2, 3, 4],
      "shape": [5, 41],
      "flags": {
        "add_explore": "distill",
        "bonus_scale": 10.0,
        "explore_reward_time_frac": 0.5,
        "distill_target": "episode_max",
        "emp_checkpoint_path": "/home/ishir/ogbench/impls/ckpts/final/empowerment/...",
        "emp_num_splus_samples": 64,
        "emp_entropy_target": false
      },
      "episode_length": null,
      "total_env_steps": 1000000,
      "runs": ["pointmaze-teleport-corner-all-squares/rlpd_noent_ann0.5_ed10/OGBench/Debug/sd000_..."]
    }
  }
}
```

Required per config: `run_tag` (the original sweep tag, so results trace back to the
launching script), `seeds` (row order), `shape`, `flags` (the agent flags that define the
config, taken from the run's own `flags.json` — not retyped by hand), `episode_length`
(`null` = the env's registered horizon) and `total_env_steps`.

`runs` (source run directories) is optional but recommended; paths are machine-local and
are provenance only — never load through them.

---

## 4. Reading an export

```python
import json, numpy as np, pathlib

root = pathlib.Path('ogbench_eval_curves')
env  = root / 'pointmaze-teleport-corner-all-squares-online-v0'
steps = np.load(env / 'steps.npy')                                   # (n_evals,)

curves = np.load(env / 'pointmaze-teleport-navigate-v0' /
                 'empowerment_distill' / 'distill_alpha10_anneal0.5.npy')   # (5, n_evals)
mean = curves.mean(0)
sem  = curves.std(0, ddof=1) / np.sqrt(curves.shape[0])

# Walk everything, method by method.
for env_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    steps = np.load(env_dir / 'steps.npy')
    for ds_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
        for method in ('crl', 'empowerment_distill', 'rnd'):
            mdir = ds_dir / method
            if not mdir.exists():
                continue
            man = json.load(open(mdir / 'manifest.json'))
            for config in sorted(man['configs']):
                arr = np.load(mdir / f'{config}.npy')
                assert arr.shape[1] == steps.size
```

## 5. Writing an export

Use `impls/scripts/export_eval_tree.py` where the runs follow this repo's
`<save_root>/<cell>/<run_tag>/OGBench/Debug/sd<seed>_s_<jobid>.<ts>/eval.csv` layout:

```bash
python scripts/export_eval_tree.py <out_root> [cell ...]
# OGBENCH_EXP_ROOT=<save_root> selects the experiment tree (default: this machine's).
```

If a machine's runs are laid out differently, write your own producer — the format is the
contract, not the script. It must:

1. derive `online env` and `offline dataset` from each run's own `flags.json`
   (`env_name`, `offline_dataset`), never from a script-local nickname table;
2. assert one shared eval grid per online env, and fail loudly instead of padding,
   truncating or interpolating;
3. skip any config whose seeds are not all finished, and report each skip (invariant 2);
4. order rows by ascending seed and record that order in the manifest;
5. emit all three method directories, empty ones included;
6. copy this document to `<export root>/README.md`.

Merging two machines' exports is a plain directory merge: the paths are content-addressed
by env / dataset / method / config, so distinct cells never collide. Two exports of the
*same* cell do collide — that is intended, and the newer, more complete one wins.
