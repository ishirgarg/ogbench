"""Export finished online runs' eval-success curves in the machine-independent tree format.

The format is specified in `scripts/eval_export_format.md`, which this script copies to
`<out_root>/README.md`. Summary of the layout it writes:

    <out_root>/<online env>/steps.npy
    <out_root>/<online env>/<offline dataset>/<method>/<config>.npy + manifest.json

`online env` and `offline dataset` come from each run's own `flags.json`, never from a
sweep-script nickname. `method` is one of the closed set crl / empowerment_distill / rnd,
and all three directories are always created so that "exported, no runs" is distinguishable
from "never exported".

Source layout expected (this repo's online sweeps):

    <exp root>/<cell>/<run_tag>/OGBench/Debug/sd<seed>_s_<jobid>.<timestamp>/{eval,train}.csv

Usage:

    python scripts/export_eval_tree.py <out_root> [cell ...]
    OGBENCH_EXP_ROOT=<dir>[:<dir>...]   experiment trees to scan (default: DEFAULT_EXP_ROOTS)
    --expect-seeds N                    seeds a config must have to count as complete
                                        (default: the most common seed count in its cell)
    --dry-run                           report what would be written, write nothing

INVARIANT 2 (completeness) is the reason this script refuses far more than it writes: a
config is exported only when every one of its seeds exists AND has reached
`total_env_steps`. Partial configs are withheld and listed on stderr, never truncated or
padded, because seeds finish in a biased order and an early snapshot misrepresents them.
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FORMAT_DOC = HERE / 'eval_export_format.md'

# Every experiment tree on this machine that holds online runs in the expected layout.
DEFAULT_EXP_ROOTS = [
    '/nas/ucb/ishirgarg/ogbench/impls/exp/distill_bonus_local_newenvs',
    '/nas/ucb/ishirgarg/ogbench/impls/exp/distill_bonus_cornerenvs',
    '/nas/ucb/ishirgarg/ogbench/impls/exp/distill_bonus_corner_slurm',
    '/nas/ucb/ishirgarg/ogbench/impls/exp/distill_bonus_cube_est2_slurm',
]

METHODS = ('crl', 'empowerment_distill', 'rnd')
METRIC = 'evaluation/success'

# Agent flags that define a config; copied verbatim into the manifest.
CONFIG_FLAGS = (
    'add_explore', 'bonus_scale', 'explore_reward_time_frac', 'distill_target',
    'emp_checkpoint_path', 'emp_num_splus_samples', 'emp_entropy_target',
)


def fmt_num(x):
    """30.0 -> '30', 0.5 -> '0.5' (config names must be stable across float formatting)."""
    f = float(x)
    return str(int(f)) if f == int(f) else repr(f)


def config_name(flags):
    """The format's level-4 grammar, derived from the run's own flags (not its tag)."""
    mode = flags.get('add_explore')
    if mode in (None, 'none', ''):
        return 'crl', 'rlpd_no_bonus'
    if mode in ('distill', 'distill-to-rlpd'):
        name = f"{mode}_alpha{fmt_num(flags['bonus_scale'])}"
        frac = flags.get('explore_reward_time_frac')
        if frac is not None:
            name += f'_anneal{fmt_num(frac)}'
        return 'empowerment_distill', name
    if mode in ('rnd',):
        name = f"alpha{fmt_num(flags['bonus_scale'])}"
        frac = flags.get('explore_reward_time_frac')
        if frac is not None:
            name += f'_anneal{fmt_num(frac)}'
        return 'rnd', name
    raise ValueError(f'unknown add_explore={mode!r}: extend config_name() and the format doc')


def estimator_id(flags):
    """A short handle for the frozen estimator, used only to break config-name collisions."""
    p = flags.get('emp_checkpoint_path')
    if not p:
        return None
    m = re.search(r'sd\d+_s_(\d+)', os.path.basename(p.rstrip('/')))
    return m.group(1) if m else os.path.basename(p.rstrip('/'))[:12]


def read_eval(path):
    rows = [r for r in csv.DictReader(open(path)) if r.get(METRIC) not in (None, '')]
    if not rows:
        return None, None
    steps = np.array([int(float(r['step'])) for r in rows], dtype=np.int64)
    vals = np.array([float(r[METRIC]) for r in rows], dtype=np.float64)
    return steps, vals


def final_train_step(run_dir):
    p = run_dir / 'train.csv'
    if not p.exists():
        return -1.0
    last = None
    for last in csv.DictReader(open(p)):
        pass
    return float(last['step']) if last else -1.0


def collect(exp_roots, only_cells):
    """One record per run directory, keyed later by (env, dataset, method, config, seed)."""
    runs = []
    for root in exp_roots:
        root = Path(root)
        if not root.is_dir():
            print(f'[export] skipping missing exp root {root}', file=sys.stderr)
            continue
        for flags_path in root.glob('*/*/OGBench/Debug/*/flags.json'):
            run_dir = flags_path.parent
            cell = flags_path.relative_to(root).parts[0]
            if only_cells and cell not in only_cells:
                continue
            eval_csv = run_dir / 'eval.csv'
            if not eval_csv.exists():
                continue
            try:
                f = json.load(open(flags_path))
            except json.JSONDecodeError:
                print(f'[export] unreadable flags.json: {run_dir}', file=sys.stderr)
                continue
            agent = f.get('agent', {}) or {}
            m = re.match(r'sd(\d+)_', run_dir.name)
            if not m:
                continue
            steps, vals = read_eval(eval_csv)
            if steps is None:
                continue
            runs.append(dict(
                run_dir=run_dir, cell=cell, exp_root=root,
                env=f['env_name'],
                dataset=f.get('offline_dataset') or 'none',
                run_tag=flags_path.relative_to(root).parts[1],
                seed=int(m.group(1)),
                flags={k: agent.get(k) for k in CONFIG_FLAGS},
                episode_length=f.get('episode_length'),
                total_env_steps=f.get('total_env_steps'),
                steps=steps, vals=vals,
                last_step=final_train_step(run_dir),
            ))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_root')
    ap.add_argument('cells', nargs='*')
    ap.add_argument('--expect-seeds', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    exp_roots = os.environ.get('OGBENCH_EXP_ROOT')
    exp_roots = exp_roots.split(':') if exp_roots else DEFAULT_EXP_ROOTS
    runs = collect(exp_roots, set(args.cells))
    if not runs:
        sys.exit('[export] no runs found')
    print(f'[export] {len(runs)} run dirs across {len(exp_roots)} exp root(s)')

    # (env, dataset, method) -> (config, estimator) -> seed -> run.
    # The estimator is part of the grouping key from the START: two sweeps of the same env that
    # differ only by frozen estimator produce identical config names, and grouping on the name
    # alone would silently interleave their seeds into one array.
    cells = defaultdict(lambda: defaultdict(dict))
    est_ids = defaultdict(set)
    for r in runs:
        method, cfg = config_name(r['flags'])
        key = (r['env'], r['dataset'], method)
        eid = estimator_id(r['flags'])
        ckey = (cfg, eid)
        prev = cells[key][ckey].get(r['seed'])
        # Reruns of the same seed: keep the one that got furthest, tie-broken by mtime, so a
        # crashed retry never shadows a finished run.
        if prev is None or ((r['last_step'], r['run_dir'].stat().st_mtime)
                            > (prev['last_step'], prev['run_dir'].stat().st_mtime)):
            cells[key][ckey][r['seed']] = r
        if method == 'empowerment_distill':
            est_ids[key].add(eid)

    # Flatten (config, estimator) -> display name. Only cells whose runs came from more than one
    # estimator get the _est<id> suffix, so single-estimator envs keep clean names.
    for key in list(cells):
        multi = len(est_ids.get(key, set())) > 1
        flat = {}
        for (cfg, eid), seeds in cells[key].items():
            name = f'{cfg}_est{eid}' if (multi and eid) else cfg
            assert name not in flat, f'config name collision: {key} {name}'
            flat[name] = seeds
        cells[key] = flat
        if multi:
            print(f'[export] {key[0]} / {key[2]}: {len(est_ids[key])} estimators '
                  f'({", ".join(sorted(str(e) for e in est_ids[key]))}) -> config names '
                  f'suffixed with _est<id>', file=sys.stderr)

    # Invariant 2: a config is exported only if every seed exists AND reached total_env_steps.
    exported, withheld = defaultdict(dict), []
    for key, cfgs in cells.items():
        counts = Counter(len(s) for s in cfgs.values())
        expect = args.expect_seeds or (counts.most_common(1)[0][0] if counts else 0)
        for cfg, seeds in cfgs.items():
            target = max((r['total_env_steps'] or 0) for r in seeds.values())
            unfinished = [s for s, r in seeds.items() if r['last_step'] < target - 1]
            if len(seeds) < expect:
                withheld.append((key, cfg, f'{len(seeds)}/{expect} seeds present'))
                continue
            if unfinished:
                withheld.append((key, cfg, f'{len(unfinished)}/{len(seeds)} seeds short of '
                                           f'{int(target):,} steps (seeds {sorted(unfinished)})'))
                continue
            exported[key][cfg] = seeds

    # Invariant 1: one shared eval grid per online env, across every dataset/method/config.
    grids = {}
    for key, cfgs in exported.items():
        for cfg, seeds in cfgs.items():
            for r in seeds.values():
                g = grids.setdefault(r['env'], r['steps'])
                if not np.array_equal(g, r['steps']):
                    sys.exit(f'[export] FATAL: ragged eval grid for {r["env"]}\n'
                             f'  {r["run_dir"]} has {r["steps"].size} evals ending '
                             f'{r["steps"][-1]}, expected {g.size} ending {g[-1]}.\n'
                             f'  Refusing to pad/truncate (invariant 1).')

    print(f'\n[export] WITHHELD {len(withheld)} incomplete config(s) (invariant 2):')
    for key, cfg, why in sorted(withheld, key=lambda x: (x[0][0], x[0][2], x[1])):
        print(f'    {key[0]} | {key[1]} | {key[2]}/{cfg}: {why}')

    out = Path(args.out_root)
    n_arrays = 0
    print(f'\n[export] WRITING to {out}')
    for env in sorted(grids):
        env_cells = {k: v for k, v in exported.items() if k[0] == env and v}
        if not env_cells:
            continue
        if not args.dry_run:
            (out / env).mkdir(parents=True, exist_ok=True)
            np.save(out / env / 'steps.npy', grids[env])
        print(f'  {env}  ({grids[env].size} evals, to {grids[env][-1]:,})')
        for dataset in sorted({k[1] for k in env_cells}):
            for method in METHODS:  # all three, empties included
                key = (env, dataset, method)
                cfgs = exported.get(key, {})
                mdir = out / env / dataset / method
                manifest = {
                    'online_env': env, 'offline_dataset': dataset, 'method': method,
                    'metric': METRIC, 'axes': '(seed, eval step)', 'steps': '../../steps.npy',
                    'configs': {},
                }
                for cfg, seeds in sorted(cfgs.items()):
                    order = sorted(seeds)
                    arr = np.stack([seeds[s]['vals'] for s in order]).astype(np.float64)
                    r0 = seeds[order[0]]
                    manifest['configs'][cfg] = {
                        'run_tag': r0['run_tag'],
                        'seeds': order,
                        'shape': list(arr.shape),
                        'flags': r0['flags'],
                        'episode_length': r0['episode_length'],
                        'total_env_steps': r0['total_env_steps'],
                        'runs': [str(seeds[s]['run_dir'].relative_to(seeds[s]['exp_root']))
                                 for s in order],
                    }
                    if not args.dry_run:
                        mdir.mkdir(parents=True, exist_ok=True)
                        np.save(mdir / f'{cfg}.npy', arr)
                    n_arrays += 1
                    print(f'      {dataset}/{method}/{cfg}.npy  {arr.shape}')
                if not args.dry_run:
                    mdir.mkdir(parents=True, exist_ok=True)
                    json.dump(manifest, open(mdir / 'manifest.json', 'w'), indent=2)
                if not cfgs:
                    print(f'      {dataset}/{method}/  (empty: exported, no complete runs)')

    if not args.dry_run:
        shutil.copyfile(FORMAT_DOC, out / 'README.md')
    print(f'\n[export] {n_arrays} config array(s) written, {len(withheld)} withheld'
          f'{" (DRY RUN, nothing written)" if args.dry_run else ""}')


if __name__ == '__main__':
    main()
