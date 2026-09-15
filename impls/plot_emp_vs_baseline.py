"""Empowerment-entropy-target online CRL vs plain online CRL (both + RLPD), 3 cells.

Baseline = plain agents/online_crl.py + RLPD (no emp_checkpoint_path), same env and
same RLPD dataset as the matching lambda-sweep cell. Empowerment runs = the lambda
sweep (jobs 1219834-1219881, submit_flat_crl_emp_lambda_sweep.sh). 4 seeds each.
Metric: evaluation/success vs env step, mean +/- 95% CI (normal approx, ddof=1) over seeds.
"""
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

EXP_ROOT = '/nas/ucb/ishirgarg/ogbench/impls/exp/OGBench/Debug'
OUT_PNG = '/tmp/claude-20178/-nas-ucb-ishirgarg-ogbench/e194423c-30c3-4b90-8347-cd8f712955c6/scratchpad/emp_vs_baseline_curves.png'

# Ordinal blue ramp (references/palette.md), 4 ordered steps clearing 2:1 on light surface.
LAMBDA_COLORS = {0.3: '#86b6ef', 1.0: '#5598e7', 3.0: '#2a78d6', 10.0: '#184f95'}
BASELINE_COLOR = '#52514e'  # text-secondary gray, dashed

CELLS = [
    dict(key='pmt_sti', title='PointMaze-Teleport-Sparse\n(RLPD: teleport-stitch)',
         env='pointmaze-teleport-sparse-online-v0', offline='pointmaze-teleport-stitch-v0'),
    dict(key='amz_nav', title='AntMaze-Medium-Corner-Sparse\n(RLPD: medium-navigate)',
         env='antmaze-medium-corner-sparse-online-v0', offline='antmaze-medium-navigate-v0'),
    dict(key='asoc_nav', title='AntSoccer-Arena-Corner\n(RLPD: arena-navigate)',
         env='antsoccer-arena-corner-online-v0', offline='antsoccer-arena-navigate-v0'),
]
SEEDS_WANTED = (0, 1, 2, 3)


def scan_runs():
    rows = []
    for d in glob.glob(os.path.join(EXP_ROOT, '*')):
        fp = os.path.join(d, 'flags.json')
        if not os.path.exists(fp):
            continue
        try:
            f = json.load(open(fp))
        except Exception:
            continue
        a = f.get('agent', {})
        if a.get('agent_name') != 'online_crl':
            continue
        rows.append(dict(
            dir=d, env=f.get('env_name'), seed=f.get('seed'),
            offline=f.get('offline_dataset'), emp_lambda=a.get('emp_lambda'),
        ))
    return rows


def load_curve(run_dir):
    p = os.path.join(run_dir, 'eval.csv')
    rows = list(csv.DictReader(open(p)))
    steps = np.array([float(r['step']) for r in rows])
    succ = np.array([float(r['evaluation/success']) for r in rows])
    return steps, succ


def mean_ci(curves):
    """curves: list of (steps, succ) with identical step grids -> (steps, mean, ci95)."""
    steps = curves[0][0]
    for s, _ in curves:
        assert np.array_equal(s, steps), 'step grids differ across seeds'
    mat = np.stack([c[1] for c in curves], axis=0)  # [seeds, T]
    n = mat.shape[0]
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    ci95 = 1.96 * sem
    return steps, mean, ci95


def pick_runs(all_runs, env, offline, emp_lambda, seeds):
    """emp_lambda=None selects the plain baseline (no emp_checkpoint_path -> emp_lambda is None)."""
    out = {}
    for r in all_runs:
        if r['env'] != env or r['offline'] != offline:
            continue
        lam = r['emp_lambda']
        lam = None if lam is None else float(lam)
        if lam != emp_lambda:
            continue
        if r['seed'] in seeds and r['seed'] not in out:
            out[r['seed']] = r['dir']
    missing = set(seeds) - out.keys()
    if missing:
        raise RuntimeError(f'{env}/{offline}/lambda={emp_lambda}: missing seeds {sorted(missing)}')
    return [out[s] for s in seeds]


def main():
    all_runs = scan_runs()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)

    for ax, cell in zip(axes, CELLS):
        base_dirs = pick_runs(all_runs, cell['env'], cell['offline'], None, SEEDS_WANTED)
        base_curve = mean_ci([load_curve(d) for d in base_dirs])
        steps, mean, ci = base_curve
        ax.plot(steps, mean, color=BASELINE_COLOR, linestyle='--', linewidth=2, label='baseline (no empowerment)')
        ax.fill_between(steps, mean - ci, mean + ci, color=BASELINE_COLOR, alpha=0.15, linewidth=0)

        for lam in (0.3, 1.0, 3.0, 10.0):
            dirs = pick_runs(all_runs, cell['env'], cell['offline'], lam, SEEDS_WANTED)
            steps, mean, ci = mean_ci([load_curve(d) for d in dirs])
            color = LAMBDA_COLORS[lam]
            ax.plot(steps, mean, color=color, linewidth=2, label=f'empowerment, $\\lambda$={lam:g}')
            ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.15, linewidth=0)

        ax.set_title(cell['title'], fontsize=10.5)
        ax.set_xlabel('env steps')
        ax.set_xlim(0, 1_000_000)
        ax.set_ylim(-0.02, 1.02)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#e5e4e0', linewidth=0.8, zorder=0)

    axes[0].set_ylabel('eval success rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Empowerment-modulated entropy target vs. plain online CRL (+RLPD), 4 seeds/config, 95% CI', y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches='tight')
    print('saved', OUT_PNG)


if __name__ == '__main__':
    main()
