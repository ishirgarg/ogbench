"""Plot the goal-conditioned SUPE runs: mean and 95% CI over seeds per arm.

Seeds log at slightly different env steps (25000 vs 25003), so steps are snapped to the logging grid (eval: 25k,
train: 5k) before averaging, and a point is drawn only once every seed of the arm has reached it. Training metrics
are smoothed with a rolling mean over `TRAIN_SMOOTH` log points per seed. The CI is mean +- t_{0.975, n-1} * sd / sqrt(n).

Usage: python plot_supe_gc.py [out.png]   (env: SUPE_ROOT, SUPE_TAG, TRAIN_SMOOTH)
"""
import glob
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get('SUPE_ROOT', '/home/ishir/ogbench/impls/exp/supe_gc_crlutd_dgx6')
OUT = sys.argv[1] if len(sys.argv) > 1 else 'supe_gc_plots.png'
TRAIN_SMOOTH = int(os.environ.get('TRAIN_SMOOTH', 5))  # 5 x 5k = 25k env steps
CELLS = ['cube_mg', 'amz_all']
GRID = {'eval': 25000, 'train': 5000}
PANELS = [
    ('eval', 'evaluation/success', 'Eval success (100 eps)'),
    ('train', 'training/episode_success', 'Train episode success (smoothed)'),
    ('train', 'training/critic/q_mean', 'Critic Q mean'),
    ('train', 'training/critic/rnd_bonus_offline_mean', 'RND bonus, offline rows'),
]
BASE = os.environ.get('SUPE_TAG', 'supe_gc_rlpd_paperaux_u4c1b1024')
LABELS = {
    BASE: 'SUPE (goal-cond.)',
    f'{BASE}_edrlpd3': '+ distill-to-rlpd α=3',
    f'{BASE}_edrlpd10': '+ distill-to-rlpd α=10',
    f'{BASE}_edrlpd30': '+ distill-to-rlpd α=30',
}
COLORS = dict(zip(LABELS, ['#4c72b0', '#55a868', '#dd8452', '#c44e52']))


def load(tag_dir, kind, key):
    """One Series per seed, indexed by the snapped env step."""
    curves = []
    for run in sorted(glob.glob(f'{tag_dir}/OGBench/Debug/sd*')):
        f = f'{run}/{kind}.csv'
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        if key not in d or len(d) == 0:
            continue
        step = (np.round(d['step'] / GRID[kind]) * GRID[kind]).astype(int)
        s = pd.Series(d[key].values, index=step).groupby(level=0).last()
        if kind == 'train' and TRAIN_SMOOTH > 1:
            s = s.rolling(TRAIN_SMOOTH, min_periods=1).mean()
        curves.append(s)
    return curves


fig, axs = plt.subplots(len(CELLS), len(PANELS), figsize=(4.4 * len(PANELS), 3.5 * len(CELLS)), squeeze=False)
for r, cell in enumerate(CELLS):
    for c, (kind, key, title) in enumerate(PANELS):
        ax = axs[r, c]
        for tag in LABELS:
            curves = load(f'{ROOT}/{cell}/{tag}', kind, key)
            if not curves:
                continue
            df = pd.concat(curves, axis=1).sort_index()
            df = df[df.count(axis=1) == len(curves)]  # only steps every seed has reached
            if df.empty:
                continue
            n = len(curves)
            x = df.index.values / 1e3
            mean = df.mean(axis=1).values
            half = stats.t.ppf(0.975, n - 1) * df.std(axis=1, ddof=1).values / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            ax.plot(x, mean, color=COLORS[tag], lw=1.8, marker='o' if kind == 'eval' else None, ms=3,
                    label=f'{LABELS[tag]} (n={n})')
            ax.fill_between(x, mean - half, mean + half, color=COLORS[tag], alpha=0.2, lw=0)
        ax.set_title(f'{cell}: {title}', fontsize=10)
        ax.set_xlabel('env steps (k)')
        if kind == 'eval':
            ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(fontsize=7, loc='lower right' if cell == 'cube_mg' else 'upper left')
fig.suptitle('Goal-conditioned SUPE (CRL critic budget, paper RM/RND schedule): mean and 95% CI over seeds', fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print(OUT)
