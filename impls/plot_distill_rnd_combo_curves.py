"""Learning curves for the distill-to-rlpd + RND combo sweep (scripts/run_distill_rnd_local_sweep.sh).

One panel per env; solid lines = the two (alpha_distill, beta_rnd) combo configs, dashed = reference
arms from the earlier distill-only sweeps on the same env / dataset / estimator (no-bonus RLPD
baseline and the constant-alpha distill-to-rlpd arm at the same or nearest alpha). Mean +- 95% CI
over seeds at every eval step (n = the seeds that have reached that step; unfinished runs simply
end early). Usage: python plot_distill_rnd_combo_curves.py [--out PNG]
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXP = '/nas/ucb/ishirgarg/ogbench/impls/exp'
COMBO = f'{EXP}/distill_rnd_combo'
BLUE, ORANGE, AQUA, YELLOW, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6b6b66'

# env -> (title, [(label, glob of run dirs, color, style)])
CELLS = {
    'cube-single-multigoal': ('cube-single-multigoal (est. 38579169)', [
        ('distill 10 + RND 1', f'{COMBO}/cube-single-multigoal/rlpd_noent_edrlpd10_rnd1/OGBench/Debug/sd*', BLUE, '-'),
        ('distill 5 + RND 0.5', f'{COMBO}/cube-single-multigoal/rlpd_noent_edrlpd5_rnd0.5/OGBench/Debug/sd*', ORANGE, '-'),
        ('distill 10 only (ref)', f'{EXP}/distill_bonus_cube_est2_slurm/cube-single-multigoal/rlpd_noent_edrlpd10/OGBench/Debug/sd*', AQUA, '--'),
        ('RLPD baseline (ref)', f'{EXP}/distill_bonus_cube_est2_slurm/cube-single-multigoal/rlpd_baseline/OGBench/Debug/sd*', GRAY, '--'),
    ]),
    'antmaze-medium-corner-all-squares': ('antmaze-medium-corner-all-squares', [
        ('distill 30 + RND 3', f'{COMBO}/antmaze-medium-corner-all-squares/rlpd_noent_edrlpd30_rnd3/OGBench/Debug/sd*', BLUE, '-'),
        ('distill 15 + RND 1.5', f'{COMBO}/antmaze-medium-corner-all-squares/rlpd_noent_edrlpd15_rnd1.5/OGBench/Debug/sd*', ORANGE, '-'),
        ('distill 30 only (ref)', f'{EXP}/distill_bonus_cornerenvs/antmaze-medium-corner-all-squares/rlpd_noent_edrlpd30/OGBench/Debug/sd*', AQUA, '--'),
        ('distill 10 only (ref)', f'{EXP}/distill_bonus_cornerenvs/antmaze-medium-corner-all-squares/rlpd_noent_edrlpd10/OGBench/Debug/sd*', YELLOW, '--'),
        ('RLPD baseline (ref)', f'{EXP}/distill_bonus_cornerenvs/antmaze-medium-corner-all-squares/rlpd_baseline/OGBench/Debug/sd*', GRAY, '--'),
    ]),
    'pointmaze-teleport-corner-all-squares': ('pointmaze-teleport-corner-all-squares (distill annealed to 0 by 500k)', [
        ('ann distill 30 + RND 1', f'{COMBO}/pointmaze-teleport-corner-all-squares/rlpd_noent_ann0.5_edrlpd30_rnd1/OGBench/Debug/sd*', BLUE, '-'),
        ('ann distill 15 + RND 0.5', f'{COMBO}/pointmaze-teleport-corner-all-squares/rlpd_noent_ann0.5_edrlpd15_rnd0.5/OGBench/Debug/sd*', ORANGE, '-'),
        ('ann distill 30 + RND 1, KILLED run', f'{COMBO}/pointmaze-teleport-corner-all-squares/rlpd_noent_ann0.5_edrlpd30_rnd1/OGBench/Debug/_killed_*/sd*', BLUE, ':'),
        ('ann distill 15 + RND 0.5, KILLED run', f'{COMBO}/pointmaze-teleport-corner-all-squares/rlpd_noent_ann0.5_edrlpd15_rnd0.5/OGBench/Debug/_killed_*/sd*', ORANGE, ':'),
    ]),
}


def load_curves(pattern):
    curves = []
    for d in sorted(glob.glob(pattern)):
        f = os.path.join(d, 'eval.csv')
        if not os.path.exists(f):
            continue
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        s = np.array([float(r['step']) for r in rows])
        y = np.array([float(r['evaluation/success']) for r in rows])
        curves.append((s, y))
    return curves


def aggregate(curves):
    steps = sorted({float(s) for c in curves for s in c[0]})
    mean, ci, n = [], [], []
    for st in steps:
        vals = [y[np.where(s == st)[0][0]] for s, y in curves if st in s]
        vals = np.array(vals)
        mean.append(vals.mean())
        ci.append(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        n.append(len(vals))
    return np.array(steps), np.array(mean), np.array(ci), np.array(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=f'{COMBO}/distill_rnd_combo_curves.png')
    args = p.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (env, (title, arms)) in zip(axes, CELLS.items()):
        for label, pattern, color, style in arms:
            curves = load_curves(pattern)
            if not curves:
                continue
            steps, mean, ci, n = aggregate(curves)
            last = int(steps[-1] / 1000)
            ax.plot(steps, mean, color=color, ls=style, lw=2.0, label=f'{label}  [n={len(curves)}, to {last}k]')
            if style == '-' or style == ':':
                ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.15, lw=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('online env steps')
        ax.set_xlim(0, 1_000_000)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper left')
    axes[0].set_ylabel('eval success (100 episodes)')
    fig.suptitle('distill-to-rlpd empowerment bonus + independent RND bonus, RLPD on -- mean +- 95% CI over seeds (references dashed, killed pointmaze runs dotted)', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
