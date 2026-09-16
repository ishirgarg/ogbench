"""Empowerment-entropy-target online CRL vs plain online CRL (both + RLPD), 3 cells.

Row 1: eval success vs env step, baseline + each lambda, mean +/- 95% CI over 4 seeds.
Row 2: per-bin alpha (the auto-tuned SAC temperature) vs env step, lambda=1.0, mean over
       4 seeds -- one line per empowerment quantile bin (light = low-empowerment states,
       dark = high-empowerment states), log y-axis since alpha spans ~1.5 orders of magnitude.
Row 3: the actual verification -- per-bin achieved entropy vs. per-bin TARGET entropy,
       averaged over the last 20 logged training rows (~100k steps, steady state) and over
       seeds, plotted against bin index (0 = lowest empowerment, 7 = highest). If the
       per-state entropy-target mechanism (agents/online_crl.py) is doing what it claims,
       the achieved-entropy line should sit on top of the target-entropy line at every bin,
       and the target itself should increase with the bin (higher empowerment -> higher
       target entropy, emp_lambda=1).

Baseline = plain agents/online_crl.py + RLPD (no emp_checkpoint_path). Empowerment runs =
the lambda sweep (jobs 1219834-1219881, submit_flat_crl_emp_lambda_sweep.sh).
"""
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

EXP_ROOT = '/nas/ucb/ishirgarg/ogbench/impls/exp/OGBench/Debug'
OUT_PNG = '/nas/ucb/ishirgarg/ogbench/impls/emp_vs_baseline_curves.png'

NUM_BINS = 8
# Row 1: ordinal blue ramp for lambda (references/palette.md, 4 ordered steps clearing 2:1 on light).
LAMBDA_COLORS = {0.3: '#86b6ef', 1.0: '#5598e7', 3.0: '#2a78d6', 10.0: '#184f95'}
BASELINE_COLOR = '#52514e'  # text-secondary gray, dashed
# Row 2: ordinal orange ramp for empowerment bin (palette.md: a second simultaneous sequential
# context takes the next categorical slot's hue, orange, as its own one-hue ramp) -- distinct
# from the blue used for lambda in row 1. Sampled light->dark, kept clear of white at the low end.
BIN_CMAP = plt.get_cmap('Oranges')
BIN_COLORS = [BIN_CMAP(0.28 + 0.62 * i / (NUM_BINS - 1)) for i in range(NUM_BINS)]
# Row 3: achieved entropy (categorical slot 1 blue) vs. its target (same gray as the row-1 baseline).
ACHIEVED_COLOR = '#2a78d6'
TARGET_COLOR = '#52514e'

CELLS = [
    dict(key='pmt_sti', title='PointMaze-Teleport-Sparse\n(RLPD: teleport-stitch)',
         env='pointmaze-teleport-sparse-online-v0', offline='pointmaze-teleport-stitch-v0'),
    dict(key='amz_nav', title='AntMaze-Medium-Corner-Sparse\n(RLPD: medium-navigate)',
         env='antmaze-medium-corner-sparse-online-v0', offline='antmaze-medium-navigate-v0'),
    dict(key='asoc_nav', title='AntSoccer-Arena-Corner\n(RLPD: arena-navigate)',
         env='antsoccer-arena-corner-online-v0', offline='antsoccer-arena-navigate-v0'),
]
SEEDS_WANTED = (0, 1, 2, 3)
REPRESENTATIVE_LAMBDA = 1.0  # which lambda arm rows 2-3 diagnose


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


def load_eval_curve(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, 'eval.csv'))))
    steps = np.array([float(r['step']) for r in rows])
    succ = np.array([float(r['evaluation/success']) for r in rows])
    return steps, succ


def load_train_rows(run_dir):
    return list(csv.DictReader(open(os.path.join(run_dir, 'train.csv'))))


def mean_ci(curves):
    """curves: list of (steps, y) with identical step grids -> (steps, mean, ci95)."""
    steps = curves[0][0]
    for s, _ in curves:
        assert np.array_equal(s, steps), 'step grids differ across seeds'
    mat = np.stack([c[1] for c in curves], axis=0)  # [seeds, T]
    n = mat.shape[0]
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return steps, mean, 1.96 * sem


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


def bin_alpha_over_steps(run_dirs):
    """[T] steps, [seeds, T, NUM_BINS] alpha -- reads training/actor/alpha_bin{i} per row."""
    steps = None
    mats = []
    for d in run_dirs:
        rows = load_train_rows(d)
        s = np.array([float(r['training/env_steps']) for r in rows])
        if steps is None:
            steps = s
        assert np.array_equal(s, steps), f'{d}: train.csv step grid differs across seeds'
        mats.append(np.array([[float(r[f'training/actor/alpha_bin{i}']) for i in range(NUM_BINS)] for r in rows]))
    return steps, np.stack(mats, axis=0)


def bin_entropy_vs_target(run_dirs, tail=20):
    """Mean over seeds and the last `tail` logged rows -> ([NUM_BINS] achieved, [NUM_BINS] target)."""
    achieved, target = [], []
    for d in run_dirs:
        rows = load_train_rows(d)[-tail:]
        achieved.append(np.mean([[float(r[f'training/actor/entropy_bin{i}']) for i in range(NUM_BINS)] for r in rows], axis=0))
        target.append(np.mean([[float(r[f'training/actor/target_entropy_bin{i}']) for i in range(NUM_BINS)] for r in rows], axis=0))
    return np.mean(achieved, axis=0), np.mean(target, axis=0)


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#e5e4e0', linewidth=0.8, zorder=0)


def main():
    all_runs = scan_runs()
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    for col, cell in enumerate(CELLS):
        ax = axes[0, col]
        base_dirs = pick_runs(all_runs, cell['env'], cell['offline'], None, SEEDS_WANTED)
        steps, mean, ci = mean_ci([load_eval_curve(d) for d in base_dirs])
        ax.plot(steps, mean, color=BASELINE_COLOR, linestyle='--', linewidth=2, label='baseline (no empowerment)')
        ax.fill_between(steps, mean - ci, mean + ci, color=BASELINE_COLOR, alpha=0.15, linewidth=0)
        for lam in (0.3, 1.0, 3.0, 10.0):
            dirs = pick_runs(all_runs, cell['env'], cell['offline'], lam, SEEDS_WANTED)
            steps, mean, ci = mean_ci([load_eval_curve(d) for d in dirs])
            color = LAMBDA_COLORS[lam]
            ax.plot(steps, mean, color=color, linewidth=2, label=f'empowerment, $\\lambda$={lam:g}')
            ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.15, linewidth=0)
        ax.set_title(cell['title'], fontsize=10.5)
        ax.set_xlim(0, 1_000_000)
        ax.set_ylim(-0.02, 1.02)
        style_ax(ax)
        if col == 0:
            ax.set_ylabel('eval success rate')

        emp_dirs = pick_runs(all_runs, cell['env'], cell['offline'], REPRESENTATIVE_LAMBDA, SEEDS_WANTED)

        ax = axes[1, col]
        steps, alpha = bin_alpha_over_steps(emp_dirs)  # [T], [seeds, T, NUM_BINS]
        alpha_mean = alpha.mean(axis=0)  # [T, NUM_BINS]
        for i in range(NUM_BINS):
            ax.plot(steps, np.clip(alpha_mean[:, i], 1e-4, None), color=BIN_COLORS[i], linewidth=1.6,
                     label=f'bin {i}' if col == 0 else None)
        ax.set_yscale('log')
        ax.set_xlim(0, 1_000_000)
        style_ax(ax)
        if col == 0:
            ax.set_ylabel(r'$\alpha$ (temperature), log scale')

        ax = axes[2, col]
        achieved, target = bin_entropy_vs_target(emp_dirs)
        bins = np.arange(NUM_BINS)
        ax.plot(bins, target, color=TARGET_COLOR, linestyle='--', linewidth=2, marker='o', markersize=5,
                 label='target entropy $H_{\\mathrm{target}}(s)$' if col == 0 else None)
        ax.plot(bins, achieved, color=ACHIEVED_COLOR, linewidth=2, marker='o', markersize=5,
                 label='achieved entropy (policy)' if col == 0 else None)
        ax.set_xticks(bins)
        ax.set_xlabel('empowerment bin (0 = lowest E, 7 = highest E)')
        style_ax(ax)
        if col == 0:
            ax.set_ylabel('entropy (nats),\nlast ~100k steps')

    axes[0, 1].set_xlabel('env steps')
    axes[1, 1].set_xlabel('env steps')

    # Each row's legend is anchored to its own middle axis (axes-fraction coords), well below
    # that axis's own xlabel, rather than a guessed figure-fraction placement -- robust to layout changes.
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(h0, l0, loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=5, frameon=False, fontsize=9)
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    axes[1, 1].legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, -0.34), ncol=8, frameon=False, fontsize=8,
                       title='empowerment bin (light = low E → dark = high E)', title_fontsize=9)
    h2, l2 = axes[2, 0].get_legend_handles_labels()
    axes[2, 1].legend(h2, l2, loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False, fontsize=9)

    fig.text(0.5, 1.135, 'Empowerment-modulated entropy target for online contrastive RL',
              ha='center', fontsize=14, fontweight='bold')
    algo_desc = (
        'Algorithm (agents/online_crl.py): a frozen offline empowerment estimator E(s) '
        r'$\approx$ I(S$^+$;A|s) (empowerment_skill checkpoint, 64 future samples/state) sets a per-state'
        '\n'
        r'SAC entropy target $H_{\mathrm{target}}(s) = H_{\mathrm{base}} + \lambda\,(E(s) - \bar{E})$, '
        'auto-tuned with one temperature $\\alpha$ per empowerment quantile bin. The term lives only in\n'
        'the actor loss -- never bootstrapped into the (purely contrastive) critic -- so '
        'high-empowerment states get more exploration entropy without any pull back toward them at eval time.'
    )
    fig.text(0.5, 1.075, algo_desc, ha='center', va='top', fontsize=10)
    fig.text(
        0.5, 1.02,
        f'Row 1: this vs. plain online CRL (+RLPD), 4 seeds/config, 95% CI  |  '
        f'Rows 2-3: verification at $\\lambda$={REPRESENTATIVE_LAMBDA:g} (mean over 4 seeds)',
        ha='center', fontsize=11,
    )
    fig.subplots_adjust(hspace=0.75, wspace=0.28)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches='tight')
    print('saved', OUT_PNG)


if __name__ == '__main__':
    main()
