"""Summarise scripts/run_distill_grad_balance_test.sh: the bonus_scale that equalises the actor gradients.

Reads every run dir `sd*_s_dgb-<cell>_<mode>_a<scale>.*` under --root, prints per run the
equalising scale (ratio of the MEAN gradient norms over an env-step window, which is steadier
than the mean of per-update ratios), the E' fit quality and the eval success, and saves a
metric-vs-env-step figure (one column per cell).

    python scripts/summarize_distill_grad_balance.py [--root .../exp/distill_grad_test] [--out fig.png]
"""

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN_RE = re.compile(r'sd\d+_s_dgb-(?P<cell>[a-z]+_[a-z]+)_(?P<mode>distill(?:-to-rlpd)?)_a(?P<scale>[0-9.e-]+?)\.\d{8}_\d{6}$')
T = 'training/'


def read_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) if r[key] not in ('', None) else np.nan for r in rows])
        except ValueError:
            pass
    return out


def window_ratio(train, num, den, lo, hi):
    step = train['step']
    sel = (step > lo) & (step <= hi)
    if not sel.any():
        return np.nan
    return float(np.nanmean(train[num][sel]) / max(np.nanmean(train[den][sel]), 1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/nas/ucb/ishirgarg/ogbench/impls/exp/distill_grad_test')
    parser.add_argument('--out', default=None)
    parser.add_argument('--skip_steps', type=int, default=10000, help='Env steps dropped from the headline ratio.')
    args = parser.parse_args()

    runs = []
    for run_dir in sorted(glob.glob(os.path.join(args.root, 'OGBench', '*', 'sd*_s_dgb-*'))):
        match = RUN_RE.search(os.path.basename(run_dir))
        if match is None:
            continue
        train = read_csv(os.path.join(run_dir, 'train.csv'))
        if 'step' not in train:
            continue
        runs.append(dict(match.groupdict(), train=train, eval=read_csv(os.path.join(run_dir, 'eval.csv')), dir=run_dir))
    if not runs:
        raise SystemExit(f'no dgb-* runs with a train.csv under {args.root}')

    header = (
        f"{'cell':9s} {'mode':16s} {'scale':>6s} {'steps':>6s} | {'alpha_eq(param)':>15s} {'first/last third':>17s} "
        f"{'alpha_eq(action)':>16s} | {'|g_crl|':>9s} {'|g_bonus|':>9s} {'|g_ent|':>9s} {'cos':>6s} | "
        f"{'E_expl_var':>10s} {'tgt_std':>8s} | eval success by step"
    )
    print(header)
    print('-' * len(header))
    for run in runs:
        tr = run['train']
        last = float(tr['step'][-1])
        lo = min(args.skip_steps, last / 2)
        third = (last - lo) / 3
        p, b, e = T + 'bonus_grad/param_norm_crl', T + 'bonus_grad/param_norm_bonus', T + 'bonus_grad/param_norm_entropy'
        sel = tr['step'] > lo
        ev = run['eval']
        success = ''
        if 'evaluation/success' in ev:
            success = ' '.join(f"{int(s) // 1000}k:{v:.2f}" for s, v in zip(ev['step'], ev['evaluation/success']))
        print(
            f"{run['cell']:9s} {run['mode']:16s} {run['scale']:>6s} {int(last):6d} | "
            f"{window_ratio(tr, p, b, lo, last):15.1f} "
            f"{window_ratio(tr, p, b, lo, lo + third):8.1f}/{window_ratio(tr, p, b, last - third, last):<8.1f} "
            f"{window_ratio(tr, T + 'bonus_grad/action_norm_crl', T + 'bonus_grad/action_norm_bonus', lo, last):16.1f} | "
            f"{np.nanmean(tr[p][sel]):9.3g} {np.nanmean(tr[b][sel]):9.3g} {np.nanmean(tr[e][sel]):9.3g} "
            f"{np.nanmean(tr[T + 'bonus_grad/cos_crl_bonus'][sel]):6.2f} | "
            f"{np.nanmean(tr[T + 'distill/explained_variance'][-3:]):10.2f} "
            f"{np.nanmean(tr[T + 'distill/target_std'][-3:]):8.3f} | {success}"
        )

    # Curves: one column per cell, rows = equalising scale, the two gradient norms, E' fit, train success.
    cells = sorted({r['cell'] for r in runs})
    panels = [
        ('alpha that equalises actor grads\n||g_crl|| / ||g_bonus(scale=1)||', None),
        ('actor-param grad norm\n(solid CRL, dashed bonus at scale 1)', None),
        ("E' explained variance", T + 'distill/explained_variance'),
        ('training episode success', T + 'episode_success'),
    ]
    fig, axes = plt.subplots(len(panels), len(cells), figsize=(5.2 * len(cells), 3.1 * len(panels)), squeeze=False)
    colors = {'distill': 'tab:blue', 'distill-to-rlpd': 'tab:orange'}
    for col, cell in enumerate(cells):
        for run in (r for r in runs if r['cell'] == cell):
            tr = run['train']
            style = dict(color=colors[run['mode']], alpha=0.9, lw=1.6, ls='-' if float(run['scale']) == 0 else ':')
            label = f"{run['mode']} a={run['scale']}"
            step = tr['step']
            axes[0, col].plot(step, tr[T + 'bonus_grad/scale_equal_param'], label=label, **style)
            axes[1, col].plot(step, tr[T + 'bonus_grad/param_norm_crl'], **{**style, 'ls': '-'}, label=label)
            axes[1, col].plot(step, tr[T + 'bonus_grad/param_norm_bonus'], **{**style, 'ls': '--'})
            for row in (2, 3):
                key = panels[row][1]
                if key in tr:
                    axes[row, col].plot(step, tr[key], label=label, **style)
        axes[0, col].set_yscale('log')
        axes[1, col].set_yscale('log')
        axes[2, col].set_ylim(-0.5, 1.02)
        axes[0, col].set_title(cell)
        for row, (title, _) in enumerate(panels):
            axes[row, col].set_ylabel(title, fontsize=8)
            axes[row, col].grid(alpha=0.3)
        axes[-1, col].set_xlabel('env steps')
        axes[0, col].legend(fontsize=7)
    fig.tight_layout()
    out = args.out or os.path.join(args.root, 'distill_grad_balance.png')
    fig.savefig(out, dpi=130)
    print(f'\nsaved {out}')

    # Performance curves: eval success and (smoothed) training-episode success, one column per cell.
    # Colour = bonus_scale (alpha), line style = mode; a=0 is the no-bonus baseline (actor ignores E').
    scales = sorted({float(r['scale']) for r in runs})
    cmap = plt.get_cmap('viridis')
    scale_color = {sc: ('black' if sc == 0 else cmap(0.15 + 0.75 * i / max(len(scales) - 1, 1))) for i, sc in enumerate(scales)}
    mode_ls = {'distill': '--', 'distill-to-rlpd': '-'}
    fig, axes = plt.subplots(2, len(cells), figsize=(5.6 * len(cells), 7.0), squeeze=False)
    for col, cell in enumerate(cells):
        for run in sorted((r for r in runs if r['cell'] == cell), key=lambda r: (float(r['scale']), r['mode'])):
            style = dict(color=scale_color[float(run['scale'])], ls=mode_ls[run['mode']], lw=1.7, alpha=0.9)
            label = f"a={run['scale']} {run['mode']}"
            ev = run['eval']
            if 'evaluation/success' in ev:
                axes[0, col].plot(ev['step'], ev['evaluation/success'], marker='o', ms=4, label=label, **style)
            tr = run['train']
            key = T + 'episode_success'
            if key in tr:
                y = tr[key]
                k = 5  # log points (2.5k env steps each) in the trailing mean
                smooth = np.array([np.nanmean(y[max(0, i - k + 1) : i + 1]) for i in range(len(y))])
                axes[1, col].plot(tr['step'], smooth, label=label, **style)
        axes[0, col].set_title(cell)
        axes[0, col].set_ylabel('eval success (20 episodes, 1 seed)')
        axes[1, col].set_ylabel('training episode success\n(trailing mean over 12.5k env steps)')
        axes[1, col].set_xlabel('env steps')
        for row in range(2):
            axes[row, col].set_ylim(-0.03, 1.0)
            axes[row, col].grid(alpha=0.3)
        axes[0, col].legend(fontsize=7, ncol=2)
    fig.suptitle("distilled empowerment bonus: solid = distill-to-rlpd, dashed = distill (online only); black = a=0 baseline")
    fig.tight_layout()
    perf_out = os.path.splitext(out)[0] + '_performance.png'
    fig.savefig(perf_out, dpi=130)
    print(f'saved {perf_out}')


if __name__ == '__main__':
    main()
