"""Exploration reward bonus (agents/online_crl.py `add_explore`) vs plain online CRL + RLPD,
pointmaze-teleport-center-online-v0 (RLPD: teleport-navigate).

Same 2x2 grid as plot_antsoccer_explore_bonus.py. The sweep (jobs 1220980-1221195, pmt_nav
cell) is still running as of 2026-09-16: only seed 0 has finished for 23/24 (mode, reward,
scale) configs (reward / max_episodic_empowerment / 0.01 has none yet), so every finished bonus
arm here is a SINGLE seed -- drawn as a thin line with no confidence band, vs. the 5-seed
baseline's shaded CI. Re-run this script once seeds 1-2 land for real per-arm CIs.
"""
import matplotlib.pyplot as plt

from explore_bonus_plot_utils import pick_baseline_dirs, plot_grid, scan_runs

OUT_PNG = '/nas/ucb/ishirgarg/ogbench/impls/pointmaze_explore_bonus_curves.png'
ENV = 'pointmaze-teleport-center-online-v0'
OFFLINE = 'pointmaze-teleport-navigate-v0'
SWEEP_SEEDS_WANTED = 3  # cap; pick_sweep_dirs uses however many (<=this) have actually finished
BASELINE_SEEDS = (0, 1, 2, 3, 4)


def main():
    all_runs = scan_runs(ENV, OFFLINE)
    baseline_dirs = pick_baseline_dirs(all_runs, BASELINE_SEEDS)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    plot_grid(
        fig, axes, all_runs, baseline_dirs, SWEEP_SEEDS_WANTED, BASELINE_SEEDS,
        f'PointMaze-Teleport-Center (RLPD: teleport-navigate): exploration reward bonus vs plain CRL+RLPD\n'
        f'baseline mean $\\pm$ 95% CI ({len(BASELINE_SEEDS)} seeds); bonus arms are a SINGLE seed each (thin line, '
        'no band), sweep still running',
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.9))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f'wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
