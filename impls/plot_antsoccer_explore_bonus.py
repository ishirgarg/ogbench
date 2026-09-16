"""Exploration reward bonus (agents/online_crl.py `add_explore`) vs plain online CRL + RLPD,
antsoccer-arena-center-online-v0 (RLPD: arena-navigate).

2x2 grid: rows = add_explore mode (reward-to-rlpd: bonus critic fit on online + RLPD rows;
reward: online rows only), cols = explore_reward (empowerment: E(s'); max_episodic_empowerment:
running max of E over the episode through s'). Every panel repeats the same plain-CRL+RLPD
baseline (dashed gray, 5 seeds) against the 6 bonus_scale arms (3 seeds each) of that
(mode, reward) cell. eval success vs env step, mean +/- 95% CI over seeds.

Data: sweep = jobs 1220980-1221195 (submit_flat_crl_explore_bonus_sweep.sh), the asoc_ctr cell,
72/72 runs complete as of 2026-09-16. emp_entropy_target=False in every sweep run, so the only
difference from the baseline is the bonus. Baseline = plain agents/online_crl.py + RLPD, no
emp_checkpoint_path at all (5 seeds, exp/OGBench/Debug, 2026-09-02/03 runs) -- picked as the
most-complete (highest env_steps) run per seed where duplicates exist on disk.
"""
import matplotlib.pyplot as plt

from explore_bonus_plot_utils import pick_baseline_dirs, plot_grid, scan_runs

OUT_PNG = '/nas/ucb/ishirgarg/ogbench/impls/antsoccer_explore_bonus_curves.png'
ENV = 'antsoccer-arena-center-online-v0'
OFFLINE = 'antsoccer-arena-navigate-v0'
SWEEP_SEEDS_WANTED = 3
BASELINE_SEEDS = (0, 1, 2, 3, 4)


def main():
    all_runs = scan_runs(ENV, OFFLINE)
    baseline_dirs = pick_baseline_dirs(all_runs, BASELINE_SEEDS)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    plot_grid(
        fig, axes, all_runs, baseline_dirs, SWEEP_SEEDS_WANTED, BASELINE_SEEDS,
        f'AntSoccer-Arena-Center (RLPD: arena-navigate): exploration reward bonus vs plain CRL+RLPD\n'
        f'mean $\\pm$ 95% CI, {SWEEP_SEEDS_WANTED} seeds/bonus arm, {len(BASELINE_SEEDS)} seeds/baseline',
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.9))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f'wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
