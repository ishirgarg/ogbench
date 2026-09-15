"""Online skill-controller learning curves on PointMaze-Teleport-navigate,
AntSoccer-Arena-navigate and Cube-Single-play (scripts/slurm/submit_pmt_asoc_cube_tes_seeds.sh
and submit_pmt_asoc_cube_entropy_sweep.sh), comparing:
  * constant (non-legacy) target_entropy_frac in {0.25, 0.5, 0.75}
    (agent.target_entropy_frac; H_target = frac * log(num_skills), use_legacy_entropy=False)
  * TES-SAC target-entropy annealing (agent.use_tes=True; Xu et al. 2021, arXiv:2112.02852):
    H_target starts at log(num_skills) and is multiplied by 0.9 whenever the batch policy
    entropy stabilises at it (tes_patience=500 gradient steps).
DDS and empowerment_skill each get two adjacent rows: success rate, then the TES target
entropy fraction (target_entropy / log(num_skills)) over the same env-step axis, with the
three constant fracs drawn as flat reference lines. 5 seeds per curve, mean +/- 95% CI. A
curve still mid-run (fewer eval/log points than its peers) is plotted as far as it has
gotten and flagged in the legend.
"""

import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ENVS = ["PointMaze-Teleport-navigate", "AntSoccer-Arena-navigate", "Cube-Single-play"]
FAMILIES = ["DDS", "empowerment"]
CKPTS = {
    ("DDS", "PointMaze-Teleport-navigate"): "ckpts/final/dds/pointmaze-teleport-navigate/sd000_s_38394362.0.20260901_181503",
    ("DDS", "AntSoccer-Arena-navigate"): "ckpts/final/dds/antsoccer-arena-navigate/sd000_s_38394360.0.20260901_181457",
    ("DDS", "Cube-Single-play"): "ckpts/final/dds/cube-single-play/sd000_s_38579172.0.20260904_235603",
    ("empowerment", "PointMaze-Teleport-navigate"): "ckpts/final/empowerment_final/pointmaze-teleport-navigate/sd000_s_38390674.0.20260901_154836",
    ("empowerment", "AntSoccer-Arena-navigate"): "ckpts/final/empowerment_final/antsoccer-arena-navigate/sd000_s_38390672.0.20260901_154836",
    ("empowerment", "Cube-Single-play"): "ckpts/final/empowerment_final/cube-single-play/sd000_s_38624008.0.20260908_013305",
}
FRACS = ["0.25", "0.5", "0.75"]
FRAC_COLORS = dict(zip(FRACS, plt.cm.viridis(np.linspace(0.15, 0.75, len(FRACS)))))
TES_COLOR = "crimson"


def _num_skills(ckpt_dir):
    return int(json.load(open(os.path.join(ckpt_dir, "flags.json")))["agent"]["num_skills"])


def _load_csv_curve(csv_path, x_col, y_col):
    xs, ys = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get(y_col, "") == "":
                continue
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def _stack_seeds(rlpd_root, csv_name, x_col, y_col, y_transform=lambda y: y):
    """(x, (n_seeds, n_points) matrix, complete, n_seeds), truncated to the shortest seed."""
    curves = []
    for d in sorted(glob.glob(os.path.join(rlpd_root, "OGBench", "*", "sd*"))):
        csv_path = os.path.join(d, csv_name)
        if not os.path.isfile(csv_path):
            continue
        xs, ys = _load_csv_curve(csv_path, x_col, y_col)
        if len(xs) > 1:  # drop a seed with only the pre-training point
            curves.append((xs, y_transform(ys)))
    if not curves:
        raise FileNotFoundError(f"No usable {csv_name} found under {rlpd_root}")
    n_min = min(len(x) for x, _ in curves)
    ref_x = curves[0][0][:n_min]
    mat = np.stack([y[:n_min] for _, y in curves], axis=0)
    complete = all(len(x) == n_min for x, _ in curves)
    return ref_x, mat, complete, len(curves)


def _plot_mean_ci(ax, x, mat, n, label, color, linewidth=2.0, linestyle="-", zorder=2):
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    ci95 = 1.96 * sem
    ax.plot(x, mean, color=color, linewidth=linewidth, linestyle=linestyle, label=label, zorder=zorder)
    ax.fill_between(x, mean - ci95, mean + ci95, color=color, alpha=0.15, zorder=zorder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tes_subdir", default="online_controller_tes/rlpd")
    parser.add_argument("--sweep_subdir_tmpl", default="online_controller_entropy_sweep/frac{frac}/rlpd")
    parser.add_argument("--output", default="pmt_asoc_cube_tes_curves.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(2 * len(FAMILIES), len(ENVS), figsize=(15, 16), sharex="col")
    for f_idx, family in enumerate(FAMILIES):
        succ_row, frac_row = 2 * f_idx, 2 * f_idx + 1
        for col, env in enumerate(ENVS):
            ckpt = CKPTS[(family, env)]
            num_skills = _num_skills(ckpt)
            log_k = np.log(num_skills)
            print(f"{family} / {env} (num_skills={num_skills}):")

            ax_succ = axes[succ_row, col]
            ax_frac = axes[frac_row, col]

            for frac in FRACS:
                sweep_root = os.path.join(ckpt, args.sweep_subdir_tmpl.format(frac=frac))
                x, mat, complete, n = _stack_seeds(sweep_root, "eval.csv", "step", "evaluation/success")
                tag = "" if complete else " [running]"
                _plot_mean_ci(ax_succ, x, mat, n, f"frac={frac} (n={n}){tag}", FRAC_COLORS[frac])
                print(f"  frac={frac} success: {n} seeds, {mat.shape[1]} eval points" + ("" if complete else " (running)"))
                ax_frac.axhline(float(frac), color=FRAC_COLORS[frac], linewidth=1.2, linestyle=":", zorder=1)

            x, mat, complete, n = _stack_seeds(os.path.join(ckpt, args.tes_subdir), "eval.csv", "step", "evaluation/success")
            tag = "" if complete else " [running]"
            _plot_mean_ci(ax_succ, x, mat, n, f"TES (n={n}){tag}", TES_COLOR, linewidth=2.6, linestyle="--", zorder=3)
            print(f"  TES success: {n} seeds, {mat.shape[1]} eval points" + ("" if complete else " (running)"))

            x, mat, complete, n = _stack_seeds(
                os.path.join(ckpt, args.tes_subdir), "train.csv", "training/env_steps",
                "training/actor/target_entropy", y_transform=lambda y: y / log_k,
            )
            tag = "" if complete else " [running]"
            _plot_mean_ci(ax_frac, x, mat, n, f"TES target/log(K) (n={n}){tag}", TES_COLOR, linewidth=2.0, zorder=3)
            print(f"  TES target_entropy_frac: {n} seeds, {mat.shape[1]} log points" + ("" if complete else " (running)"))

            if succ_row == 0:
                ax_succ.set_title(env, fontsize=11)
            ax_succ.set_ylim(-0.02, 1.02)
            ax_succ.legend(loc="best", fontsize=6.5)
            if col == 0:
                ax_succ.set_ylabel(f"{family}\nEval success rate")
                ax_frac.set_ylabel("target_entropy /\nlog(num_skills)")
            ax_frac.set_ylim(-0.02, 1.05)
            ax_frac.legend(loc="best", fontsize=6.5)
            if frac_row == axes.shape[0] - 1:
                ax_frac.set_xlabel("Online env steps")

    fig.suptitle(
        "Online skill controller: constant target_entropy_frac (0.25/0.5/0.75) vs. TES-SAC annealing\n"
        "(TES: init target = log(num_skills), drop 0.9, patience 500; arXiv:2112.02852). Dotted lines = the "
        "3 constant fracs, for scale. mean ± 95% CI over 5 seeds."
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
