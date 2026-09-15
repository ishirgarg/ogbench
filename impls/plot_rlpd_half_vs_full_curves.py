"""Online skill-controller learning curves comparing RLPD-for-half-of-training
(scripts/slurm/submit_pmt_asoc_cube_rlpd_half_entropy_sweep.sh, --rlpd_frac_time=0.5)
against RLPD-for-the-whole-run baselines that use the SAME entropy setup:
  * const0.5 : agent.target_entropy_frac=0.5, use_tes=False
      half-run  -> online_controller_rlpd_half/const0.5/rlpd_ft0.5
      full-run  -> online_controller_entropy_sweep/frac0.5/rlpd  (submit_pmt_asoc_cube_entropy_sweep.sh)
  * tes       : agent.use_tes=True (Xu et al. 2021, arXiv:2112.02852)
      half-run  -> online_controller_rlpd_half/tes/rlpd_ft0.5
      full-run  -> online_controller_tes/rlpd                   (submit_pmt_asoc_cube_tes_seeds.sh)
Also overlays, where available, a pre-existing no-RLPD reference (3 seeds, target_entropy_frac=0.75 --
NOT the const0.5/TES setup above, so it is a rough reference, not a matched arm; it only exists for
PointMaze-Teleport and AntSoccer-Arena, DDS at online_controller_rlpd_ablation/norlpd, empowerment at
online_controller_loglik_sweep/norlpd/norlpd -- there is none for Cube-Single).
DDS and empowerment_skill each get their own row, 5 seeds per curve, mean +/- 95% CI. A
curve still mid-run (fewer eval points than its peers) is plotted as far as it has gotten
and flagged "[running]" in the legend; a cell with zero finished seeds yet is skipped.
"""

import argparse
import csv
import glob
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
# (rlpd_root subdir template, label, color, linestyle) -- half-RLPD drawn solid, full-RLPD dashed.
ARMS = [
    ("online_controller_rlpd_half/const0.5/rlpd_ft0.5", "const0.5, RLPD half",  "tab:blue",   "-"),
    ("online_controller_entropy_sweep/frac0.5/rlpd",     "const0.5, RLPD full", "tab:blue",   "--"),
    ("online_controller_rlpd_half/tes/rlpd_ft0.5",       "TES, RLPD half",      "tab:red",    "-"),
    ("online_controller_tes/rlpd",                       "TES, RLPD full",      "tab:red",    "--"),
]
# Pre-existing no-RLPD reference (3 seeds, target_entropy_frac=0.75, NOT the const0.5/TES setup
# above -- a rough reference only). Missing entries (e.g. Cube-Single, either family) are skipped.
NORLPD_SUBDIR = {
    ("DDS", "PointMaze-Teleport-navigate"): "online_controller_rlpd_ablation/norlpd",
    ("DDS", "AntSoccer-Arena-navigate"): "online_controller_rlpd_ablation/norlpd",
    ("empowerment", "PointMaze-Teleport-navigate"): "online_controller_loglik_sweep/norlpd/norlpd",
    ("empowerment", "AntSoccer-Arena-navigate"): "online_controller_loglik_sweep/norlpd/norlpd",
}
NORLPD_LABEL = "No RLPD (entropy_frac=0.75, unmatched)"
NORLPD_COLOR = "gray"


def _load_curve(eval_csv):
    steps, succ = [], []
    with open(eval_csv, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            succ.append(float(row["evaluation/success"]))
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(succ)[order]


def _stack_seeds(rlpd_root):
    """(steps, (n_seeds, n_evals) success matrix, complete), truncated to the shortest seed."""
    curves = []
    for d in sorted(glob.glob(os.path.join(rlpd_root, "OGBench", "*", "sd*"))):
        eval_csv = os.path.join(d, "eval.csv")
        if not os.path.isfile(eval_csv):
            continue
        steps, succ = _load_curve(eval_csv)
        if len(steps) > 1:  # drop a seed with only the pre-training eval
            curves.append((steps, succ))
    if not curves:
        return None
    n_min = min(len(s) for s, _ in curves)
    ref_steps = curves[0][0][:n_min]
    mat = np.stack([succ[:n_min] for _, succ in curves], axis=0)
    complete = all(len(s) == n_min for s, _ in curves)
    return ref_steps, mat, complete, len(curves)


def _plot_one(ax, rlpd_root, label, color, linestyle, linewidth=2.2, alpha=0.12, zorder=2):
    stacked = _stack_seeds(rlpd_root)
    if stacked is None:
        print(f"    {label}: no finished seeds yet, skipping")
        return
    steps, mat, complete, n = stacked
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    ci95 = 1.96 * sem
    tag = "" if complete else " [running]"
    ax.plot(steps, mean, color=color, linewidth=linewidth, linestyle=linestyle,
             label=f"{label} (n={n}){tag}", zorder=zorder)
    ax.fill_between(steps, mean - ci95, mean + ci95, color=color, alpha=alpha, zorder=zorder)
    print(f"    {label}: {n} seeds, {mat.shape[1]} eval points" + ("" if complete else " (still running)"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="rlpd_half_vs_full_curves.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(len(FAMILIES), len(ENVS), figsize=(15, 9), sharey=True, sharex="col")
    for row, family in enumerate(FAMILIES):
        for col, env in enumerate(ENVS):
            ax = axes[row, col]
            ckpt = CKPTS[(family, env)]
            print(f"{family} / {env}:")
            for subdir, label, color, linestyle in ARMS:
                _plot_one(ax, os.path.join(ckpt, subdir), label, color, linestyle)
            norlpd_subdir = NORLPD_SUBDIR.get((family, env))
            if norlpd_subdir is not None:
                _plot_one(ax, os.path.join(ckpt, norlpd_subdir), NORLPD_LABEL, NORLPD_COLOR,
                          ":", linewidth=1.6, alpha=0.08, zorder=1)
            if row == 0:
                ax.set_title(env, fontsize=11)
            if row == len(FAMILIES) - 1:
                ax.set_xlabel("Online env steps")
            if col == 0:
                ax.set_ylabel(f"{family}\nEvaluation success rate")
            ax.set_ylim(-0.02, 1.02)
            ax.axvline(500000, color="gray", linewidth=0.8, linestyle=":", zorder=0)
            ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "RLPD for the first half of training vs. RLPD for the whole run, same entropy setup\n"
        "(solid = rlpd_frac_time=0.5, dashed = rlpd_frac_time=1.0; dotted vertical line = the 500k-step "
        "switch to online-only). mean ± 95% CI over up to 5 seeds."
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
