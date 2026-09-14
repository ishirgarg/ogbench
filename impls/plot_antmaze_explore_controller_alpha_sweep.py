"""Offline high-level controller AWR-alpha sweep on antmaze-medium-explore-v0 for the
3 antmaze-medium-explore checkpoints (2026-09-08 batch): 2 empowerment_final checkpoints
+ 1 DDS checkpoint, each trained at alpha in {0.3, 1, 3, 10} (single seed each), via
agents/skill_bc_relabel_controller.py:gciql (empowerment) / agents/dds_controller.py:gciql
(DDS). Plots evaluation/overall_success learning curves per alpha, one panel per
checkpoint, plus a final-step summary bar chart.
"""

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


def _latest_run_dir(alpha_root):
    run_dirs = glob.glob(os.path.join(alpha_root, "OGBench", "Debug", "sd*"))
    if not run_dirs:
        return None
    return max(run_dirs, key=os.path.getmtime)


def _load_curve(eval_csv):
    steps, succ = [], []
    with open(eval_csv, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            succ.append(float(row["evaluation/overall_success"]))
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(succ)[order]


def _load_ckpt_sweep(ckpt_dir, alphas, save_subdir):
    curves = {}
    for alpha in alphas:
        alpha_root = os.path.join(ckpt_dir, save_subdir, f"alpha{alpha}")
        run_dir = _latest_run_dir(alpha_root)
        if run_dir is None:
            print(f"WARNING: no run found under {alpha_root}")
            continue
        eval_csv = os.path.join(run_dir, "eval.csv")
        if not os.path.isfile(eval_csv):
            print(f"WARNING: no eval.csv in {run_dir}")
            continue
        curves[alpha] = _load_curve(eval_csv)
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts",
        nargs="+",
        default=[
            "ckpts/final/dds/antmaze-medium-explore/sd000_s_38624748.0.20260908_014705",
            "ckpts/final/empowerment_final/antmaze-medium-explore/sd000_s_38624747.0.20260908_014705",
            "ckpts/final/empowerment_final/antmaze-medium-explore/sd000_s_38624749.0.20260908_014705",
        ],
        help="The 3 antmaze-medium-explore checkpoints (1 DDS + 2 empowerment_final).",
    )
    parser.add_argument(
        "--labels", nargs="+", default=["DDS", "empowerment (47)", "empowerment (49)"]
    )
    parser.add_argument("--alphas", nargs="+", default=["0.3", "1", "3", "10"])
    parser.add_argument("--save_subdir", default="controller_awr_sweep")
    parser.add_argument("--output", default="antmaze_explore_controller_alpha_sweep.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, len(args.ckpts) + 1, figsize=(5 * (len(args.ckpts) + 1), 4.5))
    alpha_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(args.alphas)))

    final_success = {label: [] for label in args.labels}
    for ax, ckpt_dir, label in zip(axes[:-1], args.ckpts, args.labels):
        curves = _load_ckpt_sweep(ckpt_dir, args.alphas, args.save_subdir)
        for color, alpha in zip(alpha_colors, args.alphas):
            if alpha not in curves:
                final_success[label].append(np.nan)
                continue
            steps, succ = curves[alpha]
            ax.plot(steps, succ, color=color, linewidth=1.8, label=f"alpha={alpha}")
            final_success[label].append(succ[-1])
        ax.set_title(label)
        ax.set_xlabel("Offline train steps")
        ax.set_ylabel("Evaluation overall success")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", fontsize=8)

    bar_ax = axes[-1]
    x = np.arange(len(args.alphas))
    width = 0.8 / len(args.labels)
    ckpt_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(args.labels)))
    for i, (label, color) in enumerate(zip(args.labels, ckpt_colors)):
        bar_ax.bar(x + i * width, final_success[label], width=width, color=color, label=label)
    bar_ax.set_xticks(x + width * (len(args.labels) - 1) / 2)
    bar_ax.set_xticklabels([f"alpha={a}" for a in args.alphas])
    bar_ax.set_ylabel("Final-step evaluation overall success")
    bar_ax.set_ylim(-0.02, 1.02)
    bar_ax.set_title("Final success by alpha")
    bar_ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Offline AWR-alpha sweep on antmaze-medium-explore-v0 (seed 0, one run per alpha)"
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")

    print("\nFinal-step overall_success:")
    for label in args.labels:
        vals = ", ".join(f"alpha={a}: {v:.3f}" for a, v in zip(args.alphas, final_success[label]))
        print(f"  {label}: {vals}")


if __name__ == "__main__":
    main()
