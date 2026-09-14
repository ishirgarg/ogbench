"""Compare online skill-controller learning curves on cube-single-play across skill
families: the 5 empowerment_final checkpoints (5 seeds each) vs. the 1 DDS checkpoint
(5 seeds), all trained with agents/online_crl_skill_controller.py + RLPD on
cube-single-center-online-v0.
"""

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


def _latest_run_dir_per_seed(rlpd_root):
    """Return {seed: eval.csv path}, picking the most recently modified run per sdNNN_* dir."""
    result = {}
    for run_dir in glob.glob(os.path.join(rlpd_root, "OGBench", "Debug", "sd*")):
        m = os.path.basename(run_dir)
        seed = int(m[2:5])
        eval_csv = os.path.join(run_dir, "eval.csv")
        if not os.path.isfile(eval_csv):
            continue
        if seed not in result or os.path.getmtime(run_dir) > os.path.getmtime(os.path.dirname(result[seed])):
            result[seed] = eval_csv
    return result


def _load_curve(eval_csv):
    steps, succ = [], []
    with open(eval_csv, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            succ.append(float(row["evaluation/success"]))
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(succ)[order]


def _stack_seeds(rlpd_root, label):
    seed_files = _latest_run_dir_per_seed(rlpd_root)
    if not seed_files:
        raise FileNotFoundError(f"No eval.csv found under {rlpd_root}")
    curves = {}
    ref_steps = None
    for seed, path in sorted(seed_files.items()):
        steps, succ = _load_curve(path)
        if ref_steps is None:
            ref_steps = steps
        elif len(steps) != len(ref_steps):
            # Interpolate onto the reference grid if a seed logged a different number of evals.
            succ = np.interp(ref_steps, steps, succ)
            steps = ref_steps
        curves[seed] = succ
    n_seeds = len(curves)
    mat = np.stack([curves[s] for s in sorted(curves)], axis=0)  # (n_seeds, n_evals)
    print(f"{label}: {n_seeds} seeds {sorted(curves)}, {mat.shape[1]} eval points")
    return ref_steps, mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emp_root", default="ckpts/final/empowerment_final/cube-single-play",
        help="Env dir holding the 5 empowerment_final cube-single-play checkpoints.",
    )
    parser.add_argument(
        "--dds_ckpt", default="ckpts/final/dds/cube-single-play/sd000_s_38579172.0.20260904_235603",
        help="The single DDS cube-single-play checkpoint.",
    )
    parser.add_argument("--save_subdir", default="online_controller/rlpd")
    parser.add_argument("--output", default="online_cube_controller_curves.png")
    args = parser.parse_args()

    emp_ckpt_dirs = sorted(
        d for d in glob.glob(os.path.join(args.emp_root, "sd000_*")) if os.path.isdir(d)
    )
    if not emp_ckpt_dirs:
        raise FileNotFoundError(f"No sd000_* checkpoints found under {args.emp_root}")

    plt.figure(figsize=(8, 5.5))

    emp_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(emp_ckpt_dirs)))
    for color, ckpt_dir in zip(emp_colors, emp_ckpt_dirs):
        tag = os.path.basename(ckpt_dir).replace("sd000_s_", "").split(".")[0]
        rlpd_root = os.path.join(ckpt_dir, args.save_subdir)
        steps, mat = _stack_seeds(rlpd_root, f"empowerment ckpt {tag}")
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ci95 = 1.96 * sem
        plt.plot(steps, mean, color=color, linewidth=1.8, label=f"empowerment ({tag}, n={mat.shape[0]})")
        plt.fill_between(steps, mean - ci95, mean + ci95, color=color, alpha=0.15)

    dds_rlpd_root = os.path.join(args.dds_ckpt, args.save_subdir)
    steps, mat = _stack_seeds(dds_rlpd_root, "DDS ckpt")
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
    ci95 = 1.96 * sem
    plt.plot(steps, mean, color="crimson", linewidth=2.6, linestyle="--", label=f"DDS (n={mat.shape[0]})")
    plt.fill_between(steps, mean - ci95, mean + ci95, color="crimson", alpha=0.15)

    plt.xlabel("Online env steps")
    plt.ylabel("Evaluation success rate")
    plt.ylim(-0.02, 1.02)
    plt.title(
        "Online skill controller on cube-single-center-online-v0\n"
        "(mean ± 95% CI over 5 seeds per curve)"
    )
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
