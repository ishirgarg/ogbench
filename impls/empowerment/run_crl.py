"""Runner for CRL effective empowerment (action-level InfoNCE).

Stages: train pi_bc -> train f_dyn -> train T -> validations -> global estimate.

Standalone full-fidelity runner (exact EM, validations, BA). For the main.py agent interface see agents/empowerment_crl_flowbc.py.

Example:
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python -m empowerment.run_crl --env_name pointmaze-medium-navigate-v0 --smoke
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from empowerment.common import load_trajectory_data
from empowerment.crl_empowerment import CRLEmpowerment, sample_bc_batch, sample_triple_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env_name', default='pointmaze-medium-navigate-v0')
    p.add_argument('--bc_steps', type=int, default=100000)
    p.add_argument('--dyn_steps', type=int, default=100000)
    p.add_argument('--critic_steps', type=int, default=100000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--num_negatives', type=int, default=63)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run_name', default=None)
    p.add_argument('--save_path', default=None, help='Where to save params (default: <run_dir>/params.pkl).')
    p.add_argument('--restore_path', default=None,
                   help='Restore params (inference-exact; optimizer state is not saved, so resuming '
                        'training restarts Adam moments). Skip training stages by setting their steps to 0.')
    p.add_argument('--log_interval', type=int, default=1000)
    p.add_argument('--eval_interval', type=int, default=5000)
    p.add_argument('--global_num', type=int, default=50000)
    p.add_argument('--smoke', action='store_true', help='Tiny step counts for a smoke test.')
    args = p.parse_args()
    if args.smoke:
        args.bc_steps = min(args.bc_steps, 300)
        args.dyn_steps = min(args.dyn_steps, 300)
        args.critic_steps = min(args.critic_steps, 300)
        args.log_interval = 50
        args.eval_interval = 100
        args.global_num = 2000
    return args


class CsvLogger:
    def __init__(self, path, fields):
        self.f = open(path, 'w', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()

    def log(self, **kw):
        self.w.writerow(kw)
        self.f.flush()

    def close(self):
        self.f.close()


def main():
    args = parse_args()
    run_name = args.run_name or f'{args.env_name}_{time.strftime("%Y%m%d_%H%M%S")}'
    impls_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(impls_dir, 'empowerment', 'runs', run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f'[run] dir={run_dir}', flush=True)
    print(f'[run] args={vars(args)}', flush=True)

    data = load_trajectory_data(args.env_name, 'train')
    val_data = load_trajectory_data(args.env_name, 'val')
    print(f'[data] train size={data.size} obs_dim={data.obs_dim} act_dim={data.act_dim} '
          f'val size={val_data.size}', flush=True)

    np_rng = np.random.default_rng(args.seed)
    est = CRLEmpowerment(
        obs_dim=data.obs_dim,
        act_dim=data.act_dim,
        seed=args.seed,
        lr=args.lr,
        num_negatives=args.num_negatives,
    )
    if args.restore_path:
        est.load(args.restore_path)
        print(f'[restore] loaded params from {args.restore_path}', flush=True)

    N = args.num_negatives
    cap = np.log(N + 1.0)

    # ------------------------------------------------------------ pi_bc -----
    logger = CsvLogger(os.path.join(run_dir, 'bc.csv'), ['step', 'loss'])
    t0 = time.time()
    for step in range(1, args.bc_steps + 1):
        s, a = sample_bc_batch(data, args.batch_size, np_rng)
        loss = est.bc_train_step(s, a)
        if step % args.log_interval == 0 or step == 1 or step == args.bc_steps:
            print(f'[bc] step={step} loss={loss:.5f} ({time.time() - t0:.0f}s)', flush=True)
            logger.log(step=step, loss=loss)
    logger.close()

    # ------------------------------------------------------------ f_dyn -----
    logger = CsvLogger(os.path.join(run_dir, 'dyn.csv'), ['step', 'loss'])
    t0 = time.time()
    for step in range(1, args.dyn_steps + 1):
        s, a, sp = sample_triple_batch(data, args.batch_size, np_rng)
        loss = est.dyn_train_step(s, a, sp)
        if step % args.log_interval == 0 or step == 1 or step == args.dyn_steps:
            print(f'[dyn] step={step} loss={loss:.5f} ({time.time() - t0:.0f}s)', flush=True)
            logger.log(step=step, loss=loss)
    logger.close()

    # ------------------------------------------------------------ critic ----
    logger = CsvLogger(os.path.join(run_dir, 'critic.csv'), ['step', 'loss', 'mi_estimate', 'val_loss'])
    t0 = time.time()
    for step in range(1, args.critic_steps + 1):
        s, a, sp = sample_triple_batch(data, args.batch_size, np_rng)
        loss = est.critic_train_step(s, a, sp)
        # val_loss is logged only on steps where it is freshly computed (blank otherwise).
        val_loss = None
        if step % args.eval_interval == 0 or step == args.critic_steps:
            vs, va, vsp = sample_triple_batch(val_data, 1024, np_rng)
            val_loss = est.critic_eval_loss(vs, va, vsp)
        if step % args.log_interval == 0 or step == 1 or step == args.critic_steps:
            mi = cap - loss
            val_str = f'{val_loss:.4f}' if val_loss is not None else '-'
            print(f'[critic] step={step} loss={loss:.4f} mi={mi:.4f} val_loss={val_str} '
                  f'({time.time() - t0:.0f}s)', flush=True)
            logger.log(step=step, loss=loss, mi_estimate=mi,
                       val_loss='' if val_loss is None else val_loss)
    logger.close()

    save_path = args.save_path or os.path.join(run_dir, 'params.pkl')
    est.save(save_path)
    print(f'[save] params -> {save_path}', flush=True)

    # ------------------------------------------------------- validations ----
    print('\n=== VALIDATIONS ===', flush=True)
    results = {}

    # Converged validation loss (average over a few fresh val batches).
    val_losses = []
    for _ in range(10):
        vs, va, vsp = sample_triple_batch(val_data, 1024, np_rng)
        val_losses.append(est.critic_eval_loss(vs, va, vsp))
    val_loss = float(np.mean(val_losses))
    val_mi = cap - val_loss
    results['val_loss'] = val_loss
    results['val_mi'] = val_mi

    # Global estimate over random dataset triples (train split), plus the same
    # quantity on the held-out val split.
    g_mean, g_stderr = est.estimate_global(data, num=args.global_num, np_rng=np_rng)
    results['global_estimate'] = g_mean
    results['global_stderr'] = g_stderr
    vg_mean, vg_stderr = est.estimate_global(val_data, num=args.global_num, np_rng=np_rng)
    results['val_global_estimate'] = vg_mean
    results['val_global_stderr'] = vg_stderr
    print(f'[global] train estimate = {g_mean:.4f} +- {g_stderr:.4f} nats, '
          f'val estimate = {vg_mean:.4f} +- {vg_stderr:.4f} nats '
          f'(cap log(N+1) = {cap:.4f})', flush=True)

    # 1. Shuffle tests (pairing broken -> collapse to <= ~0; null is slightly
    # negative by Jensen, so the criterion is one-sided).
    for mode in ('bc', 'permute'):
        s_mean, s_stderr = est.estimate_global(
            data, num=args.global_num, np_rng=np_rng, shuffle=mode
        )
        results[f'shuffle_{mode}_estimate'] = s_mean
        results[f'shuffle_{mode}_stderr'] = s_stderr
        shuffle_ok = s_mean < 0.2 * max(g_mean, 1e-9) and g_mean > s_mean + 5 * (g_stderr + s_stderr)
        print(f'[shuffle:{mode}] estimate = {s_mean:.4f} +- {s_stderr:.4f} nats -> '
              f'{"PASS" if shuffle_ok else "FAIL"} (must be <= ~0, well below global)', flush=True)

    # 2. Cap check.
    cap_ok = g_mean < cap
    near_cap = g_mean > cap - 0.2
    print(f'[cap] global {g_mean:.4f} < log(N+1)={cap:.4f}: {"PASS" if cap_ok else "FAIL"}'
          + (' WARNING: within 0.2 nats of cap' if near_cap else ''), flush=True)

    # 3a. Consistency (same split, near-identity): val-split global vs cap - val_loss.
    diff = abs(vg_mean - val_mi)
    cons_ok = diff < 0.05 + 5 * vg_stderr
    print(f'[consistency] val global {vg_mean:.4f} vs log(N+1)-val_loss {val_mi:.4f} '
          f'(|diff|={diff:.4f}) -> {"PASS" if cons_ok else "CHECK"}', flush=True)
    # 3b. Generalization gap (train vs val): a large positive gap flags critic
    # overfitting, which inflates the train-side estimate.
    gap = g_mean - vg_mean
    print(f'[generalization] train - val = {gap:.4f} nats '
          f'{"(WARNING: possible critic overfitting)" if gap > 0.2 else "-> OK"}', flush=True)

    # 4. Nonnegativity.
    nonneg_ok = g_mean > -3 * g_stderr
    print(f'[nonneg] global mean {g_mean:.4f} >= 0 within noise: '
          f'{"PASS" if nonneg_ok else "FAIL"}', flush=True)

    # Spot-check estimate_crl on a few states (dataset mode + arbitrary mode).
    print('\n=== estimate_crl spot checks ===', flush=True)
    for t in data.random_nonfinal_idxs(3, np_rng):
        t = int(t)
        m, se = est.estimate_crl(data=data, t=t, M=64, np_rng=np_rng)
        print(f'[crl] dataset t={t}: {m:.4f} +- {se:.4f} nats', flush=True)
    s_query = data.observations[data.random_nonfinal_idxs(1, np_rng)[0]]
    m, se = est.estimate_crl(s=s_query, M=64)
    print(f'[crl] arbitrary state {np.round(s_query, 3).tolist()}: {m:.4f} +- {se:.4f} nats', flush=True)

    with open(os.path.join(run_dir, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for k, v in results.items():
            w.writerow([k, v])
    print(f'\n[done] results -> {os.path.join(run_dir, "results.csv")}', flush=True)


if __name__ == '__main__':
    main()
