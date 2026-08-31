"""CLI runner for Algorithm 2 (DADS effective empowerment) + Algorithm 3 (BA capacity).

Stages: EM skill discovery -> phase-2 nets -> global estimate -> validations
(shuffle test, cap, EM health, nonnegativity) -> BA capacity at sampled states
with the capacity DIAGNOSTIC C_BA(s) >= estimate_dads(s) and C_BA(s) <= log k.
The first inequality is heuristic, not a theorem: the effective estimate scores
real data futures and inflates by up to KL(post || m(.|s)) where the skill
marginal m misfits, so the check is applied with that KL as slack and the
per-state KL is reported.

Artifacts (posteriors, p(z), all net params, normalization stats) are saved to
impls/empowerment/runs/<name>/ and can be restored with --restore.

Standalone full-fidelity runner (exact EM, validations, BA).

Example:
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python -m empowerment.run_dads --env_name=pointmaze-medium-navigate-v0 --smoke
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # impls/

import jax
import numpy as np

from empowerment.ba_capacity import capacity_ba_with_spread
from empowerment.common import load_trajectory_data
from empowerment.dads_empowerment import DADS, kmeans, make_chunks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env_name', default='pointmaze-medium-navigate-v0')
    p.add_argument('--run_name', default=None)
    p.add_argument('--K', type=int, default=10, help='chunk length')
    p.add_argument('--k', type=int, default=16, help='number of discrete skills')
    p.add_argument('--em_rounds', type=int, default=50)
    p.add_argument('--mstep_steps', type=int, default=2000)
    p.add_argument('--init_mstep_steps', type=int, default=None, help='default: mstep_steps')
    p.add_argument('--phase2_steps', type=int, default=20000)
    p.add_argument('--clf_steps', type=int, default=None, help='default: phase2_steps')
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--global_samples', type=int, default=50000)
    p.add_argument('--M', type=int, default=64, help='per-state estimator samples')
    p.add_argument('--num_ba_states', type=int, default=20)
    p.add_argument('--n_mc', type=int, default=128, help='BA samples per skill')
    p.add_argument('--max_chunks', type=int, default=0, help='0 = all chunks')
    p.add_argument('--restore', action='store_true', help='restore saved artifacts and skip finished stages')
    p.add_argument('--smoke', action='store_true', help='tiny settings for an end-to-end shape/NaN check')
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.k = 8
        args.em_rounds = 3
        args.mstep_steps = 200
        args.phase2_steps = 500
        args.global_samples = 5000
        args.num_ba_states = 5
        args.n_mc = 64
        args.max_chunks = args.max_chunks or 5000
    if args.run_name is None:
        args.run_name = f'{args.env_name}_K{args.K}_k{args.k}_s{args.seed}' + ('_smoke' if args.smoke else '')
    impls_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(impls_dir, 'empowerment', 'runs', args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    phase1_path = os.path.join(run_dir, 'phase1.pkl')
    phase2_path = os.path.join(run_dir, 'phase2.pkl')
    post_path = os.path.join(run_dir, 'post.npz')
    results_path = os.path.join(run_dir, 'results.json')
    init_msteps = args.init_mstep_steps or args.mstep_steps
    clf_steps = args.clf_steps or args.phase2_steps
    log_k = float(np.log(args.k))
    results = {'args': vars(args), 'log_k': log_k}

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed + 1)

    # ---------------------------------------------------------------- Data
    data = load_trajectory_data(args.env_name)
    chunks = make_chunks(data, args.K)
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    norm_path = os.path.join(run_dir, 'norm.npz')
    if args.restore and os.path.exists(norm_path):
        # Restored artifacts were trained under the SAVED normalization; reuse it.
        saved = np.load(norm_path)
        obs_mean, obs_std = saved['obs_mean'], saved['obs_std']
    else:
        obs_mean = chunks.reshape(-1, chunks.shape[-1]).mean(0)
        obs_std = chunks.reshape(-1, chunks.shape[-1]).std(0) + 1e-6
    chunks = ((chunks - obs_mean) / obs_std).astype(np.float32)
    Nc, K, D = chunks.shape
    print(f'[data] {args.env_name}: {Nc} chunks of length {K}, obs_dim {D}', flush=True)
    np.savez(norm_path, obs_mean=obs_mean, obs_std=obs_std)

    # ---------------------------------------------------------------- Phase 1: EM
    em_health = []
    if args.restore and os.path.exists(phase2_path):
        model = DADS.load(phase2_path, seed=args.seed)
        post = np.load(post_path)['post']
        assert post.shape == (Nc, args.k), (
            f'restored post {post.shape} does not match chunks/k ({Nc}, {args.k}); '
            'check --max_chunks / --k / dataset against the saved run'
        )
        print('[restore] loaded phase-2 artifacts; skipping EM and phase 2', flush=True)
        skip_phase2 = True
    elif args.restore and os.path.exists(phase1_path):
        model = DADS.load(phase1_path, seed=args.seed)
        post = np.load(post_path)['post']
        assert post.shape == (Nc, args.k), (
            f'restored post {post.shape} does not match chunks/k ({Nc}, {args.k}); '
            'check --max_chunks / --k / dataset against the saved run'
        )
        print('[restore] loaded phase-1 artifacts; skipping EM', flush=True)
        skip_phase2 = False
    else:
        model = DADS(D, args.k, K, seed=args.seed)
        # Init: k-means on chunk displacement s_{K-1} - s_0; one M-step from hard assignments.
        disp = chunks[:, -1] - chunks[:, 0]
        assign, _ = kmeans(disp, args.k, rng)
        post = np.zeros((Nc, args.k), np.float32)
        post[np.arange(Nc), assign] = 1.0
        model.p_z = post.mean(0).clip(1e-8)
        model.p_z /= model.p_z.sum()
        t0 = time.time()
        loss = model.m_step(chunks, post, init_msteps, args.batch, rng)
        print(f'[init] k-means + M-step: nll={loss:.4f} ({time.time() - t0:.1f}s)', flush=True)

        prev_assign = assign
        for r in range(args.em_rounds):
            t0 = time.time()
            post, avg_ll, chunk_ll = model.e_step(chunks)
            new_assign = post.argmax(1)
            frac_changed = float((new_assign != prev_assign).mean())
            prev_assign = new_assign
            post, reseeded, given_up = model.update_prior_and_reseed(post, chunk_ll)
            loss = model.m_step(chunks, post, args.mstep_steps, args.batch, rng)
            em_health.append(
                dict(round=r, avg_chunk_ll=avg_ll, frac_changed=frac_changed, mstep_nll=loss,
                     reseeded=reseeded, given_up=given_up, p_z=model.p_z.tolist())
            )
            print(
                f'[EM {r:02d}] avg_chunk_ll={avg_ll:.3f} frac_changed={frac_changed:.4f} '
                f'mstep_nll={loss:.4f} reseeded={reseeded} given_up={given_up} '
                f'p_z(min/max)={model.p_z.min():.4f}/{model.p_z.max():.4f} ({time.time() - t0:.1f}s)',
                flush=True,
            )
            if frac_changed < 0.01 and r > 0:
                print('[EM] converged (<1% of chunks changed assignment)', flush=True)
                break
        # Final-guard loop. Invariant: we always EXIT immediately after an
        # E-step (never after reseed -> M-step), so the saved posteriors are
        # the exact Bayes posteriors of the saved dynamics -- reseeded hard
        # labels are only ever an intermediate training signal.
        for guard_pass in range(5):
            post, avg_ll, chunk_ll = model.e_step(chunks)
            model.p_z = post.mean(0)
            if (model.p_z > 0.5 / args.k).all() or guard_pass == 4:
                break
            post, reseeded, given_up = model.update_prior_and_reseed(post, chunk_ll)
            if not reseeded:
                # All remaining dead skills are declared unsupported; nothing
                # left to revive, and post is already an exact E-step posterior.
                break
            print(f'[EM final-guard] reseeding underused skills {reseeded} (given up: {given_up})', flush=True)
            model.m_step(chunks, post, args.mstep_steps, args.batch, rng)
        em_health.append(dict(round='final', avg_chunk_ll=avg_ll, p_z=model.p_z.tolist()))
        print(f'[EM final] avg_chunk_ll={avg_ll:.3f} p_z={np.round(model.p_z, 4)}', flush=True)
        model.save(phase1_path)
        np.savez(post_path, post=post)
        skip_phase2 = False
    results['em_health'] = em_health
    results['p_z'] = model.p_z.tolist()
    dead_skills = np.nonzero(model.p_z <= 0.5 / args.k)[0]
    exhausted = model.revive_attempts >= model.max_revive_attempts
    unexplained_dead = [int(z) for z in dead_skills if not exhausted[z]]
    n_active = int(args.k - len(dead_skills))
    results['dead_skills'] = [int(z) for z in dead_skills]
    results['n_active_skills'] = n_active
    # PASS means every skill is either alive or was declared unsupported after
    # max_revive_attempts failed revivals (an honest effective-k < k, reflected
    # in the H(p_z) ceiling) -- FAIL means a skill died without the guard
    # exhausting its revival budget.
    collapse_ok = len(unexplained_dead) == 0
    results['no_collapse'] = collapse_ok
    print(f'[check] EM collapse guard: {"PASS" if collapse_ok else "FAIL"} '
          f'({n_active}/{args.k} skills active; '
          f'unsupported after {model.max_revive_attempts} revivals: '
          f'{[int(z) for z in dead_skills if exhausted[z]]}; '
          f'unexplained dead: {unexplained_dead})', flush=True)

    # ---------------------------------------------------------------- Phase 2
    if not skip_phase2:
        t0 = time.time()
        q_loss, clf_loss = model.train_phase2(chunks, post, args.phase2_steps, clf_steps, args.batch, rng)
        print(f'[phase2] q_nll={q_loss:.4f} clf_ce={clf_loss:.4f} ({time.time() - t0:.1f}s)', flush=True)
        results['phase2'] = dict(q_nll=q_loss, clf_ce=clf_loss)
        model.save(phase2_path)

    # ---------------------------------------------------------------- Phase 3: global estimate + validations
    g_mean, g_se = model.global_estimate(chunks, post, rng, n=args.global_samples)
    s_mean, s_se = model.global_estimate(chunks, post, rng, n=args.global_samples, shuffle=True)
    h_pz = float(-(model.p_z * np.log(np.clip(model.p_z, 1e-12, None))).sum())
    print(f'[global] effective empowerment = {g_mean:.4f} +- {g_se:.4f} nats '
          f'(cap log k = {log_k:.4f}; honest ceiling H(p_z) = {h_pz:.4f})', flush=True)
    print(f'[shuffle test] broken-pairing estimate = {s_mean:.4f} +- {s_se:.4f} nats '
          f'(must be <= 0; strongly negative when skill channels are separated)', flush=True)
    results['global_estimate'] = dict(mean=g_mean, stderr=g_se)
    results['shuffle_test'] = dict(mean=s_mean, stderr=s_se)

    # For a density-model Barber-Agakov bracket the shuffled expectation is a
    # negative Jensen gap: <= 0 always, and ~0 only if per-skill channels overlap.
    # Verified empirically: shuffled draws whose random z matches the source
    # chunk's skill recover the paired estimate; mismatches are strongly negative.
    shuffle_ok = (s_mean < 3 * s_se) and (g_mean - s_mean) > 10 * max(g_se, s_se)
    print(f'[check] shuffle test <= 0 and clearly separated from true estimate: '
          f'{"PASS" if shuffle_ok else "FAIL"}', flush=True)
    cap_ok = g_mean <= log_k + 1e-6
    # The operative ceiling is the skill-usage entropy H(p_z) <= log k.
    near_cap = g_mean > min(log_k, h_pz) - 0.2
    results['h_pz'] = h_pz
    print(f'[check] global estimate <= log k: {"PASS" if cap_ok else "FAIL"}'
          + (' | WARNING: within 0.2 nats of the ceiling -- discretization binds; rerun with 2k skills'
             if near_cap else ''),
          flush=True)
    nonneg_ok = g_mean > -3 * g_se
    print(f'[check] nonnegativity within noise: {"PASS" if nonneg_ok else "FAIL"}', flush=True)
    results.update(shuffle_ok=shuffle_ok, cap_ok=cap_ok, near_cap=bool(near_cap), nonneg_ok=nonneg_ok)

    # ---------------------------------------------------------------- BA capacity at sampled states
    print(f'[BA] capacity at {args.num_ba_states} sampled chunk-start states (n_mc={args.n_mc}, 3 restarts)',
          flush=True)
    state_idxs = rng.choice(Nc, size=args.num_ba_states, replace=False)
    table = []
    header = (f'{"idx":>7} {"eff (A2)":>10} {"+-":>7} {"C_BA (A3)":>10} {"spread":>8} '
              f'{"KL(post||m)":>12} {"strict":>7} {"diag":>5}')
    print(header, flush=True)
    diag_ok = True
    n_strict_fail = 0
    for j, i in enumerate(state_idxs):
        e_mean, e_se = model.estimate_dads(chunks, post, int(i), rng, M=args.M)
        key, sub = jax.random.split(key)
        C, spread, _, _ = capacity_ba_with_spread(model, chunks[i, 0], sub, n_restarts=3, n_mc=args.n_mc)
        # m-misfit at this state: KL(post[i] || m(.|s0)). The effective estimate
        # can legitimately exceed C_BA by up to this much (see module docstring),
        # so it enters the diagnostic's slack; strict violations are reported too.
        m_row = np.asarray(model._clf_probs(model.clf_state['params'], chunks[int(i), 0:1]))[0]
        p_row = np.clip(post[int(i)], 1e-12, None)
        kl_pm = float((p_row * (np.log(p_row) - np.log(np.clip(m_row, 1e-12, None)))).sum())
        strict = (C >= e_mean - 0.05 - 2 * e_se) and (C <= log_k + 1e-6)
        ok = (C >= e_mean - 0.05 - 2 * e_se - kl_pm) and (C <= log_k + 1e-6)
        diag_ok &= ok
        n_strict_fail += int(not strict)
        table.append(dict(chunk_idx=int(i), eff=e_mean, eff_se=e_se, C_ba=C, C_spread=spread,
                          kl_post_m=kl_pm, strict_ok=bool(strict), diag_ok=bool(ok)))
        print(f'{int(i):>7} {e_mean:>10.4f} {e_se:>7.4f} {C:>10.4f} {spread:>8.4f} '
              f'{kl_pm:>12.4f} {"PASS" if strict else "FAIL":>7} {"PASS" if ok else "FAIL":>5}',
              flush=True)
    results['ba_table'] = table
    results['diag_ok'] = bool(diag_ok)
    results['n_strict_fail'] = n_strict_fail
    print(f'[check] capacity diagnostic (C_BA >= eff - KL(post||m) slack, C_BA <= log k) at all states: '
          f'{"PASS" if diag_ok else "FAIL"}; strict C_BA >= eff failures: {n_strict_fail}/{len(state_idxs)} '
          f'(strict failures are expected exactly where KL(post||m) is large)', flush=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f'[done] results written to {results_path}', flush=True)


if __name__ == '__main__':
    main()
