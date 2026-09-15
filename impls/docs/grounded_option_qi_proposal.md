# SR-IQL: Skill-Restricted IQL — a fully offline high level over the frozen `empowerment_skill` skills

Status: proposal v3, 2026-09-11. Nothing implemented.
v1 (Grounded Option Q-Iteration) and v2 (Grounded Option IQL) both required *executing*
the frozen skills in the simulator. That is off the table: we may run the skill policy's
forward pass anywhere, but never step the environment with its actions. v3 is fully
offline. The file name is kept so links stay valid.

## 0. Setting, and the observation that reorganises it

**Given, frozen:** the `empowerment_skill` actor `pi_z(a|s)`, `z in {1..K}`, `K = 50`,
deterministic mean `mu_z(s)` (const_std=True, temperature 0 at eval). Usable as a
function only. Its critic `V^z` is not used (spearman <= 0.19 as a ranker, and its
bootstrap at `mu_z(s')` is exactly the off-support query we are about to remove).

**Given:** the offline dataset `D = {(s, a, s')}` from an unknown behaviour policy
`beta`. No skill labels, no skill executions.

**Learned:** a goal-conditioned skill selector `z*(s, g)`.

**The observation.** Because the skills are fixed functions, the option MDP is a
deterministic transform of the low-level MDP. So "offline RL over skills" is *the same
problem* as "offline goal-conditioned RL at the low level with the policy class
restricted to `{pi_z}`". Two consequences:

1. **Out-of-distribution is defined in action space, not option space.** At the option
   level every `z` is untried, so "in-sample" is vacuous (this is why option-level IQL
   on imputed labels is conservatism w.r.t. the labeller, not the dynamics). At the
   action level the statement is well-defined: skill `z` is in-distribution at `s` iff
   `mu_z(s)` is covered by `beta(.|s)`. That is offline-computable.
2. **The whiteboard identity survives, one level down.** The board's
   `E_{s+ ~ p_z}[V] = E_{s+ ~ p}[ p(z|s,s+)/p(z) V ]` is not estimable offline (no
   `p(z|s,s+)` exists). Its action-level form
   `E_{a ~ pi_z}[Q(s,a)] = E_{a ~ beta}[ pi_z(a|s)/beta(a|s) Q(s,a) ]`
   is: data actions are samples from `beta`, and `pi_z(a|s)` is a forward pass.

## Why we expect this to beat the skill-BC controller

`skill_bc_relabel_controller` and SR-IQL see the same data, use the same IQL critic, and
drive the same frozen skills. They differ in one thing: **what evidence links a skill to a
data transition, and how far that evidence is trusted.**

* *Skill BC* trusts a 10-step window: it picks the one skill whose action sequence looks
  most like the window's and credits that skill with the window's *endpoint*. The
  diagnostics show both halves failing. The pick is no better than random (assigned
  skill's outcome 0.307 m vs 0.338 m random) because 50 near-identical action sequences
  cannot be told apart over a window, and the credited endpoint is 4-20x farther than the
  skill actually travels (0.71-1.21 m believed vs 0.10-0.21 m actual). The high level is
  then trained on an option MDP whose transitions were never produced by its actions,
  and its value function is calibrated to a world that moves far faster than the one it
  acts in. Nothing in the remaining pipeline can undo this: the label is fixed before
  training starts and IQL is exactly as good as its transitions.

* *SR-IQL* trusts one step: a data action is evidence about skill `z` in proportion to
  how close it is to `mu_z(s)`, the next state is the data's real next state for that
  action, and everything beyond one step is bootstrapped through `V`. Three consequences
  follow directly. (i) No skill is ever credited with an outcome it could not produce,
  because the only outcomes used are those of data actions the skill would itself take
  (F2 removed by construction). (ii) Evidence about a skill is pooled across every data
  step near `s` where behaviour did something skill-like, instead of concentrated in a
  single window's hard label, so the label-noise mechanism of F1 has no place to enter.
  (iii) `V` is the value of the best skill *the data can vouch for* at `s`, so the
  selector is pessimistic exactly where the old controller was over-confident: states
  where no skill's action is covered — the states F3 says it drives itself into.

The honest expectation is therefore not "better labels" (F5 already showed that would not
help) but a high level whose value function is finally about the skills it commands.
What it cannot do is see the states the controller creates at deployment; on that axis it
gets only the protection flat offline RL gets, which is why flat `gciql` is the right
upper reference and E0 the right first measurement.

## 1. What has to be fixed, and what fully-offline can and cannot fix

| # | failure | measurement | fully offline |
|---|---|---|---|
| F1 | BC label is a coin flip | margin 0.01 nats/step; assigned outcome (0.307 m) ~ random (0.338 m) | fixable: no label; per-step soft weights, self-normalised |
| F2 | believed vs actual endpoint | window 0.71-1.21 m, credited skill 0.10-0.21 m | fixable: value of `z` is computed from data actions that resemble `mu_z`, one step at a time, bootstrapped — never a 10-step window credited wholesale |
| F3 | controller drives itself OOD | speed 87-96% from data states, 28-89% from controller states | partly: only the protection flat offline RL gets (pessimistic `V` at uncovered states); nothing offline can *observe* those states |
| F4 | `V^z` can't rank skills | spearman <= 0.19 | fixable: goal-conditioned, in-sample `Q` replaces the occupancy kernel |
| F5 | label quality doesn't predict success | pointmaze 77%/0.20 vs antmaze 52%/0.84 | consistent: the fix is not a better label |

## 2. Why the current pipeline is importance sampling done wrong

`skill_bc_relabel_controller` labels a window by `argmax_z sum_{t<H} log pi_z(a_t|s_t)`.
That sum is the log of the H-step importance weight `prod_t pi_z(a_t|s_t)/beta(a_t|s_t)`
with the z-independent `beta` dropped. So the pipeline is **hard-argmax, 10-step, Monte
Carlo importance sampling**, with three defects:

* *hard argmax* — throws away the weight (F1);
* *10-step product* — the effective sample size of a 10-step, 50-way product weight is
  ~1 window per state, so even the soft version has no support (this is why soft labels
  alone cannot rescue it);
* *(minor) sigma = 1* — irrelevant to the argmax, but the 0.01-nat "tie" statistic is
  measured at the actor's placeholder `const_std` width and should not be read as
  "skills are indistinguishable in action space".

The fix for the product is the fix TD always gives over Monte Carlo: **weight one step,
bootstrap the rest.**

## 3. SR-IQL

Everything below runs on `GCDataset` batches `(s, a, s', g, r, m)` exactly as `gciql`
consumes them (`value_p_*` goal sampling, `gc_negative` reward, mask 0 at the goal).
No new dataset class.

### 3.1 Skill kernel — the only new ingredient

    w_z(s, a) = exp( -||a - mu_z(s)||^2 / (2 sigma_k^2) )            in (0, 1], for all K skills

`mu_z(s)` for all `z` at once is one `lax.map` over one-hots — exactly what
`chunk_skill_logliks` already does (`skill_bc_relabel_controller.py:202-231`); since
`log pi_z(a|s) = -||a - mu_z||^2/2 + const` under const_std, `w_z` is that array rescaled.
The scale is not a tuning knob: set it from the data as the RMS residual of behaviour
actions around the skill manifold, `sigma_k^2 = E_D[ min_z ||a - mu_z(s)||^2 ] / A`, and
ablate the parameter-free hard limit `w_z = 1[z = argmin_z ||a - mu_z(s)||]` (a Voronoi
partition of action space by the skills). If the two agree, the width never mattered.
Diagnostic: per-state ESS `(sum_z w_z)^2 / sum_z w_z^2`.

### 3.2 Networks

    Q_theta(s, a, g)          scalar, standard IQL critic, only ever evaluated at DATA actions
    Qz_phi(s, g)   in R^K     per-skill value: "take z's action now, then act optimally among skills"
    C_phi(s)       in R^K     coverage: how much behaviour mass sits on z's action at s
    V_psi(s, g)               value of the best *covered* skill at s
    (targets Qbar, Vbar by Polyak, as in gciql)

### 3.3 Losses

**(a) Critic, unchanged from IQL.** In-sample: `a` is a data action.

    L_Q = E_D [ ( r + gamma * m * Vbar(s', g) - Q_theta(s, a, g) )^2 ]

**(b) Per-skill value by weighted regression.** This is the action-level whiteboard
identity, self-normalised by the regression itself.

    L_Qz = E_D [ sum_z  w_z(s, a) * ( Qbar_theta(s, a, g) - Qz_phi(s, g)[z] )^2 ]

The minimiser at each `s` is `sum_a w_z Q / sum_a w_z`, i.e. `E_beta[w_z Q] / E_beta[w_z]`
— the value of `z`'s action estimated from data actions that resemble it, with no
off-support query. (Exactly: this is the `beta`-tilted skill value; the exact IS weight
would be `w_z / beta`, which needs a density model of `beta` and re-inflates variance.
The tilt toward data-dense actions is the same bias IQL carries and is welcome offline.)

**(c) Coverage.** Same regression with target 1: `C(s)[z] -> E_beta[w_z(s,a)]`, the
normaliser of (b).

    L_C = E_D [ sum_z ( w_z(s, a) - C_phi(s)[z] )^2 ]
    c(z|s) = C(s)[z] / sum_z' C(s)[z']                      the behaviour distribution over SKILLS

`c(z|s)` is the object option-level IQL was missing: a distribution over skills induced
by the data, so "in-sample max over skills" means something.

**(d) Value: IQL expectile over covered skills.**

    L_V = E_D [ sum_z  c(z|s) * l2^tau( Qzbar(s, g)[z] - V_psi(s, g) ) ],    l2^tau(u) = |tau - 1[u<0]| u^2

Compare gciql: `L_V = E_D[ l2^tau( Q(s,a) - V(s) ) ]` over data actions. Replacing "data
actions" by "coverage-weighted skills" is the whole difference, and it is what makes
`V` the value of the best skill *the data can vouch for*, rather than the best data
action (which no skill may be able to reproduce — that would re-create F2 at the value
level).

Total: `L_Q + L_Qz + L_C + L_V`, one Adam, gciql hyperparameters (`tau = 0.9`,
`discount = 0.99`, `tau_polyak = 0.005`).

### 3.4 Policy: exact argmax over K, no actor

    z*(s, g) = argmax_z [ Qz_phi(s, g)[z] + alpha * log c(z|s) ]

`alpha * log c` is DDPG+BC's behaviour term in discrete form: prefer skills the data can
vouch for at this state; `alpha = 0` is a valid ablation. Executed action:
`a = mu_{z*}(s)` through `sample_actions_with_skill` at temperature 0.

**Re-pick every step.** `V` assumes optimal switching at every step, so `skill_horizon = 1`
is the consistent choice and needs no `H`. Holding for `H > 1` is available as a
smoothing knob through the unchanged `init_eval_state` / `sample_actions_with_state`
contract, at the cost of a mismatch with what `V` assumes. (The earlier freeze of a
`skill_horizon = 1` selector — `selector-absorbs-into-standstill-skill` — was a bad-`V`
failure, F4, not a horizon failure.)

### 3.5 Where the conservatism is, stated once

Every learned quantity is fit only at data `(s, a)`: `Q` at data actions, `Qz` and `C`
by regression over data actions, `V` by expectile over skills weighted by data coverage.
The only place a skill's action is ever *used* rather than *scored* is at deployment,
`a = mu_{z*}(s)` — the same single extrapolation step every offline RL method takes when
it acts. Nothing else is extrapolated. No ensemble, no `lambda`, no model.

## 4. What this is, in one sentence each

* **vs `skill_bc_relabel_controller`:** same data, same IQL machinery; per-step soft
  weight instead of 10-step hard label; value is of the skill's action, bootstrapped,
  not of the window's endpoint.
* **vs `skill_value_controller`:** same shape (`argmax_z` of a per-skill value) with the
  occupancy kernel `V^z` replaced by an in-sample goal-conditioned `Q` — a direct fix
  for F4, and the reason it will not absorb into a stand-still skill.
* **vs flat `gciql`:** identical `L_Q`; the policy class is restricted to `{mu_z}`, and
  `V` is the best *skill* value rather than the best data-action value. Flat gciql on the
  same data is therefore the natural upper reference: the gap between it and SR-IQL is
  the price of the skill restriction, and it is measurable.
* **vs the empowerment critic:** same "evaluate `pi_z` through data transitions"
  structure; the bootstrap `Q(y|s', mu_z(s'), z)` at an off-support action is replaced by
  the weighted-regression `V`-step over data actions. That single change is the
  in-sample principle.
* **vs the whiteboard:** the identity is kept at the action level, where `p(z|.)` is a
  forward pass instead of a fiction; `V(s) <- max_z Q(s,z)` becomes the coverage-weighted
  expectile; `gamma-tilde` and `p(s)Q` are not needed.

## 5. Experiments, cheapest first

**E0 — one-step cousin, zero training (an afternoon).** Take an existing `gciql`
checkpoint on the same env/data. Eval script: `z* = argmax_z Q_gciql(s, mu_z(s), g)`,
re-picked every step, actions through the frozen skills. This queries `Q` off-support
(the thing SR-IQL avoids) and evaluates only the skill's *first* action, so it is an
optimistic bound — but it answers the prior question: *can any selector over these 50
skills approach flat offline RL?* Report alongside flat gciql and BC-relabel
(0.844 / 0.204 / 0.000).

**E1 — SR-IQL.** New agent `agents/skill_restricted_iql.py` built on `gciql.py`
(`L_Q`, expectile, target nets) + `chunk_skill_logliks` for `w_z`. Kernel:
data-estimated `sigma_k` vs hard-nearest; `alpha in {0, 0.1, 1}`, `skill_horizon in {1, 10}`,
`tau in {0.7, 0.9}`. Three envs, same seeds as the BC-relabel runs.

**E2 — ablations that test the argument.**
 (i) `V` as plain IQL expectile over data actions (drop `c(z|s)`): does restricting the
     value to covered skills matter (the F2-at-value-level claim)?
 (ii) `Qz` replaced by the off-support `Q_theta(s, mu_z(s), g)`: does in-sample-ness
     matter, or is E0 already enough?
 (iii) hard argmax over a 10-step window instead of per-step weights: recovers the
     BC-relabel label inside the new machinery — the ladder from today's pipeline to
     SR-IQL in one knob.

**E3 — offline diagnostics (no env).** Per-state ESS of `w_z`; `c(z|s)` histograms
(does coverage collapse onto a few skills?); `argmax_z Qz` vs `argmax_z Q_gciql(s,mu_z)`
agreement; label entropy vs the 3.83-3.91 nats of the BC labeller.

(Teleport-and-execute remains available as a *diagnostic* — e.g. scoring `z*` against
the oracle skill at a state, as `plot_relabel_diagnosis.py` does — but nothing in
training or in the method's claims depends on it.)

## 6. Risks and limits

* **F3 is only partly addressable offline.** `V` is pessimistic at uncovered states, so
  the selector avoids planning through them, which is all flat offline RL gets either.
  Whether that suffices on antsoccer (OOD dist 1.59 m under the old controller) is the
  main empirical question; `alpha` is the knob.
* **Kernel width.** Too small: `w_z ~ 0` everywhere and `Qz`, `C` are fit to nothing;
  too large: every skill's value collapses to the behaviour value. The data-estimated
  `sigma_k` and the hard-nearest ablation bracket it; ESS (E3) is the check.
* **Skill restriction may simply cost too much.** E0 measures the gap to flat gciql
  before anything is built. If even the optimistic bound is far below flat gciql, the
  50 skills do not span the actions the task needs and no high level fixes that.
* **`beta`-tilt.** `Qz` is the `beta`-weighted skill value, not the exact one. Correcting
  with a BC density model is possible (E2 extension) but reintroduces the variance the
  weighting was chosen to avoid.

## 9. Results (2026-09-12) — SR-IQL does not beat skill-BC

Run as E0 + E1 via `scripts/slurm/submit_sriql_e0_e1.sh` on the K=50 `empowerment_final`
checkpoints (agent `agents/skill_restricted_iql.py`; 1M steps; 50 eps x 5 tasks; results
inside each skill run dir under `gciql_flat/` and `sriql_sweep/`).

**antmaze-medium-navigate:** BC-relabel 0.844 > flat gciql 0.716 (peak 0.764) > SR-IQL best
0.540 (gauss sigma=0.88, alpha=1, tau=0.9, re-pick every step) > E0 Q-selector 0.320 (H=10) /
0.112 (H=1). Discriminative kernels were worse: hard <= 0.332, gauss sigma=0.2 <= 0.172.
alpha=0 <= 0.264 in every kernel; alpha=1 best in every kernel. Post-hoc H=10 was worse than
H=1 in 11/12 cells. The off-support ablation Q(s, mu_z(s), g) with alpha=10 scored 0.452 where
the in-sample Qz scored 0.072.

**antsoccer-arena-navigate:** flat gciql 0.180; BC-relabel 0.000; E0 0.000/0.004; every
SR-IQL cell 0.000.

**Why (measured):** E_D[min_z ||a - mu_z(s)||^2] = 0.78 / 1.13 (8-dim) with a near-uniform
nearest-skill histogram (entropy 3.88 / 3.91): all 50 skill means are roughly equidistant from
data actions, so the action-space evidence a single data step carries about a skill is weak;
per-skill values regressed from ~6 effective data actions per state are noise. The data-
estimated sigma gave ESS ~40/50 (skill-agnostic Qz), and that cell won *because* its selector
degenerates to "most behaviour-like skill, tie-broken by value". Section 2's claim that the
10-step window was the defect is therefore wrong in one respect: the window integrates weak
per-step evidence, which is why the BC label still beats any per-step kernel.

**What survives:** the coverage prior c(z|s) (alpha) is the one component that clearly helped,
and the flat critic evaluated at the skill's action ranked skills better than the in-sample
heads. A selector of the form argmax_z [Q_flat(s, mu_z(s), g) + alpha log c(z|s)] is the cheap
follow-up this suggests; the in-sample per-skill value is not.

## Appendix A — the committed-skill variant, if `H > 1` semantics are wanted

If the high level must commit to `z` for `H` steps and the value should reflect that,
evaluate the committed policy by TD, not by product weights: `Qz^(h)(s,g)`, the value of
"run `z` for `h` more steps then switch optimally", satisfies
`Qz^(h)(s,g) = E_beta[ w_z (r + gamma m Vz^(h-1)(s',g)) ] / E_beta[w_z]` with
`Vz^(0) = V`. That is `H` heads (or an `h` input) trained by the same weighted
regression; one step of weighting per Bellman step, never a product. Only worth it if
E1 shows `skill_horizon = 10` clearly beats `1`.

## Appendix B — what did not survive from v1/v2

Grounding, closed-loop rounds, any learned or ensembled option model, the
`lambda u(s,z)` penalty, the option-level expectile, `rho`, the `H`-step option TD target
`y(s,z,g)`. All of them presupposed executed skill outcomes.

## Appendix C — repo pointers

* IQL critic / expectile / targets to copy: `agents/gciql.py:25-76`
* all-K skill means in one call: `skill_bc_relabel_controller.py:202-231` (`chunk_skill_logliks`; `w_z` is its output rescaled)
* frozen actor: `agents/empowerment_skill.py` (`skill_set`, `sample_actions_with_skill`)
* eval contract: `utils/evaluation.py:120-144`
* the two existing selectors to compare against: `agents/skill_value_controller.py`, `agents/skill_bc_relabel_controller.py`
