"""Online CRL (contrastive) high-level skill controller over a frozen OGBench skill policy.

OGBench-side port of JaxGCRL's `crl_skill_controller` (`GoExploreSimple`
``agent_type="crl_skill"``): freeze a pretrained skill-conditioned policy
pi(a | s, z) and learn, online, a goal-conditioned high-level controller
pi_hi(z | s, g) over discrete skills on a Semi-MDP with fixed k-step temporal
commitment. Each SMDP transition is (s_t, z, s_{t+k}); the learner never reads the
macro reward -- the critic is purely contrastive.

Built from this repo's primitives (no JaxGCRL networks):

  * Contrastive critic Q(s, z, g): `GCDiscreteBilinearCritic` (ensemble of two)
    over (observation, one_hot(z)) and the goal observation, trained with the
    same in-batch sigmoid-BCE loss as `agents/crl.py`. Positives are future
    observations of the same episode, drawn at sample time by
    `TrajectoryReplayBuffer` with P(offset = j) proportional to (discount^k)^j over the remaining macro-rows
    (JaxGCRL: gamma measured in env steps, so gamma_macro = gamma^k per row).
  * Categorical actor pi_hi(z | s, g): `GCDiscreteActor`. Actor loss is the exact
    soft discrete objective from JaxGCRL,
        J = E_s sum_z pi(z | s, g) * (alpha * log pi(z | s, g) - Q(s, z, g)),
    with Q evaluated for *every* skill (critic head 0, no gradient).
  * alpha: `LogParam` auto-tuned with alpha * (H(pi) - H_target),
    H_target = target_entropy_frac * log(num_skills).
    The target is stated directly in the units of the policy it constrains: a categorical over
    num_skills attains its maximum entropy log(num_skills) at uniform, so target_entropy_frac in
    [0, 1] asks for that fraction of the maximum and is reachable by construction. (An unreachable
    target sends alpha to infinity, which erases the Q-learning signal from the actor loss and
    collapses the (temperature=0) eval policy onto one fixed skill regardless of state -- verified
    2026-09-02 on a K=50 checkpoint with a target of 4.0 > log(50) = 3.912.)
    `use_legacy_entropy=True` restores the pre-2026-09-13 formula,
    min(target_entropy_multiplier * action_dim, target_entropy_cap_frac * log(num_skills)), for
    reproducing old runs; see `_resolve_target_entropy`.
  * `use_tes=True`: Target Entropy Scheduled SAC (TES-SAC; Xu, Hu, Liang, McAleer, Abbeel, Fox,
    "Target Entropy Annealing for Discrete Soft Actor-Critic", NeurIPS 2021 DeepRL workshop,
    arXiv:2112.02852). The resolved `target_entropy` above is then only the INITIAL target
    H_0 (the paper uses H_0 = log|A|, i.e. target_entropy_frac=1.0); Algorithm 1 anneals it
    per gradient step from the mini-batch policy entropy e_t = mean_b H(pi(. | s_b, g_b)):
        delta = e_t - mu;  mu += (1 - lambda) * delta;  sigma^2 = lambda * (sigma^2 + (1 - lambda) * delta^2)
        if |mu - H| < mean_threshold and sigma <= std_threshold: i += 1
        if i >= T: i = 0; H *= target_discount
    (exponential moving mean / std of Finch 2009, Eqs. 9-10; mu is initialised to H_0, sigma to 0,
    and neither is reset on a drop). Paper Table 1: lambda=0.999, mean_threshold=0.01,
    std_threshold=0.05, target_discount=0.9. T ("total conditioned num") is NOT given in the paper
    and no code is public; `tes_patience` (default 500) is our choice: the moving mean itself needs ~4k steps to re-settle within 0.01
    nat of a 10%-lower target at lambda=0.999, so T mostly adds a short confirmation margin on top of that.
    The schedule state lives in `tes_state` (part of the agent pytree, so checkpoints carry it) and
    is advanced in `update` from the entropy the alpha loss just observed; the alpha loss at step t
    uses the target as of step t-1 (a one-update lag, immaterial at lambda=0.999).

Goals are full goal observations (OGBench convention): the behaviour policy sees
`info['goal']`; training goals are relabelled future observations.

Low-level execution goes through the frozen agent's own `skill_set()` /
`sample_actions_with_skill()` hooks (the contract `eval_skill_policy.py` uses), so
any checkpoint family exposing them works: `empowerment_skill` (one-hot skills),
`dds` (VQ codebook skills) and `opal` with `latent_type='discrete'` (one-hot skills
decoded by the Appendix-F BC decoder; a continuous OPAL VAE has no finite skill set
and is refused). The frozen agent is a plain pytree field, so `save_agent` writes a
full copy of it into every controller checkpoint.

RLPD labels for an `opal` checkpoint come from the OPAL posterior itself -- the same
labeller `opal_controller` uses offline: p(z | tau) by Bayes rule over the frozen
clustering mixture p_w(z) prod_t p_phi(s_t | s_{t-1}, z) (paper App. F, Eq. 71),
one label per offline k-window, sampled (`opal_label_mode='sample'`, the paper) or
taken as the argmax (`'mode'`). This mirrors the DDS setup, where the frozen
encoder + codebook labels the offline windows.

`skill_dt` (Skill Decision Transformer, `agents/skill_dt.py`) is the third family
and the one STATEFUL low level: its policy reads a K-step context of states and
re-encoded skills plus a future-skill histogram (paper Sec. A.5), so it cannot be
driven through the stateless `sample_actions_with_skill`. Instead
`init_low_level_state` / `low_level_actions_with_state` carry the paper's rollout
state across the env steps of an episode (the `MacroCollector` threads it), and a
skill z is executed by filling the unvisited tail of the histogram with z over the
remaining horizon -- exactly what `skill_dt_controller` does offline, and what the
paper's own per-skill evaluation does. RLPD labels (below) come from the frozen VQ
encoder: every state gets its codebook index, and each k-window its
`skill_dt_label_mode` reduction of those indices (default: the most frequent skill
in the window), mirroring the DDS setup where the frozen encoder labels the offline
windows.

Eval follows the repo's `init_eval_state` / `sample_actions_with_state` contract:
the argmax skill (at temperature 0) is held for `skill_commitment_k` env steps.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.dds import DDSAgent
from agents.empowerment_skill import EmpowermentAgent
from agents.opal import OPALAgent
from agents.skill_dt import SkillDTAgent
from agents.skill_dt_controller import LABEL_MODES as SKILL_DT_LABEL_MODES
from agents.skill_dt_controller import labels_from_skill_counts
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCDiscreteActor, GCDiscreteBilinearCritic, LogParam
from utils.skill_checkpoint import load_frozen_skill_agent

SKILL_AGENT_CLASSES = dict(empowerment_skill=EmpowermentAgent, dds=DDSAgent, skill_dt=SkillDTAgent, opal=OPALAgent)
OPAL_LABEL_MODES = ('sample', 'mode')
# Families whose frozen policy keeps per-episode rollout state (see `init_low_level_state`).
STATEFUL_SKILL_AGENTS = ('skill_dt',)


LEGACY_ENTROPY_KEYS = ('target_entropy_multiplier', 'target_entropy_cap_frac')


def _resolve_target_entropy(config, num_skills, ex_actions):
    """Resolve H_target for the categorical pi_hi(z | s, g).

    Default (`use_legacy_entropy=False`):

        H_target = target_entropy_frac * log(num_skills)

    stated in the units of the policy it constrains -- a categorical over num_skills maxes out at
    log(num_skills) (uniform), so any frac <= 1 is attainable by construction.

    Legacy (`use_legacy_entropy=True`), kept so pre-2026-09-13 runs stay reproducible:

        H_target = min(target_entropy_multiplier * action_dim, target_entropy_cap_frac * log(num_skills))

    where action_dim is the *low-level* env action dimensionality -- unrelated to num_skills, and
    only ever a live term when the cap does not bind.

    The two paths are mutually exclusive and never silently substitute for one another: setting a
    legacy knob without the gate (what an un-updated pre-2026-09-13 launcher does) raises, rather
    than quietly running the new formula under the old flags.

    The one exception is a config with no `use_legacy_entropy` key at all -- a `flags.json` written
    before the gate existed, replayed by the analysis scripts that rebuild an agent from its own
    checkpoint. There the legacy knobs are the only entropy knobs the run ever had, so the legacy
    path is selected automatically and the checkpoint is reproduced rather than rejected.
    """
    legacy_values = {k: config.get(k, None) for k in LEGACY_ENTROPY_KEYS}
    supplied = [k for k, v in legacy_values.items() if v is not None]

    use_legacy = config.get('use_legacy_entropy', None)
    if use_legacy is None:
        use_legacy = bool(supplied)
        if use_legacy:
            print(
                '[online_crl_skill_controller] config predates use_legacy_entropy and carries '
                f'{", ".join(supplied)}; reproducing it on the legacy target-entropy path.'
            )
    use_legacy = bool(use_legacy)

    if not use_legacy:
        if supplied:
            raise ValueError(
                f'[online_crl_skill_controller] {", ".join(supplied)} set, but use_legacy_entropy=False. '
                f'These knobs belong to the pre-2026-09-13 target-entropy formula '
                f'min(target_entropy_multiplier * action_dim, target_entropy_cap_frac * log(num_skills)) '
                f'and are ignored by the current one (target_entropy_frac * log(num_skills)). Either drop '
                f'them and pass --agent.target_entropy_frac, or pass --agent.use_legacy_entropy=True to '
                f'reproduce an old run verbatim.'
            )
        target_entropy = float(config['target_entropy_frac']) * float(np.log(num_skills))
        print(
            f'[online_crl_skill_controller] target_entropy = target_entropy_frac * log(num_skills) = '
            f"{float(config['target_entropy_frac']):.3f} * {float(np.log(num_skills)):.3f} = "
            f'{target_entropy:.3f} (num_skills={num_skills}).'
        )
        return target_entropy

    missing = [k for k in LEGACY_ENTROPY_KEYS if legacy_values[k] is None]
    if missing:
        raise ValueError(
            f'[online_crl_skill_controller] use_legacy_entropy=True requires {", ".join(missing)} to be '
            f'set explicitly (the legacy defaults were target_entropy_multiplier=0.5, '
            f'target_entropy_cap_frac=0.9). Pass them, or leave use_legacy_entropy=False to use '
            f'target_entropy_frac * log(num_skills).'
        )

    action_dim = ex_actions.shape[-1]
    uncapped_target_entropy = float(legacy_values['target_entropy_multiplier']) * float(action_dim)
    max_entropy = float(np.log(num_skills))
    target_entropy_cap = float(legacy_values['target_entropy_cap_frac']) * max_entropy
    target_entropy = min(uncapped_target_entropy, target_entropy_cap)
    # An unreachable target sends alpha to infinity, which erases the Q-learning signal from the
    # actor loss and collapses the (deterministic, temperature=0) eval policy onto one fixed skill
    # regardless of state (verified 2026-09-02: alpha reached ~1e17 by 1M steps on a K=50 checkpoint
    # with the unclamped multiplier * action_dim = 4.0 > log(50) = 3.912). The cap is what prevents
    # that here -- hence the legacy path's min().
    print(
        f'[online_crl_skill_controller] LEGACY target_entropy = min(multiplier * action_dim, '
        f'cap_frac * log(num_skills)) = min({uncapped_target_entropy:.3f}, {target_entropy_cap:.3f}) = '
        f'{target_entropy:.3f} (action_dim={action_dim}, num_skills={num_skills}); '
        f'target_entropy_frac is ignored on this path.'
    )
    return target_entropy


class OnlineCRLSkillControllerAgent(flax.struct.PyTreeNode):
    """Contrastive high-level controller pi_hi(z | s, g) over a frozen skill policy.

    Fields:
        rng: PRNG key (seeds sampling when no seed is supplied).
        network: `TrainState` over a `ModuleDict` with `actor` (GCDiscreteActor over
            the K skills), `critic` (GCDiscreteBilinearCritic) and `alpha` (LogParam).
        skill_agent: The frozen pretrained skill agent (never updated).
        config: Static configuration dictionary.
        tes_state: TES-SAC schedule state (`use_tes=True`): dict of scalar arrays `target`
            (current H_bar), `mu`, `sigma2` (exponential moving mean / variance of the batch
            policy entropy), `count` (stable steps i) and `drops` (number of target drops so far).
            None when `use_tes=False`.
    """

    rng: Any
    network: Any
    skill_agent: Any
    config: Any = nonpytree_field()
    tes_state: Any = None

    # ── Losses ────────────────────────────────────────────────────────────────

    def contrastive_loss(self, batch, grad_params):
        """In-batch contrastive critic loss over (s, one_hot(z)) vs. future goals; form of `agents/crl.py`."""
        batch_size = batch['observations'].shape[0]

        v, phi, psi = self.network.select('critic')(
            batch['observations'],
            batch['value_goals'],
            actions=batch['actions'],
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        v = jnp.exp(v)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }

    def _all_skill_values(self, observations, goals):
        """Q(s, z, g) for every skill z: (B, K), from critic head 0, no gradient.

        JaxGCRL's actor reads a single contrastive critic, so only ensemble head 0 is
        used here (both heads are still trained by the contrastive loss, as in
        agents/crl.py). phi(s, one_hot(z)) is evaluated for all K skills; psi(g) once per row.
        """
        num_skills = self.config['num_skills']
        batch_size = observations.shape[0]
        obs_b = jnp.broadcast_to(observations[:, None, ...], (batch_size, num_skills, *observations.shape[1:]))
        skills_b = jnp.broadcast_to(jnp.arange(num_skills)[None, :], (batch_size, num_skills))
        goals_b = jnp.broadcast_to(goals[:, None, ...], (batch_size, num_skills, *goals.shape[1:]))
        _, phi, psi = self.network.select('critic')(obs_b, goals_b, actions=skills_b, info=True)
        # phi: (E, B, K, d); psi: (E, B, K, d) (goal rows repeated along K).
        qs = (phi * psi[..., :1, :]).sum(-1) / jnp.sqrt(phi.shape[-1])  # (E, B, K)
        return jax.lax.stop_gradient(qs[0])

    def actor_loss(self, batch, grad_params, rng):
        """Exact soft categorical actor loss + alpha loss (JaxGCRL `crl_controller_update`)."""
        del rng  # The discrete objective enumerates every skill; nothing is sampled.
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_pi = jax.nn.log_softmax(dist.logits, axis=-1)  # (B, K)
        pi = jnp.exp(log_pi)
        entropy = -jnp.sum(pi * log_pi, axis=-1)  # (B,)  H(pi)

        q = self._all_skill_values(batch['observations'], batch['actor_goals'])  # (B, K)

        alpha = self.network.select('alpha')()
        # J = E_s sum_z pi(z | s, g) * (alpha * log pi(z | s, g) - Q(s, z, g))
        actor_loss = jnp.sum(pi * (alpha * log_pi - q), axis=-1).mean()

        alpha_param = self.network.select('alpha')(params=grad_params)
        entropy_sg = jax.lax.stop_gradient(entropy).mean()
        target_entropy = self._current_target_entropy()
        alpha_loss = alpha_param * (entropy_sg - target_entropy)

        total_loss = actor_loss + alpha_loss
        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': entropy_sg,
            'target_entropy': target_entropy,
            'q_pi_mean': jnp.sum(pi * q, axis=-1).mean(),
            'q_max_skill_mean': q.max(axis=-1).mean(),
            'pi_max_mean': jnp.mean(jnp.max(pi, axis=-1)),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    # ── TES-SAC target entropy schedule (arXiv:2112.02852, Algorithm 1) ──────

    def _current_target_entropy(self):
        """H_bar for the alpha loss: the scheduled target under `use_tes`, else the constant."""
        if self.config['use_tes']:
            return self.tes_state['target']
        return self.config['target_entropy']

    @staticmethod
    def init_tes_state(initial_target_entropy):
        """Algorithm 1, line 1: mu = H_0, sigma = 0, i = 0, H_bar = H_0."""
        h0 = jnp.asarray(initial_target_entropy, dtype=jnp.float32)
        return dict(
            target=h0,
            mu=h0,
            sigma2=jnp.zeros((), jnp.float32),
            count=jnp.zeros((), jnp.int32),
            drops=jnp.zeros((), jnp.int32),
        )

    def _tes_step(self, tes_state, batch_entropy):
        """One timestep of Algorithm 1 given e_t (the mini-batch policy entropy, Eq. 8).

        Lines 3-6 update the exponential moving mean / variance (Eqs. 9-10). Line 7: if mu is not
        within `tes_mean_threshold` of H_bar, or sigma exceeds `tes_std_threshold`, nothing else
        changes (as written, i is NOT reset by a failed check). Lines 10-14: otherwise i += 1 and,
        once i reaches `tes_patience` (T), i = 0 and H_bar *= `tes_target_discount`.
        """
        lam = jnp.asarray(self.config['tes_window_discount'], jnp.float32)
        e_t = jnp.asarray(batch_entropy, jnp.float32)
        delta = e_t - tes_state['mu']
        mu = tes_state['mu'] + (1.0 - lam) * delta
        sigma2 = lam * (tes_state['sigma2'] + (1.0 - lam) * delta**2)
        sigma = jnp.sqrt(sigma2)
        target = tes_state['target']
        stable = jnp.logical_and(
            jnp.abs(mu - target) < self.config['tes_mean_threshold'],
            sigma <= self.config['tes_std_threshold'],
        )
        count = tes_state['count'] + stable.astype(jnp.int32)
        drop = count >= int(self.config['tes_patience'])
        new_target = jnp.where(drop, target * self.config['tes_target_discount'], target)
        new_count = jnp.where(drop, 0, count)
        drops = tes_state['drops'] + drop.astype(jnp.int32)
        return dict(target=new_target, mu=mu, sigma2=sigma2, count=new_count, drops=drops)

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_agent = self.replace(network=new_network, rng=new_rng)
        if self.config['use_tes']:
            # Advance the schedule from the entropy the alpha loss just observed (Eq. 8 on this batch).
            tes_state = self._tes_step(self.tes_state, info['actor/entropy'])
            new_agent = new_agent.replace(tes_state=tes_state)
            info['actor/tes_mu'] = tes_state['mu']
            info['actor/tes_sigma'] = jnp.sqrt(tes_state['sigma2'])
            info['actor/tes_count'] = tes_state['count']
            info['actor/tes_drops'] = tes_state['drops']
        return new_agent, info

    # ── Acting: high level ────────────────────────────────────────────────────

    def _single_obs(self, observations):
        single_obs_ndim = 3 if self.config.get('encoder') is not None else 1
        return observations.ndim == single_obs_ndim

    @jax.jit
    def sample_skills(self, observations, goals, seed=None, temperature=1.0):
        """z ~ pi_hi(. | s, g) (temperature=0 -> argmax). Accepts a single obs or a batch."""
        if seed is None:
            seed = self.rng
        if goals is None:
            raise ValueError('online_crl_skill_controller needs a goal: pi_hi(z | s, g) is goal-conditioned.')
        single = self._single_obs(observations)
        obs_b = observations[None, ...] if single else observations
        goals_b = goals[None, ...] if single else goals
        dist = self.network.select('actor')(obs_b, goals_b, temperature=temperature)
        skills = dist.sample(seed=seed).astype(jnp.int32)
        return skills[0] if single else skills

    # ── Acting: low level (frozen skill policy) ───────────────────────────────

    def _low_temperature(self):
        low_temperature = float(self.config['low_temperature'])
        if self.skill_agent.config['discrete']:
            # A categorical at temperature exactly 0 divides logits by 0; the floor
            # keeps it a (near-)argmax without the NaN.
            low_temperature = max(low_temperature, 1e-6)
        return low_temperature

    def _stateful_low_level(self):
        """Whether the frozen skill policy keeps per-episode rollout state (Skill-DT's K-step context)."""
        return self.config['skill_agent_name'] in STATEFUL_SKILL_AGENTS

    def _require_stateless(self, hook):
        if self._stateful_low_level():
            raise ValueError(
                f'online_crl_skill_controller.{hook}: the frozen {self.config["skill_agent_name"]!r} policy is '
                f'stateful (its Transformer reads a K-step context, paper Sec. A.5), so it has no stateless '
                f'per-step actor. Use init_low_level_state / low_level_actions_with_state (the MacroCollector '
                f'does) or init_eval_state / sample_actions_with_state (the evaluators do).'
            )

    @jax.jit
    def low_level_actions(self, observations, skills, seed=None):
        """a ~ pi(. | s, z) from the frozen skill policy for a single observation and skill index."""
        self._require_stateless('low_level_actions')
        if seed is None:
            seed = self.rng
        skill_vectors = self.skill_agent.skill_set()  # (K, skill_width): one-hots or codebook rows
        skill_vector = skill_vectors[jnp.asarray(skills, dtype=jnp.int32)]
        return self.skill_agent.sample_actions_with_skill(
            observations, skill_vector, seed=seed, temperature=self._low_temperature()
        )

    # ── Stateful frozen low level (skill_dt) ───────────────────────────────────

    def init_low_level_state(self, max_steps=None):
        """Per-episode state of the frozen low-level policy, or None for the stateless families.

        For `skill_dt` this is the paper's Sec. A.5 rollout state (`SkillDTAgent.init_eval_state`):
        the length-K context buffers, the env-step counter that drives the timestep embedding,
        and the histogram tail over the episode horizon `max_steps` (the checkpoint's own
        `eval_max_steps` wins if set, as for a plain Skill-DT rollout). Built once per episode
        by the `MacroCollector` and by `init_eval_state`; the skill placed in it is a
        placeholder that `low_level_actions_with_state` overwrites at every step.
        """
        if not self._stateful_low_level():
            return None
        return self.skill_agent.init_eval_state(skill=0, max_steps=max_steps)

    @jax.jit
    def low_level_actions_with_state(self, observations, skills, low_state, seed=None):
        """`low_level_actions` for the stateful frozen policy: returns `(action, new_low_state)`.

        Skill-DT executes skill z the way the paper's own per-skill evaluation does (Sec. A.5):
        the unvisited tail of the future-skill histogram is filled with z over the remaining
        horizon, while the observed context (states, re-encoded skills, step counter) is kept
        -- across env steps AND across the controller's skill switches, exactly as
        `skill_dt_controller.sample_actions_with_state` does offline. The tail is a pure
        function of (z, remaining horizon), so rebuilding it every step is identical to
        rebuilding it only when z changes.
        """
        if seed is None:
            seed = self.rng
        num_skills = int(self.config['num_skills'])
        L = low_state['tail_suffix'].shape[0] - 1  # episode horizon (static)
        skill = jnp.asarray(skills, dtype=jnp.int32)
        state = {
            **low_state,
            'skill': skill,
            'tail_suffix': self.skill_agent._constant_skill_tail(skill, num_skills, L),
        }
        return self.skill_agent.sample_actions_with_state(
            observations, goals=None, agent_state=state, seed=seed, temperature=self._low_temperature()
        )

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Stateless hierarchical action: reselect the skill every step (k=1 behaviour)."""
        self._require_stateless('sample_actions')
        if seed is None:
            seed = self.rng
        high_seed, low_seed = jax.random.split(seed)
        skill = self.sample_skills(observations, goals, seed=high_seed, temperature=temperature)
        return self.low_level_actions(observations, skill, seed=low_seed)

    # ── k-step skill commitment at eval (contract used by utils/evaluation.py) ─

    def init_eval_state(self, max_steps=None):
        """Per-episode state: the committed skill and the step counter.

        For the stateful `skill_dt` low level the state is its own rollout state, which
        already carries `skill` and `count` (env steps so far) and additionally the
        Transformer context; `max_steps` is the env horizon the evaluators hand over.
        """
        if self._stateful_low_level():
            return self.init_low_level_state(max_steps=max_steps)
        return {'skill': jnp.zeros((), jnp.int32), 'count': jnp.zeros((), jnp.int32)}

    @jax.jit
    def sample_actions_with_state(self, observations, goals=None, agent_state=None, seed=None, temperature=1.0):
        """`sample_actions` with the skill held for `skill_commitment_k` env steps (single-obs eval)."""
        if seed is None:
            seed = self.rng
        if agent_state is None:
            agent_state = self.init_eval_state()
        high_seed, low_seed = jax.random.split(seed)

        k = int(self.config['skill_commitment_k'])
        reselect = (agent_state['count'] % k) == 0
        sampled = self.sample_skills(observations, goals, seed=high_seed, temperature=temperature)
        skill = jnp.where(reselect, sampled, agent_state['skill']).astype(jnp.int32)
        if self._stateful_low_level():
            # The frozen policy's state is the eval state; it advances `count` itself.
            return self.low_level_actions_with_state(observations, skill, agent_state, seed=low_seed)
        actions = self.low_level_actions(observations, skill, seed=low_seed)
        new_state = {'skill': skill, 'count': agent_state['count'] + 1}
        return actions, new_state

    # ── Offline window labelling (RLPD; see utils/rlpd.py) ───────────────────

    @jax.jit
    def chunk_skill_logliks(self, observations, actions):
        """Per-step log pi(a | s, z) of the frozen `empowerment_skill` policy for every skill: [B, K].

        The `skill_bc_relabel_controller` labeller; `SequenceDataset.relabel_chunk_skills`
        turns these into window sums with a prefix sum and takes the argmax per window.
        """
        num_skills = int(self.config['num_skills'])
        batch_size = jax.tree_util.tree_leaves(observations)[0].shape[0]
        eye = jnp.eye(num_skills)

        if self.skill_agent.config['discrete']:
            targets = actions
        else:
            # A tanh-squashed actor's log_prob is +inf at |a| == 1 (OGBench actions do hit +-1).
            targets = jnp.clip(actions, -1.0 + 1e-6, 1.0 - 1e-6)

        def loglik_for_skill(skill):
            skills_onehot = jnp.broadcast_to(eye[skill], (batch_size, num_skills))
            dist = self.skill_agent.network.select('policy')(observations, skills_onehot)
            return dist.log_prob(targets)  # [B]

        return jax.lax.map(loglik_for_skill, jnp.arange(num_skills)).T  # [B, K]

    @jax.jit
    def label_chunk_skills(self, observations_seq, actions_seq, seq_mask, seed):
        """One skill index per window from the frozen window-level labeller: int32 [B].

        `dds`:  the encoder + codebook nearest neighbour (the `dds_controller` labeller).
        `opal`: the discrete posterior p(z | tau) of the frozen clustering model, by Bayes
                rule over the mixture (paper App. F Eq. 71; the `opal_controller` labeller):
                log p_phi(tau | z) = sum_{i>=1} mask_i * log p_phi(s_{t+i} | s_{t+i-1}, z), so a
                window cut short by its trajectory end sums fewer terms. Only the STATE
                trajectory is read -- never the actions. `opal_label_mode='sample'` draws
                z ~ p(z | tau) (what the paper does), `'mode'` takes the argmax.
        """
        family = self.config['skill_agent_name']
        if family == 'dds':
            del seed
            return self.skill_agent._assign_skill(observations_seq, actions_seq, seq_mask).astype(jnp.int32)
        if family != 'opal':
            raise ValueError(f'label_chunk_skills: no window-level labeller for skill agent {family!r}.')

        num_skills = int(self.config['num_skills'])
        B, C = seq_mask.shape
        prev = observations_seq[:, :-1]
        deltas = observations_seq[:, 1:] - prev
        step_mask = seq_mask[:, 1:]
        eye = jnp.eye(num_skills)

        # `lax.map` (not vmap) over the K skills: the labelling pass pushes tens of
        # thousands of windows at a time, and vmapping would materialise K copies of
        # every hidden activation at once (same choice as opal_controller).
        def log_p_for_skill(skill):
            zs = jnp.broadcast_to(eye[skill], (B, C - 1, num_skills))
            dist = self.skill_agent.network.select('traj_model')(jnp.concatenate([prev, zs], axis=-1))
            return (dist.log_prob(deltas) * step_mask).sum(axis=-1)  # [B]

        log_p_tau = jax.lax.map(log_p_for_skill, jnp.arange(num_skills))  # [K, B]
        log_prior = jax.nn.log_softmax(self.skill_agent.network.select('skill_prior')())
        log_post = jax.nn.log_softmax(log_prior[:, None] + log_p_tau, axis=0).T  # [B, K]
        if self.config['opal_label_mode'] == 'sample':
            labels = jax.random.categorical(seed, log_post, axis=-1)
        else:
            labels = jnp.argmax(log_post, axis=-1)
        return labels.astype(jnp.int32)

    def encode_skill_indices(self, observations):
        """Discrete skill index of each state under the frozen Skill-DT VQ encoder: [B] int32.

        The hook `SequenceDataset.relabel_skill_histograms` calls (the `skill_dt_controller`
        labeller). This config has no `relabel_interval`, so main_online.py never re-runs it.
        """
        return self.skill_agent.encode_skill_indices(observations)

    def label_offline_windows(self, seq_dataset, seed=0):
        """Label every window [t, t + k) of an offline `SequenceDataset` with the frozen agent's labeller.

        Dispatches on the skill family: `empowerment_skill` uses the BC log-likelihood
        argmax, `dds` the encoder + codebook assignment, `skill_dt` the VQ encoder's
        per-state codebook indices reduced over the window (`skill_dt_label_mode`),
        `opal` the discrete posterior p(z | tau) over the window (`opal_label_mode`).
        Returns `(labels [size] int32, stats)`.
        """
        k = int(self.config['skill_commitment_k'])
        assert int(seq_dataset.config['sequence_length']) == k, (
            f'offline windows must be skill_commitment_k={k} long, got {seq_dataset.config["sequence_length"]}.'
        )
        family = self.config['skill_agent_name']
        if family == 'empowerment_skill':
            stats = seq_dataset.relabel_chunk_skills(self)
        elif family == 'dds':
            sequence_length = int(self.skill_agent.config['sequence_length'])
            if sequence_length != k:
                raise ValueError(
                    f'The DDS encoder scores windows of exactly sequence_length={sequence_length} steps, so '
                    f'offline windows cannot be labelled with skill_commitment_k={k}; use k={sequence_length} '
                    f'or a checkpoint trained with sequence_length={k}.'
                )
            stats = seq_dataset.relabel_chunk_skills_from_windows(
                self, seed=seed, num_skills=int(self.config['num_skills'])
            )
        elif family == 'opal':
            # The `opal_controller.prepare_datasets` pass: the posterior reads the whole
            # window through the per-step transition model, so any k is well defined
            # (`create` warns when k differs from the chunk_size the mixture was fit on).
            if self.config['opal_label_mode'] not in OPAL_LABEL_MODES:
                raise ValueError(
                    f'opal_label_mode must be one of {OPAL_LABEL_MODES}, got {self.config["opal_label_mode"]!r}.'
                )
            stats = seq_dataset.relabel_chunk_skills_from_windows(
                self,
                seed=seed,
                num_skills=int(self.config['num_skills']),
                chunk_bytes=int(self.config['label_chunk_bytes']),
            )
        elif family == 'skill_dt':
            # The same two-step pass as `skill_dt_controller.prepare_datasets`: one frozen
            # encoder sweep gives every state its codebook index (nothing ties k to the
            # checkpoint -- the encoder is per-state), then each window [t, t + k) takes
            # its `skill_dt_label_mode` reduction of those indices.
            num_skills = int(self.config['num_skills'])
            label_mode = self.config['skill_dt_label_mode']
            if label_mode not in SKILL_DT_LABEL_MODES:
                raise ValueError(f'skill_dt_label_mode must be one of {SKILL_DT_LABEL_MODES}, got {label_mode!r}.')
            seq_dataset.relabel_skill_histograms(
                self, chunk_bytes=int(self.config['label_chunk_bytes']), num_skills=num_skills
            )
            stats = seq_dataset.set_chunk_skills(
                labels_from_skill_counts(seq_dataset, k, label_mode), num_skills=num_skills
            )
        else:
            raise ValueError(f'No offline window labeller for skill agent {family!r}.')
        return np.asarray(seq_dataset.chunk_skills, dtype=np.int32), stats

    # ── Skill-conditioned evaluation hooks (see eval_skill_policy.py) ─────────

    def skill_set(self, seed=None, num_skills=None, observations=None):
        return self.skill_agent.skill_set(seed=seed, num_skills=num_skills, observations=observations)

    def sample_actions_with_skill(self, observations, skills, seed=None, temperature=1.0):
        # A skill_dt low level has no stateless per-step actor; sweep its skills on the
        # frozen checkpoint itself (eval_skill_policy.py --run_dir <skill_dt run>).
        self._require_stateless('sample_actions_with_skill')
        del temperature  # Reproduce the frozen policy's own execution at low_temperature.
        return self.skill_agent.sample_actions_with_skill(
            observations, skills, seed=seed, temperature=self._low_temperature()
        )

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations (also the example goals).
            ex_actions: Example batch of *low-level* actions (shapes the frozen skill agent).
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        if int(config['skill_commitment_k']) < 1:
            raise ValueError(f"skill_commitment_k must be >= 1, got {config['skill_commitment_k']}.")

        # ── Frozen low-level skill policy ────────────────────────────────────
        skill_agent, resolved = load_frozen_skill_agent(
            seed, ex_observations, ex_actions, config, SKILL_AGENT_CLASSES, caller='online_crl_skill_controller'
        )
        num_skills = resolved['num_skills']
        if resolved['agent_name'] == 'dds':
            seq_len = int(resolved['skill_config'].get('sequence_length', 0))
            if seq_len and int(config['skill_commitment_k']) != seq_len:
                print(
                    f'[online_crl_skill_controller] WARNING: skill_commitment_k={config["skill_commitment_k"]} '
                    f'differs from the DDS checkpoint\'s sequence_length={seq_len} (the horizon its skills were '
                    f'trained for; dds_controller defaults to it).'
                )

        if resolved['agent_name'] == 'opal':
            latent_type = resolved['skill_config'].get('latent_type')
            if latent_type != 'discrete':
                raise ValueError(
                    f'[online_crl_skill_controller] the opal checkpoint {resolved["ckpt_path"]} has '
                    f'latent_type={latent_type!r}; the controller is a categorical pi_hi(z | s, g) over a finite '
                    f'skill set, which only the discrete (App. F) OPAL path has. Use a latent_type=discrete run.'
                )
            chunk_size = int(resolved['skill_config'].get('chunk_size', 0))
            if chunk_size and int(config['skill_commitment_k']) != chunk_size:
                print(
                    f'[online_crl_skill_controller] WARNING: skill_commitment_k={config["skill_commitment_k"]} '
                    f'differs from the OPAL checkpoint\'s chunk_size={chunk_size} (the window its clustering '
                    f'posterior and decoder were trained on; opal_controller requires equality).'
                )

        # ── Trainable controller: actor + contrastive critic + alpha ─────────
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        ex_goals = ex_observations
        ex_skills = np.zeros((ex_observations.shape[0],), dtype=np.int32)

        actor_def = GCDiscreteActor(
            hidden_dims=tuple(config['actor_hidden_dims']),
            action_dim=num_skills,
            gc_encoder=encoders.get('actor'),
        )
        critic_def = GCDiscreteBilinearCritic(
            hidden_dims=tuple(config['value_hidden_dims']),
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
            state_encoder=encoders.get('critic_state'),
            goal_encoder=encoders.get('critic_goal'),
            action_dim=num_skills,
        )
        alpha_def = LogParam()

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_skills)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))

        # Resolved values for the agent's own use (main_online.py serialises FLAGS before
        # create, so these do not reach flags.json unless passed explicitly).
        stored_config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        stored_config['num_skills'] = num_skills
        stored_config['skill_restore_epoch'] = resolved['restore_epoch']
        stored_config['skill_checkpoint_path'] = resolved['ckpt_path']
        stored_config['skill_agent_name'] = resolved['agent_name']
        stored_config['target_entropy'] = _resolve_target_entropy(config, num_skills, ex_actions)
        # Future-goal sampling discount per macro-row: gamma^k (gamma measured in env steps).
        stored_config['goal_discount'] = float(config['discount']) ** int(config['skill_commitment_k'])

        tes_state = None
        if config['use_tes']:
            if int(config['tes_patience']) < 1:
                raise ValueError(f"tes_patience must be >= 1, got {config['tes_patience']}.")
            if not (0.0 < float(config['tes_target_discount']) < 1.0):
                raise ValueError(f"tes_target_discount must lie in (0, 1), got {config['tes_target_discount']}.")
            if not (0.0 <= float(config['tes_window_discount']) < 1.0):
                raise ValueError(f"tes_window_discount must lie in [0, 1), got {config['tes_window_discount']}.")
            tes_state = cls.init_tes_state(stored_config['target_entropy'])
            print(
                f'[online_crl_skill_controller] TES-SAC on: target_entropy={stored_config["target_entropy"]:.3f} is '
                f'the INITIAL target H_0 (paper: log(num_skills)={float(np.log(num_skills)):.3f}); '
                f'lambda={config["tes_window_discount"]}, mean_thr={config["tes_mean_threshold"]}, '
                f'std_thr={config["tes_std_threshold"]}, k={config["tes_target_discount"]}, T={config["tes_patience"]}.'
            )

        return cls(
            rng,
            network=network,
            skill_agent=skill_agent,
            config=flax.core.FrozenDict(**stored_config),
            tes_state=tes_state,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='online_crl_skill_controller',  # Agent name.
            rollout_type='macro',  # Experience collector (see utils/online_rollout.py).
            lr=3e-4,  # Learning rate (actor, critic and alpha).
            batch_size=1024,  # Batch size (macro-transitions).
            actor_hidden_dims=(512, 512, 512),  # Controller actor hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Contrastive critic hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount per env step; macro-row goal sampling uses discount ** skill_commitment_k.
            # Frozen skill policy.
            skill_checkpoint_path=ml_collections.config_dict.placeholder(str),  # Skill-agent run dir (flags.json + params_*.pkl).
            skill_restore_epoch=ml_collections.config_dict.placeholder(int),  # Pretrained epoch (None -> latest).
            num_skills=ml_collections.config_dict.placeholder(int),  # Read from the checkpoint; asserted if set.
            low_temperature=0.0,  # Temperature of the frozen low-level policy (0 -> mode).
            # SMDP.
            skill_commitment_k=20,  # Fixed temporal commitment: env steps per high-level decision.
            gamma_low=1.0,  # Intra-macro-step reward discount (bookkeeping only; the learner ignores rewards).
            target_entropy_frac=0.5,  # H_target = frac * log(num_skills) (frac <= 1 -> always reachable).
            use_legacy_entropy=False,  # True -> pre-2026-09-13 H_target (requires both knobs below).
            target_entropy_multiplier=ml_collections.config_dict.placeholder(float),  # Legacy only; raises unless use_legacy_entropy.
            target_entropy_cap_frac=ml_collections.config_dict.placeholder(float),  # Legacy only; raises unless use_legacy_entropy.
            # TES-SAC target entropy annealing (arXiv:2112.02852, Algorithm 1). With use_tes=True the
            # resolved target_entropy above is only the initial target H_0 (paper: log|A|, i.e.
            # target_entropy_frac=1.0). Values below are the paper's Table 1, except tes_patience (T),
            # which the paper leaves unspecified.
            use_tes=False,  # Anneal H_target by tes_target_discount whenever the policy entropy stabilises at it.
            tes_window_discount=0.999,  # lambda: exponential moving mean / std discount of the batch entropy.
            tes_mean_threshold=0.01,  # Drop only if |mu - H_target| < this (nats).
            tes_std_threshold=0.05,  # ... and the moving std of the batch entropy is <= this (nats).
            tes_target_discount=0.9,  # k: H_target <- k * H_target on every drop.
            tes_patience=500,  # T: gradient steps satisfying the check (not necessarily consecutive) per drop.
            # Online schedule (consumed by main_online.py); units are macro-steps.
            unroll_length=50,  # Macro-steps collected between update rounds.
            utd_ratio=1,  # Gradient steps per macro-step; each round runs unroll_length * utd_ratio updates.
            min_replay_size=1000,  # Macro-transitions collected before the first update.
            replay_size=50000,  # Replay buffer capacity in macro-transitions.
            offline_ratio=0.5,  # RLPD (--offline_dataset): fraction of every batch drawn from the offline buffer.
            # RLPD: drop offline windows the frozen skill policy cannot reproduce. The score is the
            # winning skill's BC log-likelihood per step and per action dimension (so it is comparable
            # across envs); None -> keep every window, the pre-filter behaviour.
            offline_loglik_threshold=ml_collections.config_dict.placeholder(float),
            # RLPD with a `skill_dt` checkpoint: how each offline k-window gets its codebook label
            # from the frozen VQ encoder's per-state skill indices -- 'window_mode' (the most
            # frequent skill over the window; the discrete analogue of DDS's encoder label),
            # 'end_state' (the skill of s_{t+k}) or 'future_hist' (argmax of the paper's Z_t); see
            # agents/skill_dt_controller.py. Ignored by the other families.
            skill_dt_label_mode='window_mode',
            # RLPD with a discrete `opal` checkpoint: 'sample' draws each offline k-window's label
            # z ~ p(z | tau) from the frozen clustering posterior (the paper's recipe and
            # opal_controller's default), 'mode' takes its argmax. Ignored by the other families.
            opal_label_mode='sample',
            # Observation bytes per labeller block (skill_dt encoder pass; opal posterior pass, where
            # the K-way transition model keeps a whole block's hidden activations live).
            label_chunk_bytes=64 * 1024 * 1024,
            # Observation pipeline (must match the skill checkpoint).
            discrete=False,  # Whether the low-level action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None for state-based).
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
