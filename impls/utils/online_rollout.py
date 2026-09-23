"""Experience collectors for online training: one env, one transition at a time.

Two collectors share one interface, `step(agent) -> dict`, and push compact rows
into a `TrajectoryReplayBuffer`:

  * `FlatCollector`   -- one env step per row: (s, a, r, mask, done, E(s), is_offline=0).
    Used by the flat online CRL agent. The behaviour policy is the stochastic actor
    (`temperature=1`), as in JaxGCRL's `actor_step`. `E(s)` is the row's offline
    empowerment estimate (agents/online_crl.py, `emp_checkpoint_path`): rows are written
    with a NaN placeholder and `flush_empowerment(agent)` fills the pending rows in one
    batched call, which `main_online.py` runs before every update round. The marker row
    that closes an episode (the final observation) is filled the same way, so every
    transition's E(s') exists (the reward of the exploration bonus). Each row also gets
    `episodic_max_empowerment`, the running max of E over its episode up to and including
    the row (the `max_episodic_empowerment` bonus reward); it is derived at flush time
    from the freshly written E values and the row before it. Agents without an
    estimator store 0 and never read either field. `is_offline` is 0 on every online
    row and 1 on RLPD rows (utils/rlpd.py); the exploration bonus critic masks on it.
    For the distilled bonus (`add_explore=distill*`, the agent's `uses_distill_bonus`) rows
    also carry `distill_target`, the max of E over the row's own episode (final observation
    included; the agent's `distill_target` config: 'episode_max' = the whole episode, one value
    shared by all its rows, 'future_max' = only the states after the row) and a
    `distill_target_ready` flag: the target is only known once the episode has ended, so rows are
    written NaN / 0 and the first `flush_empowerment` after the episode closes fills the whole
    episode and sets the flag. Other agents store 0 / 0 and never read them.
  * `MacroCollector`  -- one SMDP macro-step per row: the high-level agent picks a
    skill z, which the frozen low-level policy executes for `k` env steps (or
    until the episode ends). The row is (s_t, z, R, mask, done) with
    R = sum_i gamma_low^i (r_i + reward_shift) over the steps actually taken, mirroring
    `rollout_macro_step` in JaxGCRL's crl_skill_controller (`reward_shift` is 0 there;
    SUPE's agent sets -1 to train on -1/0 rewards, see agents/supe.py). Used by the online
    CRL skill controller (integer skill index) and by SUPE (a float latent u in (-1, 1)^D:
    the row's `actions` field takes the dtype/shape of the agent's `example_skill()` hook
    when it has one, else an int32 index). An agent with `sample_skills_takes_env_steps`
    gets the running env-step count in `sample_skills(env_steps=...)` (SUPE's prior warm-up).
    A frozen policy that keeps per-episode state (Skill-DT's Transformer context) exposes
    `init_low_level_state(max_steps)` / `low_level_actions_with_state(...)`; the collector
    threads that state through every env step of the episode and rebuilds it at reset.
    Macro rows carry the same `empowerment` / `episodic_max_empowerment` / `is_offline` /
    `distill_target` / `distill_target_ready` fields as flat rows, filled the same way
    (E of the macro-step's start state; the marker holds the episode's final state), so a
    macro agent can run the distilled bonus unchanged. Agents without an estimator store 0.

Both condition the behaviour policy on the episode's task goal (`info['goal']`,
the full goal observation) and never store it: training goals are relabelled
from future observations at sample time, exactly as in JaxGCRL.

Which collector an agent needs is declared by its config (`rollout_type`).
"""

import jax
import numpy as np

from utils.evaluation import env_horizon


class _EpisodeTracker:
    """Per-episode return / length / success bookkeeping shared by the collectors."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.ep_return = 0.0
        self.ep_length = 0
        self.ep_success = 0.0

    def add(self, reward, info):
        self.ep_return += float(reward)
        self.ep_length += 1
        self.ep_success = float(info.get('success', self.ep_success))

    def summary(self):
        return dict(episode_return=self.ep_return, episode_length=self.ep_length, episode_success=self.ep_success)


def _extra_row_fields():
    """The per-row fields (beyond s, a, r, mask, done) both collectors store; see the module docstring."""
    return dict(
        empowerment=np.float32(0.0),
        episodic_max_empowerment=np.float32(0.0),
        is_offline=np.float32(0.0),
        distill_target=np.float32(0.0),
        distill_target_ready=np.float32(0.0),
    )


class _Collector:
    """Shared state and the empowerment / distilled-bonus row bookkeeping of both collectors."""

    def __init__(self, env, buffer, seed):
        self.env = env
        self.buffer = buffer
        self.rng = jax.random.PRNGKey(seed)
        self.tracker = _EpisodeTracker()
        self._pending_empowerment = []  # abs indices of rows whose `empowerment` is still the NaN placeholder
        self._pending_first = []  # parallel to the above: does the row start a new episode?
        self._episode_first_row = True  # the next row written starts a new episode
        self._episode_start_abs = None  # abs index of the current episode's first row (distilled bonus only)
        self._closed_episodes = []  # (first row abs, marker abs) of episodes closed since the last flush

    def _reset_episode(self):
        observation, info = self.env.reset()
        self.observation = observation
        self.goal = info['goal']
        self.tracker.reset()
        self._episode_first_row = True

    def _mark_pending(self, abs_idx):
        """Queue a row for `flush_empowerment` (NaN placeholders until then)."""
        self._pending_empowerment.append(abs_idx)
        self._pending_first.append(self._episode_first_row)
        self._episode_first_row = False

    def _row_extras(self, agent):
        """The extra fields of a fresh online row: NaN placeholders where the agent will fill them."""
        uses_empowerment = bool(getattr(agent, 'uses_empowerment', False))
        uses_distill = bool(getattr(agent, 'uses_distill_bonus', False))
        return dict(
            # NaN until `flush_empowerment`: sampling an unfilled row would surface as a NaN loss
            # rather than silently training on a wrong entropy target.
            empowerment=np.float32(np.nan if uses_empowerment else 0.0),
            episodic_max_empowerment=np.float32(np.nan if uses_empowerment else 0.0),
            is_offline=np.float32(0.0),
            # NaN + ready=0 until the episode closes (`_flush_distill_targets`); the agent's regression
            # masks on the flag, so a NaN can only surface if a row is flagged ready without a target.
            distill_target=np.float32(np.nan if uses_distill else 0.0),
            distill_target_ready=np.float32(0.0),
        )

    def _after_add(self, agent, abs_idx):
        """Bookkeeping after a real row was written at `abs_idx`."""
        if getattr(agent, 'uses_distill_bonus', False) and self._episode_start_abs is None:
            self._episode_start_abs = abs_idx
        if getattr(agent, 'uses_empowerment', False):
            self._mark_pending(abs_idx)

    def _close_episode(self, agent, final_observation):
        """Write the marker row for the episode's final observation (+ its placeholders); returns its abs index."""
        end_abs = self.buffer.end_trajectory(final_observation)
        if getattr(agent, 'uses_empowerment', False):
            # The marker row is never an anchor but it IS the last transition's next state,
            # whose E(s') the exploration bonus reads: give it the same NaN-until-flushed slots.
            nan = np.array([np.nan], dtype=np.float32)
            self.buffer.write_field('empowerment', [end_abs], nan)
            self.buffer.write_field('episodic_max_empowerment', [end_abs], nan)
            self._mark_pending(end_abs)
        if getattr(agent, 'uses_distill_bonus', False):
            self._closed_episodes.append((self._episode_start_abs, end_abs))
            self._episode_start_abs = None
        return end_abs

    def flush_empowerment(self, agent):
        """Fill `empowerment` and `episodic_max_empowerment` of every row added since the last flush.

        One batched estimator call for E; the running max walks the pending rows in order
        (they are contiguous and in write order), restarting at each episode's first row
        and otherwise taking the max of the row's E and the previous row's running max
        (already in the buffer, or computed earlier in this walk). Returns the number of
        rows filled. A no-op for agents without an estimator (their rows were written
        with 0, not NaN, so the field is never a placeholder).
        """
        if not self._pending_empowerment:
            return 0
        abs_idxs = np.asarray(self._pending_empowerment, dtype=np.int64)
        firsts = self._pending_first
        self._pending_empowerment = []
        self._pending_first = []
        self.rng, key = jax.random.split(self.rng)
        observations = self.buffer.read_field('observations', abs_idxs)
        values = agent.empowerment_np(observations, key).astype(np.float32)
        self.buffer.write_field('empowerment', abs_idxs, values)

        running = np.empty_like(values)
        for i, (abs_idx, first) in enumerate(zip(abs_idxs, firsts)):
            if first:
                running[i] = values[i]
            else:
                prev = running[i - 1] if i > 0 and abs_idxs[i - 1] == abs_idx - 1 else (
                    self.buffer.read_field('episodic_max_empowerment', [abs_idx - 1])[0]
                )
                running[i] = max(values[i], prev)
        assert np.all(np.isfinite(running)), 'episodic running max hit an unfilled predecessor row'
        self.buffer.write_field('episodic_max_empowerment', abs_idxs, running)
        if getattr(agent, 'uses_distill_bonus', False):
            self._flush_distill_targets(agent.config['distill_target'])
        return int(len(abs_idxs))

    def _flush_distill_targets(self, kind):
        """Distilled bonus: write the E' target onto every row of the episodes closed since the last flush.

        `kind='episode_max'`: max_k E(s_k) over the whole episode, the same value on every row;
        `'future_max'`: max_{k>t} E(s_k), one reverse cummax. Runs right after the E values were
        written, so a closed episode's rows (marker included) all hold a real E. The marker
        row keeps its placeholder (it is never an anchor).
        """
        closed, self._closed_episodes = self._closed_episodes, []
        for start_abs, end_abs in closed:
            start_abs = max(start_abs, self.buffer.oldest_abs)  # rows already evicted need no target
            if start_abs >= end_abs:
                continue
            rows = np.arange(start_abs, end_abs + 1)
            values = self.buffer.read_field('empowerment', rows)
            assert np.all(np.isfinite(values)), 'distill target: a closed episode still holds an unfilled E'
            if kind == 'episode_max':
                # Rows evicted before the episode closed (episode longer than the buffer) drop out of the max.
                target = np.full(len(rows) - 1, values.max())
            else:
                target = np.maximum.accumulate(values[::-1])[::-1][1:]  # [t] = max(values[t + 1:])
            self.buffer.write_field('distill_target', rows[:-1], target.astype(np.float32))
            self.buffer.write_field('distill_target_ready', rows[:-1], np.ones(len(rows) - 1, dtype=np.float32))


class FlatCollector(_Collector):
    """Per-env-step collector for flat goal-conditioned agents."""

    def __init__(self, env, buffer, seed, discrete=False):
        super().__init__(env, buffer, seed)
        self.discrete = discrete
        self._reset_episode()

    @staticmethod
    def example_transition(example_batch, agent=None):
        """Example row (used to allocate the buffer) in this collector's layout."""
        return dict(
            observations=example_batch['observations'][0],
            actions=example_batch['actions'][0],
            rewards=np.float32(0.0),
            masks=np.float32(1.0),
            terminals=np.float32(0.0),
            **_extra_row_fields(),
        )

    def step(self, agent):
        self.rng, key = jax.random.split(self.rng)
        action = agent.sample_actions(observations=self.observation, goals=self.goal, seed=key, temperature=1.0)
        action = np.asarray(action)
        if not self.discrete:
            action = np.clip(action, -1, 1)

        next_observation, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        self.tracker.add(reward, info)

        abs_idx = self.buffer.add_transition(
            dict(
                observations=self.observation,
                actions=action,
                rewards=np.float32(reward),
                masks=np.float32(1.0 - float(terminated)),
                terminals=np.float32(done),
                **self._row_extras(agent),
            )
        )
        self._after_add(agent, abs_idx)
        self.observation = next_observation

        episode = None
        if done:
            self._close_episode(agent, next_observation)
            episode = self.tracker.summary()
            self._reset_episode()
        return dict(env_steps=1, rows=1, episode=episode)


class MacroCollector(_Collector):
    """Per-macro-step collector for a high-level skill controller over a frozen skill policy."""

    def __init__(self, env, buffer, seed, skill_commitment_k, gamma_low=1.0, reward_shift=0.0):
        super().__init__(env, buffer, seed)
        self.k = int(skill_commitment_k)
        self.gamma_low = float(gamma_low)
        self.reward_shift = float(reward_shift)
        self.env_steps = 0  # env steps taken so far (handed to agents whose skill choice depends on it)
        # Episode horizon handed to a STATEFUL frozen low-level policy (Skill-DT sizes its
        # rollout histogram with it); None if the env has no TimeLimit.
        self.horizon = env_horizon(env)
        # Per-episode state of the frozen low-level policy, or None for the stateless
        # families (empowerment_skill, dds). Built lazily on the first `step` of every
        # episode because the agent is not known at construction time.
        self.low_state = None
        self._low_state_stale = True
        # Integer skill index (the CRL controllers) or a float latent (SUPE), from the buffer's layout.
        self._integer_skills = np.issubdtype(buffer.read_field('actions', []).dtype, np.integer)
        # Goal-conditioned SUPE: every row also stores the episode's goal observation (`task_goals`).
        self._store_task_goals = 'task_goals' in buffer._data
        self.last_transition = None  # newest (s, u) row -- SUPE's per-macro-step RND update (aux_schedule='paper')
        self._reset_episode()

    def _reset_episode(self):
        super()._reset_episode()
        self.low_state = None
        self._low_state_stale = True

    @staticmethod
    def example_transition(example_batch, agent=None):
        example_skill = getattr(agent, 'example_skill', None)
        actions = np.int32(0) if example_skill is None else np.asarray(example_skill())  # skill index | latent
        row = dict(
            observations=example_batch['observations'][0],
            actions=actions,
            rewards=np.float32(0.0),
            masks=np.float32(1.0),
            terminals=np.float32(0.0),
            **_extra_row_fields(),
        )
        if getattr(agent, 'stores_task_goals', False):
            # The env goal is a full observation (`info['goal']`, same layout as `observations`).
            row['task_goals'] = np.zeros_like(example_batch['observations'][0])
        return row

    def step(self, agent):
        if self._low_state_stale:
            init_low_level_state = getattr(agent, 'init_low_level_state', None)
            self.low_state = None if init_low_level_state is None else init_low_level_state(max_steps=self.horizon)
            self._low_state_stale = False

        self.rng, skill_key, low_key = jax.random.split(self.rng, 3)
        skill_kwargs = {}
        if getattr(agent, 'sample_skills_takes_env_steps', False):
            skill_kwargs['env_steps'] = self.env_steps
        skill = agent.sample_skills(
            observations=self.observation, goals=self.goal, seed=skill_key, temperature=1.0, **skill_kwargs
        )
        skill = int(skill) if self._integer_skills else np.asarray(skill, dtype=np.float32)

        start_observation = self.observation
        macro_return = 0.0
        disc = 1.0
        terminated_any = False
        done = False
        env_steps = 0
        for _ in range(self.k):
            low_key, action_key = jax.random.split(low_key)
            if self.low_state is None:
                action = agent.low_level_actions(observations=self.observation, skills=skill, seed=action_key)
            else:
                # Stateful frozen policy: the state (a Skill-DT's K-step context and step
                # counter) persists across macro-steps within the episode.
                action, self.low_state = agent.low_level_actions_with_state(
                    observations=self.observation, skills=skill, low_state=self.low_state, seed=action_key
                )
            action = np.asarray(action)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            env_steps += 1
            self.tracker.add(reward, info)
            macro_return += disc * (float(reward) + self.reward_shift)
            disc *= self.gamma_low
            terminated_any = terminated_any or bool(terminated)
            self.observation = next_observation
            done = bool(terminated or truncated)
            if done:
                break
        self.env_steps += env_steps

        row = dict(
            observations=start_observation,
            actions=np.int32(skill) if self._integer_skills else skill,
            rewards=np.float32(macro_return),
            masks=np.float32(1.0 - float(terminated_any)),
            terminals=np.float32(done),
            **self._row_extras(agent),
        )
        if self._store_task_goals:
            row['task_goals'] = self.goal  # fixed for the whole episode (set in _reset_episode)
        abs_idx = self.buffer.add_transition(row)
        self.last_transition = dict(observations=start_observation, actions=row['actions'])
        self._after_add(agent, abs_idx)

        episode = None
        if done:
            self._close_episode(agent, self.observation)
            episode = self.tracker.summary()
            self._reset_episode()
        return dict(env_steps=env_steps, rows=1, episode=episode)


COLLECTOR_CLASSES = dict(flat=FlatCollector, macro=MacroCollector)


def example_transition(rollout_type, example_batch, agent=None):
    """The row layout (one unbatched transition) of the collector for `rollout_type`.

    `agent` lets a macro agent fix the dtype/shape of the stored skill (`example_skill()`);
    without it (or without the hook) skills are int32 indices.
    """
    if rollout_type not in COLLECTOR_CLASSES:
        raise ValueError(f'Unknown rollout_type {rollout_type!r}; expected one of {sorted(COLLECTOR_CLASSES)}.')
    return COLLECTOR_CLASSES[rollout_type].example_transition(example_batch, agent)


def make_collector(agent, env, example_batch, buffer_factory, seed):
    """Build the collector an agent's config asks for, and the buffer it writes into.

    `buffer_factory(example_transition) -> TrajectoryReplayBuffer` lets the caller
    pick the capacity while the collector fixes the row layout.
    """
    rollout_type = agent.config['rollout_type']
    buffer = buffer_factory(example_transition(rollout_type, example_batch, agent))
    if rollout_type == 'flat':
        collector = FlatCollector(env, buffer, seed, discrete=bool(agent.config['discrete']))
    else:
        collector = MacroCollector(
            env, buffer, seed,
            skill_commitment_k=agent.config['skill_commitment_k'],
            gamma_low=agent.config['gamma_low'],
            reward_shift=agent.config.get('reward_shift', 0.0),
        )
    return collector, buffer
