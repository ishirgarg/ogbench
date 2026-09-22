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
    For an agent with the RND bonus (`uses_rnd`) the collector also remembers every row
    it wrote (marker rows included) and `flush_rnd(agent)` feeds those states, in visit
    order, to the agent's running RND statistics (`agent.rnd_update_stats`), returning
    the updated agent; `main_online.py` runs it before every update round too.
  * `MacroCollector`  -- one SMDP macro-step per row: the high-level agent picks a
    skill z, which the frozen low-level policy executes for `k` env steps (or
    until the episode ends). The row is (s_t, z, R, mask, done) with
    R = sum_i gamma_low^i r_i over the steps actually taken, mirroring
    `rollout_macro_step` in JaxGCRL's crl_skill_controller. Used by the online
    CRL skill controller. A frozen policy that keeps per-episode state (Skill-DT's
    Transformer context) exposes `init_low_level_state(max_steps)` /
    `low_level_actions_with_state(...)`; the collector threads that state through
    every env step of the episode and rebuilds it at reset.

Both condition the behaviour policy on the episode's task goal (`info['goal']`,
the full goal observation) and never store it: training goals are relabelled
from future observations at sample time, exactly as in JaxGCRL.

Which collector an agent needs is declared by its config (`rollout_type`).
"""

import jax
import jax.numpy as jnp
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


class FlatCollector:
    """Per-env-step collector for flat goal-conditioned agents."""

    def __init__(self, env, buffer, seed, discrete=False):
        self.env = env
        self.buffer = buffer
        self.rng = jax.random.PRNGKey(seed)
        self.discrete = discrete
        self.tracker = _EpisodeTracker()
        self._pending_empowerment = []  # abs indices of rows whose `empowerment` is still the NaN placeholder
        self._pending_first = []  # parallel to the above: does the row start a new episode?
        self._pending_rnd = []  # abs indices (rows + marker rows) not yet fed to the agent's RND running stats
        self._pending_rnd_first = []  # parallel: is the row an episode's reset state (not any transition's s')?
        self._episode_first_row = True  # the next row written starts a new episode
        self._reset_episode()

    def _reset_episode(self):
        observation, info = self.env.reset()
        self.observation = observation
        self.goal = info['goal']
        self.tracker.reset()
        self._episode_first_row = True

    @staticmethod
    def example_transition(example_batch):
        """Example row (used to allocate the buffer) in this collector's layout."""
        return dict(
            observations=example_batch['observations'][0],
            actions=example_batch['actions'][0],
            rewards=np.float32(0.0),
            masks=np.float32(1.0),
            terminals=np.float32(0.0),
            empowerment=np.float32(0.0),
            episodic_max_empowerment=np.float32(0.0),
            is_offline=np.float32(0.0),
        )

    def _mark_pending(self, abs_idx):
        """Queue a row for `flush_empowerment` (NaN placeholders until then)."""
        self._pending_empowerment.append(abs_idx)
        self._pending_first.append(self._episode_first_row)
        self._episode_first_row = False

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
        return int(len(abs_idxs))

    RND_CHUNK = 128  # states per `rnd_update_stats` call (padded to one jit shape)

    def flush_rnd(self, agent):
        """Advance the agent's RND running stats with every state visited since the last flush.

        The pending rows (transitions plus the marker rows that close episodes) are
        contiguous and in write order, so their `observations` are the visited states in
        temporal order. An episode's first row is its reset state, not any transition's
        landing state s', so it feeds the observation stats but not the intrinsic-reward
        stream. Returns `(agent, info)`; a no-op `(agent, {})` when nothing is pending.
        """
        if not self._pending_rnd:
            return agent, {}
        abs_idxs = np.asarray(self._pending_rnd, dtype=np.int64)
        is_next = 1.0 - np.asarray(self._pending_rnd_first, dtype=np.float32)
        self._pending_rnd = []
        self._pending_rnd_first = []
        observations = self.buffer.read_field('observations', abs_idxs)

        chunk = self.RND_CHUNK
        info = {}
        for start in range(0, len(abs_idxs), chunk):
            obs = observations[start : start + chunk]
            nxt = is_next[start : start + chunk]
            n = len(obs)
            valid = np.ones((n,), dtype=np.float32)
            if n < chunk:
                pad = chunk - n
                obs = np.concatenate([obs, np.zeros((pad,) + obs.shape[1:], dtype=obs.dtype)], axis=0)
                nxt = np.concatenate([nxt, np.zeros((pad,), dtype=np.float32)], axis=0)
                valid = np.concatenate([valid, np.zeros((pad,), dtype=np.float32)], axis=0)
            agent, info = agent.rnd_update_stats(jnp.asarray(obs), jnp.asarray(nxt), jnp.asarray(valid))
        return agent, {k: float(np.asarray(v)) for k, v in info.items()}

    def step(self, agent):
        self.rng, key = jax.random.split(self.rng)
        action = agent.sample_actions(observations=self.observation, goals=self.goal, seed=key, temperature=1.0)
        action = np.asarray(action)
        if not self.discrete:
            action = np.clip(action, -1, 1)

        next_observation, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        self.tracker.add(reward, info)

        uses_empowerment = bool(getattr(agent, 'uses_empowerment', False))
        uses_rnd = bool(getattr(agent, 'uses_rnd', False))
        first_row = self._episode_first_row
        abs_idx = self.buffer.add_transition(
            dict(
                observations=self.observation,
                actions=action,
                rewards=np.float32(reward),
                masks=np.float32(1.0 - float(terminated)),
                terminals=np.float32(done),
                # NaN until `flush_empowerment`: sampling an unfilled row would surface as a NaN loss
                # rather than silently training on a wrong entropy target.
                empowerment=np.float32(np.nan if uses_empowerment else 0.0),
                episodic_max_empowerment=np.float32(np.nan if uses_empowerment else 0.0),
                is_offline=np.float32(0.0),
            )
        )
        if uses_empowerment:
            self._mark_pending(abs_idx)
        if uses_rnd:
            self._pending_rnd.append(abs_idx)
            self._pending_rnd_first.append(first_row)
        self._episode_first_row = False
        self.observation = next_observation

        episode = None
        if done:
            end_abs = self.buffer.end_trajectory(next_observation)
            if uses_empowerment:
                # The marker row is never an anchor but it IS the last transition's next state,
                # whose E(s') the exploration bonus reads: give it the same NaN-until-flushed slots.
                nan = np.array([np.nan], dtype=np.float32)
                self.buffer.write_field('empowerment', [end_abs], nan)
                self.buffer.write_field('episodic_max_empowerment', [end_abs], nan)
                self._mark_pending(end_abs)
            if uses_rnd:
                # The final observation is the last transition's landing state s'.
                self._pending_rnd.append(end_abs)
                self._pending_rnd_first.append(False)
            episode = self.tracker.summary()
            self._reset_episode()
        return dict(env_steps=1, rows=1, episode=episode)


class MacroCollector:
    """Per-macro-step collector for a high-level skill controller over a frozen skill policy."""

    def __init__(self, env, buffer, seed, skill_commitment_k, gamma_low=1.0):
        self.env = env
        self.buffer = buffer
        self.rng = jax.random.PRNGKey(seed)
        self.k = int(skill_commitment_k)
        self.gamma_low = float(gamma_low)
        self.tracker = _EpisodeTracker()
        # Episode horizon handed to a STATEFUL frozen low-level policy (Skill-DT sizes its
        # rollout histogram with it); None if the env has no TimeLimit.
        self.horizon = env_horizon(env)
        # Per-episode state of the frozen low-level policy, or None for the stateless
        # families (empowerment_skill, dds). Built lazily on the first `step` of every
        # episode because the agent is not known at construction time.
        self.low_state = None
        self._low_state_stale = True
        self._reset_episode()

    def flush_empowerment(self, agent):
        """Macro rows carry no empowerment field (the flat agent's entropy bonus only); nothing to fill."""
        return 0

    def flush_rnd(self, agent):
        """The RND bonus is the flat agent's only; nothing to feed."""
        return agent, {}

    def _reset_episode(self):
        observation, info = self.env.reset()
        self.observation = observation
        self.goal = info['goal']
        self.tracker.reset()
        self.low_state = None
        self._low_state_stale = True

    @staticmethod
    def example_transition(example_batch):
        return dict(
            observations=example_batch['observations'][0],
            actions=np.int32(0),  # skill index
            rewards=np.float32(0.0),
            masks=np.float32(1.0),
            terminals=np.float32(0.0),
        )

    def step(self, agent):
        if self._low_state_stale:
            init_low_level_state = getattr(agent, 'init_low_level_state', None)
            self.low_state = None if init_low_level_state is None else init_low_level_state(max_steps=self.horizon)
            self._low_state_stale = False

        self.rng, skill_key, low_key = jax.random.split(self.rng, 3)
        skill = int(agent.sample_skills(observations=self.observation, goals=self.goal, seed=skill_key, temperature=1.0))

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
            macro_return += disc * float(reward)
            disc *= self.gamma_low
            terminated_any = terminated_any or bool(terminated)
            self.observation = next_observation
            done = bool(terminated or truncated)
            if done:
                break

        self.buffer.add_transition(
            dict(
                observations=start_observation,
                actions=np.int32(skill),
                rewards=np.float32(macro_return),
                masks=np.float32(1.0 - float(terminated_any)),
                terminals=np.float32(done),
            )
        )

        episode = None
        if done:
            self.buffer.end_trajectory(self.observation)
            episode = self.tracker.summary()
            self._reset_episode()
        return dict(env_steps=env_steps, rows=1, episode=episode)


COLLECTOR_CLASSES = dict(flat=FlatCollector, macro=MacroCollector)


def example_transition(rollout_type, example_batch):
    """The row layout (one unbatched transition) of the collector for `rollout_type`."""
    if rollout_type not in COLLECTOR_CLASSES:
        raise ValueError(f'Unknown rollout_type {rollout_type!r}; expected one of {sorted(COLLECTOR_CLASSES)}.')
    return COLLECTOR_CLASSES[rollout_type].example_transition(example_batch)


def make_collector(agent, env, example_batch, buffer_factory, seed):
    """Build the collector an agent's config asks for, and the buffer it writes into.

    `buffer_factory(example_transition) -> TrajectoryReplayBuffer` lets the caller
    pick the capacity while the collector fixes the row layout.
    """
    rollout_type = agent.config['rollout_type']
    buffer = buffer_factory(example_transition(rollout_type, example_batch))
    if rollout_type == 'flat':
        collector = FlatCollector(env, buffer, seed, discrete=bool(agent.config['discrete']))
    else:
        collector = MacroCollector(
            env, buffer, seed,
            skill_commitment_k=agent.config['skill_commitment_k'],
            gamma_low=agent.config['gamma_low'],
        )
    return collector, buffer
