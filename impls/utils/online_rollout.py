"""Experience collectors for online training: one env, one transition at a time.

Two collectors share one interface, `step(agent) -> dict`, and push compact rows
into a `TrajectoryReplayBuffer`:

  * `FlatCollector`   -- one env step per row: (s, a, r, mask, done). Used by the
    flat online CRL agent. The behaviour policy is the stochastic actor
    (`temperature=1`), as in JaxGCRL's `actor_step`.
  * `MacroCollector`  -- one SMDP macro-step per row: the high-level agent picks a
    skill z, which the frozen low-level policy executes for `k` env steps (or
    until the episode ends). The row is (s_t, z, R, mask, done) with
    R = sum_i gamma_low^i r_i over the steps actually taken, mirroring
    `rollout_macro_step` in JaxGCRL's crl_skill_controller. Used by the online
    CRL skill controller.

Both condition the behaviour policy on the episode's task goal (`info['goal']`,
the full goal observation) and never store it: training goals are relabelled
from future observations at sample time, exactly as in JaxGCRL.

Which collector an agent needs is declared by its config (`rollout_type`).
"""

import jax
import numpy as np


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
        self._reset_episode()

    def _reset_episode(self):
        observation, info = self.env.reset()
        self.observation = observation
        self.goal = info['goal']
        self.tracker.reset()

    @staticmethod
    def example_transition(example_batch):
        """Example row (used to allocate the buffer) in this collector's layout."""
        return dict(
            observations=example_batch['observations'][0],
            actions=example_batch['actions'][0],
            rewards=np.float32(0.0),
            masks=np.float32(1.0),
            terminals=np.float32(0.0),
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

        self.buffer.add_transition(
            dict(
                observations=self.observation,
                actions=action,
                rewards=np.float32(reward),
                masks=np.float32(1.0 - float(terminated)),
                terminals=np.float32(done),
            )
        )
        self.observation = next_observation

        episode = None
        if done:
            self.buffer.end_trajectory(next_observation)
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
        self._reset_episode()

    def _reset_episode(self):
        observation, info = self.env.reset()
        self.observation = observation
        self.goal = info['goal']
        self.tracker.reset()

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
            action = np.asarray(agent.low_level_actions(observations=self.observation, skills=skill, seed=action_key))
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
