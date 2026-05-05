import collections
import os
import platform
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

import ogbench
from utils.datasets import Dataset


# ── Colored-observation augmentation ─────────────────────────────────────────
#
# A second axis of variation that lives entirely in the obs vector: append two
# extra dims that are zero almost everywhere except in 0.75-radius balls around
# four fixed (x, y) anchors, where they carry a deterministic hash of the
# agent's (x, y) ∈ [-1, 1]^2. The hash is purely a function of (x, y) — no PRNG
# state, no seed — so identical (x, y) inputs always produce identical outputs
# whether they come from the offline dataset, env.step, or info['goal'].

_COLORED_CENTERS = np.array(
    [[0.0, 4.0], [4.0, 0.0], [4.0, 12.0], [12.0, 4.0]], dtype=np.float64
)
_COLORED_RADIUS = 0.75


def _hash_xy(xy):
    """Seed-free hash (x, y) → 2 floats in [-1, 1].

    `xy` has shape [..., 2]; result has shape [..., 2]. Fractional-sin pattern
    from shader-land: looks random but is fully determined by (x, y). The two
    output channels use different frequency pairs so they decorrelate.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    h0 = np.sin(12.9898 * x + 78.233 * y) * 43758.5453
    h0 = h0 - np.floor(h0)        # [0, 1)
    h0 = h0 * 2.0 - 1.0           # [-1, 1)
    h1 = np.sin(23.1406 * x + 91.331 * y) * 13733.7283
    h1 = h1 - np.floor(h1)
    h1 = h1 * 2.0 - 1.0
    return np.stack([h0, h1], axis=-1)


def colored_obs_augment(obs):
    """Append two colored-region dims to `obs`.

    Assumes (x, y) live at indices [0, 1] of the last axis (true for the ant
    in both antmaze and antsoccer — see locomaze/ant.py:get_xy). Output dtype
    matches the input.
    """
    obs = np.asarray(obs)
    xy = obs[..., :2].astype(np.float64, copy=False)
    diffs = xy[..., None, :] - _COLORED_CENTERS                       # [..., 4, 2]
    dists = np.linalg.norm(diffs, axis=-1)                            # [..., 4]
    inside = dists.min(axis=-1) <= _COLORED_RADIUS                    # [...]
    extra = _hash_xy(xy)                                              # [..., 2]
    extra = np.where(inside[..., None], extra, np.zeros_like(extra))  # [..., 2]
    return np.concatenate([obs, extra.astype(obs.dtype, copy=False)], axis=-1)


class ColoredObsWrapper(gymnasium.Wrapper):
    """Append two `colored_obs_augment` channels to every observation.

    Augments the observations returned by `reset` and `step`, plus
    `info['goal']` (which `locomaze` BallEnv/MazeEnv set to a goal-shaped
    observation; see maze.py:427 and maze.py:682).
    """

    def __init__(self, env):
        super().__init__(env)
        old_low = env.observation_space.low
        old_high = env.observation_space.high
        new_low = np.concatenate(
            [old_low, np.array([-1.0, -1.0], dtype=old_low.dtype)], axis=-1
        )
        new_high = np.concatenate(
            [old_high, np.array([1.0, 1.0], dtype=old_high.dtype)], axis=-1
        )
        self.observation_space = Box(
            low=new_low, high=new_high, dtype=env.observation_space.dtype
        )

    def _augment_info_goal(self, info):
        if 'goal' in info and info['goal'] is not None:
            info['goal'] = colored_obs_augment(info['goal'])
        return info

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        return colored_obs_augment(ob), self._augment_info_goal(info)

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        return colored_obs_augment(ob), reward, terminated, truncated, self._augment_info_goal(info)


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name, frame_stack=None):
    """Make OGBench environment and datasets.

    A `-colored-` token in the dataset name (e.g. `antsoccer-arena-stitch-colored-v0`)
    triggers the `ColoredObsWrapper` augmentation: the underlying env+dataset
    is loaded with the `colored` token stripped, then every observation it
    surfaces — env step/reset, info['goal'], and the offline dataset arrays —
    is augmented with two extra channels.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    splits = dataset_name.split('-')
    is_colored = 'colored' in splits
    underlying_name = '-'.join(t for t in splits if t != 'colored') if is_colored else dataset_name

    # Use compact dataset to save memory.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(underlying_name, compact_dataset=True)

    if is_colored:
        env = ColoredObsWrapper(env)
        for ds in (train_dataset, val_dataset):
            ds['observations'] = colored_obs_augment(ds['observations'])
            # compact_dataset=True normally has no 'next_observations', but
            # augment defensively in case that ever changes.
            if 'next_observations' in ds:
                ds['next_observations'] = colored_obs_augment(ds['next_observations'])

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()

    return env, train_dataset, val_dataset
