"""Environment helpers for online training on OGBench envs.

Online runs never touch the offline dataset: the env comes from
`make_env_only`, and the example batch that shapes the networks comes from the
env's own spaces (`make_example_batch`). The only extra knob is the episode
horizon: JaxGCRL takes `episode_length` as a run flag, while OGBench bakes
`max_episode_steps` into the gymnasium registration, so `--episode_length`
rewrites the TimeLimit in place when given.
"""

import numpy as np

from utils.env_utils import make_env_only
from utils.evaluation import env_horizon  # noqa: F401  (re-exported: the registered/overridden horizon)


def set_time_limit(env, max_steps):
    """Set the env's TimeLimit horizon to exactly `max_steps` (raise or lower).

    Unlike `utils.evaluation.raise_time_limit`, which only ever lengthens an
    episode, this pins every `_max_episode_steps` in the wrapper chain (and the
    spec) to the requested value so `--episode_length` means what it says.
    """
    target = int(max_steps)
    e = env
    while e is not None:
        if getattr(e, '_max_episode_steps', None) is not None:
            e._max_episode_steps = target
        e = getattr(e, 'env', None)
    spec = getattr(env, 'spec', None)
    if spec is not None:
        try:
            spec.max_episode_steps = target
        except AttributeError:
            pass


def make_online_env(env_name, frame_stack=None, episode_length=None):
    """Build the training/eval env for an online run.

    Args:
        env_name: OGBench env (dataset) name; task pairs come from the env's own
            registration, so custom online task sets are separate registered envs.
        frame_stack: Number of frames to stack (must match a frozen skill checkpoint, if any).
        episode_length: If given, override the registered `max_episode_steps`.
    """
    env = make_env_only(env_name, frame_stack=frame_stack)
    if episode_length is not None:
        set_time_limit(env, episode_length)
    return env


def env_agent_xy(env):
    """The agent's planar position for trajectory plots, or None if the env has no `get_xy`."""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, 'get_xy'):
        return np.asarray(unwrapped.get_xy(), dtype=np.float32)
    return None


def env_goal_xy(env):
    """The current task goal's planar position, or None if the env does not expose it.

    On antsoccer (`BallEnv`) this is the *ball's* target cell, while `env_agent_xy`
    tracks the ant, so the plotted goal marker is where the ball must go.
    """
    unwrapped = env.unwrapped
    goal_xy = getattr(unwrapped, 'cur_goal_xy', None)
    return None if goal_xy is None else np.asarray(goal_xy, dtype=np.float32)
