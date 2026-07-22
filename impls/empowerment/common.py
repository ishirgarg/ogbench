"""Shared foundation for the offline empowerment estimators.

Conventions (see the spec):
  - natural log everywhere; estimates in nats.
  - gamma = 0.99.
  - trajectory boundaries come from `terminals`; futures are NEVER sampled
    across a boundary.
  - discounted future s+: from timestep t, Delta ~ Geometric(1 - gamma)
    (support 1, 2, ...); s+ = obs[t + Delta] from the same trajectory.
    If t + Delta overruns the trajectory (or chunk), resample Delta up to
    ~100 times, then fall back to UNIFORM over the valid range. Never clamp
    to the boundary (clamping piles mass on the terminal state).
"""

import os

import numpy as np

GAMMA = 0.99
OGBENCH_DATA_DIR = os.path.expanduser('~/.ogbench/data')


class TrajectoryData:
    """Flat compact dataset (observations, actions, terminals) + trajectory index structure.

    Attributes:
        observations: [N, obs_dim] float32.
        actions: [N, act_dim] float32.
        terminals: [N] bool/float; 1 at the LAST index of each trajectory.
        final_idx: [N] int; for each flat index t, the flat index of the last
            state of t's trajectory.
        start_idx: [N] int; flat index of the first state of t's trajectory.
    """

    def __init__(self, observations, actions, terminals):
        self.observations = np.asarray(observations, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.terminals = np.asarray(terminals)
        assert self.terminals[-1], 'last index must be a trajectory end'
        self.size = len(self.observations)

        (self.terminal_locs,) = np.nonzero(self.terminals > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        # Per-index trajectory boundaries.
        traj_of = np.searchsorted(self.terminal_locs, np.arange(self.size))
        self.final_idx = self.terminal_locs[traj_of]
        self.start_idx = self.initial_locs[traj_of]

    @property
    def obs_dim(self):
        return self.observations.shape[1]

    @property
    def act_dim(self):
        return self.actions.shape[1]

    def random_nonfinal_idxs(self, num, rng):
        """Random flat indices t with at least one valid future (t < final_idx[t]).

        Use these for query states / training triples: the final state of a
        trajectory has no valid Delta >= 1 future.
        """
        idxs = rng.integers(0, self.size, size=num)
        bad = idxs >= self.final_idx[idxs]
        while bad.any():
            idxs[bad] = rng.integers(0, self.size, size=int(bad.sum()))
            bad = idxs >= self.final_idx[idxs]
        return idxs


def load_trajectory_data(env_name, split='train'):
    """Load an OGBench compact npz dataset as TrajectoryData.

    Args:
        env_name: e.g. 'pointmaze-medium-navigate-v0'.
        split: 'train' or 'val'.
    """
    suffix = '' if split == 'train' else '-val'
    path = os.path.join(OGBENCH_DATA_DIR, f'{env_name}{suffix}.npz')
    with np.load(path) as d:
        return TrajectoryData(d['observations'], d['actions'], d['terminals'])


def sample_geometric_offsets(rng, max_offsets, gamma=GAMMA, num_resample=100):
    """Vectorized truncated-geometric offset sampler with uniform fallback.

    For each element i, draws Delta ~ Geometric(1 - gamma) on support {1, 2, ...}
    conditioned (by rejection, up to `num_resample` rounds) on
    Delta <= max_offsets[i]. Elements still unresolved after `num_resample`
    rounds fall back to Uniform{1, ..., max_offsets[i]}. No clamping.

    Args:
        rng: np.random.Generator.
        max_offsets: [B] int array of per-element maximum valid offsets (>= 1).
    Returns:
        [B] int array of offsets in [1, max_offsets].
    """
    max_offsets = np.asarray(max_offsets)
    assert (max_offsets >= 1).all(), 'every element must have at least one valid future'
    B = len(max_offsets)
    offsets = np.zeros(B, dtype=np.int64)
    unresolved = np.ones(B, dtype=bool)
    for _ in range(num_resample):
        n = int(unresolved.sum())
        if n == 0:
            break
        draws = rng.geometric(p=1 - gamma, size=n)
        ok = draws <= max_offsets[unresolved]
        idxs = np.nonzero(unresolved)[0]
        offsets[idxs[ok]] = draws[ok]
        unresolved[idxs[ok]] = False
    if unresolved.any():
        # Uniform over the valid range {1, ..., max_offset}.
        m = max_offsets[unresolved]
        offsets[unresolved] = rng.integers(1, m + 1)
    return offsets


def sample_discounted_future_idxs(data, idxs, rng, gamma=GAMMA):
    """For flat dataset indices `idxs`, sample s+ indices via the geometric rule.

    Every idx must satisfy idx < data.final_idx[idx] (use random_nonfinal_idxs).
    Returns [B] flat indices of s+ within the same trajectories.
    """
    max_offsets = data.final_idx[idxs] - idxs
    offsets = sample_geometric_offsets(rng, max_offsets, gamma=gamma)
    return idxs + offsets
