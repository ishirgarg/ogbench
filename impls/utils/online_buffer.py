"""Trajectory-aware replay buffer for online goal-conditioned RL.

A fixed-capacity ring buffer over *compact* transitions, in the spirit of
OGBench's compact datasets and `utils/datasets.ReplayBuffer`: every env step
stores a single row (observation, action, reward, mask, terminal), and the
next observation of row `i` is the observation of row `i + 1`. When an episode
ends, its final observation is appended as one extra *marker* row that is
never sampled as an anchor (`valids == 0`), exactly like the `valids` column of
OGBench's compact datasets -- so `observations[i + 1]` is always the true next
observation and the final state of a trajectory can be drawn as a goal.

Goals are relabelled at sample time as in JaxGCRL's `flatten_batch`: each
anchor's goal is a *future* observation of its own trajectory, drawn with
probability proportional to discount^dt over the rows that exist up to the
trajectory end (a truncated, renormalised geometric -- not `GCDataset`'s clipped
`min(idx + Geom, end)`, which piles the tail mass onto the final row).
Trajectories still in progress are sampled too, with their "end" taken as the
most recent row -- the in-progress window JaxGCRL's queue also exposes. The same future observation is returned under both `value_goals` and
`actor_goals`, matching JaxGCRL's use of one future state for the critic
positive and the actor's goal input.

Indexing is by *absolute* step count so the ring wrap-around needs no special
casing: a row is in the buffer iff `total - capacity <= abs_idx < total`, and a
row's future (at most one episode ahead) is always in the buffer whenever the
row itself is, provided `capacity > max episode length + 1`.
"""

import jax
import numpy as np


def _truncated_geometric_offsets(max_offsets, discount):
    """Draw offsets j in [1, d] with P(j) proportional to discount^j (inverse CDF), per row.

    `max_offsets` (d) must be >= 1. This is the row-wise distribution JaxGCRL's
    `flatten_batch` induces (`is_future * discount**dt`, renormalised by the
    categorical draw). With discount == 0 every offset is 1.
    """
    d = np.asarray(max_offsets, dtype=np.int64)
    assert np.all(d >= 1), 'every anchor must have at least one future row'
    if discount <= 0.0:
        return np.ones_like(d)
    u = np.random.rand(d.shape[0])
    # CDF(j) = (1 - discount^j) / (1 - discount^d)  =>  j = ceil(log(1 - u (1 - discount^d)) / log discount)
    offsets = np.floor(np.log1p(-u * (1.0 - discount**d)) / np.log(discount)).astype(np.int64) + 1
    return np.clip(offsets, 1, d)


class TrajectoryReplayBuffer:
    """Ring buffer of compact transitions with trajectory-aware future-goal sampling."""

    @classmethod
    def create(cls, example_transition, capacity):
        """Allocate a buffer from an example *unbatched* transition dict (one row, no leading batch dim)."""

        def alloc(example):
            example = np.asarray(example)
            return np.zeros((capacity, *example.shape), dtype=example.dtype)

        data = jax.tree_util.tree_map(alloc, dict(example_transition))
        return cls(data)

    def __init__(self, data):
        assert 'observations' in data, 'TrajectoryReplayBuffer needs an `observations` field.'
        self._data = data
        self.capacity = len(data['observations'])
        # Absolute index bookkeeping (never wraps; slot = abs_idx % capacity).
        self.total = 0  # rows ever written (real + marker)
        self.num_transitions = 0  # real (valid) rows ever written
        self._valids = np.zeros((self.capacity,), dtype=np.bool_)
        self._num_valid = 0  # running count of valid rows currently held
        self._valid_slots_cache = (-1, None)  # (total at computation, valid slot array)
        self._traj_end = np.full((self.capacity,), -1, dtype=np.int64)  # abs idx of the trajectory's last row
        self._open_rows = []  # abs indices of the in-progress trajectory (real rows + none yet)

    # ── Sizes ────────────────────────────────────────────────────────────────

    @property
    def size(self):
        """Number of rows (real + marker) currently held."""
        return min(self.total, self.capacity)

    @property
    def num_valid(self):
        """Number of sampleable (real) rows currently held."""
        return self._num_valid

    def _oldest_abs(self):
        return max(0, self.total - self.capacity)

    def _slot(self, abs_idx):
        return abs_idx % self.capacity

    # ── Writes ───────────────────────────────────────────────────────────────

    def _write_row(self, transition, valid):
        slot = self._slot(self.total)
        for key, buffer in self._data.items():
            buffer[slot] = transition[key]
        if self.total >= self.capacity and self._valids[slot]:
            self._num_valid -= 1  # evicting a valid row
        self._valids[slot] = valid
        self._num_valid += int(valid)
        self._traj_end[slot] = -1
        self.total += 1
        return self.total - 1

    def add_transition(self, transition):
        """Append one real transition of the in-progress trajectory."""
        abs_idx = self._write_row(transition, valid=True)
        self.num_transitions += 1
        self._open_rows.append(abs_idx)

    def end_trajectory(self, final_observation):
        """Close the in-progress trajectory with a marker row holding its final observation."""
        marker = {key: np.zeros_like(buffer[0]) for key, buffer in self._data.items()}
        marker['observations'] = final_observation
        end_abs = self._write_row(marker, valid=False)
        for abs_idx in self._open_rows + [end_abs]:
            if abs_idx >= self._oldest_abs():
                self._traj_end[self._slot(abs_idx)] = end_abs
        self._open_rows = []

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _valid_anchor_slots(self):
        """Slots that may serve as anchors: valid rows that already have a successor row.

        The newest row of an in-progress trajectory has no next observation yet, so it
        is excluded (JaxGCRL's `flatten_batch` likewise drops the last row of a window);
        every anchor therefore has a strictly-future goal (`goal_offsets >= 1`) and a
        true `next_observations`.
        """
        if self._valid_slots_cache[0] != self.total:
            slots = np.flatnonzero(self._valids[: self.size])
            slots = slots[slots != self._slot(self.total - 1)]
            self._valid_slots_cache = (self.total, slots)
        return self._valid_slots_cache[1]

    def sample(self, batch_size, discount, next_offset=1):
        """Sample a batch with geometric future goals.

        Args:
            batch_size: Number of anchors.
            discount: Future-goal discount in [0, 1); P(offset = j) is proportional to
                discount^j for j in [1, rows-to-trajectory-end].
            next_offset: Rows between an anchor and its `next_observations` (clamped to the
                trajectory end). 1 for ordinary rows; k for offline macro rows stored at
                every env step (see `utils/rlpd.py`).

        Returns:
            Dict with the stored fields at the anchors plus `next_observations`,
            `value_goals`, `actor_goals`, and `goal_offsets` (in rows).
        """
        assert 0.0 <= discount < 1.0, f'discount must be in [0, 1), got {discount}.'
        assert next_offset >= 1, f'next_offset must be >= 1, got {next_offset}.'
        valid_slots = self._valid_anchor_slots()
        assert len(valid_slots) > 0, 'Cannot sample from a buffer with no anchor rows.'
        oldest = self._oldest_abs()
        latest = self.total - 1

        # Rejection-free anchor draw over valid rows: index the valid slots directly.
        slots = valid_slots[np.random.randint(len(valid_slots), size=batch_size)]
        # Recover absolute indices of the chosen slots.
        abs_idxs = self._slot_to_abs(slots, oldest)

        ends = self._traj_end[slots]
        ends = np.where(ends < 0, latest, ends)  # in-progress trajectory: future up to now

        offsets = _truncated_geometric_offsets(ends - abs_idxs, discount)
        goal_abs = abs_idxs + offsets
        next_abs = np.minimum(abs_idxs + next_offset, ends)

        batch = {key: buffer[slots] for key, buffer in self._data.items()}
        batch['next_observations'] = self._data['observations'][self._slot(next_abs)]
        goals = self._data['observations'][self._slot(goal_abs)]
        batch['value_goals'] = goals
        batch['actor_goals'] = goals
        batch['goal_offsets'] = (goal_abs - abs_idxs).astype(np.int32)
        return batch

    def _slot_to_abs(self, slots, oldest):
        """Map ring slots to their absolute indices given the oldest live absolute index."""
        base = oldest - (oldest % self.capacity)
        abs_idxs = base + slots
        abs_idxs = np.where(abs_idxs < oldest, abs_idxs + self.capacity, abs_idxs)
        return abs_idxs
