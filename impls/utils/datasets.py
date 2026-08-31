import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        if self.config['frame_stack'] is not None:
            # Only support compact (observation-only) datasets.
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size, idxs=None, evaluation=False, return_goal_idxs=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
            return_goal_idxs: If True, additionally store the raw sampled goal indices under
                ``value_goal_idxs`` / ``actor_goal_idxs``. This is an additive opt-in used by
                ``SequenceDataset`` to compute per-window goal-conditioned rewards; it leaves the
                default behaviour (and every existing caller) untouched.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)
        if return_goal_idxs:
            batch['value_goal_idxs'] = value_goal_idxs
            batch['actor_goal_idxs'] = actor_goal_idxs

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config['actor_geom_sample']:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                    ],
                )

        return batch


#: Hard cap on states encoded per re-labelling call. The `chunk_bytes` budget
#: only sees observation bytes; the encoder widens each state to its hidden and
#: code dimensions, so this is what keeps the transient activations bounded
#: (65536 x 256 floats = 67 MB per buffer).
MAX_RELABEL_CHUNK = 65536


@partial(jax.jit, static_argnames=('num_skills',))
def _suffix_histograms(indices, end_of, num_skills):
    """Jitted core of :func:`trajectory_suffix_histograms`; ``end_of[t]`` is t's terminal index."""
    # int32 counts, not float32: a float32 prefix sum stops being exact past
    # 2**24 rows, which a large dataset would reach.
    onehot = jax.nn.one_hot(indices, num_skills, dtype=jnp.int32)  # [N, num_skills]
    # Inclusive prefix sums let every suffix be read off in two gathers:
    #   sum_{t'=t}^{end} = cum[end] - cum[t] + onehot[t].
    cum = jnp.cumsum(onehot, axis=0)
    suffix = (cum[end_of] - cum + onehot).astype(jnp.float32)
    return suffix / jnp.maximum(suffix.sum(-1, keepdims=True), 1.0)


def trajectory_suffix_histograms(indices, terminal_locs, num_skills, end_of=None):
    """Normalized future-skill histograms to the END of each trajectory.

    ``Z_t = normalize(sum_{t'=t}^{T} one_hot(z_{t'}))`` where ``T`` is the last
    index of ``t``'s trajectory -- the Skill-DT conditioning statistic
    (arXiv:2301.13573 Sec. 4.1, and the ``generate_histogram`` reverse cumulative
    sum in its Sec. A.5).

    Returns a DEVICE array of shape ``[len(indices), num_skills]``. It is gathered
    from once per minibatch, so keeping it on-device avoids copying the whole
    thing back to the host on every re-label; the resulting ``skill_hist_seq`` is
    then the one JAX array in an otherwise-numpy batch.

    Args:
        indices: ``[N]`` integer skill index of every state in the flat dataset.
        terminal_locs: ascending indices of each trajectory's LAST state. Must
            cover ``[0, N)``, i.e. ``terminal_locs[-1] == N - 1``; otherwise the
            trailing states have no trajectory and the (clamped, not raising)
            device gather would return silent garbage.
        num_skills: codebook size.
        end_of: optional precomputed ``[N]`` map from state to its trajectory's
            terminal index. Purely a cache -- it depends only on ``terminal_locs``.
    Returns:
        ``[N, num_skills]`` float32; every row sums to 1.
    """
    n = len(indices)
    if end_of is None:
        terminal_locs = np.asarray(terminal_locs)
        if n and len(terminal_locs) == 0:
            raise ValueError(f'terminal_locs is empty but there are {n} states.')
        if n and terminal_locs[-1] != n - 1:
            raise ValueError(
                f'terminal_locs must cover all {n} states (last terminal at index {n - 1}), '
                f'but ends at {terminal_locs[-1]}.'
            )
        end_of = terminal_locs[np.searchsorted(terminal_locs, np.arange(n))]
    return _suffix_histograms(jnp.asarray(indices), jnp.asarray(end_of), num_skills)


@dataclasses.dataclass
class SequenceDataset(GCDataset):
    """Dataset that additionally returns fixed-length sub-trajectory windows.

    Skill-discovery agents (Skill-DT, QueST, VQ-BeT, DDS, ...) need contiguous
    windows of observations/actions ``[t, t+T-1]`` from the SAME trajectory rather
    than the single transitions that ``GCDataset`` provides. This class augments
    the standard ``GCDataset`` batch (so all goal/reward keys remain available)
    with:

      - ``observations_seq``: ``[B, T, *obs_shape]`` — obs at ``idx+0 .. idx+T-1``,
        clamped to the trajectory's terminal index (reads never cross an episode
        boundary; out-of-trajectory steps repeat the terminal observation).
      - ``actions_seq``:      ``[B, T, *act_shape]`` — actions over the same window.
      - ``seq_mask``:         ``[B, T]`` float — 1.0 for in-trajectory steps, 0.0 for
        steps padded past the terminal. Agents MUST mask padded steps in losses.
      - ``timesteps_seq``:    ``[B, T]`` int32 — the ABSOLUTE step index within the
        trajectory (0 at its first state), for agents with timestep embeddings.

    For semi-MDP / option-style agents (e.g. DDS) it also emits the per-window
    goal-conditioned reward signal and the macro-step bootstrap state — all
    computed exactly as ``GCDataset`` computes them for a single transition, so
    that an ``H``-step discounted snippet return can be formed downstream:

      - ``rewards_seq``: ``[B, T]`` float — the goal-conditioned reward at each
        window step w.r.t. the SAME ``value_goals`` used for the base transition
        (``successes - 1`` if ``gc_negative`` else ``successes``).
      - ``masks_seq``:   ``[B, T]`` float — ``1 - successes`` at each window step
        (0 exactly at the step whose state index equals the value goal).
      - ``subgoal_observations``: ``[B, *obs_shape]`` — the macro-step next state
        ``s_{t+T}`` (obs at ``idx+T`` clamped to the terminal), i.e. the state the
        H-step option transitions into; used for the single-discount bootstrap.

    These extra keys are purely additive; agents that only read the window keys
    (Skill-DT, QueST, VQ-BeT) are unaffected.

    The window starts at the sampled index ``idx`` (the same index used for the
    base transition / goals), so ``observations_seq[:, 0]`` equals the base
    ``observations`` (modulo frame-stacking) and goals refer to that start state.

    Finally, agents that condition on a statistic of their OWN (continually
    changing) encoder — Skill-DT's future-skill histogram — can install
    per-state labels via :meth:`relabel_skill_histograms`; once installed, every
    batch also carries

      - ``skill_hist_seq``: ``[B, T, num_skills]`` float32 — the normalized
        histogram of skills from each window step to the END of its trajectory.

    Reads one config key:
      - ``sequence_length``: window length ``T``.
    """

    #: Per-state future-skill histograms, ``[dataset_size, num_skills]``, or None
    #: until :meth:`relabel_skill_histograms` installs them. Unannotated on
    #: purpose: it is a class attribute, not a dataclass field, so it never
    #: enters ``__init__`` and ``self.skill_hist = ...`` shadows it per instance.
    skill_hist = None
    #: Cache of the state -> trajectory-terminal-index map (depends only on
    #: ``terminal_locs``, so it survives every re-label).
    _traj_end_of = None

    def relabel_skill_histograms(self, encode_fn, num_skills, chunk_bytes=256 << 20):
        """Hindsight skill re-labelling (Skill-DT, arXiv:2301.13573 Sec. 4.1.1).

        Re-encodes every state in the dataset with the agent's *current* skill
        encoder and rebuilds the trajectory-end histograms ``Z_t``. The paper
        does this before every training iteration because "the skill encoder is
        being updated consistently and skill representations change during
        training, [so] the re-labelling of skill distributions is required to
        ensure stability in action predictions".

        Args:
            encode_fn: maps a batch of observations to ``[B]`` skill indices
                (e.g. ``SkillDTAgent.encode_skill_indices``).
            num_skills: codebook size.
            chunk_bytes: how much OBSERVATION data to encode per call, so a pixel
                dataset does not silently ask for a chunk hundreds of times
                larger than a state-based one. Note this budget covers the inputs
                only -- the encoder expands each state to its hidden and code
                widths, so device peak is a few times larger; `MAX_RELABEL_CHUNK`
                is what actually bounds those activations.
        """
        # Drop the previous histogram up front: it is about to be replaced, and
        # holding it through the encode pass is a needless resident copy
        # (~256 MB at 1M states x 64 skills).
        self.skill_hist = None

        obs = self.dataset['observations']
        per_state = max(1, int(np.prod(obs.shape[1:])) * obs.dtype.itemsize)
        if self.config['frame_stack'] is not None and not self.preprocess_frame_stack:
            per_state *= int(self.config['frame_stack'])
        chunk_size = int(np.clip(chunk_bytes // per_state, 1, min(self.size, MAX_RELABEL_CHUNK)))

        indices = np.concatenate(
            [
                np.asarray(encode_fn(self.get_observations(np.arange(start, min(start + chunk_size, self.size)))))
                for start in range(0, self.size, chunk_size)
            ]
        )
        if self._traj_end_of is None:
            self._traj_end_of = self.terminal_locs[np.searchsorted(self.terminal_locs, np.arange(self.size))]
        self.skill_hist = trajectory_suffix_histograms(
            indices, self.terminal_locs, num_skills, end_of=self._traj_end_of
        )

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        # Opt into the raw goal indices so we can replicate GCDataset's reward over the window.
        batch = super().sample(batch_size, idxs=idxs, evaluation=evaluation, return_goal_idxs=True)
        value_goal_idxs = batch.pop('value_goal_idxs')
        batch.pop('actor_goal_idxs', None)

        T = int(self.config['sequence_length'])
        # Terminal index of each sampled transition's trajectory.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        steps = np.arange(T)
        # Absolute indices for the window, clamped within the trajectory.
        raw_idxs = idxs[:, None] + steps[None, :]                 # [B, T]
        seq_idxs = np.minimum(raw_idxs, final_state_idxs[:, None])  # clamp to terminal
        seq_mask = (raw_idxs <= final_state_idxs[:, None]).astype(np.float32)  # [B, T]

        flat = seq_idxs.reshape(-1)
        obs_flat = self.get_observations(flat)
        act_flat = jax.tree_util.tree_map(lambda arr: arr[flat], self.dataset['actions'])
        batch['observations_seq'] = obs_flat.reshape((batch_size, T) + obs_flat.shape[1:])
        batch['actions_seq'] = jax.tree_util.tree_map(
            lambda arr: arr.reshape((batch_size, T) + arr.shape[1:]), act_flat
        )
        batch['seq_mask'] = seq_mask

        # Absolute within-trajectory step index of each window position.
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        batch['timesteps_seq'] = (seq_idxs - initial_state_idxs[:, None]).astype(np.int32)

        # Re-labelled future-skill histograms, when an agent has installed them.
        if self.skill_hist is not None:
            batch['skill_hist_seq'] = self.skill_hist[seq_idxs]

        # Per-window goal-conditioned reward/mask w.r.t. the SAME value goal (matches GCDataset).
        successes_seq = (seq_idxs == value_goal_idxs[:, None]).astype(np.float32)  # [B, T]
        batch['masks_seq'] = 1.0 - successes_seq
        batch['rewards_seq'] = successes_seq - (1.0 if self.config['gc_negative'] else 0.0)

        # Macro-step bootstrap state s_{t+T} (obs after the H-step window, clamped to terminal).
        subgoal_idxs = np.minimum(idxs + T, final_state_idxs)
        batch['subgoal_observations'] = self.get_observations(subgoal_idxs)

        return batch
