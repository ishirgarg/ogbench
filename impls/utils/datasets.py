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
      - ``timesteps_seq``:    ``[B, T]`` int — timestep of each window step WITHIN its
        own trajectory (0 at the reset state), for absolute-timestep embeddings.
        Padded steps repeat the terminal's timestep.
      - ``skill_hist_seq``:   ``[B, T, num_skills]`` float — the future-skill histogram
        ``Z_t = normalize(sum_{t'=t}^{T} one_hot(z_{t'}))`` to the TRAJECTORY end
        (Skill-DT, Sec. 4.1). Present ONLY after ``relabel_skill_histograms`` has
        been called at least once; absent otherwise.
      - ``chunk_skills``:     ``[B]`` int32 — the single skill that best explains the
        WHOLE window under a frozen skill-conditioned policy,
        ``argmax_z sum_{i<T} log pi(a_{t+i} | s_{t+i}, z)`` (`skill_bc_relabel_controller`),
        or — after ``relabel_chunk_skills_from_windows`` — the label a window-level
        labeller assigned to it: an int32 ``[B]`` index or a float32 ``[B, D]`` latent
        (`opal_controller`, from OPAL's posterior), or — after ``set_chunk_skills`` —
        labels the agent computed itself (`skill_dt_controller`, from the per-state
        skill counts). Present ONLY after one of those passes has been called; absent
        otherwise.
        Unlike ``skill_hist_seq`` this is a property of the window, not of each step.

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

    Reads one config key:
      - ``sequence_length``: window length ``T``.
    (and ``num_skills``, but only if ``relabel_skill_histograms`` is used.)
    """

    def __post_init__(self):
        super().__post_init__()
        # Timestep of every state within its own trajectory (0 at the reset state).
        all_idxs = np.arange(self.size)
        self.traj_timesteps = (
            all_idxs - self.initial_locs[np.searchsorted(self.initial_locs, all_idxs, side='right') - 1]
        ).astype(np.int32)
        # Forward-cumulative skill counts, filled in by `relabel_skill_histograms`.
        # None until the first re-labelling pass, in which case `skill_hist_seq` is
        # simply absent from the sampled batch.
        self.skill_cumcounts = None
        # Per-window skill labels, filled in by `relabel_chunk_skills`. None until that
        # pass runs, in which case `chunk_skills` is simply absent from the batch.
        self.chunk_skills = None

    def relabel_skill_histograms(self, agent, chunk_bytes=64 * 1024 * 1024, num_skills=None):
        """Hindsight skill re-labelling (Skill-DT, paper Sec. 4.1.1 and Alg. 1).

        Re-encodes EVERY state in the dataset with the agent's CURRENT skill
        encoder and rebuilds the trajectory-end skill histograms ``Z_t``. Called
        from ``main.py`` every ``relabel_interval`` gradient steps, so ``Z_t`` is
        at most ``relabel_interval - 1`` steps stale, exactly as in Alg. 1.

        Rather than materializing ``Z_t`` for every state (a ``[size, num_skills]``
        float array), this stores the forward cumulative counts
        ``C[i] = sum_{j < i} one_hot(z_j)`` with a leading zero row, from which the
        counts over any trajectory suffix ``[t, T]`` are ``C[T + 1] - C[t]``.
        ``sample`` normalizes those for the sampled window only.

        Args:
            agent: agent exposing ``encode_skill_indices(observations) -> [B] int``.
            chunk_bytes: observation bytes pushed through the encoder at a time.
            num_skills: codebook size (None -> ``config['num_skills']``). Controllers
                built on a frozen Skill-DT pass it explicitly, because the dataset holds
                the live agent config whose ``num_skills`` is still the placeholder the
                controller later fills in from its checkpoint.
        """
        num_skills = int(self.config['num_skills'] if num_skills is None else num_skills)

        # Chunk by BYTES, not by count: image datasets would otherwise ship
        # gigabytes to the device per chunk.
        leaves = jax.tree_util.tree_leaves(self.dataset['observations'])
        item_bytes = sum(int(np.prod(leaf.shape[1:])) * leaf.dtype.itemsize for leaf in leaves)
        if self.config['frame_stack'] is not None and not self.preprocess_frame_stack:
            item_bytes *= int(self.config['frame_stack'])
        chunk_size = int(np.clip(chunk_bytes // max(item_bytes, 1), 1024, 100000))

        if self.skill_cumcounts is None or self.skill_cumcounts.shape[1] != num_skills:
            self.skill_cumcounts = np.zeros((self.size + 1, num_skills), dtype=np.int32)
        counts = self.skill_cumcounts[1:]  # view; row i is one-hot(z_i) before the cumsum
        counts.fill(0)
        for start in range(0, self.size, chunk_size):
            idxs = np.arange(start, min(start + chunk_size, self.size))
            skills = np.asarray(jax.device_get(agent.encode_skill_indices(self.get_observations(idxs))))
            counts[idxs, skills] = 1
        np.cumsum(counts, axis=0, out=counts)

    def relabel_chunk_skills(self, agent, chunk_bytes=64 * 1024 * 1024):
        """Label every start index with the skill that best explains its whole window.

        For each index ``t`` this computes, over the length-``T`` window starting at
        ``t`` and clamped to ``t``'s own trajectory,

            z*(t) = argmax_z  sum_{i} log pi(a_{t+i} | s_{t+i}, z)

        under the FROZEN skill-conditioned policy carried by ``agent``, and stores the
        resulting ``[size]`` int32 array on the dataset so ``sample`` can emit it as
        ``chunk_skills``. Because the policy is frozen, one pass is exact — there is no
        staleness and no reason to repeat it (contrast ``relabel_skill_histograms``,
        whose encoder is still training).

        The naive cost is ``size * T * K`` actor forwards. This instead evaluates the
        per-step log-likelihood ``L[t, z] = log pi(a_t | s_t, z)`` once per index
        (``size * K`` forwards) and reads every window sum off a prefix sum of ``L``,
        so ``T`` drops out of the cost entirely. Blocks overlap by ``T - 1`` indices
        because index ``t``'s window needs log-likelihoods past the block's own end;
        each block's prefix sum restarts at zero, which is exact since both endpoints
        of every window difference lie inside the block.

        Windows that run off the end of their trajectory are shortened rather than
        padded (padded steps would contribute the terminal step's likelihood ``T - i``
        times over). A shorter window sums fewer terms, but the argmax is taken per
        ``t``, so the differing scale across ``t`` does not affect any label.

        The number of skills is taken from the agent's output width, not from the
        config: the dataset is constructed with the live agent config, where
        ``num_skills`` is still the placeholder the agent later fills in from its
        pretrained checkpoint.

        Args:
            agent: agent exposing ``chunk_skill_logliks(observations, actions) -> [B, K]``.
            chunk_bytes: observation bytes pushed through the policy at a time.

        Returns:
            dict of label statistics (entropy, coverage, most-frequent-skill share,
            and the raw per-skill counts).
        """
        T = int(self.config['sequence_length'])

        # Chunk by BYTES, matching `relabel_skill_histograms`: image datasets would
        # otherwise ship gigabytes to the device per block.
        leaves = jax.tree_util.tree_leaves(self.dataset['observations'])
        item_bytes = sum(int(np.prod(leaf.shape[1:])) * leaf.dtype.itemsize for leaf in leaves)
        if self.config['frame_stack'] is not None and not self.preprocess_frame_stack:
            item_bytes *= int(self.config['frame_stack'])
        block_size = int(np.clip(chunk_bytes // max(item_bytes, 1), 1024, 100000))

        all_idxs = np.arange(self.size)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, all_idxs)]

        labels = np.empty(self.size, dtype=np.int32)
        # K is read off the agent's own output rather than the config: the dataset holds
        # the live `FLAGS.agent`, whose `num_skills` is still the unresolved placeholder
        # at this point (the agent fills it in from its checkpoint, into its own copy).
        num_skills = None
        for start in range(0, self.size, block_size):
            end = min(start + block_size, self.size)
            # Windows starting in [start, end) read log-likelihoods out to `stop_hi`.
            stop_hi = min(end + T - 1, self.size)
            idxs = np.arange(start, stop_hi)
            logliks = np.asarray(
                jax.device_get(
                    agent.chunk_skill_logliks(
                        self.get_observations(idxs),
                        jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['actions']),
                    )
                ),
                dtype=np.float64,  # the prefix sum is a difference of large partial sums
            )  # [stop_hi - start, num_skills]
            num_skills = logliks.shape[1]

            cumulative = np.zeros((stop_hi - start + 1, num_skills), dtype=np.float64)
            np.cumsum(logliks, axis=0, out=cumulative[1:])

            starts = np.arange(start, end)
            # Exclusive window end, clamped to the trajectory's terminal. This is always
            # <= stop_hi: either stop_hi == end + T - 1 >= (end - 1) + T, or stop_hi is
            # self.size and every terminal index is below it.
            stops = np.minimum(starts + T, final_state_idxs[starts] + 1)
            window_logliks = cumulative[stops - start] - cumulative[starts - start]
            labels[start:end] = np.argmax(window_logliks, axis=1)

        self.chunk_skills = labels

        counts = np.bincount(labels, minlength=num_skills).astype(np.float64)
        probs = counts / max(counts.sum(), 1.0)
        nonzero = probs[probs > 0]
        return {
            'label_entropy': float(-(nonzero * np.log(nonzero)).sum()),
            'label_coverage': float((counts > 0).mean()),
            'label_max_frac': float(probs.max()),
            'label_counts': counts.astype(np.int64),
        }

    def set_chunk_skills(self, labels, num_skills=None):
        """Store precomputed per-start-index skill labels so ``sample`` emits ``chunk_skills``.

        For labellers that are neither a sum of per-step terms (``relabel_chunk_skills``)
        nor a function of the window alone (``relabel_chunk_skills_from_windows``) -- e.g.
        `skill_dt_controller`, whose labels are read off the per-state skill counts that
        ``relabel_skill_histograms`` already stores, so no second pass over the data is
        needed. ``labels`` must be an int32 ``[size]`` index array (or a float32
        ``[size, D]`` latent array).

        Returns:
            the same label statistics as ``relabel_chunk_skills_from_windows``.
        """
        labels = np.asarray(labels)
        if labels.shape[0] != self.size:
            raise ValueError(f'labels must have one entry per dataset index ({self.size}), got {labels.shape}.')
        self.chunk_skills = labels
        return self._chunk_skill_stats(labels, num_skills)

    @staticmethod
    def _chunk_skill_stats(labels, num_skills=None):
        """Label statistics shared by the chunk-labelling passes."""
        if labels.ndim == 1:
            minlength = int(labels.max()) + 1 if num_skills is None else int(num_skills)
            counts = np.bincount(labels, minlength=minlength).astype(np.float64)
            probs = counts / max(counts.sum(), 1.0)
            nonzero = probs[probs > 0]
            return {
                'label_entropy': float(-(nonzero * np.log(nonzero)).sum()),
                'label_coverage': float((counts > 0).mean()),
                'label_max_frac': float(probs.max()),
                'label_counts': counts.astype(np.int64),
            }
        return {
            'label_mean_norm': float(np.linalg.norm(labels, axis=-1).mean()),
            'label_std': float(labels.std(axis=0).mean()),
            'label_abs_max': float(np.abs(labels).max()),
        }

    def relabel_chunk_skills_from_windows(self, agent, seed=0, num_skills=None, chunk_bytes=64 * 1024 * 1024):
        """Label every start index with a skill drawn from a WINDOW-level labeller.

        The counterpart of ``relabel_chunk_skills`` for labellers that are not a sum of
        per-step terms and so need the whole window at once: OPAL's posteriors, i.e.
        the BiGRU encoder ``q(z|tau)`` on the continuous path and the Bayes-rule
        ``p(z|tau)`` over the trajectory mixture on the discrete one. Each start index
        ``t`` gets exactly the length-``T`` window ``sample`` would build for it
        (clamped to ``t``'s trajectory, ``seq_mask`` marking the padded steps), and

            agent.label_chunk_skills(observations_seq, actions_seq, seq_mask, seed)

        returns one label per window: an int32 ``[B]`` skill index or a float32
        ``[B, D]`` latent. Labels are stored on the dataset so ``sample`` emits them as
        ``chunk_skills`` (a ``[B]`` or ``[B, D]`` slice of the stored array). The
        labeller is frozen, so one pass is exact and never repeated.

        Args:
            agent: agent exposing ``label_chunk_skills`` as above.
            seed: PRNG seed for labellers that sample their label.
            num_skills: K for the per-skill counts of index labels (None -> max label + 1).
            chunk_bytes: window bytes pushed through the labeller at a time.

        Returns:
            dict of label statistics: for index labels the same keys as
            ``relabel_chunk_skills``; for latent labels their mean norm and per-dim std.
        """
        T = int(self.config['sequence_length'])

        # Chunk by BYTES as in `relabel_chunk_skills`; a window is T observations.
        leaves = jax.tree_util.tree_leaves(self.dataset['observations'])
        item_bytes = sum(int(np.prod(leaf.shape[1:])) * leaf.dtype.itemsize for leaf in leaves)
        if self.config['frame_stack'] is not None and not self.preprocess_frame_stack:
            item_bytes *= int(self.config['frame_stack'])
        block_size = int(np.clip(chunk_bytes // max(item_bytes * T, 1), 256, 100000))

        all_idxs = np.arange(self.size)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, all_idxs)]
        steps = np.arange(T)

        rng = jax.random.PRNGKey(seed)
        blocks = []
        for start in range(0, self.size, block_size):
            idxs = np.arange(start, min(start + block_size, self.size))
            raw_idxs = idxs[:, None] + steps[None, :]                          # [B, T]
            seq_idxs = np.minimum(raw_idxs, final_state_idxs[idxs][:, None])   # clamp to terminal
            seq_mask = (raw_idxs <= final_state_idxs[idxs][:, None]).astype(np.float32)

            flat = seq_idxs.reshape(-1)
            obs_flat = self.get_observations(flat)
            obs_seq = obs_flat.reshape((len(idxs), T) + obs_flat.shape[1:])
            act_seq = jax.tree_util.tree_map(
                lambda arr: arr[flat].reshape((len(idxs), T) + arr.shape[1:]), self.dataset['actions']
            )
            rng, block_rng = jax.random.split(rng)
            blocks.append(
                np.asarray(jax.device_get(agent.label_chunk_skills(obs_seq, act_seq, seq_mask, block_rng)))
            )
        labels = np.concatenate(blocks, axis=0)
        self.chunk_skills = labels
        return self._chunk_skill_stats(labels, num_skills)

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
        batch['timesteps_seq'] = self.traj_timesteps[seq_idxs]

        # Future-skill histogram to the trajectory end, from the last re-labelling
        # pass. Padded steps are clamped onto the terminal, so their histogram is
        # the terminal's own one-hot; agents mask them out regardless.
        if self.skill_cumcounts is not None:
            suffix_counts = (
                self.skill_cumcounts[final_state_idxs[:, None] + 1] - self.skill_cumcounts[seq_idxs]
            ).astype(np.float32)
            batch['skill_hist_seq'] = suffix_counts / np.maximum(
                suffix_counts.sum(axis=-1, keepdims=True), 1.0
            )

        # Per-window skill label from `relabel_chunk_skills` (a property of the whole
        # window, so one index per batch element rather than one per step).
        if self.chunk_skills is not None:
            batch['chunk_skills'] = self.chunk_skills[idxs]

        # Per-window goal-conditioned reward/mask w.r.t. the SAME value goal (matches GCDataset).
        successes_seq = (seq_idxs == value_goal_idxs[:, None]).astype(np.float32)  # [B, T]
        batch['masks_seq'] = 1.0 - successes_seq
        batch['rewards_seq'] = successes_seq - (1.0 if self.config['gc_negative'] else 0.0)

        # Macro-step bootstrap state s_{t+T} (obs after the H-step window, clamped to terminal).
        subgoal_idxs = np.minimum(idxs + T, final_state_idxs)
        batch['subgoal_observations'] = self.get_observations(subgoal_idxs)

        return batch
