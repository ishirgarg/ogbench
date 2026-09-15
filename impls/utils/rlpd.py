"""RLPD-style offline data mixing for online runs (JaxGCRL's `use_rlpd`).

JaxGCRL's flat `crl` agent, with `use_rlpd=True`, samples an offline OGBench
batch next to every online batch and trains on the union; offline rows get their
future goals through the same `flatten_batch` as online rows, and nothing else
about the algorithm changes (no offline pretraining, no critic ensembling, same
number of gradient steps). This is the OGBench-side port, extended to the skill
controller (JaxGCRL has no offline path for it). Decisions taken by the user on
2026-09-02:

  * The offline dataset is loaded into a second `TrajectoryReplayBuffer`, so its
    rows are relabelled with the same truncated-geometric future goals as online
    rows (rather than `GCDataset`'s goal mixture).
  * Flat agents store one offline row per env step, exactly like online rows.
  * The skill controller labels every stride-1 window [t, t + k) of each offline
    trajectory with the frozen skill agent's own labeller -- the recipe of
    `skill_bc_relabel_controller` (empowerment: argmax window BC log-likelihood),
    `dds_controller` (DDS: encoder + codebook nearest neighbour) and
    `opal_controller` (discrete OPAL: z ~ p(z | tau), the clustering posterior over
    the window's state trajectory), run through `SequenceDataset`'s relabelling
    drivers -- and stores one macro row
    (s_t, z_t, s_{t + k}) per window. Rows are env steps, so future goals are drawn
    with the per-env-step discount and `next_observations` is `k` rows ahead
    (`TrajectoryReplayBuffer.sample(next_offset=k)`); the online macro buffer, whose
    rows are macro-steps, keeps gamma^k and offset 1.
  * `MixedBatchSampler` fills an exact `offline_ratio` fraction of every batch from
    the offline buffer (RLPD's symmetric sampling; JaxGCRL only gets 50/50 in
    expectation by shuffling a concatenation).

`agent.offline_loglik_threshold` (skill controller, empowerment labeller only; None by
default, which changes nothing) additionally drops offline windows whose winning skill's
BC log-likelihood -- per step and per action dimension, so the number means the same
thing in every env -- falls below it. Dropped rows are still written to the buffer, as
non-anchor rows: they keep their trajectory contiguous so the surviving rows keep exact
k-step next observations and full future-goal horizons.

Rewards, masks and terminals of offline rows are bookkeeping only: both online
agents are purely contrastive and never read them. Offline rows carry
`rewards = 0`, and masks/terminals copied from the dataset (`terminals` marks the
last transition of a trajectory, `masks = 1 - terminals`).

Flat rows also carry `empowerment`: the frozen offline estimator's E(s) when the
flat agent runs with one (`agents/online_crl.py`, `emp_checkpoint_path`), computed
once over the whole dataset here and cached on disk next to the estimator
checkpoint (`<ckpt>/empowerment_values/`), and 0 otherwise.
"""

import dataclasses
import os
import time

import jax
import numpy as np

from utils.datasets import Dataset, SequenceDataset
from utils.env_utils import load_offline_dataset
from utils.online_buffer import TrajectoryReplayBuffer


@dataclasses.dataclass
class BufferSource:
    """A replay buffer together with the sampling arguments its rows need."""

    buffer: TrajectoryReplayBuffer
    discount: float  # future-goal discount per row
    next_offset: int = 1  # rows between an anchor and its next observation

    def sample(self, batch_size):
        return self.buffer.sample(batch_size, self.discount, next_offset=self.next_offset)


class MixedBatchSampler:
    """Exact-ratio mixing of an online and an offline source (RLPD symmetric sampling)."""

    def __init__(self, online, offline, offline_ratio):
        assert 0.0 < offline_ratio < 1.0, f'offline_ratio must be in (0, 1), got {offline_ratio}.'
        self.online = online
        self.offline = offline
        self.offline_ratio = float(offline_ratio)

    def split(self, batch_size):
        """(online rows, offline rows) of a batch; the offline share is rounded to the nearest row."""
        num_offline = int(round(batch_size * self.offline_ratio))
        num_offline = min(max(num_offline, 1), batch_size - 1)
        return batch_size - num_offline, num_offline

    def sample(self, batch_size):
        num_online, num_offline = self.split(batch_size)
        online_batch = self.online.sample(num_online)
        offline_batch = self.offline.sample(num_offline)
        assert online_batch.keys() == offline_batch.keys(), (
            f'online/offline row layouts differ: {sorted(online_batch)} vs {sorted(offline_batch)}'
        )
        return {key: np.concatenate([online_batch[key], offline_batch[key]], axis=0) for key in online_batch}


# ── Offline dataset -> SequenceDataset ───────────────────────────────────────


def _sequence_dataset_config(config, sequence_length):
    """The `GCDataset`/`SequenceDataset` config the offline data needs.

    Only `frame_stack` (observation pipeline) and `sequence_length` (window length
    for the controller's labellers) matter here; the goal-mixture keys are required
    by `GCDataset.__post_init__`'s checks but never used, since goals for offline
    rows come from `TrajectoryReplayBuffer`.
    """
    return dict(
        frame_stack=config['frame_stack'],
        sequence_length=int(sequence_length),
        discount=float(config['discount']),
        value_p_curgoal=0.0,
        value_p_trajgoal=1.0,
        value_p_randomgoal=0.0,
        value_geom_sample=True,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=True,
        gc_negative=True,
        p_aug=None,
        num_skills=config.get('num_skills'),
    )


def load_offline_sequence_dataset(dataset_name, config, sequence_length):
    """Load `dataset_name` (train split, compact) as a `SequenceDataset` with `sequence_length` windows."""
    raw = load_offline_dataset(dataset_name)
    return SequenceDataset(Dataset.create(**raw), _sequence_dataset_config(config, sequence_length))


def offline_trajectories(seq_dataset):
    """Yield (first_row, marker_row) per trajectory: real rows are [first, marker), `marker` holds the final obs.

    Compact OGBench datasets invalidate the last state of every trajectory
    (`valids == 0`), the same convention as the buffer's marker rows.
    """
    valids = np.asarray(seq_dataset.dataset['valids']) > 0
    markers = np.flatnonzero(~valids)
    starts = np.concatenate([[0], markers[:-1] + 1])
    for start, marker in zip(starts, markers):
        if marker > start:  # skip degenerate (empty) trajectories
            yield int(start), int(marker)


def _fill_buffer(buffer, seq_dataset, row_fn, keep=None):
    """Write every offline trajectory into `buffer`; `row_fn(t, start, marker) -> transition dict`.

    `keep` (a `[size]` bool array, or None for all) selects which rows are *anchors*.
    Rows outside it are still written, as non-anchor rows: dropping them outright would
    cut their trajectory in two, shortening the future-goal horizon of every earlier row
    and clamping k-step next observations at the hole. Written-but-not-anchored keeps the
    trajectory intact and merely removes the row from the sampling pool.
    """
    observations = seq_dataset.get_observations(np.arange(seq_dataset.size))
    num_rows = 0
    for start, marker in offline_trajectories(seq_dataset):
        for t in range(start, marker):
            valid = True if keep is None else bool(keep[t])
            buffer.add_transition(row_fn(t, start, marker, observations), valid=valid)
            num_rows += int(valid)
        buffer.end_trajectory(observations[marker])
    return num_rows


def _capacity(seq_dataset):
    """Rows the whole dataset occupies in a `TrajectoryReplayBuffer`: one per row incl. markers."""
    return int(seq_dataset.size)


def make_offline_flat_source(seq_dataset, example_transition, goal_discount, empowerment=None):
    """One offline row per env step: (s_t, a_t, s_{t+1}) with the flat agent's goal discount.

    `empowerment` (a `[size]` float array, or None) fills each row's `empowerment` field
    (None -> 0, the value agents without an estimator never read).
    """
    buffer = TrajectoryReplayBuffer.create(example_transition, _capacity(seq_dataset))
    actions = np.asarray(seq_dataset.dataset['actions'])
    terminals = np.asarray(seq_dataset.dataset['terminals'], dtype=np.float32)
    if empowerment is None:
        empowerment = np.zeros((seq_dataset.size,), dtype=np.float32)
    empowerment = np.asarray(empowerment, dtype=np.float32)
    assert empowerment.shape == (seq_dataset.size,), (
        f'empowerment values {empowerment.shape} do not cover the dataset ({seq_dataset.size} rows)'
    )

    def row(t, start, marker, observations):
        return dict(
            observations=observations[t],
            actions=actions[t],
            rewards=np.float32(0.0),
            masks=np.float32(1.0 - terminals[t]),
            terminals=terminals[t],
            empowerment=empowerment[t],
        )

    num_rows = _fill_buffer(buffer, seq_dataset, row)
    return BufferSource(buffer, discount=float(goal_discount), next_offset=1), num_rows


def make_offline_macro_source(seq_dataset, labels, example_transition, k, goal_discount, keep=None):
    """One offline macro row per env step t: (s_t, z_t, s_{min(t+k, end)}), `z_t` the window label.

    `goal_discount` is the per-env-step discount (rows are env steps); the k-step
    next observation comes from `next_offset=k` at sample time. `keep` restricts which
    rows are sampled as anchors (see `_fill_buffer`).
    """
    labels = np.asarray(labels)
    assert labels.shape == (seq_dataset.size,), f'expected one label per dataset row, got {labels.shape}'
    buffer = TrajectoryReplayBuffer.create(example_transition, _capacity(seq_dataset))
    terminals = np.asarray(seq_dataset.dataset['terminals'], dtype=np.float32)

    def row(t, start, marker, observations):
        last = min(t + k - 1, marker - 1)  # last env step inside the window
        return dict(
            observations=observations[t],
            actions=np.int32(labels[t]),
            rewards=np.float32(0.0),
            masks=np.float32(1.0 - terminals[last]),
            terminals=terminals[last],
        )

    num_rows = _fill_buffer(buffer, seq_dataset, row, keep=keep)
    return BufferSource(buffer, discount=float(goal_discount), next_offset=int(k)), num_rows


def make_offline_source(dataset_name, agent, example_transition, label_seed=0):
    """Build the offline `BufferSource` for `agent` (flat or macro rows, by its `rollout_type`)."""
    config = agent.config
    rollout_type = config['rollout_type']
    name = 'rlpd'

    if rollout_type == 'flat':
        if config.get('offline_loglik_threshold') is not None:
            raise ValueError(
                'agent.offline_loglik_threshold filters offline windows by how well the frozen skill '
                "policy explains them, so it needs a skill agent; rollout_type='flat' has none."
            )
        seq_dataset = load_offline_sequence_dataset(dataset_name, config, sequence_length=1)
        _check_observation_shape(seq_dataset, example_transition)
        empowerment = None
        if getattr(agent, 'uses_empowerment', False):
            empowerment = offline_empowerment_values(agent, seq_dataset, dataset_name, seed=label_seed)
        source, num_rows = make_offline_flat_source(
            seq_dataset, example_transition, config['goal_discount'], empowerment=empowerment
        )
        print(f'[{name}] offline dataset {dataset_name}: {num_rows} env-step rows in {seq_dataset.size} slots')
        return source

    if rollout_type == 'macro':
        k = int(config['skill_commitment_k'])
        seq_dataset = load_offline_sequence_dataset(dataset_name, config, sequence_length=k)
        _check_observation_shape(seq_dataset, example_transition)
        labels, stats = agent.label_offline_windows(seq_dataset, seed=label_seed)
        counts = stats.pop('label_counts', None)
        print(
            f'[{name}] labelled {seq_dataset.size} offline windows (k={k}, K={config["num_skills"]}, '
            f'labeller={config["skill_agent_name"]}): ' + ', '.join(f'{key}={v:.3f}' for key, v in stats.items())
        )
        if counts is not None:
            print(f'[{name}]   per-skill counts: {counts.tolist()}')
        keep = _loglik_keep_mask(seq_dataset, config, name)
        source, num_rows = make_offline_macro_source(
            seq_dataset, labels, example_transition, k, config['discount'], keep=keep
        )
        print(f'[{name}] offline dataset {dataset_name}: {num_rows} macro rows (stride 1) in {seq_dataset.size} slots')
        return source

    raise ValueError(f'Unknown rollout_type {rollout_type!r}.')


def offline_empowerment_values(agent, seq_dataset, dataset_name, seed=0, name='rlpd'):
    """E(s) of every dataset row under the agent's frozen estimator, cached next to the estimator checkpoint.

    The cache key names everything the numbers depend on (dataset, restore epoch, samples per
    state, estimator code path, seed), so a stale file cannot be picked up by a different
    configuration. The file is written atomically so concurrent Slurm jobs on the same
    checkpoint cannot read a partial one. Marker rows (`valids == 0`) get a value too; they
    are never sampled as anchors, so it is unused.
    """
    config = agent.config
    cache_dir = os.path.join(config['emp_checkpoint_path'], 'empowerment_values')
    cache_name = (
        f"{dataset_name}_e{config['emp_restore_epoch']}_n{int(config['emp_num_splus_samples'])}"
        f"_fast{int(bool(config['emp_fast_path']))}_seed{int(seed)}.npy"
    )
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.exists(cache_path):
        values = np.load(cache_path)
        if values.shape == (seq_dataset.size,):
            print(f'[{name}] offline empowerment: loaded cached values from {cache_path}')
            return values.astype(np.float32)
        print(f'[{name}] offline empowerment: cached {cache_path} has shape {values.shape}, recomputing')
    observations = seq_dataset.get_observations(np.arange(seq_dataset.size))
    start = time.time()
    values = agent.empowerment_np(observations, jax.random.PRNGKey(int(seed))).astype(np.float32)
    print(
        f'[{name}] offline empowerment: {len(values)} rows in {time.time() - start:.0f}s '
        f'(mean {values.mean():.4f}, min {values.min():.4f}, max {values.max():.4f})'
    )
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f'{cache_path}.tmp.{os.getpid()}'
    with open(tmp_path, 'wb') as f:  # a file handle: np.save(path) would append '.npy' to the temp name
        np.save(f, values)
    os.replace(tmp_path, cache_path)
    print(f'[{name}] offline empowerment: cached to {cache_path}')
    return values


def _loglik_keep_mask(seq_dataset, config, name):
    """Which offline windows survive `agent.offline_loglik_threshold` (None -> all of them).

    A window's score is the winning skill's BC log-likelihood per step and per action
    dimension (`SequenceDataset.relabel_chunk_skills`); windows scoring below the
    threshold are windows the frozen skill policy cannot reproduce, so the macro
    transition (s_t, z_t, s_{t+k}) the buffer would store for them is a transition no
    skill actually performs. Returning None (the default) keeps the buffer byte-for-byte
    what it was before this filter existed.
    """
    threshold = config.get('offline_loglik_threshold')
    if threshold is None:
        return None
    scores = seq_dataset.chunk_skill_scores
    if scores is None:
        raise ValueError(
            f'agent.offline_loglik_threshold needs per-window BC log-likelihoods, which only the '
            f'empowerment_skill labeller produces; got skill_agent_name={config["skill_agent_name"]!r}.'
        )
    # Marker rows (the invalidated final state of each trajectory) are not windows and are
    # never anchors anyway; excluding them keeps the reported fraction honest.
    scores = np.asarray(scores)
    is_window = np.asarray(seq_dataset.dataset['valids']) > 0
    keep = (scores >= float(threshold)) & is_window
    num_keep, num_windows = int(keep.sum()), int(is_window.sum())
    print(
        f'[{name}] offline_loglik_threshold={float(threshold):.4f}: keeping {num_keep} / {num_windows} windows '
        f'({num_keep / max(num_windows, 1):.1%}; scores are max-skill log pi per step per action dim)'
    )
    if num_keep == 0:
        window_scores = scores[is_window]
        raise ValueError(
            f'agent.offline_loglik_threshold={float(threshold)} drops every offline window (scores run '
            f'{float(window_scores.min()):.4f} to {float(window_scores.max()):.4f}); lower it.'
        )
    return keep


def _check_observation_shape(seq_dataset, example_transition):
    offline_shape = tuple(seq_dataset.get_observations(np.arange(1)).shape[1:])
    online_shape = tuple(np.asarray(example_transition['observations']).shape)
    assert offline_shape == online_shape, (
        f'offline observations {offline_shape} do not match the env observations {online_shape}; '
        f'the offline dataset must come from the same env family (and frame_stack / colored setting).'
    )
