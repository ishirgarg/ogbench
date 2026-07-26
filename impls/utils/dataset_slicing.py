"""Trajectory slicing: turn an OGBench dataset into one with shorter trajectories.

Motivation: the `-stitch-` datasets already require stitching together pieces of
different trajectories to reach far-away goals. Cutting each trajectory into
shorter sub-trajectories makes that harder still (the agent sees less of any
single long behaviour at once), which gives a controllable difficulty knob.

The sliced dataset is written to the same directory the regular datasets live in
(`~/.ogbench/data` by default) under a deterministic name, so it is generated
once per machine and simply loaded afterwards.

Naming convention
-----------------
    antmaze-medium-stitch-v0   +  slice length 50
        -> antmaze-medium-stitch-slice50-v0.npz
           antmaze-medium-stitch-slice50-v0-val.npz

Format
------
The output `.npz` has *exactly* the same keys, dtypes and semantics as the input
`.npz` (`observations`, `actions`, `terminals`, plus any of `qpos`, `qvel`,
`button_states` present in the source), so every downstream code path --
`ogbench.load_dataset` (compact and regular), `relabel_dataset`, `add_oracle_reps`,
`GCDataset`/`HGCDataset`/`SequenceDataset` (geometric + uniform goal sampling,
frame stacking, subgoal targets) -- works unchanged. Those samplers derive
trajectory boundaries purely from `terminals`, which we set at the last state of
every sub-trajectory, so goals are never sampled across a slice boundary.

Slicing scheme
--------------
OGBench stores a trajectory of `L` transitions as `L + 1` states:

    observations: [s0, s1, ..., sL]
    actions:      [a0, a1, ..., aL]      # aL is a dummy, dropped by load_dataset
    terminals:    [ 0,  0, ...,  1]

Slicing with `slice_length=K` (requires `L % K == 0`) cuts it into `L / K`
sub-trajectories whose state ranges *overlap by one state* at each cut:

    [s0..sK], [sK..s2K], [s2K..s3K], ...

The shared boundary state is what makes the split lossless: every one of the
original `L` transitions survives in exactly one sub-trajectory. The action
stored at a sub-trajectory's final state is the original action taken there --
it is never used as a transition inside that slice (that is what "truncation at
the end of each sub-trajectory" means here: the last state carries a terminal
flag, and `load_dataset` drops its action), but it reappears as the first action
of the following slice, so no data is lost.
"""

import os

import numpy as np
from ogbench.utils import DEFAULT_DATASET_DIR, download_datasets

# Keys that are per-timestep arrays and must be sliced alongside the observations.
_TIMESTEP_KEYS = ('observations', 'actions', 'terminals', 'qpos', 'qvel', 'button_states')


def sliced_dataset_name(dataset_name, slice_length):
    """Return the canonical name of the sliced version of `dataset_name`.

    The slice token is inserted just before the version token (e.g.
    `antmaze-medium-stitch-v0` -> `antmaze-medium-stitch-slice50-v0`).
    """
    splits = dataset_name.split('-')
    return '-'.join(splits[:-1] + [f'slice{int(slice_length)}'] + splits[-1:])


def parse_slice_token(dataset_name):
    """Split a name like `antmaze-medium-stitch-slice50-v0` into (base name, slice length).

    Returns `(dataset_name, None)` if the name carries no slice token.
    """
    splits = dataset_name.split('-')
    for i, token in enumerate(splits):
        if token.startswith('slice') and token[len('slice') :].isdigit():
            return '-'.join(splits[:i] + splits[i + 1 :]), int(token[len('slice') :])
    return dataset_name, None


def _slice_indices(terminals, slice_length):
    """Compute the gather indices and new terminals for a sliced dataset.

    Args:
        terminals: [N] array; 1/True at the last state of each trajectory.
        slice_length: Number of transitions per sub-trajectory.

    Returns:
        (idxs, new_terminals): `idxs` gathers the source timesteps into the sliced
        dataset; `new_terminals` flags the last state of each sub-trajectory.
    """
    (terminal_locs,) = np.nonzero(np.asarray(terminals) > 0)
    if len(terminal_locs) == 0:
        raise ValueError('Dataset has no terminal flags; cannot determine trajectory boundaries.')
    if terminal_locs[-1] != len(terminals) - 1:
        raise ValueError('Dataset does not end with a terminal flag; the last trajectory is incomplete.')

    starts = np.concatenate([[0], terminal_locs[:-1] + 1])
    num_states = terminal_locs - starts + 1  # states per trajectory
    num_transitions = num_states - 1  # transitions per trajectory (last action is a dummy)

    # Runtime enforcement: every trajectory must divide evenly into `slice_length` transitions.
    bad = np.nonzero(num_transitions % slice_length != 0)[0]
    if len(bad) > 0:
        raise ValueError(
            f'Cannot slice into length-{slice_length} sub-trajectories: '
            f'{len(bad)} of {len(num_transitions)} trajectories have a transition count that is not '
            f'divisible by {slice_length} (e.g. trajectory {int(bad[0])} has {int(num_transitions[bad[0]])} '
            f'transitions). Observed transition counts: {np.unique(num_transitions).tolist()[:10]}.'
        )
    if np.any(num_transitions <= 0):
        raise ValueError('Dataset contains a trajectory with no transitions.')

    num_slices = num_transitions // slice_length  # sub-trajectories per trajectory
    total_slices = int(num_slices.sum())

    # For each sub-trajectory: which source trajectory it came from, and its index within it.
    slice_traj = np.repeat(np.arange(len(num_slices)), num_slices)
    slice_offsets = np.concatenate([[0], np.cumsum(num_slices)[:-1]])
    slice_pos = np.arange(total_slices) - np.repeat(slice_offsets, num_slices)

    # Sub-trajectory k of trajectory j spans source states [start_j + k*K, start_j + (k+1)*K].
    slice_starts = starts[slice_traj] + slice_pos * slice_length
    idxs = (slice_starts[:, None] + np.arange(slice_length + 1)[None, :]).reshape(-1)

    new_terminals = np.zeros(total_slices * (slice_length + 1), dtype=np.asarray(terminals).dtype)
    new_terminals[slice_length :: slice_length + 1] = 1

    return idxs, new_terminals


def slice_dataset_file(src_path, dst_path, slice_length):
    """Read the dataset at `src_path`, slice it, and write it to `dst_path`."""
    src = np.load(src_path)
    idxs, new_terminals = _slice_indices(src['terminals'], slice_length)

    out = {}
    for k in _TIMESTEP_KEYS:
        if k not in src:
            continue
        out[k] = new_terminals if k == 'terminals' else src[k][idxs]

    # Sanity check: the slicing must be lossless in transitions and consistent in length.
    num_valid = int((np.asarray(src['terminals']) == 0).sum())
    new_valid = int((out['terminals'] == 0).sum())
    assert new_valid == num_valid, f'Transition count changed while slicing: {num_valid} -> {new_valid}'
    assert out['terminals'][-1], 'Sliced dataset must end on a terminal.'

    # Write atomically so that concurrent jobs (e.g. a Slurm array) never read a partial file.
    tmp_path = f'{dst_path}.tmp.{os.getpid()}.npz'
    np.savez_compressed(tmp_path, **out)
    os.replace(tmp_path, dst_path)


def make_sliced_datasets(dataset_name, slice_length, dataset_dir=DEFAULT_DATASET_DIR):
    """Ensure the sliced version of `dataset_name` exists on disk and return its path.

    Downloads the source dataset if needed, generates the sliced train/val `.npz`
    files on first use, and simply reuses them on every subsequent call.

    Args:
        dataset_name: Base dataset name (no slice token), e.g. `antmaze-medium-stitch-v0`.
        slice_length: Number of transitions per sub-trajectory.
        dataset_dir: Directory holding the datasets.

    Returns:
        Path to the sliced training dataset (the val file sits next to it with the
        usual `-val.npz` suffix), suitable for passing as `dataset_path`.
    """
    slice_length = int(slice_length)
    if slice_length <= 0:
        raise ValueError(f'slice_length must be positive, got {slice_length}.')

    dataset_dir = os.path.expanduser(dataset_dir)
    sliced_name = sliced_dataset_name(dataset_name, slice_length)

    train_dst = os.path.join(dataset_dir, f'{sliced_name}.npz')
    val_dst = os.path.join(dataset_dir, f'{sliced_name}-val.npz')

    if os.path.exists(train_dst) and os.path.exists(val_dst):
        return train_dst

    # Make sure the source dataset is available locally before slicing it.
    download_datasets([dataset_name], dataset_dir)

    for suffix, dst in (('', train_dst), ('-val', val_dst)):
        if os.path.exists(dst):
            continue
        src = os.path.join(dataset_dir, f'{dataset_name}{suffix}.npz')
        print(f'Generating sliced dataset (slice_length={slice_length}): {src} -> {dst}')
        slice_dataset_file(src, dst, slice_length)

    return train_dst
