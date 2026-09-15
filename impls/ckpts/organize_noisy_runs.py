#!/usr/bin/env python3
"""
Download + organize "noisy_policy" empowerment runs.

For each run id:
  1. Look it up on the remote scratch dir (brc) to get env_name/seed/agent hparams.
  2. Check if it already exists somewhere under ckpts/empowerment/*/noisy_policy
     (matched by flags.json's save_dir, since renamed folders lose the raw run id).
  3. If missing, scp-download it (in parallel) into a staging dir.
  4. Move/rename it into ckpts/empowerment/<env>/noisy_policy/sd<seed>_k<num_skills>_<action_noise_std>_<bc_alpha>/
     Only noisy_policy folders are ever touched/renamed -- normal run folders are left alone.

Usage:
  python3 organize_noisy_runs.py runs.txt [--jobs 8] [--dry-run]

runs.txt: one raw run id per line, e.g. sd000_s_36524712.0.20260807_120226
Optionally lines can be "run_id=SKIP" to force-exclude a run (used for collisions).
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REMOTE_HOST = "brc"
REMOTE_BASE = "/global/scratch/users/ishirgarg/ogbench/OGBench/Debug"
CKPTS_ROOT = Path(__file__).resolve().parent
EMPOWERMENT_ROOT = CKPTS_ROOT / "empowerment"
DOWNLOAD_FILES = ["wandb_run_id.txt", "train.csv", "flags.json", "eval.csv"]
PARAMS_GLOB = "params_*.pkl"


def fmt(x):
    return str(x)


def env_folder_name(env_name):
    return env_name[:-3] if env_name.endswith("-v0") else env_name


def target_name(seed, num_skills, noise, bc_alpha):
    return f"sd{int(seed):03d}_k{fmt(num_skills)}_{fmt(noise)}_{fmt(bc_alpha)}"


def remote_flags(run_id, retries=3):
    """Fetch and parse flags.json straight off the remote host, no local copy."""
    remote_path = f"{REMOTE_BASE}/{run_id}/flags.json"
    for attempt in range(retries):
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", REMOTE_HOST, f"cat {remote_path}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
    return None


def index_existing_runs():
    """Map raw run_id -> Path of its existing noisy_policy folder, by reading
    flags.json's save_dir out of every noisy_policy dir already on disk."""
    index = {}
    if not EMPOWERMENT_ROOT.exists():
        return index
    for noisy_dir in EMPOWERMENT_ROOT.glob("*/noisy_policy"):
        # Case A: flags.json directly inside noisy_policy (flat/unorganized)
        flat_flags = noisy_dir / "flags.json"
        if flat_flags.exists():
            try:
                save_dir = json.loads(flat_flags.read_text())["save_dir"]
                index[Path(save_dir).name] = noisy_dir
            except Exception:
                pass
        # Case B: flags.json inside a run subfolder
        for sub in noisy_dir.iterdir():
            if not sub.is_dir():
                continue
            f = sub / "flags.json"
            if f.exists():
                try:
                    save_dir = json.loads(f.read_text())["save_dir"]
                    index[Path(save_dir).name] = sub
                except Exception:
                    pass
    return index


def latest_checkpoint_name(run_id):
    """List params_*.pkl on the remote run dir and return the one with the highest step."""
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", REMOTE_HOST,
         f"ls {REMOTE_BASE}/{run_id}/{PARAMS_GLOB}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    names = [Path(p).name for p in result.stdout.split() if p.strip()]
    if not names:
        return None

    def step(name):
        try:
            return int(name.removeprefix("params_").removesuffix(".pkl"))
        except ValueError:
            return -1

    return max(names, key=step)


def download_run(run_id, dest_dir, dry_run=False):
    dest_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_checkpoint_name(run_id)
    files = list(DOWNLOAD_FILES) + ([checkpoint] if checkpoint else [])
    argv = ["scp", "-q"] + [f"{REMOTE_HOST}:{REMOTE_BASE}/{run_id}/{f}" for f in files] + [f"{dest_dir}/"]
    if dry_run:
        print(f"[dry-run] would run: {argv}")
        return True
    if not checkpoint:
        print(f"  ! warning: no params_*.pkl checkpoint found for {run_id}")
    result = subprocess.run(argv)
    return result.returncode == 0


def place_run(src_dir: Path, dest_dir: Path, dry_run=False):
    if dry_run:
        print(f"[dry-run] would move {src_dir} -> {dest_dir}")
        return
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        raise RuntimeError(f"target already exists: {dest_dir}")
    shutil.move(str(src_dir), str(dest_dir))


def relocate_flat_noisy_policy(noisy_dir: Path, new_name: str, dry_run=False):
    """Handle the legacy case where run files sit directly in noisy_policy/
    instead of a run subfolder -- wrap them into noisy_policy/<new_name>/."""
    tmp_holder = noisy_dir.parent / f"__tmp_{new_name}"
    if dry_run:
        print(f"[dry-run] would restructure flat {noisy_dir} -> {noisy_dir / new_name}")
        return
    tmp_holder.mkdir(parents=True)
    for item in list(noisy_dir.iterdir()):
        if item.name == ".DS_Store":
            continue
        shutil.move(str(item), str(tmp_holder / item.name))
    final = noisy_dir / new_name
    shutil.move(str(tmp_holder), str(final))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_file")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_ids, skip_ids = [], set()
    for line in Path(args.runs_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("=SKIP"):
            skip_ids.add(line.split("=")[0])
            continue
        run_ids.append(line)

    print(f"Loaded {len(run_ids)} run ids ({len(skip_ids)} explicitly skipped).")

    print("Querying remote flags.json for each run (parallel)...")
    manifest = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(remote_flags, rid): rid for rid in run_ids}
        for fut in as_completed(futs):
            rid = futs[fut]
            flags = fut.result()
            if flags is None:
                print(f"  ! could not fetch flags.json for {rid} -- SKIPPING")
                continue
            manifest[rid] = flags

    existing_index = index_existing_runs()

    plan = []  # (run_id, status, env_folder, tname, existing_path_or_None)
    for rid, flags in manifest.items():
        env_folder = env_folder_name(flags["env_name"])
        agent = flags["agent"]
        tname = target_name(flags["seed"], agent["num_skills"], agent["action_noise_std"], agent["bc_alpha"])
        existing = existing_index.get(rid)
        status = "already_local" if existing else "needs_download"
        plan.append((rid, status, env_folder, tname, existing))

    print("\nPlan:")
    for rid, status, env_folder, tname, existing in plan:
        print(f"  {rid:45s} -> {env_folder}/noisy_policy/{tname}   [{status}]")

    to_download = [p for p in plan if p[1] == "needs_download"]
    print(f"\n{len(to_download)} runs to download, {len(plan) - len(to_download)} already local.")

    staging = CKPTS_ROOT / "_staging_noisy"
    if to_download and not args.dry_run:
        staging.mkdir(exist_ok=True)

    download_ok = {}
    if to_download:
        print(f"\nDownloading {len(to_download)} runs in parallel (jobs={args.jobs})...")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {}
            for rid, status, env_folder, tname, existing in to_download:
                dest = staging / rid
                futs[pool.submit(download_run, rid, dest, args.dry_run)] = rid
            for fut in as_completed(futs):
                rid = futs[fut]
                ok = fut.result()
                download_ok[rid] = ok
                print(f"  {'ok' if ok else 'FAILED'}: {rid}")

    print("\nPlacing runs into env/noisy_policy/<target_name>...")
    for rid, status, env_folder, tname, existing in plan:
        dest_dir = EMPOWERMENT_ROOT / env_folder / "noisy_policy" / tname
        if status == "needs_download":
            if not download_ok.get(rid):
                print(f"  ! skip placement, download failed: {rid}")
                continue
            place_run(staging / rid, dest_dir, args.dry_run)
            print(f"  placed (downloaded) {rid} -> {dest_dir}")
        else:
            existing_path = existing
            if existing_path.name == tname:
                print(f"  already correctly named: {dest_dir}")
                continue
            if existing_path == EMPOWERMENT_ROOT / env_folder / "noisy_policy":
                relocate_flat_noisy_policy(existing_path, tname, args.dry_run)
                print(f"  restructured flat run {rid} -> {dest_dir}")
            else:
                place_run(existing_path, dest_dir, args.dry_run)
                print(f"  renamed {existing_path.name} -> {tname}")

    if staging.exists() and not args.dry_run:
        remaining = list(staging.iterdir())
        if not remaining:
            staging.rmdir()
        else:
            print(f"\nNote: staging dir still has leftovers: {remaining}")

    print("\nDone.")


if __name__ == "__main__":
    main()
