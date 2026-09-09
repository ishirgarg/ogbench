"""Audit every checkpoint under impls/ckpts for its expected plots.

For each run dir (anything containing flags.json + params_*.pkl) this works out
which plots *apply* from the agent and env, then reports whether they exist:

  agent empowerment_skill / dads / dv / ...  (has a skill policy + E(s))
      -> empowerment map  AND  skill paths
  agent empowerment_crl / _flowbc            (distilled E(s), no skill policy)
      -> empowerment map only
  agent dds                                  (VQ codebook skills, no E(s))
      -> skill paths only

On antsoccer every path job emits two figures -- ant paths and ball paths --
from the same rollout, so a missing ball plot is reported separately but is
fixed by re-running the same paths command (the launcher dedupes).

Supported env families: antmaze, antsoccer and pointmaze-teleport. PointMaze
has no ball and its figures carry their own names (skill_point_paths_*.png,
empowerment_pointmaze_teleport_*.png). Anything else has no skill-rollout
support here, so it is reported as N/A rather than missing.

Run with --print-cmds to emit the python command line for each missing plot.
"""
import argparse
import glob
import json
import os
import re
import sys

CKPT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpts")

# Agents that expose skill-conditioned policies usable by the path plots.
SKILL_AGENTS = {
    "empowerment_skill", "empowerment_dads", "empowerment_dv",
    "dads", "opal",
}
# Agents with a state-empowerment head but no skill policy.
MAP_ONLY_AGENTS = {"empowerment_crl", "empowerment_crl_flowbc"}
# Agents with skills but no empowerment head; rolled out by their own script.
DDS_AGENTS = {"dds"}

# Fixed antsoccer rollout start: ant at (6, 6), ball at (3, 3).
ANTSOCCER_ANT_XY = os.environ.get("ANTSOCCER_ANT_XY", "6,6")
ANTSOCCER_BALL_XY = os.environ.get("ANTSOCCER_BALL_XY", "3,3")
# Fixed pointmaze-teleport rollout start: a free, central, non-teleporter cell.
POINTMAZE_XY = os.environ.get("POINTMAZE_XY", "12,12")
# pointmaze-teleport is 36x24 in world coords, not the 20x20 the ant mazes use.
POINTMAZE_EXTENT = "--x_min 0 --x_max 36 --y_min 0 --y_max 24"


def latest_epoch(run_dir):
    eps = []
    for p in glob.glob(os.path.join(run_dir, "params_*.pkl")):
        m = re.search(r"params_(\d+)\.pkl$", os.path.basename(p))
        if m:
            eps.append(int(m.group(1)))
    return max(eps) if eps else None


def family(env_name):
    if "antsoccer" in env_name:
        return "antsoccer"
    if "antmaze" in env_name:
        return "antmaze"
    if "pointmaze" in env_name:
        return "pointmaze"
    return None


def audit(root=None):
    rows = []
    scan_root = os.path.join(CKPT_ROOT, root) if root else CKPT_ROOT
    for flags_path in sorted(glob.glob(os.path.join(scan_root, "**", "flags.json"), recursive=True)):
        run_dir = os.path.dirname(flags_path)
        rel = os.path.relpath(run_dir, CKPT_ROOT)
        epoch = latest_epoch(run_dir)
        if epoch is None:
            rows.append(dict(run=rel, run_dir=run_dir, note="no params_*.pkl", items=[]))
            continue
        with open(flags_path) as f:
            flags = json.load(f)
        agent = flags.get("agent", {}).get("agent_name", "?")
        env = flags.get("env_name", "?")
        fam = family(env)

        items = []  # (kind, expected_path_or_None, applicable, reason)
        if fam is None:
            for k in ("map", "paths", "ball"):
                items.append((k, None, False, f"env family unsupported ({env})"))
        else:
            map_png = {
                "antsoccer": "empowerment_map_e%d.png" % epoch,
                "antmaze": "empowerment_antmaze_e%d.png" % epoch,
                "pointmaze": "empowerment_pointmaze_teleport_e%d.png" % epoch,
            }[fam]
            # The ball figure only exists on antsoccer, and only where a paths
            # job runs at all.
            def with_ball(ant_png, ball_png):
                items.append(("paths", os.path.join(run_dir, ant_png), True, ""))
                if fam == "antsoccer":
                    items.append(("ball", os.path.join(run_dir, ball_png), True, ""))
                else:
                    items.append(("ball", None, False, "ball plot is antsoccer-only"))

            if agent in DDS_AGENTS:
                items.append(("map", None, False, "dds has no empowerment head"))
                with_ball("dds_skill_paths_e%d.png" % epoch, "dds_ball_paths_e%d.png" % epoch)
            elif agent in MAP_ONLY_AGENTS:
                items.append(("map", os.path.join(run_dir, map_png), True, ""))
                items.append(("paths", None, False, "%s has no skill policy" % agent))
                items.append(("ball", None, False, "%s has no skill policy" % agent))
            elif agent in SKILL_AGENTS:
                items.append(("map", os.path.join(run_dir, map_png), True, ""))
                # PointMaze paths are drawn by the pointmaze script under its
                # own name; ant families share skill_ant_paths_*.
                if fam == "pointmaze":
                    with_ball("skill_point_paths_e%d.png" % epoch, None)
                else:
                    with_ball("skill_ant_paths_e%d.png" % epoch, "skill_ball_paths_e%d.png" % epoch)
            else:
                for k in ("map", "paths", "ball"):
                    items.append((k, None, False, "unknown agent %s" % agent))

        rows.append(dict(run=rel, run_dir=run_dir, agent=agent, env=env, fam=fam,
                         epoch=epoch, items=items, note=""))
    return rows


def cmd_for(row, kind):
    """python command line that produces the missing plot for this run."""
    rd = os.path.relpath(row["run_dir"], os.path.join(CKPT_ROOT, ".."))
    script = {
        "antsoccer": "plot_empowerment_map_antsoccer.py",
        "antmaze": "plot_empowerment_map_antmaze.py",
        "pointmaze": "plot_empowerment_map_pointmaze_teleport.py",
    }[row["fam"]]
    soccer = row["fam"] == "antsoccer"
    point = row["fam"] == "pointmaze"
    ant_xy = POINTMAZE_XY if point else (ANTSOCCER_ANT_XY if soccer else "8,8")
    if row["agent"] in DDS_AGENTS:
        cmd = "plot_dds_skill_paths.py --run_dir %s/ --steps 3000 --ant_xy %s" % (rd, ant_xy)
        if point:
            return cmd + " " + POINTMAZE_EXTENT
        return cmd + (" --ball_xy %s" % ANTSOCCER_BALL_XY if soccer else "")
    if point:
        # The pointmaze script has no video branch and names its flags
        # --path_steps / --point_xy.
        if kind == "map":
            return ("%s --run_dir %s/ --grid_res 200 --batch_size 48 --no-skill_paths"
                    % (script, rd))
        return ("%s --run_dir %s/ --path_steps 3000 --point_xy %s --no-skill_map"
                % (script, rd, ant_xy))
    if kind == "map":
        return "%s --run_dir %s/ --grid_res 200 --batch_size 48 --no-skill_video --no-skill_paths" % (script, rd)
    # "paths" and "ball" come out of one rollout job.
    cmd = ("%s --run_dir %s/ --video_steps 3000 --video_ant_xy %s"
           " --no-skill_video --no-skill_map --skill_paths" % (script, rd, ant_xy))
    return cmd + (" --video_ball_xy %s" % ANTSOCCER_BALL_XY if soccer else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="Restrict the audit to this subtree of impls/ckpts "
                         "(e.g. --root final).")
    ap.add_argument("--print-cmds", action="store_true",
                    help="Print one python command per missing plot (for a launcher).")
    ap.add_argument("--force-soccer-paths", action="store_true",
                    help="With --print-cmds, also emit the paths job for every antsoccer "
                         "run whose plots already exist (they need regenerating whenever "
                         "the start positions or figure style change).")
    args = ap.parse_args()

    rows = audit(args.root)
    missing = []
    if not args.print_cmds:
        print("%-64s %-18s %-8s %-8s %-8s" % ("run", "agent", "map", "paths", "ball"))
        print("-" * 112)
    for row in rows:
        if row.get("note"):
            if not args.print_cmds:
                print("%-64s %s" % (row["run"], row["note"]))
            continue
        cells = {}
        for kind, path, applicable, reason in row["items"]:
            if not applicable:
                cells[kind] = "n/a"
            elif os.path.exists(path):
                cells[kind] = "OK"
            else:
                cells[kind] = "MISSING"
                missing.append((row, kind))
        if not args.print_cmds:
            print("%-64s %-18s %-8s %-8s %-8s" % (row["run"], row["agent"],
                                                  cells["map"], cells["paths"], cells["ball"]))

    if args.print_cmds:
        seen = set()
        if args.force_soccer_paths:
            for row in rows:
                if row.get("note") or row.get("fam") != "antsoccer":
                    continue
                if any(k == "paths" and applicable for k, _, applicable, _ in row["items"]):
                    missing.append((row, "paths"))
        for row, kind in missing:
            cmd = cmd_for(row, kind)
            if cmd in seen:   # a missing ant+ball pair is one rollout job
                continue
            seen.add(cmd)
            print(cmd)
    else:
        print("-" * 112)
        print("missing: %d plot(s) across %d run(s)"
              % (len(missing), len({r["run"] for r, _ in missing})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
