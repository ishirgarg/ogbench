#!/usr/bin/env bash
set -euo pipefail

REMOTE="brc"
REMOTE_BASE="/global/scratch/users/ishirgarg/ogbench/OGBench"
LOCAL_BASE="$HOME/Desktop/ckpts"

# method  env  run
RUNS=(
  "dds              antmaze-medium-stitch          sd000_s_35757538.0.20260721_190340"
  "dds              antmaze-medium-stitch          sd000_s_35757537.0.20260721_190341"
  "dds              antsoccer-arena-navigate       sd000_s_35757535.0.20260721_190341"
  "dds              antsoccer-arena-navigate       sd000_s_35757536.0.20260721_190341"
  "dds              antsoccer-arena-stitch         sd000_s_35757532.0.20260721_190340"
  "dds              antsoccer-arena-stitch         sd000_s_35757539.0.20260721_190340"
  "dds              antmaze-medium-navigate        sd000_s_35757533.0.20260721_190340"
  "dds              antmaze-medium-navigate        sd000_s_35757534.0.20260721_190340"
  "empowerment_crl  antsoccer-arena-navigate       sd000_s_35757549.0.20260721_190413"
  "empowerment_crl  antmaze-medium-navigate        sd000_s_35788728.0.20260724_024530"
  "empowerment      antsoccer-arena-stitch-slice50 sd000_s_35873141.0.20260726_160900"
  "empowerment      antsoccer-arena-stitch-slice50 sd000_s_35873146.0.20260726_160900"
  "empowerment      antsoccer-arena-stitch-slice50 sd000_s_35873145.0.20260726_160900"
  "empowerment      antmaze-medium-stitch-slice50  sd000_s_35873143.0.20260726_160900"
  "empowerment      antmaze-medium-stitch-slice50  sd000_s_35873142.0.20260726_160900"
  "empowerment      antmaze-medium-stitch-slice50  sd000_s_35873144.0.20260726_160900"
)

# Remote run-group directory holding a (method, env) pair's runs. Expanded by
# the remote shell, so globs are fine. Run names are unique, so "*" is a safe
# fallback for methods whose run-group naming we don't have pinned down.
group_glob() {
  case "$1" in
    dds)             printf '%s\n' "dds_${2}-v0_K*" ;;
    empowerment_crl) printf '%s\n' "emp_cmp_empowerment_crl_${2}-v0" ;;
    *)               printf '%s\n' "*" ;;
  esac
}

# Log in once: a master connection every rsync below rides on, so you
# authenticate a single time no matter how many runs are in the list.
CM="$HOME/.ssh/cm-download-$$"
cleanup() { ssh -O exit -o ControlPath="$CM" "$REMOTE" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "Opening connection to $REMOTE (authenticate once) ..."
ssh -M -N -f -o ControlPath="$CM" -o ControlPersist=yes "$REMOTE"

# macOS ships openrsync (2.6.9-compatible), which has no --info=progress2
if rsync --help 2>&1 | grep -q -- '--info'; then
  PROGRESS=(--info=progress2)
else
  PROGRESS=(--progress)
fi

i=0
for entry in "${RUNS[@]}"; do
  read -r method env run <<<"$entry"
  i=$((i + 1))

  dest="$LOCAL_BASE/$method/$env"
  mkdir -p "$dest"

  echo "[$i/${#RUNS[@]}] $method/$env/$run"
  rsync -a --partial "${PROGRESS[@]}" \
    -e "ssh -o ControlPath=$CM" \
    "$REMOTE:$REMOTE_BASE/$(group_glob "$method" "$env")/$run" \
    "$dest/"
done

echo "Done."
