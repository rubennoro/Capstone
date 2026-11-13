#!/bin/bash
# Cluster -> laptop: copy a packaged run <CLUSTER_RUNS>/<label>/RUN/ into data/FOLDER/policy/RUN/ through the transfer
# node: everything (summary, csv, plots, videos, logs, wandb/, eval/) plus the best N slim checkpoints.
# Never deletes local files; checks the fetched checkpoint sizes against checkpoints.json.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
usage: train/fetch.sh FOLDER RUN|--all [--top N] [--no-checkpoints] [--force] [--dry-run]

  FOLDER            local task folder data/FOLDER/
  RUN               run name, e.g. run_003_2026-11-13
  --all             every run in data/FOLDER/policy/ (those submitted from this laptop)
  --top N           only the N best checkpoints (top1 .. topN, ~2.5 GB each; default TOPK)
  --no-checkpoints  no checkpoints at all (only checkpoints.json)
  --force           fetch even if the run is still queued/running, or if < MIN_FREE_GB would be left
  --dry-run         print the ssh/rsync commands, run and write nothing

example: train/fetch.sh block_raise run_003_2026-11-13 --top 1
EOF
}

ALL=0 TOP="" NO_CKPT=0 ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --all) ALL=1; shift ;;
    --top) TOP=${2:?--top needs a number}; shift 2 ;;
    --no-checkpoints) NO_CKPT=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [ "$ALL" = 1 ]; then [ ${#ARGS[@]} = 1 ] || { usage >&2; exit 1; }
else [ ${#ARGS[@]} = 2 ] || { usage >&2; exit 1; }; fi
FOLDER=${ARGS[0]}
check_name folder "$FOLDER"

load_config
need_var CLUSTER_XFER CLUSTER_LOGIN CLUSTER_TOOLS CLUSTER_RUNS LOCAL_DATA_ROOT MIN_FREE_GB TOPK
TOP=${TOP:-$TOPK}
check_posint --top "$TOP"

if [ "$ALL" = 1 ]; then
  RUNS=()
  for j in "$LOCAL_DATA_ROOT/$FOLDER"/policy/*/job.json; do
    [ -f "$j" ] && RUNS+=("$(basename "$(dirname "$j")")")
  done
  [ ${#RUNS[@]} -gt 0 ] || die "no data/$FOLDER/policy/*/job.json"
else
  RUNS=("${ARGS[1]}")
fi

filters=(--include=/checkpoints/ --include=/checkpoints/checkpoints.json)
if [ "$NO_CKPT" = 0 ]; then
  for ((k = 1; k <= TOP; k++)); do filters+=("--include=/checkpoints/top${k}_*"); done
fi
filters+=(--exclude=/checkpoints/*)

# verify_ckpts DIR: every top1..TOP entry of checkpoints.json is there with the listed size
verify_ckpts() {
  python3 -I - "$1" "$TOP" <<'PY'
import json, os, sys
d, top = sys.argv[1], int(sys.argv[2])
p = os.path.join(d, "checkpoints", "checkpoints.json")
if not os.path.exists(p):
    print(f"warning: no checkpoints/checkpoints.json in {d}; checkpoints not verified", file=sys.stderr)
    sys.exit(0)
bad = []
for e in json.load(open(p)):
    if int(e["rank"]) > top:
        continue
    name = os.path.basename(e["file"])
    f = os.path.join(d, "checkpoints", name)
    if not os.path.exists(f):
        f = os.path.join(d, "checkpoints", f"top{e['rank']}_{name}")
    size = os.path.getsize(f) if os.path.exists(f) else None
    if size != int(e["size_bytes"]):
        bad.append(f"{name}: {size} bytes, expected {e['size_bytes']}")
    else:
        print(f"  ok  {os.path.basename(f)} ({size / 1e9:.2f} GB)", file=sys.stderr)
if bad:
    sys.exit("checkpoint size mismatch (run fetch.sh again; it resumes):\n  " + "\n  ".join(bad))
PY
}

fetch_one() {
  local run=$1 label rdir dst st out bytes
  check_name run "$run"
  label=$(run_label "$FOLDER" "$run")
  check_label "$label"
  rdir=$(rpath "$CLUSTER_RUNS")/$label/$run
  dst=$LOCAL_DATA_ROOT/$FOLDER/policy/$run
  step "Fetching $CLUSTER_XFER:$rdir/ -> $dst/"
  # The run's state from submit_job.sh status (it also asks Slurm, so a job killed before writing its final STATUS
  # file is reported failed/timeout instead of "running" forever)
  out=$(login_ssh "bash $(rq "$CLUSTER_TOOLS/submit_job.sh") status $(qargs --label "$label" --run "$run")") ||
    die "could not get the status of $run: does $rdir exist on the cluster? (see above)"
  if [ "$DRY_RUN" = 0 ]; then
    st=$(remote_kv "$out" STATUS)
    info "STATUS=${st:-?}"
    case $st in
      ok|failed|timeout) ;;
      *)
        [ "$FORCE" = 1 ] || die "$run is still ${st:-in an unknown state}: not fetched (--force fetches what exists so far)"
        warn "$run is still ${st:-in an unknown state}; fetching what exists so far (--force)" ;;
    esac
  fi
  bytes=$(transfer_bytes "${CLUSTER_SSH_OPTS-}" "${filters[@]}" "$CLUSTER_XFER:$rdir/" "$dst/")
  check_space "$bytes" "$dst" "run $run"
  [ "$DRY_RUN" = 1 ] || mkdir -p "$dst"
  cluster_rsync --info=progress2 "${filters[@]}" "$CLUSTER_XFER:$rdir/" "$dst/"
  if [ "$DRY_RUN" = 0 ] && [ "$NO_CKPT" = 0 ]; then verify_ckpts "$dst"; fi
  info "fetched $run"
}

if [ "$ALL" = 1 ]; then
  # one child process per run, so a refused or failed run does not stop the others
  pass=(--top "$TOP")
  [ "$NO_CKPT" = 0 ] || pass+=(--no-checkpoints)
  [ "$FORCE" = 0 ] || pass+=(--force)
  [ "$DRY_RUN" = 0 ] || pass+=(--dry-run)
  failed=0
  for r in "${RUNS[@]}"; do
    bash "$0" "$FOLDER" "$r" "${pass[@]}" || failed=$((failed + 1))
  done
  [ "$failed" = 0 ] || die "$failed of ${#RUNS[@]} runs not fetched (see above)"
else
  fetch_one "${RUNS[0]}"
fi
info "next: see train/analysis/README.md for the reports"
