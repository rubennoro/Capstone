#!/bin/bash
# Jetson -> laptop: mirror a processed task folder into data/FOLDER/ and, optionally, one packaged dataset of it into
# data/FOLDER/package/NAME/. Never deletes local files.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
usage: train/pull.sh FOLDER [--dataset NAME] [--package NAME] [--no-raw] [--force] [--dry-run]

  FOLDER          task folder on the Jetson (<JETSON_DATAPROC_ROOT>/tasks/FOLDER), e.g. block_raise or block_raise_2
  --dataset NAME  also pull the existing package <root>/datasets/<label>/NAME/ (zip sha256 is checked)
  --package NAME  first run `dataproc package NAME FOLDER` on the Jetson, then pull it (like --dataset NAME)
  --no-raw        skip the raw recordings (episode_*/raw, calibration/*/raw, failed/*/raw), the bulk of the size
  --force         copy even if less than MIN_FREE_GB would be left on the laptop
  --dry-run       print the ssh/rsync commands, run and write nothing

example: train/pull.sh block_raise --package br-nov13 --no-raw
EOF
}

FOLDER="" DATASET="" PACKAGE="" NO_RAW=0
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --dataset) DATASET=${2:?--dataset needs a name}; shift 2 ;;
    --package) PACKAGE=${2:?--package needs a name}; shift 2 ;;
    --no-raw) NO_RAW=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) [ -z "$FOLDER" ] || { usage >&2; die "unexpected argument $1"; }; FOLDER=$1; shift ;;
  esac
done
[ -n "$FOLDER" ] || { usage >&2; exit 1; }
[ -z "$DATASET" ] || [ -z "$PACKAGE" ] || [ "$DATASET" = "$PACKAGE" ] || die "give either --dataset or --package"
DATASET=${PACKAGE:-$DATASET}
check_name folder "$FOLDER"
[ -z "$DATASET" ] || check_name dataset "$DATASET"

load_config
need_var JETSON_HOST JETSON_DATAPROC_ROOT LOCAL_DATA_ROOT MIN_FREE_GB
[ -z "$PACKAGE" ] || need_var JETSON_ACTIVATE
ROOT=$(rpath "$JETSON_DATAPROC_ROOT")
LOCAL=$LOCAL_DATA_ROOT/$FOLDER

# 1. package on the Jetson
if [ -n "$PACKAGE" ]; then
  step "Packaging $PACKAGE from $FOLDER on $JETSON_HOST"
  jetson_ssh "cd && export DATAPROC_ROOT=$(rq "$ROOT") && $JETSON_ACTIVATE && dataproc package $(qargs "$PACKAGE" "$FOLDER")" ||
    die "dataproc package failed (see above). If $PACKAGE already exists on the Jetson, pull it with --dataset $PACKAGE."
fi

# 2. the task folder
filters=(--exclude=/.staging/)
[ "$NO_RAW" = 0 ] || filters+=(--exclude=/episode_*/raw/ --exclude=/calibration/*/raw/ --exclude=/failed/*/raw/)
src="$JETSON_HOST:$ROOT/tasks/$FOLDER/"
step "Pulling $src -> $LOCAL/"
bytes=$(transfer_bytes "${JETSON_SSH_OPTS-}" "${filters[@]}" "$src" "$LOCAL/")
check_space "$bytes" "$LOCAL" "task folder $FOLDER"
[ "$DRY_RUN" = 1 ] || mkdir -p "$LOCAL"
jetson_rsync --info=progress2 "${filters[@]}" "$src" "$LOCAL/"

# 3. the dataset package
if [ -n "$DATASET" ]; then
  if [ -f "$LOCAL/task.yaml" ]; then
    LABEL=$(task_label "$LOCAL/task.yaml")
    [ -n "$LABEL" ] || die "no label: in $LOCAL/task.yaml"
  elif [ "$DRY_RUN" = 1 ]; then
    LABEL="<label from task.yaml>"
  else
    die "the pulled folder has no task.yaml: $LOCAL/task.yaml"
  fi
  dst=$LOCAL/package/$DATASET
  src="$JETSON_HOST:$ROOT/datasets/$LABEL/$DATASET/"
  step "Pulling dataset $src -> $dst/"
  bytes=$(transfer_bytes "${JETSON_SSH_OPTS-}" "$src" "$dst/")
  check_space "$bytes" "$dst" "dataset $DATASET"
  [ "$DRY_RUN" = 1 ] || mkdir -p "$dst"
  jetson_rsync --info=progress2 "$src" "$dst/"
  if [ "$DRY_RUN" = 0 ]; then
    zip=$dst/$DATASET.zarr.zip
    [ -f "$zip" ] && [ -f "$dst/package.json" ] || die "$dst/ has no $DATASET.zarr.zip + package.json"
    want=$(json_get "$dst/package.json" zip.sha256)
    got=$(sha256_of "$zip")
    [ "$got" = "$want" ] || die "sha256 mismatch for $zip (got $got, package.json says $want). Delete it and pull again."
    info "sha256 ok; $(json_get "$dst/package.json" totals.episodes) episodes, $(json_get "$dst/package.json" totals.n_steps) steps"
  fi
fi

step "Done"
if [ -n "$DATASET" ]; then
  info "next: train/upload.sh $FOLDER $DATASET    (or train/go.sh $FOLDER $DATASET)"
fi
