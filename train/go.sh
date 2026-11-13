#!/bin/bash
# One command from Jetson data to a submitted training job:
# (pull.sh, if --pull/--package) -> upload.sh -> submit.sh --smoke-first (or the full run only with --no-smoke).
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
usage: train/go.sh FOLDER DATASET [--pull] [--package] [--no-raw] [--no-smoke] [--epochs N] [--note TEXT]
                   [--dry-run] [-- hydra overrides...]

  --pull      first pull the task folder and the existing package DATASET from the Jetson
  --package   first run `dataproc package DATASET FOLDER` on the Jetson, then pull (implies --pull)
  --no-raw    with --pull/--package: skip the raw recordings
  --no-smoke  submit only the full run (default: a 3-epoch smoke run first, the full run after it)
  --epochs N  epochs of the full run (default DEFAULT_EPOCHS, lowered to fit SLURM_TIME)
  --note TEXT free text stored with the run(s)
  --dry-run   print every ssh/rsync command, run and write nothing
  -- ...      extra Hydra overrides passed to submit.sh

example: train/go.sh block_raise br-nov13 --package --no-raw --note "first rig data"
EOF
}

PULL=0 PACKAGE=0 NO_RAW=0 SMOKE=1 EPOCHS="" NOTE="" HAVE_NOTE=0 ARGS=() OVERRIDES=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --pull) PULL=1; shift ;;
    --package) PACKAGE=1; PULL=1; shift ;;
    --no-raw) NO_RAW=1; shift ;;
    --no-smoke) SMOKE=0; shift ;;
    --epochs) EPOCHS=${2:?--epochs needs a number}; shift 2 ;;
    --note) NOTE=${2?--note needs a text}; HAVE_NOTE=1; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; OVERRIDES=("$@"); break ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[ ${#ARGS[@]} = 2 ] || { usage >&2; exit 1; }
FOLDER=${ARGS[0]} DATASET=${ARGS[1]}
check_name folder "$FOLDER"
check_name dataset "$DATASET"

[ "$NO_RAW" = 0 ] || [ "$PULL" = 1 ] || die "--no-raw only applies to pulling: add --pull or --package"

dry=()
[ "$DRY_RUN" = 0 ] || dry=(--dry-run)

if [ "$PULL" = 1 ]; then
  pull=("$FOLDER")
  if [ "$PACKAGE" = 1 ]; then pull+=(--package "$DATASET"); else pull+=(--dataset "$DATASET"); fi
  [ "$NO_RAW" = 0 ] || pull+=(--no-raw)
  bash "$TRAIN_DIR/pull.sh" "${pull[@]}" "${dry[@]}"
fi

bash "$TRAIN_DIR/upload.sh" "$FOLDER" "$DATASET" "${dry[@]}"

sub=("$FOLDER" "$DATASET")
if [ "$SMOKE" = 1 ]; then sub+=(--smoke-first); fi
[ -z "$EPOCHS" ] || sub+=(--epochs "$EPOCHS")
[ "$HAVE_NOTE" = 0 ] || sub+=(--note "$NOTE")
sub+=("${dry[@]}")
[ ${#OVERRIDES[@]} = 0 ] || sub+=(-- "${OVERRIDES[@]}")
bash "$TRAIN_DIR/submit.sh" "${sub[@]}"
