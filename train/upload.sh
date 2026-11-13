#!/bin/bash
# Laptop -> cluster: copy data/FOLDER/package/DATASET/{DATASET.zarr.zip,package.json} to
# <CLUSTER_DATASETS>/<label>/DATASET/ through the transfer node and check the sha256 there.
# A dataset name is never reused: a different zip already under that name is refused; the same zip is skipped.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
usage: train/upload.sh FOLDER DATASET [--dry-run]

  FOLDER     local task folder data/FOLDER/ (as pulled with train/pull.sh)
  DATASET    package in data/FOLDER/package/DATASET/
  --dry-run  print the ssh/rsync commands, run and write nothing

example: train/upload.sh block_raise br-nov13
EOF
}

ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[ ${#ARGS[@]} = 2 ] || { usage >&2; exit 1; }
FOLDER=${ARGS[0]} DATASET=${ARGS[1]}
check_name folder "$FOLDER"
check_name dataset "$DATASET"

load_config
need_var CLUSTER_XFER CLUSTER_DATASETS LOCAL_DATA_ROOT
PKG=$LOCAL_DATA_ROOT/$FOLDER/package/$DATASET
ZIP=$PKG/$DATASET.zarr.zip

if [ -f "$PKG/package.json" ] && [ -f "$ZIP" ]; then
  LABEL=$(json_get "$PKG/package.json" label_dir)
  WANT=$(json_get "$PKG/package.json" zip.sha256)
  [ -n "$LABEL" ] || die "no label_dir in $PKG/package.json"
  [ "$LABEL" != multi ] || die "$DATASET has several labels (label_dir multi); only single-label datasets are supported"
  check_name label "$LABEL"
  step "Checking the local zip ($(human_bytes "$(stat -c %s "$ZIP")"))"
  GOT=$(sha256_of "$ZIP")
  [ "$GOT" = "$WANT" ] || die "local $ZIP does not match package.json (sha256 $GOT, expected $WANT); pull it again"
elif [ "$DRY_RUN" = 1 ]; then
  warn "$PKG/ is not there yet (it would be pulled first); showing the commands with placeholders"
  LABEL="<label_dir>" WANT="<sha256>"
else
  die "missing $ZIP or package.json (pull it first: train/pull.sh $FOLDER --dataset $DATASET)"
fi

DIR=$(rpath "$CLUSTER_DATASETS")/$LABEL/$DATASET
RZIP=$DIR/$DATASET.zarr.zip
UP=$DIR/.upload

step "Checking $CLUSTER_XFER:$RZIP"
probe=$(xfer_ssh "mkdir -p $(rq "$DIR") && if [ -f $(rq "$RZIP") ]; then sha256sum $(rq "$RZIP"); else echo MISSING; fi")
probe=$(printf '%s' "$probe" | tr -d '\r' | tail -1)
if [ "$DRY_RUN" = 0 ] && [ "$probe" != MISSING ]; then
  remote=${probe%% *}
  if [ "$remote" = "$WANT" ]; then
    info "already on the cluster with the same sha256: nothing to upload"
    info "next: train/submit.sh $FOLDER $DATASET --smoke-first --note \"...\""
    exit 0
  fi
  die "a different $DATASET.zarr.zip is already on the cluster (sha256 $remote, local $WANT). Datasets are never overwritten: package under a new name."
fi

# Upload into DIR/.upload/, check the sha256 there, then move into place: a broken copy never gets the real name.
step "Uploading to $CLUSTER_XFER:$DIR/"
cluster_rsync --info=progress2 "$ZIP" "$PKG/package.json" "$CLUSTER_XFER:$UP/"
fin=$(xfer_ssh "cd $(rq "$DIR") && s=\$(sha256sum .upload/$(qargs "$DATASET.zarr.zip") | cut -d' ' -f1) &&
  if [ \"\$s\" = $(qargs "$WANT") ]; then mv .upload/package.json package.json && mv .upload/$(qargs "$DATASET.zarr.zip") $(qargs "$DATASET.zarr.zip") && rmdir .upload && echo VERIFIED; else echo \"BAD \$s\"; fi")
fin=$(printf '%s' "$fin" | tr -d '\r' | tail -1)
if [ "$DRY_RUN" = 0 ]; then
  [ "$fin" = VERIFIED ] || die "the uploaded copy has the wrong sha256 ($fin); run upload.sh again (it resumes in $DIR/.upload/)"
  info "uploaded and verified (sha256 $WANT)"
fi
info "next: train/submit.sh $FOLDER $DATASET --smoke-first --note \"...\""
