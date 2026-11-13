# shellcheck shell=bash
# Shared helpers for the train/*.sh laptop scripts. Source it; do not run it.
#
#   load_config                    config.env, then config.local.env (or $TRAIN_CONFIG_LOCAL), absolute LOCAL_DATA_ROOT
#   need_var NAME...               stop if a setting is empty or still a "<...>" placeholder
#   jetson_ssh / login_ssh / xfer_ssh CMD     run CMD (one string) on that host; with DRY_RUN=1 only print it
#   jetson_rsync / cluster_rsync ARGS...      rsync with the matching ssh options; with DRY_RUN=1 only print it
#   transfer_bytes / check_space   size estimate (rsync --dry-run --stats) and the MIN_FREE_GB check
#   rq PATH / qargs ARGS...        quote a remote path / words for the remote shell
#
# Test hooks (not for normal use): TRAIN_CONFIG_LOCAL = override file instead of train/config.local.env;
# TRAIN_CLUSTER_SRC = folder synced to CLUSTER_TOOLS instead of train/cluster/.

TRAIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'warning: %s\n' "$*" >&2; }
info() { printf '%s\n' "$*" >&2; }
step() { printf '== %s\n' "$*" >&2; }

load_config() {
  [ -f "$TRAIN_DIR/config.env" ] || die "missing $TRAIN_DIR/config.env"
  # shellcheck source=config.env
  source "$TRAIN_DIR/config.env"
  CONFIG_LOCAL=${TRAIN_CONFIG_LOCAL:-$TRAIN_DIR/config.local.env}
  if [ -f "$CONFIG_LOCAL" ]; then
    # shellcheck disable=SC1090
    source "$CONFIG_LOCAL"
  fi
  LOCAL_DATA_ROOT=$(realpath -m "$LOCAL_DATA_ROOT")
}

# need_var NAME...: every named setting must be filled in (not empty, no "<...>" placeholder left)
need_var() {
  local name val
  for name in "$@"; do
    val=${!name-}
    if [ -z "$val" ] || [[ $val == *"<"* ]]; then
      die "$name is not set (value: '${val}'). Fill it in train/config.env (or train/config.local.env)."
    fi
  done
}

# check_name WHAT VALUE: folder, dataset and run names are [A-Za-z0-9_-]+ (also keeps them safe in paths/commands)
check_name() {
  [[ $2 =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || die "invalid $1 '$2': use only letters, digits, '_' and '-'"
}

check_posint() {
  [[ $2 =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer, got '$2'"
}

# rpath PATH: a remote path as given in config.env, made relative to the remote home ("~/x" -> "x", "~" -> ".")
rpath() {
  local p=$1
  # shellcheck disable=SC2088  # a literal "~/" prefix from config.env is meant
  case $p in
    "~") p=. ;;
    "~/"*) p=${p#\~/} ;;
  esac
  printf '%s' "${p:-.}"
}

# rq PATH: rpath, quoted for the remote shell
rq() { printf '%q' "$(rpath "$1")"; }

# qargs ARGS...: each argument quoted for the remote shell, joined by spaces (safe for spaces, quotes, $, ...)
qargs() {
  local out="" a
  for a in "$@"; do out+=" $(printf '%q' "$a")"; done
  printf '%s' "${out# }"
}

# shq ARGS...: readable shell quoting for display: plain words as they are, anything else in '...'
shq() {
  local out="" a
  for a in "$@"; do
    if [[ $a =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then out+=" $a"; else out+=" '${a//\'/\'\\\'\'}'"; fi
  done
  printf '%s' "${out# }"
}

# show_cmd ARGV...: print a command the way it would be typed (dry-run output)
show_cmd() { printf '[dry-run] %s\n' "$(shq "$@")"; }

# _ssh HOST OPTS CMD: ssh with the space-separated OPTS; DRY_RUN prints instead (and prints nothing on stdout)
_ssh() {
  local host=$1 cmd=$3
  local -a opts=()
  read -ra opts <<<"$2"
  if [ "$DRY_RUN" = 1 ]; then
    show_cmd ssh "${opts[@]}" "$host" "$cmd" >&2
    return 0
  fi
  ssh "${opts[@]}" "$host" "$cmd"
}
jetson_ssh() { _ssh "$JETSON_HOST" "${JETSON_SSH_OPTS-}" "$1"; }
login_ssh() { _ssh "$CLUSTER_LOGIN" "${CLUSTER_SSH_OPTS-}" "$1"; }
xfer_ssh() { _ssh "$CLUSTER_XFER" "${CLUSTER_SSH_OPTS-}" "$1"; }

# _rsync OPTS ARGS...: rsync -a --partial over ssh with OPTS; DRY_RUN prints instead
_rsync() {
  local sshopts=$1; shift
  local -a cmd=(rsync -a --partial -e "ssh${sshopts:+ $sshopts}" "$@")
  if [ "$DRY_RUN" = 1 ]; then
    show_cmd "${cmd[@]}" >&2
    return 0
  fi
  "${cmd[@]}"
}
jetson_rsync() { _rsync "${JETSON_SSH_OPTS-}" "$@"; }
cluster_rsync() { _rsync "${CLUSTER_SSH_OPTS-}" "$@"; }

# transfer_bytes OPTS RSYNC_ARGS...: bytes rsync would copy (rsync --dry-run --stats). Prints nothing in DRY_RUN.
transfer_bytes() {
  local sshopts=$1; shift
  local -a cmd=(rsync -a --dry-run --stats --no-human-readable -e "ssh${sshopts:+ $sshopts}" "$@")
  if [ "$DRY_RUN" = 1 ]; then
    show_cmd "${cmd[@]}" >&2
    return 0
  fi
  local out
  out=$("${cmd[@]}") || die "could not list the remote files (rsync --dry-run failed, see above)"
  printf '%s\n' "$out" | sed -n 's/^Total transferred file size: \([0-9,.]*\) bytes.*/\1/p' | tr -d ',.' | head -1
}

human_bytes() { awk -v b="${1:-0}" 'BEGIN { if (b >= 1e9) printf "%.2f GB", b / 1e9; else printf "%.1f MB", b / 1e6 }'; }

# check_space BYTES DIR WHAT: refuse (unless FORCE=1) if DIR's disk would have < MIN_FREE_GB left after BYTES
check_space() {
  local bytes=$1 dir=$2 what=$3 d avail_kb left_gb
  if [ "$DRY_RUN" = 1 ]; then
    info "[dry-run] (would check that $what leaves >= ${MIN_FREE_GB} GB free on the laptop)"
    return 0
  fi
  [ -n "$bytes" ] || die "could not estimate the size of $what"
  if [ "$bytes" = 0 ]; then info "$what: nothing new to copy"; return 0; fi
  d=$dir
  while [ ! -d "$d" ]; do d=$(dirname "$d"); done
  avail_kb=$(df -Pk "$d" | awk 'NR == 2 { print $4 }')
  left_gb=$(awk -v a="$avail_kb" -v b="$bytes" 'BEGIN { printf "%.1f", (a * 1024 - b) / 1e9 }')
  info "$what: $(human_bytes "$bytes") to copy; $(human_bytes $((avail_kb * 1024))) free now, ${left_gb} GB after"
  if awk -v l="$left_gb" -v m="$MIN_FREE_GB" 'BEGIN { exit !(l < m) }'; then
    if [ "$FORCE" = 1 ]; then
      warn "less than MIN_FREE_GB=${MIN_FREE_GB} GB would be left; continuing because of --force"
    else
      die "only ${left_gb} GB would be left on the laptop (MIN_FREE_GB=${MIN_FREE_GB}). Free space, use --no-raw / --top N, or --force."
    fi
  fi
}

# json_get FILE KEY.PATH [DEFAULT]: one value from a JSON file ("totals.n_steps", "zip.sha256", "0.file")
json_get() {
  python3 -I -c '
import json, sys
v = json.load(open(sys.argv[1]))
try:
    for k in sys.argv[2].split("."):
        v = v[int(k)] if isinstance(v, list) else v[k]
except (KeyError, IndexError, ValueError, TypeError):
    if len(sys.argv) > 3:
        print(sys.argv[3]); sys.exit(0)
    sys.exit("missing key " + sys.argv[2] + " in " + sys.argv[1])
print("" if v is None else v)
' "$@"
}

# task_label TASK_YAML: the `label:` of a dataproc task.yaml
task_label() {
  [ -f "$1" ] || die "missing $1 (pull the task folder first: train/pull.sh FOLDER)"
  sed -n 's/^label:[[:space:]]*//p' "$1" | sed 's/[[:space:]]#.*$//; s/[[:space:]]*$//' | tr -d "\"'" | head -1
}

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# run_label FOLDER RUN: label of a run from its local job.json (or train_meta.json), else from the folder's task.yaml
run_label() {
  local dir=$LOCAL_DATA_ROOT/$1/policy/$2 f
  for f in "$dir/job.json" "$dir/train_meta.json"; do
    if [ -f "$f" ]; then json_get "$f" label; return; fi
  done
  if [ "$DRY_RUN" = 1 ] && [ ! -f "$LOCAL_DATA_ROOT/$1/task.yaml" ]; then echo "<label from task.yaml>"; return; fi
  task_label "$LOCAL_DATA_ROOT/$1/task.yaml"
}

# check_label LABEL: like check_name, but a dry-run may show a "<...>" placeholder for a label not known yet
check_label() {
  if [ "$DRY_RUN" = 1 ] && [[ $1 == "<"*">" ]]; then return 0; fi
  check_name label "$1"
}

# sync_tools: copy train/cluster/* + config.env (+ config.local.env) to CLUSTER_TOOLS on the login node
sync_tools() {
  local src=${TRAIN_CLUSTER_SRC:-$TRAIN_DIR/cluster} tools tmp
  tools=$(rpath "$CLUSTER_TOOLS")
  [ -f "$src/submit_job.sh" ] || [ "$DRY_RUN" = 1 ] || die "missing $src/submit_job.sh"
  step "Syncing $(basename "$src")/ + config to $CLUSTER_LOGIN:$tools/"
  # rsync replaces each file by writing a temporary copy and renaming it, so config.env / config.local.env are never
  # missing on the cluster. A stale config.local.env (none here any more) is removed only after the new files are in.
  login_ssh "mkdir -p $(rq "$tools")"
  if [ "$DRY_RUN" = 1 ]; then
    local -a extra=()
    [ ! -f "$CONFIG_LOCAL" ] || extra=("$CONFIG_LOCAL (as config.local.env)")
    cluster_rsync "$src/" "$TRAIN_DIR/config.env" "${extra[@]}" "$CLUSTER_LOGIN:$tools/"
  else
    tmp=$(mktemp -d)
    cp -a "$src/." "$tmp/"
    cp "$TRAIN_DIR/config.env" "$tmp/config.env"
    [ ! -f "$CONFIG_LOCAL" ] || cp "$CONFIG_LOCAL" "$tmp/config.local.env"
    cluster_rsync "$tmp/" "$CLUSTER_LOGIN:$tools/" || { rm -rf "$tmp"; die "could not copy the tools to the cluster"; }
    rm -rf "$tmp"
  fi
  [ -f "$CONFIG_LOCAL" ] || login_ssh "rm -f $(rq "$tools")/config.local.env"
}

# remote_kv OUTPUT KEY: value of the first KEY=value line (CRs stripped)
remote_kv() { printf '%s\n' "$1" | tr -d '\r' | sed -n "s/^$2=//p" | head -1; }
