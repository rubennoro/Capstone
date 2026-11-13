#!/bin/bash
# Tests for the laptop scripts in train/, with no network: fake `ssh` / `rsync` / `sha256sum` shims go first on PATH.
#   fake ssh    runs the remote command locally in a fake remote home (HOME set; one home for the Jetson, one shared
#               by the cluster login + transfer nodes, like the real shared home), with fake sbatch/squeue/sacct there
#   fake rsync  rewrites host:path into the matching fake home and calls the real rsync
# The cluster side is tested with a stub submit_job.sh (TRAIN_CLUSTER_SRC hook) and with the real train/cluster/ one.
#
#   bash train/tests/run_tests.sh          (TMPDIR picks where the temporary tree goes; it is removed at the end)
#   KEEP=1 bash train/tests/run_tests.sh   keep the temporary tree for inspection
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN=$(cd "$HERE/.." && pwd)
REAL_RSYNC=$(command -v rsync) || { echo "rsync is needed for the tests" >&2; exit 1; }
REAL_SHA=$(command -v sha256sum)
T=$(mktemp -d "${TMPDIR:-/tmp}/train-tests.XXXXXX")
[ "${KEEP:-0}" = 1 ] && echo "test tree: $T" || trap 'rm -rf "$T"' EXIT

PASS=0 FAIL=0 FAILED=()
ok() { PASS=$((PASS + 1)); printf '  ok    %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); FAILED+=("$1"); printf '  FAIL  %s\n' "$1"; [ -z "${2:-}" ] || printf '%s\n' "$2" | sed 's/^/        /' | tail -25; }
check() { local name=$1; shift; if "$@"; then ok "$name"; else bad "$name" "$(cat "$T/out" 2>/dev/null)"; fi; }
# run SCRIPT ARGS...: run a train script, output in $T/out, exit code in $RC
run() { bash "$TRAIN/$1" "${@:2}" >"$T/out" 2>&1; RC=$?; }
has() { grep -qF -- "$1" "$T/out"; }
calls() { wc -l <"$T/calls.log"; }
snap() { (cd "$T" && find homes local -printf '%p %s %T@\n' 2>/dev/null | sort | md5sum); }

# ------------------------------------------------------------------------------------------------ shims + fixtures
mkdir -p "$T/bin" "$T/cluster_bin" "$T/homes/jetson" "$T/homes/cluster" "$T/local"
: >"$T/calls.log"
export FAKE_LOG=$T/calls.log FAKE_HOMES=$T/homes FAKE_CLUSTER_BIN=$T/cluster_bin REAL_RSYNC REAL_SHA FAKE_T=$T

cat >"$T/bin/ssh" <<'EOF'
#!/bin/bash
while [ $# -gt 0 ]; do
  case $1 in
    -[bcDEeFIiJLlmOoopQRSWw]) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
host=$1; shift
printf 'ssh %s :: %s\n' "$host" "$*" >>"$FAKE_LOG"
case $host in
  *jetson*) h=$FAKE_HOMES/jetson extra= ;;
  *login*|*xfer*) h=$FAKE_HOMES/cluster extra=$FAKE_CLUSTER_BIN ;;
  *) echo "fake ssh: unknown host $host" >&2; exit 255 ;;
esac
cd "$h" && HOME=$h USER=tester PATH=${extra:+$extra:}$PATH exec bash -c "$*"
EOF

cat >"$T/bin/rsync" <<'EOF'
#!/bin/bash
args=() prev=
for a in "$@"; do
  if [ "$prev" != -e ] && [[ $a != -* && $a =~ ^([^/:]+):(.*)$ ]]; then
    host=${BASH_REMATCH[1]} p=${BASH_REMATCH[2]}
    case $host in
      *jetson*) h=$FAKE_HOMES/jetson ;;
      *login*|*xfer*) h=$FAKE_HOMES/cluster ;;
      *) echo "fake rsync: unknown host $host" >&2; exit 255 ;;
    esac
    case $p in /*) ;; *) a=$h/$p ;; esac
  fi
  args+=("$a"); prev=$a
done
printf 'rsync %s\n' "$*" >>"$FAKE_LOG"
exec "$REAL_RSYNC" "${args[@]}"
EOF

cat >"$T/bin/sha256sum" <<'EOF'
#!/bin/bash
printf 'sha256sum %s\n' "$*" >>"$FAKE_LOG"
exec "$REAL_SHA" "$@"
EOF

cat >"$T/cluster_bin/sbatch" <<'EOF'
#!/bin/bash
n=$(( $(cat "$FAKE_T/sbatch_n" 2>/dev/null || echo 0) + 1 )); echo "$n" >"$FAKE_T/sbatch_n"
printf '%s\n' "$@" >"$FAKE_T/sbatch_$n.args"
printf 'sbatch %s\n' "$*" >>"$FAKE_LOG"
echo "Submitted batch job $((122 + n))"
EOF
# squeue: nothing in the queue; on call number FLIP_AT it writes `ok` into FLIP_STATUS (to end status.sh --wait)
cat >"$T/cluster_bin/squeue" <<'EOF'
#!/bin/bash
n=$(( $(cat "$FAKE_T/squeue_n" 2>/dev/null || echo 0) + 1 )); echo "$n" >"$FAKE_T/squeue_n"
if [ -f "$FAKE_T/flip" ]; then read -r at file <"$FAKE_T/flip"; [ "$n" -ge "$at" ] && echo ok >"$file"; fi
exit 0
EOF
printf '#!/bin/bash\n[ -f "$FAKE_T/sacct_out" ] && cat "$FAKE_T/sacct_out"\nexit 0\n' >"$T/cluster_bin/sacct"
printf '#!/bin/bash\nexit 0\n' >"$T/cluster_bin/module"

# make_pkg.py DIR NAME LABEL N_STEPS [bad]: a fake dataproc package (zip + package.json; bad = wrong sha256)
cat >"$T/make_pkg.py" <<'EOF'
import hashlib, json, os, sys
d, name, label, n = sys.argv[1:5]
os.makedirs(d, exist_ok=True)
z = os.path.join(d, name + ".zarr.zip")
data = os.urandom(4096)
open(z, "wb").write(data)
sha = hashlib.sha256(data).hexdigest()
if len(sys.argv) > 5:
    sha = "0" * 64
rec = {"dataset": name, "label_dir": label, "labels": [label], "folders": ["block_raise"], "rig": "rig_pi_v3",
       "episodes": [], "excluded": [], "totals": {"episodes": 2, "n_steps": int(n), "per_label": {label: 2}, "excluded": 0},
       "warnings": [], "zip": {"name": name + ".zarr.zip", "size_bytes": len(data), "sha256": sha},
       "dataproc_version": "test", "created": "2026-11-13T00:00:00"}
json.dump(rec, open(os.path.join(d, "package.json"), "w"), indent=2)
EOF

# Jetson: dataproc env with a fake `dataproc package`, one task folder, two packaged datasets (one with a bad sha)
J=$T/homes/jetson
mkdir -p "$J/envs/env_dataproc/bin"
printf 'export PATH="$HOME/envs/env_dataproc/bin:$PATH"\n' >"$J/envs/env_dataproc/bin/activate"
cat >"$J/envs/env_dataproc/bin/dataproc" <<EOF
#!/bin/bash
[ "\$1" = package ] || { echo "fake dataproc: only package" >&2; exit 2; }
root=\${DATAPROC_ROOT:-\$HOME/data}
label=\$(sed -n 's/^label: *//p' "\$root/tasks/\$3/task.yaml")
d=\$root/datasets/\$label/\$2
[ ! -e "\$d" ] || { echo "dataproc package: dataset \$2 exists" >&2; exit 1; }
python3 -I "$T/make_pkg.py" "\$d" "\$2" "\$label" 37971 && echo "wrote \$d/\$2.zarr.zip"
EOF
chmod +x "$T"/bin/* "$T"/cluster_bin/* "$J/envs/env_dataproc/bin/dataproc"
TASK=$J/data/tasks/block_raise
mkdir -p "$TASK"/{calibration/pi-cal/raw,calibration/pi-cal/processed,episode_1/raw,episode_1/processed/episode.zarr}
mkdir -p "$TASK"/{episode_2/raw,episode_2/processed/episode.zarr,.staging/junk/raw,failed/pi-x/raw}
printf 'label: block_raise\nrig: rig_pi_v3\n' >"$TASK/task.yaml"
printf 'slot,pi_name\nepisode_1,a\nepisode_2,b\n' >"$TASK/episodes.csv"
for f in calibration/pi-cal/raw/cam.bin episode_1/raw/cam.bin episode_2/raw/cam.bin failed/pi-x/raw/cam.bin .staging/junk/raw/cam.bin; do
  head -c 2048 /dev/urandom >"$TASK/$f"
done
echo '{}' >"$TASK/calibration/pi-cal/processed/fit.json"
echo '{}' >"$TASK/episode_1/processed/episode.zarr/.zgroup"
echo '{}' >"$TASK/episode_2/processed/episode.zarr/.zgroup"
echo '{"stage":"slam"}' >"$TASK/failed/pi-x/reason.json"
python3 -I "$T/make_pkg.py" "$J/data/datasets/block_raise/ds1" ds1 block_raise 37971
python3 -I "$T/make_pkg.py" "$J/data/datasets/block_raise/dsbad" dsbad block_raise 37971 bad
python3 -I "$T/make_pkg.py" "$J/data/datasets/block_raise/dsbig" dsbig block_raise 1000000

# Cluster: a clone with what submit_job.sh checks for
C=$T/homes/cluster
mkdir -p "$C/CapstoneProject/Capstone/behavior_prompting/behavior_prompting/train_network" \
  "$C/CapstoneProject/Capstone/scripts" "$C/CapstoneProject/Capstone/controlSW/franka_deploy/franka_deploy"
touch "$C/CapstoneProject/Capstone/behavior_prompting/behavior_prompting/train_network/train.py" \
  "$C/CapstoneProject/Capstone/scripts/package_run.py"

# Stub cluster tools (records its argv, prints RUN=/JOB=, creates the run dir)
mkdir -p "$T/stub_tools"
cat >"$T/stub_tools/submit_job.sh" <<'EOF'
#!/bin/bash
d=$(cd "$(dirname "$0")" && pwd)
n=$(( $(cat "$d/stub_n" 2>/dev/null || echo 0) + 1 )); echo "$n" >"$d/stub_n"
printf '%s\n' "$@" >"$d/stub_argv_$n"
run=$(printf 'run_%03d_stub' "$n")
mkdir -p "$HOME/CapstoneProject/policies/block_raise/$run"
echo "stub: submitted $1"
echo "RUN=$run"
echo "JOB=$((900 + n))"
EOF

# Laptop config used by every test (TRAIN_CONFIG_LOCAL hook); TEST_* env vars vary it per test
cat >"$T/config.local.env" <<EOF
JETSON_HOST="\${TEST_JETSON_HOST:-tester@jetson.test}"
CLUSTER_USER="tester"
CLUSTER_LOGIN="tester@login.test"
CLUSTER_XFER="tester@xfer.test"
LOCAL_DATA_ROOT="\${TEST_LOCAL_ROOT:-$T/local/data}"
MIN_FREE_GB="\${TEST_MIN_FREE_GB:-1}"
EOF
export TRAIN_CONFIG_LOCAL=$T/config.local.env TRAIN_STATUS_POLL_SEC=0
export PATH="$T/bin:$PATH"
L=$T/local/data
TODAY=$(date +%F)

# --------------------------------------------------------------------------------------------------------- tests
echo "== static checks"
for f in lib.sh pull.sh upload.sh submit.sh status.sh fetch.sh go.sh analysis/eval.sh tests/run_tests.sh; do
  check "bash -n $f" bash -n "$TRAIN/$f"
done
SHELLCHECK=${SHELLCHECK:-$(command -v shellcheck)}
if [ -n "$SHELLCHECK" ]; then
  "$SHELLCHECK" -x -P SCRIPTDIR -S warning -e SC1091 "$TRAIN"/*.sh "$TRAIN/analysis/eval.sh" "$TRAIN/tests/run_tests.sh" >"$T/out" 2>&1
  check "shellcheck (warnings and errors)" test ! -s "$T/out"
else
  echo "  skip  shellcheck (not installed; SHELLCHECK=/path/to/shellcheck to use one)"
fi
for s in pull.sh upload.sh submit.sh status.sh fetch.sh go.sh analysis/eval.sh; do
  run "$s" -h; check "$s -h" eval '[ $RC = 0 ] && has usage:'
done

echo "== config"
TEST_JETSON_HOST='<user@jetson-host>' run pull.sh block_raise --dry-run
check "unfilled JETSON_HOST is named in the error" eval '[ $RC != 0 ] && has JETSON_HOST'
(export TRAIN_CONFIG_LOCAL=$T/none.env; run upload.sh block_raise ds1 --dry-run; [ $RC = 0 ]) >/dev/null
check "config.env alone (cluster defaults) loads" test $? = 0
run pull.sh "../evil" --dry-run
check "bad folder name refused" eval '[ $RC != 0 ] && has "invalid folder"'

echo "== dry-run changes nothing"
before=$(snap) n0=$(calls)
run pull.sh block_raise --package dsx --no-raw --dry-run;        r1=$RC
run upload.sh block_raise ds1 --dry-run;                           r2=$RC
run submit.sh block_raise ds1 --smoke-first --note "a b" --dry-run -- x=1; r3=$RC
run go.sh block_raise ds1 --package --no-raw --note "n" --dry-run; r4=$RC
run fetch.sh block_raise run_001_x --top 1 --dry-run;             r5=$RC
run status.sh block_raise run_001_x --dry-run;                    r6=$RC
check "dry-runs exit 0" test "$r1$r2$r3$r4$r5$r6" = 000000
check "dry-run prints commands" has "[dry-run] ssh"
check "dry-run: no ssh/rsync/sbatch was run" test "$(calls)" = "$n0"
check "dry-run: no file created or changed" test "$(snap)" = "$before"

echo "== pull"
run pull.sh block_raise --dataset ds1 --no-raw
check "pull --no-raw --dataset exits 0" test $RC = 0
check "task.yaml + processed pulled" test -f "$L/block_raise/task.yaml" -a -f "$L/block_raise/episode_1/processed/episode.zarr/.zgroup"
check "--no-raw: no episode/calibration/failed raw" test ! -e "$L/block_raise/episode_1/raw" -a ! -e "$L/block_raise/calibration/pi-cal/raw" -a ! -e "$L/block_raise/failed/pi-x/raw"
check "failed/ reason.json pulled" test -f "$L/block_raise/failed/pi-x/reason.json"
check ".staging excluded" test ! -e "$L/block_raise/.staging"
check "dataset pulled into package/ds1" test -f "$L/block_raise/package/ds1/ds1.zarr.zip" -a -f "$L/block_raise/package/ds1/package.json"
check "sha256 verified" has "sha256 ok"
check "size printed" has "to copy"
touch "$L/block_raise/local_note.txt"
run pull.sh block_raise
check "pull with raw exits 0" test $RC = 0
check "raw pulled without --no-raw" test -f "$L/block_raise/episode_1/raw/cam.bin" -a -f "$L/block_raise/calibration/pi-cal/raw/cam.bin"
check "still no .staging" test ! -e "$L/block_raise/.staging"
check "local files never deleted" test -f "$L/block_raise/local_note.txt"
run pull.sh block_raise --dataset dsbad
check "sha256 mismatch fails" eval '[ $RC != 0 ] && has "sha256 mismatch"'
run pull.sh block_raise --package ds2 --no-raw
check "pull --package runs dataproc package and pulls" eval '[ $RC = 0 ] && [ -f "$L/block_raise/package/ds2/ds2.zarr.zip" ] && has "sha256 ok"'
check "dataproc ran with DATAPROC_ROOT" grep -q "DATAPROC_ROOT=data .*dataproc package ds2 block_raise" "$T/calls.log"
run pull.sh block_raise --package ds2
check "existing package name: hint to use --dataset" eval '[ $RC != 0 ] && has "--dataset ds2"'
rm -rf "$T/local/fresh"
TEST_LOCAL_ROOT=$T/local/fresh TEST_MIN_FREE_GB=999999 run pull.sh block_raise
check "disk check refuses" eval '[ $RC != 0 ] && has "MIN_FREE_GB" && [ ! -e "$T/local/fresh/block_raise" ]'
TEST_LOCAL_ROOT=$T/local/fresh TEST_MIN_FREE_GB=999999 run pull.sh block_raise --force
check "--force overrides the disk check" eval '[ $RC = 0 ] && [ -f "$T/local/fresh/block_raise/task.yaml" ]'

echo "== upload"
RD=$C/CapstoneProject/datasets/block_raise
run upload.sh block_raise ds1
check "upload exits 0 and verifies" eval '[ $RC = 0 ] && has "uploaded and verified"'
check "zip + package.json on the cluster, no .upload left" test -f "$RD/ds1/ds1.zarr.zip" -a -f "$RD/ds1/package.json" -a ! -e "$RD/ds1/.upload"
check "remote copy identical" cmp -s "$RD/ds1/ds1.zarr.zip" "$L/block_raise/package/ds1/ds1.zarr.zip"
n0=$(calls); run upload.sh block_raise ds1
check "same sha256: skipped" eval '[ $RC = 0 ] && has "nothing to upload"'
check "same sha256: no rsync" eval '! tail -n +$((n0 + 1)) "$T/calls.log" | grep -q "^rsync"'
run upload.sh block_raise ds2
echo x >>"$RD/ds2/ds2.zarr.zip"
run upload.sh block_raise ds2
check "different sha256 on the cluster: refused" eval '[ $RC != 0 ] && has "never overwritten"'
run upload.sh block_raise nope
check "missing local package: error" eval '[ $RC != 0 ] && has "pull it first"'

echo "== submit (stub cluster tools)"
python3 -I "$T/make_pkg.py" "$L/block_raise/package/dsbig" dsbig block_raise 1000000
NOTE=$'it\'s a "quoted" note with $HOME and $(echo hacked) and `x`'
TRAIN_CLUSTER_SRC=$T/stub_tools run submit.sh block_raise ds1 --smoke-first --epochs 60 --note "$NOTE" -- 'training.lr=1e-4' 'x="a b"'
TOOLS=$C/CapstoneProject/train_tools
check "smoke-first exits 0" test $RC = 0
check "next commands printed" has "train/status.sh block_raise run_002_stub"
check "tools + config synced" test -f "$TOOLS/submit_job.sh" -a -f "$TOOLS/config.env" -a -f "$TOOLS/config.local.env"
check "sync: no rm before the copy" eval '! grep "train_tools" "$T/calls.log" | grep -q "rm -f"'
run status.sh block_raise run_001_stub --dry-run
check "ControlMaster options (local ~ in ControlPath) reach ssh" has "-o ControlMaster=auto -o 'ControlPath=~/.ssh/cm-%r@%h:%p' -o ControlPersist=10m"
check "remote config.local.env there before" test -f "$TOOLS/config.local.env"
n0=$(calls)
( export TRAIN_CONFIG_LOCAL=$T/missing.env; source "$TRAIN/lib.sh"; load_config; sync_tools ) >"$T/out" 2>&1
check "no local config.local.env: remote one removed, config.env kept" test ! -e "$TOOLS/config.local.env" -a -f "$TOOLS/config.env"
check "the stale file is removed only after the copy" eval 'tail -n +$((n0 + 1)) "$T/calls.log" | grep -E "^(rsync|ssh .*rm -f)" | tail -1 | grep -q "rm -f"'
# shellcheck disable=SC2034  # a1/a2 are used inside the eval'd checks
a1=$(cat "$TOOLS/stub_argv_1" 2>/dev/null) a2=$(cat "$TOOLS/stub_argv_2" 2>/dev/null)
check "smoke call: mode smoke, 3 epochs" eval '[[ $a1 == *$'"'"'--mode\nsmoke\n--epochs\n3\n'"'"'* ]]'
check "full call: after the smoke job (901), 60 epochs" eval '[[ $a2 == *$'"'"'--mode\nfull\n--epochs\n60\n'"'"'* && $a2 == *$'"'"'--after\n901\n'"'"'* ]]'
check "note arrives unchanged on the cluster" eval 'grep -qxF -- "$NOTE" "$TOOLS/stub_argv_1"'
check "overrides arrive after --" eval '[[ $a2 == *$'"'"'--\ntraining.lr=1e-4\nx="a b"'"'"'* ]]'
J1=$L/block_raise/policy/run_001_stub/job.json J2=$L/block_raise/policy/run_002_stub/job.json
check "job.json written for both runs" test -f "$J1" -a -f "$J2"
check "job.json fields" python3 -I -c '
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
assert a["mode"] == "smoke" and a["epochs"] == 3 and a["job_id"] == "901" and a["after"] is None
assert b["mode"] == "full" and b["epochs"] == 60 and b["job_id"] == "902" and b["after"] == "901"
assert b["note"] == sys.argv[3] and b["overrides"] == ["training.lr=1e-4", "x=\"a b\""]
assert b["folder"] == "block_raise" and b["label"] == "block_raise" and b["dataset"] == "ds1" and b["topk"] == 1
assert set(b) == {"folder","label","dataset","run","mode","epochs","topk","job_id","after","note","submitted","overrides"}
' "$J1" "$J2" "$NOTE"
TRAIN_CLUSTER_SRC=$T/stub_tools run submit.sh block_raise ds1 --epochs 300 --dry-run
check "clamp: 37971 steps, 8 h - 40 min -> 210 epochs" eval 'has "using 210" && has "--epochs 210"'
TRAIN_CLUSTER_SRC=$T/stub_tools run submit.sh block_raise dsbig --dry-run
check "clamp: 1e6 steps -> 8 epochs (55 min each)" eval 'has "60 epochs do not fit" && has "--epochs 8" && has "7:20"'
TRAIN_CLUSTER_SRC=$T/stub_tools run submit.sh block_raise ds1 --dry-run
check "default 60 epochs, estimate 2:05" eval 'has "--epochs 60" && has "2:05"'
TRAIN_CLUSTER_SRC=$T/stub_tools run submit.sh block_raise ds1 --smoke --topk 2 --after 77 --dry-run
check "--smoke/--topk/--after passed" eval 'has "--mode smoke --epochs 3 --topk 2" && has "--after 77"'
TRAIN_CLUSTER_SRC=$T/stub_tools run submit.sh block_raise ds1 --dry-run
check "default --topk is TOPK=1" has "--topk 1"

echo "== submit + status + eval (real train/cluster/submit_job.sh, fake sbatch)"
if [ -f "$TRAIN/cluster/submit_job.sh" ]; then
  RR=$C/CapstoneProject/policies/block_raise
  rm -rf "$RR"
  run submit.sh block_raise ds1 --smoke-first --note "real chain"
  check "real submit_job.sh: exits 0" test $RC = 0
  check "runs run_001_${TODAY}_smoke + run_002_$TODAY" test -f "$L/block_raise/policy/run_001_${TODAY}_smoke/job.json" -a -f "$L/block_raise/policy/run_002_$TODAY/job.json"
  check "sbatch dependency on the smoke job (123)" grep -qx -- "--dependency=afterok:123" "$T/sbatch_2.args"
  check "train_meta.json + STATUS on the cluster" test -f "$RR/run_002_$TODAY/train_meta.json" -a "$(cat "$RR/run_002_$TODAY/STATUS" 2>/dev/null)" = queued
  check "job ids parsed" python3 -I -c 'import json,sys; assert json.load(open(sys.argv[1]))["job_id"] == "124"' "$L/block_raise/policy/run_002_$TODAY/job.json"
  echo running >"$RR/run_002_$TODAY/STATUS"
  touch "$L/block_raise/policy/run_002_$TODAY/job.json"
  run status.sh block_raise
  check "status (newest run) prints STATUS=running" eval '[ $RC = 0 ] && has "run: run_002_$TODAY" && has "STATUS=running"'
  rm -f "$T/squeue_n"; echo "3 $RR/run_002_$TODAY/STATUS" >"$T/flip"
  run status.sh block_raise "run_002_$TODAY" --wait
  check "status --wait polls until ok" eval '[ $RC = 0 ] && has "STATUS=ok" && has "train/fetch.sh block_raise run_002_$TODAY"'
  rm -f "$T/flip"
  mkdir -p "$RR/run_002_$TODAY/checkpoints"; head -c 100 /dev/urandom >"$RR/run_002_$TODAY/checkpoints/top1_epoch=0001-val_action_mse_error_all=0.1.ckpt"
  run analysis/eval.sh block_raise "run_002_$TODAY" --ckpt top1 --per-demo 5 --attention
  check "eval.sh submits eval.sbatch and prints JOB=" eval '[ $RC = 0 ] && has "JOB=125" && grep -q "eval.sbatch" "$T/sbatch_3.args"'
  run go.sh block_raise ds3 --package --no-raw --note "go end to end"
  check "go.sh: package + pull + upload + smoke + full" eval '[ $RC = 0 ] && [ -f "$RD/ds3/ds3.zarr.zip" ] && [ -f "$L/block_raise/policy/run_003_${TODAY}_smoke/job.json" ] && [ -f "$L/block_raise/policy/run_004_$TODAY/job.json" ]'
  check "go.sh: full run after the smoke job" grep -qx -- "--dependency=afterok:126" "$T/sbatch_5.args"
  run go.sh block_raise ds3 --no-raw
  check "go.sh --no-raw without --pull: error" eval '[ $RC != 0 ] && has "add --pull"'
  run go.sh block_raise ds3 --no-smoke --epochs 5
  check "go.sh --no-smoke: one full run, upload skipped" eval '[ $RC = 0 ] && has "nothing to upload" && [ -f "$L/block_raise/policy/run_005_$TODAY/job.json" ] && [ ! -e "$T/sbatch_7.args" ]'
else
  echo "  skip  train/cluster/submit_job.sh not there yet"
fi

echo "== fetch"
RR=$C/CapstoneProject/policies/block_raise
# make_run NAME STATUS: a packaged run as train.sbatch leaves it
make_run() {
  local d=$RR/$1
  rm -rf "$d"; mkdir -p "$d"/{checkpoints,plots,wandb/offline-run-1,eval/top1/videos,logs}
  echo "$2" >"$d/STATUS"; echo '{}' >"$d/summary.json"; echo "# run" >"$d/README.md"
  echo x >"$d/wandb/offline-run-1/run.wandb"; echo '{}' >"$d/eval/top1/eval.json"
  local k s; for k in 1 2 3; do s=$((1000 * k)); head -c $s /dev/urandom >"$d/checkpoints/top${k}_epoch=00${k}0-val_action_mse_error_all=0.00${k}.ckpt"; done
  python3 -I -c '
import json, os, sys
d = sys.argv[1]
fs = sorted(f for f in os.listdir(d) if f.endswith(".ckpt"))
json.dump([{"rank": i + 1, "file": f, "epoch": (i + 1) * 10, "val_action_mse": 0.001 * (i + 1),
            "size_bytes": os.path.getsize(os.path.join(d, f))} for i, f in enumerate(fs)], open(os.path.join(d, "checkpoints.json"), "w"))
' "$d/checkpoints"
}
make_run run_010_f ok
run fetch.sh block_raise run_010_f --top 1
F=$L/block_raise/policy/run_010_f
check "fetch --top 1 exits 0" test $RC = 0
check "only top1 checkpoint fetched" eval '[ "$(ls "$F/checkpoints" | grep -c ckpt$)" = 1 ] && ls "$F/checkpoints" | grep -q "^top1_"'
check "checkpoints.json, wandb/, eval/, summary fetched" test -f "$F/checkpoints/checkpoints.json" -a -f "$F/wandb/offline-run-1/run.wandb" -a -f "$F/eval/top1/eval.json" -a -f "$F/summary.json"
check "checkpoint sizes verified" has "ok  top1_"
run fetch.sh block_raise run_010_f --top 3
check "--top 3 gets all three" eval '[ $RC = 0 ] && [ "$(ls "$F/checkpoints" | grep -c ckpt$)" = 3 ]'
make_run run_014_f ok
run fetch.sh block_raise run_014_f
check "default (TOPK=1) fetches only top1" eval '[ $RC = 0 ] && [ "$(ls "$L/block_raise/policy/run_014_f/checkpoints" | grep -c ckpt$)" = 1 ]'
make_run run_011_f ok
run fetch.sh block_raise run_011_f --no-checkpoints
check "--no-checkpoints: none, but checkpoints.json" eval '[ $RC = 0 ] && ! ls "$L/block_raise/policy/run_011_f/checkpoints" | grep -q ckpt$ && [ -f "$L/block_raise/policy/run_011_f/checkpoints/checkpoints.json" ]'
make_run run_012_f ok
python3 -I -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); d[0]["size_bytes"] += 1; json.dump(d, open(p, "w"))' "$RR/run_012_f/checkpoints/checkpoints.json"
run fetch.sh block_raise run_012_f --top 1
check "size mismatch fails" eval '[ $RC != 0 ] && has "size mismatch"'
make_run run_013_f running
run fetch.sh block_raise run_013_f
check "running run refused" eval '[ $RC != 0 ] && has "still running" && [ ! -e "$L/block_raise/policy/run_013_f" ]'
run fetch.sh block_raise run_013_f --force --top 1
check "--force fetches a running run" eval '[ $RC = 0 ] && [ -f "$L/block_raise/policy/run_013_f/summary.json" ]'
run fetch.sh block_raise run_999_missing
check "missing run: error" eval '[ $RC != 0 ] && has "run not found" && has "exist on the cluster"'
# a job killed before it wrote its final STATUS: the file says running, Slurm (sacct) says FAILED
make_run run_015_f running
printf '{\n  "job_id": "555"\n}\n' >"$RR/run_015_f/train_meta.json"
echo 'FAILED|1:0|00:10:00|start|end|gpu|node1' >"$T/sacct_out"
run fetch.sh block_raise run_015_f
rm -f "$T/sacct_out"
check "STATUS file running but sacct FAILED: fetched" eval '[ $RC = 0 ] && has "STATUS=failed" && [ -f "$L/block_raise/policy/run_015_f/summary.json" ]'
TEST_MIN_FREE_GB=999999 run fetch.sh block_raise run_011_f --top 1
check "fetch disk check refuses" eval '[ $RC != 0 ] && has "MIN_FREE_GB"'
if [ -f "$TRAIN/cluster/submit_job.sh" ]; then
  for r in run_001_${TODAY}_smoke run_002_$TODAY run_003_${TODAY}_smoke run_004_$TODAY run_005_$TODAY; do make_run "$r" ok; done
  echo running >"$RR/run_005_$TODAY/STATUS"
  rm -rf "$L"/block_raise/policy/run_00[0-9]_stub
  run fetch.sh block_raise --all --top 1
  check "--all fetches every finished local run, reports the running one" eval '[ $RC != 0 ] && has "1 of 5 runs not fetched" && [ -f "$L/block_raise/policy/run_004_$TODAY/checkpoints/checkpoints.json" ] && [ -f "$L/block_raise/policy/run_004_$TODAY/job.json" ]'
fi

echo
echo "summary: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ] || { printf '  failed: %s\n' "${FAILED[@]}"; exit 1; }
