#!/bin/bash
# Local tests for train/cluster/{submit_job.sh,train.sbatch,eval.sbatch}. Nothing touches a real cluster: HOME is a
# temp dir, and sbatch/squeue/sacct/module/activate/nvidia-smi/accelerate/python are stubs on PATH (the python stub
# hands everything except the repo scripts to the real python3). Needs bash, python3, setsid; shellcheck if on PATH.
#
#   bash train/tests/cluster/run_tests.sh          (prints "cluster tests: N passed, M failed")
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN=$(cd "$HERE/../.." && pwd)
CLUSTER=$TRAIN/cluster
PY3=$(command -v python3) || { echo "python3 needed"; exit 1; }
T=$(mktemp -d "${TMPDIR:-/tmp}/cluster_tests.XXXXXX")
[ -n "${KEEP_T:-}" ] || trap 'rm -rf "$T"' EXIT
DATE=$(date +%F)
PASS=0 FAIL=0
ok() { PASS=$((PASS + 1)); }
bad() { FAIL=$((FAIL + 1)); echo "FAIL: $*"; }
check() { local d=$1; shift; if "$@"; then ok; else bad "$d"; fi; }
has() { grep -qF -- "$2" "$1"; }            # file contains literal
hasnt() { ! grep -qF -- "$2" "$1"; }
line() { grep -qxF -- "$2" "$1"; }          # file has this exact line
jq_() {  # jq_ FILE PYEXPR: prints eval(PYEXPR) with d = the JSON; a tuple prints space-separated
  "$PY3" -c "import json,sys; d=json.load(open(sys.argv[1])); r=eval(sys.argv[2]); print(' '.join(map(str, r)) if isinstance(r, tuple) else r)" "$@"; }

# ------------------------------------------------------------------ fake home + tools
H=$T/home
TOOLS=$H/CapstoneProject/train_tools
mkdir -p "$TOOLS" "$H/CapstoneProject/datasets/block_raise/ds1" "$T/bin" "$T/scratch"
cp "$CLUSTER"/submit_job.sh "$CLUSTER"/train.sbatch "$CLUSTER"/eval.sbatch "$TOOLS/"
if [ -f "$TRAIN/config.env" ]; then
  cp "$TRAIN/config.env" "$TOOLS/config.env"; echo "using train/config.env"
else
  echo "train/config.env missing: using a minimal fixture"
  cat > "$TOOLS/config.env" <<'EOF'
CLUSTER_USER="tester"
CLUSTER_ROOT="CapstoneProject"
CLUSTER_REPO="${CLUSTER_ROOT}/Capstone"
CLUSTER_DATASETS="${CLUSTER_ROOT}/datasets"
CLUSTER_RUNS="${CLUSTER_ROOT}/policies"
CLUSTER_TOOLS="${CLUSTER_ROOT}/train_tools"
CLUSTER_LOGS="${CLUSTER_ROOT}/logs"
CLUSTER_CONDA_MODULE="anaconda3/2024.06"
CLUSTER_CONDA_ENV="bp"
SLURM_PARTITION="gpu"
SLURM_GRES="gpu:h200:1"
SLURM_TIME="08:00:00"
SLURM_CPUS="16"
SLURM_MEM="64G"
SLURM_ACCOUNT=""
SMOKE_PARTITION="gpu-short"
SMOKE_TIME="02:00:00"
EVAL_PARTITION="gpu-short"
EVAL_TIME="01:00:00"
DEFAULT_EPOCHS="60"
MIN_PER_EPOCH_PER_10K_STEPS="0.55"
TIME_MARGIN_MIN="40"
TOPK="3"
LOCAL_DATA_ROOT="$TRAIN_DIR/../data"
MIN_FREE_GB="5"
EOF
fi
# expected values come from the same config
eval "$(HOME=$H TRAIN_DIR=$TOOLS bash -c 'set +u; source "$1/config.env"; for v in CLUSTER_REPO CLUSTER_RUNS CLUSTER_LOGS \
  CLUSTER_DATASETS SLURM_PARTITION SLURM_GRES SLURM_TIME SLURM_CPUS SLURM_MEM SMOKE_PARTITION SMOKE_TIME EVAL_PARTITION \
  EVAL_TIME TOPK DEFAULT_EPOCHS; do printf "%s=%q\n" "E_$v" "${!v}"; done' _ "$TOOLS")"
REPO=$H/$E_CLUSTER_REPO RUNS=$H/$E_CLUSTER_RUNS/block_raise LOGS=$H/$E_CLUSTER_LOGS
mkdir -p "$REPO/behavior_prompting/behavior_prompting/train_network" "$REPO/scripts" "$REPO/controlSW/franka_deploy/franka_deploy"
touch "$REPO/behavior_prompting/behavior_prompting/train_network/train.py" "$REPO/scripts/package_run.py" "$REPO/scripts/slim_checkpoint.py"
git -C "$REPO" init -q && git -C "$REPO" -c user.name=t -c user.email=t@t add -A &&
  git -C "$REPO" -c user.name=t -c user.email=t@t commit -q -m init
COMMIT=$(git -C "$REPO" rev-parse HEAD)
ZIP=$H/$E_CLUSTER_DATASETS/block_raise/ds1/ds1.zarr.zip
mkdir -p "$(dirname "$ZIP")"; echo zip > "$ZIP"

# ------------------------------------------------------------------ stubs
cat > "$T/bin/sbatch" <<'EOF'
#!/bin/bash
[ -f "$T/sbatch_fail" ] && { echo "sbatch: error: invalid partition" >&2; exit 1; }
n=$(cat "$T/jobid" 2>/dev/null || echo 1000); n=$((n + 1)); echo $n > "$T/jobid"
printf '%s\n' "$@" > "$T/sbatch_args_$n"; cp "$T/sbatch_args_$n" "$T/sbatch_args_last"
echo "Submitted batch job $n"
EOF
cat > "$T/bin/squeue" <<'EOF'
#!/bin/bash
cat "$T/squeue.out" 2>/dev/null; exit 0
EOF
cat > "$T/bin/sacct" <<'EOF'
#!/bin/bash
cat "$T/sacct.out" 2>/dev/null; exit 0
EOF
# Simulates a concurrent submitter: the first mkdir of a run_ folder finds it already created.
cat > "$T/bin/mkdir" <<'EOF'
#!/bin/bash
if [ -f "$T/collide" ] && [ $# -eq 1 ] && [[ $1 == */run_* ]]; then rm -f "$T/collide"; /bin/mkdir "$1"; exit 1; fi
exec /bin/mkdir "$@"
EOF
printf '#!/bin/bash\necho "module $*" >> "$T/env_log"\n' > "$T/bin/module"
printf 'echo "activate $*" >> "$T/env_log"\n' > "$T/bin/activate"
cat > "$T/bin/nvidia-smi" <<'EOF'
#!/bin/bash
case "$*" in
  *name*) echo "NVIDIA H200" ;;
  *-l*) echo "timestamp, memory.used [MiB], utilization.gpu [%]"; echo "2026/11/13 10:00:00.000, 40000 MiB, 90 %"; exec sleep 300 ;;
esac
EOF
# accelerate stub = training: writes what TrainPolicyWorkspace would write into hydra.run.dir, per $T/accel_mode
cat > "$T/bin/accelerate" <<'EOF'
#!/bin/bash
printf '%s\n' "$@" > "$T/accel_args"
cat "$WATCH_STATUS" > "$T/status_during" 2>/dev/null
for a in "$@"; do case $a in hydra.run.dir=*) RD=${a#hydra.run.dir=} ;; esac; done
mode=$(cat "$T/accel_mode")
mkdir -p "$RD/checkpoints" "$RD/.hydra" "$RD/wandb/offline-run-20261113_100000-abc"
echo cfg > "$RD/.hydra/config.yaml"; echo "- a=b" > "$RD/.hydra/overrides.yaml"; echo h > "$RD/.hydra/hydra.yaml"
echo wb > "$RD/wandb/offline-run-20261113_100000-abc/run-abc.wandb"
ln -sfn offline-run-20261113_100000-abc "$RD/wandb/latest-run"
C=$RD/checkpoints
case $mode in
  ok|timeout|slimfail)
    for e in "0000 0.0100000" "0010 0.0002000" "0020 0.0003000" "0050 0.0001480" "0059 0.0004000"; do
      set -- $e; head -c $((1000 + 10#$1)) /dev/zero > "$C/epoch=$1-val_action_mse_error_all=$2.ckpt"; done
    echo x > "$C/epoch=0010-train_loss=0.123.ckpt" ;;
  fallback) echo a > "$C/epoch=0010-val_loss=0.5000000.ckpt"; echo bb > "$C/epoch=0020-val_loss=0.4000000.ckpt" ;;
esac
case $mode in
  ok|fallback|slimfail) echo l > "$C/latest.ckpt"; exit 0 ;;
  latest) echo l > "$C/latest.ckpt"; exit 0 ;;
  fail) echo "Traceback: boom"; exit 1 ;;
  timeout) trap 'echo "accelerate got TERM"; exit 143' TERM; touch "$T/accel_started"
           for i in $(seq 1 300); do sleep 0.1; done; exit 0 ;;
esac
EOF
# python stub: repo scripts are faked, everything else goes to the real python3
cat > "$T/bin/python" <<EOF
#!/bin/bash
PY3=$PY3
EOF
cat >> "$T/bin/python" <<'EOF'
case "$1" in
  */package_run.py)
    printf '%s\n' "$@" > "$T/pkg_args"
    while [ $# -gt 0 ]; do case $1 in --out) out=$2; shift 2 ;; --status) st=$2; shift 2 ;; *) shift ;; esac; done
    /bin/mkdir -p "$out/plots" "$out/logs" "$out/run_config" "$out/videos"
    echo "# run" > "$out/README.md"; echo "{\"status\": \"$st\"}" > "$out/summary.json"; echo "epoch" > "$out/epochs.csv"
    echo "epoch" > "$out/history.csv"; : > "$out/run_config/config.yaml"
    exit "$(cat "$T/pkg_rc" 2>/dev/null || echo 0)" ;;
  */slim_checkpoint.py)
    if [ -f "$T/slim_fail" ] && [[ $2 == *"$(cat "$T/slim_fail")"* ]]; then echo "slim: corrupt" >&2; exit 1; fi
    echo "$2" >> "$T/slim_calls"; head -c "$(( $(stat -c %s "$2") / 2 + 1 ))" "$2" > "$3"; exit 0 ;;
  -c)
    case "$2" in
      *libero*) cat > /dev/null; exit 0 ;;
      *robotmq*) [ -f "$T/no_robotmq" ] && exit 1; exit 0 ;;
    esac ;;
  -m)
    if [ "$2" = franka_deploy.tools.offline_eval ]; then
      printf '%s\n' "$@" > "$T/oe_args"; echo "$PWD" > "$T/oe_cwd"; echo "${PYTHONPATH:-}" > "$T/oe_pythonpath"
      [ -f "$T/oe_sleep" ] && { trap 'exit 143' TERM; touch "$T/oe_started"; for i in $(seq 1 300); do sleep 0.1; done; }
      while [ $# -gt 0 ]; do case $1 in --out) echo '{"splits": {}}' > "$2"; shift 2 ;; *) shift ;; esac; done
      exit "$(cat "$T/oe_rc" 2>/dev/null || echo 0)"
    fi ;;
  train.py)
    printf '%s\n' "$@" > "$T/rollout_args"
    for a in "$@"; do case $a in hydra.run.dir=*) RD=${a#hydra.run.dir=} ;; esac; done
    /bin/mkdir -p "$RD/vis"; echo v > "$RD/vis/rollout_umi.mp4"; echo v > "$RD/vis/attention_and_rollout_umi_x.mp4"; exit 0 ;;
esac
exec "$PY3" "$@"
EOF
chmod +x "$T"/bin/*
export T PATH="$T/bin:$PATH"

SJ() { HOME=$H USER=tester bash "$TOOLS/submit_job.sh" "$@"; }
TRAIN_ARGS=(train --folder block_raise --label block_raise --dataset ds1)

# ================================================================== submit_job.sh train
out=$(SJ "${TRAIN_ARGS[@]}" --mode full --epochs 60 --topk 3 --note "" 2>&1); rc=$?
echo "$out" > "$T/o1"
check "train full exits 0 ($out)" [ $rc -eq 0 ]
check "RUN= line" line "$T/o1" "RUN=run_001_$DATE"
check "JOB= line" line "$T/o1" "JOB=1001"
check "only one RUN= and one JOB= line" [ "$(grep -c '^RUN=\|^JOB=' "$T/o1")" -eq 2 ]
A=$T/sbatch_args_1001
for want in "--partition=$E_SLURM_PARTITION" "--time=$E_SLURM_TIME" "--gres=$E_SLURM_GRES" "--cpus-per-task=$E_SLURM_CPUS" \
  "--mem=$E_SLURM_MEM" "--output=$LOGS/%x_%j.out" "--job-name=bp_train_block_raise" "--signal=B:USR1@600" "$TOOLS/train.sbatch"; do
  check "sbatch arg $want" line "$A" "$want"
done
check "no --account when SLURM_ACCOUNT empty" hasnt "$A" "--account"
check "no --dependency without --after" hasnt "$A" "--dependency"
check "no --parsable reliance" hasnt "$A" "--parsable"
check "positional args after script" [ "$(sed -n "/train.sbatch/,\$p" "$A" | tail -n +2 | tr '\n' ' ')" = "$RUNS/run_001_$DATE block_raise ds1 run_001_$DATE full 60 3  " ]
check "config frozen into the run dir" cmp -s <(sed 1d "$RUNS/run_001_$DATE/run_config/submit_config.env") "$TOOLS/config.env"
check "logs dir created" [ -d "$LOGS" ]
M=$RUNS/run_001_$DATE/train_meta.json
check "train_meta.json valid with job id" [ "$(jq_ "$M" "d['job_id'], d['run'], d['mode'], d['epochs'], d['topk'], d['clone_git_commit']")" = "1001 run_001_$DATE full 60 3 $COMMIT" ]
check "train_meta keys" [ "$(jq_ "$M" "sorted(k for k in ['folder','label','dataset','run','mode','epochs','topk','note','overrides','submitted','clone_git_commit'] if k not in d)")" = "[]" ]
check "STATUS queued" [ "$(cat "$RUNS/run_001_$DATE/STATUS")" = queued ]

NOTE='it'"'"'s a "quoted" note; $HOME `x` \ back'
SJ "${TRAIN_ARGS[@]}" --mode full --epochs 5 --topk 2 --note "$NOTE" -- training.lr=1 'task.name=a b' > "$T/o2" 2>&1
check "second run increments" line "$T/o2" "RUN=run_002_$DATE"
A=$T/sbatch_args_1002
check "note passed to sbatch verbatim" line "$A" "$NOTE"
check "overrides last" [ "$(tail -n 2 "$A" | tr '\n' '|')" = "training.lr=1|task.name=a b|" ]
check "note in train_meta verbatim" [ "$(jq_ "$RUNS/run_002_$DATE/train_meta.json" "d['note']")" = "$NOTE" ]
check "overrides in train_meta" [ "$(jq_ "$RUNS/run_002_$DATE/train_meta.json" "'|'.join(d['overrides'])")" = "training.lr=1|task.name=a b" ]

SJ "${TRAIN_ARGS[@]}" --mode smoke --epochs 3 --topk 3 --note smoke --after 1001 > "$T/o3" 2>&1
check "smoke suffix" line "$T/o3" "RUN=run_003_${DATE}_smoke"
A=$T/sbatch_args_1003
check "smoke partition" line "$A" "--partition=$E_SMOKE_PARTITION"
check "smoke time" line "$A" "--time=$E_SMOKE_TIME"
check "smoke gres same" line "$A" "--gres=$E_SLURM_GRES"
check "--after -> --dependency" line "$A" "--dependency=afterok:1001"
check "smoke mode positional" has "$A" "smoke"
check "train_meta after" [ "$(jq_ "$RUNS/run_003_${DATE}_smoke/train_meta.json" "d['after']")" = 1001 ]

touch "$T/collide"
SJ "${TRAIN_ARGS[@]}" --mode full --epochs 5 --topk 1 --note c > "$T/o4" 2>&1
check "collision retry skips the taken name" line "$T/o4" "RUN=run_005_$DATE"
check "collision message" has "$T/o4" "run_004_$DATE was just taken"
check "taken folder left alone (no meta)" [ ! -f "$RUNS/run_004_$DATE/train_meta.json" ]

/bin/mkdir -p "$RUNS/run_008_2026-01-01"
SJ "${TRAIN_ARGS[@]}" --mode full --epochs 5 --topk 1 --note c > "$T/o5" 2>&1
check "index after 008 is 009 (no octal)" line "$T/o5" "RUN=run_009_$DATE"
SJ "${TRAIN_ARGS[@]}" --mode full --note c > "$T/o5b" 2>&1
check "defaults epochs/topk from config" [ "$(jq_ "$RUNS/run_010_$DATE/train_meta.json" "d['epochs'], d['topk']")" = "$E_DEFAULT_EPOCHS $E_TOPK" ]

echo 'TOPK=""' > "$TOOLS/config.local.env"
SJ "${TRAIN_ARGS[@]}" --mode full --epochs 5 --note k > "$T/o5c" 2>&1
check "TOPK unset -> one checkpoint (topk 1)" [ "$(jq_ "$RUNS/run_011_$DATE/train_meta.json" "d['topk']")" = 1 ] && line "$T/sbatch_args_last" 1
echo 'SLURM_ACCOUNT="acct1"' > "$TOOLS/config.local.env"
SJ "${TRAIN_ARGS[@]}" --mode full --epochs 5 --topk 1 --note a > "$T/o6" 2>&1
check "account from config.local.env" line "$T/sbatch_args_last" "--account=acct1"
echo 'CLUSTER_RUNS="<runs dir>"' > "$TOOLS/config.local.env"
SJ "${TRAIN_ARGS[@]}" --mode full --epochs 5 --topk 1 --note a > "$T/o7" 2>&1; rc=$?
check "placeholder refused, names the variable" [ $rc -ne 0 ] && has "$T/o7" "CLUSTER_RUNS"
rm -f "$TOOLS/config.local.env"

n_before=$(ls "$RUNS" | wc -l); j_before=$(cat "$T/jobid")
SJ train --folder f --label block_raise --dataset nope --mode full --epochs 5 --topk 1 --note a > "$T/o8" 2>&1; rc=$?
check "missing dataset refused" [ $rc -ne 0 ] && has "$T/o8" "dataset not found"
SJ "${TRAIN_ARGS[@]}" --mode full --epochs 5 --topk 1 --note a -- hydra.run.dir=/x > "$T/o9" 2>&1; rc=$?
check "hydra.run.dir override refused" [ $rc -ne 0 ] && has "$T/o9" "not allowed"
SJ "${TRAIN_ARGS[@]}" --mode full --epochs 5 --topk 1 --note a -- ++logging.dir=/x > "$T/o9b" 2>&1; rc=$?
check "++logging.dir override refused" [ $rc -ne 0 ]
SJ "${TRAIN_ARGS[@]}" --mode full --note a -- checkpoint.topk.k=5 > "$T/o9e" 2>&1; rc=$?
check "checkpoint.topk.k override refused (use --topk)" [ $rc -ne 0 ] && has "$T/o9e" "use --topk"
SJ "${TRAIN_ARGS[@]}" --mode bogus --note a > "$T/o9c" 2>&1; rc=$?
check "bad mode refused" [ $rc -ne 0 ]
SJ "${TRAIN_ARGS[@]}" --mode full --after 12x --note a > "$T/o9d" 2>&1; rc=$?
check "bad --after refused" [ $rc -ne 0 ]
touch "$T/sbatch_fail"
SJ "${TRAIN_ARGS[@]}" --mode full --epochs 5 --topk 1 --note a > "$T/o10" 2>&1; rc=$?
rm -f "$T/sbatch_fail"
check "sbatch failure -> nonzero, no RUN= line" [ $rc -ne 0 ] && hasnt "$T/o10" "RUN="
check "no run folder or job left behind by refusals/failures" [ "$(ls "$RUNS" | wc -l)" -eq "$n_before" ] && [ "$(cat "$T/jobid")" = "$j_before" ]

# ================================================================== submit_job.sh eval
SJ eval --label block_raise --run run_001_$DATE --dataset ds1 --ckpt top1 > "$T/e1" 2>&1; rc=$?
check "eval without packaged ckpt refused" [ $rc -ne 0 ] && has "$T/e1" "no checkpoints/top1_"
/bin/mkdir -p "$RUNS/run_001_$DATE/checkpoints"; echo c > "$RUNS/run_001_$DATE/checkpoints/top1_epoch=0050-val_action_mse_error_all=0.0001480.ckpt"
SJ eval --label block_raise --run run_001_$DATE --dataset ds1 --ckpt top1 --per-demo 5 --attention > "$T/e2" 2>&1; rc=$?
J=$(cat "$T/jobid")
check "eval submitted" [ $rc -eq 0 ] && line "$T/e2" "JOB=$J"
A=$T/sbatch_args_$J
for want in "--partition=$E_EVAL_PARTITION" "--time=$E_EVAL_TIME" "--job-name=bp_eval_block_raise" "--output=$LOGS/%x_%j.out" "$TOOLS/eval.sbatch"; do
  check "eval sbatch arg $want" line "$A" "$want"
done
check "eval positional args" [ "$(sed -n "/eval.sbatch/,\$p" "$A" | tail -n +2 | tr '\n' ' ')" = "$RUNS/run_001_$DATE/eval/top1 block_raise run_001_$DATE ds1 top1 5 1 " ]
check "eval config frozen" [ -f "$RUNS/run_001_$DATE/eval/top1/submit_config.env" ]
mv "$REPO/controlSW" "$T/controlSW.bak"
SJ eval --label block_raise --run run_001_$DATE --dataset ds1 --ckpt top1 > "$T/e3" 2>&1; rc=$?
mv "$T/controlSW.bak" "$REPO/controlSW"
check "old clone message" [ $rc -ne 0 ] && has "$T/e3" "clone is older than 4be9982"
SJ eval --label block_raise --run run_001_$DATE --dataset ds1 --ckpt best > "$T/e4" 2>&1; rc=$?
check "bad --ckpt refused" [ $rc -ne 0 ]

# ================================================================== submit_job.sh status
R=$RUNS/run_001_$DATE
printf 'step 1\n  5%%|#   | 10/200 [00:01<00:10, 9.1it/s]\rEpoch 3:  50%%|#####  | 1/2 [00:01<00:01, 1.2s/it]\nval loss 0.1\n\n== Training finished with status: ok\n' > "$LOGS/bp_train_block_raise_1001.out"
st() { : > "$T/squeue.out"; : > "$T/sacct.out"; [ -n "$2" ] && echo "$2" > "$T/squeue.out"; [ -n "$3" ] && echo "$3" > "$T/sacct.out"
  echo "$1" > "$R/STATUS"; SJ status --label block_raise --run run_001_$DATE > "$T/s" 2>&1; }
st queued "PENDING 0:00 (Priority)" "PENDING|0:0|00:00:00|Unknown|Unknown|gpu|None assigned"
check "status queued" line "$T/s" "STATUS=queued"
check "status JOB" line "$T/s" "JOB=1001"
check "status shows queue state" has "$T/s" "PENDING 0:00 (Priority)"
st queued "RUNNING 1:00 d1" "RUNNING|0:0|00:01:00|2026-11-13T10:00|Unknown|gpu|d1"
check "status queued file + RUNNING in slurm -> running" line "$T/s" "STATUS=running"
st running "RUNNING 1:00 d1" "RUNNING|0:0|00:01:00|2026-11-13T10:00|Unknown|gpu|d1"
check "status running" line "$T/s" "STATUS=running"
check "log tail keeps text" has "$T/s" "val loss 0.1"
check "log tail strips progress bars" hasnt "$T/s" "it/s]" && hasnt "$T/s" "s/it]"
check "checkpoint listed with size" has "$T/s" "top1_epoch=0050-val_action_mse_error_all=0.0001480.ckpt  2 bytes"
st ok "" "COMPLETED|0:0|04:00:00|a|b|gpu|d1"
check "status ok" line "$T/s" "STATUS=ok"
st failed "" "FAILED|1:0|00:10:00|a|b|gpu|d1"
check "status failed" line "$T/s" "STATUS=failed"
st timeout "" "COMPLETED|0:0|07:50:00|a|b|gpu|d1"
check "status timeout" line "$T/s" "STATUS=timeout"
st running "" "CANCELLED by 123|0:0|00:10:00|a|b|gpu|d1"
check "dead job with STATUS=running -> failed" line "$T/s" "STATUS=failed" && has "$T/s" "CANCELLED"
st running "" "TIMEOUT|0:0|08:00:00|a|b|gpu|d1"
check "hard timeout without packaging -> timeout" line "$T/s" "STATUS=timeout"
st queued "PENDING 0:00 (DependencyNeverSatisfied)" "PENDING|0:0|00:00:00|Unknown|Unknown|gpu|None"
check "never-satisfied dependency -> failed + scancel hint" line "$T/s" "STATUS=failed" && has "$T/s" "scancel 1001"
SJ status --label block_raise --run run_999 > "$T/s2" 2>&1; rc=$?
check "status of unknown run fails" [ $rc -ne 0 ]
: > "$T/squeue.out"; : > "$T/sacct.out"

# ================================================================== train.sbatch
# run_sbatch <run> <mode> <epochs> <topk> <note> [overrides...]; runs in the background, pid in SB_PID
start_train() {
  local run=$1; shift
  /bin/mkdir -p "$RUNS/$run/run_config"; echo queued > "$RUNS/$run/STATUS"
  [ -f "$RUNS/$run/run_config/submit_config.env" ] || cp "$TOOLS/config.env" "$RUNS/$run/run_config/submit_config.env"
  rm -f "$T/accel_args" "$T/pkg_args" "$T/slim_calls" "$T/status_during" "$T/accel_started"
  HOME=$H USER=tester BP_SCRATCH=$T/scratch SLURM_JOB_ID=$JID SLURM_JOB_NAME=bp_train_block_raise SLURMD_NODENAME=d1 \
    SLURM_JOB_PARTITION=gpu WATCH_STATUS=$RUNS/$run/STATUS \
    bash "$TOOLS/train.sbatch" "$RUNS/$run" block_raise ds1 "$run" "$@" > "$T/train_out_$run" 2>&1 &
  SB_PID=$!
}
run_train() { start_train "$@"; wait $SB_PID; SB_RC=$?; }

# ---- ok: top-3 by val_action_mse, scratch deleted
JID=2001; echo ok > "$T/accel_mode"
run_train ra full 60 3 "$NOTE" training.num_epochs=7 'task.x=a b'
F=$RUNS/ra; RD=$T/scratch/bp_runs/block_raise/ra; O=$T/train_out_ra
check "ok: exit 0 ($(tail -3 "$O"))" [ $SB_RC -eq 0 ]
check "ok: STATUS running during training" [ "$(cat "$T/status_during" 2>/dev/null)" = running ]
check "ok: final STATUS ok" [ "$(cat "$F/STATUS")" = ok ]
check "ok: env setup" has "$T/env_log" "module load anaconda3/2024.06" && has "$T/env_log" "activate bp"
for want in "--num_processes=1" "train.py" "--config-name=umi_policy_dunetp" "task.dataset.dataset_path=$ZIP" \
  "exp_name=ra" "group_tag=block_raise" "logging.mode=offline" "+logging.dir=$RD" "hydra.run.dir=$RD" \
  "checkpoint.topk.k=3" "checkpoint.save_train_loss_ckpt_every=null" "training.num_epochs=60"; do
  check "ok: accelerate arg $want" line "$T/accel_args" "$want"
done
check "ok: user overrides last" [ "$(tail -n 2 "$T/accel_args" | tr '\n' '|')" = "training.num_epochs=7|task.x=a b|" ]
check "ok: full mode has no smoke keys" hasnt "$T/accel_args" "training.val_every=1"
check "ok: package_run args" line "$T/pkg_args" "--no-checkpoint" && line "$T/pkg_args" "--note=$NOTE" && line "$T/pkg_args" "$F" &&
  line "$T/pkg_args" "$LOGS/bp_train_block_raise_2001.out" && line "$T/pkg_args" "$LOGS/bp_train_block_raise_2001_gpu.csv"
check "ok: --status ok to package_run" [ "$(grep -A1 -x -- --status "$T/pkg_args" | tail -1)" = ok ]
CJ=$F/checkpoints/checkpoints.json
check "ok: top-3 files" [ -f "$F/checkpoints/top1_epoch=0050-val_action_mse_error_all=0.0001480.ckpt" ] &&
  [ -f "$F/checkpoints/top2_epoch=0010-val_action_mse_error_all=0.0002000.ckpt" ] &&
  [ -f "$F/checkpoints/top3_epoch=0020-val_action_mse_error_all=0.0003000.ckpt" ] && [ "$(ls "$F/checkpoints" | wc -l)" -eq 4 ]
check "ok: checkpoints.json ranks/epochs/mse" [ "$(jq_ "$CJ" "[(e['rank'], e['epoch'], e['val_action_mse'], e['source']) for e in d]")" = \
  "[(1, 50, 0.000148, 'topk'), (2, 10, 0.0002, 'topk'), (3, 20, 0.0003, 'topk')]" ]
check "ok: checkpoints.json sizes match files" [ "$(jq_ "$CJ" "all(e['size_bytes'] == __import__('os').path.getsize('$F/checkpoints/' + e['file']) for e in d)")" = True ]
check "ok: train_loss ckpt ignored" [ "$(grep -c train_loss "$T/slim_calls")" -eq 0 ]
check "ok: wandb offline run copied" [ -f "$F/wandb/offline-run-20261113_100000-abc/run-abc.wandb" ] && [ ! -e "$F/wandb/latest-run" ]
check "ok: .hydra copied" [ -f "$F/run_config/hydra/hydra.yaml" ] && [ -f "$F/run_config/hydra/overrides.yaml" ]
check "ok: scratch deleted" [ ! -e "$RD" ]
check "ok: no checkpoints_on_scratch.txt" [ ! -e "$F/logs/checkpoints_on_scratch.txt" ]
check "ok: GPU logger stopped" [ -z "$(pgrep -f "^sleep 300$" -u "$(id -u)")" ]

# ---- smoke keys + job_info contents (keep scratch by failing package_run)
JID=2002; echo ok > "$T/accel_mode"; echo 3 > "$T/pkg_rc"
run_train rb smoke 3 1 "smoke note"
rm -f "$T/pkg_rc"
F=$RUNS/rb; RD=$T/scratch/bp_runs/block_raise/rb
check "smoke: val/rollout/sample every 1" line "$T/accel_args" "training.val_every=1" && line "$T/accel_args" "training.rollout_every=1" &&
  line "$T/accel_args" "training.sample_every=1" && line "$T/accel_args" "training.num_epochs=3"
check "smoke: topk=1 -> one file" [ "$(jq_ "$F/checkpoints/checkpoints.json" "len(d)")" = 1 ]
check "package_run failure: STATUS still ok, exit nonzero" [ "$(cat "$F/STATUS")" = ok ] && [ $SB_RC -ne 0 ]
check "package_run failure: scratch kept + recorded" [ -d "$RD" ] && line "$F/logs/checkpoints_on_scratch.txt" "$RD"
check "job_info.json" [ "$(jq_ "$RD/job_info.json" "d['job_id'], d['task'], d['mode'], d['gpu_name'], d['epochs']")" = "2002 block_raise smoke NVIDIA H200 3" ]

# ---- failed training: STATUS failed, scratch kept
JID=2003; echo fail > "$T/accel_mode"
run_train rc full 60 3 "x"
F=$RUNS/rc; RD=$T/scratch/bp_runs/block_raise/rc
check "fail: STATUS failed, exit nonzero" [ "$(cat "$F/STATUS")" = failed ] && [ $SB_RC -ne 0 ]
check "fail: package_run --status failed" [ "$(grep -A1 -x -- --status "$T/pkg_args" | tail -1)" = failed ]
check "fail: scratch kept + recorded" [ -d "$RD" ] && line "$F/logs/checkpoints_on_scratch.txt" "$RD"
check "fail: empty checkpoints.json" [ "$(jq_ "$F/checkpoints/checkpoints.json" "d")" = "[]" ]

# ---- fallback monitor (val_loss names)
JID=2004; echo fallback > "$T/accel_mode"
run_train rd full 60 3 "x"
F=$RUNS/rd
check "fallback: ranked by val_loss" [ "$(jq_ "$F/checkpoints/checkpoints.json" "[(e['rank'], e['epoch'], e['metric'], e['value'], e['val_action_mse']) for e in d]")" = \
  "[(1, 20, 'val_loss', 0.4, None), (2, 10, 'val_loss', 0.5, None)]" ]
check "fallback: log line" has "$T/train_out_rd" "ranking the top-K files by val_loss"
check "fallback: ok -> scratch deleted" [ ! -e "$T/scratch/bp_runs/block_raise/rd" ] && [ $SB_RC -eq 0 ]

# ---- no top-K files: latest.ckpt as top1
JID=2005; echo latest > "$T/accel_mode"
run_train re full 60 3 "x"
F=$RUNS/re
check "latest: top1_latest.ckpt" [ -f "$F/checkpoints/top1_latest.ckpt" ]
check "latest: checkpoints.json source latest" [ "$(jq_ "$F/checkpoints/checkpoints.json" "[(e['rank'], e['file'], e['epoch'], e['source']) for e in d]")" = \
  "[(1, 'top1_latest.ckpt', None, 'latest')]" ]

# ---- slim failure: STATUS ok but scratch kept, failed file not listed
JID=2006; echo slimfail > "$T/accel_mode"; echo "0.0002000" > "$T/slim_fail"
run_train rf full 60 3 "x"
rm -f "$T/slim_fail"
F=$RUNS/rf; RD=$T/scratch/bp_runs/block_raise/rf
check "slimfail: ranks 1 and 3 listed" [ "$(jq_ "$F/checkpoints/checkpoints.json" "[e['rank'] for e in d]")" = "[1, 3]" ] && [ ! -e "$F/checkpoints/top2_epoch=0010-val_action_mse_error_all=0.0002000.ckpt" ]
check "slimfail: scratch kept + recorded" [ -d "$RD" ] && line "$F/logs/checkpoints_on_scratch.txt" "$RD" && [ $SB_RC -ne 0 ]

# ---- timeout: USR1 stops training, packaging still runs, STATUS timeout
JID=2007; echo timeout > "$T/accel_mode"
start_train rg full 60 3 "x"
for i in $(seq 1 100); do [ -f "$T/accel_started" ] && break; sleep 0.1; done
check "timeout: training started" [ -f "$T/accel_started" ]
kill -USR1 $SB_PID
wait $SB_PID; SB_RC=$?
F=$RUNS/rg; RD=$T/scratch/bp_runs/block_raise/rg
check "timeout: STATUS timeout" [ "$(cat "$F/STATUS")" = timeout ] && [ $SB_RC -ne 0 ]
check "timeout: handler message + training got TERM" has "$T/train_out_rg" "Time limit approaching" && has "$T/train_out_rg" "accelerate got TERM"
check "timeout: package_run --status timeout" [ "$(grep -A1 -x -- --status "$T/pkg_args" | tail -1)" = timeout ]
check "timeout: checkpoints still packaged" [ "$(jq_ "$F/checkpoints/checkpoints.json" "len(d)")" = 3 ]
check "timeout: scratch kept" [ -d "$RD" ] && [ -f "$F/logs/checkpoints_on_scratch.txt" ]

# ---- config frozen at submit: re-syncing the tools config later must not move a queued job
JID=2008; echo ok > "$T/accel_mode"; : > "$T/env_log"
SJ "${TRAIN_ARGS[@]}" --mode full --epochs 4 --topk 1 --note "-starts with a dash" > "$T/o_fz" 2>&1
FZ=$(sed -n 's/^RUN=//p' "$T/o_fz"); FA=$(cat "$T/sbatch_args_last")
cp "$TOOLS/config.env" "$T/config.env.bak"
printf 'CLUSTER_RUNS="elsewhere"\nCLUSTER_CONDA_ENV="wrong"\nCLUSTER_REPO="nowhere"\n' >> "$TOOLS/config.env"
mapfile -t JA < <(sed -n "/train.sbatch/,\$p" <<< "$FA" | tail -n +2)
HOME=$H USER=tester BP_SCRATCH=$T/scratch SLURM_JOB_ID=$JID SLURM_JOB_NAME=bp_train_block_raise WATCH_STATUS=/dev/null \
  bash "$TOOLS/train.sbatch" "${JA[@]}" > "$T/train_out_fz" 2>&1; rc=$?
cp "$T/config.env.bak" "$TOOLS/config.env"
check "frozen config: results in the submitted run dir ($(tail -2 "$T/train_out_fz"))" [ $rc -eq 0 ] && [ "$(cat "$RUNS/$FZ/STATUS")" = ok ] && [ ! -e "$H/elsewhere" ]
check "frozen config: submitted conda env used" line "$T/env_log" "activate bp"
check "note starting with '-' reaches package_run" line "$T/pkg_args" "--note=-starts with a dash"
HOME=$H bash "$TOOLS/train.sbatch" "$RUNS/$FZ" block_raise ds1 "$FZ" full 4 1 > "$T/o_argc" 2>&1; rc=$?
check "train.sbatch refuses a short argument list" [ $rc -ne 0 ] && has "$T/o_argc" "needs 8 arguments"

# ================================================================== eval.sbatch
R=$RUNS/run_001_$DATE
printf '{"overrides": ["model.x=1", "task.y=a b"]}\n' > "$R/train_meta.json"
run_eval() {  # <tag> <per-demo> <attention>
  rm -f "$T/oe_args" "$T/rollout_args" "$T/oe_started"
  /bin/mkdir -p "$R/eval/$1"; cp "$TOOLS/config.env" "$R/eval/$1/submit_config.env"
  HOME=$H USER=tester BP_SCRATCH=$T/scratch SLURM_JOB_ID=$JID SLURM_JOB_NAME=bp_eval_block_raise SLURMD_NODENAME=d1 \
    bash "$TOOLS/eval.sbatch" "$R/eval/$1" block_raise run_001_$DATE ds1 "$@" > "$T/eval_out" 2>&1 &
  SB_PID=$!
}
CK=$R/checkpoints/top1_epoch=0050-val_action_mse_error_all=0.0001480.ckpt
JID=3001; run_eval top1 5 1; wait $SB_PID; rc=$?
E=$R/eval/top1; W=$T/scratch/bp_eval/block_raise/run_001_$DATE/top1_3001
check "eval ok ($(tail -3 "$T/eval_out"))" [ $rc -eq 0 ] && [ "$(jq_ "$E/eval_job.json" "d['status'], d['offline_eval_rc'], d['attention_rc'], d['attention']")" = "ok 0 0 True" ]
check "eval.json written" [ -s "$E/eval.json" ]
check "offline_eval run in franka_deploy with PYTHONPATH" [ "$(cat "$T/oe_cwd")" = "$REPO/controlSW/franka_deploy" ] &&
  [ "$(cut -d: -f1 "$T/oe_pythonpath")" = "$REPO/controlSW/franka_deploy" ]
for want in "--ckpt" "$CK" "--work-dir" "$W/offline_eval" "--dataset-path" "$ZIP" "--per-demo" "5" "--out" "$E/eval.json"; do
  check "offline_eval arg $want" line "$T/oe_args" "$want"
done
for want in "--config-name=umi_policy_dunetp" "+rollout=umi" "rollout.checkpoint_path=$CK" "task.dataset.dataset_path=$ZIP" \
  "hydra.run.dir=$W/rollout" "+logging.dir=$W/rollout" "logging.mode=offline" "model.x=1" "task.y=a b"; do
  check "rollout arg $want" line "$T/rollout_args" "$want"
done
check "rollout: training overrides last" [ "$(tail -n 2 "$T/rollout_args" | tr '\n' '|')" = "model.x=1|task.y=a b|" ]
check "videos copied" [ -f "$E/videos/rollout_umi.mp4" ] && [ -f "$E/videos/attention_and_rollout_umi_x.mp4" ]
check "eval work dir deleted on success" [ ! -e "$W" ]

JID=3002; run_eval top1 0 0; wait $SB_PID; rc=$?
check "eval without attention/per-demo" [ $rc -eq 0 ] && [ ! -f "$T/rollout_args" ] && hasnt "$T/oe_args" "--per-demo"

JID=3003; echo 1 > "$T/oe_rc"; run_eval top1 0 0; wait $SB_PID; rc=$?; rm -f "$T/oe_rc"
check "eval failure -> failed + work dir kept" [ $rc -ne 0 ] && [ "$(jq_ "$R/eval/top1/eval_job.json" "d['status'], d['work_dir_kept'] is not None")" = "failed True" ]

JID=3004; touch "$T/no_robotmq"; run_eval top1 0 0; wait $SB_PID; rc=$?; rm -f "$T/no_robotmq"
check "missing robotmq -> clear failure" [ $rc -ne 0 ] && has "$R/eval/top1/eval_job.json" "robotmq" && [ ! -f "$T/oe_args" ]

mv "$REPO/controlSW" "$T/controlSW.bak"
JID=3005; run_eval top1 0 0; wait $SB_PID; rc=$?
mv "$T/controlSW.bak" "$REPO/controlSW"
check "eval on old clone -> pull message" [ $rc -ne 0 ] && has "$R/eval/top1/eval_job.json" "clone is older than 4be9982"

JID=3006; touch "$T/oe_sleep"; run_eval top1 0 1
for i in $(seq 1 100); do [ -f "$T/oe_started" ] && break; sleep 0.1; done
kill -USR1 $SB_PID; wait $SB_PID; rc=$?; rm -f "$T/oe_sleep"
check "eval timeout -> status timeout, rollout skipped" [ $rc -ne 0 ] && [ "$(jq_ "$R/eval/top1/eval_job.json" "d['status']")" = timeout ] && [ ! -f "$T/rollout_args" ]

# ================================================================== static checks
for f in "$CLUSTER"/submit_job.sh "$CLUSTER"/train.sbatch "$CLUSTER"/eval.sbatch "$HERE/run_tests.sh"; do
  check "bash -n ${f##*/}" bash -n "$f"
done
if command -v shellcheck >/dev/null; then
  for f in "$CLUSTER"/submit_job.sh "$CLUSTER"/train.sbatch "$CLUSTER"/eval.sbatch; do
    check "shellcheck ${f##*/}" shellcheck -S warning "$f"
  done
else
  echo "(shellcheck not on PATH: skipped)"
fi

echo "cluster tests: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
