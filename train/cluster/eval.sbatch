#!/bin/bash
# Safe defaults only: submit_job.sh passes every resource on the sbatch command line (EVAL_* from config.env).
#SBATCH --job-name=bp_eval
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@300
# Offline evaluation of one packaged checkpoint of a run, plus (--attention) the rollout workspace's attention videos.
#
# Usage (normally via submit_job.sh eval):
#   sbatch ... eval.sbatch <eval dir> <label> <run> <dataset> <topN> <per-demo N> <attention 0|1>
# <eval dir> = FINAL/eval/<topN>; config comes only from <eval dir>/submit_config.env, frozen by submit_job.sh.
#
# Writes FINAL/eval/<topN>/ (FINAL = $HOME/CLUSTER_RUNS/<label>/<run>):
#   eval.json      franka_deploy.tools.offline_eval: the training val/action_mse metric re-computed, per-step errors in
#                  mm/deg, and with --per-demo N per-episode errors
#   videos/*.mp4   (--attention) rollout_*.mp4, attention_and_rollout_*.mp4 from RolloutPolicyWorkspace
#   eval_job.json  status (running -> ok|failed|timeout), job id, inputs, exit codes, work dir if kept
#
# Imports: behavior_prompting is installed (editable) in the bp conda env, the same way train.sbatch uses it.
# controlSW/franka_deploy has no pyproject/setup.py, so it is NOT installed: `python -m franka_deploy...` from inside
# controlSW/franka_deploy works because -m puts the cwd on sys.path; PYTHONPATH is set too so it does not depend on the
# cwd. franka_deploy refuses work dirs inside the repo clone (franka_deploy/paths.py), so its work dir is on scratch.
# franka_deploy's inference server imports behavior_prompting.deployment.detached_inference, which imports robotmq
# (requirements-inference.txt); the job checks that import first and fails with a clear message if bp lacks it.
set -o pipefail
[ $# -ge 7 ] || { echo "ERROR: eval.sbatch needs 7 arguments, got $#" >&2; exit 1; }
EVAL_DIR=${1:?eval dir}
CONF=$EVAL_DIR/submit_config.env
# shellcheck disable=SC2034  # read by config.env (LOCAL_DATA_ROOT default)
TRAIN_DIR=$EVAL_DIR
# shellcheck source=/dev/null
source "$CONF" || { echo "ERROR: cannot read $CONF" >&2; exit 1; }
# shellcheck disable=SC2088  # a literal '~/' prefix in a config value, not a tilde to expand
abs() { case $1 in /*) echo "$1" ;; '~/'*) echo "$HOME/${1#\~/}" ;; *) echo "$HOME/$1" ;; esac; }
# Positional args are read AFTER config.env, so a config key with the same name (TOPK!) cannot clobber them.
LABEL=${2:?label}
RUN_NAME=${3:?run}
DATASET=${4:?dataset}
TAG=${5:?ckpt tag topN}
PER_DEMO=${6:-0}
ATTENTION=${7:-0}

REPO=$(abs "$CLUSTER_REPO")
FD=$REPO/controlSW/franka_deploy
DATASET_ZIP=$(abs "$CLUSTER_DATASETS")/$LABEL/$DATASET/$DATASET.zarr.zip
FINAL=$(dirname "$(dirname "$EVAL_DIR")")
SCRATCH=${BP_SCRATCH:-/scratch/$USER}  # BP_SCRATCH exists only for the local tests
WORK=$SCRATCH/bp_eval/$LABEL/$RUN_NAME/${TAG}_${SLURM_JOB_ID}
mkdir -p "$EVAL_DIR" "$WORK"

CKPTS=("$FINAL"/checkpoints/"${TAG}"_*.ckpt)
CKPT=${CKPTS[0]}
STARTED=$(date -Iseconds)
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
EVAL_RC='' ATTN_RC='' MSG=''

write_job_json() {  # $1 status
  local kept=
  [ -d "$WORK" ] && kept=$WORK
  {
    printf '{\n  "status": "%s",\n  "job_id": "%s",\n  "label": "%s",\n  "run": "%s",\n  "dataset": "%s",\n' \
      "$1" "$SLURM_JOB_ID" "$LABEL" "$RUN_NAME" "$DATASET"
    printf '  "ckpt_tag": "%s",\n  "ckpt": "%s",\n  "dataset_zip": "%s",\n  "per_demo": %s,\n  "attention": %s,\n' \
      "$TAG" "${CKPT##*/}" "$DATASET_ZIP" "$PER_DEMO" "$([ "$ATTENTION" = 1 ] && echo true || echo false)"
    printf '  "node": "%s",\n  "clone_git_commit": "%s",\n  "started": "%s",\n  "finished": %s,\n' \
      "${SLURMD_NODENAME-}" "$COMMIT" "$STARTED" "$([ "$1" = running ] && echo null || echo "\"$(date -Iseconds)\"")"
    printf '  "offline_eval_rc": %s,\n  "attention_rc": %s,\n  "work_dir_kept": %s,\n  "message": "%s"\n}\n' \
      "${EVAL_RC:-null}" "${ATTN_RC:-null}" "$([ -n "$kept" ] && echo "\"$kept\"" || echo null)" "${MSG//\"/\'}"
  } > "$EVAL_DIR/eval_job.json"
}
fail() { MSG=$*; echo "ERROR: $*" >&2; write_job_json failed; exit 1; }

write_job_json running
echo "== eval $LABEL/$RUN_NAME/$TAG job $SLURM_JOB_ID on ${SLURMD_NODENAME:-?}: ${CKPT##*/}"
[ -d "$FD/franka_deploy" ] || fail "controlSW/franka_deploy missing in the clone: git -C ~/CapstoneProject/Capstone pull (clone is older than 4be9982)"
[ -f "$CKPT" ] || fail "no checkpoints/${TAG}_*.ckpt in $FINAL"
[ -f "$DATASET_ZIP" ] || fail "dataset not found: $DATASET_ZIP"

module load "$CLUSTER_CONDA_MODULE"
# shellcheck source=/dev/null
source activate "$CLUSTER_CONDA_ENV"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy
echo n | python -c "from libero.libero import get_libero_path"
python -c "import robotmq" 2>/dev/null ||
  fail "the $CLUSTER_CONDA_ENV env has no robotmq (needed by behavior_prompting.deployment.detached_inference, which franka_deploy's offline_eval loads); install it into $CLUSTER_CONDA_ENV: pip install robotmq==0.1.14"

# ---- each step in its own process group, so the USR1 handler can stop it and still write eval_job.json ----
TIMED_OUT=0 STEP_PID=
on_timeout() {
  echo "== Time limit approaching: stopping the current step"
  TIMED_OUT=1
  [ -n "$STEP_PID" ] && kill -TERM -- -"$STEP_PID" 2>/dev/null
}
trap on_timeout USR1
run_step() {  # runs "$@" in a new session; returns its exit code
  setsid "$@" &
  STEP_PID=$!
  wait $STEP_PID
  local rc=$?
  [ $TIMED_OUT = 1 ] && wait $STEP_PID 2>/dev/null
  STEP_PID=
  return $rc
}

PER_DEMO_ARGS=()
[ "$PER_DEMO" -gt 0 ] 2>/dev/null && PER_DEMO_ARGS=(--per-demo "$PER_DEMO")
cd "$FD" || fail "cannot cd to $FD"
echo "== offline_eval"
PYTHONPATH=$FD${PYTHONPATH:+:$PYTHONPATH} run_step python -m franka_deploy.tools.offline_eval \
  --ckpt "$CKPT" --work-dir "$WORK/offline_eval" --dataset-path "$DATASET_ZIP" \
  ${PER_DEMO_ARGS[@]+"${PER_DEMO_ARGS[@]}"} --out "$EVAL_DIR/eval.json"
EVAL_RC=$?

if [ "$ATTENTION" = 1 ] && [ $TIMED_OUT = 0 ]; then
  # RolloutPolicyWorkspace builds the model from THIS config, not the checkpoint's (see its TODO), so the training
  # run's user overrides (train_meta.json "overrides") are replayed here. config/rollout/umi.yaml -> rollout/base.yaml
  # (@package _global_) sets _target_=RolloutPolicyWorkspace and rollout.checkpoint_path; the UMI env runner
  # (task umi_prompt, vis_attention_map: true) replays train_rollout_max_tasks=6 dataset tasks and writes
  # <hydra run dir>/vis/{rollout_*,attention_and_rollout_*}.mp4. Slim checkpoints keep state_dicts.model only, which is
  # all load_checkpoint(exclude_keys=['optimizer']) needs.
  mapfile -t TRAIN_OVERRIDES < <(python -c 'import json, sys
for o in json.load(open(sys.argv[1])).get("overrides", []): print(o)' "$FINAL/train_meta.json" 2>/dev/null)
  echo "== attention rollout (training overrides: ${TRAIN_OVERRIDES[*]:-none})"
  cd "$REPO/behavior_prompting/behavior_prompting/train_network" || fail "clone missing at $REPO"
  run_step python train.py --config-name=umi_policy_dunetp +rollout=umi \
    rollout.checkpoint_path="$CKPT" \
    task.dataset.dataset_path="$DATASET_ZIP" \
    exp_name="${RUN_NAME}_${TAG}_rollout" group_tag="$LABEL" \
    logging.mode=offline +logging.dir="$WORK/rollout" hydra.run.dir="$WORK/rollout" \
    ${TRAIN_OVERRIDES[@]+"${TRAIN_OVERRIDES[@]}"}
  ATTN_RC=$?
  mkdir -p "$EVAL_DIR/videos"
  n=0
  for v in "$WORK"/rollout/vis/*.mp4; do [ -f "$v" ] && cp -a "$v" "$EVAL_DIR/videos/" && n=$((n + 1)); done
  echo "== copied $n video(s) to $EVAL_DIR/videos/"
  [ $ATTN_RC -eq 0 ] && [ $n -eq 0 ] && { ATTN_RC=1; MSG="rollout finished but wrote no videos"; }
fi
trap 'echo "== USR1 while finishing: ignored"' USR1

STATUS=ok
if [ $TIMED_OUT = 1 ]; then STATUS=timeout; MSG=${MSG:-"stopped at the time limit"}
elif [ "$EVAL_RC" -ne 0 ] || [ ! -s "$EVAL_DIR/eval.json" ]; then STATUS=failed; MSG=${MSG:-"offline_eval exited with $EVAL_RC"}
elif [ -n "$ATTN_RC" ] && [ "$ATTN_RC" -ne 0 ]; then STATUS=failed; MSG=${MSG:-"attention rollout exited with $ATTN_RC"}
fi
[ $STATUS = ok ] && rm -rf "$WORK"
write_job_json $STATUS
echo "== eval finished: $STATUS -> $EVAL_DIR"
[ $STATUS = ok ]
