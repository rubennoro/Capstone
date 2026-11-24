#!/usr/bin/env bash
# W2 attribution batch for ONE checkpoint: plan the first-chunk jobs and run them on the GPU under the shared lock.
#
#   unset PYTHONPATH; cd ~/Capstone-main-sim
#   sim_deploy/camera_study/attr_run.sh RUN_FOLDER [N] [LOOK ...]
#
#   RUN_FOLDER  e.g. ~/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18
#   N           paired samples per observation (default 16)
#   LOOK        sim image sets to include (sim_evals/_pipeline/W2/sets/index.json looks; default: current real_kit)
#
# Jobs (attribution_table.py ids): real__ (15 refs, training path), nb__ (60 neighbours: the tolerance), deploy__
# (deploy path, only with DEPLOY=1), the sim sets (POSES, default handheld mount mount_start robot_trials; swaps on
# SWAP_POSES, default handheld mount) of each LOOK (+ per-camera swaps), and rm__ (the real arm's static
# start sets, width 0.090; real_mount.py). first_chunk.py skips jobs already done with the same settings and inputs.
# Settings: bf16, 8 DDIM steps, exec_h 12 (what every checkpoint ran with on the real arm on 2026-11-20; exec_h does
# not change the first chunk's end point). One GPU process at a time: the whole batch holds
# sim_evals/.gpu.lock (use a short look list to keep a batch to ~10-20 min).
set -euo pipefail
RUN=${1:?RUN_FOLDER}
N=${2:-16}
shift $(( $# >= 2 ? 2 : 1 ))
LOOKS=("$@")
[ ${#LOOKS[@]} -eq 0 ] && LOOKS=(current real_kit)
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
unset PYTHONPATH
PY=~/envs/sim_deploy/bin/python
TPY=~/envs/sim_policy_deploy_torch/bin/python
RID="$(basename "$(dirname "$(dirname "$RUN")")")__$(basename "$RUN")"
W=sim_evals/_pipeline/W2/attribution/_work
mkdir -p "$W" sim_evals/_pipeline/W2/logs
$PY -m sim_deploy.camera_study.attribution_table jobs real "$W/jobs_real_${RID}.json" >/dev/null
$PY -m sim_deploy.camera_study.attribution_table jobs sets "$W/jobs_sets_${RID}.json" --looks "${LOOKS[@]}" \
  --poses ${POSES:-handheld mount mount_start robot_trials} --swap-poses ${SWAP_POSES:-handheld mount} >/dev/null
[ -f "$W/jobs_rm.json" ] || $TPY -m sim_deploy.camera_study.real_mount jobs "$W/jobs_rm.json" --widths 090 >/dev/null
J="$W/jobs_${RID}.json"
DEPLOY=${DEPLOY:-0} $PY - "$W/jobs_real_${RID}.json" "$W/jobs_sets_${RID}.json" "$W/jobs_rm.json" "$J" <<'EOF'
import json, os, sys
real, sets, rm, out = sys.argv[1:5]
r = json.load(open(real))
jobs = [j for j in r if j["id"].startswith("real__")] + [j for j in r if j["id"].startswith("nb__")]
if os.environ.get("DEPLOY") == "1":
    jobs += [j for j in r if j["id"].startswith("deploy__")]
jobs += json.load(open(sets)) + json.load(open(rm))
json.dump(jobs, open(out, "w"), indent=1)
print(len(jobs), "jobs ->", out)
EOF
LOG="sim_evals/_pipeline/W2/logs/first_chunk_${RID}_$(date +%m%d_%H%M).log"
echo "$(date +%T) $RID N=$N looks=${LOOKS[*]} -> $LOG"
flock "$REPO/sim_evals/.gpu.lock" $TPY -m sim_deploy.camera_study.first_chunk --run "$RUN" --jobs "$J" \
  --out "sim_evals/_pipeline/W2/attribution/chunks/$RID" --n "$N" --precision bf16 --steps 8 --exec-h 12 \
  >"$LOG" 2>&1
echo "$(date +%T) $RID done (rc 0; a failure exits earlier under set -e)"
