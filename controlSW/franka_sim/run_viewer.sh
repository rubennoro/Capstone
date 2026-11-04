#!/usr/bin/env bash
# Closed-loop sim (sim-rig: Panda + Franka Hand + finger kit + rig) with the scripted pick-and-place policy and the
# MuJoCo viewer. Usage: controlSW/franka_sim/run_viewer.sh [POSES=3] [--box] [further eval args, e.g. --set k=v ...]
#   --box: box fingers (+10 cm) instead of the real finger kit. Results: ~/sim_runs/scripted_<time>/ (summary.html).
# First use: builds ~/envs/env_sim with ./setup_env.sh when it is missing.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x ~/envs/env_sim/bin/python ]]; then
  echo "[run_viewer] ~/envs/env_sim not found: building it with ./setup_env.sh (one time, a few minutes)"
  ./setup_env.sh
fi
POSES=3
if [[ $# -gt 0 && $1 =~ ^[0-9]+$ ]]; then POSES=$1; shift; fi
EXTRA=()
for a in "$@"; do
  if [[ $a == --box ]]; then
    EXTRA+=(--set tooling.finger_adapter.mode=box --set tooling.finger_adapter.kit=null
            --set tooling.finger_adapter.kit_sha256=null --set tooling.finger_adapter.extension_m=0.1)
  else
    EXTRA+=("$a")
  fi
done
export DISPLAY=${DISPLAY:-:1}
OUT=~/sim_runs/scripted_$(date +%Y%m%d_%H%M%S)
rc=0
env -u PYTHONPATH -u MUJOCO_GL ~/envs/env_sim/bin/python -m franka_sim eval \
  --profile sim-rig --policy scripted_pick_place --viewer --poses "$POSES" --episodes-per-pose 1 \
  "${EXTRA[@]}" --out "$OUT" || rc=$?
echo "report: $OUT/summary.html"
exit $rc
