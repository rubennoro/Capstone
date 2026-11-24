#!/usr/bin/env bash
# sim_deploy/run.sh: one command for a sim eval (PLAN.md section 5, DESIGN.md section 1).
#
#   sim_deploy/run.sh --policy <RUN_FOLDER | hold_still | replay:<zarr>#<idx>> --rollouts N --seed S [--tag T]
#                     [--home start|deploy] [--look robot|train|current] [--only-rollouts I,J,...] ...
#   sim_deploy/run.sh --help        (this text; the driver's flags: python -m sim_deploy.eval run --help)
#
# --only-rollouts I,J (passed to the driver unchanged): run only these rollout indices (0-based) with --seed's keys
#    [seed, n], i.e. the placements of a full run's rollouts I, J (batch retries); the folders keep the indices.
# --max-new-rollouts K (passed to the driver): at most K rollouts not yet run in this run.sh call, counted over the
#    sim restarts below (a restart after exit 3 gets K minus the rollouts this call already recorded), then exit 0
#    with the eval incomplete; batch.py's pieces (piece_rollouts) continue it with --resume <eval folder>.
# --look (default robot; sim_deploy/looks.py): the station both sims load: robot = stations/sim_rig_look_robot.yaml
#    (the real robot cell), train = stations/sim_rig_look_train.yaml (the training scene), current = stations/sim_rig.yaml
#    (E7's look). The driver gets --look and --model build/<station id>/model.json (recorded in eval.yaml).
# 0. prints whether stations/start_pose.yaml equals the upstream station.yaml q_start_rad (a WARNING if
#    not; nothing is changed: `python -m sim_deploy.scene start-pose --from-git` adopts it). Upstream = env
#    SIM_DEPLOY_UPSTREAM_REPO (default ~/Capstone-final-deployment) + SIM_DEPLOY_UPSTREAM_REF (default
#    rnoro/final-deployment), sim_deploy/upstream.py
# 1. builds the model if build/<station id>/model.json is missing or older than the station file
#    (python -m sim_deploy.model build --station <look station>)
# 2. starts the sim PC: franka_control --sim --local --station sim_deploy/stations/control_sim.yaml (E4) merged
#    (P1's sim_pc/merge_station.py) with the PC key p's homing.q_start_rad into build/control_sim_<home>.merged.yaml:
#    --home start (default, P2): stations/start_pose.yaml; --home deploy: q_start_rad null (build/start_pose_null.yaml:
#    the real PC refuses c away from a set start pose); --home trainstart = start. Waits for its scene `hello` on :5651
#    with the model's sha
#    The look's build goes into the sim PC's station too (an overlay build/look_<look>.sim_pc.yaml: sim.model,
#    sim.model_json), since the sim Pi refuses a state export of another model.
# 3. starts the sim Pi: python -m sim_deploy.sim_pi --station <look station> (E5)
# 4. runs the eval driver with the arguments given (python -m sim_deploy.eval run ...), which starts and stops
#    policy_deploy itself
# Logs: sim_evals/_logs/<date>_<time>/{model,sim_pc,sim_pi,eval}.log (the eval's own folder is printed at the end).
# Stops the sim PC and sim Pi on exit and on Ctrl-C, then every other process still running that was started under
# this run.sh (env SIM_DEPLOY_RUN_TAG, eval/procs.py): policy_deploy's launcher puts frames and policy in sessions of
# their own, so a signal to run.sh's group misses them if the launcher dies without stopping them.
# Arguments go to the driver unchanged (--dry-run skips 2 and 3).
set -euo pipefail
unset PYTHONPATH

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${SIM_DEPLOY_PY:-$HOME/envs/sim_deploy/bin/python}"
SIM_PC_BIN="$REPO/deployment/control_deploy/franka_control/build-sim/franka_control"
SIM_PC_STATION="$HERE/stations/control_sim.yaml"
LOGS="$REPO/sim_evals/_logs/$(date +%Y-%m-%d_%H%M%S)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# every process started below inherits the tag; the EXIT trap stops those still running (batch.py sets one per call)
export SIM_DEPLOY_RUN_TAG="${SIM_DEPLOY_RUN_TAG:-runsh_$$_$(date +%s%N)}"

for a in "$@"; do
  if [[ "$a" == "-h" || "$a" == "--help" ]]; then
    sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0
  fi
done
dry=0
home=start
prev=""
start_pose_arg=""
max_new=""
resume_arg=""
look=robot
for a in "$@"; do
  [[ "$prev" == "--look" ]] && look="$a"
  [[ "$a" == --look=* ]] && look="${a#--look=}"
  [[ "$a" == "--dry-run" ]] && dry=1
  [[ "$prev" == "--home" ]] && home="$a"
  [[ "$a" == --home=* ]] && home="${a#--home=}"
  [[ "$prev" == "--start-pose" ]] && start_pose_arg="$a"      # P4: another start pose file (the driver reads it too)
  [[ "$a" == --start-pose=* ]] && start_pose_arg="${a#--start-pose=}"
  [[ "$prev" == "--max-new-rollouts" ]] && max_new="$a"
  [[ "$a" == --max-new-rollouts=* ]] && max_new="${a#--max-new-rollouts=}"
  [[ "$prev" == "--resume" ]] && resume_arg="$a"
  [[ "$a" == --resume=* ]] && resume_arg="${a#--resume=}"
  prev="$a"
done
START_POSE="${start_pose_arg:-$HERE/stations/start_pose.yaml}"
# P4: SIM_DEPLOY_SIM_PC_OVERLAY = a sim-only station overlay merged last (e.g. stations/homing_fast_sweep.yaml)
OVERLAY="${SIM_DEPLOY_SIM_PC_OVERLAY:-}"
case "$home" in                    # P2: start = h then p (stations/start_pose.yaml); deploy = h only, no start pose
  start|trainstart) home=start ;;
  deploy)
    START_POSE="$HERE/build/start_pose_null.yaml"
    mkdir -p "$HERE/build"
    printf '%s\n' "# generated by run.sh --home deploy: no start pose (C3 start accepted anywhere)" \
      "franka_control:" "  homing:" "    q_start_rad: null" > "$START_POSE" ;;
  *) echo "[run.sh] --home $home: start or deploy"; exit 2 ;;
esac
case "$look" in robot|train|current) ;; *) echo "[run.sh] --look $look: robot, train or current"; exit 2 ;; esac
cd "$REPO"
read -r SIM_PI_STATION MODEL_DIR < <("$PY" -m sim_deploy.looks "$look") || { echo "[run.sh] --look $look: no station"; exit 2; }
MODEL_JSON="$MODEL_DIR/model.json"
LOOK_OVERLAY="$HERE/build/look_$look.sim_pc.yaml"   # the sim PC loads the same build as the sim Pi
SIM_PC_BASE="$SIM_PC_STATION"      # P1: franka_control takes one station: base + start pose, merged at each start
SIM_PC_STATION="$HERE/build/$(basename "$SIM_PC_BASE" .yaml)_${home}_$look.merged.yaml"
export SIM_DEPLOY_SIM_PC_STATION="$SIM_PC_STATION"   # recorded in eval.yaml
mkdir -p "$LOGS"
echo "[run.sh] logs in $LOGS"
echo "[run.sh] look: $look (station $(basename "$SIM_PI_STATION"), model $(basename "$MODEL_DIR"))"

# 0. the start pose vs the real arm's (the upstream station.yaml: sim_deploy/upstream.py), read only
if [[ "$home" == start ]]; then
  "$PY" - "$START_POSE" <<'EOF' || true
import sys
from sim_deploy.scene import start_pose as sp
r = sp.compare_git(sys.argv[1])
from sim_deploy import upstream
line = f"[run.sh] {r['text']} [upstream: {upstream.describe()}]"
print(("\n" + "!" * 100 + "\n" + line + "\n" + "!" * 100) if r["same"] is False else line)
EOF
fi

pids=()
cleanup() {
  local code=$?
  for pid in "${pids[@]:-}"; do
    [[ -n "$pid" ]] && kill -INT "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]:-}"; do
    [[ -z "$pid" ]] && continue
    for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$pid" 2>/dev/null || true
  done
  "$PY" -m sim_deploy.eval.procs stop-tagged "$SIM_DEPLOY_RUN_TAG" || true   # orphans (frames, policy, launcher)
  echo "[run.sh] stopped (exit $code)"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# 1. model (the look's build)
if [[ ! -f "$MODEL_JSON" || "$SIM_PI_STATION" -nt "$MODEL_JSON" ]]; then
  echo "[run.sh] building the model of $(basename "$SIM_PI_STATION") (missing or older than the station)"
  set +e
  "$PY" -m sim_deploy.model build --station "$SIM_PI_STATION" 2>&1 | tee "$LOGS/model.log"
  build_code="${PIPESTATUS[0]}"
  set -e
  [[ "$build_code" == 0 ]] || { echo "[run.sh] the model build failed (see $LOGS/model.log)"; exit 1; }
fi
printf '%s\n' "# generated by run.sh --look $look: the sim PC loads the look's build (= the sim Pi's)" "sim:" \
  "  model: $MODEL_DIR/scene.xml" "  model_json: $MODEL_JSON" > "$LOOK_OVERLAY"
SHA="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['model_sha256'])" "$MODEL_JSON")"

start_sim() {
  for port in 5555 5620 5630 5650 5651; do
    if "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(1 if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) else 0)" "$port"; then
      echo "[run.sh] port $port is already in use: stop the other sim / policy_deploy first (DESIGN.md section 10)"; return 1
    fi
  done

  # 2. sim PC (flags: sim_deploy/sim_pc/README.md, E4; command_line.cpp has --sim, --local, --station)
  [[ -x "$SIM_PC_BIN" ]] || { echo "[run.sh] no $SIM_PC_BIN: build it (sim_deploy/sim_pc/README.md)"; exit 1; }
  "$PY" -I "$HERE/sim_pc/merge_station.py" "$SIM_PC_BASE" "$START_POSE" "$SIM_PC_STATION" "$LOOK_OVERLAY" \
    ${OVERLAY:+"$OVERLAY"} || exit 1
  echo "[run.sh] starting the sim PC (home: $home, station $(basename "$SIM_PC_STATION"))"
  ( cd "$(dirname "$SIM_PC_BIN")/.." && exec "$SIM_PC_BIN" --sim --local --station "$SIM_PC_STATION" ) \
    > "$LOGS/sim_pc.log" 2>&1 &
  pids+=($!)
  if ! "$PY" - "$SHA" <<'EOF'
import sys, time
from sim_deploy.eval.sim_link import SceneClient, SceneError
want, end = sys.argv[1], time.monotonic() + 60
while time.monotonic() < end:
    try:
        h = SceneClient(timeout_s=1.0).hello()
    except SceneError:
        time.sleep(0.5)
        continue
    if h.get("model_sha256") != want:
        sys.exit(f"[run.sh] the sim PC runs model {h.get('model_sha256')}, model.json says {want}")
    print(f"[run.sh] sim PC up: model {want[:12]}")
    sys.exit(0)
sys.exit("[run.sh] no scene hello on :5651 within 60 s (see sim_pc.log)")
EOF
  then return 1; fi
  # P3: the scene REP answers before C2/C3 are bound; a sim PC that dies after hello (e.g. "cannot bind the C2
  # endpoint ... Address already in use") used to show up 30 s later as "no pi_stream packet". Wait for its ready line.
  local pc_pid="${pids[-1]}"
  for _ in $(seq 100); do
    grep -q 'ready: C2 on' "$LOGS/sim_pc.log" 2>/dev/null && break
    if ! kill -0 "$pc_pid" 2>/dev/null; then
      echo "[run.sh] the sim PC exited after hello: $(grep -m1 'FATAL' "$LOGS/sim_pc.log" || tail -1 "$LOGS/sim_pc.log")"
      return 1
    fi
    sleep 0.1
  done
  grep -q 'ready: C2 on' "$LOGS/sim_pc.log" || { echo "[run.sh] no 'ready: C2 on' in sim_pc.log within 10 s"; return 1; }

  # 3. sim Pi (sim_deploy/sim_pi, E5: --station takes the model from build/<station id>/)
  echo "[run.sh] starting the sim Pi"
  "$PY" -m sim_deploy.sim_pi --station "$SIM_PI_STATION" > "$LOGS/sim_pi.log" 2>&1 &
  pids+=($!)
  if ! "$PY" - <<'EOF'
import sys, time, zmq
s = zmq.Context.instance().socket(zmq.SUB)
s.setsockopt(zmq.SUBSCRIBE, b"")
s.connect("tcp://127.0.0.1:5555")
sys.exit(0 if s.poll(30000) else "[run.sh] no pi_stream packet on :5555 within 30 s (see sim_pi.log)")
EOF
  then return 1; fi
  echo "[run.sh] sim Pi streaming"
}

start_sim_retry() {                  # P3: retries (another process on this machine briefly holding a sim port, e.g.
  local i n="${SIM_DEPLOY_START_TRIES:-3}" w="${SIM_DEPLOY_START_WAIT_S:-3}"   # a test suite's fake PC on :5620)
  for i in $(seq "$n"); do
    start_sim && return 0
    stop_sim
    [[ $i -lt $n ]] && { echo "[run.sh] sim start failed: retrying in $w s ($i)"; sleep "$w"; }
  done
  echo "[run.sh] the sim did not start (see $LOGS)"; exit 1
}

stop_sim() {
  for pid in "${pids[@]:-}"; do [[ -n "$pid" ]] && kill -INT "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]:-}"; do
    [[ -z "$pid" ]] && continue
    for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$pid" 2>/dev/null || true
  done
  pids=()
}

[[ $dry -eq 0 ]] && start_sim_retry

# rollouts an eval folder holds as run (batch.py's done_rollouts = the driver's --resume rule): for --max-new-rollouts
done_in() { "$PY" -c "import sys; from sim_deploy.eval.batch import done_rollouts; print(done_rollouts(sys.argv[1]))" "$1"; }
done_start=0
[[ -n "$max_new" && -n "$resume_arg" ]] && done_start="$(done_in "$resume_arg")"

# 4. the eval driver (starts and stops policy_deploy itself)
# tee -i: a Ctrl-C in the terminal must not kill the tee (the driver still prints while it writes its results)
set +e                                   # the driver's exit code is read below (pipefail would end the script)
"$PY" -m sim_deploy.eval run "$@" --look "$look" --model "$MODEL_JSON" 2>&1 | tee -i "$LOGS/eval.log"
code="${PIPESTATUS[0]}"
set -e
# E7: exit 3 = the sim PC latched a fault (as the real PC, it needs a restart): restart the sim, resume the eval
restarts=0
while [[ "$code" == 3 && $dry -eq 0 && $restarts -lt ${SIM_DEPLOY_MAX_RESTARTS:-5} ]]; do
  restarts=$((restarts + 1))
  folder="$(sed -n 's/^\[eval\] eval folder: //p' "$LOGS/eval.log" | head -1)"
  [[ -n "$folder" ]] || break
  left=()                                # --max-new-rollouts: the rest of this call's K (argparse: the last one wins)
  if [[ -n "$max_new" ]]; then
    ran=$(( $(done_in "$folder") - done_start ))
    (( ran < max_new )) || { echo "[run.sh] --max-new-rollouts $max_new: $ran rollouts run in this call"; code=0; break; }
    left=(--max-new-rollouts "$((max_new - ran))")
  fi
  echo "[run.sh] the sim PC latched a fault: restarting the sim ($restarts) and resuming $folder"
  stop_sim
  sleep 1
  start_sim_retry
  set +e
  "$PY" -m sim_deploy.eval run "$@" --look "$look" --model "$MODEL_JSON" --resume "$folder" "${left[@]}" 2>&1 \
    | tee -i -a "$LOGS/eval.log"
  code="${PIPESTATUS[0]}"
  set -e
done
exit "$code"
