"""Unattended closed-loop sim evals: a plan of checkpoints x looks, one sim chain (run.sh) at a time.

    python -m sim_deploy.eval batch --plan sim_deploy/eval/batches/w3.yaml [--dry-run]
    python -m sim_deploy.eval batch --resume <BATCH_ID>

Plan file (YAML; batches/w3.yaml, batches/smoke.yaml):

    name: w3                         # batch id = <name>_<YYYY-mm-dd_HHMM>
    checkpoints: all                 # or a list: "TASK/run_NNN" (prefix of <TASK>/policy/<run>), a run folder path,
                                     #   or {id: <one of those>, settings: {...}} to override the settings for it
    looks: [robot, train]
    rollouts: 10                     # N per checkpoint x look
    seed: 0
    starts: random                   # random (default): rollout n's eraser from rng([seed, n]) (scene/README.md);
                                     #   set: rollout n = start n of eval/start_sets/w3.yaml, the same for every
                                     #   checkpoint (eval/start_set.py; rollouts <= its 18 starts); set:<FILE or name>
                                     #   another set; passed to the driver as --starts set:<file>
    timeout: 60                      # s per rollout
    settings: {precision: bf16, diffusion_steps: 8, exec_h: 12}
    invalid_retries: 1               # re-run rollouts `evaltool score` marks invalid, this many times
    extra_args: []                   # more driver flags for every run (e.g. [--video-fps, "0"])
    load_max: 3                      # each run waits while loadavg(1) >= this
    piece_rollouts: 6                # each run.sh call runs at most this many new rollouts (driver
                                     #   --max-new-rollouts); the batch calls again for the same eval folder (--resume)
                                     #   until the run is complete; 0 = the whole run in one call (before 2026-11-22)
    gpu_cool_c: 65                   # before every run.sh call: wait until nvidia-smi temperature.gpu < this, C
    gpu_cool_max_s: 900              #   ... at most this long, then go ahead with a WARN line
    gpu_pause_s: 60                  #   ... and at least this long since the previous call ended, even if cool

Keys missing from a plan take these defaults, also on ``--resume`` (it reloads the batch's frozen plan.yaml): a batch
frozen before the pieces existed resumes with pieces of 6 and the cool-downs. Pieces and cool-downs only split the
work between run.sh calls (each continuation is the driver's --resume of the same eval folder); a rollout itself always
runs in real time, and the eval folder, eval.yaml and protocol_sha are the same as for one call (why: the laptop GPU
crashed twice, kernel Xid 62, under ~20+ min of uninterrupted policy inference + EGL rendering at 70-80 C, 2026-11-22).

Before each run.sh call: hold ``flock sim_evals/.gpu.lock``, then wait while the UI (:8770 /api/status) reports a
run in progress, any sim/policy port (5555 5610 5620 5630 5640 5650 5651) answers, or loadavg(1) >= 3. Every run
gets env SIM_DEPLOY_LAUNCHER=batch and SIM_DEPLOY_BATCH_ID. Before and after each run the GPU is checked
(``nvidia-smi`` answers, no kernel ``NVRM: Xid`` since the batch started; with pieces before and after every
run.sh call); on a fault the batch stops (exit 2,
manifest ``gpu_fault`` + ``batch_end status gpu_fault``) without scoring that run, so ``--resume`` after a reboot scores
it and retries its invalid rollouts. A kernel Xid found after a run.sh call also writes ``<eval>/gpu_fault.json``
(the first Xid's local time and line) into that run's eval folder: the fault window [xid_time, cleared). When a
batch (``--resume``) starts and its GPU check passes, it writes ``cleared`` (this boot's start if the machine rebooted
after the Xid, else now; local naive time) into every open gpu_fault.json of the batch (manifest
``gpu_fault_cleared``). ``evaltool score`` marks every rollout whose [wall_start, wall_end] overlaps a window invalid
(``gpu_fault``; no wall_end: started before the window end), so the resume re-runs them (B1, 2026-11-22). Every run
gets env SIM_DEPLOY_LAUNCHER=batch and SIM_DEPLOY_BATCH_ID (the driver writes them into eval.yaml).
run.sh restarts the sim itself on exit 3 (a latched fault; its --max-new-rollouts budget counts over those restarts).
Pieces: every run.sh call of a run gets ``--max-new-rollouts min(piece_rollouts, rollouts left)`` (calls after the first
also ``--resume <eval folder>``); log ``[i/n] <key> piece p/P (rollouts a-b)``. A call that ends incomplete after
running its whole piece is a planned continuation; a call that ends incomplete with fewer new rollouts than its piece
(run.sh gave up after its sim restarts, the watchdog, a crash) is "short", and a run with more than MAX_RESUMES short
calls is given up (manifest events before the pieces: every incomplete call was short). A call that never printed an
eval folder is a start failure (MAX_START_FAILS). Cool-down: inside the GPU lock, before every call (the first piece
too), the GPU check, then wait until the temperature is < gpu_cool_c (poll ~15 s, a log line ~every minute) and
gpu_pause_s have passed since the previous call, at most gpu_cool_max_s (then a WARN line and go ahead); an nvidia-smi
that does not answer is a GPU fault (stop, exit 2) as above.

State: sim_evals/_batches/<id>/{plan.yaml, manifest.jsonl, logs/}. manifest.jsonl is append only:
``planned`` (one per run, incl. retry runs), ``run`` (one per run.sh call: eval_dir, exit_code, start/end, complete;
since the pieces also piece, max_new, new_rollouts, short), ``gpu_cool`` (one per call: piece, start_c, end_c,
waited_s, since_last_call_s, timed_out), ``score`` (invalid rollouts from evaltool score), ``gpu_fault`` /
``gpu_fault_cleared``, ``batch_start`` / ``batch_end``. ``--resume`` replays it: complete runs are skipped, incomplete ones resumed in their eval folder.

Invalid-rollout retry: a retry run re-runs exactly the invalid rollouts with the same seed and the driver's
``--only-rollouts I,J`` (placement seed_key [seed, n], or set start n, as in the full run; folders keep the
indices); its ``planned``
line has ``retry_of`` = the run's key, ``use_rollouts`` = ``only_rollouts`` = those indices (evaltool report --batch
counts only ``use_rollouts`` of a retry and drops them from the run it retries). Manifests written before
--only-rollouts (retry = --rollouts max(invalid)+1, no ``only_rollouts``) still resume and report as before.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import yaml

from sim_deploy import REPO, ROOT
from sim_deploy.eval import procs

RUN_SH = ROOT / "run.sh"
SIM_EVALS = REPO / "sim_evals"
BATCHES = SIM_EVALS / "_batches"
REPORTS = SIM_EVALS / "_reports"
GPU_LOCK = SIM_EVALS / ".gpu.lock"
SIM_PC_BIN = REPO / "deployment" / "control_deploy" / "franka_control" / "build-sim" / "franka_control"
PD_ENVS = (Path.home() / "envs" / "sim_policy_deploy", Path.home() / "envs" / "sim_policy_deploy_torch")
PORTS = (5555, 5610, 5620, 5630, 5640, 5650, 5651)
UI_STATUS = "http://127.0.0.1:8770/api/status"
UI_BUSY = ("starting", "running", "stopping")        # ui/server.py BUSY_STATES
LOAD_MAX = 3.0
POLL_S = 5.0
MAX_RESUMES = 2            # batch-level --resume of a run after a "short" call (incomplete before its piece was done)
MAX_START_FAILS = 3        # run.sh calls that never printed an eval folder (sim did not start)
DEFAULT_SETTINGS = {"precision": "bf16", "diffusion_steps": 8, "exec_h": 12}
LOOKS = ("robot", "train", "current")
PIECE_ROLLOUTS = 6         # plan defaults (2026-11-22, two Xid 62 under sustained load): see the docstring
GPU_COOL_C = 65.0
GPU_COOL_MAX_S = 900.0
GPU_PAUSE_S = 60.0
COOL_POLL_S = 15.0
# wall-time estimate (2026-11-21 evals): sim start ~40 s + policy_deploy ~10 s + gate wait; per rollout h ~2 s,
# p ~15 s, settle/placement ~5 s, then the policy until success / fault / timeout
EST_RUN_OVERHEAD_S = 90.0
EST_ROLLOUT_OVERHEAD_S = 25.0
EST_ROLLOUT_MIN_POLICY_S = 15.0


def log(msg: str) -> None:
    print(f"[batch] {msg}", flush=True)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------------------ plan

def load_plan(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a mapping expected")
    plan = {
        "name": str(raw.get("name") or Path(path).stem),
        "checkpoints": raw.get("checkpoints", "all"),
        "looks": list(raw.get("looks") or ["robot"]),
        "rollouts": int(raw.get("rollouts", 10)),
        "seed": int(raw.get("seed", 0)),
        "starts": str(raw.get("starts", "random")),
        "timeout": float(raw.get("timeout", 60)),
        "settings": {**DEFAULT_SETTINGS, **(raw.get("settings") or {})},
        "invalid_retries": int(raw.get("invalid_retries", 1)),
        "extra_args": [str(x) for x in (raw.get("extra_args") or [])],
        "load_max": float(raw.get("load_max", LOAD_MAX)),
        "piece_rollouts": int(raw.get("piece_rollouts", PIECE_ROLLOUTS)),
        "gpu_cool_c": float(raw.get("gpu_cool_c", GPU_COOL_C)),
        "gpu_cool_max_s": float(raw.get("gpu_cool_max_s", GPU_COOL_MAX_S)),
        "gpu_pause_s": float(raw.get("gpu_pause_s", GPU_PAUSE_S)),
    }
    unknown = set(raw) - set(plan) - {"start_set_file"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    bad = [lk for lk in plan["looks"] if lk not in LOOKS]
    if bad:
        raise ValueError(f"{path}: looks {bad}: expected {', '.join(LOOKS)}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", plan["name"]):
        raise ValueError(f"{path}: name {plan['name']!r}: letters, digits, _ and - only")
    if plan["rollouts"] < 1:
        raise ValueError(f"{path}: rollouts must be >= 1")
    for k in ("piece_rollouts", "gpu_cool_max_s", "gpu_pause_s"):
        if plan[k] < 0:
            raise ValueError(f"{path}: {k} must be >= 0")
    if plan["gpu_cool_c"] <= 0:
        raise ValueError(f"{path}: gpu_cool_c must be > 0 (deg C)")
    if plan["starts"] != "random" and plan["starts"] != "set" and not plan["starts"].startswith("set:"):
        raise ValueError(f"{path}: starts {plan['starts']!r}: random, set or set:<FILE or name> (demos was replaced "
                         f"by the shared start set, 2026-11-22)")
    if plan["starts"] != "random":
        from sim_deploy.eval import start_set
        f = start_set.resolve_file(plan["starts"][4:] if plan["starts"].startswith("set:") else None)
        plan["start_set_file"] = str(f)
        if f.exists():
            n = len(start_set.load(f)["starts"])
            if plan["rollouts"] > n:
                raise ValueError(f"{path}: {f.name} has {n} starts, rollouts {plan['rollouts']}")
    _check_settings(plan["settings"], str(path))
    return plan


def _check_settings(s: dict, where: str) -> None:
    unknown = set(s) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"{where}: unknown settings {sorted(unknown)} (precision, diffusion_steps, exec_h)")
    if s["precision"] not in ("fp32", "bf16"):
        raise ValueError(f"{where}: precision {s['precision']!r}: fp32 or bf16")


def _data_root() -> Path:
    from sim_deploy.ui.catalog import DEFAULTS
    return Path(DEFAULTS["data_root"])


def _has_ckpt(run: Path) -> bool:
    return any((run / "checkpoints").glob("*.ckpt"))


def resolve_checkpoints(spec) -> list[dict]:
    """[{id, run_folder, settings_override}] in plan order."""
    if spec == "all":
        from sim_deploy.ui.catalog import Catalog
        rows = Catalog().policies(replays=False)["policies"]
        return [{"id": r["id"], "run_folder": r["run_folder"], "override": {}}
                for r in rows if r["runnable"] and not r.get("smoke")]
    if not isinstance(spec, list) or not spec:
        raise ValueError("checkpoints: 'all' or a non-empty list")
    out = []
    for item in spec:
        override = {}
        if isinstance(item, dict):
            override = dict(item.get("settings") or {})
            item = item.get("id")
        if not isinstance(item, str) or not item:
            raise ValueError(f"checkpoints: bad entry {item!r}")
        p = Path(item).expanduser()
        if p.is_dir():
            run = p.resolve()
        else:
            task, _, prefix = item.partition("/")
            pol = _data_root() / task / "policy"
            cands = sorted(r for r in pol.glob(f"{prefix}*") if r.is_dir()) if prefix else []
            usable = [r for r in cands if _has_ckpt(r)]
            if len(usable) > 1:                         # prefer the non-smoke run
                usable = [r for r in usable if not r.name.endswith("_smoke")] or usable
            if len(usable) != 1:
                raise ValueError(f"checkpoints: {item!r} matches {len(usable)} fetched runs under {pol} "
                                 f"({', '.join(r.name for r in cands) or 'none'})")
            run = usable[0]
        if not _has_ckpt(run):
            raise ValueError(f"checkpoints: {run}: no checkpoints/*.ckpt (fetch it with train/fetch.sh)")
        task = run.parent.parent.name if run.parent.name == "policy" else run.parent.name
        out.append({"id": f"{task}/{run.name}", "run_folder": str(run), "override": override})
    return out


def expand(plan: dict, ckpts: list[dict]) -> list[dict]:
    """The planned runs (checkpoint outer, look inner): partial results cover every look of a checkpoint early.
    starts set: every run carries the same start-set file (checked by the dry run and the driver)."""
    runs = []
    sf = plan.get("start_set_file")
    for c in ckpts:
        settings = {**plan["settings"], **c["override"]}
        _check_settings(settings, f"checkpoint {c['id']}")
        for look in plan["looks"]:
            runs.append({"key": f"{c['id']}|{look}", "policy_id": c["id"], "run_folder": c["run_folder"],
                         "look": look, "rollouts": plan["rollouts"], "seed": plan["seed"],
                         "timeout": plan["timeout"], "settings": settings, "tag": f"{plan['name']}_{look}",
                         "starts": "set" if sf else "random", "starts_file": sf,
                         "retry_of": None, "use_rollouts": None, "only_rollouts": None})
    return runs


def n_rollouts(r: dict) -> int:
    """The rollouts one planned run records: its --only-rollouts list, else --rollouts."""
    return len(r["only_rollouts"]) if r.get("only_rollouts") else r["rollouts"]


def command(r: dict, extra: list[str], resume: str | None = None, max_new: int | None = None) -> list[str]:
    s = r["settings"]
    cmd = [str(RUN_SH), "--policy", r["run_folder"], "--rollouts", str(r["rollouts"]), "--seed", str(r["seed"]),
           "--tag", r["tag"], "--timeout", f"{r['timeout']:g}", "--look", r["look"],
           "--precision", s["precision"], "--diffusion-steps", str(s["diffusion_steps"]), "--exec-h", str(s["exec_h"])]
    if r.get("starts") == "set":
        cmd += ["--starts", f"set:{r['starts_file']}"]
    if r.get("only_rollouts"):
        cmd += ["--only-rollouts", ",".join(str(n) for n in r["only_rollouts"])]
    cmd += extra
    if resume:
        cmd += ["--resume", resume]
    if max_new is not None:
        cmd += ["--max-new-rollouts", str(max_new)]
    return cmd


def n_calls(r: dict, piece: int) -> int:
    """run.sh calls of one run in pieces of ``piece`` rollouts (0: one call)."""
    return -(-n_rollouts(r) // piece) if piece else 1


def estimate_s(runs: list[dict], piece: int = 0) -> tuple[float, float]:
    """(lower, upper) without cool-downs: every rollout ends after EST_ROLLOUT_MIN_POLICY_S / runs to its timeout;
    the sim start overhead once per run.sh call."""
    lo = hi = 0.0
    for r in runs:
        lo += EST_RUN_OVERHEAD_S * n_calls(r, piece) + n_rollouts(r) * (
            EST_ROLLOUT_OVERHEAD_S + min(EST_ROLLOUT_MIN_POLICY_S, r["timeout"]))
        hi += EST_RUN_OVERHEAD_S * n_calls(r, piece) + n_rollouts(r) * (EST_ROLLOUT_OVERHEAD_S + r["timeout"])
    return lo, hi


def run_deadline_s(r: dict, n: int | None = None) -> float:
    """Watchdog for one run.sh call of n rollouts (default: the whole run; incl. run.sh's own restarts): twice the
    upper estimate + 10 min."""
    n = n_rollouts(r) if n is None else n
    return 2 * (EST_RUN_OVERHEAD_S + n * (EST_ROLLOUT_OVERHEAD_S + r["timeout"])) + 600


def _hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


# ------------------------------------------------------------------------------------------------- host checks

def first_xid(since: dt.datetime) -> tuple[dt.datetime | None, str | None]:
    """(local time, line) of the first kernel ``NVRM: Xid`` line since ``since``, or (None, None)."""
    try:
        r = subprocess.run(["journalctl", "-k", "-q", "--no-pager", "-o", "short-iso",
                            "--since", since.strftime("%Y-%m-%d %H:%M:%S")],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    for ln in r.stdout.splitlines():
        if "NVRM: Xid" in ln:
            return _journal_time(ln.split(" ", 1)[0]), ln.strip()
    return None, None


def _journal_time(stamp: str) -> dt.datetime | None:
    """journalctl short-iso stamp (2026-11-22T12:04:09+0100 / +01:00 / Z) -> naive local time (as rollout.json's
    wall_start)."""
    m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)(Z|[+-]\d\d:?\d\d)?$", stamp)
    if not m:
        return None
    t = dt.datetime.fromisoformat(m.group(1))
    tz = m.group(2)
    if not tz:
        return t
    if tz == "Z":
        off = dt.timedelta(0)
    else:
        sign = -1 if tz[0] == "-" else 1
        hh, mm = int(tz[1:3]), int(tz[-2:])
        off = sign * dt.timedelta(hours=hh, minutes=mm)
    return t.replace(tzinfo=dt.timezone(off)).astimezone().replace(tzinfo=None)


def gpu_fault(since: dt.datetime | None = None) -> str | None:
    """Why the GPU is unusable, or None. Two checks: ``nvidia-smi`` answers normally (after the 2026-11-22 fault it said
    "System is not in ready state" and inference took ~3 s instead of ~0.13 s), and no kernel ``NVRM: Xid`` line since
    ``since`` (the batch start). Slow inference by itself is not a fault (user decision 2026-11-22)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,pstate,clocks.sm", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0 or "not in ready state" in out or not r.stdout.strip():
            return f"nvidia-smi: {out or 'exit ' + str(r.returncode)}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"nvidia-smi: {e}"
    if since is not None:
        try:
            r = subprocess.run(["journalctl", "-k", "-q", "--no-pager", "-o", "short-iso",
                                "--since", since.strftime("%Y-%m-%d %H:%M:%S")],
                               capture_output=True, text=True, timeout=30)
            xid = [ln for ln in r.stdout.splitlines() if "NVRM: Xid" in ln]
            if xid:
                return f"kernel: {xid[0].strip()}"
        except (OSError, subprocess.TimeoutExpired):
            pass                                         # no journal access: the nvidia-smi check still runs
    return None


class GpuFault(RuntimeError):
    pass


def gpu_temperature() -> tuple[float | None, str]:
    """(nvidia-smi temperature.gpu in deg C, "") or (None, why it could not be read)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)
    out = (r.stdout + r.stderr).strip()
    try:
        if r.returncode == 0 and "not in ready state" not in out:
            return float(r.stdout.strip().splitlines()[0]), ""
    except (ValueError, IndexError):
        pass
    return None, out or f"exit {r.returncode}"


def boot_time() -> dt.datetime | None:
    """This boot's start as naive local time (same convention as rollout.json's wall_start), from /proc/stat btime."""
    try:
        for ln in Path("/proc/stat").read_text().splitlines():
            if ln.startswith("btime "):
                return dt.datetime.fromtimestamp(int(ln.split()[1])).replace(microsecond=0)
    except (OSError, ValueError):
        pass
    return None


def _read_fault(f: Path) -> dict | None:
    try:
        d = json.loads(f.read_text())
        return d if isinstance(d, dict) and d.get("xid_time") else None
    except (OSError, ValueError):
        return None


def mark_gpu_fault(eval_dir: str, since: dt.datetime) -> str | None:
    """Write <eval>/gpu_fault.json with the first kernel Xid since ``since``: the start of a fault window
    [xid_time, cleared), open-ended until clear_gpu_fault writes ``cleared`` (evaltool score marks every rollout of
    the eval whose [wall_start, wall_end] overlaps a window invalid: gpu_fault). An already cleared window moves to
    ``earlier`` when a later Xid opens a new one. None when there is no Xid line (nvidia-smi alone)."""
    from sim_deploy.evaltool import constants as C
    t, line = first_xid(since)
    if t is None or not Path(eval_dir).is_dir():
        return None
    f = Path(eval_dir) / C.GPU_FAULT_FILE
    prev = _read_fault(f)
    earlier = list((prev or {}).get("earlier") or [])
    if prev and not prev.get("cleared"):
        if dt.datetime.fromisoformat(prev["xid_time"]) <= t:
            return str(f)                                # the open window already starts at an earlier Xid: keep it
    elif prev and dt.datetime.fromisoformat(prev["cleared"]) <= t:
        earlier.append({k: v for k, v in prev.items() if k != "earlier"})   # closed window: keep it, open a new one
    rec = {"xid_time": t.isoformat(timespec="seconds"), "line": line, "written": now_iso(),
           "by": "sim_deploy/eval/batch.py"}
    if earlier:
        rec["earlier"] = earlier
    f.write_text(json.dumps(rec, indent=1))
    log(f"GPU fault: {f} (rollouts running at/after {t:%H:%M:%S} become invalid until the GPU is cleared: gpu_fault)")
    return str(f)


def clear_gpu_fault(path: str) -> str | None:
    """Close the open fault window in ``path`` (a gpu_fault.json): ``cleared`` = this boot's start when the machine
    rebooted after the Xid, else now (the GPU check just passed without a reboot). Local naive time. Returns the
    cleared time, or None when there was nothing open."""
    f = Path(path)
    d = _read_fault(f)
    if not d or d.get("cleared"):
        return None
    xid = dt.datetime.fromisoformat(d["xid_time"])
    boot = boot_time()
    c = boot if boot is not None and boot > xid else dt.datetime.now().replace(microsecond=0)
    d["cleared"] = c.isoformat(timespec="seconds")
    d["cleared_by"] = "boot after the Xid" if c == boot else "GPU check passed (no reboot since the Xid)"
    f.write_text(json.dumps(d, indent=1))
    log(f"GPU fault cleared: {f} (window {d['xid_time']} .. {d['cleared']}, {d['cleared_by']})")
    return d["cleared"]


def power_warnings() -> list[str]:
    """On battery / power-saver the sim Pi needs ~31 s to stream, run.sh waits 30 s."""
    out = []
    online = []
    for d in Path("/sys/class/power_supply").glob("*"):
        try:
            typ = (d / "type").read_text().strip()
            if typ != "Battery" and (d / "online").exists():
                online.append((d / "online").read_text().strip() == "1")
        except OSError:
            continue
    if online and not any(online):
        out.append("on battery (no AC / USB supply online): the sim Pi may miss run.sh's 30 s start window; plug in")
    try:
        prof = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=5).stdout.strip()
        if prof and prof != "performance":
            out.append(f"power profile is {prof!r}, not 'performance' (powerprofilesctl set performance)")
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def ui_state() -> str | None:
    """The UI's run state, or None if no UI answers on :8770."""
    try:
        with urllib.request.urlopen(UI_STATUS, timeout=2) as resp:
            return str(json.loads(resp.read().decode()).get("state"))
    except Exception:                                    # noqa: BLE001  (no UI, or not our UI)
        return None


def busy_ports() -> list[int]:
    out = []
    for port in PORTS:
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                out.append(port)
    return out


def busy_reasons(load_max: float = LOAD_MAX) -> list[str]:
    why = []
    st = ui_state()
    if st in UI_BUSY:
        why.append(f"the UI reports a run {st}")
    ports = busy_ports()
    if ports:
        why.append(f"ports in use: {' '.join(map(str, ports))}")
    load = os.getloadavg()[0]
    if load >= load_max:
        why.append(f"loadavg(1) {load:.2f} >= {load_max:g}")
    return why


def wait_until_free(load_max: float = LOAD_MAX) -> None:
    last, t0 = 0.0, time.monotonic()
    while True:
        why = busy_reasons(load_max)
        if not why:
            return
        if time.monotonic() - last >= 60:
            hint = (" (only the load: close other programs or set load_max in the plan, then --resume)"
                    if time.monotonic() - t0 > 900 and all(w.startswith("loadavg") for w in why) else "")
            log("waiting: " + "; ".join(why) + hint)
            last = time.monotonic()
        time.sleep(POLL_S)


class GpuLock:
    def __init__(self) -> None:
        self.f = None

    def __enter__(self):
        GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        self.f = GPU_LOCK.open("a")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"waiting for {GPU_LOCK} (another batch or flock holder)")
            fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc) -> None:
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def lock_free() -> bool:
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def evaltool_available() -> bool:
    try:
        return importlib.util.find_spec("sim_deploy.evaltool.__main__") is not None
    except (ImportError, ValueError):
        return False


# ------------------------------------------------------------------------------------------------- the manifest

class Manifest:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.path = folder / "manifest.jsonl"

    def append(self, rec: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"time": now_iso(), **rec}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for ln in self.path.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:                 # a torn last line (killed mid-write)
                continue
        return out


def done_rollouts(eval_dir: str | None) -> int:
    """Rollouts of an eval folder the driver counts as run (its --resume rule: outcome != not_run; an interrupted
    rollout counts as run, its record says stop_reason user and `evaltool score` marks it invalid -> retry)."""
    return len(done_set(eval_dir))


def done_set(eval_dir: str | None) -> set:
    """The rollout indices done_rollouts counts."""
    if not eval_dir:
        return set()
    f = Path(eval_dir) / "rollouts.jsonl"
    if not f.exists():
        return set()
    n = set()
    for ln in f.read_text().splitlines():
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if rec.get("outcome") != "not_run":
            n.add(rec.get("n"))
    return n


def _short(e: dict) -> bool:
    """A ``run`` event that ended incomplete before its piece was done (counts toward MAX_RESUMES). Events written
    before the pieces have no ``short``: every incomplete call then was one."""
    return bool(e["short"]) if "short" in e else not e.get("complete")


def _span(ns: list[int]) -> str:
    if not ns:
        return "none"
    return f"{ns[0]}-{ns[-1]}" if ns == list(range(ns[0], ns[-1] + 1)) else ",".join(map(str, ns))


# ------------------------------------------------------------------------------------------------ one run.sh call

class Interrupted(Exception):
    def __init__(self, folder: str | None = None) -> None:
        super().__init__(folder)
        self.folder = folder


def call_run_sh(cmd: list[str], env: dict, log_path: Path, deadline_s: float) -> tuple[int, str | None]:
    """Run run.sh in its own process group; everything to log_path, progress lines to stdout.
    Returns (exit code, eval folder). Ctrl-C / the watchdog send SIGINT to the group (run.sh stops the sims).
    Afterwards every process still running with this call's SIM_DEPLOY_RUN_TAG is stopped (procs.py): frames and
    policy run outside run.sh's group, and a run.sh killed after the 60 s grace cannot stop them itself."""
    folder = None
    tag = f"batch_{os.getpid()}_{time.time_ns()}"
    env = {**env, procs.TAG_VAR: tag}
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", bufsize=1, start_new_session=True)
    fired = threading.Event()

    def stop_group():
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def watchdog():
        fired.set()
        log(f"watchdog: run.sh still running after {deadline_s:.0f} s: stopping it")
        stop_group()

    timer = threading.Timer(deadline_s, watchdog)
    timer.daemon = True
    timer.start()
    try:
        with log_path.open("a") as lg:
            lg.write(f"\n### {now_iso()} {' '.join(cmd)}\n")
            for line in proc.stdout:
                lg.write(line)
                lg.flush()
                m = re.match(r"^\[eval\] eval folder: (.+)$", line)
                if m:
                    folder = folder or m.group(1).strip()
                if line.startswith(("[eval] rollout", "[eval] eval folder", "[run.sh] the sim", "[eval] error")):
                    print("    " + line.rstrip(), flush=True)
        code = proc.wait()
    except KeyboardInterrupt:
        log("Ctrl-C: stopping run.sh (it stops the sims)")
        timer.cancel()
        stop_group()
        raise Interrupted(folder)
    finally:
        timer.cancel()
        if proc.poll() is None:                          # any other error: never leave a sim chain running
            stop_group()
        left = procs.stop_tagged(tag)
        if left:
            log(f"still running after run.sh ended, stopped: {'; '.join(left)}")
            with log_path.open("a") as lg:
                lg.write(f"### {now_iso()} still running after run.sh ended, stopped: {'; '.join(left)}\n")
    if fired.is_set():
        code = code if code else 124
    return code, folder


# ------------------------------------------------------------------------------------------------------ scoring

def invalid_from_score(score: dict) -> list[int]:
    """Rollout indices score.json marks invalid. Accepts (evaltool's layout may still move):
    top-level ``invalid_rollouts: [n...]``; ``rollouts`` as a list or {n: {...}} whose items carry ``valid: bool``,
    ``invalid: bool`` or ``validity`` (bool, "valid"/"invalid", or {valid: bool})."""
    if isinstance(score.get("invalid_rollouts"), list):
        return sorted({int(n) for n in score["invalid_rollouts"]})
    ros = score.get("rollouts")
    items = list(ros.items()) if isinstance(ros, dict) else [(None, e) for e in ros or []]
    out = set()
    for k, e in items:
        if not isinstance(e, dict):
            continue
        n = e.get("n", e.get("rollout", k))
        v = e.get("valid")
        if v is None and "invalid" in e:
            v = not e["invalid"]
        if v is None and "validity" in e:
            val = e["validity"]
            v = val.get("valid") if isinstance(val, dict) else (val == "valid") if isinstance(val, str) else val
        if v is False and n is not None:
            out.add(int(n))
    return sorted(out)


def score(eval_dir: str, log_path: Path) -> list[int] | None:
    """Run `evaltool score <eval>`; the invalid rollouts, or None if it could not score."""
    cmd = [sys.executable, "-m", "sim_deploy.evaltool", "score", eval_dir]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    with log_path.open("a") as lg:
        lg.write(f"\n### {now_iso()} {' '.join(cmd)} -> {r.returncode}\n{r.stdout}{r.stderr}")
    if r.returncode != 0:
        log(f"evaltool score failed (exit {r.returncode}, see {log_path.name}): no invalid-rollout retry")
        return None
    try:
        return invalid_from_score(json.loads((Path(eval_dir) / "score.json").read_text()))
    except (OSError, ValueError) as e:
        log(f"cannot read {eval_dir}/score.json ({e}): no invalid-rollout retry")
        return None


# ------------------------------------------------------------------------------------------------------- batch

class Batch:
    def __init__(self, batch_id: str, plan: dict) -> None:
        self.id = batch_id
        self.plan = plan
        self.dir = BATCHES / batch_id
        self.logs = self.dir / "logs"
        self.m = Manifest(self.dir)
        self.can_score = evaltool_available()
        self.t_start = dt.datetime.now()
        self.last_call_end: float | None = None         # monotonic end of this process's previous run.sh call

    def env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(SIM_DEPLOY_LAUNCHER="batch", SIM_DEPLOY_BATCH_ID=self.id)
        return env

    def state(self):
        """(planned runs in order, {key: last run event}, {key: score event})"""
        planned, last, scored = [], {}, {}
        seen = set()
        for rec in self.m.read():
            ev = rec.get("event")
            if ev == "planned" and rec["run"]["key"] not in seen:
                seen.add(rec["run"]["key"])
                planned.append(rec["run"])
            elif ev == "run":
                last[rec["key"]] = rec
            elif ev == "score":
                scored[rec["key"]] = rec
        return planned, last, scored

    def execute(self) -> int:
        self.logs.mkdir(parents=True, exist_ok=True)
        if not self.can_score:
            log("WARNING: sim_deploy.evaltool is not importable yet: no scoring, no invalid-rollout retries, no report")
        for w in power_warnings():
            log("WARNING: " + w)
        self.m.append({"event": "batch_start", "pid": os.getpid()})
        t_start = self.t_start = dt.datetime.now()
        i = 0
        failed = []
        try:
            self.check_gpu(t_start, "batch start")
            self.clear_gpu_faults()                     # on --resume: close the fault windows before any scoring
            while True:
                planned, last, scored = self.state()
                if i >= len(planned):
                    break
                r = planned[i]
                i += 1
                prev = last.get(r["key"])
                if not (prev and prev.get("complete")):
                    self.check_gpu(t_start, f"before {r['key']}")
                    ok = self.run_one(r, i, len(planned), prev)
                    # after the run, before scoring: on --resume (after a reboot) the run is scored then and its
                    # never-commanded rollouts (evaltool: no_accepted_waypoints) and those that started after a
                    # kernel Xid (<eval>/gpu_fault.json -> evaltool: gpu_fault) are retried
                    ed = (self.state()[1].get(r["key"]) or {}).get("eval_dir")
                    self.check_gpu(t_start, f"after {r['key']}", eval_dir=ed)
                    if not ok:
                        failed.append(r["key"])
                        continue
                    planned, last, scored = self.state()
                if r["key"] not in scored and self.can_score:
                    self.score_one(r, last[r["key"]]["eval_dir"])
        except GpuFault as e:
            self.m.append({"event": "batch_end", "status": "gpu_fault", "reason": str(e)})
            log(f"STOPPED: GPU fault ({e}). Reboot, check nvidia-smi, then: "
                f"python -m sim_deploy.eval batch --resume {self.id}")
            return 2
        except (Interrupted, KeyboardInterrupt):
            self.m.append({"event": "batch_end", "status": "interrupted"})
            log(f"interrupted; continue with: python -m sim_deploy.eval batch --resume {self.id}")
            return 130
        self.m.append({"event": "batch_end", "status": "failed_runs" if failed else "ok", "failed": failed})
        self.report()
        log(f"done: {self.id} ({len(failed)} run(s) failed{': ' + ', '.join(failed) if failed else ''})")
        return 1 if failed else 0

    def check_gpu(self, since: dt.datetime, when: str, eval_dir: str | None = None) -> None:
        why = gpu_fault(since)
        if why:
            marked = mark_gpu_fault(eval_dir, since) if eval_dir else None
            self.m.append({"event": "gpu_fault", "when": when, "reason": why, "eval_dir": eval_dir,
                           "gpu_fault_file": marked})
            raise GpuFault(why)

    def clear_gpu_faults(self) -> None:
        """The GPU check passed: write ``cleared`` into every gpu_fault.json this batch recorded that is still open."""
        files = {e["gpu_fault_file"] for e in self.m.read() if e.get("event") == "gpu_fault" and e.get("gpu_fault_file")}
        for f in sorted(files):
            c = clear_gpu_fault(f)
            if c:
                self.m.append({"event": "gpu_fault_cleared", "gpu_fault_file": f, "cleared": c})

    def run_one(self, r: dict, i: int, n: int, prev: dict | None) -> bool:
        """Run (or continue) one planned run in pieces until complete: each run.sh call runs at most piece_rollouts
        new rollouts, after a cool-down. False when it is given up (short calls / start failures)."""
        eval_dir = prev.get("eval_dir") if prev else None
        calls = [e for e in self.m.read() if e.get("event") == "run" and e["key"] == r["key"] and not e.get("interrupted")]
        shorts = sum(1 for e in calls if e.get("eval_dir") and _short(e))
        start_fails = sum(1 for e in calls if not e.get("eval_dir"))
        log_path = self.logs / (re.sub(r"[^A-Za-z0-9_.-]+", "_", r["key"]) + ".log")
        piece, total = self.plan["piece_rollouts"], n_rollouts(r)
        indices = list(r["only_rollouts"]) if r.get("only_rollouts") else list(range(r["rollouts"]))
        while True:
            done_before = done_set(eval_dir)
            if eval_dir and len(done_before) >= total:   # e.g. Ctrl-C on a final piece's last rollout, or run.sh
                # finished after the batch died: the folder is complete although the last event says otherwise.
                # Record it (never call run.sh with nothing left: that would start a fresh, empty eval folder).
                self.m.append({"event": "run", "key": r["key"], "eval_dir": eval_dir, "exit_code": 0, "start": now_iso(),
                               "end": now_iso(), "wall_s": 0.0, "resumed": True, "done_rollouts": len(done_before),
                               "complete": True, "found_complete": True, "log": str(log_path)})
                log(f"{r['key']}: {eval_dir} already holds all {total} rollouts: recorded as complete")
                return True
            resume = eval_dir or None
            if shorts > MAX_RESUMES:
                log(f"{r['key']}: still incomplete after {shorts} calls that ended early: giving up on it")
                return False
            if start_fails >= MAX_START_FAILS:
                log(f"{r['key']}: the sim did not start {start_fails} times: giving up on it")
                return False
            left = [x for x in indices if x not in done_before]
            limit = min(piece, len(left)) if piece else None
            p = len(done_before) // piece + 1 if piece else 1
            what = (f"piece {p}/{n_calls(r, piece)} (rollouts {_span(left[:limit])})" if piece
                    else f"(rollouts {_span(left)})")
            cmd = command(r, self.plan["extra_args"], resume, limit)
            with GpuLock():
                log(f"[{i}/{n}] {r['key']} {what}" + (f" resume {resume}" if resume else ""))
                self.cool_down(r["key"], p, eval_dir)
                wait_until_free(self.plan["load_max"])
                t0, start = time.monotonic(), now_iso()
                try:
                    code, folder = call_run_sh(cmd, self.env(), log_path, run_deadline_s(r, limit))
                except Interrupted as e:                 # record the folder: --resume continues it, no new one
                    ed = e.folder or resume
                    self.m.append({"event": "run", "key": r["key"], "eval_dir": ed, "exit_code": 130,
                                   "start": start, "end": now_iso(), "wall_s": round(time.monotonic() - t0, 1),
                                   "resumed": bool(resume), "done_rollouts": done_rollouts(ed), "complete": False,
                                   "interrupted": True, "piece": p, "max_new": limit, "log": str(log_path)})
                    raise
                finally:
                    self.last_call_end = time.monotonic()
            eval_dir = folder or resume
            done = done_rollouts(eval_dir)
            new = done - len(done_before) if eval_dir else 0
            complete = bool(eval_dir) and done >= total
            short = not complete and (limit is None or new < limit)
            self.m.append({"event": "run", "key": r["key"], "eval_dir": eval_dir, "exit_code": code, "start": start,
                           "end": now_iso(), "wall_s": round(time.monotonic() - t0, 1), "resumed": bool(resume),
                           "done_rollouts": done, "complete": complete, "piece": p, "max_new": limit,
                           "new_rollouts": new, "short": short, "log": str(log_path)})
            log(f"{r['key']}: exit {code}, {done}/{total} rollouts (+{new}) in {time.monotonic() - t0:.0f} s "
                f"-> {eval_dir}" + (" (ended early)" if short else ""))
            # after every call: a fault stops the batch here (exit 2), before more pieces run on a faulted GPU
            self.check_gpu(self.t_start, f"after {r['key']} piece {p}", eval_dir=eval_dir)
            if complete:
                return True
            if not short:
                continue                                 # the next piece
            if eval_dir:
                shorts += 1
            else:
                start_fails += 1
                time.sleep(30)                           # let ports / load settle before the next start

    def cool_down(self, key: str, piece: int, eval_dir: str | None) -> None:
        """Before a run.sh call (inside the GPU lock): the GPU check, then wait until temperature.gpu < gpu_cool_c and
        gpu_pause_s since this process's previous call, at most gpu_cool_max_s. Manifest ``gpu_cool``."""
        cool, cmax, pause = self.plan["gpu_cool_c"], self.plan["gpu_cool_max_s"], self.plan["gpu_pause_s"]
        self.check_gpu(self.t_start, f"before {key} piece {piece}", eval_dir=eval_dir)
        t0 = time.monotonic()
        last = t0 - 60
        since = None if self.last_call_end is None else t0 - self.last_call_end
        first = temp = self.gpu_temp(key, piece, eval_dir)
        timed_out = False
        while True:
            now = time.monotonic()
            pause_left = 0.0 if self.last_call_end is None else pause - (now - self.last_call_end)
            if temp < cool and pause_left <= 0:
                break
            if temp >= cool and now - t0 >= cmax and pause_left <= 0:
                timed_out = True
                log(f"WARN: GPU still {temp:g} C (>= {cool:g}) after {now - t0:.0f} s (gpu_cool_max_s): going ahead")
                break
            if temp >= cool and now - last >= 60:
                log(f"cooling: GPU {temp:g} C (waiting for < {cool:g} C, {now - t0:.0f} s so far)")
                last = now
            time.sleep(max(0.5, min(COOL_POLL_S, pause_left if temp < cool else COOL_POLL_S)))
            temp = self.gpu_temp(key, piece, eval_dir)
        waited = time.monotonic() - t0
        if waited >= 1:
            log(f"cool-down: GPU {first:g} -> {temp:g} C in {waited:.0f} s")
        self.m.append({"event": "gpu_cool", "key": key, "piece": piece, "start_c": first, "end_c": temp,
                       "waited_s": round(waited, 1), "since_last_call_s": None if since is None else round(since, 1),
                       "timed_out": timed_out})

    def gpu_temp(self, key: str, piece: int, eval_dir: str | None) -> float:
        """temperature.gpu, or the GPU-fault path (manifest gpu_fault, GpuFault: the batch stops with exit 2)."""
        t, err = gpu_temperature()
        if t is not None:
            return t
        when = f"cool-down before {key} piece {piece}"
        self.check_gpu(self.t_start, when, eval_dir=eval_dir)    # nvidia-smi not ready / a kernel Xid: raises
        why = f"nvidia-smi temperature: {err}"
        self.m.append({"event": "gpu_fault", "when": when, "reason": why, "eval_dir": eval_dir, "gpu_fault_file": None})
        raise GpuFault(why)

    def score_one(self, r: dict, eval_dir: str) -> None:
        bad = score(eval_dir, self.logs / "score.log")
        if bad is None:
            return
        bad = [n for n in bad if not r.get("use_rollouts") or n in r["use_rollouts"]]
        level = r["key"].count("|retry")
        if bad:
            log(f"{r['key']}: invalid rollouts {bad}")
        if bad and level < self.plan["invalid_retries"]:  # the retry's planned line first: a crash in between
            base = r["retry_of"] or r["key"]              # re-scores on --resume and finds it planned
            key = f"{base}|retry{level + 1}"
            if not any(e.get("event") == "planned" and e["run"]["key"] == key for e in self.m.read()):
                retry = {**r, "key": key, "retry_of": base, "use_rollouts": bad, "only_rollouts": bad,
                         "tag": f"{r['tag'].split('_retry')[0]}_retry{level + 1}"}
                self.m.append({"event": "planned", "run": retry})
                log(f"queued {key}: rollouts {bad} again (same seed, --only-rollouts {','.join(map(str, bad))})")
        self.m.append({"event": "score", "key": r["key"], "eval_dir": eval_dir, "invalid": bad})

    def report(self) -> None:
        if not self.can_score:
            log("no evaltool: no report (later: python -m sim_deploy.evaltool report "
                f"--batch {self.id} --out {REPORTS / self.id})")
            return
        cmd = [sys.executable, "-m", "sim_deploy.evaltool", "report", "--batch", self.id, "--out", str(REPORTS / self.id)]
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        (self.logs / "report.log").write_text(f"{' '.join(cmd)} -> {r.returncode}\n{r.stdout}{r.stderr}")
        log(f"report: {'written to ' + str(REPORTS / self.id) if r.returncode == 0 else 'failed'} "
            f"(exit {r.returncode}, logs/report.log)")


# ------------------------------------------------------------------------------------------------------ dry run

def dry_run(plan: dict, runs: list[dict], batch_id: str) -> int:
    rows = []

    def row(name, ok, detail):
        rows.append((name, ok, detail))

    row("run.sh", "OK" if os.access(RUN_SH, os.X_OK) else "FAIL", str(RUN_SH))
    row("sim PC binary", "OK" if os.access(SIM_PC_BIN, os.X_OK) else "FAIL", str(SIM_PC_BIN))
    for e in PD_ENVS:
        row(f"env {e.name}", "OK" if (e / "bin" / "python").exists() else "FAIL", str(e))
    for look in plan["looks"]:
        from sim_deploy import looks
        mj = looks.build_dir(look) / "model.json"
        st = looks.station(look)
        fresh = mj.exists() and mj.stat().st_mtime >= st.stat().st_mtime
        row(f"model {look}", "OK" if fresh else "WARN", f"{mj}" + ("" if fresh else " (run.sh builds it first)"))
    for w in power_warnings():
        row("power", "WARN", w)
    if not any(r[0] == "power" for r in rows):
        row("power", "OK", "AC and performance profile")
    st = ui_state()
    row("UI :8770", "WARN" if st in UI_BUSY else "OK", f"state {st}" if st else "not running")
    ports = busy_ports()
    row("ports", "WARN" if ports else "OK", f"in use: {ports} (the batch waits)" if ports else "all free")
    load = os.getloadavg()[0]
    row("loadavg(1)", "WARN" if load >= plan["load_max"] else "OK",
        f"{load:.2f} (each run waits while >= {plan['load_max']:g}; plan key load_max)")
    g = gpu_fault()
    row("gpu", "FAIL" if g else "OK", g or "nvidia-smi answers (the batch checks it and the kernel log before and "
        "after each run, and stops on a fault)")
    row("gpu lock", "OK" if lock_free() else "WARN", f"{GPU_LOCK} " + ("free" if lock_free() else "held (waits)"))
    row("evaltool", "OK" if evaltool_available() else "WARN",
        "score/report available" if evaltool_available() else "not importable yet: no scoring, retries, report")

    set_rows = _start_set_rows(runs) if any(r.get("starts") == "set" for r in runs) else {}
    rows += set_rows.get("_rows", [])

    log(f"dry run of batch {batch_id} (nothing is started or written)")
    w = max(len(r[0]) for r in rows)
    for name, ok, detail in rows:
        print(f"  {name:<{w}}  {ok:<4} {detail}")
    print(f"\nplan {plan['name']}: {len(runs)} runs = {len(runs) // max(len(plan['looks']), 1)} checkpoints x "
          f"looks {plan['looks']}, N={plan['rollouts']}, seed {plan['seed']}, starts {plan.get('starts', 'random')}"
          + (f" ({plan['start_set_file']})" if plan.get("start_set_file") else "") + ", "
          f"timeout {plan['timeout']:g} s, invalid retries {plan['invalid_retries']}")
    piece = plan["piece_rollouts"]
    calls = sum(n_calls(r, piece) for r in runs)
    print(f"pieces: piece_rollouts {piece} -> {calls} run.sh calls for {len(runs)} runs"
          + ("" if piece else " (0 = one call per run)") + f"; before every call: GPU < {plan['gpu_cool_c']:g} C "
          f"(gpu_cool_c, wait at most gpu_cool_max_s {plan['gpu_cool_max_s']:g} s) and >= gpu_pause_s "
          f"{plan['gpu_pause_s']:g} s since the previous call")
    for k, r in enumerate(runs, 1):
        s = r["settings"]
        print(f"  {k:>3}. {r['policy_id']:<34} {r['look']:<6} {s['precision']}/{s['diffusion_steps']}/{s['exec_h']}  "
              f"tag {r['tag']}" + (f"  only rollouts {r['only_rollouts']}" if r.get("only_rollouts") else ""))
        if r.get("starts") == "set":
            print(f"       starts {set_rows.get((r['key'], 'line'), r['starts_file'])}")
    first = min(piece, n_rollouts(runs[0])) if runs and piece else None
    print(f"\nfirst command{' (piece 1)' if piece else ''}:\n  SIM_DEPLOY_LAUNCHER=batch SIM_DEPLOY_BATCH_ID=" + batch_id + " "
          + " ".join(command(runs[0], plan["extra_args"], None, first)) if runs else "")
    lo, hi = estimate_s(runs, piece)
    lo1, hi1 = estimate_s(runs)
    restarts = (calls - len(runs)) * EST_RUN_OVERHEAD_S
    x_lo = max(calls - 1, 0) * plan["gpu_pause_s"]
    x_hi = calls * max(plan["gpu_pause_s"], plan["gpu_cool_max_s"])
    print(f"\nwall time: {_hms(lo + x_lo)} (rollouts end after ~{EST_ROLLOUT_MIN_POLICY_S:g} s, GPU always cool) to "
          f"{_hms(hi + x_hi)} (all time out, every cool-down runs to gpu_cool_max_s), plus invalid-rollout retries and "
          f"gate waits")
    print(f"  of which the pieces add {_hms(restarts + x_lo)} to {_hms(restarts + x_hi)}: {calls - len(runs)} extra sim "
          f"starts x ~{EST_RUN_OVERHEAD_S:g} s = {_hms(restarts)}, pauses {max(calls - 1, 0)} x "
          f"{plan['gpu_pause_s']:g} s = {_hms(x_lo)}, up to {calls} x {max(plan['gpu_pause_s'], plan['gpu_cool_max_s']):g} s "
          f"= {_hms(x_hi)} if the GPU never cools below {plan['gpu_cool_c']:g} C (one call per run, no cool-downs: "
          f"{_hms(lo1)} to {_hms(hi1)})")
    bad = [r for r in rows if r[1] == "FAIL"]
    print("dry run:", "FAIL" if bad else "OK")
    return 1 if bad else 0


def _start_set_rows(runs: list[dict]) -> dict:
    """Dry run, starts set: check rows (the file loads, every start used recomputes in each look, every checkpoint is
    in the file with its split and exactly one held-out demo) and per run a line with its held-out rollout."""
    from sim_deploy.eval import start_set as sst
    out: dict = {"_rows": []}
    docs: dict = {}
    verified: set = set()
    for r in runs:
        f = Path(r["starts_file"])
        rel = str(f.relative_to(REPO)) if f.is_relative_to(REPO) else str(f)
        if f not in docs:
            try:
                docs[f] = sst.load(f)
            except (OSError, ValueError) as e:
                docs[f] = None
                out["_rows"].append(("start set", "FAIL", f"{rel}: {e} (python -m sim_deploy.eval starts build)"))
        doc = docs[f]
        if doc is None:
            out[(r["key"], "line")] = f"{rel}: MISSING"
            continue
        n_used = r["only_rollouts"] or list(range(r["rollouts"]))
        if (f, r["look"]) not in verified:
            verified.add((f, r["look"]))
            res = sst.verify(doc, sst.load_scene(r["look"]))
            probs = [f"rollout {x['rollout']}: {'; '.join(x['problems'])}" for x in res if x["problems"]]
            src = {}
            for s in doc["starts"]:
                src[s["source"]] = src.get(s["source"], 0) + 1
            out["_rows"].append((f"start set {doc['name']} ({r['look']})", "FAIL" if probs else "OK",
                                 f"{rel}: {len(res)} starts {src} recompute" + (f"; {' | '.join(probs)}" if probs else
                                                                              " within 0.5 mm / 0.1 deg, all checks pass")))
        ck = sst.checkpoint_entry(doc, r["run_folder"])
        if ck is None:
            out["_rows"].append((f"split {r['policy_id']}", "FAIL", f"{rel} was built without this checkpoint's "
                                 f"training split (held-out demo unknown)"))
        elif not ck["agree"] or ck["problems"] or len(ck["held_out"]) != 1:
            out["_rows"].append((f"split {r['policy_id']}", "WARN", f"held out {ck['held_out']}; {ck['problems']}"))
        held = [n for n in n_used if ck and n in ck["held_out"]]
        out[(r["key"], "line")] = (f"{rel}: rollouts {n_used[0]}-{n_used[-1]}" if n_used == list(range(n_used[0], n_used[-1] + 1))
                                   else f"{rel}: rollouts {n_used}") + (
            f"; held-out demo = rollout {held} (val of {ck['dataset']}, {ck['sample_type']} split "
            f"{'= recomputed' if ck['agree'] else 'DIFFERS'})" if ck else " (NOT LISTED)")
    return out


# --------------------------------------------------------------------------------------------------------- CLI

def add_arguments(p) -> None:
    p.add_argument("--plan", default=None, help="plan YAML (sim_deploy/eval/batches/*.yaml)")
    p.add_argument("--resume", default=None, metavar="BATCH_ID", help="continue sim_evals/_batches/<BATCH_ID>")
    p.add_argument("--dry-run", action="store_true", help="print the run list, wall-time estimate and checks")


def main_batch(a) -> int:
    if a.resume:
        folder = BATCHES / a.resume
        if not (folder / "plan.yaml").exists():
            raise FileNotFoundError(f"{folder}/plan.yaml: no such batch")
        if a.plan:
            log(f"--resume: using the batch's own plan.yaml, not {a.plan}")
        plan = load_plan(folder / "plan.yaml")
        b = Batch(a.resume, plan)
        planned = b.state()[0]
        if a.dry_run:
            _, last, _ = b.state()
            todo = [r for r in planned if not (last.get(r["key"]) or {}).get("complete")]
            log(f"resume {a.resume}: {len(planned) - len(todo)} of {len(planned)} planned runs complete")
            unscored = [r["key"] for r in planned if (last.get(r["key"]) or {}).get("complete")
                        and r["key"] not in b.state()[2]]
            if unscored:
                log(f"complete, scored first (a retry run is queued at the end for their invalid rollouts): "
                    f"{', '.join(unscored)}")
            for r in todo:
                d = done_rollouts((last.get(r["key"]) or {}).get("eval_dir"))
                if d:
                    log(f"incomplete, continues in its eval folder: {r['key']} ({d}/{n_rollouts(r)} rollouts done)")
            return dry_run(plan, todo, a.resume)
        log(f"resuming {a.resume}: {len(planned)} planned runs")
        return b.execute()
    if not a.plan:
        raise ValueError("--plan <yaml> or --resume <BATCH_ID>")
    plan = load_plan(Path(a.plan))
    runs = expand(plan, resolve_checkpoints(plan["checkpoints"]))
    batch_id = f"{plan['name']}_{dt.datetime.now():%Y-%m-%d_%H%M}"
    while not a.dry_run and (BATCHES / batch_id).exists():
        batch_id = f"{plan['name']}_{dt.datetime.now():%Y-%m-%d_%H%M%S}"
        time.sleep(1)
    if a.dry_run:
        return dry_run(plan, runs, batch_id)
    folder = BATCHES / batch_id
    folder.mkdir(parents=True)
    frozen = {**plan, "checkpoints": [{"id": r["run_folder"], "settings": r["settings"]}
                                      for r in runs[::max(len(plan["looks"]), 1)]]}
    (folder / "plan.yaml").write_text(f"# frozen from {a.plan} at {now_iso()}\n" + yaml.safe_dump(frozen, sort_keys=False))
    b = Batch(batch_id, plan)
    for r in runs:
        b.m.append({"event": "planned", "run": r})
    log(f"batch {batch_id}: {len(runs)} runs, state in {folder}")
    return b.execute()
