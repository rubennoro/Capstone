"""U2: start and stop one eval run (`sim_deploy/run.sh ...`) and turn what it does into Status and events (DESIGN §2-3).

    r = Runner(settings)                # ui.settings.Settings, or a dict with repo / run_sh / sim_evals (+ optional
                                        # policy_data_root, log_tail, stop_kill_after_s, driver_match)
    run_id = r.start(request)           # RunRequest dict; RunnerBusy if one is running, InvalidRequest if malformed
    r.status()                          # the Status dict of DESIGN §3 (+ a few extra keys, see _status)
    r.events()                          # queue.Queue of event dicts (thread-safe; one consumer); drain_events() too
    r.session_dir()                     # the running policy_deploy session folder (for manager/steps.jsonl) or None
    r.stop()                            # 1st: graceful (SIGINT to the eval driver); 2nd: SIGKILL the whole tree
    r.close()                           # on server exit: SIGKILL the tree if still running
    r.frames(eval_dir)                  # `evaltool frames <eval_dir>` -> [frames.jpg paths] (blocking: call in a thread)

The request may carry "target" (the server's resolved --policy value: run folder | hold_still | replay:<zarr>#<i>);
without it "policy" is resolved here (hold_still, replay:..., an absolute run folder or <task>/<run> under
policy_data_root). A run folder must have checkpoints/*.ckpt. Optional "dry_run": true adds --dry-run.

Where the progress comes from (files first, stdout only where nothing else exists):
- the eval folder: the driver's "[eval] eval folder: <path>" line, else a new `sim_evals/**/eval.yaml` after start;
- `<eval>/rollouts.jsonl` (one line per finished rollout, rewritten with the same lines at the end) -> rollout_end;
  the stdout line "[eval] rollout N: <outcome> ..." is used only if the file has no line N right after it;
- `<eval>/policy_deploy.log` (the manager's raw terminal): status-line heads ("[idle]", "[homing]", "[ep 3 running
  2.1 s] ...") -> phase, policy_deploy's "episode <id> started" -> rollout_start;
- `<eval>/metrics.json` -> Status.metrics and the run_end event;
- run.sh / driver lines ("[run.sh] starting the sim PC", "[eval] policy_deploy ready (session <path>)", ...) -> phase
  and the session folder.

run.sh gets env SIM_DEPLOY_LAUNCHER=ui (the driver records `launcher: ui` in eval.yaml, evaltool/SCHEMA.md). After the
run ends, `python -m sim_deploy.evaltool score <eval>` runs in a thread (the UI stays usable; a failure is shown, not
fatal): Status.score = {"state": "scoring"|"done"|"error", ...} with success / grasp / max lift / min z from
score.json, plus a "score" event.

Stop. The driver (`python -m sim_deploy.eval run`) handles KeyboardInterrupt: it stops the rollout in progress (`s`),
quits policy_deploy (`q`), collects the session and writes rollouts.jsonl / metrics.json / summary.html / the index
row, then exits 1; run.sh's EXIT trap then stops the sim PC and sim Pi. So the first stop sends SIGINT to the driver
process only (not the group: a group SIGINT also kills run.sh's `tee`, and the driver's next print would fail with a
broken pipe before it writes its results). Before the driver runs (sim chain coming up) it sends SIGINT to the group,
like Ctrl-C. The second stop (or `stop_kill_after_s` after the first) sends SIGKILL to every process under run.sh,
including policy_deploy, which pexpect puts in its own session.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

# ------------------------------------------------------------------------------------------------ the formats
# run.sh (sim_deploy/run.sh) and the driver (sim_deploy/eval/driver.py, log() = "[eval] <msg>")
RE_LOGS = re.compile(r"^\[run\.sh\] logs in (?P<path>\S.*)$")
RE_EVAL_DIR = re.compile(r"^\[eval\] eval folder:? (?P<path>\S.*)$")
RE_PD_READY = re.compile(r"^\[eval\] policy_deploy ready \(session (?P<path>[^)]+)\)")
RE_ROLLOUT_LINE = re.compile(r"^\[eval\] rollout (?P<n>\d+): (?P<outcome>\w+)\b")
RE_RESULTS = re.compile(r"^\[eval\] (?P<k>\d+)/(?P<n>\d+) successes; results in (?P<path>\S.*)$")
RE_ERROR = re.compile(r"^\[eval\] error: (?P<msg>.*)$")
RE_PD_FAIL = re.compile(r"^\[eval\] policy_deploy did not reach idle: (?P<msg>.*)$")
RUNSH_PHASES = [                                  # (line regex, phase)
    (re.compile(r"^\[run\.sh\] building the model"), "building the model"),
    (re.compile(r"^\[run\.sh\] starting the sim PC"), "starting the sim PC"),
    (re.compile(r"^\[run\.sh\] sim PC up"), "sim PC up"),
    (re.compile(r"^\[run\.sh\] starting the sim Pi"), "starting the sim Pi"),
    (re.compile(r"^\[run\.sh\] sim Pi streaming"), "starting the eval driver"),
    (re.compile(r"^\[eval\] eval folder:? "), "preparing (policy_deploy use)"),
    (re.compile(r"^\[eval\] interrupted"), "stopping policy_deploy"),
    (re.compile(r"^\[eval\] \d+/\d+ successes"), "results written"),
    (re.compile(r"^\[run\.sh\] stopped"), "sim stopped"),
]
LOOKS_BAD = re.compile(r"error|Error|ERROR|FAIL|Traceback|in use|no scene hello|no pi_stream|did not|not found|"
                       r"No such|refus", re.I)

# policy_deploy's terminal (eval/pd_session.py; imported when possible so a format change follows the driver)
try:
    from sim_deploy.eval.pd_session import ANSI, EP_STARTED, HEAD, READY as PD_READY
except Exception:                                   # noqa: BLE001  (a copy of the same patterns)
    ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    PD_READY = re.compile(r"policy_deploy: all \d+ processes ready \(session: (?P<path>[^)]+)\)")
    EP_STARTED = re.compile(r"^episode (?P<id>\d+) started")
    HEAD = re.compile(r"^\[(?P<head>[^\]]+)\]")

ACTIVE = ("starting", "running", "stopping")
PRECISIONS = (None, "fp32", "bf16")
LOOKS = ("robot", "train", "current")   # run.sh --look (sim_deploy/looks.py); robot is the default
# 2026-11-21 (the user): every run with the arm's settings unless the request says otherwise
POLICY_DEFAULTS = {"precision": "bf16", "diffusion_steps": 8, "exec_h": 12}
HOMES = ("start", "deploy")      # P2: start pose (h + the PC's p, default) | deploy home (h only)
HOME_ALIASES = {"trainstart": "start"}
TAG_OK = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
REPLAY_OK = re.compile(r"^replay:(?P<path>.+)#(?P<idx>\d+)$")


class RunnerError(RuntimeError):
    """Every runner problem (message for the page)."""


class RunnerBusy(RunnerError):
    """A run is already going (POST /api/run -> 409)."""


class InvalidRequest(RunnerError, ValueError):
    """The RunRequest is malformed or names a policy that cannot run."""


def _setting(settings: Any, name: str, default: Any = None) -> Any:
    """Settings dataclass attribute, its .extra, or a plain dict key."""
    if isinstance(settings, dict):
        return settings.get(name, default)
    if hasattr(settings, name):
        return getattr(settings, name)
    extra = getattr(settings, "extra", None) or {}
    return extra.get(name, default)


class Runner:
    def __init__(self, settings: Any, resolve_policy: Callable[[str], str | Path | None] | None = None) -> None:
        """settings: ui.settings.Settings or a dict. resolve_policy: optional id -> run folder (e.g. the catalog's);
        without it "<task>/<run>" is policy_data_root/<task>/policy/<run>."""
        home = Path.home()
        self.repo = Path(_setting(settings, "repo", home / "Capstone-main-sim")).expanduser()
        self.run_sh = Path(_setting(settings, "run_sh", self.repo / "sim_deploy" / "run.sh")).expanduser()
        self.sim_evals = Path(_setting(settings, "sim_evals", self.repo / "sim_evals")).expanduser()
        self.data_root = Path(_setting(settings, "policy_data_root", home / "Capstone" / "data")).expanduser()
        self.tail_n = int(_setting(settings, "log_tail", 30))
        self.kill_after_s = _setting(settings, "stop_kill_after_s", 150.0)  # after the 1st stop; None/0: never
        self.driver_match = str(_setting(settings, "driver_match", "sim_deploy.eval"))
        self.poll_s = float(_setting(settings, "runner_poll_s", 0.25))
        self.resolve_policy = resolve_policy
        self.python = str(_setting(settings, "evaltool_python", sys.executable))   # runs `-m sim_deploy.evaltool`
        self.score_timeout_s = float(_setting(settings, "score_timeout_s", 300.0))
        self._lock = threading.RLock()
        self._q: queue.Queue = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []
        self._reset(state="idle")

    # ------------------------------------------------------------------------------------------------ public
    def start(self, req: dict) -> str:
        with self._lock:
            if self._st["state"] in ACTIVE or (self._proc is not None and self._proc.poll() is None):
                raise RunnerBusy(f"a run is going ({self._st['run_id']}, {self._st['state']}): stop it first")
            req, args, notes = self._map_request(req)
            if not self.run_sh.is_file():
                raise RunnerError(f"no {self.run_sh}")
            now = dt.datetime.now()
            run_id = f"{now:%Y%m%d_%H%M%S}_{req['tag']}"
            self._reset(state="starting")
            st = self._st
            st.update(run_id=run_id, request=req, rollouts=req["rollouts"], notes=notes, phase="starting run.sh",
                      command=[str(self.run_sh), *args])
            self._t_start = time.monotonic()
            self._t_wall = time.time()
            log_dir = self.sim_evals / "_logs" / "ui"
            log_dir.mkdir(parents=True, exist_ok=True)
            st["log_file"] = str(log_dir / f"{run_id}.log")
            self._log = open(st["log_file"], "a", encoding="utf-8", errors="replace")
            self._log.write(f"# {now.isoformat(timespec='seconds')} {' '.join(st['command'])}\n"
                            f"# request {json.dumps(req)}\n")
            for n in notes:
                self._log.write(f"# note: {n}\n")
            self._log.flush()
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["PYTHONUNBUFFERED"] = "1"
            env["SIM_DEPLOY_LAUNCHER"] = "ui"           # eval.yaml launcher (evaltool/SCHEMA.md)
            env.pop("SIM_DEPLOY_BATCH_ID", None)
            try:
                self._proc = subprocess.Popen(st["command"], cwd=str(self.repo), env=env, stdin=subprocess.DEVNULL,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              start_new_session=True)
            except OSError as e:
                st.update(state="failed", last_error=f"cannot start {self.run_sh}: {e}")
                self._log.close()
                raise RunnerError(st["last_error"]) from None
            st["pid"] = self._proc.pid
            self._threads = [threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True,
                                              name="runner-stdout"),
                             threading.Thread(target=self._watch, args=(self._proc,), daemon=True,
                                              name="runner-watch")]
            for t in self._threads:
                t.start()
            return run_id

    def stop(self) -> dict:
        """1st call: graceful; 2nd: kill. Returns {"ok", "action": "none"|"graceful"|"kill", "detail"}."""
        with self._lock:
            p = self._proc
            if p is None or p.poll() is not None:
                return {"ok": True, "action": "none", "detail": "nothing is running"}
            self._stops += 1
            if self._stops == 1:
                self._st["state"] = "stopping"
                self._st["phase"] = "stopping"
                detail = self._graceful(p)
                self._log_line(f"# ui stop 1: {detail}")
                if self.kill_after_s:
                    t = threading.Timer(float(self.kill_after_s), self._kill_if_alive, args=(p,))
                    t.daemon = True
                    t.start()
                return {"ok": True, "action": "graceful", "detail": detail}
        detail = self._kill(p)
        return {"ok": True, "action": "kill", "detail": detail}

    def status(self) -> dict:
        with self._lock:
            st = dict(self._st)
            if self._t_start is not None:
                end = self._t_end if self._t_end is not None else time.monotonic()
                st["elapsed_s"] = round(end - self._t_start, 1)
            st["outcomes"] = [self._outcomes[n] for n in sorted(self._outcomes)]
            st["log_tail"] = list(self._tail)[-self.tail_n:]
            st["notes"] = list(st.get("notes") or [])
            return st

    def events(self) -> queue.Queue:
        """The event queue (dicts with "kind"); meant for one consumer (the server's broadcaster)."""
        return self._q

    def drain_events(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def session_dir(self) -> Path | None:
        """The policy_deploy session folder of the running eval (changes if the driver restarts policy_deploy)."""
        with self._lock:
            if self._st["state"] not in ("running", "stopping"):
                return None
            sd = self._st.get("session_dir")
            if sd:
                return Path(sd)
            ev = self._st.get("eval_dir")
        if ev:                                          # fallback: the newest folder under <eval>/session/
            dirs = [d for d in (Path(ev) / "session").glob("*") if d.is_dir()]
            if dirs:
                return max(dirs, key=lambda d: d.stat().st_mtime)
        return None

    def close(self) -> None:
        """On server exit (after its own stop / stop): kill whatever is still running."""
        p = self._proc
        if p is not None and p.poll() is None:
            self._kill(p)

    def busy(self) -> bool:
        with self._lock:
            return self._st["state"] in ACTIVE

    # ----------------------------------------------------------------------------------------- the request
    def _map_request(self, req: dict) -> tuple[dict, list[str], list[str]]:
        """RunRequest (DESIGN §2) -> (normalized request, run.sh arguments, notes)."""
        if not isinstance(req, dict):
            raise InvalidRequest("the request must be a JSON object")
        r = dict(req)
        notes: list[str] = []
        pol = r.get("policy")
        if not isinstance(pol, str) or not pol.strip():
            raise InvalidRequest("policy: a policy id, 'hold_still' or 'replay:<zarr>#<index>' is needed")
        pol = pol.strip()

        def num(key, kind, default, lo=None, hi=None, nullable=False):
            v = r.get(key, default)
            if v is None:
                if nullable:
                    return None
                v = default
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and float(v) != int(v)):
                raise InvalidRequest(f"{key}: expected {'an integer' if kind is int else 'a number'}, got {v!r}")
            v = kind(v)
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                raise InvalidRequest(f"{key}: {v} is outside {lo}..{hi}")
            return v

        rollouts = num("rollouts", int, 1, 1, 100)
        seed = num("seed", int, 0, 0, 2**31 - 1)
        timeout = num("timeout_s", float, 60.0, 1.0, 3600.0)
        steps = num("diffusion_steps", int, POLICY_DEFAULTS["diffusion_steps"], 1, 1000)
        exec_h = num("exec_h", int, POLICY_DEFAULTS["exec_h"], 1, 16)   # policy_deploy use: 1..ACTION_STEPS
        precision = r.get("precision") or POLICY_DEFAULTS["precision"]
        if precision not in PRECISIONS:
            raise InvalidRequest(f"precision: expected fp32, bf16 or null, got {precision!r}")
        look = r.get("look") or "robot"
        if look not in LOOKS:
            raise InvalidRequest(f"look: expected robot, train or current, got {look!r}")
        home = r.get("home") or "start"
        home = HOME_ALIASES.get(home, home) if isinstance(home, str) else home
        if home not in HOMES:
            raise InvalidRequest(f"home: expected start or deploy (trainstart = start), got {home!r}")
        tag = r.get("tag") or "ui"
        if not isinstance(tag, str) or not TAG_OK.match(tag):
            raise InvalidRequest(f"tag: 1-40 of A-Z a-z 0-9 . _ - (it ends the eval folder's name), got {tag!r}")
        tap, dry = r.get("tap", False), r.get("dry_run", False)
        if not isinstance(tap, bool) or not isinstance(dry, bool):
            raise InvalidRequest("tap / dry_run: expected true or false")

        tgt_in = r.get("target")                         # the server's resolved --policy value, when it sends one
        if tgt_in is not None and (not isinstance(tgt_in, str) or not tgt_in.strip()):
            raise InvalidRequest(f"target: expected a run folder, hold_still or replay:..., got {tgt_in!r}")
        target = self._target(tgt_in.strip() if tgt_in else pol)
        args = ["--policy", target, "--rollouts", str(rollouts), "--seed", str(seed), "--tag", tag,
                "--timeout", f"{timeout:g}"]
        if tap:
            args.append("--tap")
        if precision is not None:
            args += ["--precision", precision]
        if steps is not None:
            args += ["--diffusion-steps", str(steps)]
        if exec_h is not None:
            args += ["--exec-h", str(exec_h)]
        if dry:
            args.append("--dry-run")
        args += ["--home", home]                       # run.sh starts the sim PC to match, the driver follows it
        args += ["--look", look]                       # run.sh: the station both sims load (and the driver records)
        norm = {"policy": pol, "rollouts": rollouts, "seed": seed, "home": home, "look": look, "timeout_s": timeout,
                "tap": tap,
                "tag": tag, "precision": precision, "diffusion_steps": steps, "exec_h": exec_h}
        norm["target"] = target
        if dry:
            norm["dry_run"] = True
        return norm, args, notes

    def _target(self, pol: str) -> str:
        """Policy id -> the driver's --policy (run folder | hold_still | replay:<zarr>#<idx>)."""
        if pol == "hold_still":
            return pol
        m = REPLAY_OK.match(pol)
        if pol.startswith("replay:"):
            if not m:
                raise InvalidRequest(f"policy {pol!r}: expected replay:<zarr>#<index>")
            if not Path(m.group("path")).expanduser().exists():
                raise InvalidRequest(f"replay dataset not found: {m.group('path')}")
            return pol
        run = None
        if self.resolve_policy is not None:
            run = self.resolve_policy(pol)
        if run is None:
            p = Path(pol).expanduser()
            if p.is_absolute():
                run = p
            else:
                parts = pol.split("/")
                if len(parts) != 2 or not all(parts) or ".." in parts:
                    raise InvalidRequest(f"policy {pol!r}: expected <task>/<run>, hold_still or replay:...")
                run = self.data_root / parts[0] / "policy" / parts[1]
        run = Path(run)
        if not run.is_dir():
            raise InvalidRequest(f"policy {pol!r}: no run folder {run}")
        if not any((run / "checkpoints").glob("*.ckpt")):
            raise InvalidRequest(f"policy {pol!r}: {run} has no checkpoints/*.ckpt (fetch it with train/fetch.sh)")
        return str(run)

    # ------------------------------------------------------------------------------------------- the state
    def _reset(self, state: str) -> None:
        self._st: dict = {"state": state, "run_id": None, "request": None, "rollout": 0, "rollouts": 0, "phase": "",
                          "elapsed_s": 0.0, "outcomes": [], "last_error": None, "log_tail": [],
                          # extra keys (not in DESIGN §3 but useful to the page / U3)
                          "eval_dir": None, "logs_dir": None, "log_file": None, "session_dir": None,
                          "metrics": None, "exit_code": None, "stopped_by_user": False, "notes": [], "pid": None,
                          "command": None, "score": None}
        self._tail: collections.deque = collections.deque(maxlen=max(200, self.tail_n))
        self._outcomes: dict[int, str] = {}
        self._started: dict[int, dict] = {}            # driver n -> rollout_start event
        self._stdout_eps: dict[int, tuple[float, str]] = {}
        self._jsonl_lines = 0
        self._pd_off = 0
        self._pd_buf = ""
        self._pd_status = ""
        self._stops = 0
        self._killed = False
        self._errored = False
        self._t_start: float | None = None
        self._t_end: float | None = None
        self._t_wall = 0.0
        self._log = None

    def _emit(self, kind: str, **fields: Any) -> None:
        ev = {"kind": kind, "run_id": self._st.get("run_id"), "t": time.time(), **fields}
        self._q.put(ev)
        self._log_line(f"# event {json.dumps(ev, default=str)[:2000]}")

    def _log_line(self, line: str) -> None:
        f = self._log
        if f is not None and not f.closed:
            try:
                f.write(line + "\n")
                f.flush()
            except (OSError, ValueError):
                pass

    # -------------------------------------------------------------------------------------- stdout (run.sh)
    def _read_stdout(self, p: subprocess.Popen) -> None:
        assert p.stdout is not None
        for raw in iter(p.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            line = ANSI.sub("", line).split("\r")[-1]
            with self._lock:
                if self._proc is not p:
                    return
                self._tail.append(line)
                self._log_line(line)
                self._parse_line(line)

    def _parse_line(self, line: str) -> None:
        st = self._st
        for rx, phase in RUNSH_PHASES:
            if rx.search(line):
                if st["state"] != "stopping" or phase in ("stopping policy_deploy", "results written", "sim stopped"):
                    st["phase"] = phase
                break
        m = RE_LOGS.match(line)
        if m:
            st["logs_dir"] = m.group("path").strip()
            return
        m = RE_EVAL_DIR.match(line)
        if m:
            st["eval_dir"] = m.group("path").strip()
            if st["state"] == "starting":
                st["state"] = "running"
            return
        m = RE_PD_READY.match(line)
        if m:
            st["session_dir"] = m.group("path").strip()
            if st["state"] != "stopping":
                st["phase"] = "idle"
            return
        m = RE_ROLLOUT_LINE.match(line)
        if m:                                          # fallback for rollouts.jsonl (checked in _poll_files)
            self._stdout_eps.setdefault(int(m.group("n")), (time.monotonic(), m.group("outcome")))
            return
        m = RE_RESULTS.match(line)
        if m and not st["eval_dir"]:
            st["eval_dir"] = m.group("path").strip()
        for rx in (RE_ERROR, RE_PD_FAIL):
            m = rx.match(line)
            if m:
                st["last_error"] = line
                self._errored = True
                self._emit("error", message=m.group("msg"), line=line, source="stdout")

    # ------------------------------------------------------------------------------------------- the files
    def _watch(self, p: subprocess.Popen) -> None:
        while p.poll() is None:
            with self._lock:
                if self._proc is not p:
                    return
                try:
                    self._poll_files()
                except Exception as e:                  # noqa: BLE001 (never let the watcher die)
                    self._log_line(f"# runner: poll error {e!r}")
            time.sleep(self.poll_s)
        for t in self._threads:                         # let the stdout reader drain the last lines
            if t is not threading.current_thread():
                t.join(timeout=5.0)
        with self._lock:
            if self._proc is not p:
                return
            try:
                self._poll_files(final=True)
            except Exception as e:                      # noqa: BLE001
                self._log_line(f"# runner: final poll error {e!r}")
            self._finish(p.returncode)

    def _poll_files(self, final: bool = False) -> None:
        st = self._st
        if not st["eval_dir"]:
            found = self._find_eval_dir()
            if found:
                st["eval_dir"] = str(found)
                if st["state"] == "starting":
                    st["state"] = "running"
        if not st["eval_dir"]:
            return
        ev = Path(st["eval_dir"])
        self._poll_pd_log(ev / "policy_deploy.log")      # first: a start is numbered before its own end
        self._poll_rollouts(ev / "rollouts.jsonl", final)
        mj = ev / "metrics.json"
        if st["metrics"] is None and mj.is_file():
            try:
                st["metrics"] = json.loads(mj.read_text())
            except (OSError, json.JSONDecodeError):
                pass                                    # being written: next poll

    def _find_eval_dir(self) -> Path | None:
        """A sim_evals/**/eval.yaml newer than the start (when the driver's line was missed)."""
        if not self.sim_evals.is_dir():
            return None
        tag = (self._st.get("request") or {}).get("tag", "")
        best = None
        for pat in ("*/*/*/eval.yaml", "*/*/*/*/eval.yaml", "_dry_run/*/*/*/*/eval.yaml"):
            for f in self.sim_evals.glob(pat):
                try:
                    mt = f.stat().st_mtime
                except OSError:
                    continue
                if mt >= self._t_wall - 1.0 and f.parent.name.split("_", 2)[-1].startswith(tag):
                    if best is None or mt > best[0]:
                        best = (mt, f.parent)
        return best[1] if best else None

    def _poll_rollouts(self, path: Path, final: bool) -> None:
        lines: list[str] = []
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            lines = text.split("\n")[:-1]               # only complete lines
        recs = []
        for ln in lines:
            try:
                recs.append(json.loads(ln))
            except json.JSONDecodeError:
                break
        in_file = {int(r.get("n", i)) for i, r in enumerate(recs)}
        for i, rec in enumerate(recs):
            n = int(rec.get("n", i))
            if n not in self._outcomes:
                self._rollout_end(n, rec.get("outcome", "?"), rec, "rollouts.jsonl")
        self._jsonl_lines = max(self._jsonl_lines, len(recs))
        now = time.monotonic()
        for n, (t, outcome) in sorted(self._stdout_eps.items()):
            if n not in self._outcomes and n not in in_file and (final or now - t > 2.0):
                self._rollout_end(n, outcome, None, "stdout")

    def _rollout_end(self, n: int, outcome: str, rec: dict | None, source: str, **extra: Any) -> None:
        self._outcomes[n] = outcome
        st = self._st
        st["rollout"] = max(st["rollout"], n + 1)
        self._emit("rollout_end", n=n, rollout=n + 1, rollouts=st["rollouts"], outcome=outcome,
                   success=(rec or {}).get("success", outcome == "success"), record=rec, source=source, **extra)

    def _poll_pd_log(self, path: Path) -> None:
        """policy_deploy's raw terminal (\\r-rewritten status line + messages): phase and 'episode N started'."""
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self._pd_off:                          # rewritten: start over
            self._pd_off, self._pd_buf = 0, ""
        if size == self._pd_off:
            return
        with open(path, "rb") as f:
            f.seek(self._pd_off)
            data = f.read(size - self._pd_off)
        self._pd_off += len(data)
        self._pd_buf += data.decode("utf-8", "replace")
        parts = re.split(r"[\r\n]", self._pd_buf)
        self._pd_buf = parts.pop()
        for p in parts:
            self._pd_piece(ANSI.sub("", p).strip())
        if self._pd_buf.endswith("\x1b[K"):              # a complete status line (no \r after it yet)
            self._pd_piece(ANSI.sub("", self._pd_buf).strip())

    def _pd_piece(self, txt: str) -> None:
        if not txt:
            return
        st = self._st
        m = HEAD.match(txt)
        if m:
            if txt != self._pd_status:
                self._pd_status = txt
                if st["state"] != "stopping":
                    st["phase"] = m.group("head").strip()
                st["pd_status"] = txt
            return
        m = PD_READY.search(txt)
        if m and not st.get("session_dir"):
            st["session_dir"] = m.group("path").strip()
        m = EP_STARTED.search(txt)
        if m:
            n = len(self._outcomes)                      # rollouts are sequential: the next driver index
            while n in self._outcomes or n in self._started:
                n += 1
            if st["rollouts"] and n >= st["rollouts"]:
                return
            ev = {"n": n, "rollout": n + 1, "rollouts": st["rollouts"], "pd_rollout_id": int(m.group("id")),
                  "session_dir": st.get("session_dir"), "seed_key": [(st.get("request") or {}).get("seed"), n]}
            self._started[n] = ev
            st["rollout"] = n + 1
            self._emit("rollout_start", **ev)

    # ------------------------------------------------------------------------------------------- the end
    def _finish(self, code: int | None) -> None:
        st = self._st
        st["exit_code"] = code
        self._t_end = time.monotonic()
        user = self._stops > 0
        killed = self._stops > 1 or self._killed
        for n in sorted(self._started):                  # started but never finished: the stop ended it
            if n not in self._outcomes:
                self._rollout_end(n, "interrupted", None, "runner", synthetic=True,
                                  note="stopped from the ui before the driver recorded it" if user else
                                  "run.sh ended before the driver recorded it")
        if killed:
            st["state"] = "failed"
            st["last_error"] = "killed from the ui (second stop): results may be missing"
        elif user:
            st["state"] = "done"
            st["stopped_by_user"] = True
        elif code == 0:
            st["state"] = "done"
        else:
            st["state"] = "failed"
            st["last_error"] = f"{self._bad_line() or 'run.sh failed'} (exit {code})"
        st["phase"] = "stopped" if user else ("finished" if code == 0 else "failed")
        st["session_dir_last"] = st.get("session_dir")
        if st["state"] == "failed" and not killed and not self._errored:
            self._emit("error", message=st["last_error"], source="exit")
        ev = st["eval_dir"]
        self._emit("run_end", state=st["state"], exit_code=code, stopped_by_user=user, killed=killed,
                   eval_dir=ev, eval_rel=self._rel(ev), metrics=st["metrics"], outcomes=[self._outcomes[n] for n in
                                                                                       sorted(self._outcomes)],
                   log_file=st["log_file"], logs_dir=st["logs_dir"], last_error=st["last_error"])
        if self._log is not None:
            self._log.close()
        if ev and (Path(ev) / "rollouts.jsonl").is_file():
            st["score"] = {"state": "scoring", "eval_rel": self._rel(ev)}
            threading.Thread(target=self._score, args=(st["run_id"], ev), daemon=True, name="runner-score").start()

    # ------------------------------------------------------------------------------------------- evaltool
    def _evaltool(self, *args: str, timeout: float) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        return subprocess.run([self.python, "-m", "sim_deploy.evaltool", *args], cwd=str(self.repo), env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
                              timeout=timeout)

    @staticmethod
    def _last_line(r: subprocess.CompletedProcess) -> str:
        lines = [ln for ln in (r.stderr + r.stdout).splitlines() if ln.strip()]
        return lines[-1].strip() if lines else f"exit {r.returncode}"

    def _score(self, run_id: str, ev: str) -> None:
        """`evaltool score <ev>` after a run (its own thread): Status.score + a "score" event. Never raises."""
        res: dict = {"state": "error", "eval_rel": self._rel(ev)}
        try:
            r = self._evaltool("score", ev, timeout=self.score_timeout_s)
            if r.returncode != 0:
                res["error"] = f"evaltool score: {self._last_line(r)}"
            else:
                s = json.loads((Path(ev) / "score.json").read_text())["summary"]

                def kn(key):
                    d = s.get(key) or {}
                    return [d.get("k"), d.get("n")]
                mz = (s.get("min_z_fingers_rig_m") or {}).get("min")
                res.update(state="done", success=kn("success"), grasp=kn("grasp"),
                           max_lift_m=(s.get("max_lift_m") or {}).get("max"),
                           min_z_m=mz if mz is not None else (s.get("min_z_tcp_m") or {}).get("min"),
                           min_z_of="fingers/rig" if mz is not None else "tcp", invalid=s.get("invalid"))
        except Exception as e:                           # noqa: BLE001 - scoring is extra: show it, go on
            res["error"] = f"evaltool score: {type(e).__name__}: {e}"
        with self._lock:
            if self._st.get("run_id") != run_id:         # a newer run started meanwhile
                return
            self._st["score"] = res
            self._emit("score", **res)

    def frames(self, eval_dir: str | Path) -> list[str]:
        """`evaltool frames <eval_dir>`: the frames.jpg paths written (RunnerError with its last line on failure)."""
        try:
            r = self._evaltool("frames", str(eval_dir), timeout=self.score_timeout_s)
        except (OSError, subprocess.SubprocessError) as e:
            raise RunnerError(f"evaltool frames: {type(e).__name__}: {e}") from None
        if r.returncode != 0:
            raise RunnerError(f"evaltool frames: {self._last_line(r)}")
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip().endswith("frames.jpg")]

    def _rel(self, ev: str | None) -> str | None:
        if not ev:
            return None
        try:
            return str(Path(ev).resolve().relative_to(self.sim_evals.resolve()))
        except ValueError:
            return None

    def _bad_line(self) -> str | None:
        lines = [ln for ln in self._tail if ln.strip() and not ln.startswith("[run.sh] stopped")]
        for ln in reversed(lines):
            if LOOKS_BAD.search(ln):
                return ln.strip()
        return lines[-1].strip() if lines else None

    # ------------------------------------------------------------------------------------------- signals
    def _tree(self, root: int) -> list[int]:
        """root and every descendant (by ppid, from /proc), any session."""
        kids: dict[int, list[int]] = {}
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat", "rb") as f:
                    s = f.read().decode("utf-8", "replace")
                ppid = int(s[s.rfind(")") + 2:].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            kids.setdefault(ppid, []).append(int(d))
        out, todo = [], [root]
        while todo:
            pid = todo.pop()
            out.append(pid)
            todo += kids.get(pid, [])
        return out

    def _driver_pids(self, root: int) -> list[int]:
        out = []
        for pid in self._tree(root)[1:]:
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    argv = f.read().decode("utf-8", "replace").split("\0")
                if "-m" in argv and self.driver_match in argv and os.getpgid(pid) == root:
                    out.append(pid)
            except OSError:
                continue
        return out

    def _graceful(self, p: subprocess.Popen) -> str:
        try:
            drivers = self._driver_pids(p.pid)
        except OSError:
            drivers = []
        try:
            if drivers:
                for pid in drivers:
                    os.kill(pid, signal.SIGINT)
                return f"SIGINT to the eval driver (pid {', '.join(map(str, drivers))}): it ends the rollout, " \
                       f"quits policy_deploy and writes the results; run.sh then stops the sim"
            os.killpg(p.pid, signal.SIGINT)
            return "SIGINT to run.sh's process group (the eval driver is not running yet): run.sh stops the sim"
        except ProcessLookupError:
            return "already gone"

    def _kill_if_alive(self, p: subprocess.Popen) -> None:
        if p.poll() is None and self._proc is p:
            self._log_line(f"# ui: still running {self.kill_after_s} s after the stop: killing")
            self._kill(p)

    def _kill(self, p: subprocess.Popen) -> str:
        with self._lock:
            self._killed = True
            self._stops = max(self._stops, 2)
            self._st["phase"] = "killing"
        try:
            pids = self._tree(p.pid)
        except OSError:
            pids = [p.pid]
        for pid in pids:                                  # incl. policy_deploy (its own session, from pexpect)
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self._log_line(f"# ui stop 2: SIGKILL to {len(pids)} processes and the group {p.pid}")
        return f"SIGKILL to {len(pids)} processes (run.sh's tree and group {p.pid})"


__all__ = ["Runner", "RunnerError", "RunnerBusy", "InvalidRequest"]
