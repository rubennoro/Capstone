"""Fake Runner, Live and Catalog for `python -m sim_deploy.ui --stub` (ui/DESIGN.md §2-§3 with fake data).

They implement the APIs server.py expects (its docstring) so the server and the page run with no sim, no policy and
no other part written yet. Nothing here touches the network or writes files.
"""
from __future__ import annotations

import math
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

import cv2                                             # opencv-python-headless, in the sim env

from sim_deploy.ui.settings import Settings

BUSY = ("starting", "running", "stopping")
CHANNELS = {1: ("external", 640, 480), 2: ("main", 640, 480), 3: ("ultrawide", 640, 480), 4: ("policy", 448, 448)}


# ---------------------------------------------------------------------------------------------------------- runner

class StubRunner:
    """A fake eval run: `starting` 1.5 s, then per rollout ~4 s of phases, a random outcome, then `done`.

    stop(): the first one ends after the current rollout's phase (`stopping` 1 s, then `done`); a second kills
    (`failed`, last_error "killed")."""

    ROLLOUT_PHASES = (("home", 0.8), ("place eraser", 0.5), ("settle", 0.4), ("rollout", 2.0), ("judge", 0.3))

    def __init__(self, settings: Settings | None = None) -> None:
        self._tail = settings.log_tail if settings is not None else 30
        self._lock = threading.Lock()
        self._events: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stops = 0
        self._n_runs = 0
        self._st: dict[str, Any] = self._idle()

    def _idle(self) -> dict[str, Any]:
        return {"state": "idle", "run_id": None, "request": None, "rollout": 0, "rollouts": 0, "phase": "",
                "elapsed_s": 0.0, "outcomes": [], "last_error": None, "log_tail": []}

    # API
    def start(self, req: dict) -> str:
        with self._lock:
            if self._st["state"] in BUSY:
                raise RuntimeError(f"a run is already {self._st['state']}")
            self._n_runs += 1
            run_id = time.strftime("%Y-%m-%d_%H%M%S") + f"_stub{self._n_runs}"
            self._stops = 0
            self._st = self._idle() | {"state": "starting", "run_id": run_id, "request": dict(req),
                                       "rollouts": int(req.get("rollouts", 1)), "phase": "run.sh"}
            self._t0 = time.monotonic()
        self._log(f"[stub] run.sh --policy {req.get('target', req.get('policy'))} --rollouts {req.get('rollouts')}")
        self._thread = threading.Thread(target=self._run, name="stub-runner", daemon=True)
        self._thread.start()
        return run_id

    def stop(self) -> None:
        with self._lock:
            if self._st["state"] not in BUSY:
                return
            self._stops += 1
            if self._stops == 1:
                self._st["state"] = "stopping"
        self._log("[stub] stop requested" if self._stops == 1 else "[stub] second stop: kill")

    def status(self) -> dict:
        with self._lock:
            st = dict(self._st)
            if st["state"] in BUSY:
                st["elapsed_s"] = round(time.monotonic() - self._t0, 2)
            st["outcomes"] = list(st["outcomes"])
            st["log_tail"] = list(st["log_tail"])
            return st

    def events(self) -> queue.Queue:
        return self._events

    def session_dir(self) -> Path | None:
        with self._lock:
            if self._st["state"] in ("running", "stopping"):
                return Path("/nonexistent/stub_session") / str(self._st["run_id"])
        return None

    # the fake run
    def _log(self, line: str) -> None:
        with self._lock:
            self._st["log_tail"] = (self._st["log_tail"] + [line])[-self._tail:]

    def _set(self, **kv: Any) -> None:
        with self._lock:
            self._st.update(kv)

    def _event(self, kind: str, **kv: Any) -> None:
        self._events.put({"kind": kind, "run_id": self._st["run_id"], "t": time.time(), **kv})

    def _sleep(self, s: float) -> bool:
        """False when killed (second stop)."""
        end = time.monotonic() + s
        while time.monotonic() < end:
            if self._stops >= 2:
                return False
            time.sleep(0.05)
        return True

    def _run(self) -> None:
        rng = random.Random(int(self._st["request"].get("seed", 0)))
        n_rollouts = self._st["rollouts"]
        if not self._sleep(1.5):
            return self._killed()
        if self._stops:
            return self._finish()
        self._set(state="running")
        self._log("[stub] sim PC up, sim Pi streaming, policy_deploy [idle]")
        for n in range(1, n_rollouts + 1):
            self._set(rollout=n)
            self._event("rollout_start", rollout=n, rollouts=n_rollouts)
            self._log(f"[stub] rollout {n}/{n_rollouts} started")
            for phase, dur in self.ROLLOUT_PHASES:
                self._set(phase=phase)
                if not self._sleep(dur):
                    return self._killed()
            outcome = rng.choice(["success", "success", "timeout", "dropped"])
            with self._lock:
                self._st["outcomes"] = self._st["outcomes"] + [outcome]
            self._event("rollout_end", rollout=n, outcome={"n": n, "outcome": outcome, "success": outcome == "success",
                                                            "grasped": outcome != "timeout", "max_lift_m": 0.07})
            self._log(f"[stub] rollout {n}: {outcome}")
            if self._stops:
                break
        self._finish()

    def _finish(self) -> None:
        if self._stops:
            self._set(state="stopping", phase="s, q")
            self._sleep(1.0)
            if self._stops >= 2:
                return self._killed()
        outs = self._st["outcomes"]
        self._set(state="done", phase="")
        self._log(f"[stub] {outs.count('success')}/{len(outs)} successes")
        self._event("run_end", state="done", successes=outs.count("success"), rollouts=len(outs),
                    eval_dir=None)

    def _killed(self) -> None:
        self._set(state="failed", phase="", last_error="killed")
        self._log("[stub] killed")
        self._event("run_end", state="failed", error="killed")


# ------------------------------------------------------------------------------------------------------------ live

class StubLive:
    """Fake JPEGs on the four channels at 10 fps (a moving dot and the channel name) and fake telemetry while a
    session folder is set (set_session_dir: the runner's session_dir(), wired by the server)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._fps = settings.frame_fps if settings is not None else 10.0
        self._lock = threading.Lock()
        self._frames: dict[int, tuple[int, bytes]] = {}
        self._session: Path | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.monotonic()

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="stub-live", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest_frames(self) -> dict[int, tuple[int, bytes]]:
        with self._lock:
            return dict(self._frames)

    def set_session_dir(self, path: Path | None) -> None:
        with self._lock:
            self._session = None if path is None else Path(path)

    def telemetry(self) -> dict | None:
        with self._lock:
            if self._session is None:
                return None
        t = time.monotonic() - self._t0
        lift = max(0.0, math.sin(t * 0.8)) * 0.06
        return {"t_ns": time.time_ns(), "mode": "idle" if int(t) % 10 < 2 else "move",
                "tcp": [0.45 + 0.05 * math.sin(t), 0.02 * math.cos(t), 0.15 + lift],
                "width_m": 0.02 + 0.018 * math.sin(t * 1.3),   # within limits.yaml 0..0.04 (W0)
                "eraser": {"pose": [0.5, 0.0, 0.012 + lift, 1.0, 0.0, 0.0, 0.0], "height_m": 0.012 + lift},
                "contacts": {"finger_eraser": lift > 0.01, "table": False},
                "policy": {"infer_ms": 38.0 + 5 * math.sin(t * 3), "margin_ms": 60.0, "skipped": 0}}

    def _loop(self) -> None:
        seq = 0
        period = 1.0 / self._fps
        while not self._stop.is_set():
            started = time.monotonic()
            seq += 1
            with self._lock:
                running = self._session is not None
            frames = {ch: (seq, _fake_jpeg(name, w, h, seq, running)) for ch, (name, w, h) in CHANNELS.items()}
            with self._lock:
                self._frames = frames
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))


def _fake_jpeg(name: str, w: int, h: int, seq: int, running: bool) -> bytes:
    img = np.full((h, w, 3), (40, 34, 30) if not running else (30, 50, 30), np.uint8)
    t = seq / 10.0
    cx, cy = int(w / 2 + w / 3 * math.cos(t)), int(h / 2 + h / 3 * math.sin(t * 1.3))
    cv2.circle(img, (cx, cy), max(6, w // 30), (0, 200, 255), -1)
    if name == "policy":
        for x in (0, w // 2):
            cv2.line(img, (x, 0), (x, h), (90, 90, 90), 1)
        cv2.line(img, (0, h // 2), (w, h // 2), (90, 90, 90), 1)
    cv2.putText(img, f"STUB {name}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
    cv2.putText(img, f"#{seq} {'running' if running else 'idle'}", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else b""


# --------------------------------------------------------------------------------------------------------- catalog

class StubCatalog:
    """Fake policies, replays and eval rows (§2 shapes)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._root = (settings.policy_data_root if settings is not None else Path.home() / "Capstone" / "data")

    def policies(self) -> dict:
        def pol(task: str, run: str, kind: str, eps: int | None, mse: float | None, ckpt: bool, label: str,
                last: dict | None = None) -> dict:
            folder = self._root / task / "policy" / run
            return {"id": f"{task}/{run}", "task": task, "run_folder": str(folder), "kind": kind,
                    "train_config": {"prompting": "umi_policy_dunetp", "plain": "umi_policy_dunet"}.get(kind),
                    "dataset": f"pick1-v1-ep{eps}" if eps else None, "episodes": eps,
                    "checkpoint": str(folder / "checkpoints" / "latest.ckpt") if ckpt else None,
                    "val_mse": mse, "label": label, "last_eval": last}

        last = self.evals()["evals"][0]
        return {
            "policies": [
                pol("PICK_TEST1", "run_006_2026-11-18", "prompting", 30, 0.00020, True, "prompting, 30 eps", last),
                pol("PICK_TEST1", "run_002_2026-11-17", "prompting", 10, 0.00084, True, "prompting, 10 eps"),
                pol("PICK_TEST1", "run_022_2026-11-19", "plain", 10, None, False, "plain, 10 eps (not fetched)"),
                pol("PICK_DOUBLE", "run_001_2026-11-19", "plain", 62, None, False, "plain, 62 eps (not fetched)"),
                pol("PICK_DOUBLE", "run_999_stub", "unknown", None, None, True, "stub: unknown kind"),
            ],
            "replays": [
                {"id": f"replay:{Path.home()}/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip#{i}",
                 "dataset": "pick1-v1", "index": i, "grasp_ok": i != 1} for i in range(3)
            ],
            "builtins": ["hold_still"],
        }

    def evals(self) -> dict:
        row = {"date": "2026-11-20T00:46:00", "task": "PICK_TEST1", "run_folder": "run_006_2026-11-18",
               "policy": "PICK_TEST1/run_006_2026-11-18", "kind": "prompting", "dataset": "pick1-v1",
               "rollouts": "5", "seed": "0", "tag": "stub", "successes": "3", "success_rate": "0.600",
               "ci95_lo": "0.231", "ci95_hi": "0.882", "grasp_rate": "0.800", "lift_rate": "0.600", "faults": "0",
               "stops": "0", "timeouts": "2", "eval_dir": "PICK_TEST1/run_006_2026-11-18/2026-11-20_0046_stub"}
        base = "/evals/" + row["eval_dir"]
        return {"evals": [row | {"summary_url": base + "/summary.html", "folder_url": base + "/"}]}
