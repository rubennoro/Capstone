"""PolicyApp: start-up order, warm-up, ready file, serving, FATAL line (STRUCTURE.md 3, 4.11)."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from contracts import Observation

from policy.base_policy import Policy
from policy.errors import PolicyFatal, PolicyLoadError, SettingsError
from policy.neutral_observation import neutral_observation
from policy.policy_factory import build_policy
from policy.policy_server import PolicyServer
from policy.settings import Settings

NAME = "policy"
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
log = logging.getLogger("policy.app")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr, force=True)


def run_warm_up(policy: Policy, obs: Observation, count: int,
                clock: Callable[[], int] = time.monotonic_ns) -> list[float]:
    """Call policy.infer `count` times on `obs`, timing each (seconds), then policy.reset().

    Every failure is fatal, a contract failure of the chunk included (STRUCTURE.md 3.3): raises PolicyFatal.
    """
    times: list[float] = []
    for i in range(count):
        t0 = clock()
        try:
            policy.infer(obs)
        except Exception as e:
            log.error("warm-up inference %d/%d failed", i + 1, count, exc_info=e)
            raise PolicyFatal(f"warm-up inference failed: {type(e).__name__}: {e}",
                              "the model output is invalid or the device failed; check the checkpoint, "
                              "policy.device and the GPU") from e
        times.append((clock() - t0) / 1e9)
    try:
        policy.reset()
    except Exception as e:
        raise PolicyFatal(f"reset after warm-up failed: {type(e).__name__}: {e}", "see the log above") from e
    return times


def write_atomic(path: Path, text: str) -> None:
    """Write `<path>.tmp`, then rename it, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class PolicyApp:
    def __init__(self, station: Path, session: Path | None, ready_file: Path | None, mock: bool) -> None:
        self._station = station
        self._session = session
        self._ready_file = ready_file
        self._mock = mock
        self._stop = threading.Event()

    def run(self) -> int:
        """0: stopped normally; 1: FATAL line written."""
        setup_logging()
        try:
            return self._run()
        except KeyboardInterrupt:
            log.info("interrupted (Ctrl-C) before serving started")
            return 0
        except SettingsError as e:
            for problem in e.problems:
                log.error("settings: %s", problem)
            return _fatal("; ".join(e.problems))
        except (PolicyLoadError, PolicyFatal) as e:
            return _fatal(f"{e.what}: {e.check}")
        except Exception as e:
            log.exception("unexpected error")
            return _fatal(f"unexpected error: {type(e).__name__}: {e}: see the log above")

    def _run(self) -> int:
        settings = Settings.load(self._station, self._session, self._mock)
        settings.session_dir.mkdir(parents=True, exist_ok=True)
        log.info("kind %s%s; session folder %s", settings.active_kind, " (--mock)" if settings.mock else "",
                 settings.session_dir)

        t0 = time.monotonic()
        policy = build_policy(settings)
        log.info("policy %s built in %.2f s", settings.active_kind, time.monotonic() - t0)

        obs = neutral_observation(settings.warmup_gripper_width_m)
        times = run_warm_up(policy, obs, settings.warmup_inferences)
        device = policy.describe().get("device", "cpu")
        log.info("warm-up " + ", ".join(f"inference {i + 1}/{len(times)}: {t:.3f} s" for i, t in enumerate(times))
                 + f" on {device}")

        try:
            server = PolicyServer(policy, settings.endpoint, settings.reset_max_s)
        except Exception as e:
            raise PolicyFatal(f"cannot bind the C4 endpoint {settings.endpoint}: {type(e).__name__}: {e}",
                              "is another policy process running? check endpoints.policy in station.yaml") from e
        try:
            startup: dict[str, Any] = {
                "kind": settings.active_kind, "mock": settings.mock, "settings": settings.as_dict(),
                "policy": policy.describe(), "warmup_s": times, "endpoint": server.endpoint, "pid": os.getpid(),
            }
            write_atomic(settings.session_dir / "startup.json", json.dumps(startup, indent=1, default=str))
            if self._ready_file is not None:
                write_atomic(Path(self._ready_file), "")
            self._install_signal_handlers()
            log.info("ready: serving C4 on %s", server.endpoint)
            server.serve(self._stop, settings.poll_s)
            log.info("stopped by signal")
            return 0
        finally:
            server.close()

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, frame: object) -> None:
            self._stop.set()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


def _fatal(text: str) -> int:
    for h in logging.getLogger().handlers:
        h.flush()
    print(f"FATAL {NAME}: {text}", file=sys.stderr, flush=True)
    return 1
