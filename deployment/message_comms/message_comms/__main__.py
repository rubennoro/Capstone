"""`python -m message_comms --station FILE [--session DIR] [--ready-file FILE] [--mock]` (STRUCTURE.md section 10)."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from message_comms.episode_loop import EpisodeLoop
from message_comms.errors import ScriptError, SettingsError
from message_comms.failure_monitor import FailureMonitor
from message_comms.frames_client import FramesClient
from message_comms.franka_client import FrankaClient
from message_comms.keyboard_operator import KeyboardOperator
from message_comms.policy_client import PolicyClient
from message_comms.policy_step import PolicyStep
from message_comms.pose_converter import PoseConverter
from message_comms.scripted_operator import ScriptedOperator
from message_comms.session_log import SessionLog
from message_comms.settings import load_settings
from message_comms.state_history import StateHistory
from message_comms.state_listener import StateListener
from message_comms.step_worker import StepWorker

NAME = "message_comms"
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
log = logging.getLogger(NAME)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m message_comms", description="the deployment manager process")
    parser.add_argument("--station", required=True, type=Path, help="station.yaml")
    parser.add_argument("--session", type=Path, default=None, help="session folder (deploy run gives it)")
    parser.add_argument("--ready-file", type=Path, default=None, help="written once ready (deploy run gives it)")
    parser.add_argument("--mock", action="store_true", help="accepted; changes nothing for this process")
    return parser.parse_args(argv)


def fatal(what_failed: str, what_to_check: str) -> None:
    """The last stderr line on a failure exit (deploy/STRUCTURE.md 7.6)."""
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stderr.write(f"FATAL {NAME}: {what_failed}: {what_to_check}\n")
    sys.stderr.flush()


def _install_signals(loop: EpisodeLoop) -> dict:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous = {signal.SIGINT: signal.getsignal(signal.SIGINT), signal.SIGTERM: signal.getsignal(signal.SIGTERM)}
    signal.signal(signal.SIGINT, lambda signum, frame: loop.interrupt())
    signal.signal(signal.SIGTERM, lambda signum, frame: loop.terminate())
    return previous


def _close_worker_clients(worker: StepWorker, frames: FramesClient, policy: PolicyClient) -> None:
    """Only when the worker is idle: a worker stuck in infer still owns its sockets (left to the process exit)."""
    if not worker.busy():
        frames.close()
        policy.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)
    try:
        return _run(args)
    except Exception as exc:                 # anything unexpected (after _run's cleanup): traceback, then FATAL
        log.exception("unexpected error: %s", exc)
        fatal(f"internal error: {type(exc).__name__}: {exc}", "see the traceback in the message_comms log")
        return 1


def _run(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(args.station, args.session, args.mock)
    except SettingsError as exc:
        fatal(f"config: {exc}", "check station.yaml message_comms: and message_comms/config.yaml")
        return 1
    log.info("settings loaded from %s (mock: %s; --mock changes nothing here); session folder %s",
             settings.station_file, settings.mock, settings.session_dir)

    clock = time.monotonic_ns
    closers = []
    loop = None
    previous_signals: dict = {}
    try:
        try:
            operator = (ScriptedOperator.load(settings.operator_script, sys.stdout, clock,
                                              settings.operator_script_wait_s)
                        if settings.operator_script is not None else KeyboardOperator(sys.stdin, sys.stdout))
        except ScriptError as exc:
            fatal(f"operator script: {exc}", "check message_comms.operator_script")
            return 1
        try:
            session_log = SessionLog(settings.session_dir, datetime.now, clock)
        except OSError as exc:
            log.exception("session log: %s", exc)
            fatal(f"session folder {settings.session_dir}: cannot write: {exc}",
                  "check that the session folder (--session, or station.yaml session_root) is writable")
            return 1
        closers.append(session_log.close)
        history = StateHistory(settings.history_s)
        poll_s = settings.loop_ms / 1000
        listener = StateListener(settings.endpoints.state, history, settings.state_recv_hwm, poll_s)
        closers.append(listener.stop)
        franka = FrankaClient(settings.endpoints.commands, settings.timeouts, poll_s)
        closers.append(franka.close)
        health = FramesClient(settings.endpoints.frames, settings.timeouts.health_s)
        closers.append(health.close)
        frames = FramesClient(settings.endpoints.frames, settings.timeouts.frames_s)
        policy = PolicyClient(settings.endpoints.policy, settings.timeouts)
        converter = PoseConverter(settings.exec_h, settings.lookahead_ms)
        step = PolicyStep(frames, policy, history, converter, settings, clock)
        worker = StepWorker(step, policy, clock)
        closers.append(worker.stop)
        closers.append(lambda: _close_worker_clients(worker, frames, policy))
        monitor = FailureMonitor(settings)
        loop = EpisodeLoop(settings, franka, health, worker, history, listener, monitor, operator, session_log,
                           clock, time.sleep)
        closers.insert(0, operator.close)
        previous_signals = _install_signals(loop)
        loop.startup(args.ready_file)
        code = loop.run()
    finally:
        for close in closers:
            try:
                close()
            except Exception as exc:
                log.error("cleanup: %s", exc)
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    if code == 1 and loop is not None and loop.verdict is not None:
        fatal(loop.verdict.what_failed, loop.verdict.what_to_check)
    return code


if __name__ == "__main__":
    sys.exit(main())
