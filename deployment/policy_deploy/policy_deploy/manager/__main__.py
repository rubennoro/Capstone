# adapted from rnoro/deployment:deployment/message_comms/message_comms/__main__.py @ 91b178b
# changes: the common arguments of policy_deploy.contracts.ProcessArgs; settings from Settings.load (station + link.yaml
# + frames.yaml); builds the PC side (PcClient, PcClock, HeartbeatSender, PcListener, LinkStats, StartupChecks); the
# FATAL line through print_fatal ("FATAL manager: ...").
"""python -m policy_deploy.manager [--station FILE] [--session DIR] [--ready-file FILE] [--mock] (STRUCTURE.md 6).

--mock: the PC is on 127.0.0.1 (the fake PC, or franka_control --mock --local in the cross test), with the link.yaml
ports, and hello must say mock.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime

from policy_deploy.contracts import StationError
from policy_deploy.contracts.process_args import ProcessArgs, print_fatal, setup_logging

NAME = "manager"
log = logging.getLogger(NAME)


def _install_signals(loop) -> dict:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous = {signal.SIGINT: signal.getsignal(signal.SIGINT), signal.SIGTERM: signal.getsignal(signal.SIGTERM)}
    signal.signal(signal.SIGINT, lambda signum, frame: loop.interrupt())
    signal.signal(signal.SIGTERM, lambda signum, frame: loop.terminate())
    return previous


def main(argv: Sequence[str] | None = None) -> int:
    args, _ = ProcessArgs.parse(NAME, argv, description="the manager: episode loop, keyboard, C2 in, C3 out")
    root = logging.getLogger()
    if not root.handlers:
        setup_logging()
    try:
        return _run(args)
    except Exception as exc:                 # anything unexpected (after _run's cleanup): traceback, then FATAL
        log.exception("unexpected error: %s", exc)
        return print_fatal(NAME, f"internal error: {type(exc).__name__}: {exc}", "see the traceback in the manager log")


def _run(args: ProcessArgs) -> int:
    from policy_deploy.manager.episode_loop import EpisodeLoop
    from policy_deploy.manager.errors import ScriptError
    from policy_deploy.manager.failure_monitor import FailureMonitor
    from policy_deploy.manager.frames_client import FramesClient
    from policy_deploy.manager.heartbeat_sender import HeartbeatSender
    from policy_deploy.manager.keyboard_operator import KeyboardOperator
    from policy_deploy.manager.link_stats import LinkStats
    from policy_deploy.manager.pc_client import PcClient
    from policy_deploy.manager.pc_clock import make_pc_clock
    from policy_deploy.manager.pc_listener import PcListener
    from policy_deploy.manager.policy_client import PolicyClient
    from policy_deploy.manager.policy_step import PolicyStep
    from policy_deploy.manager.pose_converter import PoseConverter
    from policy_deploy.manager.scripted_operator import ScriptedOperator
    from policy_deploy.manager.session_log import SessionLog
    from policy_deploy.manager.settings import Settings
    from policy_deploy.manager.startup_checks import StartupChecks
    from policy_deploy.manager.state_history import StateHistory
    from policy_deploy.manager.step_worker import StepWorker

    try:
        settings = Settings.load(args.station, args.mock)
    except (StationError, ValueError) as exc:
        return print_fatal(NAME, f"config: {exc}", "check station.yaml manager: and config/manager.yaml")
    out_dir = args.out_dir(settings.session_root)
    log.info("settings from %s (mock: %s): C2 %s, C3 %s, clock %s; session folder %s", settings.station_file,
             settings.mock, settings.state_endpoint, settings.commands_endpoint, settings.clock.estimator, out_dir)

    clock = time.monotonic_ns
    closers = []
    loop = None
    previous_signals: dict = {}
    code = 1
    try:
        try:
            operator = (ScriptedOperator.load(settings.operator_script, sys.stdout, clock,
                                              settings.operator_script_wait_s)
                        if settings.operator_script is not None else KeyboardOperator(sys.stdin, sys.stdout))
        except ScriptError as exc:
            return print_fatal(NAME, f"operator script: {exc}", "check manager.operator_script")
        try:
            session_log = SessionLog(out_dir, datetime.now, clock)
        except OSError as exc:
            return print_fatal(NAME, f"session folder {out_dir}: cannot write: {exc}",
                               "check that the session folder (--session, or station.yaml session_root) is writable")
        closers.append(session_log.close)
        stats = LinkStats(settings.link_stats_window_s, clock)
        pc_clock = make_pc_clock(settings.clock.estimator, settings.clock, clock)
        history = StateHistory(settings.history_s)
        listener = PcListener(settings.state_endpoint, history, pc_clock, stats, settings.state_recv_hwm,
                              settings.listener_poll_s, clock)
        closers.append(listener.stop)
        pc = PcClient(settings.commands_endpoint, settings.c3_timeout_s, settings.timeouts.home_s,
                      settings.heartbeat_period_s, stats, clock)
        closers.append(pc.close)
        heartbeats = HeartbeatSender(pc, pc_clock, stats, settings.clock.hello_period_s,
                                     settings.clock.startup_hello_period_s, clock)
        health = FramesClient(settings.frames_endpoint, settings.timeouts.health_s)
        closers.append(health.close)
        frames = FramesClient(settings.frames_endpoint, settings.timeouts.frames_s)
        policy = PolicyClient(settings.policy_endpoint, settings.timeouts)
        converter = PoseConverter(settings.exec_h, settings.lookahead_ms)
        step = PolicyStep(frames, policy, history, converter, settings, clock, pc_clock)
        worker = StepWorker(step, policy, clock)
        closers.append(worker.stop)

        def close_worker_clients() -> None:
            """Only when the worker is idle: a worker stuck in infer still owns its sockets."""
            if not worker.busy():
                frames.close()
                policy.close()
        closers.append(close_worker_clients)
        monitor = FailureMonitor(settings.state_lost_s)
        checks = StartupChecks(settings, mock=settings.mock)
        loop = EpisodeLoop(settings, pc, heartbeats, pc_clock, stats, health, worker, history, listener, monitor,
                           operator, session_log, checks, clock, time.sleep)
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
        return print_fatal(NAME, loop.verdict.what_failed, loop.verdict.what_to_check)
    return code


if __name__ == "__main__":
    sys.exit(main())
