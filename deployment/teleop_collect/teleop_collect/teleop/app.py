# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/__main__.py @ 2ffe641 (_run)
# changes: teleop_collect STRUCTURE.md 6: builds the PC side as before (PcClient, PcClock, HeartbeatSender,
# PcListener, LinkStats, StartupChecks) plus the R1 FramesClient, E4's TaskStore, the TeleopLoop and its U1Server
# (bound before start-up, so a port in use fails at once); no policy, worker, operator or keyboard; the FATAL line names
# teleop; Ctrl-C and SIGTERM set the loop's flags; the loop's start-up writes the ready file through
# args.write_ready (ProcessArgs); a U1 port in use is its own FATAL line.
# The source's own header follows.
# adapted from rnoro/deployment:deployment/message_comms/message_comms/__main__.py @ 91b178b
# changes: the common arguments of policy_deploy.contracts.ProcessArgs; settings from Settings.load (station + link.yaml
# + frames.yaml); builds the PC side (PcClient, PcClock, HeartbeatSender, PcListener, LinkStats, StartupChecks); the
# FATAL line through print_fatal ("FATAL manager: ...").
"""run(args): teleop's process body, from ProcessArgs to the exit code (STRUCTURE.md 6)."""
from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import datetime

import zmq

from teleop_collect.contracts import ProcessArgs, StationError, print_fatal

NAME = "teleop"
log = logging.getLogger(NAME)


def _install_signals(loop) -> dict:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous = {signal.SIGINT: signal.getsignal(signal.SIGINT), signal.SIGTERM: signal.getsignal(signal.SIGTERM)}
    signal.signal(signal.SIGINT, lambda signum, frame: loop.interrupt())
    signal.signal(signal.SIGTERM, lambda signum, frame: loop.terminate())
    return previous


def run(args: ProcessArgs) -> int:
    from teleop_collect.teleop.episode.store import TaskStore
    from teleop_collect.teleop.failure_monitor import FailureMonitor
    from teleop_collect.teleop.frames_client import FramesClient
    from teleop_collect.teleop.heartbeat_sender import HeartbeatSender
    from teleop_collect.teleop.link_stats import LinkStats
    from teleop_collect.teleop.pc_client import PcClient
    from teleop_collect.teleop.pc_clock import make_pc_clock
    from teleop_collect.teleop.pc_listener import PcListener
    from teleop_collect.teleop.session_log import SessionLog
    from teleop_collect.teleop.settings import Settings
    from teleop_collect.teleop.startup_checks import StartupChecks
    from teleop_collect.teleop.state_history import StateHistory
    from teleop_collect.teleop.teleop_loop import TeleopLoop
    from teleop_collect.teleop.u1_server import U1Server

    try:
        settings = Settings.load(args.station, args.mock)
    except (StationError, ValueError) as exc:
        return print_fatal(NAME, f"config: {exc}", "check station.yaml teleop: and config/teleop.yaml")
    out_dir = args.out_dir(settings.session_root)
    log.info("settings from %s (mock: %s): C2 %s, C3 %s, R1 %s, U1 %s, clock %s; data %s; session folder %s",
             settings.station_file, settings.mock, settings.state_endpoint, settings.commands_endpoint,
             settings.r1_endpoint, settings.u1_endpoint, settings.clock.estimator, settings.data_root, out_dir)

    clock = time.monotonic_ns
    closers = []
    loop = None
    previous_signals: dict = {}
    code = 1
    try:
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
        frames = FramesClient(settings.r1_endpoint, settings.timeouts.health_s, settings.timeouts.frames_s)
        closers.append(frames.close)
        monitor = FailureMonitor(settings.state_lost_s)
        checks = StartupChecks(settings, mock=settings.mock)
        store = TaskStore(settings.data_root)
        loop = TeleopLoop(settings, pc, heartbeats, pc_clock, stats, frames, history, listener, monitor,
                          session_log, checks, store, clock, time.sleep)
        try:
            u1 = U1Server(settings.u1_endpoint, loop.dispatch)
        except zmq.ZMQError as exc:
            return print_fatal(NAME, f"U1 {settings.u1_endpoint}: cannot bind: {exc}",
                               "another teleop_collect run is using station.yaml ports.u1 (stop it, or change it)")
        closers.append(u1.close)
        loop.attach_u1(u1)
        previous_signals = _install_signals(loop)
        loop.startup(args.write_ready)
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
