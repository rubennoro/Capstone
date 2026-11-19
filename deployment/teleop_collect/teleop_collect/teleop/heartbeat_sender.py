# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/heartbeat_sender.py @ 2ffe641
# changes: imports only
"""HeartbeatSender: the cycle's one C3 keep-alive (STRUCTURE.md 8.2; TOP 3.7.3 rule H1).

beat() sends exactly one request: a hello when one is due (its sample goes to the PC clock map, its round trip to
LinkStats), else a heartbeat. In start-up mode (start-up step 4, and after every home that did not conclude, until the
map is ready again) a hello is due every startup_hello_period_s; otherwise every hello_period_s. A hello whose t_pc_ns
is below the previous one means the PC rebooted (CLOCK_MONOTONIC survives a franka_control restart): PcRestarted.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from deploy_contracts import HelloReply

from teleop_collect.teleop.errors import PcRestarted
from teleop_collect.teleop.link_stats import LinkStats
from teleop_collect.teleop.pc_client import PcClient
from teleop_collect.teleop.pc_clock import HelloSample, PcClock

TOLERANCE_NS = 2_000_000       # a hello due "every cycle" stays due despite a cycle that started 1 ms early


class HeartbeatSender:
    def __init__(self, pc: PcClient, pc_clock: PcClock, stats: LinkStats, hello_period_s: float,
                 startup_hello_period_s: float, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._pc = pc
        self._pc_clock = pc_clock
        self._stats = stats
        self._hello_ns = int(hello_period_s * 1e9)
        self._startup_hello_ns = int(startup_hello_period_s * 1e9)
        self._clock = clock
        self._startup = False
        self._last_hello_ns: int | None = None
        self._last_reply: HelloReply | None = None
        self._last_sample: HelloSample | None = None
        self.last_request = ""

    def startup_mode(self, on: bool) -> None:
        self._startup = on

    def in_startup_mode(self) -> bool:
        return self._startup

    def hello_due(self, now_ns: int) -> bool:
        if self._last_hello_ns is None:
            return True
        period = self._startup_hello_ns if self._startup else self._hello_ns
        return now_ns - self._last_hello_ns >= period - TOLERANCE_NS

    def beat(self, now_ns: int) -> HelloReply | None:
        """One C3 request: a hello when due (returns its reply), else a heartbeat (returns None)."""
        if self.hello_due(now_ns):
            self.last_request = "c3.hello/v1"
            reply, sample = self._pc.hello()
            self._last_hello_ns = now_ns
            previous = self._last_sample
            if previous is not None and sample.t_pc_ns < previous.t_pc_ns:
                raise PcRestarted(f"hello t_pc_ns went back from {previous.t_pc_ns} to {sample.t_pc_ns}",
                                  rebooted=True)
            self._last_reply, self._last_sample = reply, sample
            self._pc_clock.add_hello(sample)
            return reply
        self.last_request = "c3.heartbeat/v1"
        self._pc.heartbeat()
        return None

    def last_hello(self) -> HelloReply | None:
        return self._last_reply

    def last_sample(self) -> HelloSample | None:
        return self._last_sample
