"""LinkEmulator: deterministic "TCP over Wi-Fi" for the sim/replay senders.

Why: a wireless camera link does not drop frames (TCP) but it delays them: base latency, jitter, bandwidth backlog and
stalls (roaming, airtime contention) that hold back everything queued behind them (head-of-line blocking). The sender's
clock also differs from the host's (offset + drift), which matters for the stamped v2 protocol. This module reproduces
that, deterministically for a seed, so the receiver's de-jitter / ClockMap / health logic can be tested.

Rule (per item i, in send order; `schedule()` and the live emulator share one state machine, `_LinkState`):
    tx_end_i    = max(bucket_free, send_i) + size_i / bw          (bucket_free = previous tx_end; bw = inf -> send_i)
    base_i      = max(deliver_{i-1}, send_i + latency + jitter_i, tx_end_i)
    deliver_i   = base_i, pushed to the end of any stall event that is in progress at that time (head-of-line)
- jitter_i ~ Gamma(k=2, mean=jitter_s) (always >= 0, skewed like real Wi-Fi), own RNG stream (seed).
- Stall events form a Poisson process (rate stall_rate_hz) on the time axis measured from the first send, duration
  uniform(stall_duration_s), own RNG stream (seed+1). Because deliver is monotone, a stall delays every later item.
- Delivery is in order and nothing is dropped. All times are seconds on the sender host's monotonic clock.

Clock skew: sender_clock_ns(now) = now + offset + drift_ppm*1e-6*(now - t0), t0 = emulator creation.

Usage:
    cfg = LinkConfig.from_yaml("run/link/wifi_good.yaml")            # or a dict / sender_config()["link"]
    t = schedule(cfg, send_t, sizes)                                  # pure
    link = LinkEmulator(cfg, deliver_fn); link.submit(item, nbytes); ...; link.close()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from ..paths import REPO_ROOT

CONFIG_ROOT = REPO_ROOT / "controlSW" / "configs"


@dataclass
class LinkConfig:
    latency_s: float = 0.0
    jitter_s: float = 0.0
    stall_rate_hz: float = 0.0
    stall_duration_s: tuple = (0.05, 0.25)
    bandwidth_mbit_s: float = 0.0     # 0 = unlimited
    clock_offset_s: float = 0.0
    clock_drift_ppm: float = 0.0
    seed: int = 0

    def __post_init__(self):
        lo, hi = self.stall_duration_s
        self.stall_duration_s = (float(lo), float(hi))
        for name in ("latency_s", "jitter_s", "stall_rate_hz", "bandwidth_mbit_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"LinkConfig.{name} must be >= 0")
        if not 0 <= lo <= hi:
            raise ValueError(f"LinkConfig.stall_duration_s must satisfy 0 <= lo <= hi, got {self.stall_duration_s}")

    @classmethod
    def from_yaml(cls, path_or_dict) -> "LinkConfig":
        """Build from a run/link/*.yaml path (absolute, relative, or relative to controlSW/configs) or its dict."""
        if path_or_dict is None:
            return cls()
        if isinstance(path_or_dict, LinkConfig):
            return path_or_dict
        if isinstance(path_or_dict, (str, Path)):
            p = Path(path_or_dict)
            if not p.is_file() and (CONFIG_ROOT / p).is_file():
                p = CONFIG_ROOT / p
            if not p.is_file() and (CONFIG_ROOT / p.with_suffix(".yaml")).is_file():
                p = CONFIG_ROOT / p.with_suffix(".yaml")
            d = yaml.safe_load(p.read_text())
        else:
            d = dict(path_or_dict)
        if d.get("jitter_dist", "gamma") != "gamma":
            raise ValueError(f"jitter_dist {d['jitter_dist']!r} not supported (only gamma)")
        stall = d.get("stall") or {}
        clock = d.get("clock") or {}
        return cls(latency_s=float(d.get("latency_s", 0.0)), jitter_s=float(d.get("jitter_s", 0.0)),
                   stall_rate_hz=float(stall.get("rate_hz", 0.0)),
                   stall_duration_s=tuple(stall.get("duration_s", (0.05, 0.25))),
                   bandwidth_mbit_s=float(d.get("bandwidth_mbit_s", 0.0)),
                   clock_offset_s=float(clock.get("offset_s", 0.0)), clock_drift_ppm=float(clock.get("drift_ppm", 0.0)),
                   seed=int(d.get("seed", 0)))


class _LinkState:
    """Incremental form of the delivery rule (one call per item, in send order)."""

    def __init__(self, cfg: LinkConfig):
        self.cfg = cfg
        self.rng_j = np.random.default_rng(cfg.seed)
        self.rng_s = np.random.default_rng(cfg.seed + 1)
        self.prev = -np.inf
        self.bucket_free = -np.inf
        self.t0 = None
        self.next_stall = None  # (start offset from t0, duration)

    def _draw_stall(self, after):
        c = self.cfg
        gap = self.rng_s.exponential(1.0 / c.stall_rate_hz)
        dur = self.rng_s.uniform(*c.stall_duration_s)
        return (after + gap, dur)

    def deliver_time(self, send_t: float, size_bytes: float) -> float:
        c = self.cfg
        if self.t0 is None:
            self.t0 = send_t
            if c.stall_rate_hz > 0:
                self.next_stall = self._draw_stall(0.0)
        jit = self.rng_j.gamma(2.0, c.jitter_s / 2.0) if c.jitter_s > 0 else 0.0
        if c.bandwidth_mbit_s > 0:
            tx_end = max(self.bucket_free, send_t) + size_bytes * 8.0 / (c.bandwidth_mbit_s * 1e6)
            self.bucket_free = tx_end
        else:
            tx_end = send_t
        base = max(self.prev, send_t + c.latency_s + jit, tx_end)
        d = base
        while self.next_stall is not None and self.t0 + self.next_stall[0] <= d:
            start, dur = self.next_stall
            d = max(d, self.t0 + start + dur)       # held until the stall ends (no-op if it already ended)
            self.next_stall = self._draw_stall(start + dur)
        self.prev = d
        return self.prev


def schedule(cfg: LinkConfig, send_t: np.ndarray, sizes_bytes: np.ndarray) -> np.ndarray:
    """Deterministic delivery times for items sent at send_t (non-decreasing) with the given sizes."""
    send_t = np.asarray(send_t, dtype=np.float64)
    sizes = np.broadcast_to(np.asarray(sizes_bytes, dtype=np.float64), send_t.shape)
    if np.any(np.diff(send_t) < 0):
        raise ValueError("send_t must be non-decreasing")
    st = _LinkState(cfg)
    return np.array([st.deliver_time(float(t), float(s)) for t, s in zip(send_t, sizes)], dtype=np.float64)


class LinkEmulator:
    """Live emulator: submit() never blocks; a thread calls deliver_fn(item) at the scheduled monotonic time."""

    SPIN_S = 0.0008  # sleep until this close to the target, then yield-spin (sub-ms accuracy)

    def __init__(self, cfg: LinkConfig, deliver_fn: Callable[[Any], None]):
        self.cfg = cfg
        self._fn = deliver_fn
        self._state = _LinkState(cfg)
        self._q: deque = deque()
        self._cv = threading.Condition()
        self._closed = False
        self._t0_ns = time.monotonic_ns()
        self._submitted = 0
        self._delivered = 0
        self._bytes = 0
        self._errors = 0
        self._last_deliver_t = 0.0
        self.log: list[tuple[float, float]] = []   # (scheduled, actual) per delivered item, for tests/diagnostics
        self._th = threading.Thread(target=self._run, name="link-emulator", daemon=True)
        self._th.start()

    # ---- clock
    def to_sender_ns(self, t_mono_s: float) -> int:
        """Map a host-monotonic time (s) to this link's sender clock (ns)."""
        c = self.cfg
        t_ns = int(round(t_mono_s * 1e9))
        return t_ns + int(round(c.clock_offset_s * 1e9 + c.clock_drift_ppm * 1e-6 * (t_ns - self._t0_ns)))

    def sender_clock_ns(self, now_ns: int | None = None) -> int:
        """monotonic_ns + offset + drift*(t - t0); now_ns is injectable for tests."""
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        return self.to_sender_ns(now_ns / 1e9)

    # ---- data path
    def submit(self, item, size_bytes: int, t_send: float | None = None) -> None:
        t_send = time.monotonic() if t_send is None else t_send
        with self._cv:
            if self._closed:
                return
            t = self._state.deliver_time(t_send, size_bytes)
            self._q.append((t, item))
            self._submitted += 1
            self._bytes += int(size_bytes)
            self._last_deliver_t = t
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while not self._q and not self._closed:
                    self._cv.wait()
                if not self._q:
                    return
                if self._closed and self._drop:
                    return
                t, item = self._q[0]
            now = time.monotonic()
            while now < t:
                gap = t - now
                if gap > self.SPIN_S:
                    with self._cv:
                        if self._closed and self._drop:
                            return
                        self._cv.wait(gap - self.SPIN_S)
                else:
                    time.sleep(0)
                now = time.monotonic()
            with self._cv:
                if self._closed and self._drop:
                    return
                self._q.popleft()
            try:
                self._fn(item)
            except Exception:  # a failing receiver must not kill the link
                self._errors += 1
            actual = time.monotonic()
            with self._cv:
                self._delivered += 1
                self.log.append((t, actual))
                self._cv.notify_all()

    _drop = False

    def stats(self) -> dict:
        with self._cv:
            return {"submitted": self._submitted, "delivered": self._delivered, "queued": len(self._q),
                    "bytes": self._bytes, "errors": self._errors,
                    "backlog_s": max(0.0, self._last_deliver_t - time.monotonic()) if self._q else 0.0}

    def flush(self, timeout_s: float = 10.0) -> bool:
        """Wait until every submitted item has been delivered."""
        end = time.monotonic() + timeout_s
        with self._cv:
            while self._q or self._delivered < self._submitted:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(min(left, 0.05))
        return True

    def close(self, drain: bool = False, timeout_s: float = 10.0) -> None:
        """Stop the thread. drain=True delivers what is queued first; otherwise queued items are discarded."""
        if drain:
            self.flush(timeout_s)
        with self._cv:
            self._closed = True
            self._drop = True
            self._cv.notify_all()
        self._th.join(timeout=2.0)
