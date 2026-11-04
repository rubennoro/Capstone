"""ClockMap: map each received camera frame to its capture time on THIS host's time.monotonic() clock, exactly once.

Why: frames reach the policy host over Wi-Fi/TCP with variable delay, and the policy needs the capture time (it pairs
images with robot poses). The iPhone app (v1) sends no stamps, so the time must come from the arrival time; the rig
and sim senders (v2) stamp frames on their own clock, which must be translated. The receiver (wireless_camera.py)
calls `map()` once per frame and passes BaseCamera `camera_latency_s = 0`, so latency is subtracted here and nowhere
else. Pure (numpy + stdlib), so it is unit-tested without sockets.

Modes (`mode`):
- `arrival_minus_latency`: t = t_arrival - latency[stream]. The iPhone default; network jitter goes straight into the
  stamp.
- `arrival_dejitter`: frames are captured on a fixed grid a + k/fps, and network delay is >= 0, so the LOWER ENVELOPE
  of arrival times over the last `window_s` is the best estimate of the grid (it absorbs jitter, not the constant
  latency, which `latency[stream]` still removes). Per stream, T = 1/fps (nominal; ppm drift is < 0.1 ms per window):
    k = k_prev + 1, or + 1 + n once 3 consecutive frames arrive >= n periods late at a normal cadence (sender drops)
    a = min over the window of (t_arrival_i - k_i*T)      (refit on every sample)
    t = min(a + k*T, t_arrival) - latency[stream], and >= t_arrival - 1.4 T - latency while a drop is unconfirmed
  Why "assume delay first": Wi-Fi jitter is often close to a frame period, so rounding each arrival onto the grid
  mislabels late frames as drops (tried: p95 no better than plain arrival on wifi_busy). TCP delivers in order and
  never drops, so a late frame is almost always a delayed one; true sender drops show up as a PERSISTENT residual of
  n*T and are confirmed (re-indexed) on the 3rd such frame. Until then a + k*T is n*T early, so a frame that is
  >= 0.9 T late after a long gap (>= 1.5 T), and the late frames that follow it at normal cadence (>= 0.5 T), are
  bounded below by t_arrival - 1.4 T: the unconfirmed frames after an n-frame drop are stamped at most 1.4 periods
  early (and never later than their own jitter), whatever n is. The bound is 1.4 T, not T, because a frame delayed
  by jitter looks the same at that moment; with T it would be stamped (delay - T) late (wifi_busy p95 2.5 -> 9 ms). A drained backlog arrives at short gaps, so it is
  neither bounded nor mistaken for drops. An index that ends up too high is harmless: the next on-time frame has a
  negative residual and the min re-anchors `a` at once (the index origin is arbitrary). A gap longer than `window_s`
  (sender restart) restarts the envelope.
- `sender_stamp_ping`: v2 frames carry sender-clock stamps; the receiver pings the sender (NTP style:
  host t0 -> sender t1, t2 -> host t3). theta = ((t1-t0)+(t2-t3))/2 is (sender - host) and its error is bounded by
  half the RTT, so only the lowest-RTT 30 % of the pings in `window_s` are used (robust to asymmetric/queued RTT),
  and theta(t) is a Theil-Sen line through them (median pairwise slope = drift, median intercept), evaluated as a
  line: no single noisy sample can tilt or shift it (the earlier min-RTT anchor + least-squares slope gave +-90 ppm
  on Wi-Fi-like RTT). Fewer than 6 pings: the single minimum-RTT sample, no drift. Window: with exponential RTT
  jitter of 4 ms per direction at 2 Hz, steady-state |theta error| p95 is 0.8-1.6 ms at 20 s, 0.7-1.0 ms at 30 s and
  0.2-0.8 ms at 60 s (8 seeds), so 60 s is the default; a sensor yaml can still set ping_window_s.
    t = t_sender - theta(t_arrival) - residual[stream]
- `sender_stamp_realtime`: the sender stamps CLOCK_REALTIME and both hosts run chrony (which disciplines REALTIME but
  not MONOTONIC, design_track_b.md finding 6): t = t_sender_rt - (time.time_ns() - time.monotonic_ns())*1e-9 - residual.

Every mode's output is clamped so each stream is non-decreasing (BaseCameraClient asserts increasing timestamps,
base_camera.py:393-395); clamps are counted.

Usage:
    cm = ClockMap("arrival_dejitter", {"main": 0.08, "ultrawide": 0.08}, fps_hz=30.0, window_s=2.0)
    t_capture = cm.map("main", time.monotonic())
    cm = ClockMap("sender_stamp_ping", {}, window_s=20.0, residual_s={"main": 0.0, "ultrawide": 0.0})
    cm.add_ping(t0_ns, t1_ns, t2_ns, t3_ns); t_capture = cm.map("main", t_arrival, t_sender_ns=pkt["t_main_ns"])
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable

import numpy as np

MODES = ("arrival_minus_latency", "arrival_dejitter", "sender_stamp_ping", "sender_stamp_realtime")
_EPS_K = 0.1            # tolerance (fraction of a period) when counting dropped periods
_DROP_BOUND_T = 1.4     # unconfirmed-drop frames are stamped >= t_arrival - 1.4 T (|error| <= 1.4 T); 1.4 rather than
                        # 1.0 so jitter-delayed frames (also "late after a gap") are rarely pulled late (p95 9 -> ~3 ms)
_DROP_CONFIRM = 3       # consecutive late frames at normal cadence needed to conclude the sender dropped frames
_PING_RTT_QUANTILE = 0.3  # pings used for the offset/drift line: the lowest-RTT 30 % (simulated best of 0.1-0.5)
_MAX_DRIFT_PPM = 500.0  # a fitted drift beyond this is noise (crystals are < 100 ppm): fall back to offset only


class ClockMap:
    def __init__(self, mode: str, latency_s: dict, *, fps_hz: float | dict | None = None, window_s: float = 2.0,
                 residual_s: dict | None = None, rt_minus_mono_ns: Callable[[], int] | None = None):
        """latency_s: {stream: s} for the arrival modes; residual_s: {stream: s} for the sender modes (missing = 0).
        fps_hz: physical frame rate (float, or {stream: hz}) for arrival_dejitter. window_s: dejitter envelope window,
        or the ping window for sender_stamp_ping. rt_minus_mono_ns: test hook for sender_stamp_realtime."""
        if mode not in MODES:
            raise ValueError(f"clock mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.latency_s = {k: float(v) for k, v in (latency_s or {}).items()}
        self.residual_s = {k: float(v) for k, v in (residual_s or {}).items()}
        self.window_s = float(window_s)
        if self.window_s <= 0:
            raise ValueError("window_s must be > 0")
        if mode == "arrival_dejitter":
            if fps_hz is None:
                raise ValueError("arrival_dejitter needs fps_hz")
        self.fps_hz = fps_hz
        self._rt_minus_mono_ns = rt_minus_mono_ns or (lambda: time.time_ns() - time.monotonic_ns())
        self._last: dict[str, float] = {}
        self.monotonic_clamps = 0
        self.n_mapped = 0
        self.n_unsynced = 0
        # dejitter state per stream: deque[(k, t_arrival)], k_prev, a
        self._dj: dict[str, dict] = {}
        self.envelope_restarts = 0
        self.dropped_frames = 0
        # ping state: deque[(t_mid_host_s, theta_s, rtt_s)]
        self._pings: deque = deque()
        self.n_pings = 0
        self._fit: tuple[float, float, float] | None = None   # (t_ref_s, theta_ref_s, drift) theta(t) = ref + d*(t-t_ref)

    # ------------------------------------------------------------------------------------------------ helpers
    def _fps(self, stream: str) -> float:
        f = self.fps_hz.get(stream) if isinstance(self.fps_hz, dict) else self.fps_hz
        if not f or f <= 0:
            raise ValueError(f"arrival_dejitter: no fps for stream {stream!r}")
        return float(f)

    def _clamp(self, stream: str, t: float) -> float:
        last = self._last.get(stream)
        if last is not None and t < last:
            self.monotonic_clamps += 1
            t = last
        self._last[stream] = t
        self.n_mapped += 1
        return t

    @property
    def synced(self) -> bool:
        """False only for sender_stamp_ping before the first ping (map() then falls back to arrival time)."""
        return self.mode != "sender_stamp_ping" or self._fit is not None

    # ------------------------------------------------------------------------------------------------ map
    def map(self, stream: str, t_arrival: float, t_sender_ns: int | None = None, seq: int | None = None) -> float:
        """Capture time (host monotonic s) of a frame that arrived at t_arrival (host monotonic s)."""
        m = self.mode
        if m == "arrival_minus_latency":
            t = t_arrival - self.latency_s.get(stream, 0.0)
        elif m == "arrival_dejitter":
            t = self._dejitter(stream, t_arrival) - self.latency_s.get(stream, 0.0)
        elif m == "sender_stamp_ping":
            if t_sender_ns is None:
                raise ValueError("sender_stamp_ping needs t_sender_ns")
            theta = self.theta(t_arrival)
            if theta is None:
                self.n_unsynced += 1
                t = t_arrival - self.residual_s.get(stream, 0.0)
            else:
                t = t_sender_ns * 1e-9 - theta - self.residual_s.get(stream, 0.0)
        else:  # sender_stamp_realtime
            if t_sender_ns is None:
                raise ValueError("sender_stamp_realtime needs t_sender_ns")
            t = (t_sender_ns - self._rt_minus_mono_ns()) * 1e-9 - self.residual_s.get(stream, 0.0)
        return self._clamp(stream, t)

    def _dejitter(self, stream: str, t: float) -> float:
        T = 1.0 / self._fps(stream)
        st = self._dj.get(stream)
        if st is None or t - st["t_prev"] > self.window_s:
            if st is not None:
                self.envelope_restarts += 1
            st = {"buf": deque([(0, t)]), "k_prev": 0, "a": t, "t_prev": t, "recent": deque(maxlen=_DROP_CONFIRM)}
            self._dj[stream] = st
            return t
        buf, recent = st["buf"], st["recent"]
        k = st["k_prev"] + 1                       # assume delay, not drop (Wi-Fi jitter ~ T is common, drops rare)
        r = t - (st["a"] + k * T)
        gap = t - st["t_prev"]
        late = r >= (1.0 - _EPS_K) * T
        # possible unconfirmed sender drop: a late frame after a long gap, or still late at normal cadence right
        # after one (a drained backlog arrives at short gaps and is left to the envelope)
        suspect = late and (gap >= 1.5 * T or (st.get("suspect", False) and gap >= 0.5 * T))
        recent.append((t, r))
        if len(recent) == _DROP_CONFIRM:
            rs = [x[1] for x in recent]
            gaps = np.diff([x[0] for x in recent])
            if min(rs) >= (1.0 - _EPS_K) * T and np.all(gaps >= 0.5 * T):
                # every recent frame is >= 1 period late at a normal cadence (not a drained backlog): frames were
                # dropped by the sender. Re-index the recent ones too, so the envelope stays consistent.
                n = int(math.floor(min(rs) / T + _EPS_K))
                k += n
                for i in range(1, _DROP_CONFIRM):
                    kk, tt = buf[-i]
                    buf[-i] = (kk + n, tt)
                self.dropped_frames += n
                recent.clear()
                suspect = False
        buf.append((k, t))
        while buf and buf[0][1] < t - self.window_s:
            buf.popleft()
        st["a"] = min(ti - ki * T for ki, ti in buf)
        st["k_prev"], st["t_prev"], st["suspect"] = k, t, suspect
        est = st["a"] + k * T
        if suspect:
            # unconfirmed drop: k is too low by the dropped count n, so a + k*T is ~n*T early; t - 1.4 T bounds it
            est = max(est, t - _DROP_BOUND_T * T)
        return min(est, t)                         # never later than the arrival itself

    # ------------------------------------------------------------------------------------------------ ping sync
    def add_ping(self, t0_ns: int, t1_ns: int, t2_ns: int, t3_ns: int) -> None:
        """One clock_ping exchange: t0/t3 host monotonic ns (send/receive), t1/t2 sender ns (receive/reply)."""
        rtt = ((t3_ns - t0_ns) - (t2_ns - t1_ns)) * 1e-9
        if rtt < 0 or t3_ns < t0_ns:
            return  # impossible sample (clock jump / bad reply)
        theta = ((t1_ns - t0_ns) + (t2_ns - t3_ns)) * 0.5e-9
        t_mid = (t0_ns + t3_ns) * 0.5e-9
        self._pings.append((t_mid, theta, rtt))
        self.n_pings += 1
        while self._pings and self._pings[0][0] < t_mid - self.window_s:
            self._pings.popleft()
        self._refit()

    def _refit(self) -> None:
        """theta(t) = robust line through the lowest-RTT 30 % of the pings in the window (Theil-Sen: median pairwise
        slope = drift, median intercept), evaluated as a line. Fewer than 6 pings: the single minimum-RTT sample."""
        p = np.array(self._pings, dtype=np.float64)
        best = int(np.argmin(p[:, 2]))
        fit = (float(p[best, 0]), float(p[best, 1]), 0.0)
        if len(p) >= 6:
            sel = np.flatnonzero(p[:, 2] <= np.quantile(p[:, 2], _PING_RTT_QUANTILE))
            if len(sel) >= 3:
                x, y = p[sel, 0], p[sel, 1]
                i, j = np.triu_indices(len(x), 1)
                dx = x[j] - x[i]
                ok = dx > 0
                d = float(np.median((y[j] - y[i])[ok] / dx[ok])) if np.any(ok) else 0.0
                if abs(d) * 1e6 > _MAX_DRIFT_PPM:
                    d = 0.0
                xm = float(x.mean())
                fit = (xm, float(np.median(y - d * (x - xm))), d)
        self._fit = fit

    def theta(self, t_host_s: float) -> float | None:
        """Estimated (sender - host) clock offset in seconds at host time t_host_s; None before the first ping."""
        if self._fit is None:
            return None
        t_ref, th_ref, d = self._fit
        return th_ref + d * (t_host_s - t_ref)

    # ------------------------------------------------------------------------------------------------ stats
    def stats(self) -> dict:
        out = {"mode": self.mode, "offset_s": None, "rtt_ms_min": None, "drift_ppm": None, "n_pings": self.n_pings,
               "monotonic_clamps": self.monotonic_clamps, "n_mapped": self.n_mapped}
        if self.mode == "sender_stamp_ping":
            out["n_unsynced"] = self.n_unsynced
            if self._fit is not None:
                out["offset_s"] = self.theta(self._pings[-1][0])   # at the newest ping (no extrapolation)
                out["drift_ppm"] = self._fit[2] * 1e6
                out["rtt_ms_min"] = min(r for _, _, r in self._pings) * 1e3
        elif self.mode == "sender_stamp_realtime":
            out["offset_s"] = self._rt_minus_mono_ns() * 1e-9
        elif self.mode == "arrival_dejitter":
            out["envelope_restarts"] = self.envelope_restarts
            out["dropped_frames"] = self.dropped_frames
        return out


def clock_map_from_config(clock: dict, streams: dict | None = None) -> ClockMap:
    """Build from a sensor yaml `clock:` block (station.camera_config()["clock"]).

    Keys: mode, latency_s {stream: s}, dejitter_window_s (arrival_dejitter), ping_window_s (sender_stamp_ping),
    residual_latency_s {stream: s} (sender modes). fps for dejitter = streams[s].physical_fps_hz (default 30)."""
    mode = clock.get("mode", "arrival_minus_latency")
    window = clock.get("ping_window_s", 60.0) if mode == "sender_stamp_ping" else clock.get("dejitter_window_s", 2.0)
    fps = {s: float(v.get("physical_fps_hz", 30.0)) for s, v in (streams or {}).items()} or 30.0
    return ClockMap(mode, clock.get("latency_s") or {}, fps_hz=fps, window_s=float(window),
                    residual_s=clock.get("residual_latency_s"))
