"""RigFrameSource: two free-running rig cameras -> one iPhone-equivalent observation packet (P2 §6, brief D8).

All deployment logic lives here and has no real-env imports, so it is unit-testable anywhere. The real-env peripheral
``real_env/peripherals/rig_camera.py: RigCamera`` is a thin adapter around it.

Per packet:

* **The main camera drives.** One packet per new main frame. The packet timestamp is the main frame's *capture time*
  ``rig.capture_time_ns("main", frame.t_ns) / 1e9`` (CLOCK_MONOTONIC seconds). rig.yaml time offsets are applied here
  and only here, so the real-env config must say ``camera_latency_s: 0`` (latency applied exactly once).
* **Ultrawide pairing** (``PairingConfig.ultrawide``): ``"nearest"`` picks, among the last ``history`` ultrawide
  frames, the one whose capture time is nearest to the main capture time (the newest when all are older);
  ``"latest"`` takes the newest one. ``skew = t_main - t_uw`` is recorded; above ``max_skew_s`` a rate-limited warning
  is logged; above ``stale_after_s`` the pair is refused.
* **Remap** each physical frame to its virtual iPhone camera at the output size with the *same* ``RemapTable`` code the
  export uses (``rig.remap(name, size)``). The ultrawide remap is cached by ultrawide ``seq`` (it usually runs slower
  than the main or is paired twice).
* **Every packet holds both cameras.** On a main timeout or a stale ultrawide the previous packet is returned again
  (same images, same timestamp), up to ``max_consecutive_timeouts`` times in a row, then ``RuntimeError``.

Sync modes:

* ``"live"`` (hardware, realtime mocks): the main is read from the backend queue and fast-forwarded to the newest frame
  if the consumer fell behind (``catchup_skips``); the ultrawide queue is drained without blocking and the candidates
  come from ``read_recent(history)``.
* ``"lockstep"`` (offline replay with ``realtime=False``, used by the D10d test): every main frame is delivered in
  order, and the ultrawide is read until its newest frame is at or past the main capture time, so pairing is
  deterministic and independent of thread timing.

Backends come from ``sensor_rig.drivers.make_camera`` (RECONCILIATION R4): any object with ``start / stop /
read(timeout_s) / latest() / read_recent(n)`` and ``Frame`` items (``image, t_ns, seq, t_arrival_ns``) works, which is
how the tests inject fake sources (``sources=``).
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

log = logging.getLogger("sensor_rig.deploy")

CAMERAS = ("main", "ultrawide")
SYNC_MODES = ("live", "lockstep")
_STATS_WINDOW = 600


@dataclass(frozen=True)
class PairingConfig:
    ultrawide: Literal["nearest", "latest"] = "nearest"
    max_skew_s: float = 0.034          # warn (rate-limited) when |t_main - t_uw| exceeds this
    stale_after_s: float = 0.25        # refuse the pair when |t_main - t_uw| exceeds this
    history: int = 4                   # ultrawide frames considered for "nearest"

    def __post_init__(self):
        if self.ultrawide not in ("nearest", "latest"):
            raise ValueError(f"pairing.ultrawide must be 'nearest' or 'latest', got {self.ultrawide!r}")
        if not (self.history >= 2):
            raise ValueError(f"pairing.history must be >= 2, got {self.history}")
        if not (0 < self.max_skew_s <= self.stale_after_s):
            raise ValueError("pairing needs 0 < max_skew_s <= stale_after_s")

    @classmethod
    def from_dict(cls, d: dict | None) -> "PairingConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown pairing keys {sorted(unknown)}")
        return cls(**d)


@dataclass(frozen=True)
class Packet:
    images: dict[str, np.ndarray]      # {"main","ultrawide"}: (H,W,3) uint8 BGR, C-contiguous, virtual cameras
    t_capture_s: float                 # main capture time, CLOCK_MONOTONIC seconds
    t_capture_ns: dict[str, int]       # per camera capture time (rig offsets applied)
    seq: dict[str, int]                # backend seq per camera
    skew_s: float                      # t_main - t_uw
    repeated: bool = False             # True when this is the previous packet re-served after a timeout/stale pair


def _pct(values, q) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if len(values) else None


class _RateLimited:
    def __init__(self, period_s: float = 5.0):
        self.period_s = period_s
        self._last: dict[str, float] = {}

    def __call__(self, key: str, msg: str, *args) -> None:
        now = time.monotonic()
        if now - self._last.get(key, -1e9) >= self.period_s:
            self._last[key] = now
            log.warning(msg, *args)


class RigFrameSource:
    def __init__(self, rig, output_sizes: dict[str, tuple[int, int]] | None = None, backend: str = "argus",
                 backend_opts: dict | None = None, pairing: PairingConfig | dict | None = None,
                 frame_timeout_s: float = 0.5, max_consecutive_timeouts: int = 10, min_coverage: float = 0.98,
                 sources: dict[str, Any] | None = None, sync: str = "live", startup_timeout_s: float = 5.0):
        """
        rig: ``RigConfig``. output_sizes: ``{"main": (w, h), "ultrawide": (w, h)}`` (default: each virtual camera's
        ``deploy`` output, 960x720). backend / backend_opts: passed to ``make_camera`` for each camera; a key of
        ``backend_opts`` equal to a camera name holds per-camera overrides. sources: inject ready backends instead.
        """
        self.rig = rig
        if output_sizes is None:
            output_sizes = {n: rig.virtual[n].size("deploy") for n in CAMERAS}
        output_sizes = {str(k): (int(v[0]), int(v[1])) for k, v in dict(output_sizes).items()}
        if set(output_sizes) != set(CAMERAS):
            raise ValueError(f"output_sizes must have exactly the keys {CAMERAS}, got {sorted(output_sizes)}")
        missing = [n for n in CAMERAS if n not in rig.virtual]
        if missing:
            raise ValueError(f"rig has no virtual camera(s) {missing}")
        if sync not in SYNC_MODES:
            raise ValueError(f"sync must be one of {SYNC_MODES}, got {sync!r}")
        self.output_sizes = output_sizes
        self.pairing = pairing if isinstance(pairing, PairingConfig) else PairingConfig.from_dict(pairing)
        self.frame_timeout_s = float(frame_timeout_s)
        self.max_consecutive_timeouts = int(max_consecutive_timeouts)
        self.sync = sync
        self.startup_timeout_s = float(startup_timeout_s)
        self.backend = backend
        self.backend_opts = dict(backend_opts or {})
        self.physical = {n: rig.virtual[n].source for n in CAMERAS}

        # Remap tables: exactly the export code path at another output size (same half-pixel scaling convention).
        self.tables = {}
        for n in CAMERAS:
            table = rig.remap(n, output_sizes[n])
            if table.coverage < min_coverage:
                raise ValueError(
                    f"virtual camera {n!r} at {output_sizes[n]}: coverage {table.coverage:.4f} < min_coverage "
                    f"{min_coverage} (the physical lens does not see the whole iPhone view; see calib-check)")
            self.tables[n] = table
        if getattr(rig, "placeholder", False):
            log.warning("rig %s uses placeholder calibration: deployment images are NOT iPhone-equivalent until "
                        "calib-import has been run", getattr(rig, "source_path", None) or getattr(rig, "rig_name", "?"))

        self._sources_injected = sources is not None
        self.sources: dict[str, Any] = dict(sources) if sources is not None else {}
        self._started = False
        self._warn = _RateLimited()
        self._uw_hist: collections.deque = collections.deque(maxlen=self.pairing.history)
        self._uw_cache: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()
        self._last: Packet | None = None
        self._serve_startup = False
        self._last_new_main_mono = time.monotonic()
        self._fails = 0
        self._last_seen_seq = {n: None for n in CAMERAS}
        self.reset_stats()

    # ------------------------------------------------------------------------------------------------ lifecycle
    def _opts_for(self, name: str) -> dict:
        common = {k: v for k, v in self.backend_opts.items() if k not in CAMERAS}
        per = self.backend_opts.get(name) or {}
        opts = {**common, **dict(per)}
        if self.backend == "mock" and not (opts.get("episode_dir") or opts.get("replay")):
            # Synthetic mock: replay-only options would be rejected by MockCamera.synthetic.
            for k in ("episode_dir", "replay", "realtime", "loop", "restamp", "preserve_timestamps"):
                opts.pop(k, None)
        return opts

    def start(self) -> None:
        if self._started:
            return
        if not self._sources_injected:
            from sensor_rig.drivers.base import make_camera

            opts = {n: self._opts_for(n) for n in CAMERAS}
            if self.backend == "mock":
                # Re-stamped replays: give every camera the same live start so main <-> ultrawide offsets survive
                # (MockCamera.replay already shares the episode-wide origin).
                start_now_ns = time.monotonic_ns() + 50_000_000
                for o in opts.values():
                    if (o.get("episode_dir") or o.get("replay")) and not o.get("preserve_timestamps"):
                        o.setdefault("start_now_ns", start_now_ns)
            for n in CAMERAS:
                self.sources[n] = make_camera(self.rig, self.physical[n], backend=self.backend, **opts[n])
        for n in CAMERAS:
            if n not in self.sources:
                raise ValueError(f"no source for camera {n!r}")
            src = self.sources[n]
            if not getattr(src, "running", False):
                src.start()
        self._started = True
        self._wait_first_packet()

    def stop(self) -> None:
        for src in self.sources.values():
            try:
                src.stop()
            except Exception as e:  # pragma: no cover - best effort on shutdown
                log.warning("stopping %s failed: %r", src, e)
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def _wait_first_packet(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        main = self.sources["main"]
        while time.monotonic() < deadline:
            f = main.read(timeout_s=min(0.1, self.frame_timeout_s))
            if f is None:
                if self._exhausted(main):
                    break
                continue
            if self.sync == "live":
                f = self._catch_up(main, f)
            p = self._try_build(f)
            if p is not None:
                self._serve_startup = self.sync == "lockstep"
                return
        raise RuntimeError(f"no main+ultrawide pair within {self.startup_timeout_s:.1f} s of start "
                           f"(main frames: {self.stats_counts['main_frames']}, check both cameras)")

    # ------------------------------------------------------------------------------------------------ packets
    @property
    def last_packet(self) -> Packet | None:
        return self._last

    @property
    def exhausted(self) -> bool:
        """True when a finite (replay) main source has delivered everything."""
        return self._exhausted(self.sources.get("main"))

    @staticmethod
    def _exhausted(src) -> bool:
        return bool(getattr(src, "exhausted", False)) if src is not None else False

    def next_packet(self) -> Packet | None:
        """Packet for the next NEW main frame (blocks up to ``frame_timeout_s``).

        On timeout or a stale ultrawide: the previous packet with ``repeated=True`` (``RuntimeError`` after
        ``max_consecutive_timeouts`` in a row). ``None`` only when a finite replay source is exhausted. In lockstep
        mode the first call returns the packet built by ``start()`` (so no frame is skipped); in live mode that
        startup packet is only available as ``last_packet``.
        """
        self._require_started()
        if self._serve_startup:
            self._serve_startup = False
            return self._last
        main = self.sources["main"]
        f = main.read(timeout_s=self.frame_timeout_s)
        if f is None:
            if self._exhausted(main):
                return None
            self.stats_counts["main_timeouts"] += 1
            return self._fail("no new main frame within %.3f s" % self.frame_timeout_s)
        if self.sync == "live":
            f = self._catch_up(main, f)
        p = self._try_build(f)
        return p if p is not None else self._fail("stale ultrawide")

    def latest_packet(self) -> Packet:
        """Non-blocking: packet for the newest main frame (cached while no new main frame arrived), for fixed-rate
        publishing. Raises after ``max_consecutive_timeouts`` calls once the main has been silent for more than
        ``frame_timeout_s``."""
        self._require_started()
        if self.sync == "lockstep":
            p = self.next_packet()
            if p is None:
                raise RuntimeError("replay source exhausted")
            return p
        main = self.sources["main"]
        newest = None
        for _ in range(256):                       # drain the queue (live producers are realtime, so this ends)
            g = main.read(timeout_s=0.0)
            if g is None:
                break
            self._note_seq("main", g.seq)
            newest = g
        if newest is None:
            newest = main.latest()
        last_seq = self._last.seq["main"] if self._last is not None else None
        if newest is not None and newest.seq != last_seq:
            p = self._try_build(newest, note_seq=False)
            if p is not None:
                return p
            return self._fail("stale ultrawide")
        if time.monotonic() - self._last_new_main_mono > self.frame_timeout_s:
            self.stats_counts["main_timeouts"] += 1
            return self._fail("no new main frame for %.3f s" % (time.monotonic() - self._last_new_main_mono))
        assert self._last is not None
        return self._last

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("RigFrameSource.start() has not been called")

    def _fail(self, reason: str) -> Packet:
        self._fails += 1
        self.stats_counts["repeats"] += 1
        if self._fails > self.max_consecutive_timeouts or self._last is None:
            raise RuntimeError(f"rig camera: {reason} ({self._fails} consecutive failures, "
                               f"max_consecutive_timeouts={self.max_consecutive_timeouts})")
        self._warn("fail-" + reason.split()[0], "rig camera: %s; re-serving the previous packet (%d in a row)",
                   reason, self._fails)
        return dataclasses.replace(self._last, repeated=True)

    def _catch_up(self, main, f):
        newest = main.latest()
        while newest is not None and f.seq < newest.seq:
            g = main.read(timeout_s=0.0)
            if g is None:
                break
            self._note_seq("main", f.seq)
            self.stats_counts["catchup_skips"] += 1
            f = g
        return f

    def _note_seq(self, cam: str, seq: int) -> None:
        last = self._last_seen_seq[cam]
        if last is not None and seq > last + 1:
            self.stats_counts[f"{cam}_drops"] += seq - last - 1
        if last is None or seq > last:
            self._last_seen_seq[cam] = seq

    def _t_cap(self, cam: str, f) -> int:
        return int(self.rig.capture_time_ns(self.physical[cam], int(f.t_ns)))

    # ---- ultrawide selection
    def _uw_candidates(self, t_main_ns: int) -> list:
        uw = self.sources["ultrawide"]
        if self.sync == "lockstep":
            while not self._uw_hist or self._t_cap("ultrawide", self._uw_hist[-1]) < t_main_ns:
                g = uw.read(timeout_s=self.frame_timeout_s)
                if g is None:
                    break
                self._note_seq("ultrawide", g.seq)
                self._uw_hist.append(g)
            return list(self._uw_hist)
        for _ in range(256):
            g = uw.read(timeout_s=0.0)
            if g is None:
                break
            self._note_seq("ultrawide", g.seq)
        return list(uw.read_recent(self.pairing.history))

    def _pick_uw(self, t_main_ns: int):
        cands = self._uw_candidates(t_main_ns)
        if not cands:
            return None, None
        if self.pairing.ultrawide == "latest":
            f = cands[-1]
            return f, self._t_cap("ultrawide", f)
        best, best_t, best_d = None, None, None
        for f in cands:
            t = self._t_cap("ultrawide", f)
            d = abs(t - t_main_ns)
            if best_d is None or d < best_d or (d == best_d and t > best_t):
                best, best_t, best_d = f, t, d
        return best, best_t

    # ---- build
    def _remap(self, cam: str, img: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.tables[cam].apply(np.ascontiguousarray(img))
        self._remap_ms[cam].append((time.perf_counter() - t0) * 1e3)
        self.stats_counts[f"remap_calls_{cam}"] += 1
        return out

    def _uw_image(self, f) -> np.ndarray:
        img = self._uw_cache.get(f.seq)
        if img is None:
            img = self._remap("ultrawide", f.image)
            self._uw_cache[f.seq] = img
            while len(self._uw_cache) > self.pairing.history + 1:
                self._uw_cache.popitem(last=False)
        else:
            self._uw_cache.move_to_end(f.seq)
        return img

    def _try_build(self, f, note_seq: bool = True) -> Packet | None:
        if note_seq:
            self._note_seq("main", f.seq)
        self.stats_counts["main_frames"] += 1
        t_main = self._t_cap("main", f)
        uwf, t_uw = self._pick_uw(t_main)
        if uwf is None:
            self.stats_counts["uw_missing"] += 1
            return None
        skew_ns = t_main - t_uw
        skew_s = skew_ns / 1e9
        if abs(skew_s) > self.pairing.stale_after_s:
            self.stats_counts["uw_stale"] += 1
            return None
        if abs(skew_s) > self.pairing.max_skew_s:
            self._warn("skew", "main/ultrawide skew %.1f ms > max_skew_s %.1f ms", skew_s * 1e3,
                       self.pairing.max_skew_s * 1e3)
        if self._last is not None and t_main <= self._last.t_capture_ns["main"]:
            self._warn("nonmono", "main capture time did not increase (%d <= %d); frame ignored",
                       t_main, self._last.t_capture_ns["main"])
            self.stats_counts["non_monotonic"] += 1
            return None
        images = {"main": self._remap("main", f.image), "ultrawide": self._uw_image(uwf)}
        p = Packet(images=images, t_capture_s=t_main / 1e9, t_capture_ns={"main": t_main, "ultrawide": t_uw},
                   seq={"main": int(f.seq), "ultrawide": int(uwf.seq)}, skew_s=skew_s)
        self._last = p
        self._fails = 0
        self._last_new_main_mono = time.monotonic()
        self.stats_counts["packets"] += 1
        self._t_main.append(t_main)
        if not self._t_uw or self._t_uw[-1] != t_uw:
            self._t_uw.append(t_uw)
        self._skew.append(skew_s)
        return p

    # ------------------------------------------------------------------------------------------------ stats
    def reset_stats(self) -> None:
        self.stats_counts: collections.Counter = collections.Counter()
        self._t_main: collections.deque = collections.deque(maxlen=_STATS_WINDOW)
        self._t_uw: collections.deque = collections.deque(maxlen=_STATS_WINDOW)
        self._skew: collections.deque = collections.deque(maxlen=_STATS_WINDOW)
        self._remap_ms = {n: collections.deque(maxlen=_STATS_WINDOW) for n in CAMERAS}

    @staticmethod
    def _fps(ts) -> float | None:
        if len(ts) < 2:
            return None
        span = (ts[-1] - ts[0]) / 1e9
        return (len(ts) - 1) / span if span > 0 else None

    def stats(self) -> dict:
        """fps per camera (from capture times of used frames), skew p50/p95/max (ms), drops (seq gaps), remap ms
        p50/p99 per camera, and counters (packets, repeats, timeouts, stale ultrawide, catch-up skips, remap calls)."""
        sk = np.abs(np.asarray(self._skew, dtype=np.float64)) * 1e3
        c = self.stats_counts
        return {
            "packets": c["packets"], "repeats": c["repeats"], "main_timeouts": c["main_timeouts"],
            "uw_stale": c["uw_stale"], "uw_missing": c["uw_missing"], "catchup_skips": c["catchup_skips"],
            "non_monotonic": c["non_monotonic"],
            "fps": {"main": self._fps(self._t_main), "ultrawide": self._fps(self._t_uw)},
            "skew_ms": {"p50": _pct(sk, 50), "p95": _pct(sk, 95), "max": float(sk.max()) if len(sk) else None},
            "drops": {"main": c["main_drops"], "ultrawide": c["ultrawide_drops"]},
            "remap_ms": {n: {"p50": _pct(v, 50), "p99": _pct(v, 99)} for n, v in self._remap_ms.items()},
            "remap_calls": {n: c[f"remap_calls_{n}"] for n in CAMERAS},
        }

    def stats_line(self) -> str:
        s = self.stats()

        def f(v, fmt="{:.1f}"):
            return "-" if v is None else fmt.format(v)

        return (f"rig: packets {s['packets']} fps main {f(s['fps']['main'])} uw {f(s['fps']['ultrawide'])} | "
                f"skew ms p50 {f(s['skew_ms']['p50'])} p95 {f(s['skew_ms']['p95'])} max {f(s['skew_ms']['max'])} | "
                f"drops main {s['drops']['main']} uw {s['drops']['ultrawide']} | repeats {s['repeats']} | "
                f"remap ms main p50 {f(s['remap_ms']['main']['p50'], '{:.2f}')} "
                f"p99 {f(s['remap_ms']['main']['p99'], '{:.2f}')}")
