"""``python -m process_frames.pi_collect --endpoint tcp://PI:5555 [--out DIR] [--seconds N] [--save-every S] [--bind]``

(or ``--station FILE`` instead of ``--endpoint``, to take the Pi's address from ``endpoints.pi_frames``)

Collects the Pi's live stream in the format it sends today (not C0, PI_PAYLOAD.md): one ZeroMQ message of three
parts per packet,

    part 1  header, JSON: version, packet_no, t_send_ns, clock_offset_ns,
            main {frame_no, t_sensor_ns, exposure_us, t_mid_ns}, ultrawide {same keys}, skew_ns, uw_reused
    part 2  main camera (imx519) JPEG
    part 3  ultrawide camera (imx219) JPEG

Receive only: nothing is ever sent back to the Pi. It connects a SUB socket to ``--endpoint`` (the Pi binds a PUB);
``--bind`` binds instead, for a Pi that connects. Output goes to ``--out`` (default ``~/pi_runs/<stamp>``, or
``<session_root>/<stamp>_pi_collect`` with ``--station``). Every packet's header is written verbatim to
``headers.jsonl`` with what was seen on arrival (part sizes, decoded image sizes, problems); the JPEGs of the first
packet and then one packet every ``--save-every`` seconds are written as received. A stats line is printed every
second. A packet without the expected shape is logged and counted, never fatal. The times are not interpreted yet.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import zmq

from process_frames.errors import SettingsError
from process_frames.settings import Settings

HEADER_KEYS = ("version", "packet_no", "t_send_ns", "clock_offset_ns", "main", "ultrawide", "skew_ns", "uw_reused")
FRAME_KEYS = ("frame_no", "t_sensor_ns", "exposure_us", "t_mid_ns")
CAMERAS = ("main", "ultrawide")
N_PARTS = 3


@dataclass
class PacketReport:
    """One received message: the header as sent (None if it isn't a JSON object), the decoded images, problems."""

    arrival_ns: int
    sizes: list[int]
    header: dict | None
    images: dict[str, np.ndarray | None]
    problems: list[str] = field(default_factory=list)
    raw: list[bytes] = field(default_factory=list, repr=False)     # the parts as received (not in the record)

    def record(self) -> dict:
        """The headers.jsonl line."""
        return {"arrival_wall_ns": self.arrival_ns, "sizes": self.sizes,
                "image_wh": {cam: (None if img is None else [img.shape[1], img.shape[0]])
                             for cam, img in self.images.items()},
                "problems": self.problems, "header": self.header}


def inspect_packet(parts: Sequence[bytes], arrival_ns: int) -> PacketReport:
    """Check one multipart message against the expected shape; never raises for a bad packet."""
    parts = list(parts)
    report = PacketReport(arrival_ns, [len(p) for p in parts], None, {cam: None for cam in CAMERAS}, raw=parts)
    if len(parts) != N_PARTS:
        report.problems.append(f"{len(parts)} parts, expected {N_PARTS} (header, main JPEG, ultrawide JPEG)")
    if parts:
        report.header = _parse_header(parts[0], report.problems)
    for i, cam in enumerate(CAMERAS, start=1):
        if i >= len(parts):
            continue
        img = cv2.imdecode(np.frombuffer(parts[i], np.uint8), cv2.IMREAD_COLOR) if parts[i] else None
        if img is None:
            report.problems.append(f"part {i + 1} ({cam}) is not a decodable JPEG ({len(parts[i])} bytes)")
        report.images[cam] = img
    return report


def _parse_header(raw: bytes, problems: list[str]) -> dict | None:
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        problems.append(f"header is not JSON ({e}); first bytes {raw[:32]!r}")
        return None
    if not isinstance(header, dict):
        problems.append(f"header is JSON {type(header).__name__}, expected an object")
        return None
    _check_keys("header", header, HEADER_KEYS, problems)
    for cam in CAMERAS:
        sub = header.get(cam)
        if sub is None:
            continue
        if isinstance(sub, dict):
            _check_keys(cam, sub, FRAME_KEYS, problems)
        else:
            problems.append(f"header.{cam} is {type(sub).__name__}, expected an object")
    return header


def _check_keys(where: str, d: dict, expected: Sequence[str], problems: list[str]) -> None:
    missing = [k for k in expected if k not in d]
    extra = [k for k in d if k not in expected]
    if missing:
        problems.append(f"{where}: missing {missing}")
    if extra:
        problems.append(f"{where}: unexpected {extra}")


def _int(d: Any, *keys: str) -> int | None:
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d if isinstance(d, int) and not isinstance(d, bool) else None


class Stats:
    """Counters per stats window, and totals."""

    def __init__(self) -> None:
        self.total = {"packets": 0, "bad": 0, "packet_gaps": 0, "uw_reused": 0}
        self._last: dict[str, int | None] = {"packet_no": None, "main": None, "ultrawide": None}
        self._reset_window()

    def _reset_window(self) -> None:
        self.w = {"packets": 0, "bad": 0, "packet_gaps": 0, "uw_reused": 0, "main_gaps": 0, "uw_gaps": 0,
                  "bytes": [0, 0, 0]}
        self.w_wh: dict[str, list | None] = {cam: None for cam in CAMERAS}

    def add(self, r: PacketReport) -> None:
        self.w["packets"] += 1
        self.total["packets"] += 1
        if r.problems:
            self.w["bad"] += 1
            self.total["bad"] += 1
        for i, n in enumerate(r.sizes[:3]):
            self.w["bytes"][i] += n
        for cam, img in r.images.items():
            if img is not None:
                self.w_wh[cam] = [img.shape[1], img.shape[0]]
        h = r.header or {}
        reused = h.get("uw_reused") is True
        if reused:
            self.w["uw_reused"] += 1
            self.total["uw_reused"] += 1
        gap = self._gap("packet_no", _int(h, "packet_no"))
        self.w["packet_gaps"] += gap
        self.total["packet_gaps"] += gap
        self.w["main_gaps"] += self._gap("main", _int(h, "main", "frame_no"))
        if not reused:                              # a reused ultrawide frame repeats its frame_no
            self.w["uw_gaps"] += self._gap("ultrawide", _int(h, "ultrawide", "frame_no"))

    def _gap(self, key: str, value: int | None) -> int:
        """How many numbers were skipped since the last value (0 for the first, a repeat or a restart)."""
        last, self._last[key] = self._last[key], value if value is not None else self._last[key]
        if value is None or last is None or value <= last:
            return 0
        return value - last - 1

    def line(self, window_s: float) -> str:
        w = self.w
        n = max(w["packets"], 1)
        kb = [b / n / 1000 for b in w["bytes"]]
        wh = {cam: ("-" if v is None else f"{v[0]}x{v[1]}") for cam, v in self.w_wh.items()}
        out = (f"{w['packets'] / window_s:5.1f} pkt/s | packet_no gaps {w['packet_gaps']} | main {wh['main']} "
               f"{kb[1]:.0f} kB frame_no gaps {w['main_gaps']} | ultrawide {wh['ultrawide']} {kb[2]:.0f} kB "
               f"frame_no gaps {w['uw_gaps']} reused {w['uw_reused']} | bad {w['bad']}")
        self._reset_window()
        return out

    def summary(self) -> str:
        t = self.total
        return (f"total: {t['packets']} packets, {t['bad']} bad, {t['packet_gaps']} packet_no gaps, "
                f"{t['uw_reused']} with uw_reused")


class Collector:
    """The receive-only socket: SUB (subscribed to everything) connected to the Pi, or bound with ``bind``."""

    def __init__(self, endpoint: str, bind: bool = False, rcvhwm: int = 100) -> None:
        self._ctx = zmq.Context()
        self.socket = self._ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, rcvhwm)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        if bind:
            self.socket.bind(endpoint)
            endpoint = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        else:
            self.socket.connect(endpoint)
        self.endpoint = endpoint

    def receive_one(self, timeout_s: float) -> PacketReport | None:
        if not self.socket.poll(int(timeout_s * 1000)):
            return None
        parts = self.socket.recv_multipart()
        return inspect_packet(parts, time.time_ns())

    def close(self) -> None:
        self.socket.close()
        self._ctx.term()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m process_frames.pi_collect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", default=None, help="the Pi's sender, e.g. tcp://RaspberryPi5.local:5555 or "
                                                    "tcp://10.42.0.78:5555")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: ~/pi_runs/<stamp>; with --station: "
                                                         "<session_root>/<stamp>_pi_collect)")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long (default: until Ctrl-C)")
    p.add_argument("--save-every", type=float, default=5.0,
                   help="save the images of the first packet, then of one packet every this many seconds "
                        "(default 5; 0 = every packet). Every packet is logged in headers.jsonl either way")
    p.add_argument("--bind", action="store_true", help="bind the endpoint (e.g. tcp://0.0.0.0:5555) instead of "
                                                       "connecting, for a Pi that connects to this machine")
    p.add_argument("--station", type=Path, default=None,
                   help="optional: take the Pi's address (endpoints.pi_frames) and the output root (session_root) "
                        "from this station file when --endpoint / --out are not given")
    args = p.parse_args(argv)
    stamp = f"{_dt.datetime.now():%Y-%m-%d_%H-%M-%S}"
    if args.station is not None:
        try:
            s = Settings.load(args.station, mock=False)
        except SettingsError as e:
            print(f"pi_collect: {'; '.join(e.problems)}", file=sys.stderr)
            return 1
        endpoint = args.endpoint or s.pi_frames
        out = args.out or s.session_root / f"{stamp}_pi_collect"
    elif args.endpoint is None:
        p.error("give --endpoint tcp://<PI>:5555 (or --station FILE to use its endpoints.pi_frames)")
    else:
        endpoint = args.endpoint
        out = args.out or Path("~/pi_runs").expanduser() / stamp
    (out / "jpeg").mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    collector = Collector(endpoint, bind=args.bind)
    print(f"pi_collect: {'bound' if args.bind else 'connected'} SUB {collector.endpoint}; writing {out}", flush=True)
    stats = Stats()
    start = window_start = time.monotonic()
    last_save = None
    printed_first = False
    try:
        with (out / "headers.jsonl").open("a", encoding="utf-8") as log:
            while not stop.is_set():
                now = time.monotonic()
                if args.seconds is not None and now - start >= args.seconds:
                    break
                if now - window_start >= 1.0:
                    if stats.total["packets"] == 0:
                        if int(now - start) % 5 == 0:
                            print(f"waiting for the first packet from {collector.endpoint} ...", flush=True)
                        stats.line(now - window_start)
                    else:
                        print(stats.line(now - window_start), flush=True)
                    window_start = now
                r = collector.receive_one(0.1)
                if r is None:
                    continue
                stats.add(r)
                log.write(json.dumps(r.record()) + "\n")
                if not printed_first:
                    printed_first = True
                    print("first packet header:\n" + json.dumps(r.header, indent=2), flush=True)
                    print(f"part sizes {r.sizes}", flush=True)
                if r.problems:
                    print(f"bad packet (packet_no {_int(r.header, 'packet_no')}): {'; '.join(r.problems)}",
                          flush=True)
                if last_save is None or time.monotonic() - last_save >= args.save_every:
                    last_save = time.monotonic()
                    _save_jpegs(out / "jpeg", r)
    finally:
        collector.close()
    print(stats.summary(), flush=True)
    return 0 if stats.total["packets"] else 2


def _save_jpegs(folder: Path, r: PacketReport) -> None:
    """Write the JPEG parts byte for byte as received, named by packet_no (the arrival time if it has none)."""
    tag = _int(r.header, "packet_no")
    tag = f"packet_{tag:08d}" if tag is not None else f"arrival_{r.arrival_ns}"
    for i, cam in enumerate(CAMERAS, start=1):
        if i < len(r.raw) and r.raw[i]:
            (folder / f"{tag}_{cam}.jpg").write_bytes(r.raw[i])


if __name__ == "__main__":
    sys.exit(main())
