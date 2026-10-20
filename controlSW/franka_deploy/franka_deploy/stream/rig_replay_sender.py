"""Replay a raw rig episode as the rig's wireless stream: the same RigStreamSender code the Jetson runs (sim, mock backend).

Why: the policy host's receiver (B5/B6) must be tested against what the rig really sends: v2 packets with the rig's
capture stamps in the SENDER's clock, paired main + ultrawide frames already remapped to the virtual iPhone cameras
(960x720), clock_ping answers, bounded-queue drops. Here `sensor_rig.stream.sender.RigStreamSender` (backend "mock",
`MockCamera.replay` of a raw rig episode in real time) runs unchanged; only its transport differs: it routes through the
B4 LinkEmulator, which gives the sender clock offset/drift (on the stamps AND on the ping answers: sender clock =
link.sender_clock_ns()), delay, jitter and stalls. The emulated TCP socket buffer is bounded (`max_inflight` packets):
when the link backs up, send_packet() blocks and the sender's own queue drops the oldest packets (a rig never builds
latency).

Episode: `--episode DIR` (a raw rig episode), else from the sender config's `sim` section (run/sim/rig_raw_replay.yaml):
`raw_episode_dir`, else one generated (and cached under --cache-dir) from an iPhone episode with
sensor_rig.sim.simulator.simulate_raw_from_iphone, so no hardware data is needed.

Truth file (--truth): one JSON line per stream per packet, {"seq", "stream", "t_true"}; t_true = the frame's capture
time (rig time offsets applied) in THIS host's monotonic clock, seconds (the oracle for the receiver's stamp error).
Ready file (--ready-file, JSON, atomic, written before streaming): {start_now_ns, first_tick_ns, fps, protocol, episode,
K_main, K_uw}. start_now_ns = host monotonic ns that the episode's earliest recorded stamp maps to (replay origin);
first_tick_ns = host monotonic ns of the first main frame's capture time (the first packet sent is the next frame:
RigFrameSource.start consumes the first pair).

Usage:
  python -m franka_deploy.stream.rig_replay_sender [--episode RAW_EP | --config sender.yaml] --host 127.0.0.1
      [--ports 5555] [--protocol v2] [--fps 30] [--link run/link/wifi_good_skew.yaml] [--truth T.jsonl]
      [--ready-file F] [--loop] [--cache-dir D] [--max-s S] [--start-delay-s 1.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import yaml

from sensor_rig.stream.sender import RigStreamSender

from ..paths import REPO_ROOT
from .client import StreamClient
from .link import LinkConfig

DEFAULT_RAW_ROOT = "data/raw/2026-10-01"
DEFAULT_GENERATE = {"raw_root": DEFAULT_RAW_ROOT, "episode_index": 0, "episode_dir": None, "n_frames": 300,
                    "main_fps_hz": 30}


class LinkTransport:
    """RigStreamSender transport over StreamClient (v2) + LinkEmulator. The rig stamps are host-monotonic here, the
    client maps them to the link's skewed sender clock and writes the truth file."""

    def __init__(self, host: str, port: int, link: LinkConfig | None, truth_path: str | None = None,
                 max_inflight: int = 8, block_timeout_s: float = 1.0):
        self.host, self.port, self.link, self.truth_path = host, int(port), link, truth_path
        self.max_inflight, self.block_timeout_s = int(max_inflight), float(block_timeout_s)
        self.client: StreamClient | None = None
        self.blocked_s = 0.0

    def connect(self, hello: dict, timeout_s: float = 15.0) -> None:
        meta = {"streams": hello["streams"], "fps_hz": hello["fps_hz"], "K": hello["K"]}
        self.client = StreamClient(self.host, {"main": self.port}, "v2", link=self.link, truth_path=self.truth_path,
                                   sender=hello["sender"], v2_meta=meta)
        self.client.connect(timeout_s)

    def send_packet(self, pkt: dict, size: int) -> None:
        c = self.client
        if c._link is not None and self.max_inflight > 0:        # bounded "socket buffer"
            t0 = time.monotonic()
            while c._link.stats()["queued"] >= self.max_inflight and time.monotonic() - t0 < self.block_timeout_s:
                time.sleep(0.001)
            self.blocked_s += time.monotonic() - t0
        c.send_v2(pkt["main_jpeg"], pkt["uw_jpeg"], pkt["t_main_ns"] / 1e9, pkt["t_uw_ns"] / 1e9, pkt["seq"],
                  w=pkt["w"], h=pkt["h"])

    def answer_ping(self, data=None) -> dict:
        t1 = self.client.sender_clock_ns()
        return {"t1_ns": t1, "t2_ns": self.client.sender_clock_ns()}

    def stats(self) -> dict:
        return {**(self.client.stats() if self.client else {}), "blocked_s": round(self.blocked_s, 3)}

    def flush(self, timeout_s: float = 5.0) -> bool:
        return self.client.flush(timeout_s) if self.client else True

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


# ------------------------------------------------------------------------------------------------ episode
def find_iphone_episode(gen: dict) -> Path:
    """The raw iPhone episode to simulate from: `episode_dir` (a name under raw_root, or an absolute path) wins, else
    `episode_index` into the sorted firstreal demonstration episodes (with a video) of raw_root."""
    root = Path(str(gen.get("raw_root") or DEFAULT_RAW_ROOT))
    root = root if root.is_absolute() else REPO_ROOT / root
    d = gen.get("episode_dir")
    if d:
        p = Path(str(d))
        return p if p.is_absolute() else root / p
    eps = sorted(p for p in root.iterdir() if "firstreal_demonstration" in p.name
                 and (p / (p.name + "_rgb.mp4")).is_file()) if root.is_dir() else []
    i = int(gen.get("episode_index", 0))
    if not 0 <= i < len(eps):
        raise FileNotFoundError(f"rig_raw_replay generate.episode_index {i}: {len(eps)} firstreal demonstration "
                                f"episode(s) in {root}")
    return eps[i]


def generate_raw_episode(gen: dict, cache_dir: Path) -> Path:
    """Simulate (once, cached) a short raw rig episode from an iPhone episode; returns the raw episode directory."""
    from sensor_rig.sim.simulator import SIM_RIG_PATH, SimOptions, simulate_raw_from_iphone

    src = find_iphone_episode(gen)
    if not src.is_dir():
        raise FileNotFoundError(f"iPhone episode {src} not found (rig_raw_replay generate.*)")
    key = hashlib.sha1(json.dumps([src.name, gen.get("n_frames"), gen.get("main_fps_hz"), SIM_RIG_PATH.name],
                                  sort_keys=True).encode()).hexdigest()[:10]
    root = Path(cache_dir) / f"{src.name[:40]}_{key}"
    done = root / ".complete"
    if not done.exists():
        tmp = Path(str(root) + f".tmp{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        opts = SimOptions(max_frames=gen.get("n_frames"), main_fps=gen.get("main_fps_hz"), import_trajectory=False)
        simulate_raw_from_iphone(src, SIM_RIG_PATH, tmp, opts)
        (tmp / ".complete").write_text(src.name)
        shutil.rmtree(root, ignore_errors=True)
        os.replace(tmp, root)
    eps = sorted(p for p in root.glob("*/*") if (p / "rig.yaml").is_file())
    if not eps:
        raise RuntimeError(f"generated rig episode missing under {root}")
    return eps[0]


def resolve_episode(episode: str | None, cfg: dict, cache_dir: str | None) -> Path:
    sim = (cfg or {}).get("sim") or {}
    path = episode if episode and episode != "None" else sim.get("raw_episode_dir")
    if path:
        p = Path(str(path))
        if not (p / "rig.yaml").is_file():
            raise FileNotFoundError(f"raw rig episode {p}: no rig.yaml")
        return p
    gen = {**DEFAULT_GENERATE, **(sim.get("generate") or {})}
    return generate_raw_episode(gen, Path(cache_dir or (Path.home() / "franka_runs" / "cache" / "rig_episodes")))


# ------------------------------------------------------------------------------------------------ run
def _write_ready(path: str, d: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)


def run(args) -> dict:
    from sensor_rig.raw_episode import RawEpisode

    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    sc = cfg.get("sender") or {}
    ep_dir = resolve_episode(args.episode, cfg, args.cache_dir)
    ep = RawEpisode(ep_dir)
    rig = ep.rig()
    port = int(args.ports.split(",")[0])
    link = LinkConfig.from_yaml(args.link) if args.link else (LinkConfig.from_yaml(cfg["link"]) if cfg.get("link") else None)
    t_main, t_uw = ep.camera_times("main"), ep.camera_times("ultrawide")
    origin = min(int(t_main[0]), int(t_uw[0]))
    start_ns = args.start_now_ns if args.start_now_ns is not None else time.monotonic_ns() + int(args.start_delay_s * 1e9)
    transport = LinkTransport(args.host, port, link, args.truth, max_inflight=args.max_inflight)
    sender = RigStreamSender(
        rig, args.host, port, fps_hz=args.fps, jpeg_quality=args.jpeg_quality or int(sc.get("jpeg_quality", 90)),
        backend="mock", backend_opts={"episode_dir": str(ep_dir), "loop": bool(args.loop), "realtime": True,
                                      "start_now_ns": int(start_ns)},
        max_queue=int(sc.get("max_queue", 4)), drop_policy=sc.get("drop_policy", "drop_oldest"),
        transport=transport, sender_name="rig_replay_sender", startup_timeout_s=10.0, connect_timeout_s=args.connect_timeout)
    print(f"[rig-replay] {ep.name}: {len(t_main)} main / {len(t_uw)} ultrawide frames, link "
          f"{'none' if link is None else link}, fps {args.fps}", file=sys.stderr)
    if args.ready_file:
        K = sender.hello()["K"]
        t0_cap = int(rig.capture_time_ns(rig.virtual["main"].source, int(t_main[0])))
        _write_ready(args.ready_file, {
            "start_now_ns": int(start_ns), "first_tick_ns": int(start_ns + t0_cap - origin), "fps": args.fps,
            "protocol": "v2", "episode": ep.name, "K_main": K["main"], "K_uw": K["ultrawide"]})
    signal.signal(signal.SIGTERM, lambda *_: sender.stop())
    try:
        stats = sender.run(args.max_s)
        transport.flush(5.0)
    finally:
        sender.close()
    return {k: v for k, v in stats.items() if k != "source"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episode", default=None, help="raw rig episode directory (default: from --config sim / generated)")
    ap.add_argument("--config", default=None, help="sender.yaml written by the launcher (sender, link, sim sections)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", default="5555")
    ap.add_argument("--protocol", choices=["v2"], default="v2")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--jpeg-quality", type=int, default=0, help="0: sender.jpeg_quality of --config (default 90)")
    ap.add_argument("--link", default=None, help="run/link/*.yaml (default: the config's link, else no emulation)")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--start-now-ns", type=int, default=None, help="host monotonic ns of the replay origin")
    ap.add_argument("--start-delay-s", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=None, help="stop after S seconds")
    ap.add_argument("--max-inflight", type=int, default=8, help="emulated socket buffer (packets) before blocking")
    ap.add_argument("--cache-dir", default=None, help="where a generated raw rig episode is cached")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args(argv)
    res = run(args)
    print("[rig-replay] done " + json.dumps(res, default=str), file=sys.stderr)
    return 1 if res.get("capture_error") else 0


if __name__ == "__main__":
    sys.exit(main())
