# copied from rnoro/deployment:deployment/process_frames/process_frames/probe.py @ 91b178b
# changes: the station schema (C1 on 127.0.0.1:<ports.frames>); --station optional; --mock (no pi.address needed).
"""``python -m policy_deploy.frames.probe [--station FILE] [--mock] [--save DIR]``: ask a running frames once.

Sends one get_health and one get_frames to ``tcp://127.0.0.1:<ports.frames>``, prints the health table, the four image times, their
ages and spacings, and with ``--save`` writes the four 224x224 images as PNG, to check them by eye.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from deploy_contracts import ContractError, RemoteError, RpcClient, TransportTimeout

from policy_deploy.contracts import FramesReply, GetFramesRequest, GetHealthRequest, HealthReply

from policy_deploy.frames.errors import SettingsError
from policy_deploy.frames.settings import Settings


def _health_table(health) -> str:
    rows = ["camera     status  age_ms    fps  dropped"]
    for cam in ("main", "ultrawide"):
        r = getattr(health, cam)
        rows.append(f"{cam:<10} {r.status:<6} {r.last_frame_age_ms:7.0f} {r.fps:6.1f} {r.dropped:8d}")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m policy_deploy.frames.probe", description=__doc__)
    p.add_argument("--station", type=Path, default=None)
    p.add_argument("--mock", action="store_true", help="the station as under --mock (pi.address not needed)")
    p.add_argument("--save", type=Path, default=None, help="write the four images as PNG into this folder")
    p.add_argument("--live", type=float, nargs="?", const=5.0, default=None, metavar="HZ",
                   help="keep asking (HZ times a second, default 5) and show the four images in a window: what the "
                        "policy gets. With --save, every set is written. q or Esc in the window stops. Needs an "
                        "OpenCV with a GUI: run it with the torch environment's python")
    args = p.parse_args(argv)
    try:
        s = Settings.load(args.station, mock=args.mock)
    except SettingsError as e:
        print(f"probe: {e}", file=sys.stderr)
        return 1
    if args.live is not None:
        return _live(s, args.live, args.save)
    client = RpcClient(name="frames", endpoint=s.frames)
    try:
        health = client.request(GetHealthRequest(), HealthReply, s.probe.timeout_s).health
        print(_health_table(health))
        reply = client.request(GetFramesRequest(), FramesReply, s.probe.timeout_s)
    except (TransportTimeout, RemoteError, ContractError) as e:
        print(f"probe: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()
    now = time.monotonic_ns()
    if reply.main is None:
        print("get_frames: no images (a camera is not ok, or no frame within the picking tolerance)")
        print(_health_table(reply.health))
        return 2
    for cam, times in (("main", reply.main_t_ns), ("ultrawide", reply.ultrawide_t_ns)):
        ages = ", ".join(f"{(now - int(t)) / 1e6:.1f}" for t in times)
        spacing_ms = (int(times[-1]) - int(times[0])) / 1e6
        print(f"{cam}: t_ns {[int(t) for t in times]}, ages {ages} ms, spacing {spacing_ms:.1f} ms")
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for cam, imgs in (("main", reply.main), ("ultrawide", reply.ultrawide)):
            for i, img in enumerate(imgs):
                path = args.save / f"{cam}_{i}.png"
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"wrote {path}")
    return 0


LIVE_WINDOW = "probe --live: what the policy gets (q to stop)"
LIVE_SCALE = 2          # the 224x224 images shown at 448x448


def _live_frame(reply, now_ns: int) -> np.ndarray:
    """The four images as a 2x2 grid (rows main, ultrawide; columns the two observation steps), BGR, labelled."""
    rows = []
    for cam, imgs, times in (("main", reply.main, reply.main_t_ns), ("ultrawide", reply.ultrawide,
                                                                      reply.ultrawide_t_ns)):
        tiles = []
        for i, (img, t) in enumerate(zip(imgs, times)):
            tile = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), None, fx=LIVE_SCALE, fy=LIVE_SCALE,
                              interpolation=cv2.INTER_NEAREST)
            label = f"{cam}_{i}  age {(now_ns - int(t)) / 1e6:.0f} ms"
            cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    return np.vstack(rows)


def _live(s: Settings, hz: float, save: Path | None) -> int:
    """Ask frames for images hz times a second and show them until q, Esc or Ctrl-C."""
    from policy_deploy.frames.pi_collect import _has_gui
    if not hz > 0:
        print(f"probe: --live must be > 0, got {hz}", file=sys.stderr)
        return 1
    if not _has_gui():
        print("probe: --live needs an OpenCV with a GUI (this python's has none); use "
              "~/envs/policy_deploy_torch/bin/python", file=sys.stderr)
        return 1
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
    client = RpcClient(name="frames", endpoint=s.frames)
    print(f"probe: asking {s.frames} {hz:g} times a second; q or Esc in the window stops", flush=True)
    period = 1.0 / hz
    shown = 0
    blank = np.zeros((2 * 224 * LIVE_SCALE, 2 * 224 * LIVE_SCALE, 3), np.uint8)
    try:
        while True:
            start = time.monotonic()
            try:
                reply = client.request(GetFramesRequest(), FramesReply, s.probe.timeout_s)
            except (TransportTimeout, RemoteError, ContractError) as e:
                frame = blank.copy()
                cv2.putText(frame, f"no reply: {type(e).__name__}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                reply = None
            if reply is not None and reply.main is None:
                frame = blank.copy()
                cv2.putText(frame, "no images (a camera is not ok)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
            elif reply is not None:
                frame = _live_frame(reply, time.monotonic_ns())
                if save is not None:
                    for cam, imgs in (("main", reply.main), ("ultrawide", reply.ultrawide)):
                        for i, img in enumerate(imgs):
                            cv2.imwrite(str(save / f"{shown:06d}_{cam}_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                shown += 1
            cv2.imshow(LIVE_WINDOW, frame)
            wait_ms = max(1, int((period - (time.monotonic() - start)) * 1000))
            if (cv2.waitKey(wait_ms) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        cv2.destroyAllWindows()
    print(f"probe: {shown} image sets shown" + (f", written to {save}" if save is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
