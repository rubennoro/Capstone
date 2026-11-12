"""``python -m process_frames.probe --station FILE [--save DIR]``: ask a running process_frames once (STRUCTURE.md §12).

Sends one get_health and one get_frames to ``endpoints.frames``, prints the health table, the four image times, their
ages and spacings, and with ``--save`` writes the four 224x224 images as PNG, to check them by eye.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2

from contracts import (ContractError, FramesReply, GetFramesRequest, GetHealthRequest, HealthReply, RemoteError,
                       RpcClient, TransportTimeout)

from process_frames.errors import SettingsError
from process_frames.settings import Settings


def _health_table(health) -> str:
    rows = ["camera     status  age_ms    fps  dropped"]
    for cam in ("main", "ultrawide"):
        r = getattr(health, cam)
        rows.append(f"{cam:<10} {r.status:<6} {r.last_frame_age_ms:7.0f} {r.fps:6.1f} {r.dropped:8d}")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m process_frames.probe", description=__doc__)
    p.add_argument("--station", type=Path, required=True)
    p.add_argument("--save", type=Path, default=None, help="write the four images as PNG into this folder")
    args = p.parse_args(argv)
    try:
        s = Settings.load(args.station, mock=False)
    except SettingsError as e:
        print(f"probe: {e}", file=sys.stderr)
        return 1
    client = RpcClient(name="process_frames", endpoint=s.frames)
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


if __name__ == "__main__":
    sys.exit(main())
