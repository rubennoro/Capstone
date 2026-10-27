"""Validate that a rollout episode dir (or every episode of a session) is complete and self-consistent.

    python -m franka_deploy validate <episode_dir | session_dir> [--mode auto|real|sim]

Checks: config snapshot (deploy, bridge, controller yamls, checksums incl. ckpt sha256 + prompt index, git, package
versions); label.json matches the stop event; events.jsonl has episode_start/stop; summary.json duration and policy
calls; tap/ has exactly `policy_calls` npz files, each passing the obs/reply contract; controller_trace.npz covers
the episode at ~200 Hz with the required columns; bridge.jsonl has per-second stats inside the episode (Franka);
camera_stats.txt; the camera videos (libx264) exist and decode with frames for ~the episode duration; MainLogger
metadata exists.

Modes (W10B): `real` is the full check above. `sim` (auto: config/deploy_resolved.yaml says run_mode: sim) accepts
the two by-design differences of sim sessions: no camera video when camera.video is off (the wireless receiver records
none in sim), and a checksums.json without checkpoint entries (replay / fake policies stage no model); the prompt zarr
sha256 + prompt index are then required instead. A checkpoint entry, when present, must still carry its sha256.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from franka_deploy.tools.analyze_tap import CONTRACT_OBS, CONTRACT_REPLY

TRACE_COLS = ("t", "fk", "commanded", "width_measured", "width_commanded", "held", "fault")


MODES = ("auto", "real", "sim")


def episode_mode(d: str, mode: str = "auto") -> str:
    """real | sim. auto: run_mode of config/deploy_resolved.yaml; real when it is missing or unreadable (strict)."""
    if mode != "auto":
        return mode
    import yaml

    try:
        with open(os.path.join(d, "config", "deploy_resolved.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return "real"
    if not isinstance(cfg, dict):
        return "real"
    return "sim" if str(cfg.get("run_mode", "real")) == "sim" else "real"


def _video_expected(d: str) -> bool:
    import yaml

    try:
        cfg = yaml.safe_load(open(os.path.join(d, "config", "deploy_resolved.yaml"))) or {}
        return bool((cfg.get("camera") or {}).get("video", True))
    except (OSError, ValueError, yaml.YAMLError, AttributeError):
        return True


def validate_episode(d: str, mode: str = "auto") -> list[str]:
    """Complete episodes: everything below. Aborted episodes (label "aborted:<reason>", e.g. a killed process or
    Ctrl+C): the same, except the video may be truncated/missing and the trace may end early (it is merged from the
    1 s parts by the salvage) - but config, label, summary, events, taps and (for > 2 s) the trace must exist."""
    errs = []
    sim = episode_mode(d, mode) == "sim"
    J = lambda f: json.load(open(os.path.join(d, f)))  # noqa: E731
    lab0 = str(J("label.json").get("label", "")) if os.path.exists(os.path.join(d, "label.json")) else ""
    aborted = lab0.startswith("aborted:")
    abnormal = aborted or lab0.startswith("exception:")  # ended by a fault: the trace may end early
    req = ["summary.json", "label.json", "events.jsonl"] + ([] if aborted else ["controller_trace.npz", "camera_stats.txt"])
    for f in req:
        if not os.path.exists(os.path.join(d, f)):
            errs.append(f"missing {f}")
    for f in ("deploy.yaml", "bridge.yaml", "franka_cartesian.yaml", "checksums.json", "git.json", "versions.json"):
        if not os.path.exists(os.path.join(d, "config", f)):
            errs.append(f"missing config/{f}")
    if errs:
        return errs
    s, lab = J("summary.json"), J("label.json")
    ck = J("config/checksums.json")
    if sim and "checkpoint" not in ck:      # replay / fake policy: no model staged, the prompt zarr is still pinned
        if len(ck.get("prompt_zarr_sha256", "")) != 64 or "prompt_index" not in ck:
            errs.append("checksums.json incomplete (sim, no checkpoint: prompt_zarr_sha256 + prompt_index required)")
    elif len(ck.get("checkpoint_sha256", "")) != 64 or "prompt_index" not in ck:
        errs.append("checksums.json incomplete")
    evs = [json.loads(ln) for ln in open(os.path.join(d, "events.jsonl"))]
    kinds = [e["kind"] for e in evs]
    if "episode_start" not in kinds or ("episode_stop" not in kinds and not aborted):
        errs.append("events: no episode_start/episode_stop")
    stop = [e for e in evs if e["kind"] == "episode_stop"]
    if stop and stop[-1].get("label") not in (None, lab["label"]) and not str(lab["label"]).startswith("exception"):
        errs.append(f"label {lab['label']} != stop event {stop[-1].get('label')}")
    if s["label"] != lab["label"]:
        errs.append("summary label != label.json")
    if aborted and s.get("duration_s", 0) > 2.0 and not os.path.exists(os.path.join(d, "controller_trace.npz")):
        errs.append("aborted episode > 2 s without a (salvaged) controller trace")
    taps = sorted(glob.glob(os.path.join(d, "tap", "call_*.npz")))
    if len(taps) != s["policy_calls"]:
        errs.append(f"{len(taps)} taps != {s['policy_calls']} policy calls")
    for f in taps:
        c = np.load(f)
        for k, (shape, dtype) in CONTRACT_OBS.items():
            if f"meta_shape_{k}" not in c or tuple(c[f"meta_shape_{k}"].tolist()) != shape or str(c[f"meta_dtype_{k}"]) != dtype:
                errs.append(f"{os.path.basename(f)}: contract {k}")
        for k, shape in CONTRACT_REPLY.items():
            if c[f"reply_{k}"].shape != shape:
                errs.append(f"{os.path.basename(f)}: reply {k}")
    if not os.path.exists(os.path.join(d, "controller_trace.npz")):
        return errs
    tr = np.load(os.path.join(d, "controller_trace.npz"))
    for k in TRACE_COLS:
        if k not in tr.files:
            errs.append(f"trace missing {k}")
    if "t" in tr.files and len(tr["t"]) > 1:
        dur = float(tr["t"][-1] - tr["t"][0])
        rate = (len(tr["t"]) - 1) / max(dur, 1e-6)
        if abs(dur - s["duration_s"]) > 1.0 and not abnormal:
            errs.append(f"trace spans {dur:.1f} s vs episode {s['duration_s']:.1f} s")
        if rate < 150:
            errs.append(f"trace rate {rate:.0f} Hz < 150")
    else:
        errs.append("trace empty")
    if os.path.exists(os.path.join(d, "config", "bridge.yaml")) and not aborted:
        bj = os.path.join(d, "bridge.jsonl")
        n_stats = sum(1 for ln in open(bj) if '"stats"' in ln) if os.path.exists(bj) else 0
        if n_stats < max(1, int(s["duration_s"]) - 2):
            errs.append(f"bridge.jsonl has {n_stats} stats lines for {s['duration_s']:.0f} s")
    vids = glob.glob(os.path.join(d, "*.zarr", "*.mp4"))
    if aborted:
        vids = []  # may be truncated (encoder killed); not required
    elif vids:
        import cv2

        for v in vids:
            cap = cv2.VideoCapture(v)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 60
            ok, _ = cap.read()
            cap.release()
            if not ok or n < 0.5 * fps * max(s["duration_s"] - 1.0, 1.0):
                errs.append(f"video {os.path.basename(v)}: {n} frames for {s['duration_s']:.1f} s")
    elif not (sim and not _video_expected(d)):
        errs.append("no camera video (*.zarr/*.mp4)")
    if not aborted and not os.path.exists(os.path.join(d, "metadata.zarr")):
        errs.append("missing MainLogger metadata.zarr")
    return errs


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="franka_deploy validate")
    ap.add_argument("path")
    ap.add_argument("--mode", choices=MODES, default="auto",
                    help="auto (run_mode of config/deploy_resolved.yaml) | real (full check) | sim")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    path = a.path
    eps = [path] if os.path.exists(os.path.join(path, "summary.json")) else \
        sorted(glob.glob(os.path.join(path, "**", "episode_*"), recursive=True))
    bad = 0
    for d in eps:
        errs = validate_episode(d, a.mode)
        s = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else {}
        print(f"{'OK  ' if not errs else 'FAIL'} [{episode_mode(d, a.mode)}] {d}  label={s.get('label')} calls={s.get('policy_calls')} "
              f"dur={s.get('duration_s', 0):.1f}s size={s.get('size_mb', 0):.1f}MB rate={s.get('log_rate_mb_per_min', 0):.1f}MB/min")
        for e in errs[:10]:
            print(f"     - {e}")
        bad += bool(errs)
    print(f"{len(eps) - bad}/{len(eps)} episodes valid")
    return 0 if eps and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
