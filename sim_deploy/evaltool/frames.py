"""Key-frame strip per rollout: start / closest approach / gripper close / end, from external.mp4 (+ the tap's main
camera image nearest each key time when the rollout has tap images) -> ``<eval>/rollouts/NNN/frames.jpg``.

Times come from score.py (same rules as score.json). Video frame i is taken at t = i / video_fps (the driver records
the rate it actually rendered; the video starts with the rollout). A tap image's time is its chunk's anchor time
minus the rollout's first chunk anchor (about 0.2 s after the rollout start), so it is approximate (+- one chunk).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sim_deploy.evaltool.score import _jsonl, load_eval, score_rollout

TILE_W = 480


def _video_frame(mp4: Path, t: float, fps: float | None):
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 10.0
    i = max(0, min(n - 1 if n else 0, int(round(t * fps))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def _tap_index(rollout_dir: Path) -> list[tuple[float, Path]]:
    """[(t_s since the first chunk, npz)] for tap chunks that hold images."""
    tap = rollout_dir / "tap"
    if not tap.is_dir():
        return []
    steps = _jsonl(rollout_dir / "steps.jsonl")
    anchors = {int(s["chunk_id"]): int(s["t_anchor_ns"]) for s in steps if s.get("t_anchor_ns") is not None}
    if not anchors:
        return []
    t0 = min(anchors.values())
    out = []
    for f in tap.glob("chunk_*.npz"):
        cid = int(f.stem.split("_")[1])
        if cid not in anchors:
            continue
        try:
            with np.load(f) as z:
                if "main" not in z.files:
                    continue
        except (OSError, ValueError):
            continue
        out.append(((anchors[cid] - t0) / 1e9, f))
    return sorted(out)


def _tap_image(idx: list, t: float):
    if not idx:
        return None
    _, f = min(idx, key=lambda x: abs(x[0] - t))
    with np.load(f) as z:
        img = np.asarray(z["main"])
    while img.ndim > 3:                        # (history, H, W, C): newest
        img = img[-1]
    if img.dtype != np.uint8:
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img.reshape(img.shape[0], img.shape[1]), cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _fit(img, w: int):
    h = int(round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _label(img, text: str):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _blank(w: int, h: int, text: str):
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.putText(img, text, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def key_times(sc: dict) -> list[tuple[str, float | None]]:
    end = sc.get("duration_s")
    return [("start", 0.0), ("closest approach", sc.get("t_approach")), ("gripper close", sc.get("t_close")),
            ("end", end)]


def frames_rollout(eval_dir: Path, rec: dict) -> Path | None:
    eval_dir = Path(eval_dir)
    n = int(rec.get("n", 0))
    rollout_dir = eval_dir / "rollouts" / f"{n:03d}"
    mp4 = rollout_dir / "external.mp4"
    if not mp4.is_file():
        return None
    sc = score_rollout(eval_dir, rec)
    fps = rec.get("video_fps")
    tap = _tap_index(rollout_dir)
    cols = []
    for name, t in key_times(sc):
        if t is None:
            top = _blank(TILE_W, 360, f"{name}: n/a")
            bot = _blank(TILE_W, 270, "") if tap else None
        else:
            v = _video_frame(mp4, float(t), fps)
            top = _label(_fit(v, TILE_W), f"{name}  t = {t:.2f} s") if v is not None else _blank(TILE_W, 360, "no frame")
            ti = _tap_image(tap, float(t)) if tap else None
            bot = (_label(_fit(ti, TILE_W), f"tap main ~{t:.2f} s") if ti is not None else _blank(TILE_W, 270, "no tap image")) \
                if tap else None
        cols.append((top, bot))
    th = max(c[0].shape[0] for c in cols)
    bh = max((c[1].shape[0] for c in cols if c[1] is not None), default=0)
    tiles = []
    for top, bot in cols:
        top = cv2.copyMakeBorder(top, 0, th - top.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        if bh:
            bot = bot if bot is not None else _blank(TILE_W, bh, "")
            bot = cv2.copyMakeBorder(bot, 0, bh - bot.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            top = np.vstack([top, bot])
        tiles.append(top)
    strip = np.hstack(tiles)
    head = np.zeros((30, strip.shape[1], 3), np.uint8)
    res = "success" if sc.get("success") else ("fail" if sc.get("success") is False else "?")
    cv2.putText(head, f"{eval_dir.name} ep {n:03d} seed {json.dumps(sc.get('seed_key'))} {res} "
                f"stop {sc.get('stop_reason')} valid {sc['validity']['valid']}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out = rollout_dir / "frames.jpg"
    cv2.imwrite(str(out), np.vstack([head, strip]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def frames_eval(eval_dir: Path, rollout: int | None = None) -> list[Path]:
    _, recs = load_eval(Path(eval_dir))
    out = []
    for r in recs:
        if rollout is not None and int(r.get("n", -1)) != rollout:
            continue
        p = frames_rollout(Path(eval_dir), r)
        if p:
            out.append(p)
    return out
