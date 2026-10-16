"""calib-check: render what the virtual cameras see (files only, works headless over ssh).

Per camera ``<cam>``: ``<cam>_physical.png`` (raw frame, virtual border in green, model limit in red),
``<cam>_virtual.png`` (virtual image, uncovered pixels tinted red, 10 % grid, principal point, coverage label),
``<cam>_virtual_224.png`` (pad-to-square + resize 224, what the policy sees) and, with an iPhone reference,
``<cam>_blend.png`` / ``<cam>_edges.png``. ``report.json`` holds coverage, finger-tag depth/separation in the
virtual ultrawide, tag-corner RMS vs the reference, the rig fingerprint and the placeholder flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from sensor_rig.calib.align import TAG_IDS, detect_tags
from sensor_rig.camera_model import FisheyeEquidistant, PinholeRadtan, virtual_border_in_physical

__all__ = ["pad_resize_square", "render_check", "run_calib_check", "load_reference_iphone"]

GREEN, RED, MAGENTA, YELLOW = (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)


def pad_resize_square(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Centre-pad to a square with black, then INTER_AREA resize (iphumi ``get_image_transform_with_border``)."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def _limit_radius_px(model) -> float | None:
    """Image radius (normalised units) of the model's monotonic limit, or None if it is far outside."""
    if isinstance(model, PinholeRadtan):
        r = model.r_valid
        rd = r * (1 + model.k1 * r * r + model.k2 * r**4)
    elif isinstance(model, FisheyeEquidistant):
        rd = float(model._theta_d(np.float64(model.theta_valid)))
    else:
        return None
    return rd if rd * model.fx < 4 * np.hypot(model.width, model.height) else None


def _draw_polyline_nan(img, pts, color, thickness=2):
    seg = []
    for p in list(pts) + [np.array([np.nan, np.nan])]:
        if np.all(np.isfinite(p)):
            seg.append(p)
        elif seg:
            if len(seg) > 1:
                cv2.polylines(img, [np.round(np.array(seg) * 16).astype(np.int32)], False, color, thickness,
                              cv2.LINE_AA, shift=4)
            seg = []


def _label(img, text, org=(10, 30), scale=None):
    scale = scale or max(0.5, img.shape[1] / 1200)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _physical_panel(rig, cam, frame):
    v = rig.virtual[cam]
    model = rig.cameras[v.source].model
    out = frame.copy()
    rd = _limit_radius_px(model)
    if rd is not None:
        cv2.ellipse(out, (int(round(model.cx)), int(round(model.cy))),
                    (int(round(rd * model.fx)), int(round(rd * model.fy))), 0, 0, 360, RED, 2, cv2.LINE_AA)
    _draw_polyline_nan(out, virtual_border_in_physical(model, v.reference, v.R_phys_virt), GREEN, 3)
    _label(out, f"{v.source} physical {model.width}x{model.height} ({model.kind}); green = virtual {cam} border")
    return out


def _virtual_panel(virt_img, table, vcam, coverage):
    out = virt_img.copy()
    if not table.valid.all():
        red = np.zeros_like(out)
        red[:] = RED
        out[~table.valid] = cv2.addWeighted(out, 0.4, red, 0.6, 0)[~table.valid]
    h, w = out.shape[:2]
    for k in range(1, 10):
        cv2.line(out, (round(k * w / 10), 0), (round(k * w / 10), h - 1), (128, 128, 128), 1)
        cv2.line(out, (0, round(k * h / 10)), (w - 1, round(k * h / 10)), (128, 128, 128), 1)
    c = (int(round(vcam.cx)), int(round(vcam.cy)))
    cv2.drawMarker(out, c, YELLOW, cv2.MARKER_CROSS, max(12, w // 40), 2)
    _label(out, f"coverage = {coverage:.4f}")
    return out


def _tag_metrics(tags: dict) -> tuple[float | None, float | None]:
    if not tags:
        return None, None
    z = float(np.median([t["tvec"][2] for t in tags.values()]))
    sep = None
    if all(i in tags for i in TAG_IDS[:2]):
        sep = float(abs(tags[TAG_IDS[0]]["tvec"][0] - tags[TAG_IDS[1]]["tvec"][0]))
    return z, sep


def _draw_tags(img, tags):
    for i, t in tags.items():
        c = t["corners"]
        cv2.polylines(img, [np.round(c * 16).astype(np.int32)], True, GREEN, 2, cv2.LINE_AA, shift=4)
        _label(img, f"id{i} z={t['tvec'][2]:.3f}m", (int(c[:, 0].min()), int(c[:, 1].min()) - 6), 0.5)


def _check(rig, frames, purpose, reference, detect):
    panels, metrics = {}, {"tag_z_m": None, "tag_sep_m": None, "rms_vs_reference_px": None}
    cov_report = rig.coverage_report()
    for cam, frame in frames.items():
        if cam not in rig.virtual:
            raise KeyError(f"no virtual camera {cam!r} (have {list(rig.virtual)})")
        v = rig.virtual[cam]
        if frame.shape[:2] != tuple(rig.cameras[v.source].resolution)[::-1]:
            raise ValueError(f"{cam}: frame is {frame.shape[1]}x{frame.shape[0]}, the rig's {v.source} camera "
                             f"is {rig.cameras[v.source].resolution[0]}x{rig.cameras[v.source].resolution[1]}")
        frame = np.ascontiguousarray(frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        table = rig.remap(cam, purpose)
        vcam = rig.virtual_camera(cam, purpose)
        virt = table.apply(frame)
        panels[f"{cam}_physical"] = _physical_panel(rig, cam, frame)
        vp = _virtual_panel(virt, table, vcam, cov_report[cam])
        panels[f"{cam}_virtual_224"] = pad_resize_square(virt)
        tags = detect_tags(virt, vcam.K) if (detect and cam == "ultrawide") else {}
        if tags:
            _draw_tags(vp, tags)
            metrics["tag_z_m"], metrics["tag_sep_m"] = _tag_metrics(tags)
        panels[f"{cam}_virtual"] = vp
        ref = (reference or {}).get(cam)
        if ref is not None:
            if ref.shape[:2] != virt.shape[:2]:
                ref = cv2.resize(ref, (virt.shape[1], virt.shape[0]), interpolation=cv2.INTER_AREA)
            panels[f"{cam}_blend"] = cv2.addWeighted(virt, 0.5, ref, 0.5, 0)
            edges = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 60, 150) > 0
            e = virt.copy()
            e[edges] = MAGENTA
            panels[f"{cam}_edges"] = e
            if tags:
                rtags = detect_tags(ref, vcam.K)
                common = [i for i in tags if i in rtags]
                if common:
                    d = np.concatenate([tags[i]["corners"] - rtags[i]["corners"] for i in common])
                    metrics["rms_vs_reference_px"] = float(np.sqrt(np.mean(np.sum(d**2, axis=1))))
    return panels, metrics


def render_check(rig, frames: dict[str, np.ndarray], *, purpose="export",
                 reference: dict[str, np.ndarray] | None = None, detect_tags=True) -> dict[str, np.ndarray]:
    """BGR panel images keyed ``<cam>_physical`` / ``_virtual`` / ``_virtual_224`` (/ ``_blend`` / ``_edges``).

    ``frames``: raw physical BGR frames keyed by virtual camera name. ``reference``: iPhone frames of the same
    scene (any uniform scale of the virtual output size).
    """
    return _check(rig, frames, purpose, reference, detect_tags)[0]


def load_reference_iphone(export_dir, frame: int = 0) -> dict[str, np.ndarray]:
    """First (or ``frame``-th) main and ultrawide frame of an iPhUMI export folder (calibrations: UW only)."""
    from sensor_rig.calib.align import _decode_video, load_iphone_episode
    _, stem = load_iphone_episode(export_dir)
    out = {}
    for cam, suffix in (("main", "_rgb.mp4"), ("ultrawide", "_ultrawidergb.mp4")):
        p = Path(str(stem) + suffix)
        if p.is_file():
            fr = _decode_video(p, max_frames=frame + 1)
            if fr:
                out[cam] = fr[min(frame, len(fr) - 1)]
    return out


def _frames_from_source(rig, source, kw) -> dict[str, np.ndarray]:
    cams = [c for c in kw.get("cameras") or rig.virtual]
    if source == "images":
        images = kw["images"]
        if isinstance(images, (list, tuple)):
            images = dict(zip(["main", "ultrawide"], images))
        out = {}
        for cam, p in images.items():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"cannot read image {p}")
            out[cam] = img
        return out
    if source == "episode":
        from sensor_rig.raw_episode import RawEpisode
        ep = RawEpisode(Path(kw["episode"]))
        out = {}
        for cam in cams:
            src = rig.virtual[cam].source
            n = len(ep.camera_times(src))
            i = kw.get("frame")
            out[cam] = ep.read_image(src, n // 2 if i is None else int(i))
        return out
    if source == "live":
        from sensor_rig.drivers.base import make_camera
        out = {}
        for cam in cams:
            src = rig.virtual[cam].source
            dev = make_camera(rig, src, backend=kw.get("backend"), **(kw.get("backend_opts") or {}))
            dev.start()
            try:
                f = None
                for _ in range(int(kw.get("warmup_frames", 10))):
                    f = dev.read(timeout_s=float(kw.get("timeout_s", 2.0))) or f
                if f is None:
                    raise RuntimeError(f"no frame from camera {src}")
                out[cam] = f.image
            finally:
                dev.stop()
        return out
    raise ValueError(f"unknown source {source!r}; use live, episode or images")


def run_calib_check(rig, source: Literal["live", "episode", "images"], out_dir, *, purpose="export",
                    reference_iphone=None, detect_tags=True, **kw) -> dict:
    """Grab frames from ``source``, write the panels as PNGs and ``report.json`` into ``out_dir``; return the report.

    ``kw``: ``images`` ({cam: path} or [main, ultrawide]) | ``episode`` (raw dir) + ``frame`` |
    ``backend`` / ``backend_opts`` (live); optional ``cameras``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = _frames_from_source(rig, source, kw)
    reference = load_reference_iphone(reference_iphone) if reference_iphone else None
    panels, metrics = _check(rig, frames, purpose, reference, detect_tags)
    for name, img in panels.items():
        cv2.imwrite(str(out / f"{name}.png"), img)
    report = {
        "coverage": rig.coverage_report(),
        **metrics,
        "purpose": purpose,
        "source": source,
        "cameras": list(frames),
        "rig_fingerprint": rig.fingerprint(),
        "placeholder": rig.placeholder,
        "files": sorted(f"{n}.png" for n in panels),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
