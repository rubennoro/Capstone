"""``bench-encode``: JPEG encode cost and throughput per worker count (P1 §2.3, §3.4 budget check on the Jetson)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sensor_rig.record.jpeg import make_encoder


def synthetic_image(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Camera-like content (smooth gradients + texture + mild noise); JPEG cost depends on content."""
    import cv2

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack([
        128 + 100 * np.sin(xx / 37.0) * np.cos(yy / 53.0),
        128 + 90 * np.sin((xx + yy) / 71.0),
        128 + 80 * np.cos(xx / 23.0 - yy / 41.0),
    ], axis=-1)
    tex = cv2.GaussianBlur(rng.normal(0, 40, (height, width, 3)).astype(np.float32), (0, 0), 1.5)
    img = base + tex + rng.normal(0, 4, (height, width, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def bench_encode(width: int, height: int, quality: int = 90, n: int = 300, workers=(1, 2, 3),
                 encoder: str = "opencv", print_fn=print) -> list[dict]:
    enc = make_encoder(encoder, quality)
    imgs = [synthetic_image(width, height, s) for s in range(4)]
    for im in imgs:          # warm up
        enc.encode(im)
    results = []
    sizes = [len(enc.encode(im)) for im in imgs]
    for k in workers:
        per_frame: list[float] = []

        def job(i):
            t0 = time.perf_counter()
            enc.encode(imgs[i % len(imgs)])
            return time.perf_counter() - t0

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=k) as ex:
            per_frame = list(ex.map(job, range(n)))
        wall = time.perf_counter() - t0
        r = {"encoder": encoder, "width": width, "height": height, "quality": quality, "workers": k, "n": n,
             "ms_per_frame_p50": float(np.median(per_frame) * 1e3),
             "ms_per_frame_p99": float(np.percentile(per_frame, 99) * 1e3),
             "frames_per_s": n / wall, "kbytes_mean": float(np.mean(sizes) / 1e3)}
        results.append(r)
        if print_fn:
            print_fn(f"{encoder} {width}x{height} Q{quality} workers={k}: {r['ms_per_frame_p50']:.2f} ms/frame "
                     f"(p99 {r['ms_per_frame_p99']:.2f}), {r['frames_per_s']:.1f} frames/s, "
                     f"~{r['kbytes_mean']:.0f} KB/frame")
    return results
