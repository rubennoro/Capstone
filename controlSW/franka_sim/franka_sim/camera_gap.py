"""camera-gap (A7): how differently does the policy act on rendered vs real images? (BP: torch side; no mujoco)

Why: the camera-check numbers (MAE / SSIM) say how far renders are from the training frames; this says what that
does to the actions, which is what decides whether Stage 3 success in sim means anything. The policy is queried
offline exactly like Stage 2a (``predict_demos.predict``: deployment-format observations built from the zarr demo,
the proprio from the demo itself), with ONLY the images swapped per variant, and the SAME diffusion noise: each
variant runs on its own server started with the same ``--noise-seed-base`` (the server seeds call j with base + j and
``policy_reset`` does not reset j, so one shared server would give each variant different noise; deviation from the
spec's "one Server, reset between variants", reported). ``real_repeat`` checks that this is exact.

Variants (224 RGB images per frame that ``deploy_obs`` reads; renders from camera-check ``renders.npz``):
  real            the zarr frames (the reference)
  real_repeat     real again on a fresh server (determinism: must be 0)
  render_both     main + UW rendered (path = plain | rigb0179)
  render_main     main rendered, UW real
  render_uw       main real, UW rendered
  roi_paste       the real frames with the rendered gripper ROI (fingers + pads + tags) pasted in: isolates the
                  finger/tag design from the scene
  jpeg_q90_wire   the raw frame through the deploy wire (960x720 resize, JPEG q90, 224): the deploy floor
  jpeg_q90_224    JPEG q90 on the 224 zarr image (the B8 review floor)
Reported per variant: the TCP position gap (mm) and the width gap (mm) at k = 4 / 8 / 15 vs ``real`` (p50 / p95 /
mean), and vs the demo GT (Stage 2a-style, frames whose future lies inside the demo). No gate.

    python -m franka_sim camera-gap --policy run/policy/bp_run001 (or a resolved policy yaml) --zarr Z \\
        --renders DIR/renders.npz [--paths plain,rigb0179] [--variants ...] --seed 0 --device cuda --out gap.json
Predictions are cached by the sha1 of everything served (images, queries, checkpoint, seed, device) in
``--work-dir``/cache, so ``real`` and the JPEG floors are computed once across tooling sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

KS = (4, 8, 15)
PATH_VARIANTS = ("render_both", "render_main", "render_uw", "roi_paste")
FIXED_VARIANTS = ("real", "real_repeat", "jpeg_q90_wire", "jpeg_q90_224")
DEFAULT_WS = {"img_obs_horizon": 2, "obs_down_sample_steps": 3, "ultrawide_down_sample_steps": 1}


def _deploy():
    from franka_sim import model as _m  # noqa: F401  (puts franka_deploy on sys.path; model has no mujoco at import)
    from franka_deploy.tools import predict_demos as PD

    return PD


def jpeg224(img_rgb: np.ndarray, q: int = 90) -> np.ndarray:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img_rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def make_variant_zarr(zarr_path: str, npz, variant: str, path: str | None):
    """A predict_demos.DemoZarr whose main_image / uw_image serve the variant's images (KeyError if one is missing)."""
    PD = _deploy()

    class VariantZarr(PD.DemoZarr):
        def __init__(self):
            super().__init__(zarr_path)
            self.variant, self.rpath, self.npz = variant, path, npz
            self.served: list = []

        def _demo_of_uw(self, g: int) -> int:
            return int(np.searchsorted(self._uw_ends, g, side="right"))

        def _pick(self, real, key_sfx: str, roi_key: str):
            v = self.variant
            if v in ("real", "real_repeat"):
                out = real
            elif v == "jpeg_q90_224":
                out = jpeg224(real)
            elif v == "jpeg_q90_wire":
                out = self.npz[f"jpeg90wire_{key_sfx}"]
            else:
                stream = "main" if key_sfx.endswith("_main") else "ultrawide"
                ren = self.npz[f"{self.rpath}_{key_sfx}"]
                if v == "render_both" or (v == "render_main" and stream == "main") or (v == "render_uw" and stream == "ultrawide"):
                    out = ren
                elif v in ("render_main", "render_uw"):
                    out = real
                elif v == "roi_paste":
                    m = self.npz[roi_key]
                    out = real.copy()
                    out[m] = ren[m]
                else:
                    raise ValueError(f"unknown variant {v!r}")
            self.served.append(key_sfx)
            return np.ascontiguousarray(out)

        def main_image(self, d: int, f: int) -> np.ndarray:
            sfx = f"d{int(d)}_f{int(f)}_main"
            return self._pick(super().main_image(d, f), sfx, f"roi_{sfx}")

        def uw_image(self, g: int) -> np.ndarray:
            sfx = f"d{self._demo_of_uw(int(g))}_g{int(g)}_ultrawide"
            return self._pick(super().uw_image(g), sfx, f"roi_{sfx}")

    return VariantZarr()


def gaps(a: np.ndarray, b: np.ndarray, mask=None) -> dict:
    """{k<K>: {pos_mm: {p50, p95, mean}, width_mm: {...}}} between chunk sets a and b (n, K, 10)."""
    if mask is not None:
        a, b = a[mask], b[mask]
    out = {"n": int(len(a))}
    if not len(a):
        return out
    for k in KS:
        p = np.linalg.norm(a[:, k, :3] - b[:, k, :3], axis=1) * 1e3
        w = np.abs(a[:, k, 9] - b[:, k, 9]) * 1e3
        out[f"k{k}"] = {q: {"p50": round(float(np.percentile(x, 50)), 3), "p95": round(float(np.percentile(x, 95)), 3),
                            "mean": round(float(x.mean()), 3), "max": round(float(x.max()), 3)}
                        for q, x in (("pos_mm", p), ("width_mm", w))}
    return out


def resolve_policy(spec: str) -> dict:
    import yaml

    p = Path(spec)
    if p.is_file():
        return yaml.safe_load(p.read_text())
    from franka_sim.model import load_station

    pid = p.name.removesuffix(".yaml") if p.name else spec
    return dict(load_station({"station": {"host": "pc_laptop"}, "run": {"policy": pid}}).groups["policy"])


def _sha(pol: dict, queries, z, seed: int, device: str) -> str:
    h = hashlib.sha1()
    h.update(json.dumps({"ckpt": str(pol.get("checkpoint")), "kind": pol.get("kind"), "id": pol.get("id"),
                         "seed": seed, "device": device, "queries": queries}, sort_keys=True).encode())
    PD = _deploy()
    for q in queries:                               # every image the variant serves, in call order
        obs = PD.deploy_obs(z, z.demo(int(q["demo"])), float(q["frame"]), float(q.get("start_frame", 0.0)), DEFAULT_WS)
        h.update(np.ascontiguousarray(obs["robot0_main_camera"]).tobytes())
        if "robot0_ultrawide_camera" in obs:
            h.update(np.ascontiguousarray(obs["robot0_ultrawide_camera"]).tobytes())
    return h.hexdigest()


def run(args) -> int:
    PD = _deploy()
    t0 = time.time()
    pol = resolve_policy(args.policy)
    npz = np.load(args.renders, allow_pickle=False)
    meta = json.loads(str(npz["meta"]))
    queries = json.loads(str(npz["queries"]))
    paths = [p for p in (args.paths.split(",") if args.paths else meta["paths"]) if p != "jpeg90wire"]
    variants = args.variants.split(",") if args.variants else list(FIXED_VARIANTS) + [f"{v}@{p}" for p in paths for v in PATH_VARIANTS]
    # keep the queries whose images exist for every variant (dry pass of deploy_obs with the default workspace)
    keys = set(npz.files)
    need_paths = {v.split("@")[1] for v in variants if "@" in v} | ({"jpeg90wire"} if "jpeg_q90_wire" in variants else set())
    z0 = PD.DemoZarr(args.zarr, images=False)
    kept = []
    for q in queries:
        d, fi = int(q["demo"]), int(round(q["frame"]))
        mi = [max(0, fi - 3 * (1 - i)) for i in range(2)]
        uc, us = z0.uw_index(d, fi)
        ix = [max(us, uc - 1), uc]
        sfx = [f"d{d}_f{f}_main" for f in mi] + [f"d{d}_g{g}_ultrawide" for g in ix]
        if all(f"{p}_{s}" in keys for p in need_paths for s in sfx) and all(f"roi_{s}" in keys for s in sfx):
            kept.append(q)
    dropped = len(queries) - len(kept)
    if args.limit:
        kept = kept[: args.limit]
    print(f"[camera-gap] {len(kept)} queries ({dropped} dropped: missing images), variants {variants}", file=sys.stderr, flush=True)
    if not kept:
        raise SystemExit("camera-gap: no query has all images")
    work = Path(args.work_dir or (Path(args.out).resolve().parent / "gap_work"))
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    preds, info = {}, {}
    for v in variants:
        name, _, rp = v.partition("@")
        z = make_variant_zarr(args.zarr, npz, name, rp or None)
        sha = _sha(pol, kept, z, args.seed, args.device)
        cf = cache / f"{sha}.npz"
        tv = time.time()
        if cf.exists() and name != "real_repeat":
            r = dict(np.load(cf))
            src = "cache"
        else:
            r = PD.predict(pol, z, kept, work / f"server_{name}", seed=args.seed, device=args.device,
                           log=lambda m: print(f"[camera-gap] {v}: {m}", file=sys.stderr, flush=True))
            r = {k: r[k] for k in ("rel_chunks", "gt_chunks", "gt_valid", "infer_s", "noise_seed")}
            if name != "real_repeat":
                np.savez(cf, **r)
            src = "computed"
        preds[v] = r
        info[v] = {"source": src, "sha1": sha, "wall_s": round(time.time() - tv, 1),
                   "noise_seed_first_last": [int(r["noise_seed"][0]), int(r["noise_seed"][-1])]}
        print(f"[camera-gap] {v}: {src} in {info[v]['wall_s']} s", file=sys.stderr, flush=True)
    ref = preds["real"]
    valid = ref["gt_valid"].astype(bool)
    res = {"schema": "camera_gap/v1", "policy": {k: pol.get(k) for k in ("id", "kind", "checkpoint")}, "seed": args.seed,
           "renders": str(Path(args.renders).resolve()), "renders_meta": meta, "n_queries": len(kept), "n_dropped": dropped,
           "n_gt_valid": int(valid.sum()), "variants": info,
           "vs_real": {v: gaps(preds[v]["rel_chunks"], ref["rel_chunks"]) for v in variants if v != "real"},
           "vs_gt": {v: gaps(preds[v]["rel_chunks"], ref["gt_chunks"], valid) for v in variants},
           "noise_seeds_equal": {v: bool(np.array_equal(preds[v]["noise_seed"], ref["noise_seed"])) for v in variants},
           "wall_s": round(time.time() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(summary_text(res), flush=True)
    return 0


def summary_text(res: dict) -> str:
    lines = [f"camera-gap {res['policy'].get('id')} n={res['n_queries']} (gt valid {res['n_gt_valid']}) "
             f"sets={res['renders_meta'].get('sets')} robot={res['renders_meta'].get('robot')}",
             f"  {'variant':26s} " + " ".join(f"k{k} pos p50/p95 (w p50)".rjust(24) for k in KS) + "   | vs GT k15 pos p50/p95"]
    for v, g in [("real", None)] + list(res["vs_real"].items()):
        cells = []
        for k in KS:
            if g is None:
                cells.append("ref".rjust(24))
            else:
                e = g[f"k{k}"]
                cells.append(f"{e['pos_mm']['p50']:.1f}/{e['pos_mm']['p95']:.1f} ({e['width_mm']['p50']:.1f})".rjust(24))
        gt = res["vs_gt"][v].get("k15")
        lines.append(f"  {v:26s} " + " ".join(cells) + (f"   | {gt['pos_mm']['p50']:.1f}/{gt['pos_mm']['p95']:.1f}" if gt else ""))
    return "\n".join(lines)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", required=True, help="a resolved run/policy yaml, or a run/policy id (resolved with host pc_laptop)")
    p.add_argument("--zarr", required=True)
    p.add_argument("--renders", required=True, help="camera-check renders.npz")
    p.add_argument("--paths", default=None, help="render paths to test (default: all in the npz: plain,rigb0179)")
    p.add_argument("--variants", default=None, help="comma list (path variants as name@path); default all")
    p.add_argument("--limit", type=int, default=None, help="at most N queries")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--out", required=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
