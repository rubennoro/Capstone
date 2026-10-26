"""A7 camera-check, quick: 1 firstreal episode x 3 frames against the real zarr + raw data (skips without them).
JSON schema, finite tag errors, masks in [0, 1], SSIM(x, x) = 1, PNGs and renders.npz under tmp_path.

Run: cd controlSW/franka_sim && MUJOCO_GL=egl env -u PYTHONPATH $RE -m pytest -q tests/test_camera_check.py -s
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("av")
if os.environ.get("MUJOCO_GL") != "egl":
    pytest.skip("set MUJOCO_GL=egl", allow_module_level=True)

from franka_sim import camera_check as CC  # noqa: E402

SCRATCH = Path("/tmp/claude-1000/-home-ruben-Capstone/299e5dab-9afb-41ab-aa6c-ac52f8fb5552/scratchpad")
ZARR = Path(os.environ.get("A7_ZARR", SCRATCH / "deploy_work" / "prompt_zarr" / "firstreal-single.zarr.zip"))
RAW = CC.REPO / "data" / "raw" / "2026-10-01"
STAGE1 = CC.CONTROLSW / "docs" / "sim_plan" / "results" / "stage1" / "panda_ext0.1_my_stage1.json.gz"


def test_ssim_identity_and_symmetry():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    y = np.clip(x.astype(int) + rng.integers(-20, 21, x.shape), 0, 255).astype(np.uint8)
    assert np.allclose(CC.ssim_map(x, x), 1.0)
    assert abs(CC.ssim_map(x, y).mean() - CC.ssim_map(y, x).mean()) < 1e-12 and CC.ssim_map(x, y).mean() < 0.99
    m = CC.region_metrics(x, x, {"full": np.ones(x.shape[:2], bool), "none": np.zeros(x.shape[:2], bool)})
    assert m["full"]["mae"] == 0.0 and m["full"]["ssim"] == pytest.approx(1.0) and m["none"]["mae"] is None


def test_valid224_excludes_padding():
    v = CC.valid224(960, 720)
    assert v.shape == (224, 224) and v[:27].sum() == 0 and v[-27:].sum() == 0 and v[112].all()


@pytest.mark.skipif(not (ZARR.is_file() and RAW.is_dir()), reason="firstreal zarr / raw data not present")
def test_camera_check_quick(tmp_path):
    rc = CC.main(["--zarr", str(ZARR), "--raw", str(RAW), "--stage1", str(STAGE1), "--demos", "11",
                  "--frames-per-episode", "3", "--out", str(tmp_path)])
    assert rc == 0
    r = json.loads((tmp_path / "camera_check.json").read_text())
    assert r["schema"] == "camera_check/v1" and r["demos"] == [11] and r["n_frames"] == 3
    for k in ("episodes", "image_similarity", "hand_mask_home", "tags", "coverage", "uw_extrinsic_from_features"):
        assert k in r, k
    for t in ("0", "1"):
        e = r["tags"]["err_px"][t]
        assert e["n"] >= 1 and all(math.isfinite(e[q]) for q in ("p50", "p95", "max"))
    for lbl in ("open", "mid", "closed"):
        for s in ("main", "ultrawide"):
            h = r["hand_mask_home"][lbl][s]
            assert 0.0 <= h["hand_tooling_frac"] <= 1.0 and 0.0 <= h["arm_frac"] <= 1.0 and 0.0 <= h["finger_tag_frac"] <= 1.0
    for s in ("main", "ultrawide"):
        t = r["image_similarity"][s]
        for src in ("plain", "rigb0179", "floor_jpeg_wire", "floor_next_frame"):
            assert t[src]["full"]["mae"]["n"] == 3, (s, src)
            assert -1.0 <= t[src]["full"]["ssim"]["mean"] <= 1.0
    assert r["coverage"]["ultrawide"] == pytest.approx(1.0)
    pngs = list(tmp_path.glob("*.png"))
    assert 1 <= len(pngs) <= CC.PNG_MAX and all(p.stat().st_size < 1_000_000 for p in pngs)
    npz = np.load(tmp_path / "renders.npz")
    q = json.loads(str(npz["queries"]))
    assert len(q) == 3
    f = int(q[0]["frame"])
    assert npz[f"plain_d11_f{f}_main"].shape == (224, 224, 3) and npz[f"plain_d11_f{f - 3}_main"].dtype == np.uint8
    assert npz[f"roi_d11_f{f}_main"].dtype == bool
    print(CC.summary_text(r))
