"""T4 (D10b): exported images match the original iPhone frames where the physical lenses cover them."""

from __future__ import annotations

import av
import cv2
import numpy as np
import pytest

from fixtures_sim import *  # noqa: F401,F403
from sensor_rig.sim.compare import mask_224, masked_mae, masked_psnr, pad_resize, warp_to_K

pytestmark = pytest.mark.data
EVERY = 8  # compare every 8th frame (all frames are decoded either way)


def _frames(path):
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            yield f.to_ndarray(format="bgr24")


def _compare(exp_iter, orig_iter, pairs, K_src, K_virt, valid_export):
    """pairs: list of (export frame index, original frame index), both ascending."""
    orig_cache = {}
    want_orig = {o for _, o in pairs}
    pairs_by_exp = dict(pairs)
    ones = None
    res = {"full": [], "224": [], "psnr": []}
    oi = -1
    orig_iter = iter(orig_iter)
    for ei, img_e in enumerate(exp_iter):
        if ei not in pairs_by_exp:
            continue
        o = pairs_by_exp[ei]
        while oi < o:
            oi += 1
            frame = next(orig_iter)
            if oi in want_orig:
                orig_cache[oi] = frame
        img_o = warp_to_K(orig_cache.pop(o), K_src, K_virt)
        if ones is None:
            ones = warp_to_K(np.full(img_o.shape[:2], 255, np.uint8), K_src, K_virt) == 255
            mask = cv2.erode((valid_export & ones).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            m224 = mask_224(mask)
        res["full"].append(masked_mae(img_e, img_o, mask))
        a, b = pad_resize(img_e), pad_resize(img_o)
        res["224"].append(masked_mae(a, b, m224))
        res["psnr"].append(masked_psnr(a, b, m224))
    return {k: np.asarray(v) for k, v in res.items()}


@pytest.mark.parametrize("variant", ["demo60", "demo1278"])
def test_T4_main_images(sim_export, variant):
    se = sim_export(variant)
    i0, n = se.kept_range()
    src_idx = se.truth["main_src_index"][se.grid().src_idx[i0:i0 + n]]
    pairs = [(k, int(src_idx[k])) for k in range(0, n, EVERY)]
    rig = se.rig
    r = _compare(_frames(se.out_dir / f"{se.result.name}_rgb.mp4"), se.src.iter_main_frames(), pairs,
                 se.src.K_main, rig.virtual_camera("main", "export").K, rig.remap("main", "export").valid)
    assert len(r["full"]) == len(pairs)
    print(variant, "main full MAE mean/max", r["full"].mean(), r["full"].max(), "224 MAE", r["224"].max(),
          "PSNR min", r["psnr"].min())
    assert r["full"].max() < 8
    assert r["224"].max() < 3
    assert r["psnr"].min() > 32


@pytest.mark.parametrize("variant", ["demo60", "demo1278", "calib"])
def test_T4b_ultrawide_images(sim_export, variant):
    se = sim_export(variant)
    uw = se.json["ultrawideRGBTimes"]
    n_uw = sum(1 for s in uw if s)
    # native UW mode with no trimming: exported UW frame k == iPhone UW frame k
    assert se.truth["uw_src_frame"].tolist()[:n_uw] == list(range(n_uw))
    pairs = [(k, k) for k in range(0, n_uw, max(1, EVERY // 4))]
    rig = se.rig
    r = _compare(_frames(se.out_dir / f"{se.result.name}_ultrawidergb.mp4"), se.src.iter_uw_frames(), pairs,
                 se.src.K_uw, rig.virtual_camera("ultrawide", "export").K, rig.remap("ultrawide", "export").valid)
    print(variant, "uw full MAE mean/max", r["full"].mean(), r["full"].max(), "224 MAE", r["224"].max())
    assert r["full"].max() < 10
    assert r["224"].max() < 4


def test_T4c_coverage(sim_export):
    se = sim_export("demo60")
    assert se.result.coverage["main"] >= 0.99
    assert se.result.coverage["ultrawide"] >= 0.99
