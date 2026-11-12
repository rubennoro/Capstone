"""Live image prep at the stand-in rig's sizes: bit-identical to iPhUMI's cv_util (STRUCTURE.md §13.2)."""

from __future__ import annotations

import numpy as np
import pytest

from process_frames._dataproc._vendor.cv_util import get_image_transform_with_border
from process_frames._dataproc.imageprep import ImagePrep, letterbox, physical_model, virtual_model


def _ref(img):
    h, w = img.shape[:2]
    return get_image_transform_with_border(in_res=(w, h), out_res=(224, 224), mode="rgb")(img)


@pytest.fixture(scope="module")
def prep(standin_rig):
    return ImagePrep(standin_rig)


def test_standin_imageprep_builds(prep):
    assert prep.main.in_res == (1600, 900)
    assert prep.ultrawide.in_res == (1280, 960) and prep.ultrawide.virtual_res == (640, 480)


def test_main_224_identical_to_cv_util_at_1600x900(prep):
    img = np.random.default_rng(5).integers(0, 256, (900, 1600, 3), dtype=np.uint8)
    out = prep.prep_main(img)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert np.array_equal(out, _ref(img))


def test_ultrawide_224_identical_to_cv_util_of_the_remap(prep):
    img = np.random.default_rng(6).integers(0, 256, (960, 1280, 3), dtype=np.uint8)
    virt = prep.to_virtual(img)
    assert virt.shape == (480, 640, 3)
    assert np.array_equal(prep.prep_ultrawide(img), _ref(virt))


@pytest.mark.parametrize("size", [(1640, 1232), (1280, 960), (160, 90), (128, 96), (333, 101)])
def test_letterbox_identical_at_other_sizes(size):
    w, h = size
    img = np.random.default_rng(w + h).integers(0, 256, (h, w, 3), dtype=np.uint8)
    assert np.array_equal(letterbox(img), _ref(img))


def _blob(w, h, uv, sigma):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    g = 250.0 * np.exp(-((xx - uv[0]) ** 2 + (yy - uv[1]) ** 2) / (2 * sigma ** 2))
    return np.repeat(np.round(g).astype(np.uint8)[:, :, None], 3, axis=2)


def _centroid(img, around, radius):
    a = img[:, :, 0].astype(np.float64)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    m = (np.abs(xx - around[0]) <= radius) & (np.abs(yy - around[1]) <= radius)
    w = a * m
    return np.array([(w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()])


@pytest.mark.parametrize("P", [[0.0, 0.0, 0.5], [0.12, -0.05, 0.3], [-0.2, 0.1, 0.25]])
def test_remap_puts_a_known_point_at_the_expected_pixel_for_the_standin_lens(standin_rig, prep, P):
    phys, virt = physical_model(standin_rig, "ultrawide"), virtual_model(standin_rig)
    P = np.array(P)
    uv_phys, ok_p = phys.project(P[None])
    uv_virt, ok_v = virt.project(P[None])                 # R_phys_virt = identity in the stand-in
    assert ok_p[0] and ok_v[0]
    v = prep.to_virtual(_blob(1280, 960, uv_phys[0], sigma=3.0))
    assert np.linalg.norm(_centroid(v, uv_virt[0], 15) - uv_virt[0]) < 0.35
    out = prep.prep_ultrawide(_blob(1280, 960, uv_phys[0], sigma=8.0))
    exp = prep.ultrawide.virtual_to_out(uv_virt)[0]
    assert np.linalg.norm(_centroid(out, exp, 8) - exp) < 0.35
