"""dataproc.imageprep: letterbox bit-identical to iPhUMI, ultrawide fisheye -> virtual pinhole geometry."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from dataproc._vendor.cv_util import get_image_transform_with_border
from dataproc.config import load_rig
from dataproc.imageprep import (ImagePrep, MainPrep, UltrawidePrep, letterbox, letterbox_params, letterbox_point,
                                physical_model, virtual_model)

RIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rigs" / "rig_v1.yaml"
IPHUMI_CV_UTIL = Path("/home/ruben/Capstone/iPhUMI/python_package/iphumi/common/cv_util.py")


@pytest.fixture(scope="module")
def rig():
    return load_rig(RIG_PATH)


@pytest.fixture(scope="module")
def uw(rig):
    return UltrawidePrep(rig)


SIZES = [(1280, 720), (640, 480), (820, 616), (224, 224), (300, 500), (101, 77), (77, 101), (1920, 1080), (225, 223)]


@pytest.mark.parametrize("size", SIZES)
def test_letterbox_bit_identical_to_vendored_iphumi(size):
    w, h = size
    rng = np.random.default_rng(w * 7 + h)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    ref = get_image_transform_with_border(in_res=(w, h), out_res=(224, 224), mode="rgb")(img)
    out = letterbox(img)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert np.array_equal(out, ref)


def test_vendored_cv_util_matches_iphumi_source():
    if not IPHUMI_CV_UTIL.is_file():
        pytest.skip(f"iPhUMI checkout not found at {IPHUMI_CV_UTIL}")
    vend = Path(__file__).resolve().parents[1] / "dataproc" / "_vendor" / "cv_util.py"
    assert vend.read_text().split("\n", 1)[1] == IPHUMI_CV_UTIL.read_text()


def test_letterbox_params_and_point():
    assert letterbox_params((1280, 720)) == (1280, 280, 280, 0, 0)
    assert letterbox_params((640, 480)) == (640, 80, 80, 0, 0)
    assert letterbox_params((77, 101)) == (101, 0, 0, 12, 12)
    # the image centre stays the centre; corners map to the padded corners
    c = letterbox_point([[319.5, 239.5]], (640, 480))
    assert np.allclose(c, [[111.5, 111.5]])
    tl = letterbox_point([[-0.5, -0.5]], (640, 480))      # pixel-edge corner of the content
    assert np.allclose(tl, [[-0.5, 80 * 224 / 640 - 0.5]])


def test_main_prep(rig):
    prep = MainPrep(rig)
    w, h = rig.cameras["main"].resolution
    img = np.random.default_rng(1).integers(0, 256, (h, w, 3), dtype=np.uint8)
    ref = get_image_transform_with_border(in_res=(w, h), out_res=(224, 224), mode="rgb")(img)
    assert np.array_equal(prep.prep_main(img), ref)
    assert np.array_equal(prep(img), ref)
    with pytest.raises(ValueError):
        prep.prep_main(img[:, :-1])
    with pytest.raises(ValueError):
        prep.prep_main(img.astype(np.float32))
    # black bars: top/bottom rows of the 224 image are black for a 16:9 frame
    out = prep.prep_main(np.full((h, w, 3), 200, np.uint8))
    assert out[:40].max() == 0 and out[-40:].max() == 0 and out[112, 112, 0] == 200


def _blob_image(w, h, uv, sigma, peak=250.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    g = peak * np.exp(-((xx - uv[0]) ** 2 + (yy - uv[1]) ** 2) / (2 * sigma ** 2))
    return np.repeat(np.round(g).astype(np.uint8)[:, :, None], 3, axis=2)


def _centroid(img, around, radius):
    a = img[:, :, 0].astype(np.float64)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    m = (np.abs(xx - around[0]) <= radius) & (np.abs(yy - around[1]) <= radius)
    wgt = a * m
    s = wgt.sum()
    assert s > 0, "blob not found"
    return np.array([(wgt * xx).sum() / s, (wgt * yy).sum() / s])


def _rot_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


@pytest.mark.parametrize("R_deg", [0.0, 12.0])
def test_ultrawide_world_point_maps_to_expected_virtual_pixel(rig, R_deg):
    R = _rot_y(R_deg)
    rig2 = dataclasses.replace(rig, virtual_ultrawide=dataclasses.replace(rig.virtual_ultrawide, R_phys_virt=R))
    prep = UltrawidePrep(rig2)
    phys, virt = physical_model(rig2, "ultrawide"), virtual_model(rig2)
    assert prep.to_virtual(np.zeros((616, 820, 3), np.uint8)).shape == (480, 640, 3)
    # world points in the PHYSICAL ultrawide frame (metres)
    pts = np.array([[0.0, 0.0, 0.5], [0.12, -0.05, 0.3], [-0.2, 0.1, 0.25], [0.05, 0.15, 0.2], [-0.1, -0.12, 0.4]])
    for P in pts:
        uv_phys, ok_p = phys.project(P[None])
        d_virt = R.T @ P                                   # d_phys = R_phys_virt @ d_virt
        uv_virt, ok_v = virt.project(d_virt[None])
        assert ok_p[0] and ok_v[0]
        fish = _blob_image(820, 616, uv_phys[0], sigma=2.0)
        v = prep.to_virtual(fish)
        c = _centroid(v, uv_virt[0], radius=15)
        assert np.linalg.norm(c - uv_virt[0]) < 0.35, (P, c, uv_virt[0])
        # and through the letterbox into the 224 training image
        fish = _blob_image(820, 616, uv_phys[0], sigma=5.0)
        out = prep.prep_ultrawide(fish)
        exp = prep.virtual_to_out(uv_virt)[0]
        c = _centroid(out, exp, radius=8)
        assert np.linalg.norm(c - exp) < 0.35, (P, c, exp)


def test_ultrawide_prep_composition_and_validation(rig, uw):
    img = np.random.default_rng(3).integers(0, 256, (616, 820, 3), dtype=np.uint8)
    v = uw.to_virtual(img)
    assert v.shape == (480, 640, 3) and v.dtype == np.uint8
    assert np.array_equal(uw.prep_ultrawide(img), letterbox(v))
    assert np.array_equal(uw(img), uw.prep_ultrawide(img))
    assert np.allclose(uw.virtual_K, rig.virtual_ultrawide.K)
    # non-contiguous input is accepted (copied), wrong size rejected
    assert np.array_equal(uw.to_virtual(np.asfortranarray(img)), v)
    with pytest.raises(ValueError):
        uw.to_virtual(img[:-2])
    # letterbox rows of the 224 ultrawide image are black (640x480 -> 80 px bars -> 28 rows)
    out = uw.prep_ultrawide(np.full((616, 820, 3), 255, np.uint8))
    assert out[:27].max() == 0 and out[-27:].max() == 0


def test_image_prep_bundle(rig):
    p = ImagePrep(rig)
    main = np.zeros((720, 1280, 3), np.uint8)
    assert p.prep_main(main).shape == (224, 224, 3)
    fish = np.zeros((616, 820, 3), np.uint8)
    assert p.prep_ultrawide(fish).shape == (224, 224, 3)
    assert p.to_virtual(fish).shape == (480, 640, 3)
