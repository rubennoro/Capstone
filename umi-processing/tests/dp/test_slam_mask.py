"""The SLAM gripper mask: drawing it (dataproc.slam.mask) and applying it to the EuRoC frames (orbslam3.euroc_image).
The end-to-end runs through the orbslam3 backend are in test_geometry_slam.py."""

import cv2
import numpy as np
import pytest

from umiproc.dp.slam.mask import keep_mask, slam_keep_mask
from umiproc.dp.slam.orbslam3 import euroc_image
from test_geometry_common import make_rig

W, H = 820, 616                                        # rig_v1 ultrawide
STRIP = ((0, 462), (820, 462), (820, 616), (0, 616))   # bottom strip, rows 462..615
# the approved rig_pi_v3 mask (1280x960, Step 1 trace): two halves sharing the vertical edge x = 671, y 462..960
REAL_NATIVE = (1280, 960)
REAL_POLYGONS = (
    ((176, 960), (178, 905), (190, 860), (204, 828), (240, 784), (270, 758), (380, 624), (432, 614), (500, 558),
     (578, 514), (580, 462), (671, 462), (671, 960)),
    ((671, 462), (762, 462), (762, 524), (830, 558), (910, 606), (968, 614), (1064, 733), (1150, 728), (1204, 760),
     (1254, 866), (1280, 936), (1280, 960), (671, 960)),
)


def _rect(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _slice(shape, x0, y0, x1, y1) -> np.ndarray:
    ref = np.zeros(shape, bool)
    ref[y0:y1, x0:x1] = True
    return ref


def _within_1px(masked: np.ndarray, ref: np.ndarray) -> bool:
    """``masked`` covers all of ``ref`` and reaches at most 1 px beyond it."""
    grown = cv2.dilate(ref.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return bool(np.all(masked[ref]) and not np.any(masked & ~grown))


def _masked_rig(polygons, **edits):
    return make_rig(**{"cameras.ultrawide.slam_mask": {"polygons": [[list(p) for p in poly] for poly in polygons]},
                       **edits})


def _rgb(w, h, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------------------------- keep_mask

def test_rectangle_is_the_numpy_slice():
    m = keep_mask((_rect(100, 50, 300, 200),), (W, H), (W, H))
    assert m.shape == (H, W) and m.dtype == np.uint8 and set(np.unique(m)) == {0, 255}
    # fillPoly also draws the outline: the slice plus at most a 1 px boundary, never less than the slice
    assert _within_1px(m == 0, _slice((H, W), 100, 50, 300, 200))


def test_bottom_strip_at_half_size():
    m = keep_mask((STRIP,), (W, H), (410, 308))
    assert m.shape == (308, 410)
    assert np.flatnonzero((m == 0).any(axis=1)).tolist() == list(range(231, 308))
    assert (m[231:] == 0).all() and (m[:231] == 255).all()        # every column of rows 231..307, nothing else


def test_overlapping_polygons_mask_their_union():
    a, b = _rect(100, 100, 400, 400), _rect(250, 250, 550, 550)
    m = keep_mask((a, b), (W, H), (W, H))
    assert _within_1px(m == 0, _slice((H, W), 100, 100, 400, 400) | _slice((H, W), 250, 250, 550, 550))
    # the bug this guards against: one fillPoly call with both polygons leaves their overlap unfilled
    one = np.full((H, W), 255, np.uint8)
    cv2.fillPoly(one, [np.round((np.array(p, float) - 0.5) * 256).astype(np.int32) for p in (a, b)], 0,
                 lineType=cv2.LINE_8, shift=8)
    assert one[300, 300] == 255 and m[300, 300] == 0


@pytest.mark.parametrize("size", [REAL_NATIVE, (640, 480), (410, 308), (500, 300)])
def test_polygons_sharing_an_edge_leave_no_gap(size):
    left, right = REAL_POLYGONS
    merged = left[:-2] + right[1:-1]                    # the same outline as one polygon, without the x = 671 cut
    m = keep_mask(REAL_POLYGONS, REAL_NATIVE, size)
    assert np.array_equal(m, keep_mask((merged,), REAL_NATIVE, size))
    x_edge = int(671 * size[0] / REAL_NATIVE[0])
    y_top = int(np.ceil(462 * size[1] / REAL_NATIVE[1]))
    assert (m[y_top:, x_edge - 2:x_edge + 3] == 0).all()     # along the shared edge, top to bottom


def test_real_mask_fraction():
    """Same polygons, same fraction as the Step 1 trace measured (0.2915)."""
    assert float(np.mean(keep_mask(REAL_POLYGONS, REAL_NATIVE, REAL_NATIVE) == 0)) == pytest.approx(0.2915, abs=5e-4)


def test_per_axis_scaling():
    poly = _rect(200, 100, 600, 300)                                # on 820x616
    m = keep_mask((poly,), (W, H), (410, 616))                      # x halved, y unchanged
    assert m.shape == (616, 410) and _within_1px(m == 0, _slice((616, 410), 100, 100, 300, 300))
    m = keep_mask((poly,), (W, H), (820, 154))                      # y quartered, x unchanged
    assert m.shape == (154, 820) and _within_1px(m == 0, _slice((154, 820), 200, 25, 600, 75))


def test_cached_read_only():
    a = keep_mask((STRIP,), (W, H), (410, 308))
    assert keep_mask((STRIP,), (W, H), (410, 308)) is a
    assert keep_mask([[list(p) for p in STRIP]], [W, H], [410, 308]) is a      # lists: same cache entry
    assert not a.flags.writeable
    with pytest.raises(ValueError):
        a[0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        keep_mask((STRIP,), (W, H), (0, 308))


def test_slam_keep_mask():
    assert slam_keep_mask(make_rig()) is None
    rig = _masked_rig((STRIP,))
    m = slam_keep_mask(rig)
    assert m.shape == (H, W) and m is keep_mask((STRIP,), (W, H), (W, H))
    assert slam_keep_mask(rig, (410, 308)) is keep_mask((STRIP,), (W, H), (410, 308))


# --------------------------------------------------------------------------------------------- euroc_image

@pytest.mark.parametrize("frame_size", [(W, H), (64, 48), (1640, 1232)])
def test_euroc_image_without_mask_is_gray_and_area_resize(frame_size):
    rig, rgb = make_rig(), _rgb(*frame_size)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for size in ((W, H), (410, 308)):
        expect = gray if frame_size == size else cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        out = euroc_image(rgb, rig, size)
        assert out.dtype == np.uint8 and np.array_equal(out, expect)


# frame size, output size, margin (output px) beyond which kept pixels must be untouched: 2 px when the frame is
# downscaled; an upscaled frame's own mask (applied before the resize) has whole source pixels as its boundary,
# 6.4 output px here
@pytest.mark.parametrize("frame_size, size, margin", [((W, H), (W, H), 0), ((W, H), (410, 308), 2),
                                                       ((1640, 1232), (410, 308), 2), ((64, 48), (410, 308), 8)])
def test_euroc_image_with_mask(frame_size, size, margin):
    polygons = (STRIP, ((300, 0), (520, 0), (410, 250)))           # a strip and a slanted triangle
    rig, plain = _masked_rig(polygons), make_rig()
    rgb = _rgb(*frame_size, seed=3)
    out, ref = euroc_image(rgb, rig, size), euroc_image(rgb, plain, size)
    keep = slam_keep_mask(rig, size)
    assert out.shape == (size[1], size[0]) and (out[keep == 0] == 0).all()
    far = cv2.erode(keep, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) == 255
    assert far.mean() > 0.5 and np.array_equal(out[far], ref[far])


def test_euroc_image_no_leak_from_masked_pixels():
    """A bright patch inside the polygon must not brighten kept pixels next to it when the frame is downscaled."""
    wedge = ((0, 300), (820, 250), (820, 300))                      # a shallow edge, like parts of the real mask
    rig = _masked_rig((wedge,))
    native_keep = slam_keep_mask(rig)
    gray = np.where(native_keep == 0, 255, 40).astype(np.uint8)    # bright exactly where masked, dark elsewhere
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for size in ((410, 308), (640, 480)):
        out = euroc_image(rgb, rig, size)
        keep = slam_keep_mask(rig, size) == 255
        assert (out[~keep] == 0).all() and out[keep].max() <= 40
        # without the mask at the frame size first, the resize blends the bright pixels into kept ones
        assert cv2.resize(gray, size, interpolation=cv2.INTER_AREA)[keep].max() > 40
