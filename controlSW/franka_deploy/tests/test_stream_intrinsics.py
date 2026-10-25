"""B10: fdcal stream-intrinsics on synthetic ChArUco renders at 960x720 (board warped by a known K, JPEG q90)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from franka_deploy.fdcal import handeye as he
from franka_deploy.fdcal import stream_intrinsics as si
from franka_deploy.station import PROFILES, load_station

PKG = Path(__file__).resolve().parents[1]
BOARD = {"squares_x": 9, "squares_y": 7, "square_m": 0.03, "marker_m": 0.022, "dict": "DICT_5X5_100"}
PPM = 4000                                   # px per metre of the generated planar board image


def render_views(K, n=20, size=(960, 720), z_range=(0.30, 0.45), seed=0, quality=90):
    """n JPEG-compressed views of the board seen by a pinhole camera K (no distortion), random pose +-30 deg."""
    board = he.make_board(BOARD)
    w_m, h_m = BOARD["squares_x"] * BOARD["square_m"], BOARD["squares_y"] * BOARD["square_m"]
    flat = board.generateImage((int(w_m * PPM), int(h_m * PPM)), marginSize=0, borderBits=1)
    S = np.diag([1.0 / PPM, 1.0 / PPM, 1.0])
    rng = np.random.default_rng(seed)
    K = np.asarray(K, float)
    out = []
    tries = 0
    while len(out) < n and tries < 20 * n:
        tries += 1
        rvec = np.radians(rng.uniform(-30, 30, 3)) * np.array([1.0, 1.0, 0.5])
        R = cv2.Rodrigues(rvec)[0]
        z = rng.uniform(*z_range) * (K[0, 0] / 703.0)           # scale the distance with f so the board fills similar pixels
        # board centre target in the image: spread over the frame
        u, v = rng.uniform(0.3, 0.7) * size[0], rng.uniform(0.3, 0.7) * size[1]
        centre = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        t = centre - R @ np.array([w_m / 2, h_m / 2, 0.0])
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
        img = cv2.warpPerspective(flat, H, size, flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP if False else cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=128)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if len(si.detect_views_from_arrays([img], he.make_board(BOARD))) == 1:
            out.append(img)
    assert len(out) == n, "synthetic board not detectable"
    return out


def _write(tmp, stream, imgs):
    for i, im in enumerate(imgs):
        cv2.imwrite(str(Path(tmp) / f"{stream}_{i:03d}.png"), im)


@pytest.fixture(scope="module")
def station():
    return load_station(PROFILES["sim-iphone"])


def _fit(tmp_path, stream, K_true, station, seed=0):
    _write(tmp_path, stream, render_views(K_true, seed=seed))
    files = si._files(tmp_path, stream)
    K_st = station.get(f"sensor.streams.{stream}.K")
    variants = station.get("sensor.training_intrinsics_variants")
    return si.run_stream(stream, files, he.make_board(BOARD), K_st, variants, "none", tmp_path)


def test_main_K_recovered_and_match(tmp_path, station):
    K = station.get("sensor.streams.main.K")
    r = _fit(tmp_path, "main", K, station)
    f = r["fit"]
    assert abs(f["f"] / K[0][0] - 1) < 0.005, f
    assert abs(f["cx"] - K[0][2]) < 1.0 and abs(f["cy"] - K[1][2]) < 1.0, f
    assert r["verdict"] == "match" and "snippet" not in r
    assert r["coverage"]["cells_hit"] >= 7


def test_uw_K_recovered_and_match(tmp_path, station):
    K = station.get("sensor.streams.ultrawide.K")
    r = _fit(tmp_path, "ultrawide", K, station, seed=1)
    f = r["fit"]
    assert abs(f["f"] / K[0][0] - 1) < 0.005, f
    assert abs(f["cx"] - K[0][2]) < 1.0 and abs(f["cy"] - K[1][2]) < 1.0, f
    assert r["verdict"] == "match"


def test_two_percent_off_is_mismatch_with_snippet(tmp_path, station):
    K = np.array(station.get("sensor.streams.main.K"), float)
    K2 = K.copy()
    K2[0, 0] *= 1.02
    K2[1, 1] *= 1.02
    r = _fit(tmp_path, "main", K2, station, seed=2)
    assert abs(r["fit"]["f"] / K2[0, 0] - 1) < 0.005
    assert r["verdict"] == "mismatch" and r["vs_station"]["df_pct"] == pytest.approx(2.0, abs=0.6)
    sn = r["snippet"]
    assert sn["set"][0].startswith("sensor.streams.main.K=") and "status: measured" in sn["provenance_line"]
    assert json.loads(sn["set"][0].split("=", 1)[1])[0][0] == pytest.approx(K2[0, 0], rel=0.005)


@pytest.mark.parametrize("stream,idx", [("main", 0), ("main", 1), ("ultrawide", 0), ("ultrawide", 1)])
def test_nearest_training_variant(tmp_path, station, stream, idx):
    v = si.variant_Ks(station.get("sensor.training_intrinsics_variants"), stream)[idx]
    K = np.array(v["K"], float)
    if not v["has_c"]:                       # UW variant 2 has no c: use the station's principal point
        K[0, 2], K[1, 2] = station.get("sensor.streams.ultrawide.K")[0][2], station.get("sensor.streams.ultrawide.K")[1][2]
    r = _fit(tmp_path, stream, K, station, seed=3 + idx)
    assert r["nearest_variant"] == idx, r["vs_variants"]


def test_variants_scaled_with_half_pixel_rule(station):
    vs = si.variant_Ks(station.get("sensor.training_intrinsics_variants"), "main")
    # variant 0: f 1406.3428 at 1920x1440 -> 703.1714 at 960x720; c (953.2503 + .5) / 2 - .5 = 476.37515
    assert vs[0]["K"][0][0] == pytest.approx(703.1714, abs=1e-4)
    assert vs[0]["K"][0][2] == pytest.approx(476.37515, abs=1e-4)
    uw = si.variant_Ks(station.get("sensor.training_intrinsics_variants"), "ultrawide")
    assert uw[0]["K"][0][0] == pytest.approx(259.01538 * 1.5, abs=1e-4) and uw[1]["has_c"] is False


def test_too_few_views_is_reported(tmp_path, station):
    r = si.run_stream("main", [], he.make_board(BOARD), station.get("sensor.streams.main.K"), [], "none", tmp_path)
    assert "error" in r


def test_cli_end_to_end(tmp_path, station, capsys):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    for s in ("main", "ultrawide"):
        _write(data, s, render_views(station.get(f"sensor.streams.{s}.K"), n=8, seed=7))
    deploy = PKG / "configs" / "deploy.yaml"
    rc = si.main(["--data", str(data), "--config", str(deploy), "--profile", "sim-iphone", "--out", str(out)])
    assert rc == 0, capsys.readouterr().out
    res = json.loads((out / "stream_intrinsics.json").read_text())
    assert set(res) == {"main", "ultrawide"} and res["main"]["verdict"] == "match"
