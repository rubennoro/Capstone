"""B10: fdcal width --linear (Franka Hand: train = a*gap + b) and the hand sweep helper. The servo fit_map tests stay in
test_fdcal.py, unchanged."""
import json
import os
import threading
import time

import numpy as np
import pytest

from franka_deploy.fdcal import capture, width

OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
TRAIN_CLOSED = 0.0431
RANGE = (0.0447, 0.0844)


def sweep(a, c, noise=0.0003, gaps=np.arange(0.0, 0.0801, 0.0025), seed=0):
    # 2.5 mm steps (33 gaps): the slope's 1-sigma is 0.0003 / (sqrt(33) * 0.0235) = 0.22 %; at 5 mm steps (17 gaps) it is 0.31 %
    rng = np.random.default_rng(seed)
    return gaps, a * gaps + c + rng.normal(0, noise, len(gaps))


def test_recovers_a_and_b_within_1mm_and_half_percent():
    g, raw = sweep(0.97, 0.0431)                   # tags reproduce the training closed value
    f = width.fit_linear(g, raw, TRAIN_CLOSED)
    assert abs(f["a"] - 0.97) / 0.97 < 0.005
    assert abs(f["b"] - 0.0431) < 0.001
    assert f["checks"]["ok"] and f["checks"]["residual_p95_mm"] < 1.0
    assert abs(f["tag_offset_vs_training_mm"]) < 1.0


def test_anchor_closed_uses_train_closed_not_sweep_minimum():
    g, raw = sweep(1.0, 0.056)                     # these fingers' tags are 12.9 mm wider at gap 0
    closed = width.fit_linear(g, raw, TRAIN_CLOSED, "closed")
    rawfit = width.fit_linear(g, raw, TRAIN_CLOSED, "raw")
    assert closed["b"] == pytest.approx(TRAIN_CLOSED)                   # the training value, not min(raw) = 0.056
    assert abs(rawfit["b"] - 0.056) < 0.001 and abs(rawfit["a"] - 1.0) < 0.005
    assert closed["tag_offset_vs_training_mm"] == pytest.approx(12.9, abs=1.0)
    assert closed["a"] == pytest.approx(rawfit["a"])
    # the sweep minimum is never the closed value
    assert abs(closed["b"] - float(raw.min())) > 0.01


def test_non_monotonic_sweep_is_flagged():
    g, raw = sweep(1.0, 0.0431, noise=0.0)
    raw = raw.copy()
    raw[8] -= 0.004                                # one gap reads 4 mm narrower than its neighbours
    f = width.fit_linear(g, raw, TRAIN_CLOSED)
    assert not f["checks"]["monotonic"] and not f["checks"]["ok"]
    assert f["checks"]["max_drop_mm"] > 1.0


def test_noisy_residual_fails_the_check():
    g, raw = sweep(1.0, 0.0431, noise=0.002)
    assert not width.fit_linear(g, raw, TRAIN_CLOSED)["checks"]["residual_ok"]


def test_missing_tags_are_skipped_and_too_few_points_reported():
    g, raw = sweep(1.0, 0.0431)
    raw[::2] = np.nan
    assert width.fit_linear(g, raw, TRAIN_CLOSED)["n_points"] == int(np.isfinite(raw).sum())
    assert "error" in width.fit_linear(g[:3], np.full(3, np.nan), TRAIN_CLOSED)


def test_train_range_must_fit_in_stroke_and_sweep():
    g, raw = sweep(1.0, 0.0431, gaps=np.arange(0.0, 0.0201, 0.005))     # sweep stops at 20 mm: the range needs gap up to 41 mm
    f = width.fit_linear(g, raw, TRAIN_CLOSED)
    assert f["checks"]["train_range_in_stroke"] and not f["checks"]["train_range_in_sweep"] and not f["checks"]["ok"]
    g, raw = sweep(0.4, 0.0431)                    # a = 0.4: the 0.0844 end needs gap 103 mm > stroke
    assert not width.fit_linear(g, raw, TRAIN_CLOSED)["checks"]["train_range_in_stroke"]


def test_hold_widths_map_to_the_gap_printout():
    g, raw = sweep(1.0, 0.0431, noise=0.0)
    f = width.fit_linear(g, raw, TRAIN_CLOSED)
    lo, hi = f["hold_width_gaps_m"]
    assert (lo, hi) == pytest.approx((0.046 - 0.0431, 0.062 - 0.0431), abs=1e-6)     # A5: 3-19 mm


def test_snippet_never_writes_station_yaml():
    g, raw = sweep(1.0, 0.0431)
    sn = width.linear_snippet(width.fit_linear(g, raw, TRAIN_CLOSED), "/x/fit.json", today="2026-10-21")
    assert sn["set"][0].startswith("tooling.width_map.a=") and sn["set"][1].startswith("tooling.width_map.b=")
    assert "status: measured" in sn["provenance_line"] and "2026-10-21" in sn["provenance_line"]


def test_hand_sweep_helper_against_fake_hand_server():
    from franka_deploy.bridge.link import BridgeClient
    from franka_deploy.gripper.hand_server import FakeHand, serve

    ep = f"tcp://127.0.0.1:{5566 + OFFSET + 40}"
    hand = FakeHand(start_gap_m=0.08)
    flag = {"stop": False}
    th = threading.Thread(target=serve, args=(ep, hand, flag), daemon=True)
    hand.start(homing=False)
    th.start()
    time.sleep(0.3)
    cli = BridgeClient(ep, timeout_s=0.5)
    try:
        for i, g in enumerate((0.04, 0.0, 0.02)):
            st, done = capture.hand_move_wait(cli, g, i + 1, speed_m_s=0.2)
            assert done and st["last_result"] == "move_ok"
            assert st["gap_m"] == pytest.approx(g, abs=1e-3)
    finally:
        cli.close()
        flag["stop"] = True
        th.join(7)


def test_hand_sweep_first_move_from_open_is_not_cut_off():
    """After homing the hand is at 0.08: the first move to gap 0 at 0.02 m/s needs 4 s, longer than the old fixed 3 s."""
    from franka_deploy.bridge.link import BridgeClient
    from franka_deploy.gripper.hand_server import FakeHand, serve

    ep = f"tcp://127.0.0.1:{5566 + OFFSET + 41}"
    hand = FakeHand(start_gap_m=0.08)
    flag = {"stop": False}
    th = threading.Thread(target=serve, args=(ep, hand, flag), daemon=True)
    hand.start(homing=False)
    th.start()
    time.sleep(0.3)
    cli = BridgeClient(ep, timeout_s=0.5)
    try:
        t0 = time.monotonic()
        st, done = capture.hand_move_wait(cli, 0.0, 1, speed_m_s=0.04)       # 2 s of travel; timeout = 2 + 1 s
        assert done and st["last_result"] == "move_ok" and st["gap_m"] == pytest.approx(0.0, abs=1e-3)
        assert time.monotonic() - t0 > 1.5
    finally:
        cli.close()
        flag["stop"] = True
        th.join(7)
