"""A5 FrankaHandDriver against `hand_server --backend fake` (a separate process): never blocks the caller (slow and
dead server), open -> close -> open reaches the gap targets, kill/restart recovery, the hand_server topics, and
make_gripper on the launcher-merged controller config of the sim-iphone / real-iphone profiles.

    cd controlSW/franka_deploy && flock $LOCK env -u PYTHONPATH PYTHONPATH=. FD_TEST_PORT_OFFSET=1000 \
        $RE -m pytest -q tests/test_hand_driver.py -s
"""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from conftest import PKG_ROOT

pytest.importorskip("zmq")
pytest.importorskip("robotmq")

from franka_deploy.bridge.link import BridgeClient  # noqa: E402
from franka_deploy.gripper.franka_hand import FrankaHandDriver  # noqa: E402
from franka_deploy.gripper.hand_logic import HandPolicyConfig, WidthMap  # noqa: E402

OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
HAND_EP = f"tcp://localhost:{5566 + OFFSET}"          # the launcher passes localhost; hand_server binds 127.0.0.1


def env():
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update({"PYTHONPATH": str(PKG_ROOT), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return e


class Server:
    def __init__(self, tmp, *args):
        self.log = open(tmp / f"hand_{time.monotonic_ns()}.log", "w")
        self.p = subprocess.Popen([sys.executable, "-u", "-m", "franka_deploy.gripper.hand_server", "--endpoint", HAND_EP,
                                   "--backend", "fake", "--no-homing", *args], stdout=self.log,
                                  stderr=subprocess.STDOUT, env=env(), start_new_session=True)
        c = BridgeClient(HAND_EP.replace("localhost", "127.0.0.1"), timeout_s=0.2)
        t0 = time.monotonic()
        while True:
            try:
                c.request("hand_state")
                break
            except Exception:
                assert self.p.poll() is None and time.monotonic() - t0 < 10, "hand_server not ready"
                time.sleep(0.05)
        c.close()

    def stop(self, sig=signal.SIGINT):
        if self.p.poll() is None:
            self.p.send_signal(sig)
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait(5)
        self.log.close()


def driver(**kw):
    return FrankaHandDriver(HAND_EP, WidthMap(), HandPolicyConfig(**kw), rate_hz=50.0, rpc_timeout_s=0.1, stale_s=0.2)


def max_call_ms(drv, n=2000, w=0.084):
    worst = 0.0
    for i in range(n):
        t0 = time.perf_counter()
        drv.set_width(w)
        drv.get_width()
        worst = max(worst, time.perf_counter() - t0)
        if i % 10 == 0:
            time.sleep(0.0005)
    return worst * 1e3


def wait_for(pred, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_server_topics(tmp_path):
    s = Server(tmp_path, "--object-width-m", "0.03")
    try:
        c = BridgeClient(HAND_EP.replace("localhost", "127.0.0.1"), timeout_s=0.5)
        st = c.request("hand_state")
        assert set(st) == {"gap_m", "gap_vel_m_s", "busy", "mode", "grasped", "last_result", "seq_done", "t"}
        assert abs(st["gap_m"] - 0.08) < 1e-6 and not st["busy"]
        t0 = time.perf_counter()
        assert c.request("hand_cmd", {"kind": "grasp", "width_m": 0.032, "speed_m_s": 0.1, "force_n": 30,
                                      "eps_inner_m": 0.005, "eps_outer_m": 0.005, "seq": 3}) == {"accepted": True, "seq": 3}
        assert time.perf_counter() - t0 < 0.05                 # the REP loop never waits on the hand
        assert wait_for(lambda: c.request("hand_state")["seq_done"] == 3, 2.0)
        st = c.request("hand_state")
        assert st["mode"] == "holding" and st["grasped"] and st["last_result"] == "grasp_ok" and abs(st["gap_m"] - 0.03) < 1e-6
        c.request("hand_cmd", {"kind": "move", "width_m": 0.01, "seq": 4})            # blocked by the object
        assert wait_for(lambda: c.request("hand_state")["seq_done"] == 4, 2.0)
        assert c.request("hand_state")["last_result"] == "move_failed"
        c.request("hand_cmd", {"kind": "move", "width_m": 0.07, "seq": 5})
        time.sleep(0.1)
        c.request("hand_cmd", {"kind": "stop", "seq": 6})
        assert wait_for(lambda: c.request("hand_state")["seq_done"] == 6, 1.0)
        st = c.request("hand_state")
        assert st["mode"] == "stopped" and st["last_result"] == "stopped" and 0.03 < st["gap_m"] < 0.07
        for bad in ({"kind": "squeeze"}, {"kind": "move", "width_m": float("nan")}):
            with pytest.raises(Exception, match="hand_cmd"):
                c.request("hand_cmd", bad)
        with pytest.raises(Exception, match="unknown request"):
            c.request("get_state")
        c.close()
    finally:
        s.stop()


def test_driver_never_blocks_and_tracks_schedule(tmp_path):
    s = Server(tmp_path, "--cmd-latency-s", "0.5", "--object-width-m", "0.03")
    d = driver()
    try:
        assert wait_for(lambda: d.status()["mode"] is not None, 1.0)
        worst = max_call_ms(d)
        wm, out = WidthMap(), {"max_call_ms_slow_server": worst}
        assert worst <= 1.0, worst
        # open -> close -> open (training widths): move, grasp (stalls on the 30 mm object: grasped), release
        sched = [(0.084, wm.gap(0.084), ("move_ok",)), (0.05, 0.03, ("grasp_ok",)), (0.084, wm.gap(0.084), ("move_ok",))]
        for w, gap_target, results in sched:
            t0 = time.perf_counter()
            d.set_width(w)
            assert time.perf_counter() - t0 < 1e-3
            ok = wait_for(lambda: abs(d.get_width() - wm.train(gap_target)) < 1e-3
                          and d.status()["last_result"] in results, 3.0)
            assert ok, (w, d.status(), d.get_width())
        st = d.status()
        assert st["policy_state"] == "OPEN" and st["cmds_sent"] >= 3 and st["rpc_timeouts"] == 0
        assert abs(d.target_position() - 0.084) < 1e-9 and d.commanded_width() == 0.084
        # kill the server: calls stay <= 1 ms, stale reads and timeouts grow
        s.stop(signal.SIGKILL)
        time.sleep(0.5)
        c0 = d.status()
        worst_dead = max_call_ms(d, w=0.05)
        c1 = d.status()
        out["max_call_ms_dead_server"] = worst_dead
        assert worst_dead <= 1.0, worst_dead
        assert c1["stale_reads"] > c0["stale_reads"] and c1["rpc_timeouts"] > 0
        assert abs(d.get_width() - wm.train(wm.gap(0.084))) < 1e-3       # the last value
        # restart: recovers (fresh state; the pending close goes out)
        s = Server(tmp_path, "--start-gap-m", "0.076")
        assert wait_for(lambda: d.status()["state_age_s"] is not None and d.status()["state_age_s"] < 0.1, 2.0)
        assert wait_for(lambda: d.status()["last_result"] == "grasp_failed", 3.0), d.status()   # no object now
        assert d.status()["policy_state"] == "GRASPING"
        out.update({k: d.status()[k] for k in ("cmds_sent", "rpc_timeouts", "stale_reads")})
        print("\nA5 driver " + json.dumps(out))
    finally:
        d.close()
        s.stop()


def test_driver_without_server_does_not_raise():
    d = FrankaHandDriver(HAND_EP, WidthMap(), HandPolicyConfig(), first_state_wait_s=0.2)
    try:
        assert max_call_ms(d, 500) <= 1.0
        time.sleep(0.3)
        st = d.status()
        assert st["rpc_timeouts"] >= 1 and st["cmds_sent"] == 0 and st["mode"] is None
        assert d.get_width() == WidthMap().train(0.08)
    finally:
        t0 = time.monotonic()
        d.close()
        assert time.monotonic() - t0 < 1.5


@pytest.mark.parametrize("profile,expect_bridge", [("sim-iphone", True), ("real-iphone", False)])
def test_make_gripper_from_launcher_cfg(profile, expect_bridge):
    from franka_deploy.gripper.driver import make_gripper
    from franka_deploy.launch.launcher import build_controller_cfg, load_config, norm_endpoint
    from franka_deploy.station import station_from_deploy_cfg

    cfg = load_config(PKG_ROOT / "configs" / "deploy.yaml", profile)
    ctrl, _ = build_controller_cfg(cfg, station_from_deploy_cfg(cfg))
    g = dict(ctrl["gripper"])
    assert g["type"] == "franka_hand" and "calibration" in g      # the servo leftovers are still in the merged dict
    g["first_state_wait_s"] = 0.0                                  # nothing listens here
    drv = make_gripper(g)
    try:
        assert isinstance(drv, FrankaHandDriver)
        if expect_bridge:
            assert norm_endpoint(drv.endpoint) == norm_endpoint(cfg["bridge"]["endpoint"])
        else:
            assert drv.endpoint.endswith(":5566")
        assert drv.wm == WidthMap.from_config(g["width_map"]) and drv.delta == 0.0
        assert drv.policy.cfg.eps_inner_m == g["grasp_epsilon_inner_m"]
        assert drv.cal.to_width(0.05) == 0.05                     # identity cal (no servo table)
    finally:
        drv.close()
    del g["calibration"]
    drv = make_gripper(g)                                          # works without a calibration key
    drv.close()
