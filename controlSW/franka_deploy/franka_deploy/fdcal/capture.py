"""Data capture for the calibration tools (hardware day; the pivot drive is tested against the FakeBackend bridge).

handeye  hand-guide the arm (bridge NOT running a controller: state idle/stopped) so the fixed ChArUco board is in
         view; Enter = save the RigCamera main image + bridge get_state O_T_F; `q` = done. Writes DIR/poses.json,
         DIR/board.json, DIR/K.json, then run `fdcal handeye --data DIR`.
pivot    bridge running (`start`), fingertip centre on the marked point: drives +-angle about each TCP axis around the
         current TCP (slow interpolation through the bridge's safety filter), settles, records O_T_F at each
         extreme, returns to the centre, then runs `fdcal pivot` analysis. Writes DIR/poses.json + DIR/pivot.json.
width    servo sweep (raw positions; gripper type from franka_cartesian.yaml): per position, wait, save the RigCamera
         ultrawide image. Writes DIR/sweep.json, then run `fdcal width --data DIR`.
         Franka Hand (--hand-endpoint): per gap, hand_cmd move, wait for move_ok (<= 3 s), save the ultrawide image and the
         hand_state gap_m. Writes DIR/sweep.json (position = commanded gap, gap_m = read-back), then run
         `fdcal width --linear --data DIR ...`. The servo path (--controller-config/--positions) is unchanged.
stream   N frames of the camera server's output (what the policy sees) for `fdcal stream-intrinsics`: Enter = save when
         the ChArUco board is detected, `q` = done. Writes DIR/<stream>_NNN.png and DIR/stream.json.

    python -m franka_deploy fdcal capture handeye --out DIR [--bridge tcp://127.0.0.1:5565] [--camera tcp://localhost:16702]
    python -m franka_deploy fdcal capture pivot --out DIR --F_T_TCP configs/franka_cartesian.yaml [--angle 20]
    python -m franka_deploy fdcal capture width --out DIR --controller-config configs/franka_cartesian.yaml
        --positions 3000:1800:-100 [--settle-s 1.0]
    python -m franka_deploy fdcal capture width --out DIR --hand-endpoint tcp://localhost:5566 --gaps 0.0:0.08:0.005
        [--settle-s 1.0] [--speed-m-s 0.02]
    python -m franka_deploy fdcal capture stream --out DIR --n 30 [--camera tcp://localhost:16702] [--stream main|ultrawide|both]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from franka_deploy.fdcal import pivot as pv


def _grab(camera_ep: str, cam: str):
    import cv2

    from franka_deploy.peripherals.safe_camera_client import SafeCameraClient

    cc = SafeCameraClient("fdcal_capture", camera_ep)
    d, _ = cc.get_latest_images_dict_THWC(1, extra=0)
    return cv2.cvtColor(np.asarray(d[cam][-1]), cv2.COLOR_RGB2BGR)


def _F_T_TCP(src: str) -> np.ndarray:
    if src.endswith((".yaml", ".yml")):
        return np.array(yaml.safe_load(open(src))["F_T_TCP"], float)
    return np.array(json.load(open(src)), float)


def drive(cli, T_from, T_to, v=0.02, w=np.radians(10), rate=200.0):
    """Interpolate flange targets T_from -> T_to at <= v m/s and <= w rad/s through set_target."""
    dp = np.linalg.norm(T_to[:3, 3] - T_from[:3, 3])
    da = Rotation.from_matrix(T_from[:3, :3].T @ T_to[:3, :3]).magnitude()
    n = max(2, int(np.ceil(max(dp / v, da / w) * rate)))
    sl = Slerp([0, 1], Rotation.from_matrix(np.stack([T_from[:3, :3], T_to[:3, :3]])))
    for s in np.linspace(0, 1, n):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = sl(s).as_matrix(), (1 - s) * T_from[:3, 3] + s * T_to[:3, 3]
        cli.request("set_target", {"T": T})
        time.sleep(1.0 / rate)


def hold(cli, T, seconds, rate=200.0):
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        cli.request("set_target", {"T": T})
        time.sleep(1.0 / rate)


def capture_pivot(bridge_ep, F_T_TCP, angle=20.0, settle_s=1.0, out=None, w_deg_s=10.0, v_m_s=0.02) -> dict:
    from franka_deploy.bridge.link import BridgeClient

    cli = BridgeClient(bridge_ep, timeout_s=0.1)
    st = cli.request("get_state", timeout_s=1.0)
    if st.get("bridge_state") not in ("running", "hold"):
        raise SystemExit(f"bridge state {st.get('bridge_state')!r}: send `start` first (arm holding, zone clear)")
    O_T_F0 = np.asarray(st["O_T_EE"], float)
    center = O_T_F0 @ F_T_TCP
    poses = []
    for T_tcp in pv.pivot_commands(center, angle):
        target = T_tcp @ np.linalg.inv(F_T_TCP)
        drive(cli, O_T_F0, target, v=v_m_s, w=np.radians(w_deg_s))
        hold(cli, target, settle_s)
        poses.append(np.asarray(cli.request("get_state")["O_T_EE"], float))
        drive(cli, target, O_T_F0, v=v_m_s, w=np.radians(w_deg_s))
        hold(cli, O_T_F0, 0.3)
    res = pv.analyze(poses, F_T_TCP)
    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "poses.json").write_text(json.dumps([p.tolist() for p in poses]))
        (d / "pivot.json").write_text(json.dumps(res, indent=1))
    cli.close()
    return res


def capture_handeye(a):
    from franka_deploy.bridge.link import BridgeClient
    from franka_deploy.fdcal import handeye as he
    import cv2

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    cli = BridgeClient(a.bridge, timeout_s=0.5)
    poses = []
    board = {"squares_x": a.squares_x, "squares_y": a.squares_y, "square_m": a.square_m, "marker_m": a.marker_m,
             "dict": a.dict}
    (d / "board.json").write_text(json.dumps(board))
    (d / "K.json").write_text(json.dumps(he.K_MAIN_960.tolist()))
    b = he.make_board(board)
    while True:
        k = input(f"[{len(poses)} saved] Enter = capture, q = done: ").strip()
        if k == "q":
            break
        img = _grab(a.camera, "main")
        T = np.asarray(cli.request("get_state")["O_T_EE"], float)
        if he.board_pose_from_image(img, b, he.K_MAIN_960) is None:
            print("  board not detected: move and retry")
            continue
        name = f"{len(poses):03d}.png"
        cv2.imwrite(str(d / name), img)
        poses.append({"image": name, "O_T_F": T.tolist()})
        (d / "poses.json").write_text(json.dumps(poses))
    print(f"{len(poses)} poses in {d}; next: python -m franka_deploy fdcal handeye --data {d}")


def hand_move_wait(cli, gap_m: float, seq: int, speed_m_s: float, timeout_s: float | None = None):
    """hand_cmd move to gap_m, then poll hand_state until this command is done (seq_done >= seq and not busy) or the
    timeout (default |travel| / speed + 1 s, travel from the current gap). Returns (hand_state dict, finished: bool)."""
    if timeout_s is None:
        g0 = cli.request("hand_state").get("gap_m")
        travel = abs(float(gap_m) - float(g0)) if g0 is not None and g0 == g0 else 0.08
        timeout_s = travel / max(float(speed_m_s), 1e-4) + 1.0
    cli.request("hand_cmd", {"kind": "move", "width_m": float(gap_m), "speed_m_s": float(speed_m_s), "force_n": 0.0, "seq": int(seq)})
    t_end = time.monotonic() + timeout_s
    st = {}
    while time.monotonic() < t_end:
        st = cli.request("hand_state")
        if int(st.get("seq_done", -1)) >= seq and not st.get("busy", False):
            return st, True
        time.sleep(0.02)
    return st, False


def capture_width_hand(a):
    """Franka Hand sweep through the I-HAND endpoint (hand_server 5566, or the bridge in sim)."""
    import cv2

    from franka_deploy.bridge.link import BridgeClient

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    lo, hi, step = (float(x) for x in a.gaps.split(":"))
    cli = BridgeClient(a.hand_endpoint, timeout_s=0.5)
    sweep = []
    for i, g in enumerate(np.arange(lo, hi + step / 2, step)):
        st, done = hand_move_wait(cli, float(g), i + 1, a.speed_m_s)
        time.sleep(a.settle_s)
        st = cli.request("hand_state")
        name = f"uw_{i:03d}.png"
        cv2.imwrite(str(d / name), _grab(a.camera, "ultrawide"))
        sweep.append({"position": float(g), "gap_m": float(st.get("gap_m", float("nan"))), "image": name,
                      "result": st.get("last_result"), "move_finished": bool(done)})
        (d / "sweep.json").write_text(json.dumps(sweep))
        print(f"  gap {g * 1e3:5.1f} mm -> read {sweep[-1]['gap_m'] * 1e3:5.1f} mm ({sweep[-1]['result']})")
    cli.close()
    bad = [s for s in sweep if not s["move_finished"] or s["result"] != "move_ok"]
    if bad:
        print(f"WARNING: {len(bad)} of {len(sweep)} moves did not end in move_ok within travel / speed + 1 s")
    print(f"{len(sweep)} gaps in {d}; next: python -m franka_deploy fdcal width --linear --data {d} --deploy-config D --profile P")


def capture_stream(a):
    """N ChArUco views of the deployment stream(s) as the camera server publishes them."""
    import cv2

    from franka_deploy.fdcal import handeye as he

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    board = {"squares_x": a.squares_x, "squares_y": a.squares_y, "square_m": a.square_m, "marker_m": a.marker_m, "dict": a.dict}
    b = he.make_board(board)
    det = cv2.aruco.CharucoDetector(b)
    streams = ("main", "ultrawide") if a.stream == "both" else (a.stream,)
    saved = 0
    while saved < a.n:
        k = input(f"[{saved}/{a.n} saved] move the board / the rig, Enter = capture, q = done: ").strip()
        if k == "q":
            break
        imgs = {s: _grab(a.camera, s) for s in streams}
        good = {}
        for s, img in imgs.items():
            _, ids, _, _ = det.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            good[s] = 0 if ids is None else len(ids)
        if min(good.values()) < 6:
            print(f"  board not seen well enough (corners {good}, need >= 6 in every stream): move and retry")
            continue
        for s, img in imgs.items():
            cv2.imwrite(str(d / f"{s}_{saved:03d}.png"), img)
        saved += 1
    (d / "stream.json").write_text(json.dumps({"n": saved, "streams": list(streams), "board": board,
                                               "size_wh": [int(imgs[streams[0]].shape[1]), int(imgs[streams[0]].shape[0])] if saved else None}))
    print(f"{saved} views in {d}; next: python -m franka_deploy fdcal stream-intrinsics --data {d} --config deploy.yaml --profile P --out {d}/result")


def capture_width(a):
    import cv2

    if getattr(a, "hand_endpoint", None):
        if not a.gaps:
            raise SystemExit("--hand-endpoint needs --gaps start:stop:step (metres, e.g. 0.0:0.08:0.005)")
        return capture_width_hand(a)
    if not a.controller_config or not a.positions:
        raise SystemExit("servo sweep needs --controller-config and --positions; Franka Hand sweep needs --hand-endpoint and --gaps")

    from franka_deploy.gripper.driver import make_gripper

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.controller_config))["gripper"]
    g = make_gripper(cfg)
    lo, hi, step = (float(x) for x in a.positions.split(":"))
    sweep = []
    for i, p in enumerate(np.arange(lo, hi + (step / 2), step)):
        g._write_position(float(p))         # raw servo position (bypasses the width table on purpose)
        time.sleep(a.settle_s)
        name = f"uw_{i:03d}.png"
        cv2.imwrite(str(d / name), _grab(a.camera, "ultrawide"))
        sweep.append({"position": float(p), "read_position": float(g._read_position()), "image": name})
        (d / "sweep.json").write_text(json.dumps(sweep))
    g.close()
    print(f"{len(sweep)} positions in {d}; next: python -m franka_deploy fdcal width --data {d}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_deploy fdcal capture")
    sub = ap.add_subparsers(dest="mode", required=True)
    h = sub.add_parser("handeye")
    p = sub.add_parser("pivot")
    w = sub.add_parser("width")
    st = sub.add_parser("stream")
    for s in (h, p, w, st):
        s.add_argument("--out", required=True)
        s.add_argument("--bridge", default="tcp://127.0.0.1:5565")
        s.add_argument("--camera", default="tcp://localhost:16702")
    for s in (h, st):
        s.add_argument("--squares-x", type=int, default=9)
        s.add_argument("--squares-y", type=int, default=7)
        s.add_argument("--square-m", type=float, default=0.03)
        s.add_argument("--marker-m", type=float, default=0.022)
        s.add_argument("--dict", default="DICT_5X5_100")
    st.add_argument("--n", type=int, default=30, help="views to save")
    st.add_argument("--stream", choices=["main", "ultrawide", "both"], default="both")
    p.add_argument("--F_T_TCP", required=True)
    p.add_argument("--angle", type=float, default=20.0)
    p.add_argument("--settle-s", type=float, default=1.0)
    w.add_argument("--controller-config", default=None, help="servo sweep: franka_cartesian.yaml (gripper type)")
    w.add_argument("--positions", default=None, help="servo sweep: start:stop:step in raw servo units (closed first)")
    w.add_argument("--hand-endpoint", default=None, help="Franka Hand sweep: the I-HAND endpoint (hand_server tcp://localhost:5566)")
    w.add_argument("--gaps", default=None, help="Franka Hand sweep: start:stop:step in metres (finger gap, e.g. 0.0:0.08:0.005)")
    w.add_argument("--speed-m-s", type=float, default=0.02, help="Franka Hand sweep: move speed")
    w.add_argument("--settle-s", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.mode == "pivot":
        res = capture_pivot(a.bridge, _F_T_TCP(a.F_T_TCP), a.angle, a.settle_s, a.out)
        print(json.dumps(res, indent=1))
        return 0 if res["accept"] else 2
    {"handeye": capture_handeye, "width": capture_width, "stream": capture_stream}[a.mode](a)
    return 0
