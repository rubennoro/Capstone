"""A stand-in for the sim PC's state export, for sim Pi smoke runs only (DESIGN.md section 4 format).

    python -m sim_deploy.sim_pi.state_stub --model scene.xml [--bind tcp://127.0.0.1:5660] [--amplitude-deg 8]

Publishes ``sim.state/v1`` every 5 ms of CLOCK_MONOTONIC: the model's home qpos (model.json ``home.qpos``, else the
``home`` key) with joints 1 and 5 swinging slowly (``amplitude_deg``, 0.1 Hz) and the fingers opening and closing,
so the rendered views move. ``model_sha256`` is model.json's. Not used in a real sim run (the sim PC publishes).
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import zmq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.sim_pi.state_stub")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--bind", default="tcp://127.0.0.1:5660")
    ap.add_argument("--amplitude-deg", type=float, default=8.0)
    ap.add_argument("--period-ms", type=float, default=5.0)
    ap.add_argument("--seconds", type=float, default=0.0)
    a = ap.parse_args(argv)
    import mujoco
    doc = json.loads(a.model.with_name("model.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(a.model))
    home = doc.get("home", {}).get("qpos")
    if home is None:
        home = m.key_qpos[m.key("home").id].tolist()
    adr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in ("joint1", "joint5", "finger_joint1", "finger_joint2")
           if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0}
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(a.bind)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    P = int(a.period_ms * 1e6)
    t0 = time.monotonic_ns()
    nxt, step = t0, 0
    amp = math.radians(a.amplitude_deg)
    print(f"state_stub: PUB {a.bind}, nq {m.nq}, sha {doc['model_sha256'][:16]}..., joints {sorted(adr)}", flush=True)
    while not stop:
        if a.seconds and nxt - t0 > a.seconds * 1e9:
            break
        now = time.monotonic_ns()
        if now < nxt:
            time.sleep((nxt - now) / 1e9)
        t = (nxt - t0) / 1e9
        q = list(home)
        if "joint1" in adr:
            q[adr["joint1"]] += amp * math.sin(2 * math.pi * 0.1 * t)
        if "joint5" in adr:
            q[adr["joint5"]] += 0.5 * amp * math.sin(2 * math.pi * 0.07 * t)
        for f in ("finger_joint1", "finger_joint2"):
            if f in adr:
                q[adr[f]] = 0.01 + 0.009 * math.sin(2 * math.pi * 0.2 * t)   # gap 2..38 mm (W0: 0..0.040)
        pub.send(json.dumps({"schema": "sim.state/v1", "t_ns": nxt, "step": step,
                             "model_sha256": doc["model_sha256"], "qpos": q}).encode())
        step += int(round(a.period_ms))
        nxt += P
    pub.close(linger=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
