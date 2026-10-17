"""Mock spacemouse server: what robot_utils' SpacemouseClient expects (topic `spacemouse_state`, 8 float64 =
6 motion axes + 2 buttons), all zeros, at 100 Hz.

    python -m franka_deploy.mocks.spacemouse_server [--endpoint tcp://localhost:15557]
"""

import argparse
import signal
import time

import numpy as np
import robotmq


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="tcp://localhost:15557")
    ap.add_argument("--rate", type=float, default=100.0)
    a = ap.parse_args(argv)
    server = robotmq.RMQServer("spacemouse_server", a.endpoint)
    server.add_topic("spacemouse_state", 10.0)
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    state = np.zeros(8, np.float64)
    period = 1.0 / a.rate
    t = time.monotonic()
    print(f"[spacemouse mock] serving zeros on {a.endpoint}", flush=True)
    while not stop["v"]:
        server.put_data("spacemouse_state", state.tobytes())
        t += period
        time.sleep(max(0.0, t - time.monotonic()))


if __name__ == "__main__":
    main()
