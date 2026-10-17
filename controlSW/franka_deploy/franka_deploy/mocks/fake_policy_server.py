"""Fake policy server with the detached_inference topics, for tests (run it as its own process: robotmq calls hold
the GIL, so a client and a server in one Python process deadlock): counts every request, enforces the real
server's prompt rules (inference before a prompt -> ERROR; a second prompt without a reset -> ERROR), and replies
with an identity relative chunk (the arm holds still). Optional per-request delay and forced errors.

    python -m franka_deploy.mocks.fake_policy_server [--endpoint tcp://0.0.0.0:18765] [--delay 0.0]
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import robotmq

CONFIG = {"workspace": {"proprio_length": 1, "img_obs_horizon": 2, "obs_down_sample_steps": 3,
                        "ultrawide_down_sample_steps": 1},
          "use_prompting": True, "policy_name": "fake", "run_name": "fake", "epoch": 0, "date_str": "x", "time_str": "y"}


class FakePolicyServer:
    def __init__(self, endpoint: str = "tcp://0.0.0.0:18765", delay_s: float = 0.0, error_after: int | None = None):
        self.endpoint = endpoint
        self.server = None  # created in the serving thread (zmq sockets are not thread-safe)
        self.ready = threading.Event()
        self.delay_s = delay_s
        self.counts = {t: 0 for t in ("policy_config", "policy_reset", "policy_prompt", "policy_inference", "errors")}
        self.prompts: list[dict] = []
        self.prompted = False
        self.force_error = False
        self.error_after = error_after
        self._stop = threading.Event()

    def reply(self, topic, data):
        if topic == "policy_config":
            return CONFIG
        if topic == "policy_reset":
            self.prompted = False
            return "OK"
        if topic == "policy_prompt":
            req = robotmq.deserialize(data)
            if self.prompted:
                return "ERROR: prompt should not be provided ... Call `reset`"
            self.prompted = True
            self.prompts.append({"t": time.monotonic(), **req})
            return "OK"
        if topic == "policy_inference":
            if self.force_error or (self.error_after is not None and self.counts["policy_inference"] > self.error_after):
                return "ERROR: forced"
            if not self.prompted:
                return "ERROR: AssertionError is_prompted"
            time.sleep(self.delay_s)
            rot = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (16, 1))
            return {"action0_eef_pos": np.zeros((16, 3), np.float32), "action0_eef_rot_axis_angle": rot,
                    "action0_gripper_width": np.full((16, 1), 0.07, np.float32)}
        return ""

    def run(self):
        self.server = robotmq.RMQServer("policy_server", self.endpoint, robotmq.RMQLogLevel.WARNING)
        for t in ("policy_config", "policy_reset", "policy_prompt", "policy_inference", "export_recorded_data"):
            self.server.add_topic(t, message_remaining_time_s=60)
        self.ready.set()
        while not self._stop.is_set():
            data, topic = self.server.wait_for_request(timeout_s=0.05)
            if not topic:
                continue
            if topic in self.counts:
                self.counts[topic] += 1
            r = self.reply(topic, data)
            if isinstance(r, str) and r.startswith("ERROR"):
                self.counts["errors"] += 1
            self.server.reply_request(topic=topic, data=robotmq.serialize(r))

    def start(self):
        th = threading.Thread(target=self.run, daemon=True)
        th.start()
        self.ready.wait(5)
        return th

    def stop(self):
        self._stop.set()


def main(argv=None):
    import json
    import os
    import signal

    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="tcp://0.0.0.0:18765")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--error-after", type=int, default=None, help="reply ERROR to every inference after N")
    ap.add_argument("--stats", default=None, help="JSON file rewritten after every request (counts, prompts)")
    a = ap.parse_args(argv)
    s = FakePolicyServer(a.endpoint, a.delay, a.error_after)
    signal.signal(signal.SIGTERM, lambda *_: s.stop())
    if a.stats:
        orig = s.reply

        def reply(topic, data):
            r = orig(topic, data)
            err = isinstance(r, str) and r.startswith("ERROR")
            with open(a.stats + ".tmp", "w") as f:
                json.dump({"counts": s.counts, "prompts": s.prompts, "errors": s.counts["errors"] + int(err)}, f)
            os.replace(a.stats + ".tmp", a.stats)
            return r

        s.reply = reply
    try:
        s.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
