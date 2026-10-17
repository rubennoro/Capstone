"""Stop watcher: a separate process that HOLDS the arm within ~10 ms of a stop request.

Why a process: robotmq's blocking calls keep the Python GIL, so while the task process waits up to an inference
round trip (~0.3 s, up to the inference timeout) no other thread in it can run. This process polls (every
`poll_s`) the stop file and the tail of the task's command file; on a stop file or a `s`/`y`/`n` line it puts
{"hold": True} on the controller's "hold" topic immediately. For a stop file it then appends `s` to the command
file and removes the file, so the task ends the episode and releases the hold once its current call returns.

    python -m franka_deploy.tasks.stop_watcher --controller-endpoint E --command-file F --stop-file S
"""

from __future__ import annotations

import argparse
import os
import signal
import time

from robotmq import serialize

from franka_deploy.rmq_rpc import NoResendClient

STOP_KEYS = ("s", "y", "n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller-endpoint", required=True)
    ap.add_argument("--command-file", required=True)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--poll-s", type=float, default=0.005)
    a = ap.parse_args(argv)
    client = NoResendClient("stop_watcher", a.controller_endpoint, default_timeout_s=0.2)
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    pos = os.path.getsize(a.command_file) if os.path.exists(a.command_file) else 0
    parent = os.getppid()
    print(f"[stop_watcher] watching {a.command_file} / {a.stop_file}", flush=True)

    def hold(reason, t_req):
        try:
            client.put_data(topic="hold", data=serialize({"hold": True, "reason": reason, "t": t_req}))
            print(f"[stop_watcher] HOLD sent ({reason}) {1000 * (time.monotonic() - t_req):.1f} ms after detection", flush=True)
        except Exception as e:
            print(f"[stop_watcher] HOLD FAILED ({reason}): {e!r}", flush=True)

    while not stop["v"]:
        if os.getppid() != parent:  # the task died (even SIGKILL): do not linger
            break
        t = time.monotonic()
        if a.stop_file and os.path.exists(a.stop_file):
            hold("stop file", t)
            try:
                os.remove(a.stop_file)
            except OSError:
                pass
            with open(a.command_file, "a") as f:
                f.write("s\n")
            pos = os.path.getsize(a.command_file)  # already held for this one
        if os.path.exists(a.command_file) and os.path.getsize(a.command_file) > pos:
            with open(a.command_file) as f:
                f.seek(pos)
                data = f.read()
                pos = f.tell()
            for ln in data.splitlines():
                if ln.strip() in STOP_KEYS:
                    hold(f"key {ln.strip()!r}", t)
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
