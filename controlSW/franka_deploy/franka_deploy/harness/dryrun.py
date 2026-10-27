"""Laptop closed-loop harness: real inference server + RigCamera mock replay + MockFrankaController + mock spacemouse
+ FrankaPromptTask, each its own process (as on the Jetson), plus a timed script of keys and failure injections.

    python -m franka_deploy.harness.dryrun --work-dir W --ckpt C --prompt-zarr Z --episode E \
        --py-bp <env_bp>/bin/python --py-re <env_realenv>/bin/python [--mode integrate|replay] [--publish-mode ...]
        [--pause|--continuous] --script "3:c,300:s,305:q" [--task-args "..."] [--controller-args "..."]

Script items `t:action` (t = seconds after the task started): `c`, `C`, `r`, `R`, `s`, `y`, `n`, `q` (keys via the
command file), `stopfile` (stop via the stop file), `kill:<proc>` (SIGKILL: inference|camera|controller|spacemouse),
`stop:<proc>` (SIGTERM), `end` (finish). Writes <work>/harness.json (process exit codes, script timing).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from franka_deploy.paths import check_writable_dir

PKG = Path(__file__).resolve().parents[2]


def _env(extra_pythonpath: str = ""):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": str(PKG) + (os.pathsep + extra_pythonpath if extra_pythonpath else "")})
    return env


def _rmq_user() -> str:
    """The user field of robotmq segment names (as robotmq.utils.clear_shared_memory / the native get_user_name())."""
    try:
        return os.getlogin()
    except OSError:
        import pwd

        try:
            return pwd.getpwuid(os.getuid()).pw_name
        except KeyError:
            return str(os.getuid())


def cleanup_shm(pid: int, user: str | None = None) -> list[str]:
    """robotmq shared-memory segments (/dev/shm/rmq_<user>_<pid>_*) survive a killed camera server; remove them.
    Prefix-anchored: a segment of another process whose name carries `_<pid>_` later on is left alone (W10B)."""
    import glob

    removed = []
    prefix = f"rmq_{user if user is not None else _rmq_user()}_{int(pid)}_"
    for f in glob.glob(f"/dev/shm/{glob.escape(prefix)}*"):
        try:
            os.remove(f)
            removed.append(os.path.basename(f))
        except OSError:
            pass
    return removed


class Proc:
    def __init__(self, name, cmd, work, env):
        self.name, self.cmd = name, cmd
        self.log = open(Path(work) / f"{name}.log", "w")
        self.log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        self.log.flush()
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                  cwd=str(work), start_new_session=True)
        self.t_start = time.monotonic()

    def alive(self):
        return self.p.poll() is None

    def signal(self, sig):
        if self.alive():
            os.killpg(self.p.pid, sig)

    def stop(self, timeout=10):
        if self.alive():
            self.signal(signal.SIGTERM)
            try:
                self.p.wait(timeout)
            except subprocess.TimeoutExpired:
                self.signal(signal.SIGKILL)
                self.p.wait(5)
        self.log.close()
        self.shm_removed = cleanup_shm(self.p.pid)
        return self.p.returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt-zarr", required=True)
    ap.add_argument("--episode", required=True, help="raw (sim) rig episode dir for the camera replay")
    ap.add_argument("--py-bp", required=True)
    ap.add_argument("--py-re", required=True)
    ap.add_argument("--mode", default="integrate", choices=["integrate", "replay"])
    ap.add_argument("--replay-npz", default=None)
    ap.add_argument("--publish-mode", default="per_main_frame", choices=["per_main_frame", "fixed_rate", "auto"])
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--script", required=True)
    ap.add_argument("--task-args", default="")
    ap.add_argument("--controller-args", default="")
    ap.add_argument("--camera-overrides", default="")
    ap.add_argument("--controller", default="mock", choices=["mock", "franka"],
                    help="franka = franka_bridge (FakeBackend) + FrankaCartesian instead of MockFrankaController")
    ap.add_argument("--bridge-args", default="")
    ap.add_argument("--fake-server", default=None, help="use mocks.fake_policy_server with these extra args instead")
    ap.add_argument("--replay-server", default=None,
                    help="use tools.replay_server with these args (e.g. '--zarr Z --episode 3 --speed 0.5') instead")
    ap.add_argument("--server-work", default=None, help="inference server work dir (default <work>/server)")
    ap.add_argument("--start-delay", type=float, default=4.0, help="camera origin = now + delay (shared clock)")
    a = ap.parse_args(argv)
    work = check_writable_dir(a.work_dir, "work dir")
    work.mkdir(parents=True, exist_ok=True)
    procs: dict[str, Proc] = {}
    report = {"script": a.script, "events": []}
    t0_wall = time.monotonic()
    try:
        ready = work / "server_ready.json"
        if ready.exists():
            ready.unlink()
        sw = a.server_work or str(work / "server")
        if a.replay_server is not None:
            procs["inference"] = Proc("inference", [a.py_re, "-u", "-m", "franka_deploy.tools.replay_server"]
                                      + shlex.split(a.replay_server), work, _env())
            time.sleep(2.0)
            ready.write_text("{}")
        elif a.fake_server is not None:
            procs["inference"] = Proc("inference", [a.py_re, "-u", "-m", "franka_deploy.mocks.fake_policy_server",
                                                    "--stats", str(work / "fake_server_stats.json")]
                                      + shlex.split(a.fake_server), work, _env())
            time.sleep(1.0)
            ready.write_text("{}")
        else:
            procs["inference"] = Proc("inference", [a.py_bp, "-u", "-m", "franka_deploy.inference.server", "--ckpt",
                                                    a.ckpt, "--work-dir", sw, "--dataset-path", a.prompt_zarr,
                                                    "--endpoint", "tcp://0.0.0.0:18765", "--ready-file", str(ready)],
                                      work, _env())
        start_now_ns = time.monotonic_ns() + int(a.start_delay * 1e9) + 60_000_000_000  # set after server is ready
        procs["spacemouse"] = Proc("spacemouse", [a.py_re, "-u", "-m", "franka_deploy.mocks.spacemouse_server"], work, _env())
        t = time.monotonic()
        while not ready.exists():
            if not procs["inference"].alive():
                raise RuntimeError("inference server died during startup (see inference.log)")
            if time.monotonic() - t > 300:
                raise TimeoutError("inference server not ready in 300 s")
            time.sleep(0.5)
        report["server_ready_s"] = round(time.monotonic() - t0_wall, 1)
        start_now_ns = time.monotonic_ns() + int(a.start_delay * 1e9)
        report["start_now_ns"] = start_now_ns
        cam_cmd = [a.py_re, "-u", "-m", "real_env.peripherals.rig_camera", "wrist_mock",
                   f"backend_opts.episode_dir={a.episode}", f"publish_mode={a.publish_mode}",
                   "backend_opts.loop=true", f"+backend_opts.start_now_ns={start_now_ns}", "stats_period_s=30"]
        cam_cmd += shlex.split(a.camera_overrides)
        procs["camera"] = Proc("camera", cam_cmd, work, _env())
        if a.controller == "franka":
            cfgdir = PKG / "configs"
            procs["bridge"] = Proc("bridge", [a.py_re, "-u", "-m", "franka_deploy.bridge.franka_bridge", "--config",
                                              str(cfgdir / "bridge.yaml"), "--allow-inject"] + shlex.split(a.bridge_args),
                                   work, _env())
            time.sleep(1.0)
            ctl = [a.py_re, "-u", "-m", "franka_deploy.controllers.franka_cartesian", "--config",
                   str(cfgdir / "franka_cartesian.yaml")]
        else:
            ctl = [a.py_re, "-u", "-m", "franka_deploy.controllers.mock_franka", "--mode", a.mode, "--log",
                   str(work / "controller_log.npz")]
        if a.mode == "replay" and a.controller == "mock":
            import numpy as np

            z = np.load(a.replay_npz)
            ctl += ["--replay-npz", a.replay_npz, "--start-now-ns", str(start_now_ns), "--span-ns", str(int(z["span_ns"]))]
        ctl += shlex.split(a.controller_args)
        procs["controller"] = Proc("controller", ctl, work, _env())
        time.sleep(a.start_delay + 1.0)
        for n, p in procs.items():
            if not p.alive():
                raise RuntimeError(f"{n} died during startup (see {n}.log)")
        cmd_file, stop_file = work / "commands.txt", work / "STOP"
        cmd_file.write_text("")
        if stop_file.exists():
            stop_file.unlink()
        task = [a.py_re, "-u", "-m", "franka_deploy.tasks.run_task", "--work-dir", str(work / "task"),
                "--command-file", str(cmd_file), "--stop-file", str(stop_file), "--no-stdin",
                "--continuous" if a.continuous else "--pause"] + shlex.split(a.task_args.replace("{START_NOW_NS}", str(start_now_ns)))
        procs["task"] = Proc("task", task, work, _env())
        t_task = time.monotonic()
        items = []
        for it in a.script.split(","):
            ts, act = it.split(":", 1)
            items.append((float(ts), act.strip()))
        for ts, act in sorted(items):
            while time.monotonic() - t_task < ts:
                if not procs["task"].alive():
                    break
                time.sleep(0.01)
            ev = {"t": round(time.monotonic() - t_task, 3), "action": act, "task_alive": procs["task"].alive()}
            if act == "end":
                report["events"].append(ev)
                break
            if act == "stopfile":
                stop_file.write_text("s")
                ev["t_mono"] = time.monotonic()
            elif act.startswith("kill:") or act.startswith("stop:") or act.startswith("sigint:"):
                kind, name = act.split(":")
                procs[name].signal({"kill": signal.SIGKILL, "stop": signal.SIGTERM, "sigint": signal.SIGINT}[kind])
                ev["t_mono"] = time.monotonic()
            elif act == "reflex":
                from franka_deploy.bridge.link import BridgeClient

                ev["t_mono"] = time.monotonic()
                ev["reply"] = str(BridgeClient("tcp://127.0.0.1:5565", timeout_s=0.5).request("inject_reflex"))
            else:
                with open(cmd_file, "a") as f:
                    f.write(act + "\n")
                ev["t_mono"] = time.monotonic()
            report["events"].append(ev)
        if procs["task"].alive():
            with open(cmd_file, "a") as f:
                f.write("q\n")
            try:
                procs["task"].p.wait(20)
            except subprocess.TimeoutExpired:
                report["task_quit_timeout"] = True
    finally:
        for n in ("task", "camera", "controller", "bridge", "spacemouse", "inference"):
            if n in procs:
                report.setdefault("exit_codes", {})[n] = procs[n].stop()
        report["total_s"] = round(time.monotonic() - t0_wall, 1)
        (work / "harness.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
