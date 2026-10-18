"""One-command deployment launcher: `python -m franka_deploy run --config deploy.yaml [--mock]`.

Start order (each with a readiness check + timeout; any failure aborts and tears everything down):
  bridge -> controller -> camera -> inference (READY after warm-up) -> spacemouse -> task.
Preflight (launch/preflight.py): static checks before anything starts (dirs outside the repo, disk budget, clean
/dev/shm, free ports, checkpoint, prompt-zarr sha256, imagecodecs pin, rig.yaml not the default, agent at 20 Hz),
then per-component checks as they come up (bridge not in reflex, FK near home, camera rate + frame selection,
checkpoint staged in the work dir).
Supervisor: a child dying -> HOLD through the controller's hold topic first (or the bridge watchdog if the
controller itself died), then ordered teardown (task, controller, camera, inference, spacemouse, bridge last),
/dev/shm cleanup, port check, summary json, non-zero exit.
Terminal: the launcher owns it; keys c C s y n r R q are appended to the task's command file (s/y/n are turned into a
HOLD by the stop watcher within ~10 ms); `?` prints help. Ctrl+C at any phase: clean exit <= 3 s; a second Ctrl+C
kills everything immediately.
Session dir (under work_root): config/ snapshot (copied into every episode), logs/, server/, task/episodes/...
"""

from __future__ import annotations

import copy
import json
import os
import select
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from franka_deploy.launch import preflight as pf
from franka_deploy.paths import REPO_ROOT, check_writable_dir
from franka_deploy.station import StationError, station_from_deploy_cfg

PKG = Path(__file__).resolve().parents[2]
KEYS = "cCsynrRq"
TEARDOWN_ORDER = ("task", "controller", "camera", "inference", "spacemouse", "bridge")


class Abort(RuntimeError):
    pass


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def station_contradictions(target: str, fragment: dict, override: dict, prefix: str = "") -> list[dict]:
    """Override leaves that set a different value than the station fragment holds at the same path."""
    out = []
    for k, v in (override or {}).items():
        path = f"{prefix}{k}"
        if k not in fragment:
            continue
        if isinstance(v, dict) and isinstance(fragment[k], dict):
            out += station_contradictions(target, fragment[k], v, path + ".")
        elif v != fragment[k]:
            out.append({"target": target, "path": path, "station": fragment[k], "override": v})
    return out


def run_mode_of(cfg: dict) -> str:
    """`run_mode: sim|real`; real when a station exists and the key is missing (the safe option)."""
    return str(cfg.get("run_mode", "real"))


def load_config(path, profile: str) -> dict:
    raw = yaml.safe_load(open(path))
    cfg = deep_merge({k: v for k, v in raw.items() if k != "profiles"}, raw.get("profiles", {}).get(profile, {}))
    cfg["profile"] = profile
    cfg["config_path"] = str(Path(path).resolve())
    return cfg


POLICY_KINDS = ("bp_server", "replay", "fake", "command")
COMMAND_PLACEHOLDERS = ("python", "endpoint", "ready_file", "work_dir", "checkpoint", "prompt_zarr")


def resolve_policy(cfg: dict, station=None) -> dict:
    """The policy slot as one dict {id, kind, python, ready_timeout_s, ...}.

    With a station `run.policy` its group is used. Without one the deploy.yaml keys mean what they always did:
    `inference.replay` (the --replay-episode flag) -> kind replay, otherwise `checkpoint`/`prompt_zarr` -> bp_server.
    The `--replay-episode` flag also wins over a station policy (it is an explicit operator choice).
    """
    inf, c = cfg.get("inference", {}), cfg
    rp = inf.get("replay")
    if rp:
        return {"id": "deploy.yaml:replay", "kind": "replay", "ready_timeout_s": 60,
                "episode_index": rp["episode"], "speed_scale": rp.get("speed", 1.0),
                "start_frame_index": rp.get("start_frame", 0), "zarr": rp.get("zarr")}
    if station is not None and "policy" in station.groups:
        pol = station.get("policy")
        pol.setdefault("id", station.ids.get("policy"))
        return pol
    return {"id": "deploy.yaml", "kind": "bp_server", "python": c["python"]["bp"], "checkpoint": c["checkpoint"],
            "prompt_zarr": c["prompt_zarr"], "device": inf.get("device", "cuda"),
            "ready_timeout_s": inf.get("ready_timeout_s", 300)}


def policy_python(cfg: dict, pol: dict) -> str:
    """`python` of a policy: a key of cfg.python (bp, realenv) or a path (what `${host.python.bp}` resolves to)."""
    py = pol.get("python") or cfg["python"]["realenv" if pol.get("kind") in ("replay", "fake") else "bp"]
    return cfg["python"].get(py, py) if isinstance(py, str) else py


def command_template(pol: dict) -> list[str]:
    cmd = pol.get("cmd")
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    return [str(a) for a in (cmd or [])]


def policy_problems(cfg: dict, pol: dict) -> list[str]:
    """Static validation of a resolved policy (no process, no file reads except the checkpoint existence)."""
    kind = pol.get("kind")
    if kind not in POLICY_KINDS:
        return [f"policy {pol.get('id')}: unknown kind {kind!r} (known: {', '.join(POLICY_KINDS)})"]
    out = []
    if kind == "command":
        tpl = command_template(pol)
        text = " ".join(tpl)
        out += [f"policy {pol.get('id')}: cmd lacks {{{k}}}" for k in ("endpoint", "ready_file") if "{" + k + "}" not in text]
        try:
            _ = [a.format_map({k: "x" for k in COMMAND_PLACEHOLDERS}) for a in tpl]
        except (KeyError, IndexError, ValueError) as e:
            out.append(f"policy {pol.get('id')}: cmd has an unknown or malformed placeholder ({e}); "
                       f"known: {', '.join(COMMAND_PLACEHOLDERS)}")
    if kind == "bp_server":
        ok, detail = pf.checkpoint_ok(pol.get("checkpoint") or cfg.get("checkpoint"))
        if not ok:
            out.append(f"policy {pol.get('id')}: checkpoint: {detail}")
    return out


def port_of(ep):
    return ep


class Child:
    def __init__(self, name, cmd, log_path, env, cwd):
        self.name, self.cmd = name, cmd
        self.log = open(log_path, "w")
        self.log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        self.log.flush()
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                  cwd=str(cwd), start_new_session=True)
        self.expected_exit = False

    def alive(self):
        return self.p.poll() is None

    def signal(self, sig):
        if self.alive():
            try:
                os.killpg(self.p.pid, sig)
            except ProcessLookupError:
                pass

    def stop(self, timeout_s):
        self.expected_exit = True
        if self.alive():
            self.signal(signal.SIGINT)
            try:
                self.p.wait(timeout_s)
            except subprocess.TimeoutExpired:
                self.signal(signal.SIGKILL)
                try:
                    self.p.wait(2)
                except subprocess.TimeoutExpired:
                    pass
        try:
            self.log.close()
        except Exception:
            pass
        return self.p.returncode


class Launcher:
    def __init__(self, cfg: dict, interactive: bool | None = None, script: str | None = None):
        self.cfg = cfg
        self.station = station_from_deploy_cfg(cfg)   # None for profiles without `station:`
        self.policy = resolve_policy(cfg, self.station)
        self.t0 = time.monotonic()
        self.children: dict[str, Child] = {}
        self.events: list[dict] = []
        self.sigints = 0
        self.interrupted = False
        self.exit_code = 0
        self.exit_reason = None
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.script = script
        self.work_root = Path(cfg["work_root"])
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session = self.work_root / f"session_{stamp}"
        self.cmd_file = self.session / "commands.txt"
        self.stop_file = self.session / "STOP"
        self.summary = {"profile": cfg["profile"], "session": str(self.session), "phases": {}, "preflight": {},
                        "policy": {"id": self.policy.get("id"), "kind": self.policy.get("kind")}}

    # ------------------------------------------------------------------ utils
    def log(self, kind, **kw):
        ev = {"t": round(time.monotonic() - self.t0, 3), "kind": kind, **kw}
        self.events.append(ev)
        print(f"[launcher {ev['t']:7.2f}s] {kind} {json.dumps(kw, default=str) if kw else ''}", flush=True)

    def check(self, name, result):
        ok, detail = result
        self.summary["preflight"][name] = {"ok": bool(ok), "detail": detail}
        self.log("preflight", check=name, ok=bool(ok), detail=detail)
        if not ok:
            raise Abort(f"preflight {name} failed: {detail}")

    def env(self):
        e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        cache = self.work_root / "cache"
        e.update({"PYTHONPATH": str(PKG), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                  "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": str(cache / "xdg"), "MPLCONFIGDIR": str(cache / "mpl"),
                  "NUMBA_CACHE_DIR": str(cache / "numba"), "HF_HOME": str(cache / "hf"), "TORCH_HOME": str(cache / "torch"),
                  "LIBERO_CONFIG_PATH": str(cache / "libero_unused"), "PIP_NO_CACHE_DIR": "1"})
        extra = [str(p) for p in self.cfg.get("extra_path", [])]
        if extra:
            e["PATH"] = os.pathsep.join(extra + [e.get("PATH", "")])
        return e

    def start(self, name, cmd):
        self.children[name] = Child(name, cmd, self.session / "logs" / f"{name}.log", self.env(), self.session)
        self.log("started", child=name, pid=self.children[name].p.pid)

    def wait_until(self, name, fn, timeout_s, what):
        t = time.monotonic()
        while True:
            if self.interrupted:
                raise Abort("interrupted during startup")
            if name in self.children and not self.children[name].alive():
                raise Abort(f"{name} died during startup (see logs/{name}.log)")
            try:
                r = fn()
                if r:
                    self.summary["phases"][name] = round(time.monotonic() - t, 2)
                    self.log("ready", child=name, after_s=round(time.monotonic() - t, 2))
                    return r
            except Exception:
                pass
            if time.monotonic() - t > timeout_s:
                raise Abort(f"{name} not ready within {timeout_s} s ({what})")
            time.sleep(0.1)

    # ------------------------------------------------------------------ session snapshot
    def snapshot(self):
        cfgd = self.session / "config"
        cfgd.mkdir(parents=True, exist_ok=True)
        (self.session / "logs").mkdir(exist_ok=True)
        c = self.cfg
        bridge = yaml.safe_load(open(self.resolve(c["bridge"]["config"])))
        st, overrides = self.station, []
        if st is not None:   # precedence: bridge.yaml < station fragment < deploy.yaml bridge.override
            frag = st.bridge_fragment()
            overrides += station_contradictions("bridge", frag, c["bridge"].get("override", {}))
            bridge = deep_merge(bridge, frag)
        bridge = deep_merge(bridge, c["bridge"].get("override", {}))
        bridge["endpoint"] = c["bridge"]["endpoint"]
        (cfgd / "bridge.yaml").write_text(yaml.safe_dump(bridge))
        ctrl = yaml.safe_load(open(self.resolve(c["controller"]["config"])))
        if st is not None:   # precedence: franka_cartesian.yaml < station fragment < deploy.yaml controller.override
            frag = st.controller_fragment()
            overrides += station_contradictions("controller", frag, c["controller"].get("override", {}))
            ctrl = deep_merge(ctrl, frag)
        ctrl = deep_merge(ctrl, c["controller"].get("override", {}))
        ctrl["bridge_endpoint"] = c["bridge"]["endpoint"]
        ctrl["controller"]["endpoint"] = c["controller"]["endpoint"]
        if c["home"].get("pose"):
            ctrl["home_tcp_xyz_wxyz"] = list(c["home"]["pose"])
        (cfgd / "franka_cartesian.yaml").write_text(yaml.safe_dump(ctrl))
        (cfgd / "deploy_resolved.yaml").write_text(yaml.safe_dump(c))
        if st is not None:
            (cfgd / "station_resolved.yaml").write_text(st.to_yaml())
            self.summary["station_fingerprint"] = st.fingerprint()
            self.summary["preflight"]["station_overrides"] = overrides
            if overrides:
                self.log("station_overrides", overrides=overrides)
        shutil.copy(c["config_path"], cfgd / "deploy.yaml")
        if c["camera"].get("rig_config_path"):
            try:
                shutil.copy(self.rig_path(), cfgd / "rig.yaml")
            except Exception:
                pass
        g = lambda *a: subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True).stdout  # noqa: E731
        import hashlib

        diff = g("diff", "HEAD", "--", "controlSW")
        untracked = g("ls-files", "--others", "--exclude-standard", "--", "controlSW").split()
        h = hashlib.sha256(diff.encode())
        for f in sorted(untracked):
            p = REPO_ROOT / f
            if p.is_file():
                h.update(f.encode())
                h.update(p.read_bytes())
        pf.write_json(cfgd / "git.json", {"describe": g("describe", "--always", "--dirty").strip(),
                                          "head": g("rev-parse", "HEAD").strip(),
                                          "controlSW_diff_sha256": h.hexdigest(), "controlSW_untracked_files": len(untracked)})
        vers = {}
        for k in ("bp", "realenv"):
            r = subprocess.run([c["python"][k], "-m", "pip", "freeze"], capture_output=True, text=True, env=self.env())
            vers[k] = r.stdout.splitlines()
        pf.write_json(cfgd / "versions.json", vers)
        self.cmd_file.write_text("")

    def resolve(self, p):
        p = Path(str(p))
        return p if p.is_absolute() else PKG / p

    def rig_path(self):
        r = self.cfg["camera"].get("rig_config_path")
        if r is None and self.cfg["camera"].get("episode_dir"):
            return str(Path(self.cfg["camera"]["episode_dir"]) / "rig.yaml")
        return r

    # ------------------------------------------------------------------ phases
    def static_preflight(self):
        c = self.cfg
        self.check("dirs_outside_repo", pf.dirs_outside_repo(self.work_root, self.session))
        self.check("disk_free", pf.disk_free(self.work_root, c.get("disk_budget_gb", 2.0)))
        self.check("shm_clean", pf.shm_clean())
        eps = [c["bridge"]["endpoint"], c["controller"]["endpoint"], c["camera"]["endpoint"], c["inference"]["endpoint"],
               c["spacemouse"]["endpoint"], c["camera"]["logger_endpoint"]]
        self.check("ports_free", pf.ports_free(eps))
        pol, kind = self.policy, self.policy.get("kind")
        self.check("policy_valid", (not (probs := policy_problems(c, pol)), "; ".join(probs) or f"{pol.get('id')} ({kind})"))
        if kind == "bp_server":
            ckpt, zarr = pol.get("checkpoint") or c["checkpoint"], pol.get("prompt_zarr") or c["prompt_zarr"]
            self.check("checkpoint", pf.checkpoint_ok(ckpt))
            ok, sha = pf.prompt_zarr_sha(zarr, pol.get("prompt_zarr_sha256", c.get("prompt_zarr_sha256")),
                                         self.work_root / "cache")
            self.check("prompt_zarr_sha256", (ok, sha))
            ck_sha = pf.sha256_file(ckpt, self.work_root / "cache")
            pf.write_json(self.session / "config" / "checksums.json", {"checkpoint": str(Path(ckpt).resolve()),
                          "checkpoint_sha256": ck_sha, "prompt_zarr": str(Path(zarr).resolve()),
                          "prompt_zarr_sha256": sha, "prompt_index": c["task"]["prompt_index"]})
            self.check("imagecodecs_pin", pf.imagecodecs_pin(policy_python(c, pol)))
        elif kind == "replay":
            zp = Path(pol.get("zarr") or pol.get("prompt_zarr") or c["prompt_zarr"])
            ok, sha = pf.prompt_zarr_sha(str(zp), pol.get("prompt_zarr_sha256", c.get("prompt_zarr_sha256")),
                                         self.work_root / "cache")
            self.check("prompt_zarr_sha256", (ok, sha))
            pf.write_json(self.session / "config" / "checksums.json", {   # no model in a replay: no checkpoint entries
                "prompt_zarr": str(zp.resolve()), "prompt_zarr_sha256": sha, "prompt_index": c["task"]["prompt_index"]})
        self.check("rig_not_default", pf.rig_not_default(self.rig_path()))
        self.check("agent_hz", pf.agent_hz_ok(c["task"].get("agent_hz", 20)))
        if self.station is not None:
            self.check("station_valid", (True, f"{self.station.ids} fingerprint {self.station.fingerprint()}"))
            self.check("station_placeholders", pf.station_placeholders(self.station, run_mode_of(c),
                                                                       c.get("allow_placeholders", [])))
            self.check("tooling_sensor_compat", pf.tooling_sensor_compat(self.station))

    def policy_command(self, pol: dict | None = None) -> tuple[list[str], float]:
        """Command line and ready timeout (s) for a policy kind. Pure: starts nothing."""
        c, pol = self.cfg, pol or self.policy
        kind, py = pol.get("kind"), policy_python(self.cfg, pol)
        endpoint, ready = c["inference"]["endpoint"], str(self.session / "server_ready.json")
        work = str(self.session / "server")
        ckpt, zarr = pol.get("checkpoint") or c.get("checkpoint"), pol.get("prompt_zarr") or c.get("prompt_zarr")
        timeout = float(pol.get("ready_timeout_s", 300))
        if kind == "bp_server":
            device = c["inference"].get("device", "cuda")
            if self.station is not None and "policy" in self.station.groups:
                device = self.station.get("host.inference_device", device)
            return [py, "-u", "-m", "franka_deploy.inference.server", "--ckpt", ckpt, "--work-dir", work,
                    "--dataset-path", zarr, "--endpoint", endpoint, "--device", device, "--ready-file", ready], timeout
        if kind == "replay":  # open-loop action replay: same topics, the demo's own future instead of a model
            return [py, "-u", "-m", "franka_deploy.tools.replay_server", "--zarr", pol.get("zarr") or zarr,
                    "--episode", str(pol.get("episode_index", 0)), "--speed", str(pol.get("speed_scale", 1.0)),
                    "--exec-h", str(c["task"].get("exec_h", 8)), "--start-frame", str(pol.get("start_frame_index", 0)),
                    "--endpoint", endpoint, "--ready-file", ready] + ([] if c["task"].get("pause", True) else ["--continuous"]), timeout
        if kind == "fake":  # constant hold chunks, no GPU and no checkpoint
            return [py, "-u", "-m", "franka_deploy.mocks.fake_policy_server", "--endpoint", endpoint,
                    "--ready-file", ready, "--delay", str(pol.get("delay_s", 0.0))], timeout
        if kind == "command":
            sub = {"python": py, "endpoint": endpoint, "ready_file": ready, "work_dir": work,
                   "checkpoint": str(ckpt or ""), "prompt_zarr": str(zarr or "")}
            return [a.format_map(sub) for a in command_template(pol)], timeout
        raise Abort(f"unknown policy kind {kind!r} (known: {', '.join(POLICY_KINDS)})")

    def start_policy(self, pol: dict | None = None):
        pol = pol or self.policy
        cmd, timeout = self.policy_command(pol)
        ready = self.session / "server_ready.json"
        self.start("inference", cmd)
        self.wait_until("inference", lambda: ready.exists(), timeout, f"policy ready file ({pol.get('kind')})")
        if pol.get("kind") == "bp_server":
            self.check("checkpoint_staged", pf.checkpoint_staged(self.session / "server",
                                                                 pol.get("checkpoint") or self.cfg["checkpoint"]))

    def start_all(self):
        c, py = self.cfg, self.cfg["python"]["realenv"]
        logs = self.session / "logs"
        bcmd = [py, "-u", "-m", "franka_deploy.bridge.franka_bridge", "--config", str(self.session / "config" / "bridge.yaml"),
                "--log-jsonl", str(logs / "bridge.jsonl")] + (["--allow-inject"] if c["bridge"].get("allow_inject") else [])
        self.start("bridge", bcmd)
        from franka_deploy.bridge.link import BridgeClient

        bc = BridgeClient(c["bridge"]["endpoint"].replace("0.0.0.0", "127.0.0.1"), timeout_s=0.3)
        st = self.wait_until("bridge", lambda: bc.request("get_state"), 15, "get_state")
        self.check("bridge_not_reflex", pf.bridge_not_reflex(st))

        self.start("controller", [py, "-u", "-m", "franka_deploy.controllers.franka_cartesian", "--config",
                                  str(self.session / "config" / "franka_cartesian.yaml")])
        from franka_deploy.rmq_rpc import NoResendClient
        from robotmq import deserialize

        self.ctrl_client = NoResendClient("launcher", c["controller"]["endpoint"], default_timeout_s=0.1)

        def ctrl_status():
            raw, _ = self.ctrl_client.peek_data(topic="status", n=-1)
            return deserialize(raw[0]) if raw else None

        self.wait_until("controller", ctrl_status, 20, "status topic")
        raw, _ = self.ctrl_client.peek_data(topic="eef_pose", n=-1)
        fk = deserialize(raw[0])["eef_xyz_wxyz"]
        self.check("fk_near_home", pf.fk_near_home(fk, c["home"]["xyz"], c["home"]["max_dist_m"]))

        cam = c["camera"]
        ccmd = [py, "-u", "-m", "real_env.peripherals.rig_camera", cam["config"], f"server_endpoint={cam['endpoint']}",
                f"logger_endpoint={cam['logger_endpoint']}", "stats_period_s=5"]
        if cam.get("rig_config_path"):
            ccmd.append(f"rig_config_path={cam['rig_config_path']}")
        if cam.get("episode_dir"):
            ccmd += [f"backend_opts.episode_dir={cam['episode_dir']}", "backend_opts.loop=true"]
        ccmd += list(cam.get("overrides", []))
        self.start("camera", ccmd)
        from franka_deploy.peripherals.safe_camera_client import SafeCameraClient

        cc = self.wait_until("camera", lambda: SafeCameraClient("launcher_cam", cam["endpoint"]), 30, "camera config")
        self.wait_until("camera", lambda: cc.get_latest_images_dict_THWC(4, extra=0), 15, "frames")
        self.check("camera_rate_and_selection", pf.camera_ok(cc))

        self.start_policy()

        sm = c["spacemouse"]
        if sm.get("mock", False):
            self.start("spacemouse", [py, "-u", "-m", "franka_deploy.mocks.spacemouse_server", "--endpoint", sm["endpoint"]])
        else:
            self.start("spacemouse", [py, "-u", "-m", "robot_utils.teleop_utils.spacemouse.spacemouse_server"])
        smc = NoResendClient("launcher_sm", sm["endpoint"], default_timeout_s=0.1)
        self.wait_until("spacemouse", lambda: smc.peek_data(topic="spacemouse_state", n=-1)[0], 10, "spacemouse_state")

        t, tready = c["task"], self.session / "task_ready.json"
        tcmd = [py, "-u", "-m", "franka_deploy.tasks.run_task", "--work-dir", str(self.session / "task"),
                "--task-name", t["name"], "--prompt-index", str(t["prompt_index"]), "--exec-h", str(t.get("exec_h", 8)),
                "--anchor", t.get("anchor", "target"), "--inference-timeout", str(t.get("inference_timeout_s", 2.0)),
                "--width-proprio", t.get("width_proprio", "commanded"), "--v-max", str(t.get("v_max", 0.25)),
                "--w-max", str(t.get("w_max", 1.0)), "--tap-images-every", str(t.get("tap_images_every", 0)),
                "--command-file", str(self.cmd_file), "--stop-file", str(self.stop_file), "--no-stdin",
                "--policy-endpoint", c["inference"]["endpoint"].replace("0.0.0.0", "localhost"), "--camera-endpoint", cam["endpoint"],
                "--controller-endpoint", c["controller"]["endpoint"], "--spacemouse-endpoint", sm["endpoint"],
                "--session-dir", str(self.session / "config"), "--bridge-log", str(logs / "bridge.jsonl"),
                "--camera-log", str(logs / "camera.log"), "--ready-file", str(tready),
                "--home-xyz", *[str(x) for x in c["home"]["xyz"]], "--home-max-dist", str(c["home"]["max_dist_m"]),
                "--pause" if t.get("pause", True) else "--continuous"]
        if self.station is not None:
            ta = self.station.task_args()
            for flag, key in (("--lookahead-s", "lookahead_s"), ("--home-gripper-m", "home_gripper_m"),
                              ("--camera-latency-s", "camera_latency_s"), ("--policy-latency-s", "policy_latency_s")):
                if key in ta:
                    tcmd += [flag, str(ta[key])]
        if cam.get("video", False):
            tcmd += ["--video-logger-endpoint", cam["logger_endpoint"]]
        self.start("task", tcmd)
        self.wait_until("task", lambda: tready.exists(), 60, "task ready file")

    # ------------------------------------------------------------------ supervision
    def hold_now(self, reason):
        t = time.monotonic()
        try:
            from robotmq import serialize

            if "controller" in self.children and self.children["controller"].alive():
                self.ctrl_client.put_data(topic="hold", data=serialize({"hold": True, "reason": reason, "t": t}))
                self.log("hold_sent", reason=reason, ms=round(1000 * (time.monotonic() - t), 1))
            else:
                self.log("hold_by_bridge_watchdog", reason=reason)
        except Exception as e:
            self.log("hold_failed", reason=reason, error=repr(e))

    def forward(self, k):
        if k == "?":
            print("keys: c start episode | C single step | s stop | y stop+success | n stop+fail | r reset | "
                  "R reset policy | q quit | Ctrl+C exit", flush=True)
            return
        if k in KEYS or k.startswith("note:"):
            with open(self.cmd_file, "a") as f:
                f.write(k + "\n")
            self.log("key", key=k)

    def _stdin_thread(self):
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._term_restore = (fd, old)
        try:
            while not self._done:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    self.forward(sys.stdin.read(1))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _script_thread(self):
        t0 = time.monotonic()
        for item in self.script.split(","):
            ts, act = item.split(":", 1)
            while time.monotonic() - t0 < float(ts) and not self._done:
                time.sleep(0.01)
            if self._done:
                return
            if act.startswith("kill:") or act.startswith("sigint_launcher"):
                if act.startswith("kill:"):
                    name = act.split(":")[1]
                    self.log("inject_kill", child=name, t_mono=time.monotonic())
                    self.children[name].signal(signal.SIGKILL)
                else:
                    os.kill(os.getpid(), signal.SIGINT)
            elif act == "reflex":
                from franka_deploy.bridge.link import BridgeClient

                BridgeClient(self.cfg["bridge"]["endpoint"].replace("0.0.0.0", "127.0.0.1"), 0.5).request("inject_reflex")
                self.log("inject_reflex")
            else:
                self.forward(act)

    def supervise(self):
        while True:
            if self.interrupted:
                self.exit_code, self.exit_reason = 130, "ctrl_c"
                return
            for name, ch in self.children.items():
                if not ch.alive() and not ch.expected_exit:
                    rc = ch.p.returncode
                    if name == "task" and rc == 0:
                        self.exit_code, self.exit_reason = 0, "quit"
                        ch.expected_exit = True
                        return
                    self.log("child_died", child=name, returncode=rc, t_mono=time.monotonic())
                    self.hold_now(f"{name} died")
                    self.exit_code, self.exit_reason = 1, f"{name} died (rc {rc})"
                    return
            time.sleep(0.05)

    def teardown(self):
        quick = self.interrupted
        budget = 0.3 if quick else 5.0
        codes = {}
        for name in TEARDOWN_ORDER:
            if name in self.children:
                if name == "task" and not quick and self.children[name].alive():
                    self.forward("q")
                    try:
                        self.children[name].p.wait(3)
                    except subprocess.TimeoutExpired:
                        pass
                codes[name] = self.children[name].stop(1.2 if (quick and name == "task") else budget)
        removed = []
        for ch in self.children.values():
            from franka_deploy.harness.dryrun import cleanup_shm

            removed += cleanup_shm(ch.p.pid)
        self.summary["child_exit_codes"] = codes
        try:
            from franka_deploy.eval.salvage import salvage_session

            self.summary["salvage"] = salvage_session(str(self.session), (self.exit_reason or "shutdown").replace(" ", "_"))
        except Exception as e:
            self.summary["salvage_error"] = repr(e)
        self.summary["shm_removed"] = removed
        c = self.cfg
        eps = [c["bridge"]["endpoint"], c["controller"]["endpoint"], c["camera"]["endpoint"], c["inference"]["endpoint"],
               c["spacemouse"]["endpoint"], c["camera"]["logger_endpoint"]]
        self.summary["ports_free_after"] = pf.ports_free(eps)
        self.summary["shm_clean_after"] = pf.shm_clean()

    def on_sigint(self, *_):
        self.sigints += 1
        self.interrupted = True
        if self.sigints >= 2:  # second Ctrl+C: force (kill all, reap briefly, clean shm, exit)
            from franka_deploy.harness.dryrun import cleanup_shm

            for ch in self.children.values():
                ch.signal(signal.SIGKILL)
            t = time.monotonic() + 1.0
            for ch in self.children.values():
                try:
                    ch.p.wait(max(0.01, t - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
                cleanup_shm(ch.p.pid)
            os._exit(131)

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.on_sigint)
        signal.signal(signal.SIGTERM, self.on_sigint)
        self._done = False
        th = None
        try:
            check_writable_dir(self.work_root, "work_root")
            self.session.mkdir(parents=True, exist_ok=True)
            self.snapshot()
            self.static_preflight()
            self.start_all()
            self.summary["ready_s"] = round(time.monotonic() - self.t0, 2)
            self.log("READY", after_s=self.summary["ready_s"], session=str(self.session))
            if self.interactive:
                th = threading.Thread(target=self._stdin_thread, daemon=True)
                th.start()
            if self.script:
                threading.Thread(target=self._script_thread, daemon=True).start()
            self.supervise()
        except Abort as e:
            self.exit_code = 130 if self.interrupted else 2
            self.exit_reason = str(e)
            self.log("abort", reason=str(e))
            if self.children:
                self.hold_now("abort")
        except Exception as e:
            self.exit_code, self.exit_reason = 3, repr(e)
            self.log("launcher_error", error=repr(e))
            self.hold_now("launcher error")
        finally:
            self._done = True
            t_td = time.monotonic()
            self.teardown()
            self.summary.update(exit_code=self.exit_code, exit_reason=self.exit_reason,
                                teardown_s=round(time.monotonic() - t_td, 2), total_s=round(time.monotonic() - self.t0, 2),
                                events=self.events)
            if self.session.exists():
                pf.write_json(self.session / "launcher_summary.json", self.summary)
            self.log("exit", code=self.exit_code, reason=self.exit_reason, teardown_s=self.summary["teardown_s"])
        return self.exit_code


def check_config(cfg: dict) -> int:
    """Validate a deploy config without hardware (e.g. the Jetson profile on the laptop)."""
    problems, notes = [], []
    req = ["work_root", "python", "checkpoint", "prompt_zarr", "task", "camera", "bridge", "controller", "inference",
           "spacemouse", "home"]
    problems += [f"missing key {k}" for k in req if k not in cfg]
    if problems:
        print("\n".join(problems))
        return 1
    try:
        station = station_from_deploy_cfg(cfg)
    except StationError as e:
        print(f"FAIL station: {e}")
        return 1
    pol = resolve_policy(cfg, station)
    r = [("dirs_outside_repo", pf.dirs_outside_repo(cfg["work_root"]))]
    if pol["kind"] == "bp_server":   # the checkpoint and prompt zarr only matter for a real model
        r.append(("checkpoint", pf.checkpoint_ok(pol.get("checkpoint") or cfg["checkpoint"])))
    r.append(("agent_hz", pf.agent_hz_ok(cfg["task"].get("agent_hz", 20))))
    if pol["kind"] in ("bp_server", "replay"):
        zp = Path(pol.get("zarr") or pol.get("prompt_zarr") or cfg["prompt_zarr"])
        r.append(("prompt_zarr_exists", (zp.is_file(), str(zp))))
    if station is not None or pol["kind"] != "bp_server":
        probs = policy_problems(cfg, pol)
        r.append(("policy", (not probs, "; ".join(probs) or f"{pol.get('id')} ({pol['kind']})")))
    for k in ("bp", "realenv"):
        r.append((f"python_{k}", (Path(cfg["python"][k]).exists(), cfg["python"][k])))
    bcfg = deep_merge(yaml.safe_load(open(PKG / cfg["bridge"]["config"] if not Path(cfg["bridge"]["config"]).is_absolute()
                                          else cfg["bridge"]["config"])), cfg["bridge"].get("override", {}))
    btype = bcfg.get("backend", {}).get("type")
    r.append(("bridge_backend", (btype in ("fake", "panda_py"), str(btype))))
    if btype == "panda_py":
        pp = bcfg["backend"].get("panda_py", {})
        r.append(("panda_py_hostname", (bool(pp.get("hostname")), str(pp.get("hostname")))))
        r.append(("panda_py_versions_pinned", (bool(bcfg["backend"].get("panda_py_version") and bcfg["backend"].get("libfranka_version")),
                                               f"{bcfg['backend'].get('panda_py_version')} / {bcfg['backend'].get('libfranka_version')}")))
        notes.append("panda-py import / FCI connection: hardware-only (not checked)")
    cam = cfg["camera"]
    r.append(("camera_config", (cam.get("config") in ("wrist_right", "wrist_mock"), str(cam.get("config")))))
    if cam.get("config") == "wrist_right":
        notes.append("Argus cameras: hardware-only (not checked)")
        rp = cam.get("rig_config_path")
        r.append(("rig_not_default", pf.rig_not_default(rp) if rp and Path(str(rp)).exists()
                  else (False, f"rig config {rp} missing or default")))
    st = station
    if st is not None:
        mode = run_mode_of(cfg)
        crit = {p for p, _ in st.placeholders(critical_only=True)}
        print(f"NOTE station {st.ids} fingerprint {st.fingerprint()} run_mode {mode}")
        for path, p in st.placeholders(critical_only=True):
            print(f"NOTE   CRITICAL {p.status:11s} {path}")
        print(f"NOTE   {len(crit)} critical, {len(st.placeholders())} placeholder/hw value(s) in total "
              f"(python -m franka_deploy.station validate --all -v lists them)")
        r.append(("station_placeholders", pf.station_placeholders(st, mode, cfg.get("allow_placeholders", []))))
        r.append(("tooling_sensor_compat", pf.tooling_sensor_compat(st)))
    ok = True
    for name, (good, detail) in r:
        print(f"{'OK  ' if good else 'FAIL'} {name}: {detail}")
        ok &= bool(good)
    for n in notes:
        print(f"NOTE {n}")
    return 0 if ok else 1
