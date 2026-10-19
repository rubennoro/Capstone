"""One-command deployment launcher: `python -m franka_deploy run --config deploy.yaml [--mock]`.

Start order (each with a readiness check + timeout; any failure aborts and tears everything down):
  bridge -> [hand] -> controller -> camera [-> sender] -> inference (READY after warm-up) -> spacemouse -> task.
  camera.kind rig_local (default): RigCamera; wireless: the B5 receiver (peripherals/wireless_camera.py, camera.yaml =
  station.camera_config()). Sim profiles add a `sender` child that imitates the sensor over the run's emulated link
  (run.sim.image_source: video_replay -> stream.iphone_replay_sender; rig_raw_replay -> B7; render -> A7/B8); its death
  is not fatal (like a phone dropping off Wi-Fi: the receiver goes stale and the task's image-age guard holds).
  `hand` (franka_deploy.gripper.hand_server, A5) runs when the gripper is franka_hand on an endpoint other than the
  bridge's (real hardware, 5566).
Preflight (launch/preflight.py): static checks before anything starts (dirs outside the repo, disk budget, clean
/dev/shm, free ports, checkpoint, prompt-zarr sha256, imagecodecs pin, rig.yaml not the default, agent at 20 Hz),
then per-component checks as they come up (bridge not in reflex, FK near home, camera rate + frame selection,
checkpoint staged in the work dir; wireless: websocket upgrade, fps, image age, reconnects, v2 clock sync + hello).
Supervisor: a child dying -> HOLD through the controller's hold topic first (or the bridge watchdog if the
controller itself died), then ordered teardown (TEARDOWN_ORDER: task, controller, sender, camera, inference,
spacemouse, hand, bridge last),
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
SIM_ROOT = PKG.parent / "franka_sim"           # Track A package dir (on the child PYTHONPATH when it exists)
KEYS = "cCsynrRq"
TEARDOWN_ORDER = ("task", "controller", "sender", "camera", "inference", "spacemouse", "hand", "bridge")
CAMERA_KINDS = ("rig_local", "wireless")
# run.sim.image_source -> (sender module, its source file, the phase that provides it). The launcher starts the sender
# in sim profiles only; a missing file is reported by check_config / preflight ("not available yet").
SENDERS = {
    "video_replay": ("franka_deploy.stream.iphone_replay_sender", PKG / "franka_deploy" / "stream" / "iphone_replay_sender.py", "B4"),
    "rig_raw_replay": ("franka_deploy.stream.rig_replay_sender", PKG / "franka_deploy" / "stream" / "rig_replay_sender.py", "B7"),
    "render": ("franka_sim.senders.render_sender", SIM_ROOT / "franka_sim" / "senders" / "render_sender.py", "A7/B8"),
}
HAND_SERVER = PKG / "franka_deploy" / "gripper" / "hand_server.py"            # A5 (real Franka Hand, port 5566)
MUJOCO_BACKEND = SIM_ROOT / "franka_sim" / "backend.py"                        # A4 (MuJoCoBackend)
SIM_STATE_ENDPOINT = "tcp://127.0.0.1:5570"                                    # A6 sim-state PUB (INTERFACES.md section 2)
DEFAULT_RAW_ROOT = "data/raw/2026-10-01"


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


def split_script(script: str) -> list[str]:
    """'t:act,t:act' items; a comma only separates items when the next one starts with '<seconds>:', so an action
    may carry JSON (e.g. 'sim_reset:{"eraser_xy_yaw": [0.5, 0.0, 0.3]}')."""
    import re

    return [x for x in re.split(r",(?=\s*\d+(?:\.\d+)?:)", script) if x.strip()]


def camera_kind(cfg: dict) -> str:
    """`camera.kind`: rig_local (default when absent: today's RigCamera command) | wireless (B5 receiver)."""
    return str(cfg["camera"].get("kind", "rig_local"))


def norm_endpoint(ep) -> str:
    """tcp://localhost:N, tcp://0.0.0.0:N and tcp://127.0.0.1:N are the same local endpoint."""
    host, _, port = str(ep).replace("tcp://", "").rpartition(":")
    return f"tcp://{'127.0.0.1' if host in ('localhost', '0.0.0.0', '*', '127.0.0.1') else host}:{port}"


def wire_protocol(p: str) -> str:
    """Sensor yaml transport.protocol (iphumi_socketio_v1|v2) -> the senders' --protocol v1|v2."""
    p = str(p)
    if p.endswith("v1"):
        return "v1"
    if p.endswith("v2"):
        return "v2"
    raise ValueError(f"unknown transport.protocol {p!r}")


def build_bridge_cfg(cfg: dict, station=None) -> tuple[dict, list]:
    """Merged bridge.yaml (precedence: bridge.yaml < station fragment (+ station home) < deploy.yaml bridge.override)
    and the override leaves that contradict the station. Pure (reads the yaml files only)."""
    bridge = yaml.safe_load(open(resolve_pkg(cfg["bridge"]["config"])))
    overrides = []
    if station is not None:
        frag = station.bridge_fragment()
        home = cfg.get("_station_home") or {}
        btype = deep_merge(bridge, cfg["bridge"].get("override", {})).get("backend", {}).get("type")
        if home.get("home_O_T_EE_xyz_wxyz") and btype in ("fake", "mujoco"):   # the sim backends start at the home
            frag["home_O_T_EE_xyz_wxyz"] = home["home_O_T_EE_xyz_wxyz"]
        overrides += station_contradictions("bridge", frag, cfg["bridge"].get("override", {}))
        bridge = deep_merge(bridge, frag)
    bridge = deep_merge(bridge, cfg["bridge"].get("override", {}))
    bridge["endpoint"] = cfg["bridge"]["endpoint"]
    return bridge, overrides


def build_controller_cfg(cfg: dict, station=None) -> tuple[dict, list]:
    """Merged franka_cartesian.yaml (precedence: file < station fragment < deploy.yaml controller.override). Pure."""
    ctrl = yaml.safe_load(open(resolve_pkg(cfg["controller"]["config"])))
    overrides = []
    if station is not None:
        frag = station.controller_fragment()
        overrides += station_contradictions("controller", frag, cfg["controller"].get("override", {}))
        ctrl = deep_merge(ctrl, frag)
    ctrl = deep_merge(ctrl, cfg["controller"].get("override", {}))
    ctrl["bridge_endpoint"] = cfg["bridge"]["endpoint"]
    ctrl["controller"]["endpoint"] = cfg["controller"]["endpoint"]
    if cfg["home"].get("pose"):
        ctrl["home_tcp_xyz_wxyz"] = list(cfg["home"]["pose"])
    return ctrl, overrides


def resolve_pkg(p) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else PKG / p


def _quat_wxyz_to_R(q):
    import numpy as np

    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _R_to_quat_wxyz(R):
    import numpy as np

    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q) / np.linalg.norm(q)
    lead = q[0] if abs(q[0]) > 1e-9 else q[int(np.argmax(np.abs(q)))]   # canonical sign (w >= 0, else largest > 0)
    return [float(x) + 0.0 for x in (q if lead >= 0 else -q)]


def station_home(station, F_T_TCP) -> dict | None:
    """Home from the station: TCP position robot.home_tcp.xyz_m, TCP orientation = the EE orientation of
    robot.bridge.home_O_T_EE_xyz_wxyz (EE z down) composed with the controller's F_T_TCP rotation, and the matching
    EE start pose home_O_T_EE = T_home_TCP @ inv(F_T_TCP) for the sim backends (FakeBackend / MuJoCo), so a station
    profile starts at the station home (B2 open question, B6 carry-over). None without robot.home_tcp."""
    import numpy as np

    ht = station.get("robot.home_tcp", None)
    if not ht:
        return None
    ee = station.get("robot.bridge.home_O_T_EE_xyz_wxyz", [0.45, 0.0, 0.45, 0.0, 1.0, 0.0, 0.0])
    F = np.asarray(F_T_TCP, float)
    T = np.eye(4)
    T[:3, :3] = _quat_wxyz_to_R(ee[3:]) @ F[:3, :3]
    T[:3, 3] = np.asarray(ht["xyz_m"], float)
    E = T @ np.linalg.inv(F)
    r = lambda v: [round(float(x), 6) + 0.0 for x in v]  # noqa: E731
    return {"xyz": r(T[:3, 3]), "max_dist_m": float(ht["max_dist_m"]),
            "pose": r(list(T[:3, 3]) + _R_to_quat_wxyz(T[:3, :3])),
            "home_O_T_EE_xyz_wxyz": r(list(E[:3, 3]) + _R_to_quat_wxyz(E[:3, :3]))}


def apply_station_home(cfg: dict, station) -> dict | None:
    """With a station, cfg.home comes from it (xyz + max_dist always; the full pose only for the sim backends: on
    hardware `r` would move the arm to a pose derived from placeholder values, so deploy.yaml home.pose stays).
    Stores the derived values in cfg["_station_home"] (used by build_bridge_cfg). Returns them."""
    if station is None:
        return None
    ctrl, _ = build_controller_cfg(cfg, station)
    home = station_home(station, ctrl["F_T_TCP"])
    if home is None:
        return None
    btype = deep_merge(yaml.safe_load(open(resolve_pkg(cfg["bridge"]["config"]))),
                       cfg["bridge"].get("override", {})).get("backend", {}).get("type")
    cfg["home"] = {**cfg.get("home", {}), "xyz": home["xyz"], "max_dist_m": home["max_dist_m"]}
    if btype in ("fake", "mujoco"):
        cfg["home"]["pose"] = home["pose"]
    cfg["_station_home"] = home
    return home


def resolve_replay_episode(sim: dict) -> Path:
    """The raw iPhone episode of run.sim video_replay: `episode_dir` (a dir name under raw_root, or an absolute path)
    wins; otherwise `episode_index` into the sorted firstreal demonstration episodes of raw_root (default
    data/raw/2026-10-01, relative to the repo root) that have their _rgb.mp4."""
    root = Path(str(sim.get("raw_root") or DEFAULT_RAW_ROOT))
    root = root if root.is_absolute() else REPO_ROOT / root
    d = sim.get("episode_dir")
    if d:
        p = Path(str(d))
        return p if p.is_absolute() else root / p
    eps = sorted(p for p in root.iterdir() if "firstreal_demonstration" in p.name
                 and (p / (p.name + "_rgb.mp4")).is_file()) if root.is_dir() else []
    i = int(sim.get("episode_index", 0))
    if not 0 <= i < len(eps):
        raise FileNotFoundError(f"run.sim episode_index {i}: {len(eps)} firstreal demonstration episode(s) in {root}")
    return eps[i]


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
        self.station_home = apply_station_home(cfg, self.station)   # station profiles start at the station home
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

    def env(self, child: str | None = None):
        """Child environment. PYTHONPATH = franka_deploy (+ franka_sim when the dir exists). User site-packages are
        off, except for the children listed in deploy.yaml `user_site_children` (e.g. [camera, sender] on the laptop,
        where flask lives in ~/.local)."""
        e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        cache = self.work_root / "cache"
        pp = os.pathsep.join([str(PKG)] + ([str(SIM_ROOT)] if SIM_ROOT.is_dir() else []))
        e.update({"PYTHONPATH": pp, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                  "TRANSFORMERS_OFFLINE": "1", "XDG_CACHE_HOME": str(cache / "xdg"), "MPLCONFIGDIR": str(cache / "mpl"),
                  "NUMBA_CACHE_DIR": str(cache / "numba"), "HF_HOME": str(cache / "hf"), "TORCH_HOME": str(cache / "torch"),
                  "LIBERO_CONFIG_PATH": str(cache / "libero_unused"), "PIP_NO_CACHE_DIR": "1"})
        extra = [str(p) for p in self.cfg.get("extra_path", [])]
        if extra:
            e["PATH"] = os.pathsep.join(extra + [e.get("PATH", "")])
        if child is not None and child in (self.cfg.get("user_site_children") or []):
            e.pop("PYTHONNOUSERSITE", None)
        return e

    def start(self, name, cmd):
        self.children[name] = Child(name, cmd, self.session / "logs" / f"{name}.log", self.env(name), self.session)
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
        st = self.station
        # precedence: bridge.yaml < station fragment < deploy.yaml bridge.override (the same for the controller)
        bridge, overrides = build_bridge_cfg(c, st)
        (cfgd / "bridge.yaml").write_text(yaml.safe_dump(bridge))
        ctrl, o2 = build_controller_cfg(c, st)
        overrides += o2
        self.bridge_cfg, self.ctrl_cfg = bridge, ctrl
        (cfgd / "franka_cartesian.yaml").write_text(yaml.safe_dump(ctrl))
        (cfgd / "deploy_resolved.yaml").write_text(yaml.safe_dump(c))
        if st is not None:
            (cfgd / "station_resolved.yaml").write_text(st.to_yaml())
            if self.station_home:
                self.summary["station_home"] = self.station_home
            if camera_kind(c) == "wireless":   # the receiver config IS the sensor group (station.camera_config)
                (cfgd / "camera.yaml").write_text(yaml.safe_dump(st.camera_config(c["camera"]["endpoint"],
                                                                                  c["camera"]["logger_endpoint"])))
            if run_mode_of(c) == "sim" and "sensor" in st.groups:
                (cfgd / "sender.yaml").write_text(yaml.safe_dump(st.sender_config()))
                if "link" in st.groups:
                    (cfgd / "link.yaml").write_text(yaml.safe_dump(st.get("link")))
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

    # ------------------------------------------------------------------ wireless / sim helpers
    def _bridge(self) -> dict:
        if getattr(self, "bridge_cfg", None) is None:
            self.bridge_cfg = build_bridge_cfg(self.cfg, self.station)[0]
        return self.bridge_cfg

    def _ctrl(self) -> dict:
        if getattr(self, "ctrl_cfg", None) is None:
            self.ctrl_cfg = build_controller_cfg(self.cfg, self.station)[0]
        return self.ctrl_cfg

    def backend_type(self) -> str | None:
        return self._bridge().get("backend", {}).get("type")

    def hand_endpoint(self) -> str | None:
        """The hand_server endpoint when a `hand` child runs: gripper franka_hand with an endpoint other than the
        bridge's (real hardware: 5566). In sim the bridge serves the hand topics itself (endpoint = bridge)."""
        g = self._ctrl().get("gripper") or {}
        if g.get("type") != "franka_hand" or not g.get("endpoint"):
            return None
        return None if norm_endpoint(g["endpoint"]) == norm_endpoint(self.cfg["bridge"]["endpoint"]) else g["endpoint"]

    def image_source(self) -> str | None:
        """run.sim.image_source in sim profiles with a station (the sender slot), else None (no sender)."""
        if self.station is None or run_mode_of(self.cfg) != "sim" or camera_kind(self.cfg) != "wireless":
            return None
        return self.station.get("sim.image_source", None)

    def receiver_endpoints(self) -> list[str]:
        if camera_kind(self.cfg) != "wireless" or self.station is None:
            return []
        tr = self.station.get("sensor.transport")
        streams = ["main", "ultrawide"] if wire_protocol(tr["protocol"]) == "v1" else ["main"]
        return [f"tcp://{tr['bind_host']}:{int(tr['ports'][s])}" for s in streams]

    def endpoints(self) -> list[str]:
        """Every port this session binds (preflight ports_free and the teardown check). Profiles without a station,
        wireless camera, MuJoCo backend or hand child get exactly the pre-B6 list."""
        c = self.cfg
        eps = [c["bridge"]["endpoint"], c["controller"]["endpoint"], c["camera"]["endpoint"], c["inference"]["endpoint"],
               c["spacemouse"]["endpoint"], c["camera"]["logger_endpoint"]]
        eps += self.receiver_endpoints()
        if self.backend_type() == "mujoco":
            eps.append(c.get("sim_state_endpoint", SIM_STATE_ENDPOINT))
        if self.hand_endpoint():
            eps.append(self.hand_endpoint())
        return eps

    def stats_path(self) -> Path:
        return self.session / "logs" / "camera_stats.json"

    def sender_command(self) -> list[str]:
        """The sim sender child (imitates the sensor over the run's emulated link), by run.sim.image_source."""
        src, st, c = self.image_source(), self.station, self.cfg
        if src not in SENDERS:
            raise Abort(f"run.sim.image_source {src!r}: no sender (known: {', '.join(SENDERS)})")
        mod, path, phase = SENDERS[src]
        if not path.is_file():
            raise Abort(f"sender for image_source {src} ({mod}) not available yet ({phase})")
        py, cfgd, logs = c["python"]["realenv"], self.session / "config", self.session / "logs"
        tr, main = st.get("sensor.transport"), st.get("sensor.streams.main")
        proto = wire_protocol(tr["protocol"])
        ports = ",".join(str(int(tr["ports"][s])) for s in ("main", "ultrawide") if s in tr["ports"])   # v2 uses main
        common = ["--host", "127.0.0.1", "--ports", ports, "--protocol", proto, "--fps", f"{float(main['physical_fps_hz']):g}"]
        if (cfgd / "link.yaml").exists():
            common += ["--link", str(cfgd / "link.yaml")]
        common += ["--truth", str(logs / "sender_truth.jsonl"), "--ready-file", str(self.session / "sender_ready.json"),
                   "--loop"]   # always: the sender is the device and must outlive every operator episode
        if src == "video_replay":
            ep = resolve_replay_episode(st.get("sim"))
            q = main.get("jpeg_quality", 0.9)
            return [py, "-u", "-m", mod, "--episode", str(ep), *common, "--size", f"{main['width_px']}x{main['height_px']}",
                    "--jpeg-quality", str(int(round(100 * q)) if q <= 1 else int(q))]
        extra = ["--episode", str(st.get("sim.raw_episode_dir"))] if src == "rig_raw_replay" else []
        return [py, "-u", "-m", mod, "--config", str(cfgd / "sender.yaml"), *extra, *common]

    # ------------------------------------------------------------------ phases
    def static_preflight(self):
        c = self.cfg
        self.check("dirs_outside_repo", pf.dirs_outside_repo(self.work_root, self.session))
        self.check("disk_free", pf.disk_free(self.work_root, c.get("disk_budget_gb", 2.0)))
        self.check("shm_clean", pf.shm_clean())
        self.check("ports_free", pf.ports_free(self.endpoints()))
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
        kind = camera_kind(c)
        if kind not in CAMERA_KINDS:
            self.check("camera_kind", (False, f"camera.kind {kind!r} (known: {', '.join(CAMERA_KINDS)})"))
        if kind == "rig_local":
            self.check("rig_not_default", pf.rig_not_default(self.rig_path()))
        self.check("agent_hz", pf.agent_hz_ok(c["task"].get("agent_hz", 20)))
        if self.station is not None:
            self.check("station_valid", (True, f"{self.station.ids} fingerprint {self.station.fingerprint()}"))
            self.check("station_placeholders", pf.station_placeholders(self.station, run_mode_of(c),
                                                                       c.get("allow_placeholders", [])))
            self.check("tooling_sensor_compat", pf.tooling_sensor_compat(self.station))
        if kind == "wireless":
            if self.station is None or "sensor" not in self.station.groups:
                self.check("camera_kind", (False, "camera.kind wireless needs a station with a sensor group"))
            self.check("websocket_available", pf.websocket_available(c["python"]["realenv"], self.env("camera")))
        src = self.image_source()
        if src is not None:
            mod, path, phase = SENDERS.get(src, (None, None, None))
            self.check("sender_available", (bool(path and path.is_file()),
                                            f"{src}: {mod} " + ("present" if path and path.is_file()
                                                                else f"not available yet ({phase})")))
        if self.hand_endpoint():
            self.check("hand_server_available", (HAND_SERVER.is_file(), f"{HAND_SERVER.name} "
                                                 + ("present" if HAND_SERVER.is_file() else "missing (A5)")))
        if self.backend_type() == "mujoco":
            self.check("mujoco_backend_available", (MUJOCO_BACKEND.is_file(), f"franka_sim {'present' if SIM_ROOT.is_dir() else 'missing'}"
                                                    f" / MuJoCoBackend {'present' if MUJOCO_BACKEND.is_file() else 'missing (A4)'}"))

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
        if c["bridge"].get("allow_sim_reset"):
            bcmd += self.allow_sim_reset_flag()
        self.start("bridge", bcmd)
        from franka_deploy.bridge.link import BridgeClient

        bc = BridgeClient(c["bridge"]["endpoint"].replace("0.0.0.0", "127.0.0.1"), timeout_s=0.3)
        st = self.wait_until("bridge", lambda: bc.request("get_state"), 15, "get_state")
        self.check("bridge_not_reflex", pf.bridge_not_reflex(st))

        hand_ep = self.hand_endpoint()
        if hand_ep:   # real Franka Hand: libfranka Gripper in its own process (A5), before the controller uses it
            hcmd = [py, "-u", "-m", "franka_deploy.gripper.hand_server", "--endpoint", hand_ep, "--backend",
                    "libfranka" if self.backend_type() == "panda_py" else "fake"]
            host = (self._bridge().get("backend", {}).get("panda_py") or {}).get("hostname")
            if self.station is not None:
                host = self.station.get("robot.fci.hostname", host)
            if host:
                hcmd += ["--hostname", str(host)]
            self.start("hand", hcmd)
            hc = BridgeClient(norm_endpoint(hand_ep), timeout_s=0.3)
            self.wait_until("hand", lambda: hc.request("hand_state") is not None, 20, "hand_state")

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
        wireless = camera_kind(c) == "wireless"
        if wireless:   # B5 receiver: camera.yaml = station.camera_config() (written by snapshot)
            ccmd = [py, "-u", "-m", "franka_deploy.peripherals.wireless_camera", "--config",
                    str(self.session / "config" / "camera.yaml"), "--stats-path", str(self.stats_path())]
            if run_mode_of(c) == "sim":
                ccmd += ["--truth-path", str(logs / "camera_truth.jsonl")]
        else:
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
        if self.image_source() is not None:   # sim: the sender imitates the sensor (after the receiver is up)
            self.start("sender", self.sender_command())
            sready = self.session / "sender_ready.json"
            self.wait_until("sender", lambda: sready.exists(), float(cam.get("sender_ready_timeout_s", 60)), "sender ready file")
        self.wait_until("camera", lambda: cc.get_latest_images_dict_THWC(4, extra=0),
                        float(cam.get("frames_timeout_s", 15 if not wireless or self.image_source() else 60)), "frames")
        self.check("camera_rate_and_selection", pf.camera_ok(cc))
        if wireless:
            self.wireless_checks()

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
                              ("--camera-latency-s", "camera_latency_s"), ("--policy-latency-s", "policy_latency_s"),
                              ("--max-image-age-s", "max_image_age_s")):
                if key in ta:
                    tcmd += [flag, str(ta[key])]
        if cam.get("video", False):
            tcmd += ["--video-logger-endpoint", cam["logger_endpoint"]]
        self.start("task", tcmd)
        self.wait_until("task", lambda: tready.exists(), 60, "task ready file")

    def wireless_checks(self):
        """After the camera is up: link health (websocket, fps, age, reconnects, sizes), v2 clock sync and hello."""
        st = self.station
        tr, streams, health = st.get("sensor.transport"), st.get("sensor.streams"), st.get("sensor.health")
        proto = wire_protocol(tr["protocol"])
        socket_streams = ["main", "ultrawide"] if proto == "v1" else ["main"]
        self.check("wireless_health_ok", pf.wireless_health_ok(
            self.stats_path(), {s: float(v["physical_fps_hz"]) for s, v in streams.items()}, float(health["max_age_s"]),
            socket_streams, min_fps_ratio=float(health.get("min_fps_ratio", 0.9)),
            stats_period_s=float(health.get("stats_period_s", 1.0))))
        if proto == "v2":
            self.check("clock_ok", pf.clock_ok(self.stats_path()))
            self.check("hello_problems", pf.hello_check((pf.read_stats(self.stats_path()) or {}).get("hello_problems"),
                                                        run_mode_of(self.cfg)))

    def allow_sim_reset_flag(self) -> list[str]:
        """["--allow-sim-reset"] when franka_bridge --help lists it (A6 adds the bridge side), else [] + a debug log."""
        try:
            r = subprocess.run([self.cfg["python"]["realenv"], "-m", "franka_deploy.bridge.franka_bridge", "--help"],
                               capture_output=True, text=True, timeout=60, env=self.env("bridge"))
            if "--allow-sim-reset" in r.stdout:
                return ["--allow-sim-reset"]
        except Exception as e:
            self.log("debug", msg=f"franka_bridge --help failed: {e!r}")
            return []
        self.log("debug", msg="bridge.allow_sim_reset set but franka_bridge has no --allow-sim-reset yet (A6): skipped")
        return []

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
        for item in split_script(self.script):
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
            elif act == "sim_reset" or act.startswith("sim_reset:"):
                self.sim_reset(act.partition(":")[2])
            else:
                self.forward(act)

    def sim_reset(self, text: str = ""):
        """`--script` action sim_reset[:<json>] -> bridge topic sim_reset (A6; refused unless --allow-sim-reset)."""
        from franka_deploy.bridge.link import BridgeClient

        try:
            data = json.loads(text) if text.strip() else {}
            rep = BridgeClient(self.cfg["bridge"]["endpoint"].replace("0.0.0.0", "127.0.0.1"), 0.5).request("sim_reset", data)
            self.log("sim_reset", data=data, reply=rep)
        except Exception as e:
            self.log("sim_reset_failed", data=text, error=repr(e))

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
                    if name == "sender":   # the sim device dropped out: like a phone leaving Wi-Fi, not fatal. The
                        ch.expected_exit = True   # receiver goes stale and the task's image-age guard holds; HOLD now too
                        self.log("sender_died", returncode=rc, t_mono=time.monotonic())
                        self.summary["sender_died"] = {"returncode": rc, "t": round(time.monotonic() - self.t0, 3)}
                        self.hold_now("sender died")
                        continue
                    self.log("child_died", child=name, returncode=rc, t_mono=time.monotonic())
                    self.hold_now(f"{name} died")
                    self.exit_code, self.exit_reason = 1, f"{name} died (rc {rc})"
                    return
            time.sleep(0.05)

    def teardown(self):
        quick = self.interrupted
        budget = 0.3 if quick else 5.0
        codes = {}
        if camera_kind(self.cfg) == "wireless" and self.session.exists():   # link health as the session ran
            self.summary["camera_stats"] = pf.read_stats(self.stats_path())
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
        try:
            eps = self.endpoints()
        except Exception:   # e.g. aborted before the configs could be read: the always-present ports
            c = self.cfg
            eps = [c["bridge"]["endpoint"], c["controller"]["endpoint"], c["camera"]["endpoint"],
                   c["inference"]["endpoint"], c["spacemouse"]["endpoint"], c["camera"]["logger_endpoint"]]
        self.summary["ports_free_after"] = pf.ports_free(eps)
        if camera_kind(self.cfg) == "wireless":   # after the sender stopped (stale ticks grow, fps 0)
            self.summary["camera_stats_final"] = pf.read_stats(self.stats_path())
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


def _import_check(python: str, module: str) -> tuple[bool, str]:
    if not Path(python).exists():
        return False, f"python {python} not found"
    r = subprocess.run([python, "-c", f"import {module}; print(getattr({module}, '__version__', 'ok'))"],
                       capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    return r.returncode == 0, (f"{module} {r.stdout.strip()} (python.realenv)" if r.returncode == 0
                               else f"import {module} failed: {(r.stderr.strip().splitlines() or ['?'])[-1]}")


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
    home = apply_station_home(cfg, station)
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
    r.append(("bridge_backend", (btype in ("fake", "panda_py", "mujoco"), str(btype))))
    if btype == "mujoco":   # A4: franka_sim.backend.MuJoCoBackend, mujoco importable in python.realenv
        r.append(("mujoco_import", _import_check(cfg["python"]["realenv"], "mujoco")))
        r.append(("mujoco_backend", (SIM_ROOT.is_dir() and MUJOCO_BACKEND.is_file(),
                                     f"franka_sim {'present' if SIM_ROOT.is_dir() else 'missing'} / MuJoCoBackend "
                                     f"(franka_sim/backend.py) {'present' if MUJOCO_BACKEND.is_file() else 'missing (A4)'}")))
        if not cfg["bridge"].get("allow_sim_reset"):
            notes.append("bridge.allow_sim_reset is off: sim_reset script actions will be refused")
    ctrl, _ = build_controller_cfg(cfg, station)
    gtype = (ctrl.get("gripper") or {}).get("type")
    if station is not None or gtype == "franka_hand":   # (no new output line for the legacy profiles)
        r.append(("controller_gripper", (gtype in ("fake", "feetech", "dynamixel", "franka_hand"), str(gtype))))
    if gtype == "franka_hand":
        drv = (PKG / "franka_deploy" / "gripper" / "driver.py").read_text()
        r.append(("franka_hand_driver", ("franka_hand" in drv, "controller gripper driver franka_hand "
                                         + ("present" if "franka_hand" in drv else "missing (A5)"))))
        gep = ctrl["gripper"].get("endpoint")
        if gep and norm_endpoint(gep) != norm_endpoint(cfg["bridge"]["endpoint"]):
            r.append(("hand_server", (HAND_SERVER.is_file(), f"hand child on {gep}: franka_deploy.gripper.hand_server "
                                      + ("present" if HAND_SERVER.is_file() else "missing (A5)"))))
        else:
            notes.append(f"hand: endpoint {gep} = the bridge: the bridge backend serves hand_cmd/hand_state (A5), no hand child")
    if btype == "panda_py":
        pp = bcfg["backend"].get("panda_py", {})
        r.append(("panda_py_hostname", (bool(pp.get("hostname")), str(pp.get("hostname")))))
        r.append(("panda_py_versions_pinned", (bool(bcfg["backend"].get("panda_py_version") and bcfg["backend"].get("libfranka_version")),
                                               f"{bcfg['backend'].get('panda_py_version')} / {bcfg['backend'].get('libfranka_version')}")))
        notes.append("panda-py import / FCI connection: hardware-only (not checked)")
    cam = cfg["camera"]
    kind = camera_kind(cfg)
    if kind == "rig_local":
        r.append(("camera_config", (cam.get("config") in ("wrist_right", "wrist_mock"), str(cam.get("config")))))
        if cam.get("config") == "wrist_right":
            notes.append("Argus cameras: hardware-only (not checked)")
            rp = cam.get("rig_config_path")
            r.append(("rig_not_default", pf.rig_not_default(rp) if rp and Path(str(rp)).exists()
                      else (False, f"rig config {rp} missing or default")))
    elif kind == "wireless":
        has_sensor = station is not None and "sensor" in station.groups
        r.append(("camera_kind", (has_sensor, "wireless (B5 receiver, camera.yaml from station.camera_config)"
                                  if has_sensor else "wireless needs a station with a sensor group")))
        if Path(cfg["python"]["realenv"]).exists():
            env = {**os.environ, "PYTHONNOUSERSITE": "1"}
            if "camera" in (cfg.get("user_site_children") or []):
                env.pop("PYTHONNOUSERSITE")
            r.append(("websocket_available", pf.websocket_available(cfg["python"]["realenv"], env)))
        if has_sensor:
            tr = station.get("sensor.transport")
            notes.append(f"wireless: {tr['protocol']} on {tr['bind_host']}:{dict(tr['ports'])}, clock "
                         f"{station.get('sensor.clock.mode')}, max_image_age_s {station.get('sensor.task.max_image_age_s')}")
            if run_mode_of(cfg) == "sim":
                src = station.get("sim.image_source", None)
                mod, path, phase = SENDERS.get(src, (None, None, None))
                ok = bool(path and path.is_file())
                detail = f"{src}: {mod} " + ("present" if ok else f"not available yet ({phase})") if mod else \
                    f"run.sim.image_source {src!r}: no sender slot (known: {', '.join(SENDERS)})"
                if ok and src == "video_replay":
                    try:
                        detail += f", episode {resolve_replay_episode(station.get('sim')).name}"
                    except (OSError, ValueError) as e:
                        ok, detail = False, f"{src}: {e}"
                r.append(("sender", (ok, detail)))
            else:
                notes.append("real mode: no sender; the device (iPhone app / rig) streams to the receiver ports")
    else:
        r.append(("camera_kind", (False, f"camera.kind {kind!r} (known: {', '.join(CAMERA_KINDS)})")))
    if home:
        notes.append(f"home from the station: TCP xyz {home['xyz']} (max {home['max_dist_m']} m)"
                     + (f", sim start home_O_T_EE {home['home_O_T_EE_xyz_wxyz']}" if btype in ("fake", "mujoco") else ""))
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
