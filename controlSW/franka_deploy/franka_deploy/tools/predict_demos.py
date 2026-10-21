"""Offline policy predictions on recorded demos (Stage 2a of the sim evaluator), for ANY policy kind (env_bp side).

    python -m franka_deploy.tools.predict_demos --policy POLICY.yaml --zarr Z (--demos 0,3 | --split val|all [--limit N])
        [--every 8] [--queries Q.json] [--seed 0] [--work-dir W] [--device cuda] --out pred.npz

POLICY.yaml = the resolved ``run/policy`` group (id, kind, python, checkpoint, prompt_zarr, prompt_index, cmd,
episode_index, ready_timeout_s, contract.{K,exec_h,uses_prompt}); JSON works too.

Kinds:
  bp_server / command  a policy server process is started (bp_server: franka_deploy.inference.server; command: the
                       ``cmd`` template with {python} {endpoint} {ready_file} {work_dir} {checkpoint} {prompt_zarr}) and
                       queried over the robotmq protocol exactly as the deployment task does: policy_config,
                       policy_reset, policy_prompt, policy_inference with deployment-format observations. This makes
                       the stage policy-agnostic: anything that serves the I-POLICY topics can be evaluated.
  replay               no server: the demo's own future, ``action[k] = inv(P(f)) P(f + 3k)`` (replay_server convention,
                       ``ReplayServer.chunk``), error zero by construction.
  fake                 no server: the identity chunk (hold), constant width 0.07 (``mocks.fake_policy_server``).

Observations (what ``real_env`` PolicyAgent.predict_actions sends, built from the zarr instead of the robot; same
keys, shapes and dtypes as ``tools.analyze_tap.CONTRACT_OBS``): proprio history = the demo TCP pose / width at frames
f - 3*(h-1-i) (h = proprio_length + 1; clipped at the demo start), pos3 + rot6d relative to the last (identity), width,
rot6d relative to the demo pose at ``start_frame`` (the model's *_wrt_start input), and the 224x224 zarr images the
training sampler would use for frame f (main: frames f-3, f; ultrawide: the upsample_index frame uc and uc-1, clipped
at the episode start). ``start_frame`` defaults to 0 (training: the episode start); Stage 2b passes the live start.
Frames may be fractional: poses/widths are interpolated (live teacher forcing interpolates too), images use round(f).

Queries: default for each demo, frames 0, S, 2S, ... while f + 3*(K-1) <= N-1 with S = 3 * ``--every`` (agent steps at
20 Hz = 3 demo frames at 60 Hz; ``--every`` defaults to the policy's contract.exec_h). ``--queries Q.json`` gives an
ordered list ``[{"demo": 3, "frame": 41.5, "start_frame": 0.0}, ...]`` instead (Stage 2b: the frames of the live calls,
in call order, so a server started with the live ``--noise-seed-base`` draws the same diffusion noise per call).

Output pred.npz: demo (n,) int, frame (n,) float, start_frame (n,), rel_chunks (n,K,10) = pos3 + rot6d + width of the
policy, gt_chunks (n,K,10) = the demo's own future in the same convention, gt_valid (n,) bool (the future lies inside
the demo), infer_s (n,), noise_seed (n,) (-1 when the policy gives none), meta (JSON string: policy id/kind, demo
names, zarr, seed, split, ...).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

STRIDE = 3          # demo frames (60 Hz) per agent step (20 Hz)
K_DEFAULT = 16
PKG = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------------------------ zarr access
class DemoZarr:
    """Low-dim arrays in memory, images lazily. Needs zarr + the imagecodecs codecs (env_bp) for images only."""

    def __init__(self, path: str, images: bool = True):
        import zarr

        self.path = str(path)
        if images:
            try:
                from behavior_prompting.common.imagecodecs_numcodecs import register_codecs

                register_codecs(verbose=False)
            except ImportError:  # pragma: no cover
                pass
        self.root = zarr.open(zarr.ZipStore(self.path, mode="r"), mode="r")
        self.names = [str(x) for x in self.root["meta/episode_names"][:]]
        self.ends = np.asarray(self.root["meta/episode_ends"][:], dtype=np.int64)
        pos = self.root["data/gripper_right_eef_pos"][:].astype(float)
        rv = self.root["data/gripper_right_eef_rot_axis_angle"][:].astype(float)
        self.width_all = self.root["data/gripper_right_gripper_width"][:, 0].astype(float)
        self.R_all = Rotation.from_rotvec(rv)
        self.pos_all = pos
        self.has_uw = "data/camera_right_ultrawide_rgb" in self.root
        self._uw_ends = np.asarray(self.root["meta/episode_ends_camera_right_ultrawide_rgb"][:], dtype=np.int64) if self.has_uw else None
        self._up = self.root["meta/upsample_index_camera_right_ultrawide_rgb"][:].astype(np.int64) if self.has_uw else None

    def start(self, d: int) -> int:
        return 0 if d == 0 else int(self.ends[d - 1])

    def length(self, d: int) -> int:
        return int(self.ends[d]) - self.start(d)

    def demo(self, d: int) -> dict:
        s, e = self.start(d), int(self.ends[d])
        T = np.tile(np.eye(4), (e - s, 1, 1))
        T[:, :3, :3] = self.R_all[s:e].as_matrix()
        T[:, :3, 3] = self.pos_all[s:e]
        return {"idx": d, "name": self.names[d], "T": T, "w": self.width_all[s:e], "s": s}

    def main_image(self, d: int, f: int) -> np.ndarray:
        return np.asarray(self.root["data/camera_right_main_rgb"][self.start(d) + int(f)])

    def uw_index(self, d: int, f: int) -> tuple[int, int]:
        """(global uw frame of main frame f, global index of the episode's first uw frame)."""
        us = 0 if d == 0 else int(self._uw_ends[d - 1])
        return int(self._up[self.start(d) + int(f)]), us

    def uw_image(self, g: int) -> np.ndarray:
        return np.asarray(self.root["data/camera_right_ultrawide_rgb"][int(g)])


def interp_pose(T: np.ndarray, f: float) -> np.ndarray:
    f = float(np.clip(f, 0, len(T) - 1))
    i = int(np.floor(f))
    j = min(i + 1, len(T) - 1)
    a = f - i
    out = np.eye(4)
    out[:3, 3] = (1 - a) * T[i, :3, 3] + a * T[j, :3, 3]
    if a == 0 or i == j:
        out[:3, :3] = T[i, :3, :3]
    else:
        out[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([T[i, :3, :3], T[j, :3, :3]])))(a).as_matrix()
    return out


def interp_w(w: np.ndarray, f: float) -> float:
    return float(np.interp(float(np.clip(f, 0, len(w) - 1)), np.arange(len(w)), w))


def rot6d_of(R: np.ndarray) -> np.ndarray:
    """First two ROWS of R, flattened (training / robot_utils convention)."""
    return np.asarray(R)[:2].reshape(-1)


def gt_chunk(T: np.ndarray, w: np.ndarray, f: float, K: int = K_DEFAULT) -> tuple[np.ndarray, bool]:
    """(K,10) demo future relative to the pose at f (action 0 = identity), width absolute; valid = inside the demo."""
    inv0 = np.linalg.inv(interp_pose(T, f))
    out = np.zeros((K, 10))
    for k in range(K):
        fk = f + STRIDE * k
        rel = inv0 @ interp_pose(T, fk)
        out[k, :3], out[k, 3:9], out[k, 9] = rel[:3, 3], rot6d_of(rel[:3, :3]), interp_w(w, fk)
    return out, bool(f + STRIDE * (K - 1) <= len(T) - 1 + 1e-9)


def deploy_obs(z: DemoZarr, dem: dict, f: float, start_frame: float, ws: dict, task_name: str = "block_raise") -> dict:
    """The observation dict a PolicyAgent sends for demo ``dem`` at (fractional) frame ``f`` (see the module docstring)."""
    T, w = dem["T"], dem["w"]
    hp = int(ws.get("proprio_length", 1)) + 1
    himg = int(ws.get("img_obs_horizon", 2))
    ds = int(ws.get("obs_down_sample_steps", STRIDE))
    uds = int(ws.get("ultrawide_down_sample_steps", 1))
    fr = [f - ds * (hp - 1 - i) for i in range(hp)]
    poses = [interp_pose(T, x) for x in fr]
    inv_last, inv_start = np.linalg.inv(poses[-1]), np.linalg.inv(interp_pose(T, start_frame))
    rel = [inv_last @ P for P in poses]
    wrt = [inv_start @ P for P in poses]
    obs = {
        "robot0_eef_pos": np.array([r[:3, 3] for r in rel], np.float32),
        "robot0_eef_rot_axis_angle": np.array([rot6d_of(r[:3, :3]) for r in rel], np.float32),
        "robot0_gripper_width": np.array([[interp_w(w, x)] for x in fr], np.float32),
        "robot0_eef_rot_axis_angle_wrt_start": np.array([rot6d_of(r[:3, :3]) for r in wrt], np.float32),
        "episode_idx": np.array(0),
        "task_name": task_name,
    }
    fi = int(round(float(np.clip(f, 0, len(T) - 1))))
    mi = [max(0, fi - ds * (himg - 1 - i)) for i in range(himg)]
    obs["robot0_main_camera"] = np.stack([z.main_image(dem["idx"], m) for m in mi])
    if z.has_uw:
        uc, us = z.uw_index(dem["idx"], fi)
        nv = min(himg, (uc - us) // uds + 1)
        ix = list(range(uc - (nv - 1) * uds, uc + 1, uds))
        ix = [ix[0]] * (himg - len(ix)) + ix
        obs["robot0_ultrawide_camera"] = np.stack([z.uw_image(g) for g in ix])
    return obs


# ------------------------------------------------------------------------------------------------ policy kinds
def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def server_command(pol: dict, endpoint: str, ready: str, work: str, noise_seed_base: int, device: str) -> list[str]:
    py = str(pol.get("python") or sys.executable)
    ckpt, zarr = pol.get("checkpoint"), pol.get("prompt_zarr")
    kind = pol["kind"]
    if kind == "bp_server":
        return [py, "-u", "-m", "franka_deploy.inference.server", "--ckpt", str(ckpt), "--work-dir", work,
                "--dataset-path", str(zarr), "--endpoint", endpoint, "--device", device, "--ready-file", ready,
                "--noise-seed-base", str(int(noise_seed_base))]
    if kind == "command":
        cmd = pol.get("cmd")
        cmd = shlex.split(cmd) if isinstance(cmd, str) else list(cmd or [])
        sub = {"python": py, "endpoint": endpoint, "ready_file": ready, "work_dir": work,
               "checkpoint": str(ckpt or ""), "prompt_zarr": str(zarr or "")}
        return [str(a).format_map(sub) for a in cmd]
    raise ValueError(f"policy kind {kind!r} has no server")


class Server:
    """A policy server child + a robotmq client, like the launcher's `inference` child + the task's PolicyAgent."""

    def __init__(self, pol: dict, work: Path, noise_seed_base: int, device: str):
        self.pol, self.work = pol, Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.endpoint = f"tcp://127.0.0.1:{free_port()}"
        self.ready = self.work / "server_ready.json"
        if self.ready.exists():
            self.ready.unlink()
        cmd = server_command(pol, self.endpoint, str(self.ready), str(self.work / "server"), noise_seed_base, device)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(PKG)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.log = open(self.work / "server.log", "w")
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env, cwd=str(self.work))
        t0, to = time.monotonic(), float(pol.get("ready_timeout_s", 300))
        while not self.ready.exists():
            if self.p.poll() is not None:
                raise RuntimeError(f"policy server died during startup (rc {self.p.returncode}); see {self.work / 'server.log'}")
            if time.monotonic() - t0 > to:
                self.close()
                raise TimeoutError(f"policy server not ready within {to} s; see {self.work / 'server.log'}")
            time.sleep(0.2)
        import robotmq

        self.rmq = robotmq
        self.client = robotmq.RMQClient("predict_demos", self.endpoint)

    def request(self, topic: str, data, timeout_s: float = 60.0):
        rep = self.rmq.deserialize(self.client.request_with_data(topic=topic, data=self.rmq.serialize(data), timeout_s=timeout_s))
        if isinstance(rep, str) and rep.startswith("ERROR"):
            raise RuntimeError(f"{topic}: {rep}")
        return rep

    def split_info(self) -> dict | None:
        p = self.work / "server" / "split_info.json"
        return json.loads(p.read_text()) if p.exists() else None

    def close(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(10)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.log.close()


def default_queries(z: DemoZarr, demos: list[int], every_steps: int, K: int) -> list[dict]:
    out = []
    for d in demos:
        n = z.length(d)
        f = 0
        while f + STRIDE * (K - 1) <= n - 1:
            out.append({"demo": int(d), "frame": float(f), "start_frame": 0.0})
            f += STRIDE * every_steps
    return out


def chunk_from_reply(rep: dict) -> np.ndarray:
    pos = np.asarray(rep["action0_eef_pos"], float)
    rot = np.asarray(rep["action0_eef_rot_axis_angle"], float)
    wid = np.asarray(rep["action0_gripper_width"], float).reshape(len(pos), -1)[:, :1]
    return np.concatenate([pos, rot, wid], axis=1)


def predict(pol: dict, z: DemoZarr, queries: list[dict], work: Path, seed: int = 0, device: str = "cuda", log=print,
            server: "Server | None" = None) -> dict:
    """Predictions for ``queries``. ``server``: an already started Server (the caller closes it), else one is started
    for the server kinds and closed here."""
    kind = pol["kind"]
    K = int(((pol.get("contract") or {}).get("K")) or K_DEFAULT)
    cache: dict[int, dict] = {}
    dem = lambda d: cache.setdefault(int(d), z.demo(int(d)))  # noqa: E731
    n = len(queries)
    rel = np.zeros((n, K, 10))
    gt = np.zeros((n, K, 10))
    valid = np.zeros(n, bool)
    infer = np.zeros(n)
    seeds = -np.ones(n, np.int64)
    for i, q in enumerate(queries):
        dm = dem(q["demo"])
        gt[i], valid[i] = gt_chunk(dm["T"], dm["w"], float(q["frame"]), K)
    srv = None
    try:
        if kind in ("bp_server", "command"):
            srv = server or Server(pol, work, seed, device)
            cfg = srv.request("policy_config", True, 30.0)
            ws = cfg.get("workspace", {})
            use_prompt = bool((pol.get("contract") or {}).get("uses_prompt", cfg.get("use_prompting", False)))
            srv.request("policy_reset", "RESET", 30.0)
            if use_prompt:
                srv.request("policy_prompt", {"task_name": pol.get("task_name", "block_raise"),
                                              "prompt_index": int(pol.get("prompt_index", 0))}, 120.0)
            for i, q in enumerate(queries):
                obs = deploy_obs(z, dem(q["demo"]), float(q["frame"]), float(q.get("start_frame", 0.0)), ws)
                t0 = time.monotonic()
                rep = srv.request("policy_inference", obs, 60.0)
                infer[i] = time.monotonic() - t0
                rel[i] = chunk_from_reply(rep)[:K]
                if "noise_seed" in rep:
                    seeds[i] = int(np.asarray(rep["noise_seed"]))
                if (i + 1) % 20 == 0:
                    log(f"  predicted {i + 1}/{n}")
            info = {"split_info": srv.split_info()}
        elif kind == "replay":
            for i, q in enumerate(queries):
                rel[i] = gt[i]
            info = {}
        elif kind == "fake":
            rel[:, :, 3:9] = rot6d_of(np.eye(3))
            rel[:, :, 9] = 0.07
            info = {}
        else:
            raise ValueError(f"unknown policy kind {kind!r}")
    finally:
        if srv is not None and server is None:
            srv.close()
    return {"rel_chunks": rel, "gt_chunks": gt, "gt_valid": valid, "infer_s": infer, "noise_seed": seeds, "info": info}


def resolve_demos(z: DemoZarr, a, split_info: dict | None) -> list[int]:
    if a.demos:
        return [int(x) for x in a.demos.split(",")]
    idx = list(range(len(z.names)))
    if a.split == "val" and split_info:
        val = [i for i in idx if z.names[i] in split_info and not split_info[z.names[i]]]
        if val:
            idx = val
    return idx[: a.limit] if a.limit else idx


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--demos", default=None)
    ap.add_argument("--split", default="val", choices=["val", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=None, help="agent steps between queries (default contract.exec_h)")
    ap.add_argument("--queries", default=None, help="JSON list of {demo, frame, start_frame}: replaces the default frames")
    ap.add_argument("--seed", type=int, default=0, help="--noise-seed-base of the policy server (call index j draws seed+j)")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pol = yaml.safe_load(Path(a.policy).read_text())
    work = Path(a.work_dir or (Path(a.out).resolve().parent / "predict_work"))
    z = DemoZarr(a.zarr, images=pol["kind"] in ("bp_server", "command"))
    K = int(((pol.get("contract") or {}).get("K")) or K_DEFAULT)
    every = int(a.every or (pol.get("contract") or {}).get("exec_h") or 8)
    split_info = None
    srv = Server(pol, work, a.seed, a.device) if pol["kind"] in ("bp_server", "command") else None
    try:
        if srv is not None:
            split_info = srv.split_info()
        if a.queries:
            queries = json.loads(Path(a.queries).read_text())
            demos = sorted({int(q["demo"]) for q in queries})
        else:
            demos = resolve_demos(z, a, split_info)
            queries = default_queries(z, demos, every, K)
        t0 = time.time()
        res = predict(pol, z, queries, work, a.seed, a.device, server=srv)
    finally:
        if srv is not None:
            srv.close()
    meta = {"policy_id": pol.get("id"), "policy_kind": pol["kind"], "zarr": os.path.abspath(a.zarr), "seed": a.seed,
            "split": "explicit" if a.demos or a.queries else a.split, "demos": demos,
            "demo_names": {str(d): z.names[d] for d in demos}, "every_steps": every, "K": K,
            "n_queries": len(queries), "wall_s": round(time.time() - t0, 1),
            "split_info_used": bool(split_info), "checkpoint": pol.get("checkpoint")}
    np.savez(a.out, demo=np.array([q["demo"] for q in queries], np.int64),
             frame=np.array([q["frame"] for q in queries], float),
             start_frame=np.array([q.get("start_frame", 0.0) for q in queries], float),
             rel_chunks=res["rel_chunks"], gt_chunks=res["gt_chunks"], gt_valid=res["gt_valid"], infer_s=res["infer_s"],
             noise_seed=res["noise_seed"], meta=json.dumps(meta))
    print(f"[predict_demos] {pol['kind']} {pol.get('id')}: {len(queries)} chunks on demos {demos} -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
