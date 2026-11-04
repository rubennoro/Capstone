"""Policy contract check: does a policy server speak the protocol the Franka task expects?

    python -m franka_deploy policy-contract --endpoint tcp://localhost:18765 [--prompt-index 0] [--task-name block_raise]
        [--obs dummy|tap:DIR] [--expect run/policy/<id>.yaml] [--reset-max-s 0.5] [--inference-timeout 5] [--out contract.json]

The client behaves like SafePolicyAgent: one request per call, a per-topic timeout, never resent (rmq_rpc.NoResendClient).
Checks, in order (a pass/fail table goes to stdout, the JSON to --out, exit 0 when nothing failed):
  config_keys                   policy_config has workspace.{proprio_length,img_obs_horizon,obs_down_sample_steps,
                                ultrawide_down_sample_steps} and use_prompting
  reset_latency                 policy_reset answers OK in < reset_max_s (default 0.5), measured 5 times, max reported
  inference_before_prompt       (use_prompting) an inference before any prompt returns an ERROR string. A server that
                                answers anyway does not gate on prompts at all: warning, and the second-prompt check is skipped
  prompt_ok                     a prompt {task_name, prompt_index} returns OK
  second_prompt_rejected        (use_prompting) a second prompt without a reset returns an ERROR string
  inference_ok                  inference with a CONTRACT_OBS-shaped dummy obs (or the first call of a tap directory) replies
  reply_keys / reply_shape      every CONTRACT_REPLY key, with K steps (K = contract.K, 16 unless --expect says otherwise)
  reply_finite                  no NaN / inf
  rot6d_norm / rot6d_orthogonal rows of every rot6d have norm in [0.9, 1.1] and |dot| < 0.1
  width_range                   absolute width in [0, 0.12] m
  action0_identity              (warning only) action 0 within 2 cm and 5 degrees of identity
  export_topic                  export_recorded_data answers (warning when it does not)
With --expect the policy yaml's contract {K, uses_prompt, reset_max_s} must also match (expect_* checks).
CONTRACT_OBS / CONTRACT_REPLY are imported from tools/analyze_tap.py, never copied.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from franka_deploy.tools.analyze_tap import CONTRACT_OBS, CONTRACT_REPLY, rot6d_to_mat

CONFIG_KEYS = ("proprio_length", "img_obs_horizon", "obs_down_sample_steps", "ultrawide_down_sample_steps")
TIMEOUTS = {"policy_config": 5.0, "policy_reset": 5.0, "policy_prompt": 60.0, "policy_inference": 5.0,
            "export_recorded_data": 5.0}
RESET_REPEATS = 5
ROT_NORM = (0.9, 1.1)
ROT_DOT_MAX = 0.1
WIDTH_RANGE = (0.0, 0.12)
IDENT_POS_M, IDENT_ROT_DEG = 0.02, 5.0


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail: str = "", **extra):
        self.checks.append({"name": name, "status": status, "detail": detail, **extra})

    def ok(self, name, good: bool, detail="", **extra):
        self.add(name, "pass" if good else "fail", detail, **extra)

    @property
    def failed(self):
        return [c for c in self.checks if c["status"] == "fail"]

    def table(self) -> str:
        return "\n".join(f"{c['status'].upper():5s} {c['name']:26s} {c['detail']}" for c in self.checks)


def dummy_obs(task_name: str = "block_raise") -> dict:
    """An observation with exactly the CONTRACT_OBS keys, shapes and dtypes (identity pose, black images)."""
    obs = {}
    for k, (shape, dtype) in CONTRACT_OBS.items():
        if k == "task_name":
            obs[k] = np.array(task_name, dtype=f"<U{max(len(task_name), 1)}")
        elif k.endswith("rot_axis_angle") or k.endswith("rot_axis_angle_wrt_start"):
            obs[k] = np.tile(np.array([1, 0, 0, 0, 1, 0], dtype), (shape[0], 1))
        elif k == "robot0_gripper_width":
            obs[k] = np.full(shape, 0.07, dtype)
        else:
            obs[k] = np.zeros(shape, dtype)
    return obs


def tap_obs(tap_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(tap_dir, "call_*.npz")))
    if not files:
        raise FileNotFoundError(f"no call_*.npz in {tap_dir}")
    z = np.load(files[0], allow_pickle=False)
    obs = {k[len("obs_"):]: z[k] for k in z.files if k.startswith("obs_")}
    missing = [k for k in CONTRACT_OBS if k not in obs]
    if missing:
        raise KeyError(f"{files[0]} lacks obs keys {missing}")
    return obs


def load_expect(spec: str) -> dict:
    """`run/policy/<id>.yaml` (relative to controlSW/configs), a path, or a bare policy id -> its `contract` dict."""
    import yaml

    from franka_deploy.station import STATION_ROOT

    for p in (Path(spec), STATION_ROOT / spec, STATION_ROOT / "run" / "policy" / f"{spec}.yaml"):
        if p.is_file():
            return dict(yaml.safe_load(p.read_text()).get("contract") or {})
    raise FileNotFoundError(f"--expect {spec}: not a file (also tried under {STATION_ROOT})")


class Rpc:
    """One request per call, per-topic timeout, never resent. call() -> (kind, value): ok | error | timeout."""

    def __init__(self, endpoint: str, timeouts: dict):
        from franka_deploy.rmq_rpc import NoResendClient

        self.client = NoResendClient("policy_contract", endpoint.replace("0.0.0.0", "localhost"), timeouts=timeouts)

    def call(self, topic: str, payload):
        import robotmq

        from franka_deploy.rmq_rpc import RpcError, RpcTimeout

        t0 = time.monotonic()
        try:
            raw = self.client.request_with_data(topic=topic, data=robotmq.serialize(payload))
        except RpcTimeout as e:
            return "timeout", str(e), time.monotonic() - t0
        except RpcError as e:
            return "error", str(e), time.monotonic() - t0
        dt = time.monotonic() - t0
        reply = robotmq.deserialize(raw)
        if isinstance(reply, str) and reply.startswith("ERROR"):
            return "error", reply, dt
        return "ok", reply, dt


def _short(v, n=90):
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def check_reply(rep: Report, reply, K: int):
    """Steps 6 (reply part): keys, K, finiteness, rot6d, width, action 0."""
    if not isinstance(reply, dict):
        rep.ok("reply_keys", False, f"reply is {type(reply).__name__}, not a dict")
        return
    missing = [k for k in CONTRACT_REPLY if k not in reply]
    rep.ok("reply_keys", not missing, f"missing {missing}" if missing else f"all of {sorted(CONTRACT_REPLY)}")
    if missing:
        return
    arr = {k: np.asarray(reply[k]) for k in CONTRACT_REPLY}
    bad = {k: tuple(a.shape) for k, a in arr.items() if a.shape != (K, CONTRACT_REPLY[k][1])}
    got_k = {a.shape[0] for a in arr.values() if a.ndim}
    rep.ok("reply_shape", not bad, f"K={K}, wrong shapes {bad}" if bad else f"K={K}: "
           + ", ".join(f"{k} {tuple(a.shape)}" for k, a in arr.items()), expected_K=K, got_K=sorted(got_k))
    finite = all(np.isfinite(a.astype(float)).all() for a in arr.values())
    rep.ok("reply_finite", finite, "" if finite else "NaN or inf in the reply")
    if bad or not finite:
        return
    r = arr["action0_eef_rot_axis_angle"].astype(float)
    n1, n2 = np.linalg.norm(r[:, :3], axis=1), np.linalg.norm(r[:, 3:], axis=1)
    nmin, nmax = float(min(n1.min(), n2.min())), float(max(n1.max(), n2.max()))
    rep.ok("rot6d_norm", ROT_NORM[0] <= nmin and nmax <= ROT_NORM[1], f"row norms in [{nmin:.3f}, {nmax:.3f}], want {ROT_NORM}")
    dot = float(np.abs(np.sum(r[:, :3] * r[:, 3:], axis=1) / np.maximum(n1 * n2, 1e-12)).max())
    rep.ok("rot6d_orthogonal", dot < ROT_DOT_MAX, f"max |cos(row1, row2)| {dot:.3f}, want < {ROT_DOT_MAX}")
    w = arr["action0_gripper_width"].astype(float)
    rep.ok("width_range", WIDTH_RANGE[0] <= w.min() and w.max() <= WIDTH_RANGE[1],
           f"width in [{w.min():.4f}, {w.max():.4f}] m, want {WIDTH_RANGE}")
    p0 = float(np.linalg.norm(arr["action0_eef_pos"][0]))
    ang = float(np.degrees(np.arccos(np.clip((np.trace(rot6d_to_mat(r[0])) - 1) / 2, -1, 1))))
    rep.add("action0_identity", "pass" if p0 <= IDENT_POS_M and ang <= IDENT_ROT_DEG else "warn",
            f"action 0: {p0 * 100:.2f} cm, {ang:.2f} deg from identity (limit {IDENT_POS_M * 100:g} cm / {IDENT_ROT_DEG:g} deg)")


def run_contract(endpoint: str, prompt_index: int = 0, task_name: str = "block_raise", obs="dummy",
                 expect: dict | None = None, reset_max_s: float | None = None, inference_timeout_s: float = 5.0) -> dict:
    expect = expect or {}
    K = int(expect.get("K", 16))
    reset_max = float(reset_max_s if reset_max_s is not None else expect.get("reset_max_s", 0.5))
    rep = Report()
    timeouts = {**TIMEOUTS, "policy_inference": inference_timeout_s}
    rpc = Rpc(endpoint, timeouts)

    # 1. config
    kind, cfg, dt = rpc.call("policy_config", True)
    ws = cfg.get("workspace") if kind == "ok" and isinstance(cfg, dict) else None
    problems = [] if kind == "ok" else [f"{kind}: {_short(cfg)}"]
    if kind == "ok":
        if not isinstance(ws, dict):
            problems.append("no workspace dict")
        else:
            problems += [f"workspace.{k} missing" for k in CONFIG_KEYS if k not in ws]
        if not isinstance(cfg, dict) or "use_prompting" not in cfg:
            problems.append("use_prompting missing")
    rep.ok("config_keys", not problems, "; ".join(problems) or f"workspace {dict(ws)}, use_prompting {cfg['use_prompting']}")
    if problems:
        return finish(rep, endpoint, K, reset_max, None)
    prompting = bool(cfg["use_prompting"])
    if "uses_prompt" in expect:
        rep.ok("expect_uses_prompt", bool(expect["uses_prompt"]) == prompting,
               f"yaml uses_prompt {expect['uses_prompt']}, server use_prompting {prompting}")

    # 2. reset latency
    times, bad = [], None
    for _ in range(RESET_REPEATS):
        kind, val, dt = rpc.call("policy_reset", "RESET")
        times.append(dt)
        if kind != "ok" or val != "OK":
            bad = f"{kind}: {_short(val)}"
            break
    tmax = max(times)
    rep.ok("reset_latency", bad is None and tmax < reset_max,
           bad or f"max {tmax:.3f} s over {len(times)} resets, want < {reset_max} s", max_s=round(tmax, 4))
    if bad:
        return finish(rep, endpoint, K, reset_max, None)

    try:
        o = dummy_obs(task_name) if obs == "dummy" else tap_obs(obs.split(":", 1)[1])
    except (OSError, KeyError) as e:
        rep.ok("obs", False, str(e))
        return finish(rep, endpoint, K, reset_max, None)

    # 3. inference before a prompt
    enforced = False
    if prompting:
        kind, val, _ = rpc.call("policy_inference", o)
        enforced = kind == "error"
        if enforced:
            rep.ok("inference_before_prompt", True, f"ERROR as required: {_short(val)}")
        elif kind == "ok":
            rep.add("inference_before_prompt", "warn", "answered without a prompt: the server does not gate on prompts "
                    "(stateless); the second-prompt rule cannot be tested")
            rpc.call("policy_reset", "RESET")
        else:
            rep.ok("inference_before_prompt", False, f"{kind}: {_short(val)}")
    else:
        rep.add("inference_before_prompt", "skip", "use_prompting false")

    # 4. prompt
    kind, val, _ = rpc.call("policy_prompt", {"task_name": task_name, "prompt_index": prompt_index})
    good = kind == "ok" and val == "OK"
    if prompting or good:
        rep.ok("prompt_ok", good, f"{kind}: {_short(val)}")
    else:
        rep.add("prompt_ok", "warn", f"a non-prompting policy should ignore a prompt, got {kind}: {_short(val)}")

    # 5. second prompt without reset
    if prompting and enforced:
        kind, val, _ = rpc.call("policy_prompt", {"task_name": task_name, "prompt_index": prompt_index})
        rep.ok("second_prompt_rejected", kind == "error", f"{kind}: {_short(val)}")
    else:
        rep.add("second_prompt_rejected", "skip", "use_prompting false" if not prompting else "server does not gate on prompts")

    # 6. inference
    kind, val, dt = rpc.call("policy_inference", o)
    rep.ok("inference_ok", kind == "ok", f"{dt:.3f} s" if kind == "ok" else f"{kind}: {_short(val)}",
           seconds=round(dt, 4))
    reply = val if kind == "ok" else None
    if reply is not None:
        check_reply(rep, reply, K)
    if "K" in expect:
        got = None
        if isinstance(reply, dict) and "action0_eef_pos" in reply:
            got = int(np.asarray(reply["action0_eef_pos"]).shape[0])
        rep.ok("expect_K", got == K, f"yaml K {K}, reply K {got}")

    # 7. export topic
    kind, val, _ = rpc.call("export_recorded_data", time.strftime("%Y-%m-%d_%H-%M-%S"))
    rep.add("export_topic", "pass" if kind == "ok" else "warn", f"{kind}: {_short(val)}" if kind != "ok" else "answers")
    rpc.call("policy_reset", "RESET")   # leave the server without a prompt
    return finish(rep, endpoint, K, reset_max, prompting)


def finish(rep: Report, endpoint, K, reset_max, prompting) -> dict:
    return {"endpoint": endpoint, "K": K, "reset_max_s": reset_max, "use_prompting": prompting,
            "passed": not rep.failed, "failed": [c["name"] for c in rep.failed], "checks": rep.checks,
            "table": rep.table()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_deploy policy-contract", description=__doc__.split("\n\n")[0])
    ap.add_argument("--endpoint", required=True, help="robotmq endpoint of the policy server, e.g. tcp://localhost:18765")
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--task-name", default="block_raise")
    ap.add_argument("--obs", default="dummy", help="dummy | tap:DIR (first call_*.npz of a task tap directory)")
    ap.add_argument("--expect", default=None, help="run/policy/<id>.yaml (or an id): its contract {K, uses_prompt, "
                                                   "reset_max_s} must match the server")
    ap.add_argument("--reset-max-s", type=float, default=None, help="default 0.5, or the yaml's contract.reset_max_s")
    ap.add_argument("--inference-timeout", type=float, default=5.0, help="seconds to wait for each inference reply")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    a = ap.parse_args(argv)
    if not (a.obs == "dummy" or a.obs.startswith("tap:")):
        ap.error("--obs must be dummy or tap:DIR")
    res = run_contract(a.endpoint, a.prompt_index, a.task_name, a.obs, load_expect(a.expect) if a.expect else None,
                       a.reset_max_s, a.inference_timeout)
    print(res["table"])
    print(f"CONTRACT {'PASS' if res['passed'] else 'FAIL ' + ', '.join(res['failed'])} ({a.endpoint})")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
