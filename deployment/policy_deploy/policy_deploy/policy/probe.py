# adapted from rnoro/deployment:deployment/policy/policy/probe.py @ 91b178b
# (there: new; replaces rnoro/main-sim:controlSW/franka_deploy/franka_deploy/tools/policy_contract.py @ 935c0b3)
# changes: module path, the new station (--station optional); the config line prints the policy's kind; the infer line
# reports mean / p50 / p95 / max over --infers N
"""`python -m policy_deploy.policy.probe [--station FILE] [--infers N]`: talk C4 to a running policy, like the manager.

Sends config, reset (5 times, timed), prompt (only if config says uses_prompt), infer with the neutral observation
(N times, timed), then a final reset, and prints one line per step. Light environment, no torch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Sequence, TextIO

import numpy as np
from deploy_contracts import Ack, ContractError, RemoteError, RpcClient, TransportTimeout

from policy_deploy.contracts import (ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS, ActionChunk, PolicyConfig,
                                     PolicyConfigRequest, Prompt, Reset)
from policy_deploy.policy.errors import SettingsError
from policy_deploy.policy.neutral_observation import neutral_observation
from policy_deploy.policy.settings import Settings


class StepFailed(Exception):
    pass


def timing_summary(ms: Sequence[float]) -> str:
    """"mean 12.3 ms, p50 12.0 ms, p95 15.1 ms, max 16.2 ms" over the given times."""
    a = np.asarray(ms, dtype=np.float64)
    return (f"mean {a.mean():.1f} ms, p50 {np.percentile(a, 50):.1f} ms, p95 {np.percentile(a, 95):.1f} ms, "
            f"max {a.max():.1f} ms")


class Probe:
    def __init__(self, client: RpcClient, gripper_width_m: float, timeout_s: float, out: TextIO,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._client = client
        self._width = gripper_width_m
        self._timeout_s = timeout_s
        self._out = out
        self._clock = clock

    def run(self, resets: int, infers: int, task_name: str | None, prompt_index: int | None) -> int:
        """0 if every step passed, else 1."""
        try:
            config = self._request("config", PolicyConfigRequest(), PolicyConfig)
            expected = (OBS_STEPS, ACTION_STEPS, STEP_MS, IMAGE_SIZE)
            got = (config.obs_steps, config.action_steps, config.step_ms, config.image_size)
            if got != expected:
                raise StepFailed(f"FAIL config: sizes {got} != C4 {expected}")
            self._line(f"PASS config: kind {config.kind}, inputs {list(config.inputs)}, "
                       f"uses_prompt {config.uses_prompt}")
            reset_ms = [self._timed("reset", Reset(), Ack)[1] for _ in range(resets)]
            if resets:
                self._line(f"PASS reset {max(reset_ms):.1f} ms (max of {resets})")
            if config.uses_prompt:
                if task_name is None or prompt_index is None:
                    raise StepFailed("FAIL prompt: the policy uses prompts: pass --task-name and --prompt-index")
                self._request("prompt", Prompt(task_name=task_name, prompt_index=prompt_index), Ack)
                self._line(f"PASS prompt {task_name!r} {prompt_index}")
            obs = neutral_observation(self._width)
            infer_ms, clipped = [], 0
            for _ in range(infers):
                chunk, ms = self._timed("infer", obs, ActionChunk)
                infer_ms.append(ms)
                clipped += int(chunk.width_clipped)
            if infers:
                self._line(f"PASS infer {timing_summary(infer_ms)} ({infers} chunks, width_clipped total {clipped})")
            self._request("reset", Reset(), Ack)
            self._line("PASS final reset (the policy is left reset)")
            return 0
        except StepFailed as e:
            self._line(str(e))
            return 1

    def _timed(self, step: str, msg: object, expect: type):
        t0 = self._clock()
        reply = self._request(step, msg, expect)
        return reply, (self._clock() - t0) / 1e6

    def _request(self, step: str, msg: object, expect: type):
        try:
            reply = self._client.request(msg, expect, timeout_s=self._timeout_s)
        except RemoteError as e:
            raise StepFailed(f"FAIL {step}: ErrorReply: {e.reply.error}") from e
        except (TransportTimeout, ContractError) as e:
            raise StepFailed(f"FAIL {step}: {type(e).__name__}: {e}") from e
        if isinstance(reply, Ack) and not reply.ok:
            raise StepFailed(f"FAIL {step}: refused: {reply.reason}")
        return reply

    def _line(self, text: str) -> None:
        self._out.write(text + "\n")
        self._out.flush()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m policy_deploy.policy.probe", description=__doc__.split("\n\n")[0])
    ap.add_argument("--station", type=Path, default=None,
                    help="the station file (default: deployment/policy_deploy/station.yaml)")
    ap.add_argument("--infers", type=int, default=10, help="number of infer requests (timed: mean/p50/p95/max)")
    ap.add_argument("--resets", type=int, default=5, help="number of timed reset requests")
    ap.add_argument("--timeout-s", type=float, default=30.0, help="per-request timeout")
    ap.add_argument("--task-name", default=None)
    ap.add_argument("--prompt-index", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        # mock=True: the probe needs only the endpoint and the warm-up width, not policy.kind or a checkpoint
        settings = Settings.load(a.station, mock=True)
    except SettingsError as e:
        print("FAIL settings: " + "; ".join(e.problems))
        return 1
    client = RpcClient("policy", settings.endpoint)
    try:
        return Probe(client, settings.warmup_gripper_width_m, a.timeout_s, sys.stdout).run(
            a.resets, a.infers, a.task_name, a.prompt_index)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
