# adapted from rnoro/deployment:deployment/message_comms/tests/fake_servers.py @ 91b178b (renamed manager_servers)
# changes: FakeFrankaControl dropped (the tests use the in-process FakePc); imports; + make_fake_pc().
"""Fake frames and policy on real sockets (tcp://127.0.0.1:*), for test_flow.py and test_main.py (the PC is the
in-process FakePc of policy_deploy.fake_pc).

Each answers its contract's requests; `faults` make it misbehave in one way at a chosen moment, so the failure rows
can be driven through the real manager main().
"""
from __future__ import annotations

import time

import numpy as np

from deploy_contracts import Ack, ErrorReply, decode, pack_obj
from manager_fakes import health, make_chunk, make_frames, policy_config
from manager_raw_server import RawServer

from policy_deploy.contracts import HealthReply

MS = 1_000_000


class FakeProcessFrames:
    """C1. modes: "ok", "stale", "down", "silent" (stops answering)."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.calls: list[str] = []
        self.server = RawServer(self._answer)
        self.endpoint = self.server.endpoint

    def _report(self):
        if self.mode == "stale":
            return health("stale", "ok")
        if self.mode == "down":
            return health("down", "stale")
        return health()

    def _answer(self, data: bytes):
        req = decode(data)
        self.calls.append(req.SCHEMA)
        if self.mode == "silent":
            return None
        if req.SCHEMA == "c1.get_health/v1":
            return HealthReply(health=self._report())
        t1 = time.monotonic_ns() - 40 * MS
        return make_frames((t1 - 50 * MS, t1), self._report())

    def close(self) -> None:
        self.server.close()


class FakePolicy:
    """C4, a hold-still policy. on_infer(n) may return a fault for the n-th infer (1-based):
    "hang" (no reply), "error" (ErrorReply), "broken" (non-unit rot6d), "version" (c4.chunk/v2), "odd" (5 cm)."""

    def __init__(self, uses_prompt: bool = False) -> None:
        self.uses_prompt = uses_prompt
        self.calls: list[str] = []
        self.infers = 0
        self.faults: dict[int, str] = {}
        self.server = RawServer(self._answer)
        self.endpoint = self.server.endpoint

    def _answer(self, data: bytes):
        req = decode(data)
        self.calls.append(req.SCHEMA)
        if req.SCHEMA == "c4.config/v1":
            return policy_config(uses_prompt=self.uses_prompt)
        if req.SCHEMA in ("c4.reset/v1", "c4.prompt/v1"):
            return Ack(ok=True, reason="")
        self.infers += 1
        fault = self.faults.get(self.infers)
        if fault == "hang":
            return None
        if fault == "error":
            return ErrorReply(request=req.SCHEMA, error="CUDA error: device lost", field="")
        if fault == "broken":
            d = make_chunk().to_dict()
            d["eef_rot6d"] = np.full((16, 6), 3.0, dtype=np.float32)
            return pack_obj(d)
        if fault == "version":
            d = make_chunk().to_dict()
            d["schema"] = "c4.chunk/v2"
            return pack_obj(d)
        if fault == "odd":
            pos = np.zeros((16, 3))
            pos[:, 0] = 0.05
            return make_chunk(pos=pos, width_clipped=2)
        return make_chunk()

    def close(self) -> None:
        self.server.close()


def make_fake_pc(tmp_path, **config):
    """An in-process FakePc on free ports (tcp://127.0.0.1:*), started; config overrides config/fake_pc.yaml."""
    from policy_deploy.contracts import load_config
    from policy_deploy.fake_pc.fake_pc import FakePc
    from policy_deploy.fake_pc.settings import FakePcSettings
    base = load_config("fake_pc", None)
    base.update(config)
    settings = FakePcSettings.from_config(base, tmp_path, state_endpoint="tcp://127.0.0.1:*",
                                          commands_endpoint="tcp://127.0.0.1:*")
    fake = FakePc(settings, commit="0123456789abcdef0123456789abcdef01234567")
    fake.start()
    return fake
