# copied from rnoro/deployment:deployment/message_comms/message_comms/frames_client.py @ 91b178b
# changes: process name frames; imports
"""FramesClient: C1 to frames (STRUCTURE.md 4.2)."""
from __future__ import annotations

from deploy_contracts import RpcClient

from policy_deploy.contracts import FramesReply, GetFramesRequest, GetHealthRequest, HealthReply, HealthReport

PROCESS = "frames"


class FramesClient:
    def __init__(self, endpoint: str, timeout_s: float) -> None:
        self._rpc = RpcClient(PROCESS, endpoint)
        self._timeout_s = timeout_s

    def get_frames(self) -> FramesReply:
        return self._rpc.request(GetFramesRequest(), FramesReply, self._timeout_s)

    def get_health(self) -> HealthReport:
        return self._rpc.request(GetHealthRequest(), HealthReply, self._timeout_s).health

    def close(self) -> None:
        self._rpc.close()
