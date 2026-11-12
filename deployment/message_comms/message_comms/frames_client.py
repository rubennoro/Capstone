"""FramesClient: C1 to process_frames (STRUCTURE.md 4.2)."""
from __future__ import annotations

from contracts import FramesReply, GetFramesRequest, GetHealthRequest, HealthReply, HealthReport, RpcClient

PROCESS = "process_frames"


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
