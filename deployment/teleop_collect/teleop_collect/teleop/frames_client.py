# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/frames_client.py @ 2ffe641
# changes: R1 instead of C1 (teleop_collect STRUCTURE.md 4.2): start_recording / stop_recording / get_health, the
# health timeout for get_health and timeouts.frames_s for start/stop (stop_recording takes a shorter one for the
# abort paths); one REQ socket, main thread only. The source's own header follows.
# copied from rnoro/deployment:deployment/message_comms/message_comms/frames_client.py @ 91b178b
# changes: process name frames; imports
"""FramesClient: R1 to frames (STRUCTURE.md 4.2), from teleop's main thread only."""
from __future__ import annotations

from deploy_contracts import Ack, RpcClient

from teleop_collect.contracts import FramesHealth, GetFramesHealth, RecordingSummary, StartRecording, StopRecording

PROCESS = "frames"


class FramesClient:
    def __init__(self, endpoint: str, health_timeout_s: float, frames_timeout_s: float) -> None:
        self._rpc = RpcClient(PROCESS, endpoint)
        self.endpoint = endpoint
        self._health_timeout_s = health_timeout_s
        self._frames_timeout_s = frames_timeout_s

    def start_recording(self, episode_dir: str) -> Ack:
        """A refusal is returned (ok false and the reason), not raised."""
        return self._rpc.request(StartRecording(episode_dir=str(episode_dir)), Ack, self._frames_timeout_s)

    def stop_recording(self, timeout_s: float | None = None) -> RecordingSummary:
        """RemoteError when frames is not recording. timeout_s: a shorter wait (the abort paths); default frames_s."""
        return self._rpc.request(StopRecording(), RecordingSummary,
                                 self._frames_timeout_s if timeout_s is None else timeout_s)

    def get_health(self) -> FramesHealth:
        return self._rpc.request(GetFramesHealth(), FramesHealth, self._health_timeout_s)

    def close(self) -> None:
        self._rpc.close()
