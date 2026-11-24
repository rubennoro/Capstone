# adapted from rnoro/deployment:deployment/message_comms/message_comms/errors.py @ 91b178b
# changes: SettingsError is policy_deploy.contracts' StationError; + PcRestarted (C2 seq going back, or a hello t_pc_ns
# going back: the PC rebooted).
"""The errors the manager adds to deploy_contracts' (STRUCTURE.md 6, 8)."""
from __future__ import annotations

from policy_deploy.contracts import StationError as SettingsError  # noqa: F401  (one error type for settings)


class Refused(RuntimeError):
    """A server answered Ack(ok=false)."""

    def __init__(self, process: str, request: str, reason: str) -> None:
        self.process = process
        self.request = request
        self.reason = reason
        super().__init__(f"{process}: {request} refused: {reason}")


class RequestCancelled(RuntimeError):
    """A long request (home) was cancelled because the operator or the launcher asked to stop; its reply is dropped."""

    def __init__(self, process: str, request: str) -> None:
        self.process = process
        self.request = request
        super().__init__(f"{process}: {request} cancelled (reply dropped)")


class PcRestarted(RuntimeError):
    """franka_control restarted (C2 seq went back), or the PC rebooted (a hello t_pc_ns went back)."""

    def __init__(self, detail: str, rebooted: bool = False) -> None:
        self.detail = detail
        self.rebooted = rebooted
        super().__init__(detail)


class ScriptError(ValueError):
    """A broken operator script."""


class ScriptTimeout(ScriptError):
    """A scripted wait, or a key EpisodeLoop keeps refusing, not done within operator_script_wait_s."""
