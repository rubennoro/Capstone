# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/errors.py @ 2ffe641
# changes: DecodeError dropped (images are never decoded); + RecordingRefused (FrameRecorder.start, R1 refusals).
# The source's own header follows.
# adapted from rnoro/deployment:deployment/process_frames/process_frames/errors.py @ 91b178b
# changes: FATAL lines name the process "frames"; SettingsError kept for the frames settings.
"""The errors of this process. A bad rig file is the copied loader's ConfigError."""

from __future__ import annotations


class FatalError(RuntimeError):
    """This process's own unrecoverable problem: exit 1 with the line ``FATAL frames: <what>: <check>``."""

    def __init__(self, what: str, check: str) -> None:
        super().__init__(f"{what}: {check}")
        self.what = what
        self.check = check


class SettingsError(ValueError):
    """config/frames_process.yaml or the station's frames section is invalid; ``problems`` lists every bad key."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


class RecordingRefused(Exception):
    """A recording cannot start; the message is the R1 refusal text (STRUCTURE.md 4.2), e.g. ``already recording``."""
