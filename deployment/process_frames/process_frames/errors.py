"""The errors of this process (STRUCTURE.md §5.1). A bad rig file is the copied loader's ConfigError."""

from __future__ import annotations


class FatalError(RuntimeError):
    """This process's own unrecoverable problem: exit 1 with the line ``FATAL process_frames: <what>: <check>``."""

    def __init__(self, what: str, check: str) -> None:
        super().__init__(f"{what}: {check}")
        self.what = what
        self.check = check


class DecodeError(Exception):
    """A FrameDecoder could not decode one frame."""

    def __init__(self, camera: str, seq: int, detail: str) -> None:
        super().__init__(f"{camera} frame seq {seq}: {detail}")
        self.camera = camera
        self.seq = seq
        self.detail = detail


class SettingsError(ValueError):
    """config.yaml or the station's process_frames section is invalid; ``problems`` lists every bad key."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)
