"""FrankaControlBinary: runs franka_control --version and --check-config and parses them (control_deploy/STRUCTURE.md
6.3), and builds its run command (6.4). The wrapper never re-implements the config merge: effective values come from
--check-config."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TIMEOUT_S = 10.0   # both answer in milliseconds; this only catches a hang
VERSION_RE = re.compile(r"^franka_control commit=(\S+) dirty=(yes|no) libfranka=(\S+)$")


class BinaryError(RuntimeError):
    """The binary could not be run, hung, or printed something unexpected (one line)."""


@dataclass(frozen=True)
class VersionInfo:
    commit: str          # 40 hex or "unknown"
    dirty: bool
    libfranka: str       # "0.9.2" or "none"
    line: str


@dataclass(frozen=True)
class CheckConfigResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    config: dict[str, Any] | None = None   # the YAML of --check-config when ok

    def effective(self, *keys: str, default: Any = None) -> Any:
        """A value of the merged franka_control section, e.g. effective("load", "mass_kg")."""
        node: Any = (self.config or {}).get("franka_control")
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def contracts(self) -> dict[str, Any]:
        return dict((self.config or {}).get("contracts") or {})


class FrankaControlBinary:
    def __init__(self, path: Path | str, timeout_s: float = TIMEOUT_S) -> None:
        self.path = Path(path)
        self.timeout_s = timeout_s

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run([str(self.path), *args], capture_output=True, text=True, timeout=self.timeout_s,
                                  stdin=subprocess.DEVNULL, cwd="/")
        except subprocess.TimeoutExpired:
            raise BinaryError(f"{self.path.name} {' '.join(args)} did not answer within {self.timeout_s:g} s") from None
        except OSError as exc:
            raise BinaryError(f"cannot run {self.path}: {exc.strerror or exc}") from None

    def version(self) -> VersionInfo:
        done = self._run(["--version"])
        line = done.stdout.strip()
        m = VERSION_RE.match(line)
        if done.returncode != 0 or m is None:
            raise BinaryError(f"unexpected --version output {line!r} (exit {done.returncode})")
        return VersionInfo(commit=m.group(1), dirty=m.group(2) == "yes", libfranka=m.group(3), line=line)

    def check_config(self, station: Path | str, mock: bool, local: bool) -> CheckConfigResult:
        args = ["--check-config", "--station", str(station)] + (["--mock"] if mock else []) + \
               (["--local"] if local else [])
        done = self._run(args)
        if done.returncode == 0:
            try:
                return CheckConfigResult(ok=True, config=yaml.safe_load(done.stdout) or {})
            except yaml.YAMLError as exc:
                raise BinaryError(f"--check-config printed invalid YAML ({exc})") from None
        problems = [line for line in done.stdout.splitlines() if line.strip()]
        if not problems:
            problems = [(done.stderr.strip().splitlines() or [f"exit {done.returncode}"])[-1]]
        return CheckConfigResult(ok=False, problems=problems)

    def command(self, station: Path | str, session: Path | str, ready_file: Path | str, mock: bool, local: bool,
                keyboard: bool = True) -> list[str]:
        """The run command of control_deploy run (6.4 step 3)."""
        cmd = [str(self.path), "--station", str(station), "--session", str(session), "--ready-file", str(ready_file)]
        if keyboard:
            cmd.append("--keyboard")
        if mock:
            cmd.append("--mock")
        if local:
            cmd.append("--local")
        return cmd
