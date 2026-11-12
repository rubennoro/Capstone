"""How `deploy` starts and checks the policy process (STRUCTURE.md 8). Not part of the `policy` package.

Imports only the standard library, `deploy` and `contracts` (deploy/STRUCTURE.md 4): the package itself may need
torch, so its checks run in a subprocess, `python -m policy.self_check`, in the policy's own environment.
"""

from __future__ import annotations

import json
import subprocess

from deploy import CheckResult, ManagedProcess, check_helpers

STATUSES = ("pass", "warn", "fail", "info")


class PolicyProcess(ManagedProcess):
    NAME = "policy"
    READY_HINT = ("checkpoint load + warm-up inference; check policy.kind, policy.checkpoint, policy.device "
                  "and the GPU")

    def program(self) -> list[str]:
        return [self.station.python("light" if self.mock else "policy"), "-m", "policy"]

    def checks(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        kind_unset = not self.mock and self.config.get("kind") is None
        if kind_unset:
            results.append(CheckResult(self.NAME, "kind", "fail",
                                       "policy.kind is not set: choose diffusion or hold_still in station.yaml"))
        results.append(check_helpers.port_free(self.NAME, self.station.endpoint("policy")))
        if not kind_unset:
            results.extend(self._self_check())
        return results

    def _self_check(self) -> list[CheckResult]:
        python = self.program()[0]
        cmd = [python, "-m", "policy.self_check", "--station", str(self.station.path)] + (["--mock"] if self.mock
                                                                                          else [])
        timeout_s = float(self.config["self_check_timeout_s"])
        try:
            proc = subprocess.run(cmd, cwd=self.folder, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            return [self._self_check_failed(f"timed out after {timeout_s:g} s", _text(e.stderr))]
        except OSError as e:
            return [self._self_check_failed(f"could not run {python}: {e}", "")]
        if proc.returncode != 0:
            return [self._self_check_failed(f"exited with code {proc.returncode}", proc.stderr)]
        results = []
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                name, status, detail = item["name"], item["status"], item["detail"]
                if status not in STATUSES or not isinstance(name, str) or not isinstance(detail, str):
                    raise ValueError(line)
            except (ValueError, KeyError, TypeError):
                return [self._self_check_failed(f"printed a line that is not a check: {line[:200]!r}", proc.stderr)]
            results.append(CheckResult(self.NAME, name, status, detail))
        if not results:
            return [self._self_check_failed("printed no checks", proc.stderr)]
        return results

    def _self_check_failed(self, what: str, stderr: str) -> CheckResult:
        tail = " | ".join(line for line in stderr.strip().splitlines()[-3:]) if stderr else ""
        detail = f"python -m policy.self_check {what}"
        if tail:
            detail += f": {tail}"
        detail += " (is the policy package installed in that Python? pip install -e policy)"
        return CheckResult(self.NAME, "self_check", "fail", detail)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


PROCESS_CLASS = PolicyProcess
