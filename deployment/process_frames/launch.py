"""How deploy/ starts and checks process_frames (process_frames/STRUCTURE.md §11). Not part of the package.

Imports only the standard library, yaml, deploy and contracts. Sends and receives no contract messages: whether the
Pi answers time_sync and streams frames that match the rig file is checked by the process itself at start-up.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from deploy import CheckResult, ManagedProcess, check_helpers, parse_endpoint


class ProcessFramesProcess(ManagedProcess):
    NAME = "process_frames"
    READY_HINT = "frames from both cameras; check the Pi stream, Wi-Fi and process_frames.rig_file"

    def program(self) -> list[str]:
        return [self.station.python("light"), "-m", "process_frames"]

    def _fail(self, name: str, detail: str) -> CheckResult:
        return CheckResult(self.NAME, name, "fail", detail)

    def checks(self) -> list[CheckResult]:
        if not self.config:
            return [self._fail("settings", "station.yaml's process_frames section is invalid (see overrides)")]
        python = self.station.python("light")
        timeout = float(self.config["check"]["subprocess_timeout_s"])
        out = self._rig_checks(python, timeout)
        imports = check_helpers.python_imports(self.NAME, python, ["process_frames", "av", "cv2", "scipy", "zmq"],
                                               timeout)
        if imports.status == "fail" and "av (" in imports.detail:
            imports = self._fail("imports", imports.detail + "; PyAV missing: pip install av==17.1.0")
        out.append(imports)
        decoder, stream = self.config["decoder"], self.config["stream"]
        if stream == "pi_v1":
            detail = f"{decoder} (stream pi_v1: the Pi's JSON header + JPEG frames, clock from t_send_ns)"
        else:
            detail = f"{decoder} (stream c0)" if decoder == "h264" else f"{decoder} (stream c0, backup, PLAN §9)"
        out.append(check_helpers.info(self.NAME, "decoder", detail))
        out.append(check_helpers.port_free(self.NAME, self.station.endpoint("frames")))
        if self.mock:
            out.extend(self._mock_checks())
        else:
            endpoint = self.station.endpoint("pi_frames")
            host, port = parse_endpoint(endpoint)
            r = check_helpers.host_reachable(self.NAME, host, port, float(self.config["check"]["reachable_timeout_s"]))
            if r.status == "fail":
                r = self._fail("Pi reachable", f"no TCP connection to {host}:{port}: is the Pi on, on this Wi-Fi, "
                                               f"and its sender running?")
            else:
                r = CheckResult(self.NAME, "Pi reachable", r.status, r.detail)
            out.append(r)
            out.append(check_helpers.info(self.NAME, "camera source", f"Pi at {host}"))
        return out

    def _rig_checks(self, python: str, timeout: float) -> list[CheckResult]:
        rig = self.config_path("rig_file")
        exists = check_helpers.file_exists(self.NAME, "rig file", rig)
        if exists.status == "fail":
            return [exists]
        try:
            done = subprocess.run([python, "-m", "process_frames.rig_file", str(rig), "--json"], capture_output=True,
                                  text=True, timeout=timeout, stdin=subprocess.DEVNULL, cwd=self.folder)
            lines = done.stdout.strip().splitlines()
            result = json.loads(lines[-1]) if lines else None
        except subprocess.TimeoutExpired:
            return [self._fail("rig file", f"python -m process_frames.rig_file took over {timeout:g} s")]
        except (OSError, json.JSONDecodeError) as e:
            return [self._fail("rig file", f"cannot run python -m process_frames.rig_file: {e}")]
        if not isinstance(result, dict):
            why = (done.stderr.strip().splitlines() or ["no output"])[-1]
            return [self._fail("rig file", f"python -m process_frames.rig_file failed: {why}")]
        if not result.get("ok"):
            return [self._fail("rig file", str(result.get("error")))]
        cams = ", ".join(f"{role} {c['sensor']} {c['width']}x{c['height']}@{c['fps']:g}"
                         for role, c in result["cameras"].items())
        return [CheckResult(self.NAME, "rig file", "pass", f"{result['rig_id']}: {cams}")]

    def _mock_checks(self) -> list[CheckResult]:
        out = []
        if self.config.get("mock_episode") is None:
            out.append(self._fail("mock episode", "process_frames.mock_episode is not set"))
            episode = None
        else:
            episode = self.config_path("mock_episode")
            if not (Path(episode) / "meta.json").is_file():
                out.append(self._fail("mock episode", f"not found (no meta.json): {episode}"))
            else:
                out.append(CheckResult(self.NAME, "mock episode", "pass", f"found: {episode}"))
        for name in ("mock_pi_frames", "mock_pi_time_sync"):
            out.append(check_helpers.port_free(self.NAME, self.station.endpoint(name)))
        out.append(check_helpers.info(self.NAME, "camera source",
                                      f"FakePiSender, stand-in Pi from {episode} (mock)"))
        return out


PROCESS_CLASS = ProcessFramesProcess
