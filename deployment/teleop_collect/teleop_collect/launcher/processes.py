# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/launcher/processes.py @ 2ffe641
# changes: the four teleop_collect processes (FakePcProcess, FramesProcess, TeleopProcess, UiProcess; no
# PolicyProcess, no ManagerProcess); their rows are STRUCTURE.md 10.1's: frames without the imports row and with the
# R1 and preview ports instead of the C1 port (the rig file check runs teleop_collect.frames.rig_file); teleop (from
# ManagerProcess) without the terminal / operator-script row and without a self_check, with the U1 port and
# `teleop_collect.teleop.pc_probe --check --json` (not --mock); ui (new) with the HTTP port; the ready hints name this
# package's processes, and frames' real-run hint adds "(C button, LED blue)" for the Pi; imports. The source's own
# header follows.
# adapted from rnoro/deployment:deployment/process_frames/launch.py, policy/launch.py, message_comms/launch.py
# @ 91b178b
# changes: one class per laptop process in one file (FakePcProcess, FramesProcess, PolicyProcess, ManagerProcess);
# their checks are the rows of STRUCTURE.md 9.2; heavy checks run as subprocesses (frames.rig_file --json,
# policy.self_check --json, manager.self_check --json, manager.pc_probe --check --json); no c0/H.264/time-sync rows;
# the manager's heartbeat and homing rules moved into manager.self_check and pc_probe (link.yaml and the PC's hello,
# not franka_control's config); torch is python.torch for kind diffusion.
"""The laptop's processes as the launcher starts and checks them (STRUCTURE.md 10.1)."""
from __future__ import annotations

from pathlib import Path

from deploy_contracts import LinkSettings

from teleop_collect.launcher import check_helpers
from teleop_collect.launcher.check_result import CheckResult
from teleop_collect.launcher.managed_process import ManagedProcess

LOCAL_HOST = "127.0.0.1"
PC_PROBE_CHECK_TIMEOUT_S = 30.0         # pc_probe --check: up to 2 x watchdog of rule H2, then 2 s of hellos
PC_HINT = "check that control_deploy run is running on the PC, ASUS_5g, and link.yaml pc.address"


class FakePcProcess(ManagedProcess):
    """The Python stand-in for franka_control, --mock only (STRUCTURE.md 10.2)."""

    NAME = "fake_pc"
    READY_HINT = "C2 and C3 bound on 127.0.0.1 and the first C2 published; check the fake_pc log"

    def checks(self) -> list[CheckResult]:
        try:
            link = LinkSettings.load()
        except (ValueError, OSError) as e:
            return [CheckResult(self.NAME, "ports", "fail", f"contracts/link.yaml: {e}")]
        out = []
        for port, what in ((link.state_port, "C2"), (link.commands_port, "C3")):
            r = check_helpers.port_free(self.NAME, f"tcp://{LOCAL_HOST}:{port}", f"{what} port")
            out.append(r)
        return out


class FramesProcess(ManagedProcess):
    NAME = "frames"
    READY_HINT = "frames from both cameras; check the Pi on ASUS_5g (C button, LED blue) and station.yaml pi.address"

    def ready_hint(self) -> str:
        if self.mock:
            return "frames from both cameras; check the fake Pi (frames.mock_episode) in the frames log"
        return self.READY_HINT

    def checks(self) -> list[CheckResult]:
        python = self.python()
        timeout = float(self.config["check"]["subprocess_timeout_s"])
        out = [check_helpers.rig_file_check(self.NAME, python, self.config_path("rig_file"), timeout, self.folder)]
        out.append(check_helpers.port_free(self.NAME, self.station.r1_endpoint(), "R1 port"))
        out.append(check_helpers.port_free(self.NAME, self.station.preview_endpoint(), "preview port"))
        if self.mock:
            out.extend(self._mock_checks())
        else:
            host, port = self.station.pi_address, self.station.pi_port
            r = check_helpers.host_reachable(self.NAME, str(host), port, float(self.config["check"]
                                                                                ["reachable_timeout_s"]))
            if r.status == "fail":
                r = CheckResult(self.NAME, "Pi reachable", "fail",
                                f"no TCP connection to {host}:{port}: is the Pi on ASUS_5g and its sender running? "
                                f"(station.yaml pi.address)")
            else:
                r = CheckResult(self.NAME, "Pi reachable", "pass", r.detail)
            out.append(r)
        return out

    def _mock_checks(self) -> list[CheckResult]:
        out = []
        episode = self.config_path("mock_episode")
        if episode is None:
            out.append(CheckResult(self.NAME, "mock episode", "fail", "frames.mock_episode is not set"))
        elif not (Path(episode) / "meta.json").is_file():
            out.append(CheckResult(self.NAME, "mock episode", "fail",
                                   f"not found (no meta.json): {episode}: set frames.mock_episode in station.yaml"))
        else:
            out.append(CheckResult(self.NAME, "mock episode", "pass", f"found: {episode}"))
        out.append(check_helpers.port_free(self.NAME, self.station.mock_pi_endpoint(), "fake Pi port"))
        return out


class TeleopProcess(ManagedProcess):
    NAME = "teleop"
    READY_HINT = ("start-up: C2 from the PC, hello checks, clock map, the home, both cameras ok, data_root "
                  "writable; see the teleop log")

    def __init__(self, *args, fake_pc: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fake_pc = fake_pc                       # --mock: is the fake PC started by this run?

    def homes_on_start(self) -> bool:
        """teleop.home.on_start: the whole arm homes once during start-up (PLAN 8)."""
        home = self.config.get("home")
        return isinstance(home, dict) and home.get("on_start") is True

    def checks(self) -> list[CheckResult]:
        out = [check_helpers.port_free(self.NAME, self.station.u1_endpoint(), "U1 port")]
        if self.mock:
            detail = ("fake PC on 127.0.0.1, started by run" if self.fake_pc
                      else "PC on 127.0.0.1 started outside the launcher (cross test)")
            out.append(CheckResult(self.NAME, "PC", "info", detail))
        else:
            cmd = [self.python(), "-m", "teleop_collect.teleop.pc_probe", "--station", str(self.station.path),
                   "--check", "--json"]
            out += check_helpers.json_check_lines(self.NAME, cmd, PC_PROBE_CHECK_TIMEOUT_S, self.folder,
                                                  label="PC", hint=PC_HINT)
        return out


class UiProcess(ManagedProcess):
    NAME = "ui"
    READY_HINT = "the page served on 127.0.0.1 and teleop answering U1 get_status; see the ui log"

    def checks(self) -> list[CheckResult]:
        return [check_helpers.port_free(self.NAME, f"tcp://{LOCAL_HOST}:{self.station.http_port}", "HTTP port")]


PROCESS_CLASSES: dict[str, type[ManagedProcess]] = {
    "fake_pc": FakePcProcess, "frames": FramesProcess, "teleop": TeleopProcess, "ui": UiProcess,
}
