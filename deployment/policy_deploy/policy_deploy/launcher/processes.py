# adapted from rnoro/deployment:deployment/process_frames/launch.py, policy/launch.py, message_comms/launch.py
# @ 91b178b
# changes: one class per laptop process in one file (FakePcProcess, FramesProcess, PolicyProcess, ManagerProcess);
# their checks are the rows of STRUCTURE.md 9.2; heavy checks run as subprocesses (frames.rig_file --json,
# policy.self_check --json, manager.self_check --json, manager.pc_probe --check --json); no c0/H.264/time-sync rows;
# the manager's heartbeat and homing rules moved into manager.self_check and pc_probe (link.yaml and the PC's hello,
# not franka_control's config); torch is python.torch for kind diffusion.
# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/launcher/processes.py @ f632576
# changes: FramesProcess(fake_pi=...) as ManagerProcess(fake_pc=...): under --mock with launcher.fake_pi false the
# rows are those of a real run (Pi reachable at pi.address:pi.port) instead of the fake Pi's; the "kind not set" hint
# names replay (sim_deploy)
"""The laptop's processes as the launcher starts and checks them (STRUCTURE.md 9.1, 9.2)."""
from __future__ import annotations

import sys
from pathlib import Path

from deploy_contracts import LinkSettings, TcpFrame

from policy_deploy.contracts.station import StationError
from policy_deploy.launcher import check_helpers
from policy_deploy.launcher.check_result import CheckResult
from policy_deploy.launcher.managed_process import ManagedProcess

LOCAL_HOST = "127.0.0.1"
MANAGER_SELF_CHECK_TIMEOUT_S = 20.0     # manager.self_check reads files only
PC_PROBE_CHECK_TIMEOUT_S = 30.0         # pc_probe --check: up to 2 x watchdog of rule H2, then 2 s of hellos


class FakePcProcess(ManagedProcess):
    """The Python stand-in for franka_control, --mock only (STRUCTURE.md 10)."""

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
    READY_HINT = "frames from both cameras; check the Pi on ASUS_5g and station.yaml pi.address"

    def __init__(self, *args, fake_pi: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fake_pi = fake_pi                       # --mock: does frames start the fake Pi? (launcher.fake_pi)

    def ready_hint(self) -> str:
        if self.mock and self.fake_pi:
            return "frames from both cameras; check the fake Pi (frames.mock_episode) in the frames log"
        return self.READY_HINT

    def checks(self) -> list[CheckResult]:
        python = self.python()
        timeout = float(self.config["check"]["subprocess_timeout_s"])
        out = [check_helpers.rig_file_check(self.NAME, python, self.config_path("rig_file"), timeout, self.folder)]
        imports = check_helpers.python_imports(self.NAME, python, ["cv2", "scipy", "zmq", "av"], timeout)
        if imports.status == "fail":
            imports = CheckResult(self.NAME, "imports", "fail",
                                  imports.detail + ': install the light env: uv pip install -e '
                                                   '"deployment/policy_deploy[light,test]"')
        out.append(imports)
        out.append(check_helpers.port_free(self.NAME, self.station.frames_endpoint(), "C1 port"))
        if self.mock and self.fake_pi:
            out.extend(self._mock_checks())
        elif self.station.pi_address is None:                # only possible under --mock (Station requires it)
            out.append(CheckResult(self.NAME, "Pi reachable", "fail",
                                   "station.yaml pi.address is not set: required with launcher.fake_pi false"))
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


class PolicyProcess(ManagedProcess):
    NAME = "policy"
    READY_HINT = ("model loaded, warmed up, C4 answers; check policy.kind, policy.checkpoint, policy.device "
                  "and the GPU")

    def kind(self) -> str | None:
        """The kind that runs: policy.mock_kind under --mock, else policy.kind."""
        return self.config.get("mock_kind") if self.mock else self.config.get("kind")

    def python(self) -> str:
        """python.torch for kind diffusion (raises StationError if unset); python.light for everything else."""
        return self.station.python("torch" if self.kind() == "diffusion" else "light")

    def checks(self) -> list[CheckResult]:
        out: list[CheckResult] = []
        kind = self.kind()
        if kind is None:                             # otherwise the self-check prints the kind row
            out.append(CheckResult(self.NAME, "kind", "fail",
                                   "policy.kind is not set: choose hold_still, diffusion or replay in station.yaml"))
        if not self.mock and kind is not None:
            settled = self._settled()
            if settled is None:
                out.append(CheckResult(self.NAME, "settled", "fail", "contracts/frames.yaml cannot be read"))
            elif kind != "hold_still" and not settled:
                out.append(CheckResult(self.NAME, "settled", "fail",
                                       f"trained policy refused on the real arm: contracts/frames.yaml settled is "
                                       f"false (PLAN 9): use policy.kind hold_still, or settle the TCP"))
            else:
                out.append(CheckResult(self.NAME, "settled", "pass",
                                       f"frames.yaml settled: {str(settled).lower()}, kind {kind}"))
        if kind == "diffusion":
            try:
                out.append(check_helpers.executable(self.NAME, "torch env", Path(self.station.python("torch"))))
            except StationError as e:
                out.append(CheckResult(self.NAME, "torch env", "fail", "; ".join(e.problems)))
        if kind is not None:
            out.extend(self._self_check())
        out.append(check_helpers.port_free(self.NAME, self.station.policy_endpoint(), "C4 port"))
        return out

    @staticmethod
    def _settled() -> bool | None:
        try:
            return TcpFrame.load().settled
        except (ValueError, OSError):
            return None

    def _self_check(self) -> list[CheckResult]:
        try:
            python = self.python()
        except StationError as e:
            return [CheckResult(self.NAME, "self_check", "info", f"not run: {'; '.join(e.problems)}")]
        cmd = [python, "-m", "policy_deploy.policy.self_check", "--station", str(self.station.path)]
        cmd += ["--mock"] if self.mock else []
        cmd.append("--json")
        return check_helpers.json_check_lines(
            self.NAME, cmd, float(self.config["self_check_timeout_s"]), self.folder,
            hint="is policy_deploy installed editable in that Python?")


class ManagerProcess(ManagedProcess):
    NAME = "manager"
    READY_HINT = ("start-up: C2 from the PC, hello checks, clock map, policy config, both cameras ok; see the "
                  "manager log")
    USES_TERMINAL = True

    def __init__(self, *args, fake_pc: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fake_pc = fake_pc                       # --mock: is the fake PC started by this run?

    def checks(self) -> list[CheckResult]:
        out = [self._terminal_row()]
        cmd = [self.python(), "-m", "policy_deploy.manager.self_check", "--station", str(self.station.path)]
        cmd += ["--mock"] if self.mock else []
        cmd.append("--json")
        out += check_helpers.json_check_lines(self.NAME, cmd, MANAGER_SELF_CHECK_TIMEOUT_S, self.folder)
        if self.mock:
            detail = ("fake PC on 127.0.0.1, started by run" if self.fake_pc
                      else "PC on 127.0.0.1 started outside the launcher (cross test)")
            out.append(CheckResult(self.NAME, "PC", "info", detail))
        else:
            cmd = [self.python(), "-m", "policy_deploy.manager.pc_probe", "--station", str(self.station.path),
                   "--check", "--json"]
            out += check_helpers.json_check_lines(self.NAME, cmd, PC_PROBE_CHECK_TIMEOUT_S, self.folder,
                                                  label="PC", hint="check that control_deploy run is running on "
                                                                   "the PC, ASUS_5g, and link.yaml pc.address")
        return out

    def _terminal_row(self) -> CheckResult:
        if self.config.get("operator_script") is not None:
            return check_helpers.file_exists(self.NAME, "operator script", self.config_path("operator_script"))
        if not sys.stdin.isatty():
            return CheckResult(self.NAME, "terminal", "fail", "no terminal for the keyboard: run policy_deploy run "
                                                              "in a terminal, or set manager.operator_script")
        return CheckResult(self.NAME, "terminal", "pass", "keyboard on this terminal")


PROCESS_CLASSES: dict[str, type[ManagedProcess]] = {
    "fake_pc": FakePcProcess, "frames": FramesProcess, "policy": PolicyProcess, "manager": ManagerProcess,
}
