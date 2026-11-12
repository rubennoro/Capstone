"""How deploy/ starts and checks franka_control (franka_control/STRUCTURE.md §11). Not part of any package.

Imports only the standard library, yaml, deploy and contracts. Never moves the arm and sends no contract messages:
the binary checks the arm itself at start-up (a problem there is its FATAL line).
"""
from __future__ import annotations

import resource
import subprocess
from pathlib import Path

from deploy import CheckResult, ManagedProcess, check_helpers

ARM_COMMAND_PORT = 1337        # libfranka's command port on the arm (a fact of the FCI, not a setting)
REACH_TIMEOUT_S = 2.0          # the arm-reachable check (STRUCTURE.md §11)
BINARY_TIMEOUT_S = 10.0        # `--version` / `--check-config` answer in milliseconds; this only catches a hang
RTPRIO_MIN = 99
REALTIME_FLAG = Path("/sys/kernel/realtime")


class FrankaControlProcess(ManagedProcess):
    NAME = "franka_control"
    READY_HINT = ("connected to the arm (or FakeArm) and holding its pose; check franka_control.robot_ip, "
                  "FCI enabled in Desk, brakes unlocked, user stop released, and the log")

    def program(self) -> list[str]:
        return [str(self.config_path("binary"))]

    def _fail(self, name: str, detail: str) -> CheckResult:
        return CheckResult(self.NAME, name, "fail", detail)

    def _run(self, args: list[str]) -> subprocess.CompletedProcess | str:
        """The binary's output, or a one-line reason it could not be run."""
        try:
            return subprocess.run([str(self.config_path("binary")), *args], capture_output=True, text=True,
                                  timeout=BINARY_TIMEOUT_S, stdin=subprocess.DEVNULL, cwd=self.folder)
        except subprocess.TimeoutExpired:
            return f"{' '.join(args)} did not answer within {BINARY_TIMEOUT_S:g} s"
        except OSError as exc:
            return f"cannot run {self.config_path('binary')}: {exc}"

    def checks(self) -> list[CheckResult]:
        if not self.config:
            return [self._fail("settings", "station.yaml's franka_control section is invalid (see overrides)")]
        out = [self._version_check(), self._config_check()]
        out.append(check_helpers.port_free(self.NAME, self.station.endpoint("state")))
        out.append(check_helpers.port_free(self.NAME, self.station.endpoint("commands")))
        robot_ip = str(self.config["robot_ip"])
        realtime = self.config["realtime"]
        if self.mock:
            out.append(check_helpers.info(self.NAME, "backups", "arm: FakeArm, gripper: FakeGripper (--mock)"))
            return out
        reach = check_helpers.host_reachable(self.NAME, robot_ip, ARM_COMMAND_PORT, REACH_TIMEOUT_S)
        if reach.status == "fail":
            out.append(self._fail("arm reachable", f"arm not reachable at {robot_ip}:{ARM_COMMAND_PORT}: check the "
                                                   f"cable, the PC's static IP, FCI enabled in Desk"))
        else:
            out.append(CheckResult(self.NAME, "arm reachable", "pass", reach.detail))
        if realtime == "enforce":
            out.append(self._rt_kernel_check())
            out.append(self._rtprio_check())
        else:
            out.append(check_helpers.warn(
                self.NAME, "realtime",
                "real-time ignore mode: expect occasional reflexes under load; late cycles are shown as C2 "
                "`loop_late` on the status line and logged every second"))
        if float(self.config["load"]["mass_kg"]) == 0.0:
            out.append(check_helpers.warn(self.NAME, "payload", "no payload set: the rig's weight is not compensated"))
        out.append(check_helpers.info(self.NAME, "backups", f"arm: LibfrankaArm at {robot_ip}, realtime {realtime}"))
        return out

    def _version_check(self) -> CheckResult:
        want = str(self.config["libfranka_version"])
        done = self._run(["--version"])
        if isinstance(done, str):
            return self._fail("version", f"{done}: rebuild: franka_control/build.sh")
        words = done.stdout.split()
        built = words[-1] if len(words) >= 2 and words[-2] == "libfranka" else None
        if built is None:
            return self._fail("version", f"unexpected --version output {done.stdout.strip()!r}: rebuild: "
                                         f"franka_control/build.sh")
        if built == want:
            return CheckResult(self.NAME, "version", "pass", done.stdout.strip())
        detail = f"built with libfranka {built}, config.yaml wants {want}: rebuild: franka_control/build.sh"
        if self.mock:
            return check_helpers.info(self.NAME, "version", detail + " (fine for --mock)")
        return self._fail("version", detail)

    def _config_check(self) -> CheckResult:
        args = ["--check-config", "--station", str(self.station.path)] + (["--mock"] if self.mock else [])
        done = self._run(args)
        if isinstance(done, str):
            return self._fail("config", done)
        if done.returncode == 0:
            return CheckResult(self.NAME, "config", "pass", "the binary accepts the merged config")
        problems = [line for line in done.stdout.splitlines() if line.strip()]
        if not problems:
            problems = [(done.stderr.strip().splitlines() or [f"exit {done.returncode}"])[-1]]
        more = f" ({len(problems)} problems)" if len(problems) > 1 else ""
        return self._fail("config", problems[0] + more)

    def _rt_kernel_check(self) -> CheckResult:
        try:
            flag = REALTIME_FLAG.read_text().strip()
        except OSError:
            flag = ""
        if flag == "1":
            return CheckResult(self.NAME, "rt kernel", "pass", "PREEMPT_RT kernel")
        return self._fail("rt kernel", "no PREEMPT_RT kernel: install one, or set franka_control.realtime: ignore "
                                       "(backup, PLAN §9)")

    def _rtprio_check(self) -> CheckResult:
        soft, _ = resource.getrlimit(resource.RLIMIT_RTPRIO)
        if soft == resource.RLIM_INFINITY or soft >= RTPRIO_MIN:
            limit = "unlimited" if soft == resource.RLIM_INFINITY else str(soft)
            return CheckResult(self.NAME, "rtprio", "pass", f"rtprio limit {limit}")
        return self._fail("rtprio", f"rtprio limit {soft} < {RTPRIO_MIN}: add the user to the realtime group with "
                                    f"rtprio {RTPRIO_MIN} in /etc/security/limits.conf, log in again")


PROCESS_CLASS = FrankaControlProcess
