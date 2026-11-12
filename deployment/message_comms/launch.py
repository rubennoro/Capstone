# message_comms/launch.py: how deploy/ starts and checks this process (STRUCTURE.md section 9, deploy section 4).
# Imports only the standard library, yaml, deploy and contracts: never the message_comms package.
from __future__ import annotations

import sys

import deploy
from contracts import ACTION_STEPS, STEP_MS
from deploy import CheckResult, ManagedProcess, check_helpers

IMPORT_CHECK_TIMEOUT_S = 20.0
TIMEOUT_KEYS = ("commands_s", "health_s", "frames_s", "policy_hang_s", "config_s", "reset_s", "prompt_s", "home_s",
                "startup_s")
EXEC_H_RANGE = (1, ACTION_STEPS)
POSE_SOURCES = ("target", "measured")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class MessageCommsProcess(ManagedProcess):
    NAME = "message_comms"
    READY_HINT = ("connected to franka_control, process_frames and policy; fresh arm state, both cameras ok, "
                  "policy config accepted; check the other three processes' logs and endpoints in station.yaml")
    USES_TERMINAL = True

    def program(self) -> list[str]:
        return [self.station.python("light"), "-m", "message_comms"]

    def checks(self) -> list[CheckResult]:
        """Quick; starts nothing; --mock changes nothing (this is the real process under test)."""
        name = self.NAME
        cfg = self.config
        out = [check_helpers.python_imports(name, self.station.python("light"),
                                            ["message_comms", "contracts", "numpy", "scipy", "yaml"],
                                            IMPORT_CHECK_TIMEOUT_S)]
        script = cfg.get("operator_script")
        if script is not None:
            out.append(check_helpers.file_exists(name, "operator_script", self.config_path("operator_script")))
        elif not sys.stdin.isatty():
            out.append(CheckResult(name, "terminal", "fail", "no terminal for the keyboard: run deploy run in a "
                                   "terminal, or set message_comms.operator_script"))
        out.extend(self._value_checks(cfg))
        out.extend(self._franka_checks(cfg))
        out.append(check_helpers.info(name, "exec_h", f"exec_h = {cfg.get('exec_h')} "
                                      f"({_seconds_per_step(cfg.get('exec_h'))} per step; PLAN section 9 backup)"))
        out.append(check_helpers.info(name, "pose_source", f"pose_source = {cfg.get('pose_source')}"))
        out.append(check_helpers.info(name, "operator", "operator: keyboard" if script is None
                                      else f"operator: script {self.config_path('operator_script')}"))
        out.append(check_helpers.info(name, "tap", "tap: on" if cfg.get("save_tap") else "tap: off"))
        return out

    def franka_folder(self):
        """franka_control's folder, whose config.yaml holds watchdog.limit_s and homing.timeout_s."""
        return deploy.DEPLOYMENT_ROOT / "franka_control"

    def _value_checks(self, cfg) -> list[CheckResult]:
        problems = []
        exec_h = cfg.get("exec_h")
        is_int = isinstance(exec_h, int) and not isinstance(exec_h, bool)
        if not (is_int and EXEC_H_RANGE[0] <= exec_h <= EXEC_H_RANGE[1]):
            problems.append(f"exec_h must be between {EXEC_H_RANGE[0]} and {EXEC_H_RANGE[1]}, got {exec_h!r}")
        if cfg.get("pose_source") not in POSE_SOURCES:
            problems.append(f"pose_source must be one of {', '.join(POSE_SOURCES)}, got {cfg.get('pose_source')!r}")
        timeouts = cfg.get("timeouts") or {}
        for key in TIMEOUT_KEYS:
            value = timeouts.get(key)
            if not (_number(value) and value > 0):
                problems.append(f"timeouts.{key} must be > 0, got {value!r}")
        for key in ("odd_chunk_max_pos_m", "odd_chunk_max_rot_deg"):
            value = cfg.get(key)
            if not (_number(value) and value > 0):
                problems.append(f"{key} must be > 0, got {value!r}")
        every = cfg.get("tap_images_every")
        if not (isinstance(every, int) and not isinstance(every, bool) and every >= 1):
            problems.append(f"tap_images_every must be >= 1, got {every!r}")
        if problems:
            return [CheckResult(self.NAME, "settings", "fail", "; ".join(problems))]
        return [CheckResult(self.NAME, "settings", "pass", "values in range")]

    def _franka_checks(self, cfg) -> list[CheckResult]:
        """Heartbeat gap and homing against franka_control's real settings (STRUCTURE.md section 8)."""
        try:
            fc = deploy.load_process_config(self.franka_folder(), self.station.section("franka_control"))
            limit_s = float(fc["watchdog"]["limit_s"])
            homing_s = float(fc["homing"]["timeout_s"])
        except Exception as exc:          # a missing or broken franka_control config: say so, don't crash the table
            return [CheckResult(self.NAME, "franka_control limits", "fail",
                                f"cannot read franka_control/config.yaml watchdog.limit_s / homing.timeout_s: {exc}")]
        timeouts = cfg.get("timeouts") or {}
        try:
            gap_s = cfg["loop_ms"] / 1000 + timeouts["commands_s"] + timeouts["health_s"]
            home_s = float(timeouts["home_s"])
        except (KeyError, TypeError) as exc:
            return [CheckResult(self.NAME, "heartbeat gap", "fail", f"settings incomplete: {exc}")]
        out = []
        if gap_s < limit_s:
            out.append(CheckResult(self.NAME, "heartbeat gap", "pass",
                                   f"longest heartbeat gap {gap_s:.3f} s < franka_control watchdog.limit_s "
                                   f"{limit_s} s"))
        else:
            out.append(CheckResult(self.NAME, "heartbeat gap", "fail",
                                   f"longest heartbeat gap loop_ms/1000 + commands_s + health_s = {gap_s:.3f} s is not "
                                   f"below franka_control watchdog.limit_s {limit_s} s: lower message_comms timeouts"))
        if home_s > homing_s:
            out.append(CheckResult(self.NAME, "homing timeout", "pass",
                                   f"timeouts.home_s {home_s} s > franka_control homing.timeout_s {homing_s} s"))
        else:
            out.append(CheckResult(self.NAME, "homing timeout", "fail",
                                   f"timeouts.home_s {home_s} s must be above franka_control homing.timeout_s "
                                   f"{homing_s} s"))
        return out


def _seconds_per_step(exec_h) -> str:
    return f"{exec_h * STEP_MS / 1000:.2f} s" if isinstance(exec_h, int) else "?"


PROCESS_CLASS = MessageCommsProcess
