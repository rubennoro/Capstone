"""PcChecks: every row of `control_deploy check` (control_deploy/STRUCTURE.md 6.3). Never moves the arm, never connects
to FCI. The rows version, config, ports, arm reachable, rt kernel, rtprio, realtime, payload and backups are ported
from rnoro/deployment:deployment/franka_control/launch.py:42-118 @ 91b178b.

Everything that touches the machine goes through `Probes`, so the tests inject each input.
"""
from __future__ import annotations

import ipaddress
import os
import resource
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import deploy_contracts
import yaml
from deploy_contracts import ContractLimits, LinkSettings, TcpFrame

from control_deploy import check_helpers, firewall, netinfo, paths
from control_deploy.binary import BinaryError, CheckConfigResult, FrankaControlBinary, VersionInfo
from control_deploy.check_result import CheckResult
from control_deploy.station import Station

P = "control_deploy"
ARM_COMMAND_PORT = 1337        # libfranka's command port on the arm (a fact of the FCI, not a setting)
REACH_TIMEOUT_S = 2.0
RTPRIO_MIN = 99
REALTIME_FLAG = Path("/sys/kernel/realtime")
FCI_NETWORK = ipaddress.ip_network("172.16.0.0/24")
BUILD = "control_deploy/franka_control/build.sh"


def _git(args: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-C", str(paths.repo_root()), *args], capture_output=True, text=True,
                              timeout=10.0, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_head() -> str | None:
    done = _git(["rev-parse", "HEAD"])
    return done.stdout.strip() if done is not None and done.returncode == 0 else None


def git_dirty() -> bool:
    """Tracked changes in deployment/contracts/ or deployment/control_deploy/ (what build_info.h's dirty flag covers)."""
    done = _git(["status", "--porcelain", "--untracked-files=no", "--", str(paths.contracts_dir()),
                 str(paths.deploy_dir())])
    return bool(done is not None and done.returncode == 0 and done.stdout.strip())


def rt_flag() -> str:
    try:
        return REALTIME_FLAG.read_text().strip()
    except OSError:
        return ""


def rtprio_limit() -> int:
    soft, _ = resource.getrlimit(resource.RLIMIT_RTPRIO)
    return soft


def todo_keys(config_text: str) -> list[str]:
    """The dotted keys of config.yaml lines still marked TODO (comment-only lines are skipped)."""
    stack: list[tuple[int, str]] = []
    out: list[str] = []
    for raw in config_text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if not stripped or stripped.startswith("-"):
            continue
        key = stripped.split(":", 1)[0].strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        if "TODO" in raw.split("#", 1)[-1] and "#" in raw:
            out.append(".".join(k for _, k in stack))
    return out


def _has_key(tree: Any, dotted: str) -> bool:
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


@dataclass
class Probes:
    """Every input from the machine. The defaults are the real ones."""

    contracts_dir: Callable[[], Path] = deploy_contracts.paths.contracts_dir
    expected_contracts_dir: Callable[[], Path] = paths.contracts_dir
    git_head: Callable[[], str | None] = git_head
    git_dirty: Callable[[], bool] = git_dirty
    binary: Callable[[Path], FrankaControlBinary] = FrankaControlBinary
    local_addresses: Callable[[], dict[str, list[str]]] = netinfo.local_addresses
    port_free: Callable[[str], CheckResult] = field(default=lambda ep: check_helpers.port_free(P, ep))
    read_firewall: Callable[[], firewall.LiveTable] = firewall.read_live
    ufw_enabled: Callable[[], bool] = firewall.ufw_enabled
    host_reachable: Callable[[str, int, float], CheckResult] = field(
        default=lambda host, port, timeout: check_helpers.host_reachable(P, host, port, timeout))
    rt_flag: Callable[[], str] = rt_flag
    rtprio_limit: Callable[[], int] = rtprio_limit
    config_text: Callable[[], str] = field(default=lambda: paths.config_file().read_text())


def _worst(parts: list[tuple[str, str]]) -> tuple[str, str]:
    order = {"fail": 3, "warn": 2, "info": 1, "pass": 0}
    status = max((s for s, _ in parts), key=lambda s: order[s])
    return status, "; ".join(t for _, t in parts)


class PcChecks:
    def __init__(self, station_path: Path | str | None, mock: bool, local: bool, probes: Probes | None = None) -> None:
        self.station_path = Path(station_path) if station_path else paths.default_station()
        self.mock = mock
        self.local = local
        self.p = probes or Probes()
        self.station: Station | None = None
        self.link: LinkSettings | None = None
        self.frame: TcpFrame | None = None
        self.version: VersionInfo | None = None
        self.config: CheckConfigResult | None = None
        self.binary_path: Path | None = None

    def r(self, name: str, status: str, detail: str) -> CheckResult:
        return CheckResult(P, name, status, detail)

    # ---- the rows, in table order ----

    def run(self) -> list[CheckResult]:
        out = [self.env_row(), self.contracts_row(), self.station_row(), self.binary_row()]
        binary_ok = out[-1].status == "pass"
        if binary_ok:
            out.append(self.version_row())
            self.run_check_config()
            out.append(self.baked_paths_row())
            out.append(self.config_row())
        if self.frame is not None:
            out.append(self.tcp_frame_row())
        if not self.mock:
            out.append(self.todo_row())
        cfg_ok = self.config is not None and self.config.ok
        if not self.mock and cfg_ok:
            out.append(self.payload_row())
        if not self.local and self.link is not None:
            out.append(self.link_address_row())
        if self.link is not None:
            out.extend(self.port_rows())
        if not self.local and self.link is not None:
            out.append(self.firewall_row())
        if not self.mock:
            out.append(self.fci_network_row())
            if cfg_ok:
                out.append(self.arm_reachable_row())
                if self.config.effective("realtime") == "enforce":
                    out.extend([self.rt_kernel_row(), self.rtprio_row()])
                else:
                    out.append(self.r("realtime", "warn", "real-time ignore mode: expect occasional reflexes; late "
                                                          "cycles show on the status line"))
        if self.station is not None:
            out.append(self.session_root_row())
        out.append(self.backups_row())
        return out

    def env_row(self) -> CheckResult:
        try:
            have = Path(self.p.contracts_dir()).resolve()
        except FileNotFoundError as exc:
            return self.r("env", "fail", f"{exc}")
        want = Path(self.p.expected_contracts_dir()).resolve()
        if have != want:
            return self.r("env", "fail", f"deploy_contracts comes from {have}: uv pip install -e deployment/contracts "
                                         f"into ~/envs/control_deploy")
        return self.r("env", "pass", f"deploy_contracts from {have}")

    def contracts_row(self) -> CheckResult:
        d = Path(self.p.contracts_dir())
        try:
            ContractLimits.load(d / "limits.yaml")
            self.frame = TcpFrame.load(d / "frames.yaml")
            self.link = LinkSettings.load(d / "link.yaml")
        except (ValueError, OSError, KeyError) as exc:
            return self.r("contracts", "fail", str(exc))
        return self.r("contracts", "pass", "limits.yaml, frames.yaml, link.yaml load")

    def station_row(self) -> CheckResult:
        try:
            self.station = Station.load(self.station_path)
        except ValueError as exc:
            return self.r("station", "fail", str(exc))
        return self.r("station", "pass", f"{self.station.path}")

    def binary_row(self) -> CheckResult:
        try:
            self.binary_path = paths.binary(self.station.franka_control if self.station else None)
        except (OSError, KeyError) as exc:
            return self.r("binary", "fail", f"config.yaml binary: {exc}")
        res = check_helpers.executable(P, "binary", self.binary_path)
        if res.status != "pass":
            return self.r("binary", "fail", f"{res.detail}: build it: {BUILD}")
        return res

    def version_row(self) -> CheckResult:
        try:
            self.version = self.p.binary(self.binary_path).version()
        except BinaryError as exc:
            return self.r("version", "fail", f"{exc}: rebuild: {BUILD}")
        v = self.version
        want_lib = str(yaml_safe(self.p.config_text()).get("libfranka_version"))
        parts: list[tuple[str, str]] = [("pass", v.line)]
        if v.libfranka != want_lib:
            text = f"built with libfranka {v.libfranka}, config.yaml wants {want_lib}: rebuild: {BUILD}"
            parts.append(("info", text + " (fine for --mock)") if self.mock else ("fail", text))
        head = self.p.git_head()
        if v.commit == "unknown":
            parts.append(("warn", "commit=unknown (built without git): the laptop's hello check sees no commit"))
        elif head is not None and v.commit != head:
            parts.append(("fail", f"built from {v.commit}, checkout is {head}: rebuild after every pull: {BUILD}"))
        if v.dirty or self.p.git_dirty():
            parts.append(("warn", "uncommitted changes in contracts/ or control_deploy/ (built dirty or dirty now): "
                                  "the laptop's hello check sees only the commit"))
        status, detail = _worst(parts)
        return self.r("version", status, detail)

    def run_check_config(self) -> None:
        try:
            self.config = self.p.binary(self.binary_path).check_config(self.station_path, self.mock, self.local)
        except BinaryError as exc:
            self.config = CheckConfigResult(ok=False, problems=[str(exc)])

    def baked_paths_row(self) -> CheckResult:
        if not self.config.ok:
            return self.r("baked paths", "info", "not checked: --check-config failed (see config)")
        baked = self.config.contracts()
        here = Path(self.p.expected_contracts_dir()).resolve()
        want = {"contracts_dir": here, "limits_file": here / "limits.yaml", "frames_file": here / "frames.yaml",
                "link_file": here / "link.yaml"}
        wrong = [f"{k} {baked.get(k)}" for k, v in want.items()
                 if baked.get(k) is None or Path(str(baked[k])).resolve() != v]
        if wrong:
            return self.r("baked paths", "fail", f"the checkout moved since the build ({', '.join(wrong)}; checkout "
                                                 f"{here}): delete franka_control/build* and rerun build.sh")
        return self.r("baked paths", "pass", f"the binary reads {here}")

    def config_row(self) -> CheckResult:
        if self.config.ok:
            return self.r("config", "pass", "the binary accepts the merged config and the contract files")
        problems = self.config.problems
        more = f" ({len(problems)} problems)" if len(problems) > 1 else ""
        return self.r("config", "fail", problems[0] + more)

    def tcp_frame_row(self) -> CheckResult:
        if self.frame.settled:
            return self.r("tcp frame", "pass", "frames.yaml settled: true")
        return self.r("tcp frame", "info", "frames.yaml settled: false: TCP pending PLAN §9 (fine for hold still and "
                                           "console)")

    def todo_row(self) -> CheckResult:
        overrides = self.station.franka_control if self.station else {}
        left = [k for k in todo_keys(self.p.config_text()) if not _has_key(overrides, k)]
        if left:
            return self.r("todo", "warn", f"still shipped defaults: {', '.join(left)}: measure them (PLAN §12)")
        return self.r("todo", "pass", "no shipped default left (every TODO key is set in the station)")

    def payload_row(self) -> CheckResult:
        mass = float(self.config.effective("load", "mass_kg", default=0.0) or 0.0)
        if mass > 0.0:
            return self.r("payload", "pass", f"load.mass_kg {mass:g}")
        return self.r("payload", "warn", "no payload set: the rig's weight is not compensated (the arm sags a few mm)")

    def link_address_row(self) -> CheckResult:
        try:
            addresses = self.p.local_addresses()
        except (OSError, RuntimeError, ValueError) as exc:
            return self.r("link address", "fail", f"cannot list this PC's addresses: {exc}")
        address = self.link.pc_address
        for iface, addrs in addresses.items():
            if address in addrs:
                return self.r("link address", "pass", f"link.yaml pc.address {address} is on {iface}")
        others = ", ".join(f"{i} has {', '.join(a)}" for i, a in addresses.items() if i != "lo") or "no other address"
        return self.r("link address", "fail", f"link.yaml pc.address {address} is not on this PC ({others}): reserve "
                                              f"the address on ASUS_5g or update link.yaml")

    def _bind_host(self) -> str:
        if self.local:
            return "127.0.0.1"
        try:
            on_pc = any(self.link.pc_address in a for a in self.p.local_addresses().values())
        except (OSError, RuntimeError, ValueError):
            on_pc = False
        return self.link.pc_address if on_pc else "*"   # absent: whether the port is taken anywhere

    def port_rows(self) -> list[CheckResult]:
        host = self._bind_host()
        out = []
        for port in (self.link.state_port, self.link.commands_port):
            res = self.p.port_free(f"tcp://{host}:{port}")
            out.append(self.r("ports", res.status, res.detail))
        return out

    def firewall_row(self) -> CheckResult:
        live = self.p.read_firewall()
        ufw = "; ufw is enabled: also allow the laptop there" if self.p.ufw_enabled() else ""
        if live.state == "absent":
            return self.r("firewall", "fail", f"no table {firewall.TABLE} (print it with control_deploy firewall){ufw}")
        if live.state == "unreadable":
            return self.r("firewall", "warn", f"cannot read the firewall without root: compare sudo nft list table "
                                              f"{firewall.TABLE} with control_deploy firewall{ufw}")
        diffs = firewall.compare(live.rules, self.link)
        if diffs:
            return self.r("firewall", "fail", f"table {firewall.TABLE} differs from link.yaml: {'; '.join(diffs)} "
                                              f"(print it with control_deploy firewall){ufw}")
        if ufw:
            return self.r("firewall", "warn", f"table {firewall.TABLE} matches link.yaml{ufw}")
        return self.r("firewall", "pass", f"table {firewall.TABLE} matches link.yaml")

    def fci_network_row(self) -> CheckResult:
        try:
            addresses = self.p.local_addresses()
        except (OSError, RuntimeError, ValueError) as exc:
            return self.r("fci network", "fail", f"cannot list this PC's addresses: {exc}")
        for iface, addrs in addresses.items():
            for a in addrs:
                if ipaddress.ip_address(a) in FCI_NETWORK:
                    return self.r("fci network", "pass", f"{iface} has {a}")
        return self.r("fci network", "fail", "no 172.16.0.x address: sudo nmcli con up franka")

    def arm_reachable_row(self) -> CheckResult:
        robot_ip = str(self.config.effective("robot_ip"))
        reach = self.p.host_reachable(robot_ip, ARM_COMMAND_PORT, REACH_TIMEOUT_S)
        if reach.status == "fail":
            return self.r("arm reachable", "fail", f"arm not reachable at {robot_ip}:{ARM_COMMAND_PORT}: check the "
                                                   f"cable, the PC's static IP, FCI enabled in Desk")
        return self.r("arm reachable", "pass", reach.detail)

    def rt_kernel_row(self) -> CheckResult:
        if self.p.rt_flag() == "1":
            return self.r("rt kernel", "pass", "PREEMPT_RT kernel")
        return self.r("rt kernel", "fail", "no PREEMPT_RT kernel: install one, or set franka_control.realtime: ignore "
                                           "in control_deploy/station.yaml (backup, PLAN §13)")

    def rtprio_row(self) -> CheckResult:
        soft = self.p.rtprio_limit()
        if soft == resource.RLIM_INFINITY or soft >= RTPRIO_MIN:
            limit = "unlimited" if soft == resource.RLIM_INFINITY else str(soft)
            return self.r("rtprio", "pass", f"rtprio limit {limit}")
        return self.r("rtprio", "fail", f"rtprio limit {soft} < {RTPRIO_MIN}: add the user to the realtime group with "
                                        f"rtprio {RTPRIO_MIN} in /etc/security/limits.conf, log in again")

    def session_root_row(self) -> CheckResult:
        root = self.station.session_root
        probe = root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if probe.is_dir() and os.access(probe, os.W_OK | os.X_OK):
            return self.r("session root", "pass", f"{root} is writable")
        return self.r("session root", "fail", f"{root} is not writable (nearest existing folder {probe}): fix "
                                              f"session_root in {self.station_path.name}")

    def backups_row(self) -> CheckResult:
        binds = "127.0.0.1" if self.local else (self.link.pc_address if self.link else "?")
        if self.mock:
            return self.r("backups", "info", f"arm: FakeArm, gripper: FakeGripper (--mock); binds {binds}")
        robot_ip = self.config.effective("robot_ip", default="?") if self.config else "?"
        realtime = self.config.effective("realtime", default="?") if self.config else "?"
        return self.r("backups", "info", f"arm: LibfrankaArm at {robot_ip}, realtime {realtime}; binds {binds}")


def yaml_safe(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}
