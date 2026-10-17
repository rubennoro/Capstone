"""Pass/fail check of a Jetson set up by scripts/jetson_deploy_setup.sh (run after the reboot, as the deploy user).

Checks: L4T R36, PREEMPT_RT kernel, kernel cmdline (isolcpus/nohz_full/rcu_nocbs on the RT core), Franka NIC IRQs on
the RT core, nvpmodel MAXN, rtprio/memlock limits, static IP on the Franka link (+ robot ping), Argus on both
cameras (10 frames each through nvarguscamerasrc), CUDA + torch (a small GPU matmul), imagecodecs == 2023.9.18 (the
training pin; prompt decoding differs otherwise).

    python -m franka_deploy.tools.jetson_verify [--rt-core 5] [--robot-ip 172.16.0.2] [--jetson-ip 172.16.0.1]
        [--cameras 0,1] [--json out.json] [--dry-run]
Exit 0 only if every check passes. --dry-run lists the checks and the commands they use, runs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
from pathlib import Path

IMAGECODECS_PIN = "2023.9.18"


def _read(p) -> str:
    try:
        return Path(p).read_text()
    except OSError:
        return ""


def _sh(cmd, timeout=20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


# ---- pure checks (unit-tested with captured text) -------------------------------------------------------------------
def check_l4t(text: str):
    m = re.search(r"# R(\d+) \(release\), REVISION: ([\d.]+)", text)
    if not m:
        return False, "no /etc/nv_tegra_release (not a Jetson)"
    return m.group(1) == "36", f"L4T R{m.group(1)}.{m.group(2)}" + ("" if m.group(1) == "36" else " (need R36 / JetPack 6)")


def check_rt(uname_v: str, realtime_flag: str):
    ok = "PREEMPT_RT" in uname_v or realtime_flag.strip() == "1"
    return ok, uname_v.strip()[:80]


def check_cmdline(cmdline: str, core: int):
    want = [f"isolcpus={core}", f"nohz_full={core}", f"rcu_nocbs={core}"]
    toks = cmdline.split()
    missing = [w for w in want if not any(t == w or (t.startswith(w.split("=")[0] + "=") and t.split("=")[1].split(",")[-1] == str(core)) for t in toks)]
    return not missing, "ok" if not missing else f"missing {missing}"


def check_irqs(interrupts: str, affinities: dict, core: int, pattern=r"eth\d|eqos|nvethernet|mgbe|enP|r8168|r8169"):
    irqs = [ln.split(":")[0].strip() for ln in interrupts.splitlines() if re.search(pattern, ln) and ":" in ln]
    if not irqs:
        return False, "no NIC IRQ found in /proc/interrupts"
    bad = {i: affinities.get(i, "?") for i in irqs if affinities.get(i, "").strip() != str(core)}
    return not bad, f"IRQs {irqs}" + (f"; not on core {core}: {bad}" if bad else f" on core {core}")


def check_limits(rtprio, memlock):
    inf = resource.RLIM_INFINITY
    ok = rtprio >= 80 and (memlock == inf or memlock >= 1 << 30)
    return ok, f"rtprio {rtprio}, memlock {'unlimited' if memlock == inf else memlock}"


def check_nvpmodel(q: str):
    m = re.search(r"NV Power Mode:\s*(\S+)", q)
    if not m:
        return False, "nvpmodel not available"
    return m.group(1).upper().startswith("MAXN"), f"mode {m.group(1)}"


def check_ip(ip_addr_out: str, jetson_ip: str):
    return (f"inet {jetson_ip}/" in ip_addr_out), (f"{jetson_ip} configured" if f"inet {jetson_ip}/" in ip_addr_out
                                                   else f"{jetson_ip} not on any interface")


def check_imagecodecs(version: str | None):
    return version == IMAGECODECS_PIN, f"imagecodecs {version} (pin {IMAGECODECS_PIN})"


# ---- probes ---------------------------------------------------------------------------------------------------------
def probe_camera(sensor_id: int):
    if not shutil.which("gst-launch-1.0"):
        return False, "gst-launch-1.0 missing"
    rc, out = _sh(["gst-launch-1.0", "-q", "nvarguscamerasrc", f"sensor-id={sensor_id}", "num-buffers=10", "!",
                   "video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1", "!", "fakesink"], timeout=20)
    return rc == 0, "10 frames" if rc == 0 else out.strip().splitlines()[-1][:120] if out.strip() else f"rc {rc}"


def probe_torch():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return False, f"import torch failed: {e}"
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__}: CUDA not available"
    a = torch.randn(256, 256, device="cuda")
    v = float((a @ a).sum().item())
    return v == v, f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


def probe_imagecodecs():
    try:
        import imagecodecs

        return imagecodecs.__version__
    except Exception:  # noqa: BLE001
        return None


CHECKS = [
    ("l4t", "L4T R36 (JetPack 6)", "cat /etc/nv_tegra_release"),
    ("preempt_rt", "PREEMPT_RT kernel", "uname -v; cat /sys/kernel/realtime"),
    ("cmdline", "isolcpus/nohz_full/rcu_nocbs on the RT core", "cat /proc/cmdline"),
    ("nic_irq", "Franka NIC IRQs on the RT core", "/proc/interrupts + /proc/irq/N/smp_affinity_list"),
    ("nvpmodel", "nvpmodel MAXN", "nvpmodel -q"),
    ("limits", "rtprio >= 80, memlock unlimited", "getrlimit(RLIMIT_RTPRIO / RLIMIT_MEMLOCK)"),
    ("static_ip", "static IP on the Franka link", "ip -4 addr"),
    ("robot_ping", "robot reachable", "ping -c 3 -W 1 <robot-ip>"),
    ("camera_N", "Argus frames on each camera", "gst-launch-1.0 nvarguscamerasrc sensor-id=N num-buffers=10 ! fakesink"),
    ("cuda_torch", "CUDA + torch", "import torch; GPU matmul"),
    ("imagecodecs", f"imagecodecs == {IMAGECODECS_PIN}", "import imagecodecs"),
]


def run_all(a) -> list[dict]:
    res = []

    def add(name, ok_detail):
        res.append({"check": name, "pass": bool(ok_detail[0]), "detail": ok_detail[1]})

    add("l4t", check_l4t(_read("/etc/nv_tegra_release")))
    add("preempt_rt", check_rt(_sh(["uname", "-v"])[1], _read("/sys/kernel/realtime")))
    add("cmdline", check_cmdline(_read("/proc/cmdline"), a.rt_core))
    inter = _read("/proc/interrupts")
    aff = {}
    for ln in inter.splitlines():
        i = ln.split(":")[0].strip()
        if i.isdigit():
            aff[i] = _read(f"/proc/irq/{i}/smp_affinity_list").strip()
    add("nic_irq", check_irqs(inter, aff, a.rt_core))
    add("nvpmodel", check_nvpmodel(_sh(["nvpmodel", "-q"])[1] if shutil.which("nvpmodel") else ""))
    add("limits", check_limits(resource.getrlimit(resource.RLIMIT_RTPRIO)[1], resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]))
    add("static_ip", check_ip(_sh(["ip", "-4", "addr"])[1], a.jetson_ip))
    rc, _ = _sh(["ping", "-c", "3", "-W", "1", a.robot_ip], timeout=10)
    add("robot_ping", (rc == 0, a.robot_ip + (" reachable" if rc == 0 else " unreachable")))
    for c in a.cameras:
        add(f"camera_{c}", probe_camera(c))
    add("cuda_torch", probe_torch())
    add("imagecodecs", check_imagecodecs(probe_imagecodecs()))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt-core", type=int, default=5)
    ap.add_argument("--robot-ip", default="172.16.0.2")
    ap.add_argument("--jetson-ip", default="172.16.0.1")
    ap.add_argument("--cameras", default="0,1", type=lambda s: [int(x) for x in s.split(",") if x != ""])
    ap.add_argument("--json", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run:
        for name, what, how in CHECKS:
            print(f"{name:12s} {what:45s} via: {how}")
        return 0
    res = run_all(a)
    for r in res:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:12s} {r['detail']}")
    n_fail = sum(not r["pass"] for r in res)
    print(f"\n{len(res) - n_fail}/{len(res)} passed")
    if a.json:
        Path(a.json).write_text(json.dumps({"host": os.uname().nodename, "checks": res}, indent=1))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
