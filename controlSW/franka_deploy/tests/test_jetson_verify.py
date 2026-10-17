"""jetson_verify: the pure pass/fail checks on captured Jetson output."""
import resource

from franka_deploy.tools import jetson_verify as jv

REL = "# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025\n"


def test_l4t_and_rt():
    assert jv.check_l4t(REL) == (True, "L4T R36.4.3")
    assert not jv.check_l4t(REL.replace("R36", "R35"))[0] and not jv.check_l4t("")[0]
    assert jv.check_rt("#1 SMP PREEMPT_RT Tue Jan  7 2025", "")[0]
    assert jv.check_rt("#1 SMP PREEMPT", "1\n")[0]
    assert not jv.check_rt("#1 SMP PREEMPT Tue", "")[0]


def test_cmdline():
    base = "root=/dev/nvme0n1p1 rw rootwait console=ttyTCU0,115200 "
    assert jv.check_cmdline(base + "isolcpus=5 nohz_full=5 rcu_nocbs=5", 5)[0]
    assert jv.check_cmdline(base + "isolcpus=managed_irq,domain,5 nohz_full=5 rcu_nocbs=5", 5)[0]
    ok, d = jv.check_cmdline(base + "isolcpus=5 nohz_full=5", 5)
    assert not ok and "rcu_nocbs=5" in d
    assert not jv.check_cmdline(base + "isolcpus=4 nohz_full=5 rcu_nocbs=5", 5)[0]


def test_irqs():
    inter = ("           CPU0       CPU1\n 220:     123     0   GICv3 226 Level     2310000.ethernet.common_irq\n"
             " 221:  9999       0   GICv3 227 Level     eth0\n  13:   5   0  GICv3  arch_timer\n")
    pat = r"eth\d|ethernet"
    assert jv.check_irqs(inter, {"220": "5", "221": "5"}, 5, pat)[0]
    ok, d = jv.check_irqs(inter, {"220": "5", "221": "0-5"}, 5, pat)
    assert not ok and "221" in d
    assert not jv.check_irqs("  13: 5 0 arch_timer\n", {}, 5)[0]


def test_limits_power_ip_codecs():
    assert jv.check_limits(99, resource.RLIM_INFINITY)[0]
    assert not jv.check_limits(0, 65536)[0]
    assert jv.check_nvpmodel("NV Power Mode: MAXN_SUPER\n2\n")[0]
    assert not jv.check_nvpmodel("NV Power Mode: 15W\n0\n")[0] and not jv.check_nvpmodel("")[0]
    ip = "2: eth0: <BROADCAST,MULTICAST,UP>\n    inet 172.16.0.1/24 brd 172.16.0.255 scope global eth0\n"
    assert jv.check_ip(ip, "172.16.0.1")[0] and not jv.check_ip(ip, "172.16.0.10")[0]
    assert jv.check_imagecodecs("2023.9.18")[0] and not jv.check_imagecodecs("2025.3.30")[0]
    assert not jv.check_imagecodecs(None)[0]
