"""CLI: --help and bare import work without mujoco; check-assets is wired."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "franka_sim", *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=60)


def test_help_exits_zero():
    r = _run("--help")
    assert r.returncode == 0, r.stderr
    assert "check-assets" in r.stdout


def test_import_does_not_pull_mujoco():
    code = "import sys, franka_sim, franka_sim.cli, franka_sim.__main__; assert 'mujoco' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_check_assets_command():
    r = _run("check-assets")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_register_and_dispatch():
    from franka_sim import cli

    seen = []
    cli.register("zz-test", lambda p: p.add_argument("--x", type=int, default=3),
                 lambda a: seen.append(a.x) or 0, help="t")
    assert cli.main(["zz-test", "--x", "5"]) == 0
    assert seen == [5]
    assert cli.main([]) == 2
