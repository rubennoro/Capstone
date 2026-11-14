# adapted from rnoro/deployment:deployment/policy/tests/test_self_check.py @ 91b178b
# changes: module path, the new station; --json (and the readable form without it); + the gpu memory, gpu processes
# and precision rows of kind diffusion, with nvidia-smi stubbed; the environment hint names python.torch.
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest
from policy_fakes import BLOCKER_PRELUDE

from policy_deploy.policy import self_check
from policy_deploy.policy.self_check import SelfCheck, gpu_info, gpu_processes

STATUSES = {"pass", "warn", "fail", "info"}
TORCH_PY = {"light": sys.executable, "torch": sys.executable}


def no_smi(args):
    raise FileNotFoundError("nvidia-smi")


def smi(info="NVIDIA GeForce RTX 3050 Ti Laptop GPU, 7, 4096, [N/A]", apps=""):
    def run(args):
        return info + "\n" if args[0].startswith("--query-gpu") else apps
    return run


def _run(station, mock: bool, nvidia_smi=no_smi) -> list[dict]:
    out = io.StringIO()
    SelfCheck(station, mock, out, as_json=True, nvidia_smi=nvidia_smi).run()
    return [json.loads(line) for line in out.getvalue().splitlines()]


def _blocked(station, mock: bool = False) -> list[dict]:
    script = BLOCKER_PRELUDE + "\nfrom policy_deploy.policy.self_check import main\nsys.exit(main(sys.argv[1:]))\n"
    args = [sys.executable, "-c", script, "--json", "--station", str(station)] + (["--mock"] if mock else [])
    proc = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines()]


def test_json_line_format_and_hold_still_passes(policy_station):
    lines = _run(policy_station({"kind": "hold_still"}), mock=False)
    assert all(set(item) == {"name", "status", "detail"} and item["status"] in STATUSES for item in lines)
    assert [(i["name"], i["status"]) for i in lines] == [("settings", "pass"), ("kind", "info")]


def test_mock_needs_nothing_else(policy_station):
    lines = _run(policy_station({"kind": "diffusion"}), mock=True)
    assert [i["status"] for i in lines] == ["pass", "info"] and "(--mock)" in lines[1]["detail"]


def test_settings_problems_fail(policy_station):
    lines = _run(policy_station({"kind": "hold_still", "chekpoint": 1}), mock=False)
    assert lines == [{"name": "settings", "status": "fail",
                      "detail": "station.yaml: policy.chekpoint: not a key of config/policy.yaml"}]


def test_readable_form_without_json(policy_station, capsys):
    assert self_check.main(["--station", str(policy_station({"kind": "hold_still"}))]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("PASS  settings: ") and out[1].startswith("INFO  kind: policy kind: hold_still")


def test_diffusion_with_torch_blocked_fails_with_the_environment_hint(policy_station, tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"x")
    lines = _blocked(policy_station({"kind": "diffusion", "checkpoint": str(ckpt), "diffusion_steps": 8},
                                    python=TORCH_PY))
    by_name = {i["name"]: i for i in lines}
    assert by_name["checkpoint"]["status"] == "pass"
    imports = by_name["imports"]
    assert imports["status"] == "fail" and "torch environment" in imports["detail"]
    assert "python.torch" in imports["detail"] and "torch" in imports["detail"]
    assert "device" not in by_name                                       # skipped without torch
    assert by_name["precision"] == {"name": "precision", "status": "info", "detail": "precision: fp32"}
    assert by_name["diffusion_steps"] == {"name": "diffusion_steps", "status": "info",
                                          "detail": "diffusion steps: 8, PLAN 13 third backup"}
    assert by_name["gpu memory"]["status"] == "info" and "gpu processes" in by_name


def test_missing_checkpoint_fails(policy_station, tmp_path):
    lines = _blocked(policy_station({"kind": "diffusion", "checkpoint": str(tmp_path / "missing.ckpt")},
                                    python=TORCH_PY))
    by_name = {i["name"]: i for i in lines}
    assert by_name["checkpoint"]["status"] == "fail" and "checkpoint not found" in by_name["checkpoint"]["detail"]


def test_wrong_suffix_fails(policy_station, tmp_path):
    (tmp_path / "best.pt").write_bytes(b"x")
    lines = _run(policy_station({"kind": "diffusion", "checkpoint": str(tmp_path / "best.pt")}, python=TORCH_PY),
                 mock=False)
    by_name = {i["name"]: i for i in lines}
    assert by_name["checkpoint"]["status"] == "fail" and ".ckpt" in by_name["checkpoint"]["detail"]


def _diffusion_rows(policy_station, tmp_path, nvidia_smi, **section):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"x")
    rows = _run(policy_station({"kind": "diffusion", "checkpoint": str(ckpt), **section}, python=TORCH_PY),
                mock=False, nvidia_smi=nvidia_smi)
    return {i["name"]: i for i in rows}


def test_gpu_rows_free_memory_and_no_other_process(policy_station, tmp_path):
    by_name = _diffusion_rows(policy_station, tmp_path, smi())
    assert by_name["gpu memory"]["status"] == "info"
    assert by_name["gpu memory"]["detail"] == ("NVIDIA GeForce RTX 3050 Ti Laptop GPU: 4089 MiB free of 4096 MiB "
                                               "(7 MiB used); power limit n/a")
    assert by_name["gpu processes"] == {"name": "gpu processes", "status": "pass",
                                        "detail": "no other process on the GPU"}


def test_gpu_rows_warn_above_300_mb_used_by_others(policy_station, tmp_path):
    apps = "1234, /usr/bin/firefox, 250\n5678, python, 120\n"
    by_name = _diffusion_rows(policy_station, tmp_path, smi(apps=apps))
    row = by_name["gpu processes"]
    assert row["status"] == "warn" and "2 other process(es) use 370 MiB" in row["detail"]
    assert "firefox (pid 1234, 250 MiB)" in row["detail"]
    by_name = _diffusion_rows(policy_station, tmp_path, smi(apps="1234, firefox, 250\n"))
    assert by_name["gpu processes"]["status"] == "pass"


def test_own_pid_is_not_another_process(policy_station, tmp_path):
    by_name = _diffusion_rows(policy_station, tmp_path, smi(apps=f"{os.getpid()}, python, 900\n"))
    assert by_name["gpu processes"]["status"] == "pass"


def test_gpu_rows_without_nvidia_smi_and_on_cpu(policy_station, tmp_path):
    by_name = _diffusion_rows(policy_station, tmp_path, no_smi)
    assert by_name["gpu memory"]["status"] == "info" and "not available" in by_name["gpu memory"]["detail"]
    assert by_name["gpu processes"]["status"] == "info"
    by_name = _diffusion_rows(policy_station, tmp_path, smi(), device="cpu", precision="bf16")
    assert "gpu memory" not in by_name and "gpu processes" not in by_name
    assert by_name["precision"]["detail"].startswith("precision: bf16 (autocast")


@pytest.mark.parametrize("text,want", [
    ("NVIDIA X, 100, 4096, 35.00", {"name": "NVIDIA X", "memory_used_mb": 100.0, "memory_total_mb": 4096.0,
                                    "power_limit_w": 35.0}),
    ("NVIDIA X, [N/A], 4096, [N/A]", {"name": "NVIDIA X", "memory_used_mb": None, "memory_total_mb": 4096.0,
                                      "power_limit_w": None}),
    ("", None),
])
def test_gpu_info_parsing(text, want):
    assert gpu_info(lambda args: text) == want


def test_gpu_processes_parsing():
    assert gpu_processes(lambda args: "12, a, 10\nbad line\n13, b, [N/A]\n") == [
        {"pid": 12, "name": "a", "used_mb": 10.0}, {"pid": 13, "name": "b", "used_mb": None}]
    assert gpu_processes(no_smi) is None
