from __future__ import annotations

import io
import json
import subprocess
import sys

from fakes import BLOCKER_PRELUDE
from policy.self_check import SelfCheck

STATUSES = {"pass", "warn", "fail", "info"}


def _run(station, mock: bool) -> list[dict]:
    out = io.StringIO()
    SelfCheck(station, mock, out).run()
    return [json.loads(line) for line in out.getvalue().splitlines()]


def _blocked(station, mock: bool = False) -> list[dict]:
    script = BLOCKER_PRELUDE + "\nfrom policy.self_check import main\nsys.exit(main(sys.argv[1:]))\n"
    args = [sys.executable, "-c", script, "--station", str(station)] + (["--mock"] if mock else [])
    proc = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines()]


def test_json_line_format_and_hold_still_passes(station_file):
    lines = _run(station_file({"kind": "hold_still"}), mock=False)
    assert all(set(item) == {"name", "status", "detail"} and item["status"] in STATUSES for item in lines)
    assert [(i["name"], i["status"]) for i in lines] == [("settings", "pass"), ("kind", "info")]


def test_mock_needs_nothing_else(station_file):
    lines = _run(station_file({"kind": "diffusion"}), mock=True)
    assert [i["status"] for i in lines] == ["pass", "info"] and "(--mock)" in lines[1]["detail"]


def test_settings_problems_fail(station_file):
    lines = _run(station_file({"kind": "hold_still", "chekpoint": 1}), mock=False)
    assert lines == [{"name": "settings", "status": "fail",
                      "detail": "station.yaml: policy.chekpoint: not a key of policy/config.yaml"}]


def test_diffusion_with_torch_blocked_fails_with_the_environment_hint(station_file, tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"x")
    lines = _blocked(station_file({"kind": "diffusion", "checkpoint": str(ckpt), "diffusion_steps": 8}))
    by_name = {i["name"]: i for i in lines}
    assert by_name["checkpoint"]["status"] == "pass"
    imports = by_name["imports"]
    assert imports["status"] == "fail" and "policy environment" in imports["detail"]
    assert "python.policy" in imports["detail"] and "torch" in imports["detail"]
    assert "device" not in by_name                                       # skipped without torch
    assert by_name["diffusion_steps"] == {"name": "diffusion_steps", "status": "info",
                                          "detail": "diffusion steps: 8, PLAN §9 backup"}


def test_missing_checkpoint_fails(station_file, tmp_path):
    lines = _blocked(station_file({"kind": "diffusion", "checkpoint": str(tmp_path / "missing.ckpt")}))
    by_name = {i["name"]: i for i in lines}
    assert by_name["checkpoint"]["status"] == "fail" and "checkpoint not found" in by_name["checkpoint"]["detail"]


def test_wrong_suffix_fails(station_file, tmp_path):
    (tmp_path / "best.pt").write_bytes(b"x")
    lines = _run(station_file({"kind": "diffusion", "checkpoint": str(tmp_path / "best.pt")}), mock=False)
    by_name = {i["name"]: i for i in lines}
    assert by_name["checkpoint"]["status"] == "fail" and ".ckpt" in by_name["checkpoint"]["detail"]
