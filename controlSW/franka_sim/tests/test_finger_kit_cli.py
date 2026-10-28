"""A9d finger-kit CLI: check / pin / show / report on a make_box_kit kit; check-assets verifies assets/finger_kits.

Run: cd controlSW/franka_sim && MUJOCO_GL=egl env -u PYTHONPATH $RE -m pytest -q tests/test_finger_kit_cli.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
pytest.importorskip("mujoco")

from franka_sim import cli, finger_kit as FK, finger_kit_cli as CLI  # noqa: E402
from franka_sim.tools import check_assets  # noqa: E402


def _run(*argv: str) -> int:
    from franka_sim.__main__ import main

    return main(list(argv))


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    return FK.make_box_kit(tmp_path_factory.mktemp("kit") / "box")


def _kit_sets(kit_dir: Path, mode="kit"):
    return ["--set", f"tooling.finger_adapter.mode={mode}", "--set", f"tooling.finger_adapter.kit={kit_dir}",
            "--set", f'tooling.finger_adapter.kit_sha256="{FK.kit_digest(kit_dir)}"',
            "--set", "tooling.finger_adapter.adapter_length_m=0.02"]


def test_registered():
    assert _run() == 2 or "finger-kit" in cli.registered()
    assert "finger-kit" in cli.registered()


def test_check_prints_kit(kit, capsys):
    assert _run("finger-kit", "check", str(kit)) == 0
    out = capsys.readouterr().out
    assert "kit box_equivalent" in out and "role=contact" in out and "bbox_mm" in out and "volume_cm3" in out
    assert "camera_face=+1" in out and "camera_face=-1" in out and "tip_mm" in out and "tag id=0" in out
    assert f"digest: {FK.kit_digest(kit)}" in out


def test_check_fails_cleanly(tmp_path, kit, capsys):
    bad = tmp_path / "bad"
    shutil.copytree(kit, bad)
    (bad / "kit.yaml").write_text("schema: finger_kit/v1\nid: x\n")
    assert _run("finger-kit", "check", str(bad)) == 1
    cap = capsys.readouterr()
    assert "FAILED" in cap.err and "Traceback" not in cap.err
    assert _run("finger-kit", "check", str(tmp_path / "missing")) == 1


def test_pin_writes_source_and_prints_sha(tmp_path, kit, capsys):
    d = tmp_path / "k"
    shutil.copytree(kit, d)
    (d / "SOURCE.md").unlink()
    assert _run("finger-kit", "pin", str(d)) == 0
    assert f"kit_sha256: {FK.kit_digest(d)}" in capsys.readouterr().out
    assert (d / "SOURCE.md").is_file()


def _report(tmp_path, *extra):
    out = tmp_path / "rep"
    assert _run("finger-kit", "report", "--profile", "sim-iphone", "--out", str(out), *extra) == 0
    return json.loads((out / "finger_report.json").read_text()), out


def test_report_box(tmp_path):
    rep, out = _report(tmp_path)
    n = rep["numbers"]
    assert rep["schema"] == "finger_report/v1" and rep["mode"] == "box" and rep["finger_kit"] is None
    for k in ("mode", "extension_m", "pad_length_m", "pad_offset_m", "flange_to_tcp_m", "flange_to_tip_m",
              "tag_separation_closed_m", "tcp_at", "solref", "solimp", "F_T_TCP_xyz_mm", "C_T_TCP_xyz_mm",
              "F_T_cam_main_xyz_mm", "F_T_cam_uw_rpy_deg"):
        assert k in n, k
        assert set(n[k]) == {"value", "unit", "status", "source"}
    assert n["tcp_at"]["value"] == "pad_centre" and "kit_id" not in n
    assert n["flange_to_tip_m"]["value"] == pytest.approx(n["flange_to_tcp_m"]["value"] + n["pad_length_m"]["value"] / 2)
    assert n["tag_separation_closed_m"]["value"] > 0.04
    assert n["C_T_TCP_xyz_mm"]["status"] == "measured"
    md = (out / "finger_report.md").read_text()
    assert "| number | value |" in md and "| tcp_at | pad_centre |" in md


def test_report_stock(tmp_path):
    rep, _ = _report(tmp_path, "--set", "tooling.finger_adapter.mode=stock")
    n = rep["numbers"]
    assert rep["mode"] == "stock" and n["tcp_at"]["value"] == "stock_pads"
    assert n["pad_length_m"]["value"] > 0 and "tag_separation_closed_m" not in n


def test_report_kit(tmp_path, kit):
    rep, out = _report(tmp_path, *_kit_sets(kit))
    n = rep["numbers"]
    assert rep["mode"] == "kit" and rep["finger_kit"]["id"] == "box_equivalent"
    assert n["kit_sha256"]["value"] == FK.kit_digest(kit) and n["kit_variant"]["value"] == "full"
    assert n["tcp_at"]["value"] == "tip" and n["multiccd"]["value"].startswith("on")
    assert n["adapter_length_m"]["value"] == 0.02 and n["adapter_length_m"]["status"] == "placeholder"
    assert n["dy_l_m"]["status"] == "derived (model)"                 # lateral_offset_m auto
    assert n["flange_to_tip_m"]["value"] == pytest.approx(rep["finger_kit"]["flange_to_tip_m"])
    assert n["mass_pad_l_kg"]["value"] > 0 and n["tag_separation_closed_m"]["value"] > 0.04
    assert "| kit_id | box_equivalent |" in (out / "finger_report.md").read_text()


def test_show_writes_pngs(tmp_path, kit):
    out = tmp_path / "show"
    assert _run("finger-kit", "show", "--profile", "sim-iphone", "--out", str(out), *_kit_sets(kit)) == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == sorted(f"{v}_{g}.png" for v in (*CLI.VIEWS, "main", "uw") for g in CLI.GAPS)
    assert all((out / p).stat().st_size > 1000 for p in pngs)
    txt = (out / "show.txt").read_text()
    assert "frame M_l" in txt and "frame tcp" in txt and "frame flange" in txt


def _assets(tmp_path, kit) -> Path:
    a = tmp_path / "assets"
    (a / "finger_kits").mkdir(parents=True)
    (a / "x.txt").write_text("hi\n")
    (a / "SOURCE.md").write_text(f"| x.txt | 3 | {check_assets.sha256_file(a / 'x.txt')} |\n")
    shutil.copytree(kit, a / "finger_kits" / "box")
    return a


def test_check_assets_kits(tmp_path, kit):
    a = _assets(tmp_path, kit)
    n, problems = check_assets.check(a)
    assert problems == [] and n == 1
    assert check_assets.check_kits(a)[0] == 1
    # tampered mesh
    stl = a / "finger_kits" / "box" / "pad.stl"
    stl.write_text(stl.read_text() + " ")
    _, problems = check_assets.check(a)
    assert any("finger_kits/box" in p and "mismatch: pad.stl" in p for p in problems), problems
    assert check_assets.run(type("A", (), {"assets": str(a)})()) == 1


def test_check_assets_kit_needs_source_and_no_extras(tmp_path, kit):
    a = _assets(tmp_path, kit)
    (a / "finger_kits" / "box" / "SOURCE.md").unlink()
    assert any("missing SOURCE.md" in p for p in check_assets.check(a)[1])
    CLI.cmd_pin(type("A", (), {"dir": str(a / "finger_kits" / "box")})())
    assert check_assets.check(a)[1] == []
    (a / "finger_kits" / "box" / "extra.bin").write_bytes(b"1")
    assert any("extra file" in p for p in check_assets.check(a)[1])
