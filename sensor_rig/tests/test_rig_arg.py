"""Every subcommand that takes a rig file reports a missing / unparseable one the same way: one message, exit 2."""

from __future__ import annotations

import pytest

from sensor_rig.cli import main


def _argv(cmd, rig, tmp):
    out = str(tmp / "out")
    return {
        "collect": ["collect", "--mock", "--rig", rig, "--out", out, "--session", "rig1020"],
        "record": ["record", "--mock", "--rig", rig, "--out", out, "--session", "rig1020", "--side", "left",
                   "--duration", "1"],
        "stamp-offset": ["stamp-offset", "--mock", "--rig", rig, "--out", out],
        "process-session": ["process-session", "--raw", out, "--session", "rig1020", "--out", out, "--vio",
                            "import-dir", "--traj-dir", out, "--rig", rig],
        "export": ["export", "--raw", out, "--out", str(tmp / "exp"), "--rig", rig],
        "pose-prep": ["pose-prep", "--raw", out, "--out", out, "--rig", rig],
        "pose-import": ["pose-import", "--raw", out, "--traj", out, "--rig", rig],
        "sim": ["sim", "--iphone", out, "--out", out, "--rig", rig],
        "calib-import": ["calib-import", "--rig", rig, "--camchain", out, "--out", out],
        "calib-check": ["calib-check", "--rig", rig, "--images", "a.png", "b.png", "--out", out],
        "calib-align": ["calib-align", "--rig", rig, "--iphone-calib", out, "--rig-calib", out],
        "calib-tcp": ["calib-tcp", "--rig", rig, "--tcp-in-main-cv", "0", "0", "0.1"],
        "bench-remap": ["bench-remap", "--rig", rig],
        "rig-validate": ["rig-validate", rig],
    }[cmd]


CMDS = ["collect", "record", "stamp-offset", "process-session", "export", "pose-prep", "pose-import", "sim",
        "calib-import", "calib-check", "calib-align", "calib-tcp", "bench-remap", "rig-validate"]


@pytest.mark.parametrize("cmd", CMDS)
@pytest.mark.parametrize("kind", ["missing", "unparseable"])
def test_bad_rig_file_one_message_exit_2(cmd, kind, tmp_path, capsys):
    if kind == "missing":
        rig = str(tmp_path / "nonexistent.yaml")
    else:
        p = tmp_path / "broken.yaml"
        p.write_text("cameras: [unclosed\n  - : :\n")
        rig = str(p)
    if cmd == "rig-validate" and kind == "unparseable":
        pytest.skip("rig-validate is the validator: it reports invalid content itself (exit 1, existing contract)")
    rc = main(_argv(cmd, rig, tmp_path))
    err = capsys.readouterr().err
    assert rc == 2, err
    assert f"rig file not found/invalid: {rig}" in err and "sensor-rig rig-validate" in err or cmd == "rig-validate"
    assert "Traceback" not in err
    if kind == "missing":
        assert "file not found" in err


def test_rig_validate_invalid_content_still_exit_1(tmp_path, capsys):
    p = tmp_path / "broken.yaml"
    p.write_text("cameras: [unclosed\n")
    assert main(["rig-validate", str(p)]) == 1
    assert "ERROR" in capsys.readouterr().err
