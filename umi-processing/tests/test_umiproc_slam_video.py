"""``umiproc slam-video``: a video of a map's run folders (one video frame per fed frame), the recording found from the
map's report, the ``--first-map`` cut marked on the whole-clip run, and the input errors (exit 2)."""

import json
import shutil

import cv2
import pytest

from umiproc import cli as umicli
from umiproc.map import build as B
from umiproc.viz import slam_video as SV
from test_umiproc_map_command import _map, _pass_m4_m5, _two_piece_fake, env, env40  # noqa: F401 - fixtures


def _frames(path):
    c = cv2.VideoCapture(str(path))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, img = c.read()
    c.release()
    return n, img


@pytest.fixture
def first_map_folder(env40, monkeypatch):
    _pass_m4_m5(monkeypatch)
    o, _ = _map(env40, name="episode_000002", runner=_two_piece_fake(), first_map=True)
    assert o.passed
    return o, env40


def test_video_of_each_run_with_the_recording_from_the_report(first_map_folder, capsys):
    o, env = first_map_folder
    fm = o.report["first_map"]
    for name, n in ((B.MAPPING_FULL_DIR, fm["frames_full"]), (B.MAPPING_DIR, fm["frames"]),
                    (B.RELOCALIZE_DIR, fm["frames"])):
        assert umicli.main(["slam-video", str(o.where / name)]) == 0
        out = o.where / name / SV.VIDEO_NAME
        frames, img = _frames(out)
        assert frames == n and img.shape == (SV.H_HDR + SV.H_IMG + SV.H_TL, SV.W_IMG + SV.W_MAP, 3)
    text = capsys.readouterr().out
    assert f"--first-map cut after fed frame {fm['frames'] - 1}" in text      # only on mapping_full
    assert text.count("--first-map cut") == 1


def test_render_counts_and_explicit_recording(first_map_folder, tmp_path):
    o, env = first_map_folder
    out = tmp_path / "v.mp4"
    st = SV.render(o.where / B.MAPPING_FULL_DIR, env["src"] / "episode_000002", out, speed=2.0)
    fm = o.report["first_map"]
    assert st["frames"] == fm["frames_full"] and st["fps"] == 60.0 and st["cut"] == fm["frames"] - 1
    assert st["saved_map"] + st["later_map"] + st["not_tracking"] == st["frames"]
    assert st["new_maps"] == 1                                # the fake's state 0 at frame 24
    assert _frames(out)[0] == st["frames"]


def test_input_errors(first_map_folder, tmp_path, capsys):
    o, env = first_map_folder
    run = o.where / B.MAPPING_DIR
    assert umicli.main(["slam-video", str(tmp_path / "nope")]) == 2
    lone = tmp_path / "lone"
    shutil.copytree(run, lone)                               # no report.json next to it
    assert umicli.main(["slam-video", str(lone)]) == 2
    assert "pass --recording" in capsys.readouterr().err
    assert umicli.main(["slam-video", str(lone), "--recording", str(tmp_path)]) == 2     # no meta.json
    assert "ultrawide camera" in capsys.readouterr().err
    rep = json.loads((o.where / B.REPORT_NAME).read_text())
    rep["source"] = str(tmp_path / "gone")
    (o.where / B.REPORT_NAME).write_text(json.dumps(rep))
    assert umicli.main(["slam-video", str(run)]) == 2
    assert "is not there any more" in capsys.readouterr().err
    assert umicli.main(["slam-video", str(run), "--recording", str(env["src"] / "episode_000002"), "--speed",
                        "0"]) == 2
