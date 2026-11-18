"""``umiproc slam-video`` and ``umiproc slam-3d`` by task folder (into ``tasks/<folder>/analysis/``): a video and the
plot-run figure of a map's runs (one video frame per fed frame, the frames from the map's ``raw/`` copy), the
``--first-map`` cut marked on the whole-clip run, the 3-D page from the stored map points, the RUN_DIR fallback (a
failed map in ``maps_failed/``), and the input errors (exit 2)."""

import json
import shutil

import cv2
import pytest

from umiproc import cli as umicli
from umiproc.map import build as B
from umiproc.viz import slam_3d as S3
from umiproc.viz import slam_video as SV
from umiproc.viz import target as T
from umiproc_testlib import FakeSlam, fake_slam_build
from test_umiproc_map_command import TWO_MAPS, _map, _pass_m4_m5, _two_piece_fake, env, env40  # noqa: F401 - fixtures


def _frames(path):
    c = cv2.VideoCapture(str(path))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, img = c.read()
    c.release()
    return n, img


def _page_data(path):
    html = path.read_text()
    start = html.index("const D = ") + len("const D = ")
    return json.loads(html[start:html.index(";\n", start)])


@pytest.fixture
def first_map_folder(env40, monkeypatch):
    _pass_m4_m5(monkeypatch)
    o, _ = _map(env40, name="episode_000002", runner=_two_piece_fake(), first_map=True)
    assert o.passed and o.report["first_map"]["applied"]
    return o, env40


def _root(env):
    return ["--root", str(env["root"].path)]


def test_video_and_figure_of_each_run_by_folder_name(first_map_folder, capsys):
    o, env = first_map_folder
    pytest.importorskip("matplotlib")
    shutil.rmtree(env["src"] / "episode_000002")             # the Pi folder is gone: the map's raw/ copy is used
    assert umicli.main(["slam-video", o.folder, "--run", "all"] + _root(env)) == 0
    an = env["root"].tasks / o.folder / T.ANALYSIS_DIR
    fm = o.report["first_map"]
    for name, n in ((B.MAPPING_FULL_DIR, fm["frames_full"]), (B.MAPPING_DIR, fm["frames"]),
                    (B.RELOCALIZE_DIR, fm["frames"])):
        frames, img = _frames(an / SV.video_name(name))
        assert frames == n and img.shape == (SV.H_HDR + SV.H_IMG + SV.H_TL, SV.W_IMG + SV.W_MAP, 3)
        assert (an / SV.figure_name(name)).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    text = capsys.readouterr().out
    assert f"cut after fed frame {fm['frames'] - 1}" in text and text.count("cut after fed frame") == 1
    assert f"wrote {an / SV.video_name(B.MAPPING_DIR)}" in text and f"wrote {an / SV.figure_name(B.MAPPING_DIR)}" in text
    assert str(o.where / "raw") in text
    # the default run is mapping only
    for p in an.iterdir():
        p.unlink()
    assert umicli.main(["slam-video", o.folder] + _root(env)) == 0
    assert sorted(p.name for p in an.iterdir()) == sorted([SV.video_name("mapping"), SV.figure_name("mapping")])


def test_render_counts_and_explicit_recording(first_map_folder, tmp_path):
    o, env = first_map_folder
    out = tmp_path / "v.mp4"
    st = SV.render(o.where / B.MAPPING_FULL_DIR, env["src"] / "episode_000002", out, speed=2.0)
    fm = o.report["first_map"]
    assert st["frames"] == fm["frames_full"] and st["fps"] == 60.0 and st["cut"] == fm["frames"] - 1
    assert st["saved_map"] + st["later_map"] + st["not_tracking"] == st["frames"]
    assert st["new_maps"] == 1                                # the fake's state 0 at frame 24
    assert _frames(out)[0] == st["frames"]
    # the converted raw/ copy and the Pi folder give the same video and the same frames CSV times
    assert T.ultrawide_video(o.where / "raw")[0].name == "cam1.mp4"
    assert T.ultrawide_video(env["src"] / "episode_000002")[0].name == "cam0.mp4"


def test_3d_page_by_folder_name(first_map_folder, capsys):
    o, env = first_map_folder
    an = env["root"].tasks / o.folder / T.ANALYSIS_DIR
    assert umicli.main(["slam-3d", o.folder] + _root(env)) == 0
    d = _page_data(an / S3.HTML_NAME)
    fm = o.report["first_map"]
    assert len(d["points"]["xyz"]) == 20 and d["meta"]["map_points"] == 20
    assert d["title"] == f"{o.folder}: mapping run" and len(d["frames"]) == fm["frames"] - 2   # frames 0, 1: not posed
    assert umicli.main(["slam-3d", o.folder, "--run", "mapping_full"] + _root(env)) == 0
    assert len(_page_data(an / "slam_3d_mapping_full.html")["frames"]) > len(d["frames"])
    text = capsys.readouterr().out
    assert f"wrote {an / S3.HTML_NAME}" in text and "20 points" in text
    with pytest.raises(SystemExit):                                                        # not a choice
        umicli.main(["slam-3d", o.folder, "--run", "relocalize"] + _root(env))


def test_3d_page_path_only_without_stored_points(env, monkeypatch, capsys):
    _pass_m4_m5(monkeypatch)
    fake_slam_build(env["tmp"] / "fork_old", monkeypatch, save_points=False)
    o, _ = _map(env)
    assert o.passed
    assert umicli.main(["slam-3d", o.folder] + _root(env)) == 0
    d = _page_data(env["root"].tasks / o.folder / T.ANALYSIS_DIR / S3.HTML_NAME)
    assert d["points"]["xyz"] == [] and d["meta"] is None and len(d["frames"]) > 0
    assert "no map_points.csv" in capsys.readouterr().out


def test_run_dir_fallback_for_a_failed_map(env, capsys):
    """A failed map (M1: two maps) has no task folder: its run folder in maps_failed/ is given; output next to it."""
    o, _ = _map(env, runner=FakeSlam(maps=lambda mapping: TWO_MAPS))
    assert o.exit_code == B.EXIT_FAILED and o.where.parent.parent.name == B.FAILED_DIR
    run = o.where / B.MAPPING_DIR
    assert umicli.main(["slam-video", str(run)]) == 0           # the recording from the report's source
    assert _frames(o.where / T.ANALYSIS_DIR / SV.video_name(B.MAPPING_DIR))[0] == o.report["runs"]["mapping"][
        "frames_fed"]
    assert umicli.main(["slam-3d", str(run)]) == 0
    assert len(_page_data(o.where / T.ANALYSIS_DIR / S3.HTML_NAME)["points"]["xyz"]) == 20
    # the map attempt folder itself, with --run
    assert umicli.main(["slam-video", str(o.where), "--run", "relocalize"]) == 0
    assert (o.where / T.ANALYSIS_DIR / SV.video_name(B.RELOCALIZE_DIR)).is_file()
    assert "wrote" in capsys.readouterr().out
    assert not (env["root"].path / "tasks").exists() or not any((env["root"].path / "tasks").iterdir())


def test_input_errors(first_map_folder, tmp_path, capsys):
    o, env = first_map_folder
    run = o.where / B.MAPPING_DIR
    assert umicli.main(["slam-video", "no_such_folder"] + _root(env)) == 2
    assert "no task folder 'no_such_folder'" in capsys.readouterr().err
    assert umicli.main(["slam-3d", "no_such_folder"] + _root(env)) == 2
    lone = tmp_path / "lone"
    shutil.copytree(run, lone)                               # no report.json or raw/ next to it
    assert umicli.main(["slam-video", str(lone)]) == 2
    assert "pass --recording" in capsys.readouterr().err
    assert umicli.main(["slam-video", str(lone), "--recording", str(tmp_path)]) == 2     # no meta.json
    assert "ultrawide camera" in capsys.readouterr().err
    assert umicli.main(["slam-3d", str(lone)]) == 0           # estimated times; output in lone/../analysis
    assert (tmp_path / T.ANALYSIS_DIR / S3.page_name("lone")).is_file()        # named after the run folder
    (o.where / "raw" / "meta.json").rename(tmp_path / "meta.json")      # no usable raw/ copy: the report's source
    rep = json.loads((o.where / B.REPORT_NAME).read_text())
    rep["source"] = str(tmp_path / "gone")
    (o.where / B.REPORT_NAME).write_text(json.dumps(rep))
    assert umicli.main(["slam-video", o.folder] + _root(env)) == 2
    assert "is not there any more" in capsys.readouterr().err
    assert umicli.main(["slam-video", o.folder, "--recording", str(env["src"] / "episode_000002"), "--speed",
                        "0"] + _root(env)) == 2
