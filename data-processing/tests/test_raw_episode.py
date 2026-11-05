import json
import os
import subprocess

import numpy as np
import pytest

import fixtures
from fixtures import (
    build_episode, corrupt_hash, drop_csv_row, make_non_monotonic, make_seq_gap, read_meta, remove_file,
    write_manifest, write_meta,
)
from dataproc.raw_episode import (
    PROBLEM_CODES, REQUIRED_FILES, RawEpisode, RawEpisodeError, parse_meta,
)


@pytest.fixture
def ep(tmp_path):
    return build_episode(tmp_path / "s01_episode_000001")


def report(path, **kw):
    return RawEpisode.open(path).validate(**kw)


def codes(path, **kw):
    return report(path, **kw).codes()


# --------------------------------------------------------------------------------------------- valid episode

def test_valid_episode_is_clean(ep):
    r = report(ep)
    assert r.ok and r.problems == ()
    assert r.to_dict() == {"path": str(ep), "ok": True, "problems": []}


def test_reader_api(ep):
    e = RawEpisode.open(ep)
    assert e.name == ep.name and e.is_complete and e.is_verified
    m = e.meta
    assert (m.schema, m.session_id, m.episode_number, m.rig_id, m.clock) == (
        "dataproc.raw.v1", "s01", 1, "rig_v1", "CLOCK_MONOTONIC")
    assert m.cameras["cam0"].role == "main" and m.cameras["cam1"].role == "ultrawide"
    assert m.cameras["cam0"].resolution == (64, 36) and m.cameras["cam1"].fps == 30.0
    assert m.camera("ultrawide") == "cam1" and m.camera("main") == "cam0"
    with pytest.raises(KeyError):
        m.camera("depth")
    assert m.start_unix_ns == m.wall_anchor.unix_ns + (m.button_start_ns - m.wall_anchor.clock_ns)
    f = e.frames("cam0")
    assert len(f) == 12 and f.t_ns.dtype == np.int64 and np.array_equal(f.frame_idx, np.arange(12))
    assert np.all(np.diff(f.t_ns) > 0) and np.all(f.exposure_ns == 5_000_000)
    assert e.imu.gyro.shape == (len(e.imu), 3) and e.imu.accel.dtype == np.float64
    assert e.imu.accel[:, 2].mean() == pytest.approx(9.81, abs=0.05)
    with pytest.raises(KeyError):
        e.frames("cam2")
    assert e.video_path("cam1") == ep / "cam1.mp4"


def test_big_stamps_stay_exact(ep):
    """int64 parsing: ns stamps beyond 2**53 must not go through float."""
    header, *rows = (ep / "cam0_frames.csv").read_text().splitlines()
    big = 999_999_999_999_000_001                     # 18 digits, far beyond 2**53
    rows = [f"{i},{big + i},{100 + i},5000000" for i in range(len(rows))]
    (ep / "cam0_frames.csv").write_text("\n".join([header, *rows]) + "\n")
    assert RawEpisode.open(ep).frames("cam0").t_ns[0] == big


def test_read_video_frames(ep):
    e = RawEpisode.open(ep)
    allf = list(e.read_video_frames("cam1"))
    assert [i for i, _ in allf] == list(range(6))
    assert allf[0][1].shape == (48, 64, 3) and allf[0][1].dtype == np.uint8
    sel = list(e.read_video_frames("cam1", indices=[0, 2, 5]))
    assert [i for i, _ in sel] == [0, 2, 5]
    assert np.array_equal(sel[1][1], allf[2][1])
    assert list(e.read_video_frames("cam1", indices=[])) == []
    # near-lossless content check on frame 3: red channel is a known ramp
    expected_r = (np.mgrid[0:48, 0:64][1] * 4 + 3 * 8) % 256
    assert np.abs(allf[3][1][..., 0].astype(int) - expected_r).mean() < 6
    with pytest.raises(ValueError):
        list(e.read_video_frames("cam1", indices=[2, 1]))
    with pytest.raises(ValueError):
        list(e.read_video_frames("cam1", indices=[-1]))
    with pytest.raises(RawEpisodeError, match="ends before frame 6"):
        list(e.read_video_frames("cam1", indices=[1, 6]))


def test_read_video_frames_unreadable(ep):
    (ep / "cam0.mp4").write_bytes(b"not a video")
    with pytest.raises(RawEpisodeError):
        list(RawEpisode.open(ep).read_video_frames("cam0"))


def test_open_requires_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        RawEpisode.open(tmp_path / "missing")


def test_every_code_has_a_valid_severity():
    assert set(PROBLEM_CODES.values()) <= {"error", "warning"}


# --------------------------------------------------------------------------------------------- files + manifest

@pytest.mark.parametrize("name", [n for n in REQUIRED_FILES if n != "COMPLETE"])
def test_missing_file(ep, name):
    remove_file(ep, name)
    r = report(ep)
    assert not r.ok
    assert any(p.code == "missing_file" and p.file == name for p in r.errors)


def test_missing_data_file_also_fails_manifest(ep):
    remove_file(ep, "cam1.mp4")
    assert {"missing_file", "manifest_listed_missing"} <= codes(ep)


def test_missing_complete(ep):
    remove_file(ep, "COMPLETE")
    e = RawEpisode.open(ep)
    assert not e.is_complete
    assert codes(ep) == {"incomplete"}


def test_missing_verified_is_not_a_problem(ep):
    remove_file(ep, ".verified")
    assert not RawEpisode.open(ep).is_verified and report(ep).ok


def test_bad_hash(ep):
    corrupt_hash(ep, "imu.csv")
    r = report(ep)
    assert [(p.code, p.file) for p in r.errors] == [("hash_mismatch", "imu.csv")]
    assert report(ep, check_manifest=False).ok


def test_extra_file_and_rsync_temp(ep):
    (ep / "notes.txt").write_text("x")
    (ep / ".cam0.mp4.Xa81kd").write_bytes(b"partial")
    (ep / "sub").mkdir()
    r = report(ep)
    assert sorted(p.file for p in r.errors if p.code == "extra_file") == [".cam0.mp4.Xa81kd", "notes.txt"]
    assert [p.file for p in r.errors if p.code == "not_regular_file"] == ["sub"]


def test_extra_listed_file_is_allowed(ep):
    (ep / "pi_log.txt").write_text("ok")
    write_manifest(ep)
    assert report(ep).ok


def test_manifest_missing_entry(ep):
    lines = [ln for ln in (ep / "MANIFEST.sha256").read_text().splitlines() if not ln.endswith("imu.csv")]
    (ep / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    assert {"manifest_missing_entry", "extra_file"} == codes(ep)


@pytest.mark.parametrize("line", ["garbage", "abc  imu.csv", "0" * 64 + "  ../imu.csv", "0" * 64 + "  COMPLETE",
                                  "0" * 64 + "  sub/x"])
def test_manifest_malformed(ep, line):
    with open(ep / "MANIFEST.sha256", "a") as f:
        f.write(line + "\n")
    assert "manifest_malformed" in codes(ep)


def test_manifest_accepts_binary_marker_and_uppercase(ep):
    text = (ep / "MANIFEST.sha256").read_text().splitlines()
    out = [f"{ln[:64].upper()} *{ln[66:]}" for ln in text]
    (ep / "MANIFEST.sha256").write_text("\n".join(out) + "\n")
    assert report(ep).ok


def test_verify_manifest_without_manifest(ep):
    remove_file(ep, "MANIFEST.sha256")
    assert [p.code for p in RawEpisode.open(ep).verify_manifest()] == ["missing_file"]


# --------------------------------------------------------------------------------------------- meta.json

def _edit_meta(ep, fn):
    m = read_meta(ep)
    fn(m)
    write_meta(ep, m)


def test_meta_not_json(ep):
    (ep / "meta.json").write_text("{nope")
    write_manifest(ep)
    e = RawEpisode.open(ep)
    assert "meta_invalid" in e.validate().codes()
    with pytest.raises(RawEpisodeError):
        e.meta


def test_meta_schema_version(ep):
    _edit_meta(ep, lambda m: m.update(schema="dataproc.raw.v0"))
    assert codes(ep) == {"schema_version"}


@pytest.mark.parametrize("edit", [
    lambda m: m.pop("session_id"),
    lambda m: m.update(episode_number=-1),
    lambda m: m.update(episode_number=True),
    lambda m: m.update(rig_id="rig v1"),
    lambda m: m.update(clock="CLOCK_REALTIME"),
    lambda m: m.pop("wall_anchor"),
    lambda m: m["wall_anchor"].update(unix_ns=1.5),
    lambda m: m.update(button_start_ns="0"),
    lambda m: m["cameras"]["cam0"].update(fps=0),
    lambda m: m["cameras"]["cam1"].update(resolution=[64]),
    lambda m: m["cameras"]["cam1"].pop("frame_count"),
    lambda m: m["cameras"]["cam1"].update(dropped_frames=-2),
    lambda m: m["imu"].pop("rate_hz"),
    lambda m: m.update(imu=[]),
    lambda m: m.update(cameras="x"),
])
def test_meta_invalid_fields(ep, edit):
    _edit_meta(ep, edit)
    r = report(ep, decode_videos=False)
    assert "meta_invalid" in r.codes() and not r.ok
    assert all(p.file == "meta.json" for p in r.problems if p.code == "meta_invalid")


def test_meta_unknown_keys_are_kept(ep):
    _edit_meta(ep, lambda m: m.update(pi_software="v1.2", extra={"a": 1}))
    e = RawEpisode.open(ep)
    assert e.validate().ok and e.meta.raw["pi_software"] == "v1.2"


def test_meta_optional_sensor(ep):
    _edit_meta(ep, lambda m: m["cameras"]["cam0"].pop("sensor"))
    e = RawEpisode.open(ep)
    assert e.validate().ok and e.meta.cameras["cam0"].sensor is None


@pytest.mark.parametrize("edit", [
    lambda m: m["cameras"]["cam0"].update(role="ultrawide"),
    lambda m: m["cameras"].update(cam2=m["cameras"]["cam0"]),
    lambda m: m["cameras"].pop("cam1"),
])
def test_camera_roles(ep, edit):
    _edit_meta(ep, edit)
    assert "camera_roles" in codes(ep, decode_videos=False)


def test_parse_meta_non_object():
    meta, probs = parse_meta(None)
    assert meta is None and [p.code for p in probs] == ["meta_invalid"]


# --------------------------------------------------------------------------------------------- CSVs + counts

def test_count_mismatch_rows_vs_meta_and_decoded(ep):
    drop_csv_row(ep, "cam0_frames.csv")
    r = report(ep)
    assert r.codes() == {"count_mismatch", "decoded_count_mismatch"}
    cm = next(p for p in r.problems if p.code == "count_mismatch")
    assert cm.details == {"rows": 11, "meta": 12}
    dm = next(p for p in r.problems if p.code == "decoded_count_mismatch")
    assert dm.details == {"decoded": 12, "rows": 11}
    assert report(ep, decode_videos=False).codes() == {"count_mismatch"}


def test_imu_count_mismatch(ep):
    drop_csv_row(ep, "imu.csv")
    assert codes(ep) == {"count_mismatch"}


def test_decoded_count_mismatch_only(ep):
    _edit_meta(ep, lambda m: m["cameras"]["cam1"].update(frame_count=7))
    header, *rows = (ep / "cam1_frames.csv").read_text().splitlines()
    rows.append(f"6,{int(rows[-1].split(',')[1]) + 33_333_333},106,5000000")
    (ep / "cam1_frames.csv").write_text("\n".join([header, *rows]) + "\n")
    write_manifest(ep)
    assert codes(ep) == {"decoded_count_mismatch"}


def test_video_resolution_mismatch(ep):
    _edit_meta(ep, lambda m: m["cameras"]["cam0"].update(resolution=[1280, 720]))
    assert codes(ep) == {"video_resolution_mismatch"}


def test_video_unreadable(ep):
    (ep / "cam1.mp4").write_bytes(b"\x00" * 100)
    write_manifest(ep)
    assert codes(ep) == {"video_unreadable"}
    assert report(ep, decode_videos=False).ok


@pytest.mark.parametrize("name", ["cam0_frames.csv", "imu.csv"])
def test_non_monotonic(ep, name):
    make_non_monotonic(ep, name, row=3)
    r = report(ep)
    assert r.codes() == {"t_ns_not_monotonic"}
    assert r.problems[0].details == {"row": 3} and r.problems[0].file == name


def test_frame_idx_wrong(ep):
    header, *rows = (ep / "cam1_frames.csv").read_text().splitlines()
    rows[2] = "5" + rows[2][1:]
    (ep / "cam1_frames.csv").write_text("\n".join([header, *rows]) + "\n")
    write_manifest(ep)
    assert codes(ep) == {"frame_idx"}


def test_csv_header_wrong(ep):
    text = (ep / "imu.csv").read_text().replace("t_ns,sample_idx", "timestamp,sample_idx", 1)
    (ep / "imu.csv").write_text(text)
    write_manifest(ep)
    e = RawEpisode.open(ep)
    assert e.validate().codes() == {"csv_header"}
    with pytest.raises(RawEpisodeError):
        e.imu


@pytest.mark.parametrize("bad_row", ["1,2,3", "x,1,2,3", "1.5,1,2,3"])
def test_csv_parse_errors(ep, bad_row):
    with open(ep / "cam0_frames.csv", "a") as f:
        f.write(bad_row + "\n")
    write_manifest(ep)
    assert "csv_parse" in codes(ep, decode_videos=False)


def test_imu_non_finite(ep):
    with open(ep / "imu.csv", "a") as f:
        f.write("9999999999999,999,nan,0,0,0,0,0\n")
    write_manifest(ep)
    assert "csv_parse" in codes(ep, decode_videos=False)


def test_empty_stream(ep):
    (ep / "cam1_frames.csv").write_text("frame_idx,t_ns,sensor_seq,exposure_ns\n")
    _edit_meta(ep, lambda m: m["cameras"]["cam1"].update(frame_count=0))
    assert "empty_stream" in codes(ep, decode_videos=False)


# --------------------------------------------------------------------------------------------- gaps

def test_sensor_seq_gap_is_a_warning_with_sizes(ep):
    make_seq_gap(ep, "cam0_frames.csv", at=4, missing=2)
    r = report(ep)
    assert r.ok
    assert r.codes() == {"sensor_seq_gap", "drop_counter_mismatch"}
    g = next(p for p in r.warnings if p.code == "sensor_seq_gap")
    assert g.details["n_gaps"] == 1 and g.details["total_missing"] == 2 and g.details["max_missing"] == 2
    assert g.details["gaps"] == [{"after_row": 3, "missing": 2, "dt_ms": pytest.approx(1000 / 60, abs=0.01)}]
    _edit_meta(ep, lambda m: m["cameras"]["cam0"].update(dropped_frames=2))
    assert codes(ep) == {"sensor_seq_gap"}


def test_imu_sample_gap(ep):
    make_seq_gap(ep, "imu.csv", at=10, missing=5)
    make_seq_gap(ep, "imu.csv", at=20, missing=1)
    _edit_meta(ep, lambda m: m["imu"].update(dropped_samples=6))
    r = report(ep)
    assert r.ok and r.codes() == {"imu_sample_gap"}
    d = r.warnings[0].details
    assert (d["n_gaps"], d["total_missing"], d["max_missing"]) == (2, 6, 5)
    assert [g["after_row"] for g in d["gaps"]] == [9, 19]


def test_sequence_not_increasing(ep):
    make_seq_gap(ep, "cam1_frames.csv", at=3, missing=-1)
    assert "sequence_not_increasing" in codes(ep)


def test_drop_counter_without_gaps(ep):
    _edit_meta(ep, lambda m: m["imu"].update(dropped_samples=3))
    r = report(ep)
    assert r.ok and r.codes() == {"drop_counter_mismatch"}


# --------------------------------------------------------------------------------------------- button times

def test_button_order(ep):
    _edit_meta(ep, lambda m: m.update(button_stop_ns=m["button_start_ns"]))
    assert codes(ep) == {"button_order"}


def test_button_outside_stream_is_a_warning(ep):
    t_end = int((ep / "imu.csv").read_text().splitlines()[-1].split(",")[0])
    _edit_meta(ep, lambda m: m.update(button_stop_ns=t_end + 50_000_000))
    r = report(ep)
    assert r.ok and r.codes() == {"button_outside_stream"}
    assert {p.details["stream"] for p in r.warnings} == {"cam0", "cam1", "imu"}
    imu_w = next(p for p in r.warnings if p.details["stream"] == "imu")
    assert imu_w.details["early_stop_ms"] == pytest.approx(50.0) and imu_w.details["late_start_ms"] == 0


def test_button_no_overlap(ep):
    _edit_meta(ep, lambda m: m.update(button_start_ns=10 ** 15, button_stop_ns=10 ** 15 + 1))
    r = report(ep)
    assert "button_no_overlap" in r.codes() and not r.ok


def test_problem_json_serialisable(ep):
    make_seq_gap(ep)
    corrupt_hash(ep, "cam0.mp4")
    json.dumps(report(ep).to_dict())


# --------------------------------------------------------------------------------------------- fix round (R-intake)

def test_symlinked_data_file_is_rejected(ep, tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_bytes((ep / "imu.csv").read_bytes())
    (ep / "imu.csv").unlink()
    os.symlink(outside, ep / "imu.csv")
    e = RawEpisode.open(ep)
    r = e.validate()
    assert not r.ok and ("not_regular_file", "imu.csv") in {(p.code, p.file) for p in r.errors}
    assert "missing_file" not in r.codes() and "hash_mismatch" not in r.codes()
    with pytest.raises(RawEpisodeError):                 # never read through the symlink
        e.imu
    assert "not_regular_file" in e.validate(check_manifest=False).codes()


def test_data_file_is_directory(ep):
    remove_file(ep, "imu.csv")
    (ep / "imu.csv").mkdir()
    r = report(ep)
    assert ("not_regular_file", "imu.csv") in {(p.code, p.file) for p in r.errors}


def test_symlinked_marker_does_not_count(ep, tmp_path):
    remove_file(ep, ".verified")
    (tmp_path / "v").touch()
    os.symlink(tmp_path / "v", ep / ".verified")
    e = RawEpisode.open(ep)
    assert not e.is_verified and "not_regular_file" in e.validate().codes()


def test_symlinked_episode_dir(ep, tmp_path):
    link = tmp_path / "link_ep"
    os.symlink(ep, link)
    assert report(link).codes() == {"episode_is_symlink"}
    assert report(str(link) + "/").codes() == {"episode_is_symlink"}


@pytest.mark.parametrize("name", [".hidden", "-x", "_x", "a b", "ép"])
def test_bad_episode_name(tmp_path, name):
    ep = build_episode(tmp_path / name)
    assert report(ep).codes() == {"bad_episode_name"}


@pytest.mark.parametrize("name, col", [("imu.csv", 0), ("imu.csv", 1), ("cam0_frames.csv", 1),
                                       ("cam0_frames.csv", 3)])
@pytest.mark.parametrize("row", [0, 2])
def test_int_overflow_is_csv_parse(ep, name, col, row):
    header, *rows = (ep / name).read_text().splitlines()
    parts = rows[row].split(",")
    parts[col] = "9" * 19
    rows[row] = ",".join(parts)
    (ep / name).write_text("\n".join([header, *rows]) + "\n")
    write_manifest(ep)
    e = RawEpisode.open(ep)
    assert e.validate().codes() == {"csv_parse"}


def test_float_overflow_is_csv_parse(ep):
    with open(ep / "imu.csv", "a") as f:
        f.write("9999999999999,999,1e999,0,0,0,0,0\n")
    write_manifest(ep)
    assert "csv_parse" in codes(ep, decode_videos=False)


def _set_exposure(ep, row, value):
    header, *rows = (ep / "cam0_frames.csv").read_text().splitlines()
    parts = rows[row].split(",")
    parts[3] = str(value)
    rows[row] = ",".join(parts)
    (ep / "cam0_frames.csv").write_text("\n".join([header, *rows]) + "\n")
    write_manifest(ep)


@pytest.mark.parametrize("value", [0, -5, 16_666_667, 20_000_000])     # cam0 runs at 60 fps
def test_exposure_invalid(ep, value):
    _set_exposure(ep, 2, value)
    r = report(ep)
    assert r.codes() == {"exposure_invalid"}
    assert r.errors[0].details == {"row": 2, "n_rows": 1}


def test_exposure_just_below_period_is_fine(ep):
    _set_exposure(ep, 2, 16_666_666)
    assert report(ep).ok


def test_manifest_dot_slash_and_bom(ep):
    text = (ep / "MANIFEST.sha256").read_text().replace("  ", "  ./")
    (ep / "MANIFEST.sha256").write_bytes(b"\xef\xbb\xbf" + text.encode())
    assert report(ep).ok
    subprocess.run(["sha256sum", "-c", "--quiet", "MANIFEST.sha256"], cwd=ep, check=True)


def test_manifest_crlf(ep):
    t = (ep / "MANIFEST.sha256").read_text().replace("\n", "\r\n")
    (ep / "MANIFEST.sha256").write_bytes(t.encode())
    assert report(ep).ok


@pytest.mark.parametrize("prefix", ["/abs/", "././", ".//", "sub/"])
def test_manifest_other_paths_still_malformed(ep, prefix):
    t = (ep / "MANIFEST.sha256").read_text().replace("imu.csv", prefix + "imu.csv")
    (ep / "MANIFEST.sha256").write_text(t)
    assert "manifest_malformed" in codes(ep, decode_videos=False)


@pytest.mark.parametrize("flag", [[], ["-b"]])
def test_manifest_from_real_sha256sum(ep, flag):
    names = ["cam0.mp4", "cam1.mp4", "cam0_frames.csv", "cam1_frames.csv", "imu.csv", "meta.json"]
    out = subprocess.run(["sha256sum", *flag, *names], cwd=ep, capture_output=True, check=True).stdout
    (ep / "MANIFEST.sha256").write_bytes(out)
    assert report(ep).ok


def test_manifest_listed_name_with_space_and_unicode(ep):
    (ep / "pi log.txt").write_text("hi")
    (ep / "logé.txt").write_text("hi")
    write_manifest(ep)
    assert report(ep).ok


def test_truncated_mp4(ep):
    data = (ep / "cam0.mp4").read_bytes()
    (ep / "cam0.mp4").write_bytes(data[: len(data) // 2])
    assert {"hash_mismatch", "video_unreadable"} <= codes(ep)
    write_manifest(ep)
    assert codes(ep) & {"video_unreadable", "decoded_count_mismatch"}


def test_truncated_faststart_mp4(tmp_path):
    import av

    ep = build_episode(tmp_path / "e")
    out = tmp_path / "fs.mp4"
    with av.open(str(ep / "cam0.mp4")) as i, av.open(str(out), "w", options={"movflags": "faststart"}) as o:
        s = o.add_stream_from_template(i.streams.video[0])
        for pkt in i.demux(i.streams.video[0]):
            if pkt.dts is None:
                continue
            pkt.stream = s
            o.mux(pkt)
    d = out.read_bytes()
    (ep / "cam0.mp4").write_bytes(d[: int(len(d) * 0.6)])
    write_manifest(ep)
    assert not report(ep).ok


def test_vfr_mp4_counts_every_frame(tmp_path):
    import av
    from fractions import Fraction

    ep = build_episode(tmp_path / "e", n_cam0=20)
    frames = fixtures._frames(20, fixtures.SIZES["cam0"], 0)
    pts = np.cumsum([1] + [16 if i % 3 else 40 for i in range(19)])
    with av.open(str(ep / "cam0.mp4"), "w") as c:
        st = c.add_stream("libx264", rate=60)
        st.width, st.height, st.pix_fmt = 64, 36, "yuv420p"
        st.time_base = Fraction(1, 1000)
        for f, p in zip(frames, pts):
            vf = av.VideoFrame.from_ndarray(f, format="rgb24")
            vf.pts, vf.time_base = int(p), Fraction(1, 1000)
            for pkt in st.encode(vf):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    write_manifest(ep)
    assert report(ep).ok
    assert len(list(RawEpisode.open(ep).read_video_frames("cam0"))) == 20


def test_static_frames_are_all_counted(tmp_path):
    ep = build_episode(tmp_path / "e", n_cam0=15)
    fixtures._write_mp4(ep / "cam0.mp4", [fixtures._frames(1, fixtures.SIZES["cam0"], 0)[0]] * 15, 60)
    write_manifest(ep)
    assert report(ep).ok


def test_validate_writes_nothing(ep):
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in ep.iterdir()}
    e = RawEpisode.open(ep)
    e.validate()
    list(e.read_video_frames("cam0"))
    assert {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in ep.iterdir()} == before


def test_read_only_episode(ep):
    for p in ep.iterdir():
        os.chmod(p, 0o444)
    os.chmod(ep, 0o555)
    try:
        assert report(ep).ok
    finally:
        os.chmod(ep, 0o755)


def test_meta_nan_and_float_ns(ep):
    t = (ep / "meta.json").read_text().replace('"fps": 60', '"fps": NaN')
    (ep / "meta.json").write_text(t)
    write_manifest(ep)
    assert "meta_invalid" in codes(ep, decode_videos=False)
    m = json.loads((ep / "meta.json").read_text().replace("NaN", "60"))
    m["button_start_ns"] = float(m["button_start_ns"])
    write_meta(ep, m)
    assert "meta_invalid" in codes(ep, decode_videos=False)
