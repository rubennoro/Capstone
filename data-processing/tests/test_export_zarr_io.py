"""Per-episode zarr artifact, readers (vendored + behavior_prompting ReplayBuffer), atomic output, validation."""

from __future__ import annotations

import importlib.util
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from dataproc._vendor.replay_buffer import ReplayBuffer
from dataproc.export.frames import ultrawide_substream
from dataproc.export.zarr_writer import (EPISODE_ZARR_SCHEMA, IMAGE_SHAPE, EpisodeArrays, build_task_zarr_zip,
                                         load_episode_zarr, write_episode_zarr)

REPO = Path(__file__).resolve().parents[2]          # the Capstone checkout
BP_REPLAY_BUFFER = REPO / "behavior_prompting/behavior_prompting/common/replay_buffer.py"
IPHUMI_REPLAY_BUFFER = REPO / "iPhUMI/python_package/iphumi/common/replay_buffer.py"
IPHUMI_CODECS = REPO / "iPhUMI/python_package/iphumi/common/imagecodecs_numcodecs.py"


def smooth_frames(rng, n):
    """Low-frequency images (JPEG XL level 99 is near-lossless on these)."""
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    out = np.empty((n,) + IMAGE_SHAPE, np.uint8)
    for i in range(n):
        a, b, c = rng.uniform(10, 40, 3)
        ph = rng.uniform(0, 6.28, 3)
        out[i] = np.stack([128 + 90 * np.sin(xx / a + ph[0]), 128 + 90 * np.sin(yy / b + ph[1]),
                           128 + 90 * np.sin((xx + yy) / c + ph[2])], -1).astype(np.uint8)
    return out


def make_episode(rng, name, n, label="block_raise"):
    t = np.round(np.arange(n) * 1e9 / 60).astype(np.int64)
    t_uw = int(rng.integers(-40, 10)) * 1_000_000 + np.round(np.arange(n // 2 + 6) * 1e9 / 30).astype(np.int64)
    sel, up = ultrawide_substream(t, t_uw)
    return EpisodeArrays(name=name, task_label=label, main_rgb=smooth_frames(rng, n),
                         uw_rgb=smooth_frames(rng, len(sel)), uw_upsample_local=up,
                         eef_pos=rng.normal(size=(n, 3)), eef_rot_axis_angle=rng.normal(size=(n, 3)) + 1.0,
                         gripper_width=rng.random(n) * 0.08)


@pytest.fixture(scope="module")
def eps():
    rng = np.random.default_rng(42)
    return [make_episode(rng, f"pi_ep_{i}", n) for i, n in enumerate([40, 25, 33])]


@pytest.fixture(scope="module")
def built(eps, tmp_path_factory):
    d = tmp_path_factory.mktemp("io")
    paths = []
    for i, e in enumerate(eps):
        (d / f"episode_{i}").mkdir()
        paths.append(write_episode_zarr(e, d / f"episode_{i}" / "episode.zarr"))
    out_mem = d / "mem.zarr.zip"
    out_path = d / "paths.zarr.zip"
    build_task_zarr_zip(eps, out_mem)
    build_task_zarr_zip(paths, out_path)
    return d, paths, out_mem, out_path


def _assert_close(dec, ref):
    assert dec.shape == ref.shape and dec.dtype == np.uint8
    diff = np.abs(dec.astype(np.int16) - ref.astype(np.int16))
    assert diff.mean() < 1.0, diff.mean()
    per_frame = np.abs(dec.reshape(len(dec), -1, 3).mean(1) - ref.reshape(len(ref), -1, 3).mean(1)).max()
    assert per_frame < 0.5, per_frame


# --------------------------------------------------------------------------------------------- episode artifact

def test_episode_zarr_roundtrip(eps, built):
    d, paths, _, _ = built
    root = zarr.open_group(str(paths[0]), mode="r")
    assert root.attrs["schema"] == EPISODE_ZARR_SCHEMA and root.attrs["name"] == "pi_ep_0"
    assert root["main_rgb"].chunks == (1,) + IMAGE_SHAPE
    assert root["main_rgb"].compressor.get_config()["id"] == "imagecodecs_jpegxl"
    back = load_episode_zarr(paths[0])
    e = eps[0]
    assert back.name == e.name and back.task_label == e.task_label
    for k in ("uw_upsample_local", "eef_pos", "eef_rot_axis_angle", "gripper_width"):
        assert np.array_equal(getattr(back, k), getattr(e, k)) and getattr(back, k).dtype == getattr(e, k).dtype
    _assert_close(back.main_rgb, e.main_rgb)
    _assert_close(back.uw_rgb, e.uw_rgb)
    # no temp dirs left behind
    assert [p.name for p in paths[0].parent.iterdir()] == ["episode.zarr"]


def test_episode_zarr_rewrite_replaces_atomically(eps, tmp_path):
    p = tmp_path / "episode.zarr"
    write_episode_zarr(eps[1], p)
    write_episode_zarr(eps[2], p)
    assert load_episode_zarr(p).name == "pi_ep_2"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["episode.zarr"]


def test_load_rejects_non_episode_zarr(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_episode_zarr(tmp_path / "missing.zarr")
    zarr.group(store=zarr.DirectoryStore(str(tmp_path / "other.zarr")))
    with pytest.raises(ValueError):
        load_episode_zarr(tmp_path / "other.zarr")


# --------------------------------------------------------------------------------------------- zip contents

def test_chunk_copy_equals_in_memory_encoding(built):
    """Per-episode-zarr inputs give exactly the same zip contents as in-memory inputs (chunks copied verbatim)."""
    _, paths, out_mem, out_path = built
    za, zb = zipfile.ZipFile(out_mem), zipfile.ZipFile(out_path)
    assert sorted(za.namelist()) == sorted(zb.namelist())
    for n in za.namelist():
        assert za.read(n) == zb.read(n), n
    # and the image chunks are the per-episode chunks, byte for byte
    src = Path(paths[1]) / "main_rgb" / "0.0.0.0"
    assert zb.read("data/camera_right_main_rgb/40.0.0.0") == src.read_bytes()


def test_vendored_replay_buffer_reads_zip(eps, built):
    _, _, out, _ = built
    rb = ReplayBuffer.create_from_path(str(out))
    assert rb.n_episodes == 3 and rb.n_steps == sum(e.n_steps for e in eps)
    assert rb.is_key_upsampled("camera_right_ultrawide_rgb")
    assert not rb.is_key_upsampled("camera_right_main_rgb")
    ends = np.cumsum([e.n_steps for e in eps])
    uw_ends = np.cumsum([e.n_uw for e in eps])
    _assert_close(rb["camera_right_main_rgb"][ends[0]:ends[1]], eps[1].main_rgb)
    _assert_close(rb["camera_right_ultrawide_rgb"][uw_ends[1]:], eps[2].uw_rgb)
    # upsample mapping: the main step -> its ultrawide frame
    steps = np.array([0, ends[0], ends[0] + 7, ends[-1] - 1])
    uw = rb.map_upsample_index("camera_right_ultrawide_rgb", steps)
    assert uw.tolist() == [0, uw_ends[0], uw_ends[0] + eps[1].uw_upsample_local[7], uw_ends[-1] - 1]
    np.testing.assert_array_equal(rb["gripper_right_gripper_width"][:], np.concatenate([e.gripper_width for e in eps]))
    # in-memory copy (numpy backend) also passes the constructor checks
    mem = ReplayBuffer.copy_from_path(str(out))
    assert mem.backend == "numpy" and mem.n_episodes == 3
    assert mem.get_task_to_episode_idxs().tolist() == [0, 1, 2]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_behavior_prompting_replay_buffer_reads_zip(eps, built):
    if not BP_REPLAY_BUFFER.is_file():
        pytest.skip(f"behavior_prompting not found at {BP_REPLAY_BUFFER}")
    try:
        mod = _load_module(BP_REPLAY_BUFFER, "_bp_replay_buffer")
    except ImportError as e:                                   # only zarr / numcodecs / numpy are needed
        pytest.skip(f"behavior_prompting replay_buffer not importable: {e}")
    _, _, out, _ = built
    rb = mod.ReplayBuffer.create_from_path(str(out))
    assert rb.n_episodes == 3 and rb.n_tasks == 3
    assert rb.task_names[:].tolist() == ["block_raise"] * 3
    assert rb.episode_names[:].tolist() == [e.name for e in eps]
    assert rb.is_key_upsampled("camera_right_ultrawide_rgb")
    _assert_close(rb["camera_right_main_rgb"][:eps[0].n_steps], eps[0].main_rgb)
    assert rb.get_task_labels_idxs().tolist() == np.repeat(np.arange(3), [e.n_steps - 1 for e in eps]).tolist()
    assert rb.get_episode_idxs().tolist() == np.repeat(np.arange(3), [e.n_steps for e in eps]).tolist()


def test_iphumi_codec_decodes_and_vendored_copy_is_exact(eps, built):
    """The original iPhUMI codec module decodes our chunks; vendored replay_buffer.py is the iPhUMI file."""
    if not IPHUMI_CODECS.is_file():
        pytest.skip(f"iPhUMI not found at {IPHUMI_CODECS}")
    vend = Path(__file__).resolve().parents[1] / "dataproc" / "_vendor" / "replay_buffer.py"
    assert vend.read_text().split("\n", 2)[2] == IPHUMI_REPLAY_BUFFER.read_text()
    orig = _load_module(IPHUMI_CODECS, "_iphumi_codecs")
    _, _, out, _ = built
    zf = zipfile.ZipFile(out)
    codec = orig.JpegXl(level=99, numthreads=1)
    dec = np.asarray(codec.decode(zf.read("data/camera_right_ultrawide_rgb/0.0.0.0")))
    _assert_close(dec.reshape((1,) + IMAGE_SHAPE), eps[0].uw_rgb[:1])
    from dataproc.export.zarr_writer import image_compressor
    assert codec.get_config() == image_compressor().get_config()


# --------------------------------------------------------------------------------------------- atomicity / errors

def test_failed_build_leaves_no_trace_and_keeps_old_zip(eps, tmp_path):
    out = tmp_path / "t.zarr.zip"
    out.write_bytes(b"old")
    bad = EpisodeArrays(name="pi_ep_0", task_label="x", main_rgb=eps[1].main_rgb, uw_rgb=eps[1].uw_rgb,
                        uw_upsample_local=eps[1].uw_upsample_local, eef_pos=eps[1].eef_pos,
                        eef_rot_axis_angle=eps[1].eef_rot_axis_angle, gripper_width=eps[1].gripper_width)
    with pytest.raises(ValueError, match="duplicate"):
        build_task_zarr_zip([eps[0], bad], out)
    assert out.read_bytes() == b"old"
    with pytest.raises(FileNotFoundError):
        build_task_zarr_zip([eps[0], tmp_path / "nope.zarr"], out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.zarr.zip"]
    with pytest.raises(ValueError):
        build_task_zarr_zip([], out)
    with pytest.raises(ValueError):
        build_task_zarr_zip(eps, tmp_path / "t.zip")
    with pytest.raises(FileNotFoundError):
        build_task_zarr_zip(eps, tmp_path / "missing_dir" / "t.zarr.zip")


def test_successful_build_overwrites_and_cleans_up(eps, tmp_path):
    out = tmp_path / "t.zarr.zip"
    out.write_bytes(b"old")
    s = build_task_zarr_zip(eps[:1], out, num_workers=1)
    assert s["n_episodes"] == 1 and zipfile.is_zipfile(out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.zarr.zip"]


def _kw(e, **over):
    d = dict(name=e.name, task_label=e.task_label, main_rgb=e.main_rgb, uw_rgb=e.uw_rgb,
             uw_upsample_local=e.uw_upsample_local, eef_pos=e.eef_pos, eef_rot_axis_angle=e.eef_rot_axis_angle,
             gripper_width=e.gripper_width)
    d.update(over)
    return d


def test_episode_arrays_validation(eps):
    e = eps[0]
    up = e.uw_upsample_local
    ok = EpisodeArrays(**_kw(e, eef_pos=e.eef_pos.astype(np.float64), gripper_width=e.gripper_width[:, 0]))
    assert ok.eef_pos.dtype == np.float32 and ok.gripper_width.shape == (e.n_steps, 1)
    bad = [
        dict(name=""), dict(task_label=""),
        dict(main_rgb=e.main_rgb.astype(np.float32)), dict(main_rgb=e.main_rgb[:, :100]),
        dict(main_rgb=e.main_rgb[:1], uw_upsample_local=up[:1], eef_pos=e.eef_pos[:1],
             eef_rot_axis_angle=e.eef_rot_axis_angle[:1], gripper_width=e.gripper_width[:1]),
        dict(uw_upsample_local=up[:-1]), dict(uw_upsample_local=up.astype(np.float64)),
        dict(uw_upsample_local=np.minimum(up, up[-1] - 1)),          # last != M - 1
        dict(uw_upsample_local=up[::-1].copy()),                     # decreasing
        dict(uw_upsample_local=up - 1),                              # negative
        dict(eef_pos=e.eef_pos[:, :2]),
        dict(eef_pos=np.where(np.arange(e.n_steps)[:, None] == 3, np.nan, e.eef_pos)),
        dict(eef_rot_axis_angle=np.zeros_like(e.eef_rot_axis_angle)),
        dict(gripper_width=e.gripper_width[:-1]),
    ]
    for over in bad:
        with pytest.raises(ValueError):
            EpisodeArrays(**_kw(e, **over))


# --------------------------------------------------------------------------------------------- failure injection

def test_replace_failure_restores_old_episode(eps, tmp_path, monkeypatch):
    """N1: if renaming the new episode.zarr into place fails, the old one is put back and no temp dirs remain."""
    import dataproc.export.zarr_writer as zw

    p = tmp_path / "episode.zarr"
    write_episode_zarr(eps[1], p)
    real_rename = os.rename

    def flaky_rename(src, dst):
        if Path(src).name.startswith(".episode.zarr.tmp-") and Path(dst) == p:
            raise OSError("injected rename failure")
        return real_rename(src, dst)

    monkeypatch.setattr(zw.os, "rename", flaky_rename)
    with pytest.raises(OSError, match="injected"):
        write_episode_zarr(eps[2], p)
    monkeypatch.setattr(zw.os, "rename", real_rename)
    assert load_episode_zarr(p).name == "pi_ep_1"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["episode.zarr"]


def test_zip_temp_names_unique_within_process(eps, tmp_path, monkeypatch):
    """N2: a second build of the same output, started while the first still holds its temp zip (same process,
    same pid), gets a distinct temp zip (mkstemp) and both complete."""
    import dataproc.export.zarr_writer as zw

    out = tmp_path / "t.zarr.zip"
    seen = []
    real_replace = os.replace

    def nesting_replace(src, dst):
        if not Path(src).name.startswith(".t.zarr.zip.tmp-"):     # zarr's DirectoryStore also uses os.replace
            return real_replace(src, dst)
        seen.append(Path(src).name)
        if len(seen) == 1:                   # first build is about to publish: run a second one now
            assert Path(src).exists()
            build_task_zarr_zip([eps[1]], out, num_workers=1)
            assert Path(src).exists(), "the nested build must not touch the outer temp zip"
        return real_replace(src, dst)

    monkeypatch.setattr(zw.os, "replace", nesting_replace)
    build_task_zarr_zip([eps[0]], out, num_workers=1)
    assert len(seen) == 2 and seen[0] != seen[1]
    assert ReplayBuffer.create_from_path(str(out)).episode_names[:].tolist() == ["pi_ep_0"]   # outer wins (last)
    assert sorted(x.name for x in tmp_path.iterdir()) == ["t.zarr.zip"]


def test_zip_failure_during_save_cleans_temp(eps, tmp_path, monkeypatch):
    """A failure while writing the zip keeps the old output and removes the temp zip and the temp store."""
    out = tmp_path / "t.zarr.zip"
    out.write_bytes(b"old")

    def boom(self, store, **kw):
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(ReplayBuffer, "save_to_store", boom)
    with pytest.raises(RuntimeError, match="injected"):
        build_task_zarr_zip(eps[:1], out)
    assert out.read_bytes() == b"old"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["t.zarr.zip"]
