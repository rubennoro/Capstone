"""The task .zarr.zip against the iPhone golden schema (scripts/regression/golden/iphone_firstreal_5ep.json)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from dataproc._vendor.replay_buffer import get_optimal_chunks
from dataproc.export.frames import ultrawide_substream
from dataproc.export.zarr_writer import IMAGE_SHAPE, EpisodeArrays, build_task_zarr_zip, write_episode_zarr

REPO = Path(__file__).resolve().parents[2]          # the Capstone checkout
GOLDEN = REPO / "scripts/regression/golden/iphone_firstreal_5ep.json"
IMAGE_KEYS = {"data/camera_right_main_rgb", "data/camera_right_ultrawide_rgb"}


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.is_file():
        pytest.skip(f"golden file not found at {GOLDEN}")
    return json.loads(GOLDEN.read_text())["zarr"]


def random_episode(rng, name, n, label="block_raise", uw_offset_ns=None):
    t = np.round(np.arange(n) * 1e9 / 60).astype(np.int64)
    off = int(rng.integers(-60, 20)) * 1_000_000 if uw_offset_ns is None else uw_offset_ns
    t_uw = off + np.round(np.arange(n // 2 + 6) * 1e9 / 30).astype(np.int64)
    sel, up = ultrawide_substream(t, t_uw)
    return EpisodeArrays(name=name, task_label=label,
                         main_rgb=rng.integers(0, 256, (n,) + IMAGE_SHAPE, dtype=np.uint8),
                         uw_rgb=rng.integers(0, 256, (len(sel),) + IMAGE_SHAPE, dtype=np.uint8),
                         uw_upsample_local=up, eef_pos=rng.normal(size=(n, 3)),
                         eef_rot_axis_angle=rng.normal(size=(n, 3)) + 2.0, gripper_width=rng.random((n, 1)) * 0.08)


def array_info(z: zarr.Group, path: str) -> dict:
    a = z[path]
    return {"shape": list(a.shape), "dtype": str(a.dtype), "chunks": list(a.chunks),
            "compressor": a.compressor.get_config() if a.compressor is not None else None,
            "filters": [f.get_config() for f in (a.filters or [])], "attrs": dict(a.attrs)}


def all_arrays(z: zarr.Group) -> list[str]:
    return sorted(f"{g}/{k}" for g in z.group_keys() for k in z[g].array_keys())


def raw_chunks_sha256(zip_path: Path, path: str) -> str:
    """Same hash as scripts/regression/iphone_pipeline_check.py."""
    zf = zipfile.ZipFile(zip_path)
    h = hashlib.sha256()
    for n in sorted(zf.namelist()):
        if n.startswith(path + "/") and not n.rsplit("/", 1)[-1].startswith("."):
            h.update(n.encode())
            h.update(zf.read(n))
    return h.hexdigest()


@pytest.fixture(scope="module")
def three_episode_zip(tmp_path_factory):
    rng = np.random.default_rng(0)
    eps = [random_episode(rng, f"2026-11-04T10-0{i}-00.000Z_rig_ep{i}", n) for i, n in enumerate([95, 64, 130])]
    d = tmp_path_factory.mktemp("zip3")
    out = d / "block_raise.zarr.zip"
    # mix inputs: in-memory, per-episode zarr (chunk copy), in-memory
    p1 = write_episode_zarr(eps[1], d / "episode.zarr")
    summary = build_task_zarr_zip([eps[0], p1, eps[2]], out)
    return out, eps, summary


def test_schema_matches_golden(golden, three_episode_zip):
    out, eps, summary = three_episode_zip
    z = zarr.open(str(out), mode="r")
    assert dict(z.attrs) == golden["root_attrs"] == {}
    assert sorted(z.group_keys()) == golden["groups"] == ["data", "labels", "meta"]
    assert list(z["labels"].array_keys()) == []
    assert all_arrays(z) == sorted(golden["arrays"])
    for path, g in golden["arrays"].items():
        info = array_info(z, path)
        assert info["dtype"] == g["dtype"], path
        assert info["compressor"] == g["compressor"], path
        assert info["filters"] == g["filters"], path
        assert info["attrs"] == g["attrs"], path
        assert len(info["shape"]) == len(g["shape"]) and info["shape"][1:] == g["shape"][1:], path
        if path in IMAGE_KEYS:
            assert info["chunks"] == g["chunks"] == [1, 224, 224, 3], path
        else:                                                    # the iPhUMI 8 KiB rechunk rule
            exp = list(get_optimal_chunks(tuple(info["shape"]), np.dtype(info["dtype"]), target_chunk_bytes=8192))
            assert info["chunks"] == exp, path
    assert summary["n_episodes"] == 3 and summary["n_steps"] == sum(e.n_steps for e in eps)


def test_meta_semantics(three_episode_zip):
    out, eps, _ = three_episode_zip
    z = zarr.open(str(out), mode="r")
    m = z["meta"]
    n = np.array([e.n_steps for e in eps])
    mu = np.array([e.n_uw for e in eps])
    ends = np.cumsum(n)
    assert m["episode_ends"][:].tolist() == ends.tolist()
    assert m["episode_names"][:].tolist() == [e.name for e in eps]
    assert m["task_names"][:].tolist() == ["block_raise"] * 3
    assert m["task_lengths"][:].tolist() == (n - 1).tolist()
    assert m["task_labels_ends"][:].tolist() == np.cumsum(n - 1).tolist()
    assert m["task_data_ends"][:].tolist() == (ends - 1).tolist()
    assert not m["is_episode_error_correction"][:].any() and not m["is_task_error_correction"][:].any()
    uw_ends = np.cumsum(mu)
    assert m["episode_ends_camera_right_ultrawide_rgb"][:].tolist() == uw_ends.tolist()
    up = m["upsample_index_camera_right_ultrawide_rgb"][:]
    ds = m["downsample_index_camera_right_ultrawide_rgb"][:]
    assert len(up) == ends[-1] and len(ds) == uw_ends[-1] == z["data/camera_right_ultrawide_rgb"].shape[0]
    assert z["data/camera_right_main_rgb"].shape[0] == ends[-1]
    s = su = 0
    for e, ee, ue in zip(eps, ends, uw_ends):
        assert np.array_equal(up[s:ee], e.uw_upsample_local + su)       # global = local + previous uw total
        assert up[s] >= su and up[ee - 1] == ue - 1
        dl = ds[su:ue]
        assert np.all((dl >= s) & (dl < ee))                             # downsample stays in its episode
        assert np.array_equal(up[dl], np.arange(su, ue))                 # first step that uses each uw frame
        assert np.all(dl[1:] > dl[:-1])
        s, su = ee, ue
    for k, attr in (("gripper_right_eef_pos", "eef_pos"), ("gripper_right_eef_rot_axis_angle", "eef_rot_axis_angle"),
                    ("gripper_right_gripper_width", "gripper_width")):
        assert np.array_equal(z["data"][k][:], np.concatenate([getattr(e, attr) for e in eps]))


def _golden_episodes(golden):
    """Episodes with the golden's exact lengths and per-episode ultrawide maps (images constant)."""
    a = golden["arrays"]
    ends = np.array(a["meta/episode_ends"]["values"])
    uw_ends = np.array(a["meta/episode_ends_camera_right_ultrawide_rgb"]["values"])
    up = np.array(a["meta/upsample_index_camera_right_ultrawide_rgb"]["values"])
    names = a["meta/episode_names"]["values"]
    pos = np.array(a["data/gripper_right_eef_pos"]["values"], np.float32)
    rot = np.array(a["data/gripper_right_eef_rot_axis_angle"]["values"], np.float32)
    wid = np.array(a["data/gripper_right_gripper_width"]["values"], np.float32)
    img = np.full((1,) + IMAGE_SHAPE, 90, np.uint8)
    eps, s, su = [], 0, 0
    for name, e, ue in zip(names, ends, uw_ends):
        eps.append(EpisodeArrays(name=name, task_label=a["meta/task_names"]["values"][0],
                                 main_rgb=np.broadcast_to(img, (e - s,) + IMAGE_SHAPE),
                                 uw_rgb=np.broadcast_to(img, (ue - su,) + IMAGE_SHAPE),
                                 uw_upsample_local=up[s:e] - su, eef_pos=pos[s:e], eef_rot_axis_angle=rot[s:e],
                                 gripper_width=wid[s:e]))
        s, su = e, ue
    return eps


def test_golden_replay_is_byte_identical(golden, tmp_path):
    """Built from the golden's own per-episode lengths, ultrawide maps and lowdim values, every array has the
    golden's shape and chunks, and every meta and lowdim array is byte-identical (same raw chunk sha256)."""
    out = tmp_path / "golden.zarr.zip"
    build_task_zarr_zip(_golden_episodes(golden), out)
    z = zarr.open(str(out), mode="r")
    assert all_arrays(z) == sorted(golden["arrays"])
    for path, g in golden["arrays"].items():
        info = array_info(z, path)
        for k in ("shape", "dtype", "chunks", "compressor", "filters", "attrs"):
            assert info[k] == g[k], (path, k)
        if path not in IMAGE_KEYS:
            assert raw_chunks_sha256(out, path) == g["raw_chunks_sha256"], path
