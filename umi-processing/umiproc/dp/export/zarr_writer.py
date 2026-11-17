"""Per-episode artifact and the task-level ``.zarr.zip`` (the training replay buffer).

**Per-episode artifact** (``episode_N/processed/episode.zarr``, a zarr v2 ``DirectoryStore``), written by
:func:`write_episode_zarr`, read by :func:`load_episode_zarr`::

    .zattrs                 {"schema": "dataproc.episode_zarr.v1", "name": <Pi episode dir name>, "task_label": ...}
    main_rgb                (N, 224, 224, 3) uint8, chunks (1, 224, 224, 3), imagecodecs_jpegxl level 99
    uw_rgb                  (M, 224, 224, 3) uint8, same chunks/codec
    uw_upsample_local       (N,) int64      per 60 Hz step, index into uw_rgb (export/frames.py)
    eef_pos                 (N, 3) float32  TCP position in the world, metres
    eef_rot_axis_angle      (N, 3) float32  TCP rotation vector in the world
    gripper_width           (N, 1) float32  metres

The images are compressed with exactly the codec of the final zip, so :func:`build_task_zarr_zip` copies their
chunks byte for byte (no second lossy JPEG XL generation, no re-encoding time). The directory is written under a
temporary name and renamed into place (near-atomic replace, see ``_replace_dir``), so a crash never leaves a
half-written ``episode.zarr``.

**Task zip** (:func:`build_task_zarr_zip`): the same flow as iPhUMI's ``gen_replay_buffer_zarr.py`` with the
vendored ``ReplayBuffer``: a temporary ``DirectoryStore``; per episode ``add_episode`` (lowdim data, no
compressor; one task ``[0, N - 1)`` named ``task_label``) then the ultrawide ``upsample_index`` /
``episode_ends`` / ``downsample_index`` meta; then the image arrays (``require_dataset``, chunk 1 frame, JPEG XL
level 99, ``numthreads=1``); then the 8 KiB rechunking of every meta array and every non-image data array; then
``save_to_store`` into a ``zarr.ZipStore`` written to a temporary file and renamed to ``out_zip_path``. The temp
directory is always removed. Output schema: docs/DESIGN.md "Output zarr contract" (== the iPhone golden).
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import zarr

from .._vendor.imagecodecs_numcodecs import JpegXl, register_codecs
from .._vendor.replay_buffer import ReplayBuffer, get_optimal_chunks, rechunk_recompress_array

register_codecs(verbose=False)

__all__ = ["EpisodeArrays", "EPISODE_ZARR_SCHEMA", "IMAGE_SHAPE", "image_compressor", "write_episode_zarr",
           "load_episode_zarr", "build_task_zarr_zip", "MAIN_KEY", "UW_KEY", "POS_KEY", "ROT_KEY", "WIDTH_KEY"]

EPISODE_ZARR_SCHEMA = "dataproc.episode_zarr.v1"
IMAGE_SHAPE = (224, 224, 3)
SIDE = "right"
MAIN_KEY = f"camera_{SIDE}_main_rgb"
UW_KEY = f"camera_{SIDE}_ultrawide_rgb"
POS_KEY = f"gripper_{SIDE}_eef_pos"
ROT_KEY = f"gripper_{SIDE}_eef_rot_axis_angle"
WIDTH_KEY = f"gripper_{SIDE}_gripper_width"
_META_TARGET_CHUNK_BYTES = 8 * 1024    # gen_replay_buffer_zarr.py
_DATA_TARGET_CHUNK_BYTES = 8 * 1024


def image_compressor() -> JpegXl:
    """The image codec of iPhUMI's replay buffer: ``JpegXl(level=99, numthreads=1)``."""
    return JpegXl(level=99, numthreads=1)


# --------------------------------------------------------------------------------------------- episode arrays

@dataclass(eq=False)
class EpisodeArrays:
    """One processed episode on the 60 Hz grid (N steps) with its ~10 Hz ultrawide sub-stream (M frames).

    ``__post_init__`` casts the lowdim arrays to float32 / int64 and validates everything (``ValueError``):
    shapes, uint8 images of 224x224x3, N >= 2 (the task is ``[0, N - 1)``), finite lowdim values, no zero rotation
    vector (iPhUMI's ``pose_6d_to_4x4`` is NaN there), and ``uw_upsample_local`` non-decreasing, >= 0 and
    ``uw_upsample_local[-1] == M - 1`` (iPhUMI derives M from it).
    """

    name: str                       # Pi episode dir name (meta/episode_names)
    task_label: str                 # task.yaml label (meta/task_names)
    main_rgb: np.ndarray            # (N, 224, 224, 3) uint8
    uw_rgb: np.ndarray              # (M, 224, 224, 3) uint8
    uw_upsample_local: np.ndarray   # (N,) int64
    eef_pos: np.ndarray             # (N, 3) float32
    eef_rot_axis_angle: np.ndarray  # (N, 3) float32
    gripper_width: np.ndarray       # (N, 1) float32

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name: must be a non-empty string")
        if not isinstance(self.task_label, str) or not self.task_label:
            raise ValueError("task_label: must be a non-empty string")
        for k in ("main_rgb", "uw_rgb"):
            a = getattr(self, k)
            if not isinstance(a, np.ndarray) or a.dtype != np.uint8 or a.ndim != 4 or a.shape[1:] != IMAGE_SHAPE:
                raise ValueError(f"{k}: expected (n, 224, 224, 3) uint8, got "
                                 f"{getattr(a, 'shape', None)} {getattr(a, 'dtype', None)}")
        n, m = len(self.main_rgb), len(self.uw_rgb)
        if n < 2:
            raise ValueError(f"main_rgb: need at least 2 steps, got {n}")
        if m < 1:
            raise ValueError("uw_rgb: empty")
        up = np.asarray(self.uw_upsample_local)
        if up.dtype.kind not in "iu" or up.shape != (n,):
            raise ValueError(f"uw_upsample_local: expected ({n},) integers, got {up.shape} {up.dtype}")
        up = up.astype(np.int64)
        if up[0] < 0 or np.any(np.diff(up) < 0):
            raise ValueError("uw_upsample_local: must be >= 0 and non-decreasing")
        if int(up[-1]) != m - 1:
            raise ValueError(f"uw_upsample_local: last value must be M - 1 = {m - 1}, got {int(up[-1])}")
        self.uw_upsample_local = up
        for k, d in (("eef_pos", 3), ("eef_rot_axis_angle", 3), ("gripper_width", 1)):
            a = np.asarray(getattr(self, k))
            if k == "gripper_width" and a.shape == (n,):
                a = a[:, None]
            if a.shape != (n, d):
                raise ValueError(f"{k}: expected ({n}, {d}), got {a.shape}")
            a = a.astype(np.float32)
            if not np.all(np.isfinite(a)):
                raise ValueError(f"{k}: non-finite values")
            setattr(self, k, a)
        if np.any(np.linalg.norm(self.eef_rot_axis_angle, axis=1) == 0.0):
            raise ValueError("eef_rot_axis_angle: zero rotation vector (NaN in iPhUMI pose_6d_to_4x4)")

    @property
    def n_steps(self) -> int:
        return len(self.main_rgb)

    @property
    def n_uw(self) -> int:
        return len(self.uw_rgb)

    def lowdim(self) -> dict[str, np.ndarray]:
        """The ``add_episode`` data dict (replay-buffer keys)."""
        return {POS_KEY: self.eef_pos, ROT_KEY: self.eef_rot_axis_angle, WIDTH_KEY: self.gripper_width}


# --------------------------------------------------------------------------------------------- helpers

def _write_frames(arr: zarr.Array, offset: int, frames: np.ndarray, num_workers: int) -> None:
    """``arr[offset + i] = frames[i]`` (one chunk per frame), encoded in a thread pool like iPhUMI."""
    def put(i: int) -> None:
        arr[offset + i] = frames[i]
    if num_workers <= 1:
        for i in range(len(frames)):
            put(i)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(put, range(len(frames))))


def _replace_dir(tmp: Path, final: Path) -> None:
    """Rename ``tmp`` onto ``final`` (near-atomic: an existing ``final`` is first moved aside to a unique
    ``.<name>.old-*`` dir, removed after the second rename; if that rename fails, the old dir is put back)."""
    old = None
    if final.exists():
        old = Path(tempfile.mkdtemp(prefix=f".{final.name}.old-", dir=final.parent))
        os.rmdir(old)                                  # reserve a unique name, then move final onto it
        os.rename(final, old)
    try:
        os.rename(tmp, final)
    except BaseException:
        if old is not None and not final.exists():
            os.rename(old, final)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


# --------------------------------------------------------------------------------------------- per-episode zarr

def write_episode_zarr(ep: EpisodeArrays, path: Union[str, os.PathLike], num_workers: int = 4) -> Path:
    """Write ``ep`` to the per-episode zarr directory ``path`` (atomically replaced). Returns ``path``."""
    path = Path(path)
    tmp = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    try:
        root = zarr.group(store=zarr.DirectoryStore(str(tmp)), overwrite=True)
        root.attrs.update({"schema": EPISODE_ZARR_SCHEMA, "name": ep.name, "task_label": ep.task_label})
        for key, frames in (("main_rgb", ep.main_rgb), ("uw_rgb", ep.uw_rgb)):
            arr = root.zeros(key, shape=frames.shape, chunks=(1,) + IMAGE_SHAPE, dtype=np.uint8,
                             compressor=image_compressor())
            _write_frames(arr, 0, frames, num_workers)
        for key in ("uw_upsample_local", "eef_pos", "eef_rot_axis_angle", "gripper_width"):
            v = getattr(ep, key)
            root.array(key, v, chunks=v.shape, compressor=None)
        _replace_dir(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return path


def _open_episode_zarr(path: Union[str, os.PathLike]) -> zarr.Group:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path}: no per-episode zarr")
    root = zarr.open_group(store=zarr.DirectoryStore(str(path)), mode="r")
    if root.attrs.get("schema") != EPISODE_ZARR_SCHEMA:
        raise ValueError(f"{path}: schema {root.attrs.get('schema')!r} != {EPISODE_ZARR_SCHEMA!r}")
    return root


def load_episode_zarr(path: Union[str, os.PathLike]) -> EpisodeArrays:
    """Read a per-episode zarr back into (validated) :class:`EpisodeArrays`, decoding the images."""
    root = _open_episode_zarr(path)
    return EpisodeArrays(name=root.attrs["name"], task_label=root.attrs["task_label"],
                         main_rgb=root["main_rgb"][:], uw_rgb=root["uw_rgb"][:],
                         uw_upsample_local=root["uw_upsample_local"][:], eef_pos=root["eef_pos"][:],
                         eef_rot_axis_angle=root["eef_rot_axis_angle"][:], gripper_width=root["gripper_width"][:])


# --------------------------------------------------------------------------------------------- task zip

@dataclass
class _Source:
    name: str
    task_label: str
    lowdim: dict
    upsample_local: np.ndarray
    n: int
    m: int
    ep: EpisodeArrays | None = None         # in memory: encode the images
    group: zarr.Group | None = None         # per-episode zarr: copy the compressed chunks


def _source(item) -> _Source:
    if isinstance(item, EpisodeArrays):
        return _Source(item.name, item.task_label, item.lowdim(), item.uw_upsample_local, item.n_steps, item.n_uw,
                       ep=item)
    root = _open_episode_zarr(item)
    up = root["uw_upsample_local"][:]
    m = root["uw_rgb"].shape[0]
    # Validate through EpisodeArrays with image placeholders of the right lengths (cheap: broadcast views).
    blank = np.broadcast_to(np.zeros(IMAGE_SHAPE, np.uint8), (1,) + IMAGE_SHAPE)
    chk = EpisodeArrays(name=root.attrs["name"], task_label=root.attrs["task_label"],
                        main_rgb=np.broadcast_to(blank, (root["main_rgb"].shape[0],) + IMAGE_SHAPE),
                        uw_rgb=np.broadcast_to(blank, (m,) + IMAGE_SHAPE), uw_upsample_local=up,
                        eef_pos=root["eef_pos"][:], eef_rot_axis_angle=root["eef_rot_axis_angle"][:],
                        gripper_width=root["gripper_width"][:])
    return _Source(chk.name, chk.task_label, chk.lowdim(), chk.uw_upsample_local, chk.n_steps, m, group=root)


def _same_encoding(src: zarr.Array, dst: zarr.Array) -> bool:
    return (src.dtype == dst.dtype and src.chunks == dst.chunks and src.shape[1:] == dst.shape[1:]
            and src.compressor is not None and dst.compressor is not None
            and src.compressor.get_config() == dst.compressor.get_config()
            and not src.filters and not dst.filters and src.order == dst.order
            and src.fill_value == dst.fill_value
            and getattr(src, "_dimension_separator", ".") == getattr(dst, "_dimension_separator", "."))


def _copy_frames(src: zarr.Array, dst: zarr.Array, offset: int, num_workers: int) -> None:
    """Copy ``src`` into ``dst[offset:]``: byte-for-byte chunk copy when the encodings match, else re-encode."""
    n = src.shape[0]
    if not _same_encoding(src, dst):
        for a in range(0, n, 256):
            _write_frames(dst, offset + a, src[a:a + 256], num_workers)
        return
    s_store, d_store = src.store, dst.store
    s_pre = f"{src.path}/" if src.path else ""
    d_pre = f"{dst.path}/" if dst.path else ""
    tail = ".0" * (len(src.shape) - 1)
    for i in range(n):
        k = f"{s_pre}{i}{tail}"
        if k in s_store:                              # a missing chunk is fill_value in both arrays
            d_store[f"{d_pre}{offset + i}{tail}"] = s_store[k]


def build_task_zarr_zip(episodes: Sequence[Union[EpisodeArrays, str, os.PathLike]],
                        out_zip_path: Union[str, os.PathLike], *, tmp_dir: Union[str, os.PathLike, None] = None,
                        num_workers: int = 4) -> dict:
    """Build the training ``.zarr.zip`` from episodes, in order.

    ``episodes``: :class:`EpisodeArrays` and/or per-episode zarr paths (``processed/episode.zarr``). The temp
    DirectoryStore lives in ``tmp_dir`` (default: next to the output, so the final rename stays on one
    filesystem and a tmpfs ``/tmp`` is not filled). ``out_zip_path``'s directory must exist. Returns a summary
    ``{"path", "n_episodes", "n_steps", "n_ultrawide", "episode_names"}``.
    """
    out = Path(out_zip_path)
    if not out.name.endswith(".zarr.zip"):
        raise ValueError(f"{out}: the output must end in .zarr.zip")
    if not out.parent.is_dir():
        raise FileNotFoundError(f"{out.parent}: output directory does not exist")
    sources = [_source(e) for e in episodes]
    if not sources:
        raise ValueError("build_task_zarr_zip: no episodes")
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"build_task_zarr_zip: duplicate episode names {sorted({n for n in names if names.count(n) > 1})}")

    with warnings.catch_warnings():     # zarr 2.18 FutureWarning about the vendored dtype=str meta arrays (benign)
        warnings.filterwarnings("ignore", message="missing object_codec", category=FutureWarning)
        return _build(sources, names, out, tmp_dir, num_workers)


def _build(sources, names, out: Path, tmp_dir, num_workers: int) -> dict:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.tmp-", dir=str(out.parent))   # unique per call
    os.close(fd)
    tmp_zip = Path(tmp_name)
    try:
        work = Path(tempfile.mkdtemp(prefix=".dataproc_zarr_", dir=str(tmp_dir) if tmp_dir else str(out.parent)))
    except BaseException:
        tmp_zip.unlink()
        raise
    try:
        rb = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(work)))
        buffer_start = 0
        uw_start = 0
        offsets = []
        for s in sources:
            n = s.n
            rb.add_episode(data=s.lowdim, tasks=[{"name": s.task_label, "start_idx": 0, "end_idx": n - 1,
                                                  "labels": {}}],
                           compressors=None, episode_name=s.name, is_error_correction=False)
            offsets.append((buffer_start, uw_start))
            buffer_start += n

            # gen_replay_buffer_zarr.py: registered after add_episode, so episode_ends holds this episode's end.
            uw_idx = s.upsample_local
            n_uw = int(uw_idx[-1]) + 1
            downsample_mapping = np.searchsorted(uw_idx, np.arange(n_uw, dtype=np.int64), side="left")
            meta = rb.meta
            ep_ends = meta["episode_ends"]
            prev_main_end = int(np.asarray(ep_ends)[-2]) if len(np.asarray(ep_ends)) >= 2 else 0
            upsample_meta_key = f"upsample_index_{UW_KEY}"
            episode_ends_uw_key = f"episode_ends_{UW_KEY}"
            downsample_meta_key = f"downsample_index_{UW_KEY}"
            prev_uw_end = int(np.asarray(meta[episode_ends_uw_key])[-1]) if upsample_meta_key in meta else 0
            rb.append_attribute(meta, upsample_meta_key, uw_idx + prev_uw_end)
            rb.append_attribute(meta, episode_ends_uw_key, prev_uw_end + n_uw)
            rb.append_attribute(meta, downsample_meta_key, downsample_mapping + prev_main_end)
            uw_start += n_uw

        main_arr = rb.data.require_dataset(name=MAIN_KEY, shape=(buffer_start,) + IMAGE_SHAPE,
                                           chunks=(1,) + IMAGE_SHAPE, compressor=image_compressor(),
                                           dtype=np.uint8)
        uw_arr = rb.data.require_dataset(name=UW_KEY, shape=(uw_start,) + IMAGE_SHAPE,
                                         chunks=(1,) + IMAGE_SHAPE, compressor=image_compressor(), dtype=np.uint8)
        for s, (mo, uo) in zip(sources, offsets):
            if s.ep is not None:
                _write_frames(main_arr, mo, s.ep.main_rgb, num_workers)
                _write_frames(uw_arr, uo, s.ep.uw_rgb, num_workers)
            else:
                _copy_frames(s.group["main_rgb"], main_arr, mo, num_workers)
                _copy_frames(s.group["uw_rgb"], uw_arr, uo, num_workers)

        # gen_replay_buffer_zarr.py rechunking (8 KiB chunks for random access; image arrays keep chunk 1).
        for key in list(rb.meta.array_keys()):
            arr = rb.meta[key]
            optimal = get_optimal_chunks(shape=arr.shape, dtype=arr.dtype, target_chunk_bytes=_META_TARGET_CHUNK_BYTES)
            if optimal != arr.chunks:
                rechunk_recompress_array(rb.meta, key, chunks=optimal)
        for key in list(rb.data.array_keys()):
            arr = rb.data[key]
            if arr.chunks[0] == 1:
                continue
            optimal = get_optimal_chunks(shape=arr.shape, dtype=arr.dtype, target_chunk_bytes=_DATA_TARGET_CHUNK_BYTES)
            if optimal != arr.chunks:
                rechunk_recompress_array(rb.data, key, chunks=optimal)

        # Re-open so the ReplayBuffer constructor checks the finished layout (upsample/downsample lengths etc.).
        rb = ReplayBuffer(root=zarr.open_group(store=zarr.DirectoryStore(str(work)), mode="r"))
        with zarr.ZipStore(str(tmp_zip), mode="w") as zip_store:
            rb.save_to_store(store=zip_store)
        os.replace(tmp_zip, out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if tmp_zip.exists():
            tmp_zip.unlink()
    return {"path": str(out), "n_episodes": len(sources), "n_steps": int(buffer_start), "n_ultrawide": int(uw_start),
            "episode_names": names}

