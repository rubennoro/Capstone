"""Read the low-dim arrays of a umiproc package (.zarr.zip or .zarr folder) without the zarr package: uncompressed
zarr v2, C order (the same reader as the copy's policy_deploy/policy/replay_policy.py _ZarrReader, without its
policy_deploy imports, so it runs in ~/envs/sim_deploy)."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

POS_KEY = "data/gripper_right_eef_pos"                    # policy.yaml key_map.obs (copy): eef_pos
ROT_KEY = "data/gripper_right_eef_rot_axis_angle"         # eef_rot6d (stored as an axis-angle)
WIDTH_KEY = "data/gripper_right_gripper_width"            # gripper_width_m
DATA_HZ = 60.0                                            # umiproc packages: 60 Hz low-dim rows


def parse_replay_target(target: str) -> tuple[Path, int]:
    """'replay:<dataset>#<index>' (policy_deploy use syntax) -> (dataset path, index)."""
    if not target.startswith("replay:") or "#" not in target:
        raise ValueError(f"expected replay:<dataset>#<index>, got {target!r}")
    path, _, idx = target[len("replay:"):].rpartition("#")
    return Path(path).expanduser(), int(idx)


class ZarrLowDim:
    def __init__(self, path) -> None:
        self.path = Path(path).expanduser()
        self._zip = None if self.path.is_dir() else zipfile.ZipFile(self.path)

    def _read(self, name: str) -> bytes | None:
        try:
            return self._zip.read(name) if self._zip is not None else (self.path / name).read_bytes()
        except (KeyError, FileNotFoundError):
            return None

    def array(self, key: str) -> np.ndarray:
        raw = self._read(f"{key}/.zarray")
        if raw is None:
            raise KeyError(f"{self.path.name}: no array {key}")
        meta = json.loads(raw)
        if meta.get("compressor") is not None or meta.get("filters") or meta.get("order", "C") != "C":
            raise ValueError(f"{self.path.name}: {key} is compressed or filtered")
        dtype = np.dtype(meta["dtype"])
        shape, chunks = list(meta["shape"]), list(meta["chunks"])
        sep = meta.get("dimension_separator", ".")
        out = np.full(shape, meta.get("fill_value") or 0, dtype=dtype)
        for i in range(math.ceil(shape[0] / chunks[0]) if shape[0] else 0):
            data = self._read(f"{key}/" + sep.join([str(i)] + ["0"] * (len(shape) - 1)))
            if data is None:
                continue
            block = np.frombuffer(data, dtype=dtype).reshape([chunks[0]] + shape[1:])
            lo, hi = i * chunks[0], min(shape[0], (i + 1) * chunks[0])
            out[lo:hi] = block[:hi - lo]
        return out

    def strings(self, key: str) -> list[str]:
        """A 1-d object array stored with zarr's ``vlen-utf8`` filter (meta/episode_names, meta/task_names): per chunk
        a little-endian uint32 item count, then per item a uint32 byte length and the UTF-8 bytes."""
        raw = self._read(f"{key}/.zarray")
        if raw is None:
            raise KeyError(f"{self.path.name}: no array {key}")
        meta = json.loads(raw)
        if meta.get("compressor") is not None or [f.get("id") for f in meta.get("filters") or []] != ["vlen-utf8"] \
                or len(meta["shape"]) != 1:
            raise ValueError(f"{self.path.name}: {key} is not an uncompressed 1-d vlen-utf8 array")
        out: list[str] = []
        sep = meta.get("dimension_separator", ".")
        for i in range(math.ceil(meta["shape"][0] / meta["chunks"][0]) if meta["shape"][0] else 0):
            data = self._read(f"{key}/" + sep.join([str(i)]))
            if data is None:
                raise ValueError(f"{self.path.name}: {key} chunk {i} missing")
            n, o = int.from_bytes(data[:4], "little"), 4
            for _ in range(n):
                k = int.from_bytes(data[o:o + 4], "little")
                out.append(data[o + 4:o + 4 + k].decode("utf-8"))
                o += 4 + k
        return out[:meta["shape"][0]]

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def load_episode(dataset, index: int) -> dict:
    """{'poses': (n, 4, 4) recorded TCP poses, 'widths': (n,), 'rows': [s, e], 'n_episodes', 'name'}."""
    r = ZarrLowDim(dataset)
    try:
        ends = r.array("meta/episode_ends").astype(np.int64)
        if not 0 <= index < len(ends):
            raise IndexError(f"{Path(dataset).name}: episode {index} not in 0..{len(ends) - 1}")
        s, e = (int(ends[index - 1]) if index > 0 else 0), int(ends[index])
        pos, rot, w = r.array(POS_KEY)[s:e], r.array(ROT_KEY)[s:e], r.array(WIDTH_KEY)[s:e, 0]
        try:
            name = r.strings("meta/episode_names")[index]
        except (KeyError, ValueError, IndexError):
            name = ""
    finally:
        r.close()
    T = np.zeros((e - s, 4, 4))
    T[:, :3, :3] = Rotation.from_rotvec(rot.astype(np.float64)).as_matrix()
    T[:, :3, 3] = pos
    T[:, 3, 3] = 1.0
    return {"poses": T, "widths": w.astype(np.float64), "rows": [s, e], "n_episodes": len(ends), "name": name}


def n_episodes(dataset) -> int:
    r = ZarrLowDim(dataset)
    try:
        return len(r.array("meta/episode_ends"))
    finally:
        r.close()
