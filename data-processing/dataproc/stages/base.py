"""The stage interface and the objects every stage receives (docs/PROCESS.md).

* :class:`Stage`: ``name``, ``version``, ``deps``; ``inputs(ctx, ep) -> dict`` (what goes into the stage's
  ``inputs_hash``, besides the manifest, the dataproc version and the ``deps`` hashes); ``run(ctx, ep) -> dict``
  (the ``outputs`` summary recorded in ``status.json``). A stage that cannot produce its output raises
  :class:`StageFailed` (``failed``), one that finds the episode unusable raises it with ``rejected``.
* :class:`ProcessContext`: everything shared by the run (task, rig, SLAM backend, calibration, one ImagePrep).
* :class:`Episode`: one ``episode_N``: paths, the opened raw episode, and the products of earlier stages (kept in
  memory when they ran in this call, otherwise loaded from ``processed/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dataproc.files import read_json

__all__ = ["StageFailed", "Stage", "Calibration", "ProcessContext", "Episode", "rig_sections"]


class StageFailed(Exception):
    """Raised by :meth:`Stage.run`: ``status`` is ``failed`` (no output) or ``rejected`` (episode unusable)."""

    def __init__(self, status: str, reasons: list[str], message: str = "", details: dict | None = None):
        if status not in ("failed", "rejected"):
            raise ValueError(f"StageFailed status must be failed or rejected, got {status!r}")
        super().__init__(message or ", ".join(reasons))
        self.status, self.reasons, self.details = status, list(reasons), dict(details or {})


class Stage:
    """One per-episode processing step. Subclasses set ``name`` (the ``status.json`` key), ``deps`` (stages whose
    ``inputs_hash`` is chained into this one, and whose re-run forces this one to re-run) and implement
    :meth:`inputs` and :meth:`run`. Bump ``version`` when the code changes what the stage writes."""

    name = "stage"
    version = 1
    deps: tuple[str, ...] = ()

    def inputs(self, ctx: "ProcessContext", ep: "Episode") -> dict:
        raise NotImplementedError

    def run(self, ctx: "ProcessContext", ep: "Episode") -> dict:
        raise NotImplementedError


# --------------------------------------------------------------------------------------------- context

def _cam_section(rig, role: str) -> dict:
    c = rig.cameras[role]
    return {"resolution": list(c.resolution), "fps": c.fps,
            "model": {"type": c.model.type, "intrinsics": list(c.model.intrinsics),
                      "distortion": list(c.model.distortion)},
            "T_imu_cam": c.T_imu_cam, "timestamp_ref": c.timestamp_ref, "time_offset_ns": c.time_offset_ns}


def rig_sections(rig) -> dict:
    """Canonical (parsed) rig yaml sections for the stage hashes. ``provenance`` and ``notes`` are left out: they
    do not change any output."""
    n, v, a = rig.imu.noise, rig.virtual_ultrawide, rig.aruco
    return {
        "rig_id": rig.rig_id,
        "cameras.main": _cam_section(rig, "main"),
        "cameras.ultrawide": _cam_section(rig, "ultrawide"),
        "cameras.resolution_fps": {r: {"resolution": list(c.resolution), "fps": c.fps}
                                   for r, c in sorted(rig.cameras.items())},
        "virtual_ultrawide": {"resolution": list(v.resolution), "intrinsics": list(v.intrinsics),
                              "R_phys_virt": v.R_phys_virt},
        "tcp": {"T_cam_tcp": rig.T_cam_tcp},
        "imu": {"sensor": rig.imu.sensor, "time_offset_ns": rig.imu.time_offset_ns,
                "noise": [n.gyroscope_noise_density, n.gyroscope_random_walk, n.accelerometer_noise_density,
                          n.accelerometer_random_walk, n.inflation_noise, n.inflation_random_walk]},
        "imu.time_offset_ns": rig.imu.time_offset_ns,
        "aruco": {"dict": a.dict, "marker_size_m": a.marker_size_m, "left_id": a.left_id, "right_id": a.right_id,
                  "nominal_z_m": a.nominal_z_m},
        "world_offset": {"T_world_offset": rig.T_world_offset},
        "gripper": (None if rig.gripper is None else
                    {"max_opening_m": rig.gripper.max_opening_m,
                     "opening_tolerance_m": rig.gripper.opening_tolerance_m,
                     **({} if rig.gripper.closed_width_m is None else
                        {"closed_width_m": rig.gripper.closed_width_m})}),
    }


@dataclass
class Calibration:
    """Result of the calibration step. ``state``: ``done`` | ``no_calibration`` | ``calibration_failed``;
    ``token`` is what the width stage hashes (the gripper_range.json bytes hash, or the failure)."""

    state: str
    name: str | None = None
    reason: str | None = None
    gripper_range: Any = None
    token: str = ""
    ran: bool = False

    def to_report(self) -> dict:
        d = {"state": self.state, "name": self.name, "reason": self.reason, "ran": self.ran}
        if self.gripper_range is not None:
            gr = self.gripper_range
            d["gripper_range"] = {"min_width": gr.min_width, "max_width": gr.max_width,
                                  "n_both_frames": gr.fit_stats.get("n_both_frames")}
        return d


@dataclass
class ProcessContext:
    """Shared by every step of one ``dataproc process`` call."""

    root: Any                        # DataRoot
    folder: str
    task_dir: Path
    task: Any                        # TaskConfig
    rig: Any                         # RigConfig
    ledger: list[dict]
    backend: Any                     # SlamBackend
    backend_name: str
    force: bool
    log: Callable = print
    calibration: Calibration | None = None
    sections: dict = field(init=False)
    _prep: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.sections = rig_sections(self.rig)

    @property
    def prep(self):
        """One :class:`~dataproc.imageprep.ImagePrep` per run (its remap table is the expensive part)."""
        if self._prep is None:
            from dataproc.imageprep import ImagePrep
            self._prep = ImagePrep(self.rig)
        return self._prep

    def pi_name(self, ep_dir: Path) -> str:
        """``pi_name`` of the slot in ``episodes.csv``; without one, the Pi's naming convention from meta.json."""
        import json

        for r in self.ledger:
            if r.get("slot") == ep_dir.name and r.get("kind") == "demonstration" and r.get("pi_name"):
                return r["pi_name"]
        try:
            meta = json.loads((ep_dir / "raw" / "meta.json").read_text(encoding="utf-8"))
            return f"{meta['session_id']}_episode_{int(meta['episode_number']):06d}"
        except Exception:  # noqa: BLE001
            return f"{self.folder}_{ep_dir.name}"


# --------------------------------------------------------------------------------------------- episode

class Episode:
    """One ``episode_N``. ``products`` holds what stages produced in this call; the accessors fall back to the
    files the stages wrote into ``processed/``."""

    def __init__(self, ep_dir: Path, pi_name: str):
        self.dir = Path(ep_dir)
        self.slot = self.dir.name
        self.pi_name = pi_name
        self.raw_dir = self.dir / "raw"
        self.processed = self.dir / "processed"
        self.products: dict[str, Any] = {}
        self._raw = None

    @property
    def raw(self):
        """The opened :class:`~dataproc.raw_episode.RawEpisode` (opened on first use)."""
        if self._raw is None:
            from dataproc.raw_episode import RawEpisode
            self._raw = RawEpisode.open(self.raw_dir)
        return self._raw

    def manifest_bytes(self) -> bytes:
        from dataproc.raw_episode import MANIFEST

        try:
            return (self.raw_dir / MANIFEST).read_bytes()
        except OSError:
            return b""

    def _get(self, key: str, load: Callable[[], Any]):
        if key not in self.products:
            self.products[key] = load()
        return self.products[key]

    def _npz(self, fname: str) -> dict:
        with np.load(self.processed / fname, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def grid(self) -> np.ndarray:
        return self._get("grid", lambda: np.load(self.processed / "grid.npy"))

    def slam(self) -> dict:
        """``{trajectory, t_frames_ns, tracked, problems}``."""
        def load():
            from dataproc.geometry.trajectory import Trajectory

            z = self._npz("slam/slam.npz")
            res = read_json(self.processed / "slam" / "result.json", {})
            return {"trajectory": Trajectory(z["t_ns"], z["p"], z["q_xyzw"], frame="imu", world="z_up_gravity"),
                    "t_frames_ns": z["t_frames_ns"], "tracked": z["tracked"], "problems": res.get("problems", [])}
        return self._get("slam", load)

    def pose(self) -> dict:
        return self._get("pose", lambda: self._npz("pose.npz"))

    def width(self) -> dict:
        return self._get("width", lambda: self._npz("width.npz"))

    def kept(self) -> dict:
        """The validate stage's kept span ``{i0, i1, ...}`` (grid rows ``i0:i1``)."""
        return self._get("kept", lambda: read_json(self.processed / "validate.json")["kept"])
