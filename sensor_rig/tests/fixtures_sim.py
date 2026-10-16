"""Fixtures for the D10a/b acceptance tests (WP-C): simulate a raw rig episode from a real iPhone export, export it,
and give the tests the export, the raw episode, the simulator's truth and the original iPhone episode.

Import into a test module with ``from fixtures_sim import *  # noqa: F401,F403`` (conftest.py is WP-A's file).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from sensor_rig.export.exporter import ExportOptions, ExportResult, export_episode
from sensor_rig.export.grid import build_grid
from sensor_rig.pose.trajectory import Trajectory
from sensor_rig.raw_episode import RawEpisode
from sensor_rig.sim.iphone_reader import IphoneEpisode
from sensor_rig.sim.simulator import SIM_RIG_MAIN30_PATH, SIM_RIG_PATH, SimOptions, load_truth, \
    simulate_raw_from_iphone
from sensor_rig.timefmt import iso_ms

# Real episodes (P1 §5): demos whose main K equals the virtual K (1406.34) allow direct image comparison.
DEMO_1406 = "2026-10-01T21-39-22.626Z_49242_firstreal_demonstration_left"
DEMO_1278 = "2026-10-01T21-14-52.701Z_73132_firstreal_demonstration_left"
CALIB = "2026-10-01T21-10-44.057Z_15647_firstreal_grippercalibration_left"
UNIT_FRAMES = 240

VARIANTS: dict[str, dict] = {
    "demo60": dict(src=DEMO_1406, rig=SIM_RIG_PATH, sim={}, export={}),
    "demo1278": dict(src=DEMO_1278, rig=SIM_RIG_PATH, sim={}, export={}),
    "main30": dict(src=DEMO_1406, rig=SIM_RIG_MAIN30_PATH, sim={}, export={}),
    "traj_ultrawide": dict(src=DEMO_1406, rig=SIM_RIG_PATH, sim=dict(traj_frame="ultrawide"), export={}),
    "traj_uw_rate": dict(src=DEMO_1406, rig=SIM_RIG_PATH, sim=dict(traj_rate="uw"),
                         export=dict(max_pose_gap_s=0.15)),
    "jitter": dict(src=DEMO_1406, rig=SIM_RIG_PATH, sim=dict(ts_jitter_ns=500_000, seed=3), export={}),
    "init_delay": dict(src=DEMO_1406, rig=SIM_RIG_PATH, sim=dict(vio_init_delay_s=0.5), export={}),
    "gap": dict(src=DEMO_1406, rig=SIM_RIG_PATH, sim=dict(vio_gap=(2.0, 0.3)), export={}),
    "uw_hold": dict(src=DEMO_1406, rig=SIM_RIG_PATH, sim=dict(uw_mode="30fps_hold", seed=5), export={}),
    "calib": dict(src=CALIB, rig=SIM_RIG_PATH, sim={}, export={}),
}


@dataclass
class SimExport:
    variant: str
    src: IphoneEpisode
    raw: RawEpisode
    truth: dict
    result: ExportResult
    out_root: Path
    json: dict | None = None
    out_dir: Path | None = None
    extra: dict = field(default_factory=dict)

    @property
    def rig(self):
        return self.raw.rig()

    def grid_ns(self) -> np.ndarray:
        """The export's full (untrimmed) grid on the corrected rig clock, recomputed from the raw stamps."""
        t_main = self.rig.capture_time_ns("main", self.raw.camera_times("main"))
        return build_grid(t_main).t_ns

    def grid(self):
        return build_grid(self.rig.capture_time_ns("main", self.raw.camera_times("main")))

    def kept_range(self) -> tuple[int, int]:
        """(i0, n): exported frame k is grid slot i0 + k."""
        n = int(self.json["frameCount"])
        if not self.json["poseTimes"]:  # calibrations are never trimmed
            return 0, n
        off = self.raw.wall_offset_ns()
        strings = [iso_ms(int(t) + off) for t in self.grid_ns()]
        return strings.index(self.json["poseTimes"][0]), n

    def exported_poses(self) -> np.ndarray:
        return np.asarray(self.json["poseTransforms"], dtype=np.float64).reshape(-1, 4, 4)

    def reference_poses(self) -> np.ndarray:
        """Original ARKit poses at the exported frames' *true* times (native grid: the source slot's pose; fill grid:
        originals interpolated (linear + SLERP) at the grid times)."""
        i0, n = self.kept_range()
        T_o = self.truth["T_arkit"]
        g = self.grid()
        if g.mode == "native":
            return T_o[self.truth["main_src_index"][i0:i0 + n]]
        traj = Trajectory.from_T(self.truth["t_ns"], T_o, frame="imu")
        T, valid = traj.sample(g.t_ns[i0:i0 + n], max_gap_ns=10**12)
        assert valid.all()
        return T


def run_variant(variant: str, iphone_data: Path, root: Path, max_frames: int = UNIT_FRAMES) -> SimExport:
    v = VARIANTS[variant]
    src_dir = Path(iphone_data) / v["src"]
    if not src_dir.is_dir():
        pytest.skip(f"iPhone episode {v['src']} not in {iphone_data}")
    raw_root = root / variant / "raw"
    out_root = root / variant / "export"
    opts = SimOptions(max_frames=max_frames, **v["sim"])
    raw_dir = simulate_raw_from_iphone(src_dir, v["rig"], raw_root, opts)
    res = export_episode(raw_dir, out_root, ExportOptions(**v["export"]))
    se = SimExport(variant, IphoneEpisode(src_dir), RawEpisode(raw_dir), load_truth(raw_dir), res, out_root)
    if res.status == "exported":
        se.out_dir = res.out_dir
        with open(res.out_dir / f"{res.name}.json") as f:
            se.json = json.load(f)
    return se


@pytest.fixture(scope="session")
def sim_export(iphone_data, tmp_path_factory):
    """``sim_export(variant)`` -> :class:`SimExport` (simulated once per session and cached)."""
    root = tmp_path_factory.mktemp("sim_export")
    cache: dict[str, SimExport] = {}

    def get(variant: str) -> SimExport:
        if variant not in cache:
            cache[variant] = run_variant(variant, iphone_data, root)
        return cache[variant]

    return get
