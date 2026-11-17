"""The real umiproc_slam on real recordings (slow: about a minute and a half on the laptop; frames are fed in real
time). Skipped when the binary, the vocabulary or the recordings are absent.

Map: the mapping recording's EuRoC folder with the gripper mask and the table tag (13, 0.16 m); then, when that map
has keyframes, localize the second recording against it. Asserted: the run finishes, its summary agrees with the
inputs, and the trajectory CSV lines up with times.txt. Not asserted: how well it tracks (it varies run to run).

Inputs (environment variables, defaults = the laptop's 2026-11-15 eraser_pick_2 recordings, read only):
  UMIPROC_TEST_EUROC_MAP, UMIPROC_TEST_EUROC_LOC   EuRoC folders (times.txt, mav0/, orbslam3_settings.yaml)
  UMIPROC_TEST_MASK                                gripper mask PNG at 1280x960 (255 = masked)
"""

import os
from pathlib import Path

import numpy as np
import pytest

from umiproc.slam.runner import default_binary, default_vocabulary, run_umi_slam
from umiproc.slam.trajectory_csv import read_times

_DATA = Path.home() / "Downloads/data/tasks/eraser_pick_2/failed"
EUROC_MAP = Path(os.environ.get("UMIPROC_TEST_EUROC_MAP",
                                _DATA / "pi-2026-11-15_episode_000120/processed/slam/work"))
EUROC_LOC = Path(os.environ.get("UMIPROC_TEST_EUROC_LOC",
                                _DATA / "pi-2026-11-15_episode_000119/processed/slam/work"))
MASK = Path(os.environ.get("UMIPROC_TEST_MASK", "/tmp/claude-1000/-home-ruben-Capstone-data-work/"
                           "6b9540ba-9ac3-4e29-baef-c4535f45cc0e/scratchpad/umi_mask_1280x960.png"))

_missing = [str(p) for p in (default_binary(), default_vocabulary(), EUROC_MAP / "times.txt",
                             EUROC_LOC / "times.txt", MASK) if not p.is_file()]
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(bool(_missing), reason=f"needs {', '.join(_missing)}")]


def _check_run(res, euroc: Path):
    times = read_times(euroc / "times.txt")
    s = res.summary
    assert s is not None, res.log_path.read_text()[-2000:]
    assert res.returncode == 0 == s.exit_code and not res.timed_out
    assert (s.frames_total, s.frames_fed) == (len(times), len(times)) and not s.stopped_early
    assert s.t0_ns == int(times[0])
    assert sum(s.state_counts.values()) == s.frames_fed
    assert s.lost_frames == s.frames_fed - s.state_counts.get(2, 0)
    tr = res.trajectory()                               # refuses a misaligned file
    assert tr.n_frames == len(times) and np.array_equal(tr.t_ns, times)
    assert tr.n_tracked <= s.frames_fed - s.lost_frames    # a reset un-tracks frames whose state stayed OK
    p = s.pacing                                        # whether the machine kept up (reported, not judged)
    assert p is not None and 0 < p.track_ms_median <= p.track_ms_p95 <= p.track_ms_max
    assert 0 <= p.frames_over_period <= s.frames_fed and p.feed_wall_s > 0
    assert p.recording_rate_hz == pytest.approx((len(times) - 1) / ((times[-1] - times[0]) * 1e-9), rel=1e-3)
    assert 0 < p.feed_rate_hz <= p.recording_rate_hz * 1.05    # real-time pacing never feeds faster
    return tr


def _pacing(s) -> str:
    p = s.pacing
    return (f"tracking {p.track_ms_median:.1f}/{p.track_ms_p95:.1f}/{p.track_ms_max:.1f} ms (median/p95/max), "
            f"{p.frames_over_period}/{s.frames_fed} frames over their period, fed at {p.feed_rate_hz:.2f} of "
            f"{p.recording_rate_hz:.2f} frames/s")


def test_map_then_localize(tmp_path):
    res_map = run_umi_slam(EUROC_MAP, EUROC_MAP / "orbslam3_settings.yaml", tmp_path / "map",
                           save_map=tmp_path / "map.osa", mask_img=MASK, init_tag_id=13, init_tag_size=0.16)
    tr_map = _check_run(res_map, EUROC_MAP)
    s = res_map.summary
    assert s.options["init_tag_id"] == 13 and s.options["mask_img"] == str(MASK.resolve())
    assert res_map.map_saved
    assert len(res_map.log.of_kind("map_initialized")) >= 1
    print(f"\nmap: {tr_map.n_tracked}/{tr_map.n_frames} tracked, maps {s.maps}, "
          f"resets {len(res_map.log.resets)}, VIBA done {res_map.log.viba_completed}; {_pacing(s)}")
    if not s.maps_with_keyframes:
        pytest.skip("the mapping run ended without a map with keyframes (it happens run to run); "
                    "localization not tried")

    res_loc = run_umi_slam(EUROC_LOC, EUROC_LOC / "orbslam3_settings.yaml", tmp_path / "loc",
                           load_map=tmp_path / "map.osa", mask_img=MASK)
    tr_loc = _check_run(res_loc, EUROC_LOC)
    assert res_loc.summary.options["load_map"] == str((tmp_path / "map.osa").resolve())
    assert not res_loc.map_saved
    print(f"localize: {tr_loc.n_tracked}/{tr_loc.n_frames} tracked, relocalized at "
          f"{res_loc.log.relocalization_times_s} s, maps {res_loc.summary.maps}; {_pacing(res_loc.summary)}")
