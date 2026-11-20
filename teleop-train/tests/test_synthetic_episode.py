"""The shared test helper builds what it promises (tests/synthetic_episode.py)."""
from __future__ import annotations

import av
import numpy as np

import synthetic_episode as se


def test_episode_files_and_frame_index_roundtrip(tmp_path):
    task = se.make_task(tmp_path)
    ep = se.make_episode(task, 3)
    d = ep.episode_dir
    assert d.name == "episode_003"
    for f in ("meta.yaml", "COMPLETE", "robot_states.csv", "main.mkv", "main_frames.csv", "ultrawide.mkv",
              "ultrawide_frames.csv", "packets.csv", "commands.csv", "keys.csv", "clock_pi.csv", "clock_pc.csv",
              "events.jsonl"):
        assert (d / f).exists(), f
    with av.open(str(d / "main.mkv")) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    assert len(frames) == len(ep.main_t_ns)
    assert [se.decode_index(f) for f in frames] == list(range(len(frames)))
    assert frames[0].shape == (se.HEIGHT, se.WIDTH, 3)
    rows = (d / "robot_states.csv").read_text().splitlines()
    assert rows[0].split(",") == list(se.STATE_COLUMNS)
    assert len(rows) - 1 == len(ep.t_state_ns)


def test_truth_trajectory():
    tr = se.Truth(se.T0_NS, 2.0, 2.0)
    assert tr.duration_s == 8.5
    np.testing.assert_allclose(tr.position([tr.motion_start_ns])[0], se.HOME_POS)
    np.testing.assert_allclose(tr.position([tr.motion_end_ns])[0], se.HOME_POS + [0.1, 0, 0])
    np.testing.assert_allclose(tr.width([tr.motion_end_ns])[0], 0.02)
    turn = (tr.rotation([tr.motion_end_ns]) * se.HOME_ROT.inv()).magnitude()[0]
    np.testing.assert_allclose(np.degrees(turn), 15.0)
    np.testing.assert_allclose(tr.measured([se.T0_NS])[0, :3, 3] - tr.target([se.T0_NS])[0, :3, 3],
                               se.MEASURED_OFFSET_M)
