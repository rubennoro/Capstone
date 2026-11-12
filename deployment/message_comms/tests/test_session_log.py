import json
from datetime import datetime

import numpy as np

from fakes import FakeClock, make_chunk, make_frames, policy_config, pose, health
from message_comms.policy_step import StepDone, StepJob, StepTiming
from message_comms.pose_converter import PoseConverter
from message_comms.session_log import EpisodeRecord, Outcome, SessionLog, StepRecord
from contracts import SetWaypointsReply


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def make_log(settings):
    clock = FakeClock()
    return SessionLog(settings.session_dir, datetime.now, clock), clock


def step_record(i, sent=True):
    return StepRecord(1, i, 5, "target", 1.0, 70.0, 5.0, 90.0, 110.0, sent, "" if sent else "camera_stale: x",
                      8, 4, 0, 0, 30.0)


def test_the_four_files_and_each_outcome(settings):
    log, clock = make_log(settings)
    log.write_session(settings, policy_config())
    for i, outcome in enumerate(Outcome, start=1):
        log.start_episode(i, pose())
        clock.advance(2.5)
        rec = log.end_episode(outcome, f"reason {i}", {"steps_sent": 7, "late": 1, "odd_chunks": 2,
                                                       "width_clips": 3, "skips.camera_stale": 4,
                                                       "shortened": 5, "pulled_back": 6, "loop_late": 8,
                                                       "anchor_fallbacks": 9})
        assert isinstance(rec, EpisodeRecord)
    log.log_step(step_record(1))
    log.log_event("home", "done")
    for name in ("session.yaml", "episodes.jsonl", "steps.jsonl", "events.jsonl"):
        assert (settings.session_dir / name).exists()
    eps = lines(settings.session_dir / "episodes.jsonl")
    assert [e["outcome"] for e in eps] == ["success", "fail", "stopped", "aborted"]
    e = eps[0]
    assert e["reason"] == "reason 1" and e["duration_s"] == 2.5 and e["steps_sent"] == 7
    assert e["skips"] == {"camera_stale": 4} and e["width_clips"] == 3 and e["odd_chunks"] == 2
    assert e["shortened"] == 5 and e["pulled_back"] == 6 and e["loop_late"] == 8 and e["anchor_fallbacks"] == 9
    assert np.array_equal(e["start_pose"], pose()) and datetime.fromisoformat(e["wall_start"])
    assert "policy_config" in (settings.session_dir / "session.yaml").read_text()
    log.close()


def test_lines_flushed_at_once(settings):
    log, _ = make_log(settings)
    log.log_event("startup", "x")
    log.log_step(step_record(1))
    assert len(lines(settings.session_dir / "events.jsonl")) == 1     # readable before close
    assert lines(settings.session_dir / "steps.jsonl")[0]["chunk_id"] == 1
    log.close()


def test_open_episode_closed_on_exit(settings):
    log, _ = make_log(settings)
    log.start_episode(1, pose())
    log.close()
    eps = lines(settings.session_dir / "episodes.jsonl")
    assert eps[0]["outcome"] == "aborted"


def test_log_steps_off(make_settings):
    s = make_settings(log_steps=False)
    log, _ = make_log(s)
    log.write_session(s, policy_config())
    log.log_step(step_record(1))
    assert (s.session_dir / "steps.jsonl").read_text() == ""
    log.close()


def done_step(chunk_id, episode=1):
    t = (9_950_000_000, 10_000_000_000)
    frames = make_frames(t)
    conv = PoseConverter(8, 0)
    obs, anchor = conv.to_observation(np.stack([pose(), pose()]), np.array([0.05, 0.05]), pose(), frames, t[1],
                                      "target")
    chunk = make_chunk()
    wp = conv.to_waypoints(chunk, anchor, chunk_id)
    return StepDone(StepJob(episode, chunk_id, pose(), None), wp, anchor, health(),
                    StepTiming(0, 1, 2, 3, 40.0), 0.0, obs, chunk, frames.main_t_ns, frames.ultrawide_t_ns)


def reply(chunk_id):
    return SetWaypointsReply(chunk_id=chunk_id, accepted=4, shortened=0, skipped=4, rejected=())


def test_no_tap_folder_without_save_tap(settings):
    log, _ = make_log(settings)
    log.write_session(settings, policy_config())
    log.save_tap(done_step(1), reply(1))
    assert not (settings.session_dir / "tap").exists()
    log.close()


def test_tap_files_and_images_every(make_settings):
    s = make_settings(save_tap=True, tap_images_every=3)
    log, _ = make_log(s)
    log.write_session(s, policy_config())
    log.start_episode(2, pose())
    for i in range(1, 8):
        log.save_tap(done_step(i, episode=2), reply(i))
    files = sorted((s.session_dir / "tap" / "ep_0002").glob("chunk_*.npz"))
    assert len(files) == 7
    with_images = []
    for i in range(1, 8):
        with np.load(s.session_dir / "tap" / "ep_0002" / f"chunk_{i}.npz") as z:
            for key in ("eef_pos", "eef_rot6d", "eef_rot6d_wrt_start", "gripper_width_m", "main_t_ns",
                        "ultrawide_t_ns", "anchor_T", "anchor_t_ns", "anchor_source", "start_pose", "chunk_eef_pos",
                        "chunk_eef_rot6d", "chunk_gripper_width_m", "chunk_width_clipped", "waypoints_t_ns",
                        "waypoints_O_T_TCP", "waypoints_gripper_width_m", "reply_accepted", "reply_skipped"):
                assert key in z.files
            assert z["chunk_eef_pos"].shape == (16, 3) and z["waypoints_t_ns"].shape == (8,)
            if "main" in z.files:
                with_images.append(i)
                assert z["main"].shape == (2, 224, 224, 3) and "ultrawide" in z.files
    assert with_images == [1, 4, 7]
    log.close()
