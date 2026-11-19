# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/frames/test_frame_times.py @ 2ffe641
# changes: imports; the times are checked on the RecordItem (no PacketBuffer, no RawDecoder); + the ultrawide mapped mid
# adapted from rnoro/deployment:deployment/process_frames/tests/test_frame_times.py @ 91b178b
# changes: imports; ClockMap from contracts.clock_fit (t_remote_ns); t_ns (laptop clock); the pair gap is mid-exposure based
"""Mid-exposure times match the training formula t_pi_ns - exposure_ns // 2 + time_offset_ns (STRUCTURE.md 7.2)."""

from __future__ import annotations

from frames_helpers import raw_frame, raw_packet, write_rig
from teleop_collect.frames._dataproc.config import load_rig
from teleop_collect.frames.camera_health import CameraHealth
from teleop_collect.contracts.clock_fit import ClockMap, ClockSample
from teleop_collect.frames.receiver import Receiver
from teleop_collect.frames.recording_packet_source import RecordingPacketSource

OFFSET = 1_000_000_000_000          # Pi minus laptop
DEFAULT_EXP = 5_000_000


def receiver_for(rig, packets, clock):
    cm = ClockMap(60.0, 0.3, 10.0, 500.0, 10.0)
    cm.add_sample(ClockSample(t_send_ns=clock(), t_remote_ns=clock() + OFFSET, t_recv_ns=clock()))
    items = []
    health = {c: CameraHealth(c, 0.35, 2.0, 1.0, clock) for c in ("main", "ultrawide")}
    rx = Receiver(RecordingPacketSource(packets), rig, cm, health, DEFAULT_EXP, 1.0, clock, sinks=[items.append])
    return rx, items


def test_rig_formula(small_rig):
    assert small_rig.camera_time_ns("main", 10_000_000_000, 8_000_001) == 10_000_000_000 - 4_000_000
    assert small_rig.camera_time_ns("ultrawide", 7, 0) == 7


def test_start_of_exposure_rig_gives_plus(tmp_path):
    def change(d):
        d["cameras"]["main"]["timestamp_ref"] = "start_of_exposure"
        d["cameras"]["main"]["time_offset_ns"] = 250
    rig = load_rig(write_rig(tmp_path, "rig_soe", change))
    assert rig.camera_time_ns("main", 10_000, 4_000) == 10_000 + 2_000 + 250


def test_packet_time_is_the_main_formula_minus_clock_offset(small_rig, fake_clock):
    """The packet time is the main frame's mid-exposure time; the pair's own capture time is only kept."""
    t_pi = fake_clock.now + OFFSET - 5_000_000
    pk = [raw_packet(raw_frame("main", 0, t_pi, exposure_ns=8_000_001),
                     [raw_frame("ultrawide", 0, t_pi - 11_000_000, exposure_ns=3_000_000)], 0),
          raw_packet(raw_frame("main", 1, t_pi + 33_333_333, exposure_ns=0), [], 0)]
    rx, items = receiver_for(small_rig, pk, fake_clock)
    assert rx.receive_one(0.0)
    p = items[-1]
    assert p.main_t_mid_pi_ns == t_pi - 8_000_001 // 2
    assert p.main_t_mid_ns == t_pi - 8_000_001 // 2 - OFFSET
    assert p.ultrawide.info.t_pi_ns == t_pi - 11_000_000
    assert p.ultrawide_t_mid_pi_ns == t_pi - 11_000_000 - 1_500_000
    assert p.ultrawide_t_mid_ns == t_pi - 11_000_000 - 1_500_000 - OFFSET
    assert p.pair_gap_ms * 1e6 == (11_000_000 + 1_500_000 - 8_000_001 // 2)       # mid-exposures
    assert rx.receive_one(0.0)
    exp = min(DEFAULT_EXP, round(1e9 / 30) // 2)          # exposure 0: the default, capped at half a period
    assert exp == DEFAULT_EXP
    assert items[-1].main_t_mid_ns == t_pi + 33_333_333 - exp // 2 - OFFSET


def test_default_exposure_is_capped_at_half_a_period(tmp_path, fake_clock):
    rig = load_rig(write_rig(tmp_path, "rig_fast", lambda d: d["cameras"]["main"].update(fps=200)))
    rx, items = receiver_for(rig, [raw_packet(raw_frame("main", 0, fake_clock.now + OFFSET))], fake_clock)
    assert rx.default_exposure_ns() == round(1e9 / 200) // 2 == 2_500_000
    rx.receive_one(0.0)
    assert items[-1].main_t_mid_pi_ns == fake_clock.now + OFFSET - 2_500_000 // 2
