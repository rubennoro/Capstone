"""Mid-exposure times match the training formula t_pi_ns - exposure_ns // 2 + time_offset_ns (STRUCTURE.md §6.1)."""

from __future__ import annotations

import threading

import pytest

from conftest import RawDecoder, raw_frame, write_rig
from process_frames._dataproc.config import load_rig
from process_frames.camera_health import CameraHealth
from process_frames.clock_map import ClockMap, ClockSample
from process_frames.frame_buffer import FrameBuffer
from process_frames.receiver import Receiver
from process_frames.replay_source import ReplaySource

OFFSET = 1_000_000_000_000          # Pi minus PC
DEFAULT_EXP = 5_000_000


def receiver_for(rig, frames, clock):
    cm = ClockMap(60.0, 0.3, 10.0, 500.0, 10.0)
    cm.add_sample(ClockSample(t_send_ns=clock(), t_pi_ns=clock() + OFFSET, t_recv_ns=clock()))
    buffers = {c: FrameBuffer(c, 1.0) for c in ("main", "ultrawide")}
    health = {c: CameraHealth(c, 0.35, 2.0, 1.0, clock) for c in ("main", "ultrawide")}
    rx = Receiver(ReplaySource(frames), {c: RawDecoder() for c in buffers}, rig, cm, buffers, health, DEFAULT_EXP,
                  clock)
    return rx, buffers


def test_rig_formula(small_rig):
    assert small_rig.camera_time_ns("main", 10_000_000_000, 8_000_001) == 10_000_000_000 - 4_000_000
    assert small_rig.camera_time_ns("ultrawide", 7, 0) == 7


def test_start_of_exposure_rig_gives_plus(tmp_path):
    def change(d):
        d["cameras"]["main"]["timestamp_ref"] = "start_of_exposure"
        d["cameras"]["main"]["time_offset_ns"] = 250
    rig = load_rig(write_rig(tmp_path, "rig_soe", change))
    assert rig.camera_time_ns("main", 10_000, 4_000) == 10_000 + 2_000 + 250


def test_receiver_buffered_time_is_formula_minus_clock_offset(small_rig, fake_clock):
    t_pi = fake_clock.now + OFFSET - 5_000_000
    frames = [raw_frame("main", 0, t_pi, exposure_ns=8_000_001), raw_frame("ultrawide", 0, t_pi + 11, exposure_ns=0)]
    rx, buffers = receiver_for(small_rig, frames, fake_clock)
    assert rx.receive_one(0.0) and rx.receive_one(0.0)
    m = buffers["main"].newest()
    assert m.t_pi_mid_ns == t_pi - 8_000_001 // 2
    assert m.t_pc_ns == t_pi - 8_000_001 // 2 - OFFSET
    u = buffers["ultrawide"].newest()
    exp = min(DEFAULT_EXP, round(1e9 / 30) // 2)          # exposure 0: the default, capped at half a period
    assert exp == DEFAULT_EXP
    assert u.t_pc_ns == t_pi + 11 - exp // 2 - OFFSET


def test_default_exposure_is_capped_at_half_a_period(tmp_path, fake_clock):
    rig = load_rig(write_rig(tmp_path, "rig_fast", lambda d: d["cameras"]["main"].update(fps=200)))
    rx, buffers = receiver_for(rig, [raw_frame("main", 0, fake_clock.now + OFFSET)], fake_clock)
    assert rx.default_exposure_ns("main") == round(1e9 / 200) // 2 == 2_500_000
    rx.receive_one(0.0)
    assert buffers["main"].newest().t_pi_mid_ns == fake_clock.now + OFFSET - 2_500_000 // 2
