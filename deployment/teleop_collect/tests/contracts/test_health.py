# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/contracts/test_c1.py @ 2ffe641
# changes: only the health Records (test_camera_health_report; the nested-path test through R1's FramesHealth instead
# of C1's HealthReply); the C1 message tests dropped; imports.
"""The cameras' health Records (STRUCTURE.md 4): CameraHealthReport and HealthReport."""
from __future__ import annotations

from deploy_contracts import decode, encode
from station_helpers import contract_error

from teleop_collect.contracts import CAMERAS, CameraHealthReport, FramesHealth, HealthReport


def cam(status="ok", age=10.0, fps=30.0, dropped=0):
    return CameraHealthReport(status=status, last_frame_age_ms=age, fps=fps, dropped=dropped)


def health(main="ok", ultrawide="ok"):
    return HealthReport(main=cam(main), ultrawide=cam(ultrawide))


def test_camera_health_report():
    contract_error(lambda: cam(status="dead"), "status", "value")
    contract_error(lambda: cam(age=-1.0), "last_frame_age_ms", "value")
    contract_error(lambda: cam(fps=-0.1), "fps", "value")
    contract_error(lambda: cam(dropped=-1), "dropped", "value")
    contract_error(lambda: cam(fps=30), "fps", "type")


def test_health_and_nested_paths():
    assert CAMERAS == ("main", "ultrawide")
    h = FramesHealth(health=health("stale", "ok"), synced=True, recording=False, recording_error="", frames_main=0,
                     frames_ultrawide=0)
    assert decode(encode(h)) == h
    d = h.to_dict()
    d["health"]["ultrawide"]["status"] = "gone"
    contract_error(lambda: FramesHealth.from_dict(d), "health.ultrawide.status", "value")
