import numpy as np
import pytest

from contracts import (IMAGE_SIZE, OBS_STEPS, CameraHealthReport, FramesReply, GetFramesRequest, GetHealthRequest,
                       HealthReply, HealthReport, decode, encode)
from helpers import contract_error


def cam(status="ok", age=10.0, fps=30.0, dropped=0):
    return CameraHealthReport(status=status, last_frame_age_ms=age, fps=fps, dropped=dropped)


def health(main="ok", ultrawide="ok"):
    return HealthReport(main=cam(main), ultrawide=cam(ultrawide))


def images():
    return np.zeros((OBS_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def times(t0=100):
    return np.array([t0, t0 + 50_000_000], dtype=np.int64)


def reply(**change):
    d = dict(main=images(), main_t_ns=times(), ultrawide=images(), ultrawide_t_ns=times(), health=health())
    d.update(change)
    return FramesReply(**d)


NONE = dict(main=None, main_t_ns=None, ultrawide=None, ultrawide_t_ns=None)


def test_requests():
    assert decode(encode(GetFramesRequest())) == GetFramesRequest()
    assert decode(encode(GetHealthRequest())) == GetHealthRequest()


def test_shapes_use_the_size_constants():
    r = reply()
    assert r.main.shape == (OBS_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3) == (2, 224, 224, 3)
    contract_error(lambda: reply(main=np.zeros((3, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)), "main", "shape")
    contract_error(lambda: reply(ultrawide_t_ns=np.zeros(3, np.int64)), "ultrawide_t_ns", "shape")
    contract_error(lambda: reply(main=images().astype(np.float32)), "main", "dtype")


def test_both_ok_has_all_images():
    assert decode(encode(reply())) == reply()
    for name in NONE:
        contract_error(lambda: reply(**{name: None}), name, "value")


@pytest.mark.parametrize("main,ultrawide", [("stale", "ok"), ("ok", "down"), ("down", "stale")])
def test_not_ok_means_no_images_at_all(main, ultrawide):
    r = FramesReply(health=health(main, ultrawide), **NONE)
    assert decode(encode(r)).main is None
    contract_error(lambda: reply(health=health(main, ultrawide)), "main", "value")
    contract_error(lambda: FramesReply(health=health(main, ultrawide), **{**NONE, "ultrawide_t_ns": times()}),
                   "ultrawide_t_ns", "value")


def test_times_strictly_increase():
    contract_error(lambda: reply(main_t_ns=np.array([5, 5], np.int64)), "main_t_ns", "value")
    contract_error(lambda: reply(ultrawide_t_ns=np.array([9, 3], np.int64)), "ultrawide_t_ns", "value")


def test_camera_health_report():
    contract_error(lambda: cam(status="dead"), "status", "value")
    contract_error(lambda: cam(age=-1.0), "last_frame_age_ms", "value")
    contract_error(lambda: cam(fps=-0.1), "fps", "value")
    contract_error(lambda: cam(dropped=-1), "dropped", "value")
    contract_error(lambda: cam(fps=30), "fps", "type")


def test_health_reply_and_nested_paths():
    h = HealthReply(health=health("stale", "ok"))
    assert decode(encode(h)) == h
    d = h.to_dict()
    d["health"]["ultrawide"]["status"] = "gone"
    contract_error(lambda: HealthReply.from_dict(d), "health.ultrawide.status", "value")
