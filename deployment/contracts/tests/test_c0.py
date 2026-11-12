import dataclasses

import pytest

from contracts import Frame, TimeSyncReply, TimeSyncRequest, decode, encode
from helpers import contract_error


def frame(**change):
    d = dict(camera="main", sensor="imx519", seq=0, t_pi_ns=1_000, exposure_ns=0, width=1600, height=900,
             encoding="h264", keyframe=False, data=b"\x00\x00\x01\x41")
    d.update(change)
    return Frame(**d)


def test_good_frames():
    assert decode(encode(frame())) == frame()
    assert frame(encoding="jpeg", keyframe=True).encoding == "jpeg"
    assert frame(camera="ultrawide", sensor="imx219").camera == "ultrawide"
    assert [f.name for f in dataclasses.fields(Frame)] == ["camera", "sensor", "seq", "t_pi_ns", "exposure_ns",
                                                            "width", "height", "encoding", "keyframe", "data"]


@pytest.mark.parametrize("field,value", [("camera", "left"), ("sensor", "imx477"), ("encoding", "png"),
                                         ("exposure_ns", -1), ("width", 0), ("height", -2), ("seq", -1)])
def test_enums_and_ranges(field, value):
    contract_error(lambda: frame(**{field: value}), field, "value")


def test_jpeg_must_be_a_keyframe():
    contract_error(lambda: frame(encoding="jpeg", keyframe=False), "keyframe", "value")


def test_data_non_empty_bytes():
    contract_error(lambda: frame(data=b""), "data", "value")
    contract_error(lambda: frame(data="abc"), "data", "type")


def test_time_sync():
    assert decode(encode(TimeSyncRequest())) == TimeSyncRequest()
    assert TimeSyncRequest().to_dict() == {"schema": "c0.time_sync/v1"}
    assert decode(encode(TimeSyncReply(t_pi_ns=-5))).t_pi_ns == -5
    contract_error(lambda: TimeSyncReply(t_pi_ns=1.5), "t_pi_ns", "type")
