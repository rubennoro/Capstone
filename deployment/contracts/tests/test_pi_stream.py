"""pi_stream/v1: the Pi's 3-part wire format (JSON header, main JPEG, ultrawide JPEG)."""
from __future__ import annotations

import copy
import json

import pytest

from contracts import ContractError, PiStreamHeader, PiStreamPacket, jpeg_size

# A header as the real Pi sent it (2026-11-12, both cameras 640x480 @ 30).
REAL = {"version": 1, "packet_no": 800, "t_send_ns": 1269437832529, "clock_offset_ns": 1794539164607668815,
        "main": {"frame_no": 800, "t_sensor_ns": 1269392013000, "exposure_us": 25640, "t_mid_ns": 1269379193000},
        "ultrawide": {"frame_no": 799, "t_sensor_ns": 1269412129000, "exposure_us": 33287,
                      "t_mid_ns": 1269395485500},
        "skew_ns": 16292500, "uw_reused": False}


def tiny_jpeg(w: int, h: int) -> bytes:
    """A minimal byte string with SOI, an APP0 segment and a baseline SOF0 header (enough for jpeg_size)."""
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0\x00\x11\x08" + h.to_bytes(2, "big") + w.to_bytes(2, "big") + b"\x03" + b"\x01\x22\x00" * 3
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xda\x00\x02" + b"\x00" * 8 + b"\xff\xd9"


def parts(h: dict | None = None) -> list[bytes]:
    return [json.dumps(h if h is not None else REAL).encode(), tiny_jpeg(640, 480), tiny_jpeg(640, 480)]


def broken(path: str, value=None, delete=False) -> dict:
    h = copy.deepcopy(REAL)
    keys = path.split(".")
    d = h
    for k in keys[:-1]:
        d = d[k]
    if delete:
        del d[keys[-1]]
    else:
        d[keys[-1]] = value
    return h


def test_the_real_pi_header_parses_and_round_trips():
    p = PiStreamPacket.from_parts(parts())
    h = p.header
    assert h.packet_no == 800 and h.main.frame_no == 800 and h.ultrawide.frame_no == 799 and h.uw_reused is False
    assert h.main.t_mid_ns == h.main.t_sensor_ns - h.main.exposure_us * 1000 // 2       # the Pi's own rule
    assert json.loads(h.to_json()) == REAL
    assert PiStreamPacket.from_parts(p.to_parts()) == p


@pytest.mark.parametrize("msg, field, problem", [
    ([b"{}"], "parts", "value"),
    (parts() + [b"x"], "parts", "value"),
    ([b"not json", b"a", b"b"], "parts[0]", "value"),
    ([b"[1, 2]", b"a", b"b"], "parts[0]", "not_a_map"),
    ([json.dumps(REAL).encode(), b"", b"b"], "parts[1]", "value"),
    ([json.dumps(REAL).encode(), b"a", b""], "parts[2]", "value"),
])
def test_part_problems(msg, field, problem):
    with pytest.raises(ContractError) as e:
        PiStreamPacket.from_parts(msg)
    assert (e.value.field, e.value.problem, e.value.schema) == (field, problem, "pi_stream/v1")


@pytest.mark.parametrize("h, field, problem", [
    (broken("skew_ns", delete=True), "skew_ns", "missing"),
    (broken("main.t_mid_ns", delete=True), "main.t_mid_ns", "missing"),
    (broken("extra", 1), "extra", "unknown_field"),
    (broken("ultrawide.extra", 1), "ultrawide.extra", "unknown_field"),
    (broken("version", 2), "version", "schema_version"),
    (broken("packet_no", "800"), "packet_no", "type"),
    (broken("packet_no", True), "packet_no", "type"),
    (broken("packet_no", -1), "packet_no", "value"),
    (broken("main.frame_no", 1.5), "main.frame_no", "type"),
    (broken("main.exposure_us", -5), "main.exposure_us", "value"),
    (broken("uw_reused", 0), "uw_reused", "type"),
    (broken("main", [1]), "main", "not_a_map"),
])
def test_header_problems(h, field, problem):
    with pytest.raises(ContractError) as e:
        PiStreamHeader.from_json(json.dumps(h).encode())
    assert (e.value.field, e.value.problem) == (field, problem)


def test_jpeg_size_reads_the_frame_header():
    assert jpeg_size(tiny_jpeg(640, 480)) == (640, 480)
    assert jpeg_size(tiny_jpeg(1600, 1200)) == (1600, 1200)
    for bad in (b"", b"\x89PNG....", b"\xff\xd8\xff\xda\x00\x02\x00\x00"):
        with pytest.raises(ContractError):
            jpeg_size(bad)


def test_jpeg_size_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    ok, buf = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))
    assert ok and jpeg_size(buf.tobytes()) == (640, 480)
    ok, buf = cv2.imencode(".jpg", np.zeros((96, 128, 3), np.uint8), [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
    assert ok and jpeg_size(buf.tobytes()) == (128, 96)
