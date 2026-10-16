"""T14: ICM-42688-P protocol (FIFO parsing on hand-built bytes, scaling, timestamp unwrap, config sequence) and the
chip-clock mapper / FIFO decoder timing (synthetic drift)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from sensor_rig.drivers import icm42688_proto as P
from sensor_rig.drivers.chip_clock import ChipClockMapper, fit_lower_envelope
from sensor_rig.drivers.icm42688_spi import ImuFifoDecoder, parse_spi_device

H3 = 0x68   # 0110_10xx: accel + gyro + ODR timestamp (packet 3)
H4 = 0x78   # 0111_10xx: + HEADER_20 (packet 4)


def pkt3(acc, gyr, temp=0, tmst=0, header=H3) -> bytes:
    return struct.pack(">B3h3hbH", header, *acc, *gyr, temp, tmst)


def pkt4(acc20, gyr20, temp16=0, tmst=0, header=H4) -> bytes:
    hi = []
    ext = []
    for a, g in zip(acc20, gyr20):
        ua, ug = a & 0xFFFFF, g & 0xFFFFF
        hi.append((ua >> 4, ug >> 4))
        ext.append(((ua & 0xF) << 4) | (ug & 0xF))
    acc_hi = [h[0] for h in hi]
    gyr_hi = [h[1] for h in hi]
    b = struct.pack(">B3H3HhH", header, *acc_hi, *gyr_hi, temp16, tmst)
    return b + bytes(ext)


# ---------------------------------------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------------------------------------
def test_parse_packet3_basic_and_empty_stop():
    buf = pkt3((1, -2, 3), (-4, 5, -6), temp=10, tmst=1234) + pkt3((7, 8, 9), (10, 11, 12), temp=-3, tmst=2234)
    buf += bytes([0x80] + [0] * 15) + pkt3((0, 0, 0), (0, 0, 0))   # empty marker: stop there
    pk, used = P.parse_fifo(buf, 16)
    assert used == 32
    assert len(pk) == 2
    assert pk[0].accel_raw == (1, -2, 3) and pk[0].gyro_raw == (-4, 5, -6)
    assert pk[0].temp_raw == 10 and pk[0].tmst == 1234 and pk[0].header == H3
    assert pk[1].temp_raw == -3 and pk[1].tmst == 2234


def test_parse_invalid_and_partial():
    good = pkt3((100, 200, 300), (1, 2, 3), tmst=5)
    bad = pkt3((-32768, 0, 0), (1, 2, 3), tmst=6)
    bad_g = pkt3((1, 2, 3), (0, -32768, 0), tmst=7)
    buf = good + bad + bad_g + good[:9]          # trailing partial packet
    pk, used = P.parse_fifo(buf, 16)
    assert used == 48                            # partial packet left for the next read
    assert [p.tmst for p in pk] == [5]
    pk_all, _ = P.parse_fifo(buf, 16, keep_invalid=True)
    assert [p.valid for p in pk_all] == [True, False, False]
    assert [p.tmst for p in pk_all] == [5, 6, 7]


def test_parse_bad_header():
    with pytest.raises(P.FifoFormatError):
        P.parse_fifo(pkt3((0, 0, 0), (0, 0, 0), header=0x48), 16)      # accel only header for a 16-byte stream
    # a bad header after good packets: stop, consume the good ones only
    buf = pkt3((1, 1, 1), (1, 1, 1)) + pkt3((0, 0, 0), (0, 0, 0), header=0x20)
    pk, used = P.parse_fifo(buf, 16)
    assert len(pk) == 1 and used == 16
    # a 20-bit header in a 16-byte stream is out of sync
    with pytest.raises(P.FifoFormatError):
        P.parse_fifo(pkt3((0, 0, 0), (0, 0, 0), header=H4), 16)


def test_parse_packet4_20bit():
    acc = (123457, -98764, 524284)
    gyr = (-2, 4096 * 16 + 6, -524286)
    buf = pkt4(acc, gyr, temp16=-1325, tmst=65535)
    assert len(buf) == 20
    pk, used = P.parse_fifo(buf + pkt4(acc, gyr, tmst=1)[:7], 20)
    assert used == 20 and len(pk) == 1
    p = pk[0]
    assert p.hires and p.accel_raw == acc and p.gyro_raw == gyr and p.temp_raw == -1325 and p.tmst == 65535
    inv = pkt4((P.INVALID_20, 0, 0), (0, 0, 0))
    assert P.parse_fifo(inv, 20)[0] == []


def test_scaling():
    p = P.FifoPacket((2048, -1024, 0), (16384, 0, -32767), 20, 0, H3)
    a, g, t = P.scale(p, 16, 2000)
    np.testing.assert_allclose(a, [9.80665, -9.80665 / 2, 0.0])
    np.testing.assert_allclose(g, [np.deg2rad(1000.0), 0.0, np.deg2rad(-32767 / 16.384)])
    assert t == pytest.approx(20 / 2.07 + 25)
    a, g, _ = P.scale(p, 8, 500)
    assert a[0] == pytest.approx(9.80665 / 2) and g[0] == pytest.approx(np.deg2rad(250.0))
    ph = P.FifoPacket((32768, 0, 0), (262, 0, 0), 1324, 0, H4, hires=True)
    a, g, t = P.scale(ph, 4, 250)            # hi-res ignores the configured ranges (DS-000347 §6.1)
    assert a[0] == pytest.approx(9.80665) and g[0] == pytest.approx(np.deg2rad(262 / 262.144))
    assert t == pytest.approx(1324 / 132.48 + 25)
    with pytest.raises(ValueError):
        P.gyro_lsb_per_dps(300)


# ---------------------------------------------------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------------------------------------------------
def test_unwrap_three_wraps_in_chunks():
    rng = np.random.default_rng(1)
    true = 40_000 + np.cumsum(rng.integers(990, 1010, 260))     # ~260 ms > 3 wraps of 65.536 ms
    raw = true % 65536
    uw = P.TimestampUnwrapper()
    out = []
    i = 0
    while i < len(raw):
        k = int(rng.integers(1, 9))
        out.append(uw(raw[i:i + k]))
        i += k
    out = np.concatenate(out)
    assert (true[-1] - true[0]) > 3 * 65536
    np.testing.assert_array_equal(out - out[0], true - true[0])
    assert np.all(np.diff(out) > 0)


def test_unwrap_hint_and_tick():
    uw = P.TimestampUnwrapper()
    t_true = 5 * 65536 + 1234
    out = uw([t_true % 65536, (t_true + 1000) % 65536], hint_us=t_true + 20_000)
    np.testing.assert_array_equal(out, [t_true, t_true + 1000])
    uw16 = P.TimestampUnwrapper(tick_us=16)
    np.testing.assert_array_equal(uw16([65535, 0, 1]), [65535 * 16, 65536 * 16, 65537 * 16])


# ---------------------------------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------------------------------
def test_config_sequence_hires_1khz():
    seq = P.config_sequence(1000, 16, 2000, hires=True)
    assert seq == [
        (0x76, 0x00), (0x11, 0x01), (0x76, 0x00), (0x4C, 0x33), (0x4F, 0x06), (0x50, 0x06), (0x52, 0x11),
        (0x54, 0x21), (0x5F, 0x57), (0x16, 0x40), (0x4B, 0x02), (0x4E, 0x0F),
    ]
    assert P.POST_WRITE_DELAYS_S[0x11] >= 0.001 and P.POST_WRITE_DELAYS_S[0x4E] >= 0.045


def test_config_sequence_other_rates():
    seq = dict(P.config_sequence(200, 8, 500, accel_bw="odr_div_2", gyro_bw="low_latency_1"))
    assert seq[0x4F] == (2 << 5) | 0x07 == 0x47
    assert seq[0x50] == (1 << 5) | 0x07 == 0x27
    assert seq[0x52] == 0x0F
    assert seq[0x5F] == 0x47                      # no HIRES
    assert dict(P.config_sequence(500, 4, 1000))[0x4F] == (1 << 5) | 0x0F
    assert dict(P.config_sequence(1000, 16, 2000, tmst_res_16us=True))[0x54] == 0x29
    with pytest.raises(ValueError):
        P.config_sequence(400, 16, 2000)
    with pytest.raises(ValueError):
        P.config_sequence(1000, 32, 2000)


def test_spi_commands_and_device():
    assert P.spi_read_cmd(P.REG_FIFO_COUNTH, 2) == [0xAE, 0, 0]
    assert P.spi_read_cmd(P.REG_WHO_AM_I, 1) == [0xF5, 0]
    assert P.spi_write_cmd(P.REG_PWR_MGMT0, 0x0F) == [0x4E, 0x0F]
    assert parse_spi_device("/dev/spidev0.1") == (0, 1)
    with pytest.raises(ValueError):
        parse_spi_device("/dev/ttyUSB0")


# ---------------------------------------------------------------------------------------------------------------------
# chip clock mapping
# ---------------------------------------------------------------------------------------------------------------------
def _simulate_reads(ppm: float, seconds: float, seed: int = 0, poll_ms: float = 5.0, delay_mean_ms: float = 1.0,
                    overhead_us: float = 20.0, odr_hz: float = 1000.0):
    """Chip samples every 1/odr on the chip clock; host = h0 + chip_us * 1000 * (1 + ppm). Polls every poll_ms with an
    exponential scheduling delay; each read reports the newest sample latched before it."""
    rng = np.random.default_rng(seed)
    h0 = 7_000_000_000_000
    scale = 1000.0 * (1 + ppm * 1e-6)
    period_us = 1e6 / odr_hz
    n_polls = int(seconds * 1000 / poll_ms)
    reads = []
    for j in range(1, n_polls):
        t_read = h0 + j * poll_ms * 1e6 + rng.exponential(delay_mean_ms * 1e6)
        k_last = int(np.floor((t_read - h0) / scale / period_us))
        reads.append((k_last, int(t_read + overhead_us * 1e3)))
    return h0, scale, period_us, reads


@pytest.mark.parametrize("ppm", [50.0, -8000.0])
def test_chip_clock_mapper_drift(ppm):
    h0, scale, period_us, reads = _simulate_reads(ppm, 6.0)
    m = ChipClockMapper()
    errs = []
    for k_last, host_ns in reads:
        chip_us = int(round(k_last * period_us))
        m.add(chip_us, host_ns)
        t_host = host_ns - h0
        if t_host > 2e9:   # after 2 s warm-up
            ks = np.arange(max(0, k_last - 4), k_last + 1)
            chip = np.round(ks * period_us).astype(np.int64)
            truth = h0 + chip * scale
            errs.append(np.abs(m.map(chip) - truth).max())
    assert max(errs) < 100_000, f"max error {max(errs) / 1e3:.1f} us"
    assert abs(m.slope_ppm - ppm) < 30


def test_lower_envelope_exact_line():
    chip = np.arange(0, 10_000_000, 5000, dtype=np.int64)
    host = 123_456_789 + (chip * 1000.3).astype(np.int64)
    host[::7] += 3000                                   # positive delays only
    y0, a, b, x0 = fit_lower_envelope(chip, host, 990, 1010)
    assert b == pytest.approx(1000.3, abs=1e-3)
    # clamp
    _, _, b2, _ = fit_lower_envelope(chip, host, 999.0, 1000.1)
    assert b2 == pytest.approx(1000.1)


def test_fifo_decoder_end_to_end_with_overflow():
    """Fake FIFO bytes from a drifting chip clock -> decoder output times match the truth; an overflow (150 ms of
    samples lost, > 2 timestamp wraps) is recovered from the host clock."""
    ppm = 300.0
    h0, scale, period_us, reads = _simulate_reads(ppm, 5.0, seed=3)
    dec = ImuFifoDecoder(packet_size=16, accel_fs_g=16, gyro_fs_dps=2000)
    next_k = 0
    all_t, all_truth = [], []
    lost_from, lost_to = 3000, 3150                    # samples [3000, 3150) lost to a FIFO overflow
    pending_overflow = False
    batch = None
    for k_last, host_ns in reads:
        if k_last < next_k:
            continue
        if next_k <= lost_from <= k_last or lost_from <= next_k < lost_to:
            if k_last < lost_to:                       # still overflowing: nothing usable this read
                next_k = k_last + 1
                pending_overflow = True
                continue
            next_k = lost_to
            pending_overflow = True
        ks = np.arange(next_k, k_last + 1)
        chip = np.round(ks * period_us).astype(np.int64)
        buf = b"".join(pkt3((0, 0, 2048), (0, 0, 0), 25, int(c) % 65536) for c in chip)
        batch = dec.process(buf, host_ns, overflow=pending_overflow)
        pending_overflow = False
        assert batch is not None and len(batch) == len(ks)
        all_t.append(batch.t_ns)
        all_truth.append(h0 + chip * scale)
        next_k = k_last + 1
    t = np.concatenate(all_t)
    truth = np.concatenate(all_truth)
    assert np.all(np.diff(t) > 0)
    warm = truth > h0 + 2e9
    err = np.abs(t[warm] - truth[warm])
    assert err.max() < 100_000, f"max error {err.max() / 1e3:.1f} us"
    assert dec.stats["fifo_overflows"] == 1
    np.testing.assert_allclose(batch.accel[:, 2], 9.80665)


def test_hires_and_16bit_packets_agree():
    """The same physical sample decoded from a 16-byte packet (+-16 g, +-2000 dps) and from a 20-byte packet (16-bit
    value << 4 plus extension nibbles) gives the same SI value (up to the extra resolution)."""
    acc16 = (2048, -1000, 12345)             # 1 g on x
    gyr16 = (164, -16384, 7)                 # 10 dps on x, -1000 dps on y
    acc20 = tuple((v << 4) | 0x4 for v in acc16)   # extension bits: 18-bit accel, low two bits 0
    gyr20 = tuple((v << 4) | 0x6 for v in gyr16)   # 19-bit gyro, LSB 0
    p16 = P.parse_fifo(pkt3(acc16, gyr16), 16)[0][0]
    p20 = P.parse_fifo(pkt4(acc20, gyr20), 20)[0][0]
    a16, g16, _ = P.scale(p16, 16, 2000)
    a20, g20, _ = P.scale(p20, 16, 2000)
    np.testing.assert_allclose(a20, a16, atol=9.80665 * 16 / 32768)     # within one 16-bit LSB
    np.testing.assert_allclose(g20, g16, atol=np.deg2rad(2000 / 32768))
    assert a16[0] == pytest.approx(9.80665) and g16[0] == pytest.approx(np.deg2rad(164 / 16.384))
