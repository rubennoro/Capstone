"""TDK ICM-42688-P register map, configuration sequence and FIFO decoding (pure; unit-tested on fake bytes).

Source / verification
---------------------
Values checked against **TDK InvenSense DS-000347, ICM-42688-P datasheet, Revision 1.6** (PDF copy; the current
TDK revision is 1.8, 2023-07-27, which we could not download - re-check the items marked [rev1.6] against it):

* Register addresses (bank 0): DEVICE_CONFIG 0x11, INT_CONFIG 0x14, FIFO_CONFIG 0x16, TEMP_DATA1 0x1D, INT_STATUS 0x2D,
  FIFO_COUNTH 0x2E, FIFO_DATA 0x30, SIGNAL_PATH_RESET 0x4B, INTF_CONFIG0 0x4C, INTF_CONFIG1 0x4D, PWR_MGMT0 0x4E,
  GYRO_CONFIG0 0x4F, ACCEL_CONFIG0 0x50, GYRO_ACCEL_CONFIG0 0x52, TMST_CONFIG 0x54, FIFO_CONFIG1 0x5F,
  WHO_AM_I 0x75 (= 0x47), REG_BANK_SEL 0x76. Datasheet §14 + Linux ``inv_icm42600.h`` + Zephyr ``icm4268x_reg.h``.
* GYRO_CONFIG0 / ACCEL_CONFIG0: FS in bits 7:5, ODR in bits 3:0; ODR codes 32k 1, 16k 2, 8k 3, 4k 4, 2k 5, 1k 6,
  200 7, 100 8, 50 9, 25 10, 12.5 11, 500 15. FS codes gyro 2000/1000/500/250/125 dps = 0..4; accel 16/8/4/2 g = 0..3
  (datasheet §14.37-14.38).
* Sensitivities (datasheet tables 1-2): gyro 16.4/32.8/65.5/131/262 LSB/dps, accel 2048/4096/8192/16384 LSB/g. We use
  the exact values 32768/FS (the Linux driver does too); 16.4 is the datasheet's rounding of 16.384.
* 20-bit FIFO (packet 4, datasheet §6.1): FS forced to +-2000 dps / +-16 g; invalid = -524288. The 20-bit word
  holds 19-bit gyro data (LSB always 0, i.e. value << 1) and 18-bit accel data (two LSBs 0, i.e. value << 2). The
  datasheet's 131 LSB/dps and 8192 LSB/g apply to those 19-/18-bit values, so on the raw 20-bit word the scale is
  2x131 = 262.144 LSB/dps and 4x8192 = 32768 LSB/g, i.e. exactly 16x the 16-bit +-2000 dps / +-16 g sensitivities
  (the Zephyr driver does the same). The static gravity check in the hardware checklist confirms it (|a| = 9.81).
* FIFO packet 3 (16 B): header, accel XYZ (BE int16), gyro XYZ (BE int16), temp int8, timestamp BE uint16.
  Packet 4 (20 B): header, accel XYZ [19:4], gyro XYZ [19:4], temp BE int16, timestamp BE uint16, then 3 bytes
  ``accel[3:0] << 4 | gyro[3:0]`` for X, Y, Z (datasheet §6.1).
* FIFO header (§6.2): bit7 HEADER_MSG (1 = FIFO empty), bit6 ACCEL, bit5 GYRO, bit4 HEADER_20, bits3:2 TIMESTAMP_FSYNC
  (10 = ODR timestamp, 11 = FSYNC time), bit1 ODR_ACCEL changed, bit0 ODR_GYRO changed.
* Temperature: FIFO 8-bit ``raw / 2.07 + 25``; 16-bit (registers and packet 4) ``raw / 132.48 + 25`` degC.
* Invalid samples with INTF_CONFIG0.FIFO_HOLD_LAST_DATA_EN = 0: -32768 (16-bit), -524288 (20-bit).
* INTF_CONFIG0 (reset 0x30): bit6 FIFO_COUNT_REC (0 = bytes), bit5 FIFO_COUNT_ENDIAN (1 = BE), bit4 SENSOR_DATA_ENDIAN
  (1 = BE), bits1:0 UI_SIFS_CFG (11 = disable I2C).
* FIFO_CONFIG1 (reset 0x00): bit6 FIFO_RESUME_PARTIAL_RD, bit4 FIFO_HIRES_EN, bit3 FIFO_TMST_FSYNC_EN, bit2 TEMP_EN,
  bit1 GYRO_EN, bit0 ACCEL_EN. With FIFO_TMST_FSYNC_EN = 0 the FIFO carries ODR timestamps only (§6.4 table).
* TMST_CONFIG (reset 0x23): bit4 TMST_TO_REGS_EN, bit3 TMST_RES (0 = 1 us), bit2 TMST_DELTA_EN, bit1 TMST_FSYNC_EN,
  bit0 TMST_EN. We write 0x21: keep reserved bit5 at its reset value, 1 us absolute timestamps, FSYNC off.
* FIFO_CONFIG bits 7:6 FIFO_MODE (01 = stream). SIGNAL_PATH_RESET bit1 FIFO_FLUSH. PWR_MGMT0 bits 3:2 gyro mode,
  1:0 accel mode (11 = low noise) -> 0x0F; "do not issue register writes for 200 us" after turning sensors on;
  "gyroscope needs to be kept ON for a minimum of 45 ms". Soft reset: "wait 1 ms" (§14.1).
* GYRO_ACCEL_CONFIG0 (reset 0x11): accel UI filter BW bits 7:4, gyro bits 3:0; 1 = max(400 Hz, ODR)/4.
* FIFO size 2048 B (§6.3); SPI read = address | 0x80 (§ SPI operational features); max SCLK 24 MHz.
* Internal clock accuracy (§ CLKIN): ODR (and the 1 us timestamp tick) can be off by up to +-1 % with the PLL
  (+-8 % on the RC oscillator). The host-side clock mapper must therefore allow percent-level slope, not ppm.

Not verified (behaviour assumptions, check on hardware): that FIFO_COUNT read latches the count used for the
following burst read; that stream mode overwrites the oldest data on overflow (we treat overflow as data loss either
way and resynchronise timestamps from the host clock).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665
DEG2RAD = np.pi / 180.0

# ---- registers (bank 0) ----
REG_DEVICE_CONFIG = 0x11
REG_INT_CONFIG = 0x14
REG_FIFO_CONFIG = 0x16
REG_TEMP_DATA1 = 0x1D
REG_INT_STATUS = 0x2D
REG_FIFO_COUNTH = 0x2E
REG_FIFO_COUNTL = 0x2F
REG_FIFO_DATA = 0x30
REG_SIGNAL_PATH_RESET = 0x4B
REG_INTF_CONFIG0 = 0x4C
REG_INTF_CONFIG1 = 0x4D
REG_PWR_MGMT0 = 0x4E
REG_GYRO_CONFIG0 = 0x4F
REG_ACCEL_CONFIG0 = 0x50
REG_GYRO_ACCEL_CONFIG0 = 0x52
REG_TMST_CONFIG = 0x54
REG_FIFO_CONFIG1 = 0x5F
REG_WHO_AM_I = 0x75
REG_BANK_SEL = 0x76

WHO_AM_I_VALUE = 0x47
SPI_READ = 0x80
FIFO_SIZE_BYTES = 2048

# ---- bit fields ----
DEVICE_CONFIG_SOFT_RESET = 0x01
FIFO_MODE_BYPASS = 0x00
FIFO_MODE_STREAM = 0x40
SIGNAL_PATH_RESET_FIFO_FLUSH = 0x02
INTF_CONFIG0_FIFO_HOLD_LAST_DATA_EN = 0x80
INTF_CONFIG0_FIFO_COUNT_REC = 0x40
INTF_CONFIG0_FIFO_COUNT_BE = 0x20
INTF_CONFIG0_SENSOR_DATA_BE = 0x10
INTF_CONFIG0_UI_SIFS_DISABLE_I2C = 0x03
PWR_MGMT0_GYRO_LN = 0x0C
PWR_MGMT0_ACCEL_LN = 0x03
TMST_CONFIG_RESERVED_BIT5 = 0x20
TMST_CONFIG_TO_REGS_EN = 0x10
TMST_CONFIG_RES_16US = 0x08
TMST_CONFIG_DELTA_EN = 0x04
TMST_CONFIG_FSYNC_EN = 0x02
TMST_CONFIG_EN = 0x01
FIFO_CONFIG1_RESUME_PARTIAL_RD = 0x40
FIFO_CONFIG1_HIRES_EN = 0x10
FIFO_CONFIG1_TMST_FSYNC_EN = 0x08
FIFO_CONFIG1_TEMP_EN = 0x04
FIFO_CONFIG1_GYRO_EN = 0x02
FIFO_CONFIG1_ACCEL_EN = 0x01

HEADER_MSG = 0x80
HEADER_ACCEL = 0x40
HEADER_GYRO = 0x20
HEADER_20 = 0x10
HEADER_TS_MASK = 0x0C
HEADER_TS_ODR = 0x08
HEADER_TS_FSYNC = 0x0C

# ---- codes ----
ODR_CODES = {32000: 0x01, 16000: 0x02, 8000: 0x03, 4000: 0x04, 2000: 0x05, 1000: 0x06, 200: 0x07, 100: 0x08,
             50: 0x09, 25: 0x0A, 500: 0x0F}
GYRO_FS_CODES = {2000: 0, 1000: 1, 500: 2, 250: 3, 125: 4, 62.5: 5, 31.25: 6, 15.625: 7}
ACCEL_FS_CODES = {16: 0, 8: 1, 4: 2, 2: 3}
UI_FILTER_BW_CODES = {"odr_div_2": 0, "odr_div_4": 1, "odr_div_5": 2, "odr_div_8": 3, "odr_div_10": 4,
                      "odr_div_16": 5, "odr_div_20": 6, "odr_div_40": 7,
                      # 14: low latency, Dec2 at max(400 Hz, ODR); 15: low latency, Dec2 at max(200 Hz, 8 x ODR)
                      "low_latency_0": 14, "low_latency_1": 15, "low_latency_400": 14, "low_latency_200_8x": 15}

# Scale of the raw 20-bit FIFO word (= 16 x the 16-bit sensitivity at +-2000 dps / +-16 g, see docstring).
HIRES_GYRO_LSB_PER_DPS = 16 * 32768.0 / 2000.0     # 262.144 (= 2 x 131 on the 19-bit value)
HIRES_ACCEL_LSB_PER_G = 16 * 32768.0 / 16.0        # 32768   (= 4 x 8192 on the 18-bit value)
INVALID_16 = -32768
INVALID_20 = -524288
TEMP_FIFO8_LSB_PER_C = 2.07
TEMP_16_LSB_PER_C = 132.48

PACKET3_SIZE = 16
PACKET4_SIZE = 20
PACKET_FORMATS = {"16byte": PACKET3_SIZE, "packet3": PACKET3_SIZE, "20byte_highres_ts": PACKET4_SIZE,
                  "packet4": PACKET4_SIZE}

# Delays to respect after writing a register (seconds).
SOFT_RESET_DELAY_S = 0.002      # datasheet: 1 ms
PWR_ON_DELAY_S = 0.050          # 200 us no-write window + gyro start-up (gyro must stay on >= 45 ms)
POST_WRITE_DELAYS_S = {REG_DEVICE_CONFIG: SOFT_RESET_DELAY_S, REG_PWR_MGMT0: PWR_ON_DELAY_S}


def gyro_lsb_per_dps(fs_dps: float) -> float:
    if fs_dps not in GYRO_FS_CODES:
        raise ValueError(f"invalid gyro full scale {fs_dps} dps (valid: {sorted(GYRO_FS_CODES)})")
    return 32768.0 / float(fs_dps)


def accel_lsb_per_g(fs_g: int) -> float:
    if fs_g not in ACCEL_FS_CODES:
        raise ValueError(f"invalid accel full scale {fs_g} g (valid: {sorted(ACCEL_FS_CODES)})")
    return 32768.0 / float(fs_g)


def packet_size(fmt: str | int) -> int:
    if isinstance(fmt, int):
        if fmt not in (PACKET3_SIZE, PACKET4_SIZE):
            raise ValueError("packet size must be 16 or 20")
        return fmt
    if fmt not in PACKET_FORMATS:
        raise ValueError(f"unknown FIFO packet format {fmt!r} (valid: {sorted(PACKET_FORMATS)})")
    return PACKET_FORMATS[fmt]


def config_sequence(odr_hz: int, accel_fs_g: int, gyro_fs_dps: float, hires: bool = False,
                    accel_bw: str | int = "odr_div_4", gyro_bw: str | int = "odr_div_4",
                    tmst_res_16us: bool = False) -> list[tuple[int, int]]:
    """Register writes ``(reg, value)`` that configure the chip for FIFO streaming. Respect
    ``POST_WRITE_DELAYS_S[reg]`` after each write.

    Order: bank 0, soft reset, bank 0, interface (BE data + count in bytes, I2C off), gyro/accel FS+ODR, UI filters,
    timestamps (1 us, absolute, FSYNC off), FIFO content (accel+gyro+temp [+hires], partial reads allowed), FIFO
    stream mode, FIFO flush, power on (gyro+accel low-noise).
    """
    if odr_hz not in ODR_CODES:
        raise ValueError(f"invalid ODR {odr_hz} Hz (valid: {sorted(ODR_CODES)})")
    odr = ODR_CODES[odr_hz]
    afs = ACCEL_FS_CODES.get(accel_fs_g)
    gfs = GYRO_FS_CODES.get(gyro_fs_dps)
    if afs is None or gfs is None:
        raise ValueError(f"invalid full scale: accel {accel_fs_g} g, gyro {gyro_fs_dps} dps")
    abw = UI_FILTER_BW_CODES[accel_bw] if isinstance(accel_bw, str) else int(accel_bw)
    gbw = UI_FILTER_BW_CODES[gyro_bw] if isinstance(gyro_bw, str) else int(gyro_bw)
    tmst = TMST_CONFIG_RESERVED_BIT5 | TMST_CONFIG_EN | (TMST_CONFIG_RES_16US if tmst_res_16us else 0)
    fifo1 = (FIFO_CONFIG1_RESUME_PARTIAL_RD | FIFO_CONFIG1_TEMP_EN | FIFO_CONFIG1_GYRO_EN | FIFO_CONFIG1_ACCEL_EN
             | (FIFO_CONFIG1_HIRES_EN if hires else 0))
    return [
        (REG_BANK_SEL, 0x00),
        (REG_DEVICE_CONFIG, DEVICE_CONFIG_SOFT_RESET),
        (REG_BANK_SEL, 0x00),
        (REG_INTF_CONFIG0, INTF_CONFIG0_FIFO_COUNT_BE | INTF_CONFIG0_SENSOR_DATA_BE | INTF_CONFIG0_UI_SIFS_DISABLE_I2C),
        (REG_GYRO_CONFIG0, (gfs << 5) | odr),
        (REG_ACCEL_CONFIG0, (afs << 5) | odr),
        (REG_GYRO_ACCEL_CONFIG0, ((abw & 0x0F) << 4) | (gbw & 0x0F)),
        (REG_TMST_CONFIG, tmst),
        (REG_FIFO_CONFIG1, fifo1),
        (REG_FIFO_CONFIG, FIFO_MODE_STREAM),
        (REG_SIGNAL_PATH_RESET, SIGNAL_PATH_RESET_FIFO_FLUSH),
        (REG_PWR_MGMT0, PWR_MGMT0_GYRO_LN | PWR_MGMT0_ACCEL_LN),
    ]


def spi_read_cmd(reg: int, n: int) -> list[int]:
    """SPI transfer for reading ``n`` bytes starting at ``reg`` (address | 0x80, then n dummy bytes)."""
    return [(reg & 0x7F) | SPI_READ] + [0] * n


def spi_write_cmd(reg: int, value: int) -> list[int]:
    return [reg & 0x7F, value & 0xFF]


@dataclass
class FifoPacket:
    accel_raw: tuple[int, int, int]   # 16-bit, or 20-bit for packet 4
    gyro_raw: tuple[int, int, int]
    temp_raw: int                     # int8 (packet 3) or int16 (packet 4)
    tmst: int                         # 16-bit timestamp field
    header: int
    hires: bool = False
    valid: bool = True                # False if accel or gyro holds the invalid marker

    @property
    def tmst_is_fsync(self) -> bool:
        return (self.header & HEADER_TS_MASK) == HEADER_TS_FSYNC


class FifoFormatError(ValueError):
    pass


def _s16(b: bytes, i: int) -> int:
    v = (b[i] << 8) | b[i + 1]
    return v - 0x10000 if v & 0x8000 else v


def parse_fifo(buf: bytes, packet_size: int = PACKET3_SIZE, keep_invalid: bool = False
               ) -> tuple[list[FifoPacket], int]:
    """Decode consecutive accel+gyro FIFO packets (packet 3: 16 B, packet 4: 20 B).

    Stops at an "empty" header (bit7), at a trailing partial packet, or at a header that does not match the configured
    packet type (raises ``FifoFormatError`` if that happens on the very first packet, since then the stream is out of
    sync and the caller should flush the FIFO; otherwise the bad tail is left unconsumed). Packets with invalid samples
    are dropped unless ``keep_invalid`` (they still carry a valid timestamp). Returns ``(packets, bytes_consumed)``.
    """
    if packet_size not in (PACKET3_SIZE, PACKET4_SIZE):
        raise ValueError("packet_size must be 16 or 20")
    hires = packet_size == PACKET4_SIZE
    out: list[FifoPacket] = []
    pos = 0
    n = len(buf)
    while pos + packet_size <= n:
        h = buf[pos]
        if h & HEADER_MSG:   # FIFO empty (reads past the end return 0x80/0xFF fill): stop, consume nothing more
            break
        if (h & (HEADER_ACCEL | HEADER_GYRO)) != (HEADER_ACCEL | HEADER_GYRO) or \
                (not hires and (h & HEADER_20)) or (h & HEADER_TS_MASK) not in (HEADER_TS_ODR, HEADER_TS_FSYNC):
            if not out and pos == 0:
                raise FifoFormatError(f"unexpected FIFO header 0x{h:02x} for {packet_size}-byte packets")
            break
        p = buf[pos:pos + packet_size]
        if hires:
            ext = p[17:20]
            acc = []
            gyr = []
            for ax in range(3):
                a = (((p[1 + 2 * ax] << 8) | p[2 + 2 * ax]) << 4) | (ext[ax] >> 4)
                g = (((p[7 + 2 * ax] << 8) | p[8 + 2 * ax]) << 4) | (ext[ax] & 0x0F)
                acc.append(a - (1 << 20) if a & (1 << 19) else a)
                gyr.append(g - (1 << 20) if g & (1 << 19) else g)
            temp = _s16(p, 13)
            tmst = (p[15] << 8) | p[16]
            invalid = INVALID_20
        else:
            acc = [_s16(p, 1), _s16(p, 3), _s16(p, 5)]
            gyr = [_s16(p, 7), _s16(p, 9), _s16(p, 11)]
            temp = p[13] - 256 if p[13] & 0x80 else p[13]
            tmst = (p[14] << 8) | p[15]
            invalid = INVALID_16
        valid = invalid not in acc and invalid not in gyr
        if valid or keep_invalid:
            out.append(FifoPacket(tuple(acc), tuple(gyr), temp, tmst, h, hires, valid))
        pos += packet_size
    return out, pos


def scale(p: FifoPacket, accel_fs_g: int, gyro_fs_dps: float) -> tuple[np.ndarray, np.ndarray, float]:
    """One packet -> (accel m/s^2 (3,), gyro rad/s (3,), temperature degC)."""
    a, g, t = scale_arrays([p], accel_fs_g, gyro_fs_dps)
    return a[0], g[0], float(t[0])


def scale_arrays(packets: list[FifoPacket], accel_fs_g: int, gyro_fs_dps: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised ``scale``: (accel (N,3) m/s^2, gyro (N,3) rad/s, temp (N,) degC)."""
    if not packets:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    acc = np.array([p.accel_raw for p in packets], dtype=np.float64)
    gyr = np.array([p.gyro_raw for p in packets], dtype=np.float64)
    tmp = np.array([p.temp_raw for p in packets], dtype=np.float64)
    if packets[0].hires:
        a_lsb, g_lsb, t_lsb = HIRES_ACCEL_LSB_PER_G, HIRES_GYRO_LSB_PER_DPS, TEMP_16_LSB_PER_C
    else:
        a_lsb, g_lsb, t_lsb = accel_lsb_per_g(accel_fs_g), gyro_lsb_per_dps(gyro_fs_dps), TEMP_FIFO8_LSB_PER_C
    return acc / a_lsb * G, gyr / g_lsb * DEG2RAD, tmp / t_lsb + 25.0


class TimestampUnwrapper:
    """16-bit chip timestamps (ticks of ``tick_us``) -> monotone int64 microseconds, across calls.

    Consecutive samples are assumed less than one wrap (65.536 ms at 1 us ticks) apart, which holds as long as no
    samples are lost. After a FIFO overflow call with ``hint_us`` (expected chip time of the first sample, e.g. from
    the host clock) to pick the wrap count nearest to it.
    """

    WRAP = 1 << 16

    def __init__(self, tick_us: int = 1):
        self.tick_us = int(tick_us)
        self.last_raw: int | None = None
        self.last_ticks: int | None = None   # unwrapped ticks of the last sample

    def reset(self) -> None:
        self.last_raw = None
        self.last_ticks = None

    def __call__(self, tmst16, hint_us: int | None = None) -> np.ndarray:
        raw = np.asarray(tmst16, dtype=np.int64).reshape(-1) & 0xFFFF
        if len(raw) == 0:
            return np.zeros(0, np.int64)
        out = np.empty(len(raw), np.int64)
        if hint_us is not None or self.last_ticks is None:
            hint_ticks = 0 if hint_us is None else int(hint_us) // self.tick_us
            base = hint_ticks - (hint_ticks % self.WRAP)
            cand = base + int(raw[0])
            # nearest of cand - WRAP, cand, cand + WRAP to the hint
            first = min((cand - self.WRAP, cand, cand + self.WRAP), key=lambda c: abs(c - hint_ticks))
            if hint_us is None:
                first = int(raw[0])
        else:
            first = self.last_ticks + ((int(raw[0]) - self.last_raw) % self.WRAP)
        out[0] = first
        if len(raw) > 1:
            d = np.diff(raw) % self.WRAP
            out[1:] = first + np.cumsum(d)
        self.last_raw = int(raw[-1])
        self.last_ticks = int(out[-1])
        return out * self.tick_us
