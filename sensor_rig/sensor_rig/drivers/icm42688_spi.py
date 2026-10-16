"""Icm42688Spi: ICM-42688-P over userspace spidev, FIFO burst reads, chip timestamps mapped to CLOCK_MONOTONIC.

Jetson wiring (R4 §2): 40-pin header SPI1 (pins 19 MOSI, 21 MISO, 23 SCK, 24 CS0) enabled with jetson-io ->
``/dev/spidev0.0``. ``spidev`` is imported lazily in ``_open`` (``pip install spidev``).

Reader thread, every ``poll_ms`` (default 5 ms):
1. read FIFO_COUNT (2 bytes, big-endian, bytes) -> ``host_ns = now`` right after it,
2. burst-read ``floor(count / packet) * packet`` bytes from FIFO_DATA (chunks <= 4096 B, spidev's default bufsiz),
3. ``ImuFifoDecoder.process(bytes, host_ns)``: parse -> unwrap 16-bit timestamps -> feed ``ChipClockMapper`` with
   (chip time of the newest packet, host_ns) -> map every sample to CLOCK_MONOTONIC -> ``ImuBatch``.
``fifo_overflows`` counts reads with count >= FIFO size - one packet (samples were lost; the unwrapper is then
re-synchronised from the host clock). ``invalid_samples`` counts packets with the -32768 / -524288 marker.

The decoder is pure and unit-tested with fake FIFO bytes; only ``Icm42688Spi._open/_run`` touch hardware.
NOT RUN ON HARDWARE YET. See ``sensor_rig/record/HARDWARE_CHECKLIST.md``.
"""

from __future__ import annotations

import re
import time

import numpy as np

from sensor_rig.drivers import icm42688_proto as P
from sensor_rig.drivers.base import BackendError, ImuBackend, ImuBatch
from sensor_rig.drivers.chip_clock import ChipClockMapper

SPI_CHUNK = 4096


class ImuFifoDecoder:
    """Pure FIFO-bytes -> ImuBatch pipeline (parse, unwrap, clock map), stateful across reads."""

    def __init__(self, packet_size: int = P.PACKET3_SIZE, accel_fs_g: int = 16, gyro_fs_dps: float = 2000,
                 tick_us: int = 1, mapper: ChipClockMapper | None = None):
        self.packet_size = P.packet_size(packet_size)
        self.accel_fs_g = accel_fs_g
        self.gyro_fs_dps = gyro_fs_dps
        self.unwrap = P.TimestampUnwrapper(tick_us)
        self.mapper = mapper or ChipClockMapper()
        self.last_t_ns: int | None = None
        self.batch_index = 0
        self.stats = {"invalid_samples": 0, "fifo_overflows": 0, "parse_errors": 0, "fsync_stamps": 0,
                      "monotonic_fixes": 0, "packets": 0}

    def process(self, buf: bytes, host_ns: int, overflow: bool = False) -> ImuBatch | None:
        """Decode one FIFO read. ``host_ns``: CLOCK_MONOTONIC right after the FIFO_COUNT read that sized ``buf``.

        Raises ``P.FifoFormatError`` if the data is out of sync (caller flushes the FIFO and calls ``resync()``).
        """
        try:
            packets, _ = P.parse_fifo(buf, self.packet_size, keep_invalid=True)
        except P.FifoFormatError:
            self.stats["parse_errors"] += 1
            self.unwrap.reset()
            raise
        if not packets:
            return None
        self.stats["packets"] += len(packets)
        self.stats["fsync_stamps"] += sum(p.tmst_is_fsync for p in packets)
        hint = None
        if overflow:
            self.stats["fifo_overflows"] += 1
            if self.mapper.ready:
                # newest packet is ~now; first packet is (n-1) ODR periods earlier: let the unwrapper pick the wrap
                # count consistent with the host clock for the newest one by unwrapping relative to it.
                hint = self.mapper.chip_from_host(host_ns)
            else:
                self.unwrap.reset()
        if hint is not None:
            raw = np.array([p.tmst for p in packets], np.int64)
            rel = np.concatenate([[0], np.cumsum(np.diff(raw) % P.TimestampUnwrapper.WRAP)]) * self.unwrap.tick_us
            self.unwrap.reset()
            chip_first_hint = hint - int(rel[-1])
            chip_us = self.unwrap(raw, hint_us=chip_first_hint)
        else:
            chip_us = self.unwrap([p.tmst for p in packets])
        self.mapper.add(int(chip_us[-1]), int(host_ns))
        valid = np.array([p.valid for p in packets], bool)
        self.stats["invalid_samples"] += int((~valid).sum())
        t_ns = self.mapper.map(chip_us)
        # enforce strictly increasing output across refits
        if self.last_t_ns is not None and t_ns[0] <= self.last_t_ns:
            self.stats["monotonic_fixes"] += 1
        floor = (self.last_t_ns + 1) if self.last_t_ns is not None else t_ns[0]
        t_ns = np.maximum.accumulate(np.maximum(t_ns, floor))
        dup = np.nonzero(np.diff(t_ns) <= 0)[0]
        for i in dup:   # rare: equal stamps after clamping
            t_ns[i + 1] = t_ns[i] + 1
        self.last_t_ns = int(t_ns[-1])
        good = [p for p, v in zip(packets, valid) if v]
        if not good:
            return None
        accel, gyro, temp = P.scale_arrays(good, self.accel_fs_g, self.gyro_fs_dps)
        n = len(good)
        aux = {"chip_us": chip_us[valid], "temp_c": temp, "batch": np.full(n, self.batch_index, np.int64),
               "host_read_ns": np.full(n, int(host_ns), np.int64)}
        self.batch_index += 1
        return ImuBatch(t_ns[valid].astype(np.int64), gyro, accel, aux)


def parse_spi_device(dev: str) -> tuple[int, int]:
    """'/dev/spidev0.0' -> (0, 0)."""
    m = re.search(r"spidev(\d+)\.(\d+)$", str(dev))
    if not m:
        raise ValueError(f"cannot parse SPI device {dev!r} (expected /dev/spidevB.D)")
    return int(m.group(1)), int(m.group(2))


class Icm42688Spi(ImuBackend):
    def __init__(self, spi_device: str = "/dev/spidev0.0", spi_speed_hz: int = 10_000_000, spi_mode: int = 0,
                 odr_hz: int = 1000, accel_fs_g: int = 16, gyro_fs_dps: float = 2000,
                 packet: str | int = "16byte", accel_bw: str = "odr_div_4", gyro_bw: str = "odr_div_4",
                 poll_ms: float = 5.0, tick_us: int = 1, clock=None, mapper_max_ppm: float = 20000.0):
        super().__init__(odr_hz, clock=clock)
        self.spi_device = spi_device
        self.spi_speed_hz = int(spi_speed_hz)
        self.spi_mode = int(spi_mode)
        self.odr_hz = int(odr_hz)
        self.accel_fs_g = int(accel_fs_g)
        self.gyro_fs_dps = gyro_fs_dps
        self.packet_size = P.packet_size(packet)
        self.hires = self.packet_size == P.PACKET4_SIZE
        if self.hires and (self.accel_fs_g != 16 or float(gyro_fs_dps) != 2000.0):
            import warnings

            warnings.warn("20-byte hi-res FIFO packets force +-16 g / +-2000 dps (DS-000347 §6.1); configured ranges "
                          "only affect the data registers", stacklevel=2)
        self.accel_bw = accel_bw
        self.gyro_bw = gyro_bw
        self.poll_s = float(poll_ms) / 1000.0
        self.tick_us = int(tick_us)
        if self.tick_us not in (1, 16):
            raise ValueError("timestamp resolution must be 1 or 16 us")
        self.decoder = ImuFifoDecoder(self.packet_size, self.accel_fs_g, gyro_fs_dps, tick_us=self.tick_us,
                                      mapper=ChipClockMapper(max_ppm=mapper_max_ppm))
        self._spi = None

    @classmethod
    def from_rig(cls, rig, **opts) -> "Icm42688Spi":
        imu = rig.imu
        fifo = dict(getattr(imu, "fifo", None) or {})
        kw = dict(spi_device=imu.spi_device, spi_speed_hz=imu.spi_speed_hz, spi_mode=imu.spi_mode,
                  odr_hz=int(imu.odr_hz), accel_fs_g=int(imu.accel_range_g), gyro_fs_dps=imu.gyro_range_dps,
                  packet=fifo.get("packet", "16byte"), accel_bw=getattr(imu, "accel_ui_filter_bw", "odr_div_4"),
                  gyro_bw=getattr(imu, "gyro_ui_filter_bw", "odr_div_4"),
                  poll_ms=float(fifo.get("poll_period_ms", 5)), tick_us=int(fifo.get("timestamp_resolution_us", 1)))
        kw.update(opts)
        return cls(**kw)

    # ---- SPI primitives (hardware only) ----
    def _read(self, reg: int, n: int) -> bytes:
        return bytes(self._spi.xfer2(P.spi_read_cmd(reg, n))[1:])

    def _write(self, reg: int, val: int) -> None:
        self._spi.xfer2(P.spi_write_cmd(reg, val))
        d = P.POST_WRITE_DELAYS_S.get(reg)
        if d:
            time.sleep(d)

    def _read_fifo(self, nbytes: int) -> bytes:
        out = bytearray()
        while nbytes > 0:
            k = min(nbytes, SPI_CHUNK - 1)
            k -= k % self.packet_size
            if k <= 0:
                break
            out += self._read(P.REG_FIFO_DATA, k)
            nbytes -= k
        return bytes(out)

    def _open(self) -> None:  # pragma: no cover - needs hardware
        try:
            import spidev
        except ImportError as e:
            raise BackendError("the icm42688_spi backend needs the 'spidev' package (pip install spidev)") from e
        bus, dev = parse_spi_device(self.spi_device)
        spi = spidev.SpiDev()
        spi.open(bus, dev)
        spi.max_speed_hz = self.spi_speed_hz
        spi.mode = self.spi_mode
        self._spi = spi
        who = self._read(P.REG_WHO_AM_I, 1)[0]
        if who != P.WHO_AM_I_VALUE:
            spi.close()
            raise BackendError(f"ICM-42688-P not found on {self.spi_device}: WHO_AM_I=0x{who:02x}, expected 0x47")
        for reg, val in P.config_sequence(self.odr_hz, self.accel_fs_g, self.gyro_fs_dps, hires=self.hires,
                                          accel_bw=self.accel_bw, gyro_bw=self.gyro_bw,
                                          tmst_res_16us=self.tick_us == 16):
            self._write(reg, val)

    def _close(self) -> None:  # pragma: no cover - needs hardware
        if self._spi is not None:
            try:
                self._write(P.REG_PWR_MGMT0, 0x00)
            finally:
                self._spi.close()
                self._spi = None

    def _run(self) -> None:  # pragma: no cover - needs hardware
        clock = self.clock
        overflow_bytes = P.FIFO_SIZE_BYTES - self.packet_size
        next_poll = time.monotonic()
        while not self._stopping.is_set():
            next_poll += self.poll_s
            dt = next_poll - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                next_poll = time.monotonic()
            cnt_b = self._read(P.REG_FIFO_COUNTH, 2)
            host_ns = clock.now_ns()
            count = (cnt_b[0] << 8) | cnt_b[1]
            n = (count // self.packet_size) * self.packet_size
            if n == 0:
                continue
            overflow = count >= overflow_bytes
            buf = self._read_fifo(n)
            try:
                batch = self.decoder.process(buf, host_ns, overflow=overflow)
            except P.FifoFormatError:
                self._write(P.REG_SIGNAL_PATH_RESET, P.SIGNAL_PATH_RESET_FIFO_FLUSH)
                continue
            self.stats.update({k: self.decoder.stats[k] for k in ("fifo_overflows", "invalid_samples")})
            if batch is not None:
                self._push(batch)

    def info(self) -> dict:
        return {"backend": "icm42688_spi", "odr_hz": self.odr_hz, "accel_fs_g": 16 if self.hires else self.accel_fs_g,
                "gyro_fs_dps": 2000 if self.hires else self.gyro_fs_dps, "fifo_packet_bytes": self.packet_size,
                "spi_device": self.spi_device, "clock_fit": self.decoder.mapper.summary(),
                "fifo_overflows": self.decoder.stats["fifo_overflows"],
                "invalid_samples": self.decoder.stats["invalid_samples"]}
