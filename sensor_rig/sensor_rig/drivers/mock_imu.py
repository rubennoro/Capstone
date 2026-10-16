"""MockImu: a static IMU (gravity + white noise) or one synthesised from a trajectory (P1 §2.2).

Samples lie on the grid ``t0 + round(k * 1e9 / rate_hz)`` (``t0`` = clock time at ``start()``); a batch with every
sample up to "now" is emitted every ``batch_period_s`` (default 5 ms) of clock time, like a FIFO drained by a poller.
"""

from __future__ import annotations

import numpy as np

from sensor_rig.drivers.base import ImuBackend, ImuBatch

G = 9.80665
# Datasheet-class white-noise densities (ICM-42688-P: 2.8 mdps/sqrt(Hz), 70 ug/sqrt(Hz))
DEFAULT_GYRO_ND = 4.9e-5     # rad/s/sqrt(Hz)
DEFAULT_ACCEL_ND = 6.9e-4    # m/s^2/sqrt(Hz)


class MockImu(ImuBackend):
    def __init__(self, rate_hz: float, sample_fn, clock=None, batch_period_s: float = 0.005,
                 n_samples: int | None = None, description: dict | None = None):
        """``sample_fn(k_array, t_ns_array) -> (gyro (N,3), accel (N,3))``."""
        super().__init__(rate_hz, clock=clock)
        self._sample_fn = sample_fn
        self.batch_period_ns = int(batch_period_s * 1e9)
        self.n_samples = n_samples
        self._description = description or {}
        self.t_start_ns: int | None = None
        self._src_t: np.ndarray | None = None   # explicit source stamps (from_trajectory)
        self._preserve = False

    def _run(self) -> None:
        clock = self.clock
        t0 = clock.now_ns()
        self.t_start_ns = t0
        period = 1e9 / self.rate_hz
        k = 0
        next_emit = t0 + self.batch_period_ns
        while not self._stopping.is_set():
            clock.sleep_until(next_emit)
            now = clock.now_ns()
            k_end = int(np.floor((now - t0) / period)) + 1      # samples with t <= now
            if self.n_samples is not None:
                k_end = min(k_end, self.n_samples)
            if k_end > k:
                ks = np.arange(k, k_end)
                if self._src_t is None:
                    t = t0 + np.round(ks * period).astype(np.int64)
                elif self._preserve:
                    t = self._src_t[ks]
                else:
                    t = t0 + (self._src_t[ks] - self._src_t[0])
                gyro, accel = self._sample_fn(ks, t)
                self._push(ImuBatch(t, np.asarray(gyro, float), np.asarray(accel, float), {}))
                k = k_end
            if self.n_samples is not None and k >= self.n_samples:
                return
            next_emit += self.batch_period_ns
            if next_emit < now:   # fell behind (slow host): catch up without bursting forever
                next_emit = now + self.batch_period_ns

    def info(self) -> dict:
        d = {"backend": "mock", "odr_hz": self.rate_hz}
        d.update(self._description)
        return d

    @classmethod
    def static(cls, rate_hz: float = 1000.0, R_world_imu: np.ndarray | None = None,
               gyro_noise_density: float = DEFAULT_GYRO_ND, accel_noise_density: float = DEFAULT_ACCEL_ND,
               gyro_bias=(0.0, 0.0, 0.0), accel_bias=(0.0, 0.0, 0.0), seed: int = 0, clock=None,
               **kw) -> "MockImu":
        """At rest: accel = R_world_imu^T (0, 0, +g) (specific force, z-up world) + noise; gyro = bias + noise.

        Noise std per sample = density * sqrt(rate_hz) (white noise sampled at the ODR).
        """
        R = np.eye(3) if R_world_imu is None else np.asarray(R_world_imu, float)
        f_imu = R.T @ np.array([0.0, 0.0, G])
        rng = np.random.default_rng(seed)
        sg = gyro_noise_density * np.sqrt(rate_hz)
        sa = accel_noise_density * np.sqrt(rate_hz)
        bg = np.asarray(gyro_bias, float)
        ba = np.asarray(accel_bias, float)

        def sample_fn(ks, t):
            n = len(ks)
            gyro = bg + rng.normal(0.0, sg, (n, 3))
            accel = f_imu + ba + rng.normal(0.0, sa, (n, 3))
            return gyro, accel

        return cls(rate_hz, sample_fn, clock=clock, description={"mode": "static"}, **kw)

    @classmethod
    def from_trajectory(cls, times_ns: np.ndarray, T_W_S: np.ndarray, rate_hz: float = 1000.0,
                        noise: dict | None = None, seed: int = 0, clock=None, preserve_timestamps: bool = False,
                        **kw) -> "MockImu":
        """IMU synthesised from a trajectory (pose of the IMU frame S in a z-up world) with
        ``sensor_rig.sim.imu_synth.synth_imu`` (imported lazily; owned by WP-C).

        The synthesised samples are replayed at the clock; stamps are re-based to ``start`` unless
        ``preserve_timestamps``.
        """
        from sensor_rig.sim.imu_synth import synth_imu

        t_src, gyro_src, accel_src = synth_imu(np.asarray(times_ns, np.int64), np.asarray(T_W_S, float), rate_hz,
                                               noise, seed)
        t_src = np.asarray(t_src, np.int64)
        mock = cls(rate_hz, lambda ks, t: (gyro_src[ks], accel_src[ks]), clock=clock, n_samples=len(t_src),
                   description={"mode": "trajectory"}, **kw)
        mock._src_t = t_src
        mock._preserve = bool(preserve_timestamps)
        return mock
