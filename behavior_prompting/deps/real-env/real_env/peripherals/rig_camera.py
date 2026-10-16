"""RigCamera: the Jetson sensor rig as a drop-in replacement for the iPhone wrist camera.

Thin adapter around ``sensor_rig.deploy.RigFrameSource`` (all capture, pairing, remap and timestamp logic lives in the
``sensor_rig`` package; see sensor_rig/docs/deployment.md). Output contract = the iPhone peripheral's:
``{"main", "ultrawide"}`` (H, W, 3) uint8 BGR frames rectified to the iPhone 16 Pro intrinsics at the configured size
(960x720), and ONE CLOCK_MONOTONIC timestamp = capture time of the main frame (rig.yaml offsets applied). Configs:
``configs/peripherals/rig_wrist_right.yaml`` (hardware) and ``rig_wrist_mock.yaml`` (no hardware).

Run: ``rig_camera wrist_right`` or ``rig_camera wrist_mock [hydra overrides]``. Needs ``pip install -e sensor_rig``.
"""

import json
import os
import sys
import time
from typing import Any, Literal

import cv2
import hydra
import numpy as np
from cv2.typing import MatLike
from omegaconf import DictConfig, ListConfig, OmegaConf

from real_env.peripherals.base_camera import BaseCamera


def _plain(x: Any) -> Any:
    if isinstance(x, (DictConfig, ListConfig)):
        return OmegaConf.to_container(x, resolve=True)
    return x


class RigCamera(BaseCamera):
    def __init__(
        self,
        rig_config_path: str | None = None,
        backend: str = "argus",
        backend_opts: dict | None = None,
        publish_mode: Literal["auto", "per_main_frame", "fixed_rate"] = "auto",
        pairing: dict | None = None,
        frame_timeout_s: float = 0.5,
        max_consecutive_timeouts: int = 10,
        min_coverage: float = 0.98,
        video_codec: str | None = "libx264",
        num_cv_threads: int | None = None,
        stats_period_s: float = 5.0,
        sync: str = "live",
        sources: dict | None = None,
        **kwargs,
    ):
        """
        rig_config_path: calibrated rig.yaml (absolute, relative to cwd, or a bare name in sensor_rig/configs/).
            ``null`` = ``<backend_opts.episode_dir>/rig.yaml`` when replaying a raw episode, else rig_default.yaml.
        backend: ``argus`` (CSI cameras) or ``mock`` (replay ``backend_opts.episode_dir`` or a synthetic pattern).
        publish_mode: ``per_main_frame`` (one packet per new main frame), ``fixed_rate`` (packets at
            ``camera_configs.main.fps``, newest frames; for a 30 fps physical main behind an fps-60 config), or
            ``auto`` (fixed_rate when the physical main fps is below the configured fps).
        sources: test hook, injected camera backends (bypasses ``backend``).
        Remaining kwargs go to BaseCamera (name, server_endpoint, logger_endpoint, camera_latency_s, camera_configs,
        display_resolution_WH, config_str).
        """
        super().__init__(**kwargs)
        from sensor_rig.deploy.frame_source import RigFrameSource
        from sensor_rig.rig_config import DEFAULT_RIG_PATH, load_rig

        backend_opts = dict(_plain(backend_opts) or {})
        pairing = _plain(pairing)

        if set(self.camera_configs) != {"main", "ultrawide"}:
            raise ValueError(f"RigCamera camera_configs must be exactly main + ultrawide, got {sorted(self.camera_configs)}")
        for cam, cc in self.camera_configs.items():
            if cc.get("type") != "rgb":
                raise ValueError(f"camera_configs.{cam}.type must be rgb, got {cc.get('type')!r}")

        if num_cv_threads is not None:
            cv2.setNumThreads(int(num_cv_threads))

        if self.camera_latency_s != 0:
            print(f"[{self.name}] WARNING: camera_latency_s={self.camera_latency_s} is non-zero. RigCamera timestamps are "
                  "already true capture times (rig.yaml cameras.*.time offsets), so this value is subtracted as an extra "
                  "measured residual only. Use 0.0 unless you measured a residual.")

        if rig_config_path is None:
            episode_dir = backend_opts.get("episode_dir")
            rig_config_path = os.path.join(episode_dir, "rig.yaml") if episode_dir else str(DEFAULT_RIG_PATH)
        self.rig = load_rig(rig_config_path)
        print(f"[{self.name}] rig: {self.rig.source_path} (fingerprint {self.rig.fingerprint()[:12]}, "
              f"placeholder={self.rig.placeholder})")

        output_sizes = {cam: (int(cc["width"]), int(cc["height"])) for cam, cc in self.camera_configs.items()}
        self.publish_fps = float(self.camera_configs["main"]["fps"])
        if publish_mode == "auto":
            phys_fps = float(self.rig.cameras[self.rig.virtual["main"].source].fps)
            publish_mode = "fixed_rate" if phys_fps < 0.9 * self.publish_fps else "per_main_frame"
        if publish_mode not in ("per_main_frame", "fixed_rate"):
            raise ValueError(f"publish_mode must be auto, per_main_frame or fixed_rate, got {publish_mode!r}")
        self.publish_mode = publish_mode

        # No NVENC on the Orin Nano: VideoLogger reads self.codec only when a recording starts.
        if video_codec:
            self.video_logger.codec = video_codec

        self.stats_period_s = float(stats_period_s)
        self._last_stats = time.monotonic()
        self._next_tick: float | None = None
        self.source = RigFrameSource(
            self.rig, output_sizes=output_sizes, backend=backend, backend_opts=backend_opts, pairing=pairing,
            frame_timeout_s=frame_timeout_s, max_consecutive_timeouts=max_consecutive_timeouts,
            min_coverage=min_coverage, sources=sources, sync=sync,
        )
        self.source.start()
        print(f"[{self.name}] RigCamera started: backend={backend} publish_mode={self.publish_mode} "
              f"fps={self.publish_fps:g} sizes={output_sizes}")

    def get_images_bgr_hwc_and_timestamp(self) -> tuple[dict[str, MatLike], float]:
        if self.publish_mode == "fixed_rate":
            period = 1.0 / self.publish_fps
            now = time.monotonic()
            if self._next_tick is None or now - self._next_tick > period:
                self._next_tick = now          # first call, or we fell behind by more than a period: re-phase
            else:
                time.sleep(max(0.0, self._next_tick - now))
            self._next_tick += period
            packet = self.source.latest_packet()
        else:
            packet = self.source.next_packet()
            if packet is None:
                raise RuntimeError(f"[{self.name}] replay source exhausted")
        now = time.monotonic()
        if self.stats_period_s > 0 and now - self._last_stats >= self.stats_period_s:
            self._last_stats = now
            print(f"\n[{self.name}] {self.source.stats_line()}")
        return dict(packet.images), packet.t_capture_s

    def reset(self):
        print(f"\n[{self.name}] reset: {json.dumps(self.source.stats(), default=str)}")
        self.source.reset_stats()

    def run(self):
        try:
            super().run()
        finally:
            self.source.stop()


def run_rig_camera():
    """Runs the rig camera server: ``rig_camera [name] [hydra overrides...]`` with config ``rig_<name>.yaml``."""
    np.set_printoptions(precision=4)
    os.environ["HYDRA_FULL_ERROR"] = "1"
    name = sys.argv[1] if len(sys.argv) > 1 else "wrist_right"
    with hydra.initialize(config_path="../configs/peripherals", version_base=None):
        cfg = hydra.compose(config_name=f"rig_{name}", overrides=sys.argv[2:])
    print(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    camera: RigCamera = hydra.utils.instantiate(cfg)
    camera.run()


if __name__ == "__main__":
    run_rig_camera()
