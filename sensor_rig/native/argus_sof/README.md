# argus_sof: sensor start-of-frame timestamps via libargus

**Status: written, never built or run** (no Jetson in the development session). The Python side (protocol parser,
shared-memory ring reader, the `argus_sof` camera backend, clock conversion) is unit-tested with a fake helper
process (`tests/fake_argus_sof.py`).

Why: the `argus` backend (GStreamer `nvarguscamerasrc` → appsink) stamps buffers when they *arrive*, roughly 30 ms
after mid-exposure, with load-dependent jitter (gaps G2 / F8 in the plan). This helper reads
`ICaptureMetadata::getSensorTimestamp()` for each frame, i.e. the sensor's own start-of-frame time.

```bash
sensor_rig/native/argus_sof/build.sh        # on the Jetson; needs nvidia-l4t-jetson-multimedia-api
```

Then, in rig.yaml:

```yaml
capture:
  backends: {main: argus_sof, ultrawide: argus_sof, imu: icm42688_spi}
  argus_sof: {helper: /home/<you>/Capstone/sensor_rig/native/argus_sof/build/argus_sof}   # optional
```

`sensor-rig doctor` checks that the helper exists and is executable. Afterwards, run `sensor-rig stamp-offset`
again (≥ 60 s).

What the stamp means: Argus documents `getSensorTimestamp()` as the time the first data of the frame arrives from
the sensor, i.e. the start of readout. It is not mid-exposure and not the start of exposure. The rig convention is
`t_capture = mid-exposure = t_stamp + offset_prior_ns`, so `offset_prior_ns` is **not 0**. Expect roughly
−(exposure/2) for the first row plus up to +readout/2 (~11 ms on IMX219 mode 3) for the image centre with a rolling
shutter, so a few ms, either sign. **Don't assume a value: the stamp-offset LED test measures the real offset**
(with the LED near the image centre) and prints the patch. The jitter should drop well below the ~1 ms of appsink
stamps.

Protocol and shared-memory layout: see the header of `argus_sof.cpp`. Things to verify on hardware:
1. The timestamp domain. L4T documents the sensor timestamp as TSC/MONOTONIC_RAW based; sensor_rig maps it with
   `clock.RawToMono`, refitted every 2 s. The stamp-offset run confirms the mapping (a stable offset with no drift).
2. That `copyToNvBuffer` + CPU copy keeps up with 2 × 1640x1232 @ 30 fps (≈ 3 MB per frame).
3. Two helpers (one per sensor) at the same time: `nvargus-daemon` serves both.

Samples it follows (from `/usr/src/jetson_multimedia_api` on JetPack 6): `argus/samples/userAutoExposure`
(per-frame `ICaptureMetadata`), `argus/samples/cudaHistogram` and `samples/09_argus_camera_jpeg` (EGLStream
`FrameConsumer`, `IImageNativeBuffer` → NvBufSurface), and `argus/samples/multiSensor`.
