// SPDX-License-Identifier: MIT
/// @file rig/types.hpp
/// @brief Core value types shared by every module: time, images, IMU samples, summary statistics.
/// FROZEN CONTRACT (T0).
///
/// Conventions used throughout `rig::` (PIPELINE.md section 5, "One clock"):
/// - **Time:** `TimeNs` = signed 64-bit nanoseconds of `CLOCK_MONOTONIC_RAW` unless a name says otherwise
///   (`*_unix_ns` = CLOCK_REALTIME, `*_mono_ns` = CLOCK_MONOTONIC). `kInvalidTime` marks "unknown".
/// - **Units:** SI unless the name carries a unit suffix (`_ns`, `_ms`, `_s`, `_hz`, `_g`, `_dps`, `_c`, `_m`).
/// - **Frames:** camera frames are OpenCV optical (x right, y down, z forward). The IMU body frame is the LSM6DSOX
///   sensor frame as the chip reports it (no axis remap); `T_imu_cam` in the config places cameras in it.
/// - **Unknown values:** integers use -1, floating point uses NaN, times use kInvalidTime.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include <Eigen/Core>

#include "rig/span.hpp"
#include "rig/status.hpp"

namespace rig {

/// Nanoseconds of CLOCK_MONOTONIC_RAW (see rig/time/clock.hpp).
using TimeNs = int64_t;

/// "No timestamp". Compare with `==`; never do arithmetic on it.
inline constexpr TimeNs kInvalidTime = std::numeric_limits<int64_t>::min();

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// Standard gravity used for every g <-> m/s^2 conversion (BIPM; same constant as ST's drivers).
inline constexpr double kStandardGravity = 9.80665;

// ---------------------------------------------------------------------------------------------------------------
// Provenance of calibration-like values (config, packages, manifests).
// ---------------------------------------------------------------------------------------------------------------

/// Where a calibration value came from. YAML strings in parentheses.
enum class Provenance : uint8_t {
  kNominal = 0,   ///< ("nominal") lens/sensor spec sheet; a guess until calibrated (PIPELINE D7).
  kCad = 1,       ///< ("cad") from the CAD of the mount / finger kit (PIPELINE D6).
  kKalibr = 2,    ///< ("kalibr") estimated by Kalibr.
  kMeasured = 3,  ///< ("measured") measured on hardware by one of our tools (Allan variance, LED test, arm test).
  kManual = 4,    ///< ("manual") hand-entered by an operator.
};

constexpr const char* to_string(Provenance p) noexcept {
  switch (p) {
    case Provenance::kNominal: return "nominal";
    case Provenance::kCad: return "cad";
    case Provenance::kKalibr: return "kalibr";
    case Provenance::kMeasured: return "measured";
    case Provenance::kManual: return "manual";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------------------------------------------

/// Pixel layout of `ImageFrame::data`. YAML strings in parentheses.
enum class PixelFormat : uint8_t {
  kUnknown = 0,
  kNV12 = 1,   ///< ("nv12") 2 planes: Y (W x H, 1 B/px), interleaved UV (W/2 x H/2 samples, 2 B each). BT.601 range
               ///< as delivered by the ISP; the Argus/GStreamer backends' native output and the JPEG encoder input.
  kGRAY8 = 2,  ///< ("gray8") 1 plane, 1 B/px.
  kBGR8 = 3,   ///< ("bgr8") 1 plane, 3 B/px, OpenCV channel order.
  kRGB8 = 4,   ///< ("rgb8") 1 plane, 3 B/px.
  kYUYV = 5,   ///< ("yuyv") 1 plane, 2 B/px packed 4:2:2.
  kJPEG = 6,   ///< ("jpeg") compressed bytes in plane 0; width/height describe the decoded image; stride = 0.
};

constexpr const char* to_string(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::kUnknown: return "unknown";
    case PixelFormat::kNV12: return "nv12";
    case PixelFormat::kGRAY8: return "gray8";
    case PixelFormat::kBGR8: return "bgr8";
    case PixelFormat::kRGB8: return "rgb8";
    case PixelFormat::kYUYV: return "yuyv";
    case PixelFormat::kJPEG: return "jpeg";
  }
  return "unknown";
}

/// Number of planes a format uses (0 for kUnknown).
constexpr int num_planes(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::kNV12: return 2;
    case PixelFormat::kGRAY8:
    case PixelFormat::kBGR8:
    case PixelFormat::kRGB8:
    case PixelFormat::kYUYV:
    case PixelFormat::kJPEG: return 1;
    case PixelFormat::kUnknown: return 0;
  }
  return 0;
}

/// One plane inside an ImageFrame buffer.
struct Plane {
  std::size_t offset = 0;  ///< Byte offset of the plane's first row from `ImageFrame::data.get()`.
  std::size_t stride = 0;  ///< Bytes between the starts of consecutive rows (>= width * bytes per sample).
  int width = 0;           ///< Samples per row in this plane (NV12 UV plane: W/2 interleaved UV pairs).
  int height = 0;          ///< Rows in this plane (NV12 UV plane: H/2).
};

/// Maximum planes per frame (NV12 = 2; one spare for future I420).
inline constexpr int kMaxPlanes = 3;

/// IMX519 lens focus state for a frame (PLAN T2 `Imx519Focus`). YAML/CSV strings in parentheses.
enum class FocusStatus : uint8_t {
  kUnknown = 0,          ///< ("unknown") not determined (e.g. mock/replay without metadata).
  kNotApplicable = 1,    ///< ("n/a") fixed-focus lens (IMX219 ultrawide).
  kLockedVcmI2c = 2,     ///< ("vcm_i2c") position written to the VCM at I2C 0x0C by userspace at start.
  kLockedV4l2 = 3,       ///< ("v4l2") position set through the V4L2 focus_absolute control.
  kFixedAtPowerUp = 4,   ///< ("power_up") VCM not reachable (`UU`, no V4L2 control); lens where it powered up.
  kFailed = 5,           ///< ("failed") a focus write was attempted and failed.
};

constexpr const char* to_string(FocusStatus s) noexcept {
  switch (s) {
    case FocusStatus::kUnknown: return "unknown";
    case FocusStatus::kNotApplicable: return "n/a";
    case FocusStatus::kLockedVcmI2c: return "vcm_i2c";
    case FocusStatus::kLockedV4l2: return "v4l2";
    case FocusStatus::kFixedAtPowerUp: return "power_up";
    case FocusStatus::kFailed: return "failed";
  }
  return "unknown";
}

/// Quantization range of the frame's YCbCr samples (CONTRACT CHANGE by T2, R1 F16; additive). Decides how a consumer
/// converts NV12/YUYV/GRAY8 to RGB or JPEG: kFull = JFIF / "extended range" (Y, Cb, Cr all 0..255; what a baseline
/// JPEG stores, so a JPEG encoded straight from these samples decodes correctly), kLimited = BT.601 video range
/// (Y 16..235, CbCr 16..240). RGB/BGR frames are always full range. YAML/CSV strings in parentheses.
enum class ColorRange : uint8_t {
  kUnknown = 0,  ///< ("unknown") not stated by the source.
  kLimited = 1,  ///< ("limited") BT.601 video range.
  kFull = 2,     ///< ("full") JFIF full range (Argus NVBUF_COLOR_FORMAT_NV12_ER; mock/replay).
};

constexpr const char* to_string(ColorRange r) noexcept {
  switch (r) {
    case ColorRange::kUnknown: return "unknown";
    case ColorRange::kLimited: return "limited";
    case ColorRange::kFull: return "full";
  }
  return "unknown";
}

/// Per-frame sensor metadata (Argus ICaptureMetadata where available; mock/replay fill what they know).
struct FrameMeta {
  int64_t exposure_ns = -1;      ///< Sensor exposure time [ns] (Argus getSensorExposureTime); -1 = unknown.
  double analog_gain = kNaN;     ///< Sensor analog gain [x] (Argus getSensorAnalogGain).
  double digital_gain = kNaN;    ///< ISP digital gain [x] (Argus getIspDigitalGain).
  int64_t readout_ns = -1;       ///< Rolling-shutter readout time first->last row [ns] (Argus getFrameReadoutTime).
  int sensor_mode = -1;          ///< Index into CameraBackend::modes() in use.
  int focus_position = -1;       ///< Lens position written (VCM DAC code / V4L2 focus_absolute value); -1 = n/a.
  FocusStatus focus_status = FocusStatus::kUnknown;
  ColorRange color_range = ColorRange::kUnknown;  ///< YCbCr range of the pixel data (T2 addition, R1 F16).
};

/// One camera frame. Value type, cheap to copy: pixel storage is shared and immutable.
///
/// Timestamps (all CLOCK_MONOTONIC_RAW ns; PIPELINE 5):
/// - `t_sof_ns`: sensor start of frame (first row starts exposing) as reported by the sensor/driver
///   (Argus getSensorTimestamp; GStreamer: buffer PTS + base time converted, ~30 ms late, fallback only).
/// - `t_mid_ns`: image time used everywhere downstream = t_sof + exposure/2 + readout/2 (middle of the
///   middle row's exposure; assumption documented by T2 and verified on hardware day, BRINGUP step 13).
/// - `t_arrival_ns`: host time the frame became available to the backend (now_raw_ns() on dequeue).
///
/// Ownership: `data` is a shared, read-only buffer. Backends may recycle buffers through a custom deleter (buffer
/// pool); consumers must drop their copy promptly (bounded queues guarantee this) and must never write through it.
/// Threading: immutable after the backend publishes it; safe to share between threads.
struct ImageFrame {
  std::string camera_id;            ///< Config key of the camera ("main", "uw").
  uint64_t seq = 0;                 ///< Backend frame counter, starts at 0 on start(); gaps = drops.
  TimeNs t_sof_ns = kInvalidTime;   ///< See above.
  TimeNs t_mid_ns = kInvalidTime;   ///< See above.
  TimeNs t_arrival_ns = kInvalidTime;  ///< See above.
  int width = 0;                    ///< Image width [px].
  int height = 0;                   ///< Image height [px].
  PixelFormat fmt = PixelFormat::kUnknown;
  std::array<Plane, kMaxPlanes> planes{};  ///< First num_planes(fmt) entries are valid.
  std::shared_ptr<const uint8_t[]> data;   ///< Pixel bytes (all planes). nullptr = empty frame.
  std::size_t size = 0;             ///< Bytes valid in `data`.
  FrameMeta meta;

  /// True if the frame has pixel data, a known format and a valid t_mid.
  bool valid() const noexcept {
    return data != nullptr && size > 0 && fmt != PixelFormat::kUnknown && width > 0 && height > 0 &&
           t_mid_ns != kInvalidTime;
  }
  /// Pointer to the first byte of plane `i`. @pre 0 <= i < num_planes(fmt) and valid().
  const uint8_t* plane_data(int i) const noexcept { return data.get() + planes[static_cast<std::size_t>(i)].offset; }
};

// ---------------------------------------------------------------------------------------------------------------
// IMU
// ---------------------------------------------------------------------------------------------------------------

/// Bit flags in ImuSample::flags.
enum ImuSampleFlags : uint32_t {
  kImuFlagNone = 0,
  kImuFlagAccSaturated = 1u << 0,    ///< Any accel axis at the full-scale limit (raw +32767 / -32768).
  kImuFlagGyroSaturated = 1u << 1,   ///< Any gyro axis at the full-scale limit.
  kImuFlagAfterGap = 1u << 2,        ///< First sample after a detected gap / overrun (previous samples lost).
  kImuFlagTimeExtrapolated = 1u << 3,  ///< t_ns mapped before the clock fit was ready (nominal slope only).
  kImuFlagAccStale = 1u << 4,        ///< acc_mps2 is a repeat of the previous accel word (gyro-only FIFO slot).
  kImuFlagGyroStale = 1u << 5,       ///< gyro_rps is a repeat of the previous gyro word.
};

/// One synchronized accel + gyro sample in SI units.
///
/// Time: `t_ns` = CLOCK_MONOTONIC_RAW, mapped from the chip timestamp by ChipClockMapper (PIPELINE 5).
/// `chip_ticks` = the unwrapped (64-bit) LSM6DSOX timestamp counter value paired with this sample (one tick is
/// nominally 25 us, trimmed per chip; DS12814 / AN5272). Axes: IMU body frame (chip axes). Threading: value type.
/// Eigen::Vector3d has no alignment requirement, so ImuSample is safe in std::vector.
struct ImuSample {
  TimeNs t_ns = kInvalidTime;
  int64_t chip_ticks = -1;
  Eigen::Vector3d gyro_rps = Eigen::Vector3d::Zero();  ///< Angular rate [rad/s].
  Eigen::Vector3d acc_mps2 = Eigen::Vector3d::Zero();  ///< Specific force [m/s^2] (reads +g up at rest).
  float temp_c = std::numeric_limits<float>::quiet_NaN();  ///< Die temperature [deg C]; NaN if not sampled.
  uint32_t flags = kImuFlagNone;  ///< ImuSampleFlags bits.
};

// ---------------------------------------------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------------------------------------------

/// Summary of a scalar distribution over a window. The unit is the unit of the field it is stored in
/// (e.g. `inter_frame_ns` -> ns). Percentiles may be approximate (histogram/reservoir) but must be within 1 % of the
/// value range. NaN fields when count == 0.
struct SummaryStats {
  uint64_t count = 0;
  double min = kNaN;
  double max = kNaN;
  double mean = kNaN;
  double p50 = kNaN;
  double p99 = kNaN;
};

}  // namespace rig
