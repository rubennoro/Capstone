// SPDX-License-Identifier: MIT
/// @file rig/drivers/camera/capture_plan.hpp
/// @brief Pure translation of a CameraConfig + chosen SensorMode into the exact sensor/ISP controls (Argus
/// ISourceSettings / IAutoControlSettings / IDenoiseSettings / IEdgeEnhanceSettings), and the equivalent
/// nvarguscamerasrc / override GStreamer launch strings. Unit-tested on the laptop; applied by the argus/ and gst/
/// backends. Owned by T2.
///
/// Locking rules (PLAN T2, PIPELINE 5):
/// - frame duration fixed: setFrameDurationRange([d, d]), d = round(1e9 / fps), clamped into the mode's range;
/// - exposure_ns > 0: exposure range [e, e]; analog_gain > 0: gain range [g, g] (the AE then has nothing to move);
///   0 leaves the mode's full range to the AE (bring-up only);
/// - ISP digital gain range [isp_digital_gain, isp_digital_gain] (1-1 by default);
/// - AE lock / AWB lock: applied from the first frame if nothing is left for that loop to converge (AE: both
///   exposure and gain fixed; AWB: mode "off"). Otherwise the lock is DEFERRED: the stream runs unlocked for
///   kSettleNs so AE/AWB converge, then a new repeating request with the lock is submitted (start() returns after).
/// - denoise / edge enhancement OFF unless configured (DENOISE_MODE_FAST / EDGE_ENHANCE_MODE_FAST when on);
///   AE antibanding OFF (it would otherwise quantize exposure to the mains period).
/// Threading: pure functions.
#pragma once

#include <cstdint>
#include <string>

#include "rig/drivers/camera_backend.hpp"

namespace rig {

/// How long start() lets AE/AWB (and the VCM after a focus write) settle before locking and delivering [ns].
inline constexpr int64_t kSettleNs = 1'000'000'000;
/// Frames discarded at start even when nothing needs to settle (timestamp-domain classification uses them).
inline constexpr int kMinSettleFrames = 5;

/// Argus AwbMode names accepted in `awb_mode` (Argus/Types.h AWB_MODE_*), with the nvarguscamerasrc wbmode
/// integer (gst-inspect: 0 off, 1 auto, 2 incandescent, 3 fluorescent, 4 warm-fluorescent, 5 daylight,
/// 6 cloudy-daylight, 7 twilight, 8 shade, 9 manual). Returns -1 for an unknown name.
int awb_mode_index(const std::string& name);

struct CapturePlan {
  SensorMode mode;                 ///< The mode in use.
  int64_t frame_duration_ns = 0;   ///< [d, d].
  int64_t exposure_min_ns = 0;     ///< Exposure range given to the sensor [ns].
  int64_t exposure_max_ns = 0;
  double gain_min = 0.0;           ///< Analog gain range [x].
  double gain_max = 0.0;
  double isp_gain_min = 1.0;       ///< ISP digital gain range [x].
  double isp_gain_max = 1.0;
  bool exposure_locked = false;    ///< exposure_min == exposure_max (from config).
  bool gain_locked = false;
  std::string awb_mode = "auto";   ///< Validated awb_mode.
  int wbmode = 1;                  ///< nvarguscamerasrc wbmode integer.
  bool ae_lock_requested = false;
  bool awb_lock_requested = false;
  bool ae_lock_deferred = false;   ///< AE lock applied after kSettleNs (AE had something to converge).
  bool awb_lock_deferred = false;  ///< AWB lock applied after kSettleNs.
  bool denoise = false;
  bool edge_enhance = false;
  int fifo_length = 4;             ///< EGLStream FIFO length (buffer_count).

  bool needs_settle() const noexcept { return ae_lock_deferred || awb_lock_deferred; }
};

/// Validates config against `mode` and builds the plan. kInvalidArgument (with "cameras.<id>.driver.<key>: ..."
/// context) if the exposure exceeds the frame duration or the mode's exposure range, the gain is outside the mode's
/// range, isp_digital_gain < 1, fps is outside the mode's range, awb_mode is unknown, or buffer_count < 1.
Result<CapturePlan> build_capture_plan(const CameraConfig& config, const SensorMode& mode);

/// GStreamer caps format name for a PixelFormat ("NV12", "GRAY8", "BGR", "RGB", "YUY2"); empty if unsupported
/// (JPEG, unknown).
std::string gst_format_name(PixelFormat f);

/// "N/D" framerate string (60 -> "60/1", 29.97 -> "30000/1001").
std::string gst_framerate(double fps);

/// Full launch string ending in `appsink name=<sink_name>`:
/// - source_override empty: `nvarguscamerasrc sensor-id=.. sensor-mode=.. exposuretimerange="e e" gainrange="g g"
///   ispdigitalgainrange="d d" aelock=.. awblock=.. wbmode=.. tnr-mode=0|1 ee-mode=0|1 aeantibanding=0 !
///   video/x-raw(memory:NVMM),width=W,height=H,framerate=F,format=NV12 ! nvvidconv ! video/x-raw,format=NV12
///   [! videoconvert ! video/x-raw,format=X] ! appsink ...` (sensor-mode omitted if plan.mode.index < 0; deferred
///   locks start as false and are set at runtime);
/// - otherwise `<override> ! videoconvert ! videoscale ! video/x-raw,format=X,width=W,height=H,framerate=F ! appsink`.
/// The appsink is `sync=false max-buffers=<fifo_length> drop=false emit-signals=false`.
std::string build_gst_pipeline(const CameraConfig& config, const CapturePlan& plan,
                               const std::string& sink_name = "rig_sink");

}  // namespace rig
