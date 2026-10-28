// SPDX-License-Identifier: MIT
/// @file rig/drivers/camera_backend.hpp
/// @brief Camera driver interface: Argus (primary), GStreamer (fallback), mock and replay backends, IMX519 focus.
/// FROZEN CONTRACT (T0); implemented by T2 in common/drivers/camera/.
///
/// Lifecycle: make_camera(config) -> start() -> read()... -> stop(). A backend may be restarted after stop().
/// Threading:
/// - start(), stop(), modes() are called from the owning thread (CaptureService).
/// - read() is called from exactly one consumer thread (the camera's capture thread).
/// - latest(), stats(), status(), config() are thread-safe and callable from any thread at any time.
/// - stop() may be called while another thread is blocked in read(); read() then returns std::nullopt promptly.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

/// Which implementation to use. YAML strings in parentheses.
enum class CameraBackendKind : uint8_t {
  kArgus = 0,   ///< ("argus") libargus EGLStream consumer; needs RIG_WITH_ARGUS (Jetson only).
  kGst = 1,     ///< ("gst") GStreamer nvarguscamerasrc ! nvvidconv ! appsink (or gst.source_override); RIG_WITH_GST.
  kMock = 2,    ///< ("mock") deterministic synthetic frames.
  kReplay = 3,  ///< ("replay") frames from a package raw/cam_* or EuRoC-style folder.
};

/// One sensor mode as reported by the driver (Argus ISensorMode) or a mode table (mock/replay).
struct SensorMode {
  int index = -1;                 ///< Driver mode index (Argus sensor-mode / nvarguscamerasrc sensor-mode).
  int width = 0;                  ///< Output width [px].
  int height = 0;                 ///< Output height [px].
  double min_fps = kNaN;          ///< From the frame-duration range [Hz].
  double max_fps = kNaN;          ///< [Hz].
  int64_t min_exposure_ns = -1;   ///< Exposure range [ns].
  int64_t max_exposure_ns = -1;
  double min_analog_gain = kNaN;  ///< Analog gain range [x].
  double max_analog_gain = kNaN;
  int bit_depth = -1;             ///< Sensor pixel bit depth (10 for both IMX519 and IMX219 RAW10 modes).
  bool full_fov = true;           ///< False for cropped modes (IMX219 1280x720@60 and 1920x1080@30 are crops;
                                  ///< sensor_rig picamera2_logic.py:99-104). From T2's mode tables.
  std::string description;        ///< Free text from the driver, for logs.
};

/// IMX519 focus lock (PLAN T2 `Imx519Focus`). Mirrors rig.yaml `cameras.<id>.driver.focus`.
enum class FocusMode : uint8_t {
  kNone = 0,   ///< ("none") fixed-focus lens; nothing is written (IMX219 ultrawide).
  kFixed = 1,  ///< ("fixed") write `position` once at start(); autofocus never runs (PIPELINE 5).
};

struct FocusConfig {
  FocusMode mode = FocusMode::kNone;
  int position = -1;          ///< Lens position code (VCM DAC value; range per T2's cited Arducam Focuser.py).
  int i2c_bus = -1;           ///< Linux I2C bus of the camera's mux channel (e.g. 9 or 10 on the Orin Nano).
  int i2c_addr = 0x0C;        ///< VCM 7-bit address (PIPELINE 3).
  bool allow_v4l2_fallback = true;  ///< If the VCM is kernel-bound (`UU`), try V4L2 focus_absolute on the subdev.
  std::string v4l2_subdev;    ///< Subdev node for the fallback, e.g. "/dev/v4l-subdev1"; empty = auto-detect.
};

/// GStreamer backend options. Mirrors `cameras.<id>.driver.gst`.
struct GstCameraConfig {
  /// Replaces the `nvarguscamerasrc ...` source element(s) when non-empty (laptop tests, e.g.
  /// "videotestsrc is-live=true pattern=ball"). The backend appends the caps/convert/appsink tail itself.
  std::string source_override;
};

/// Mock backend options. Mirrors `cameras.<id>.driver.mock`.
struct MockCameraConfig {
  std::string pattern = "checker";  ///< "checker" | "gradient" | "noise" | "counter" (frame number burned in).
  int64_t jitter_ns = 0;            ///< Uniform +/- jitter added to the ideal SOF times [ns].
  int drop_every_n = 0;             ///< Skip one frame every N frames (seq and time gap, as a real drop); 0 = never.
  int stall_after_frames = -1;      ///< Stop producing after this many frames (a stall); -1 = never.
  int64_t stall_ns = 0;             ///< Stall length [ns] before frames resume; 0 with stall_after_frames = forever.
  uint64_t seed = 1;                ///< RNG seed (jitter, noise pattern).
};

/// Replay backend options. Mirrors `cameras.<id>.driver.replay`.
struct ReplayCameraConfig {
  std::string path;       ///< Package raw/cam_<id> dir, or an EuRoC-style folder with data.csv + images.
  bool realtime = true;   ///< Pace frames by their timestamps (true) or deliver as fast as read() is called.
  bool loop = false;      ///< Restart at the end (timestamps keep increasing).
  bool retime = true;     ///< Shift timestamps so the first frame is at now_raw_ns() at start().
};

/// Everything a backend needs. Mirrors rig.yaml `cameras.<id>.driver` (except `id`, which is the map key).
struct CameraConfig {
  std::string id;                       ///< Config key ("main", "uw"); copied into ImageFrame::camera_id.
  CameraBackendKind backend = CameraBackendKind::kMock;
  std::string sensor;                   ///< "imx519" | "imx219" (selects T2's mode tables and validation).
  int sensor_id = 0;                    ///< Argus sensor index / nvarguscamerasrc sensor-id (CSI port order).
  int width = 0;                        ///< Requested output width [px].
  int height = 0;                       ///< Requested output height [px].
  double fps = 0.0;                     ///< Requested frame rate [Hz]; frame duration fixed at 1/fps.
  int sensor_mode = -1;                 ///< Force a mode index; -1 = select by width x height @ fps.
  PixelFormat pixel_format = PixelFormat::kNV12;  ///< Output format delivered by read().
  int64_t exposure_ns = 0;              ///< Locked exposure [ns] (range min = max); 0 = leave AE free (bring-up only).
  double analog_gain = 0.0;             ///< Locked analog gain [x] (range min = max); 0 = leave free.
  double isp_digital_gain = 1.0;        ///< ISP digital gain range min = max [x].
  bool ae_lock = true;                  ///< Argus AE lock.
  bool awb_lock = true;                 ///< Argus AWB lock.
  std::string awb_mode = "auto";        ///< Argus AWB mode before lock: "off" | "auto" | "daylight" | ... (T2 maps).
  bool denoise = false;                 ///< ISP noise reduction (off: raw texture for SLAM/tags; PLAN T2).
  bool edge_enhance = false;            ///< ISP edge enhancement.
  int buffer_count = 4;                 ///< Driver/EGLStream buffers in flight.
  double drop_gap_factor = 1.5;         ///< An SOF gap > factor / fps counts as a drop (with or without seq gap).
  FocusConfig focus;
  GstCameraConfig gst;
  MockCameraConfig mock;
  ReplayCameraConfig replay;
};

/// Running counters. All counts since start(). Copy returned by stats(); thread-safe.
struct CameraStats {
  uint64_t frames = 0;              ///< Frames delivered by read().
  uint64_t drops_seq_gap = 0;       ///< Missing sequence numbers reported by the driver.
  uint64_t drops_time_gap = 0;      ///< SOF gaps > drop_gap_factor / fps without a seq gap.
  uint64_t drops_backend_queue = 0; ///< Frames the backend discarded because read() was not called in time.
  uint64_t drops_error = 0;         ///< Frames lost to driver/capture errors.
  uint64_t errors = 0;              ///< Non-fatal errors (e.g. one failed capture request).
  double fps_estimate = kNaN;       ///< Frame rate over the last 1 s of SOF times [Hz].
  SummaryStats inter_frame_ns;      ///< SOF-to-SOF interval over the last window [ns].
  SummaryStats latency_ns;          ///< t_arrival - t_sof over the last window [ns].
  TimeNs last_t_sof_ns = kInvalidTime;
  int sensor_mode = -1;             ///< Mode index in use.
  int64_t exposure_ns = -1;         ///< Last frame's exposure readback [ns] (confirms the lock).
  double analog_gain = kNaN;        ///< Last frame's gain readback [x].
  FocusStatus focus_status = FocusStatus::kUnknown;
  int focus_position = -1;

  uint64_t drops_total() const noexcept { return drops_seq_gap + drops_time_gap + drops_backend_queue + drops_error; }
};

/// Abstract camera. See the file comment for lifecycle and threading.
class CameraBackend {
 public:
  virtual ~CameraBackend() = default;

  /// The configuration given to make_camera(), with `sensor_mode` resolved after start().
  virtual const CameraConfig& config() const = 0;

  /// Opens the device, selects the mode (fails with kNotFound listing the available modes if none matches),
  /// applies exposure/gain/AWB/ISP locks and focus, and starts streaming. kFailedPrecondition if already started.
  virtual Status start() = 0;

  /// Stops streaming and releases the device. Idempotent; unblocks a concurrent read().
  virtual Status stop() = 0;

  /// Next frame in capture order, waiting up to `timeout`. std::nullopt on timeout, after stop(), or after a fatal
  /// error (then status() is not OK). Frames are never reordered; skipped frames are counted in stats().
  virtual std::optional<ImageFrame> read(std::chrono::nanoseconds timeout) = 0;

  /// Most recent frame delivered by read() (std::nullopt before the first). Does not consume.
  virtual std::optional<ImageFrame> latest() const = 0;

  virtual CameraStats stats() const = 0;

  /// OK while healthy; the first fatal error otherwise (device lost, Argus daemon died, replay file corrupt).
  virtual Status status() const = 0;

  /// Sensor modes available on this device. Callable before start() (opens the provider briefly if needed).
  virtual Result<std::vector<SensorMode>> modes() const = 0;
};

/// Creates the backend selected by `config.backend`. Does not start it. kUnimplemented if that backend was not
/// compiled in (RIG_WITH_ARGUS / RIG_WITH_GST), kInvalidArgument for an inconsistent config.
/// Implemented by T2 (common/drivers/camera/camera_factory.cpp; T0 ships a stub).
Result<std::unique_ptr<CameraBackend>> make_camera(const CameraConfig& config);

/// Chooses the mode for `config` from `modes`: the forced index if config.sensor_mode >= 0, else the full-FOV mode
/// with exactly width x height whose fps range contains config.fps. kNotFound with a message listing the modes
/// otherwise. Pure function (table-driven tests in T2).
Result<SensorMode> select_sensor_mode(const std::vector<SensorMode>& modes, const CameraConfig& config);

/// t_mid = t_sof + exposure/2 + readout/2 (PIPELINE 5); returns t_sof if exposure or readout is unknown (< 0).
constexpr TimeNs mid_exposure_time(TimeNs t_sof_ns, int64_t exposure_ns, int64_t readout_ns) noexcept {
  if (t_sof_ns == kInvalidTime) return kInvalidTime;
  if (exposure_ns < 0 || readout_ns < 0) return t_sof_ns;
  return t_sof_ns + exposure_ns / 2 + readout_ns / 2;
}

}  // namespace rig
