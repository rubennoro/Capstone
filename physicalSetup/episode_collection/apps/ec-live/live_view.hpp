// SPDX-License-Identifier: MIT
/// @file live_view.hpp
/// @brief ec-live (L1): the pure part of the live viewer: rate meters, the 5 s IMU history, the overlay and status
/// texts, the mosaic layout and the mosaic renderer. No sensors, threads or windows here, so the tests in
/// tests/live/ can drive it directly (they compile live_view.cpp into the test area).
///
/// Threading: none of these types are thread-safe; ec-live uses them from its single render (main) thread.
#pragma once

#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "rig/config/rig_config.hpp"
#include "rig/drivers/camera_backend.hpp"
#include "rig/drivers/imu_backend.hpp"
#include "rig/status.hpp"
#include "rig/types.hpp"

namespace eclive {

using rig::TimeNs;

constexpr TimeNs kSecondNs = 1000000000LL;

// ------------------------------------------------------------------------------------------------------------------
// Config overrides
// ------------------------------------------------------------------------------------------------------------------

/// `--mock`: every camera backend -> mock (sizes, fps, pixel format and exposure kept), the IMU -> sim with motion
/// on (so the plots move on the laptop). Edits the struct only, never the yaml.
void apply_mock_overrides(rig::RigConfig& config);

/// Camera ids in display order: role main first, then ultrawide, then any other (config map order).
std::vector<std::string> display_order(const rig::RigConfig& config);

// ------------------------------------------------------------------------------------------------------------------
// Measurements
// ------------------------------------------------------------------------------------------------------------------

/// Rate of a monotonic counter over a sliding window (default 1 s): add (time, count) pairs, rate = dcount / dt over
/// the oldest pair still inside the window. 0 until two pairs span some time.
class RateMeter {
 public:
  explicit RateMeter(TimeNs window_ns = kSecondNs) : window_ns_(window_ns) {}
  void add(TimeNs t_ns, uint64_t count);
  double rate_hz() const;
  void clear() { pts_.clear(); }

 private:
  TimeNs window_ns_;
  std::deque<std::pair<TimeNs, uint64_t>> pts_;
};

/// One IMU point for the plots (accel m/s^2, gyro deg/s).
struct ImuPoint {
  TimeNs t_ns = rig::kInvalidTime;
  float acc[3] = {0, 0, 0};
  float gyro_dps[3] = {0, 0, 0};
};

/// Rolling history of the last `span_ns` (default 5 s) of IMU samples, plus the latest sample and temperature.
class ImuHistory {
 public:
  explicit ImuHistory(TimeNs span_ns = 5 * kSecondNs) : span_ns_(span_ns) {}
  /// Samples with t_ns == kInvalidTime update latest() but are not plotted.
  void add(const rig::ImuSample& s);
  /// Drops points older than now - span.
  void prune(TimeNs now_ns);
  const std::deque<ImuPoint>& points() const { return pts_; }
  bool has_sample() const { return has_latest_; }
  const rig::ImuSample& latest() const { return latest_; }
  /// Last non-NaN temperature [deg C]; NaN if none yet.
  float temperature_c() const { return temp_c_; }
  TimeNs span_ns() const { return span_ns_; }

 private:
  TimeNs span_ns_;
  std::deque<ImuPoint> pts_;
  rig::ImuSample latest_;
  bool has_latest_ = false;
  float temp_c_ = std::numeric_limits<float>::quiet_NaN();
};

// ------------------------------------------------------------------------------------------------------------------
// What the renderer shows (filled by the app each tick)
// ------------------------------------------------------------------------------------------------------------------

enum class Severity : uint8_t { kOk = 0, kInfo = 1, kWarn = 2, kBad = 3 };

struct TextLine {
  std::string text;
  Severity severity = Severity::kOk;
};

struct CameraView {
  std::string id;
  std::string role;               ///< "main" | "ultrawide".
  std::string backend;            ///< "argus" | "gst" | "mock" | "replay".
  int width = 0, height = 0;      ///< Configured capture size.
  double fps = 0;                 ///< Configured fps.
  bool started = false;           ///< Its capture service started.
  std::string error;              ///< Start failure or runtime fatal error ("" = none).
  TimeNs started_ns = rig::kInvalidTime;  ///< When it started (stall clock before the first frame).

  // Newest frame (only meaningful when has_frame).
  bool has_frame = false;
  uint64_t frames_seen = 0;       ///< Frames published by the capture thread (hub counter).
  uint64_t seq = 0;
  TimeNs t_mid_ns = rig::kInvalidTime;
  TimeNs last_rx_ns = rig::kInvalidTime;  ///< Host time ec-live last saw a new frame.
  rig::FrameMeta meta;
  cv::Mat image;                  ///< BGR (any size; scaled to the tile when rendered).
  std::string convert_error;      ///< frame_to_bgr failure, shown on the tile.

  double measured_fps = 0;        ///< Hub frame counter over 1 s.
  bool has_stats = false;
  rig::CameraStats stats;
};

struct ImuView {
  bool enabled = false;           ///< An IMU was requested (not --no-imu).
  std::string id;
  std::string backend;            ///< "lsm6dsox_i2c" | "sim" | "replay".
  double odr_hz = 0;
  bool started = false;
  std::string error;
  double measured_rate_hz = 0;    ///< Hub sample counter over 1 s.
  bool has_stats = false;
  rig::ImuStats stats;
  TimeNs started_ns = rig::kInvalidTime;
  TimeNs last_rx_ns = rig::kInvalidTime;
};

struct RenderOptions {
  int width = 1600;               ///< Mosaic width [px].
  int max_camera_height = 720;    ///< Cap on the camera row height [px].
  int imu_panel_height = 230;     ///< IMU strip height [px] (when an IMU was requested).
  double stall_s = 1.0;           ///< "NO FRAMES" after this long without a new frame [s].
  int gap = 4;                    ///< Gap between tiles [px].
};

struct Layout {
  cv::Size size;                  ///< Whole mosaic.
  std::vector<cv::Rect> tiles;    ///< One per camera, in the order given.
  std::vector<cv::Rect> images;   ///< Aspect-kept image rect inside each tile (centred).
  cv::Rect imu;                   ///< IMU strip; empty without an IMU.
};

/// Largest rect with `src`'s aspect ratio that fits in `box`, centred in it.
cv::Rect fit_rect(cv::Size src, const cv::Rect& box);
/// Mosaic layout: cameras side by side in equal-width columns (each scaled to fit, aspect kept, row height capped at
/// max_camera_height), the IMU strip under them. With no cameras, the mosaic is the IMU strip only.
Layout compute_layout(const RenderOptions& opt, const std::vector<cv::Size>& camera_sizes, bool with_imu);

/// Overlay lines of a camera tile (id/mode, fps, drops, exposure/gain, focus, colour range, frame age).
std::vector<TextLine> camera_overlay_lines(const CameraView& cam, TimeNs now_ns);
/// "NO FRAMES for N.N s" if the camera started but delivered nothing new for > stall_s; "" otherwise.
std::string stall_text(const CameraView& cam, TimeNs now_ns, double stall_s);
/// Text lines of the IMU panel.
std::vector<TextLine> imu_panel_lines(const ImuView& imu, const ImuHistory& hist);
/// One-line terminal status (rates, drops, |a|), printed once per second.
std::string status_line(const std::vector<CameraView>& cams, const ImuView& imu, const ImuHistory& hist,
                        TimeNs now_ns, TimeNs t0_ns, double stall_s);

/// Symmetric plot range: the smallest of 1, 2, 5 x 10^k that is >= max(max_abs * 1.1, min_range).
double plot_range(double max_abs, double min_range);

/// Draws the whole mosaic (CV_8UC3, layout.size).
cv::Mat render_mosaic(const Layout& layout, const std::vector<CameraView>& cams, const ImuView& imu,
                      const ImuHistory& hist, const RenderOptions& opt, TimeNs now_ns);

/// Converts the newest frame for display: frame_to_bgr (honours FrameMeta::color_range), then INTER_AREA down to
/// `target` (no-op when target is empty or the same size). Large NV12 frames are first halved in the YCbCr
/// domain (both planes), which keeps the range handling of frame_to_bgr and quarters its cost.
rig::Result<cv::Mat> frame_to_display(const rig::ImageFrame& frame, cv::Size target);

/// Backend name strings as in rig.yaml.
const char* camera_backend_name(rig::CameraBackendKind k);
const char* imu_backend_name(rig::ImuBackendKind k);

/// A one-line "what to check" hint for a sensor that failed to start ("" if none applies).
std::string failure_hint(bool is_camera, const std::string& backend, const rig::Status& error);

/// Wraps `text` into lines of at most `max_chars` characters (breaking at spaces where possible).
std::vector<std::string> wrap_text(const std::string& text, std::size_t max_chars);

}  // namespace eclive
