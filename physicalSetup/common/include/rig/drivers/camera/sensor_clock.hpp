// SPDX-License-Identifier: MIT
/// @file rig/drivers/camera/sensor_clock.hpp
/// @brief Camera timestamp semantics and clock-domain conversion into the rig clock (CLOCK_MONOTONIC_RAW,
/// now_raw_ns()). Pure logic, unit-tested on the laptop; used by the Argus and GStreamer backends. Owned by T2.
///
/// ## What the Argus timestamp means (r36.4.3 headers, Argus/CaptureMetadata.h:393-397)
/// `ICaptureMetadata::getSensorTimestamp()`: "the start timestamp for the sensor (in nanoseconds). This is the time
/// that the first data from this capture arrives from the sensor." On a rolling-shutter sensor the first data is row
/// 0 being read out, i.e. the END of row 0's exposure, not its start. `getFrameReadoutTime()` (:367-371) is "from the
/// beginning of the first line to the beginning of the last line". So, with E = exposure and R = readout:
///   row r exposes during [ts - E + r/(H-1)*R, ts + r/(H-1)*R]; the middle row's mid-exposure is ts - E/2 + R/2.
/// The frozen ImageFrame defines `t_sof_ns` as "first row starts exposing" and t_mid = t_sof + E/2 + R/2
/// (mid_exposure_time). Hence argus_start_of_exposure() = ts - E, which makes mid_exposure_time() return
/// ts - E/2 + R/2. (Feeding ts itself as t_sof would make t_mid one full exposure late: 8 ms on main, 4 ms on uw.)
/// [H: verify with the LED/stamp-offset test, BRINGUP step 13: a residual constant offset goes to
///  cameras.<id>.time.offset_ns, never into the raw stamps.]
///
/// ## Clock domain (uncertain on r36; resolved at runtime, verified on H)
/// - NVIDIA staff disagree: getSensorTimestamp() is "the kernel timestamp (i.e. MONOTONIC_RAW) of sensor
///   start-of-frame" (JerryChang) versus "CLOCK_REALTIME" (ShaneCCC), both in
///   https://forums.developer.nvidia.com/t/get-frame-system-time/268916?page=2 (2023).
/// - `Ext::ISensorTimestampTsc::getSensorSofTimestampTsc()` (Argus/Ext/SensorTimestampTsc.h) is the VI hardware SOF
///   in the Tegra TSC domain; MONOTONIC_RAW = TSC - offset_ns with offset_ns from
///   /sys/devices/system/clocksource/clocksource0/offset_ns (same thread; and
///   https://forums.developer.nvidia.com/t/camera-sof-timestamp-in-the-future/329038 for r36.4.0 precision issues).
/// - SensorClockResolver therefore classifies getSensorTimestamp() against MONOTONIC_RAW / MONOTONIC / REALTIME read
///   at frame arrival (a plausible domain gives -max_future <= arrival - ts <= max_latency), prefers MONOTONIC_RAW, falls
///   back to TSC - offset_ns, and as a last resort converts MONOTONIC/REALTIME with a per-frame clock anchor.
///   Everything goes through SensorClockResolver::to_raw_ns() so H can change the policy in one place.
///   TODO-verify(H, BRINGUP 13): which domain is chosen on L4T 36.4.3 and that the TSC cross-check is ~0.
/// Threading: value types; SensorClockResolver is not thread-safe (one per camera, owned by the capture thread).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "rig/status.hpp"
#include "rig/time/clock.hpp"
#include "rig/types.hpp"

namespace rig {

/// Clock a driver timestamp is in.
enum class SensorClockDomain : uint8_t {
  kUnknown = 0,
  kMonotonicRaw = 1,  ///< CLOCK_MONOTONIC_RAW (the rig clock; no conversion).
  kMonotonic = 2,     ///< CLOCK_MONOTONIC (NTP-slewed; converted with a per-frame anchor).
  kRealtime = 3,      ///< CLOCK_REALTIME (wall clock; converted with a per-frame anchor; NTP steps break it).
  kTscOffset = 4,     ///< Tegra TSC ns minus clocksource0/offset_ns (= MONOTONIC_RAW per NVIDIA).
};

const char* to_string(SensorClockDomain d) noexcept;

/// Maximum plausible arrival - timestamp latency used for classification [ns]. Generous: Argus delivers ~1-3 frame
/// periods after SOF; 9 fps modes have 111 ms periods.
inline constexpr int64_t kMaxPlausibleCameraLatencyNs = 1'000'000'000;

/// How far a sensor stamp may lie in the FUTURE of its arrival and still be plausible [ns] (R1 F3). On r36.4.0 the
/// VI SOF stamp was reported slightly ahead of CLOCK_MONOTONIC_RAW (forum 329038: offset_ns truncation in the TSC
/// conversion), so a strict latency >= 0 rule could reject the right domain. Such frames are accepted and counted
/// in CameraStats::timestamp_anomalies by the backend.
inline constexpr int64_t kMaxFutureCameraStampNs = 2'000'000;

/// Domain whose clock, read at frame arrival (`at_arrival`), puts `ts_ns` in
/// [arrival - max_latency, arrival + max_future]. MONOTONIC_RAW wins ties (MONOTONIC differs from it only by NTP
/// slew, often < max_latency).
SensorClockDomain classify_sensor_clock(uint64_t ts_ns, const ClockAnchor& at_arrival,
                                        int64_t max_latency_ns = kMaxPlausibleCameraLatencyNs,
                                        int64_t max_future_ns = kMaxFutureCameraStampNs) noexcept;

/// Converts `ts_ns` of domain `d` into MONOTONIC_RAW using the anchor taken at arrival. kTscOffset requires
/// `tsc_offset_ns`. Returns kInvalidTime for kUnknown or missing inputs.
TimeNs sensor_clock_to_raw_ns(uint64_t ts_ns, SensorClockDomain d, const ClockAnchor& at_arrival,
                              std::optional<int64_t> tsc_offset_ns = std::nullopt) noexcept;

/// Start of row-0 exposure from Argus' first-data time (see file comment): ts - exposure. Unchanged if exposure < 0.
constexpr TimeNs argus_start_of_exposure(TimeNs first_data_raw_ns, int64_t exposure_ns) noexcept {
  if (first_data_raw_ns == kInvalidTime) return kInvalidTime;
  return exposure_ns < 0 ? first_data_raw_ns : first_data_raw_ns - exposure_ns;
}

/// Reads <sysroot>/sys/devices/system/clocksource/clocksource0/offset_ns (Jetson). kNotFound if absent.
Result<int64_t> read_tsc_offset_ns(const std::string& sysroot = "");

/// GStreamer: buffer PTS is pipeline running time; with the pipeline clock forced to the MONOTONIC system clock
/// (GstSystemClock clock-type=GST_CLOCK_TYPE_MONOTONIC) the capture clock time is base_time + running_time
/// (GStreamer design doc "synchronisation": running_time = clock_time - base_time). Then MONOTONIC -> RAW with the
/// anchor. For nvarguscamerasrc this PTS is a buffer time inside the source, ~30 ms after mid-exposure and jittery
/// (sensor_rig argus_logic.py:11-16; PLAN T2): fallback only.
TimeNs gst_clock_time_to_raw_ns(uint64_t base_time_ns, uint64_t running_time_ns, const ClockAnchor& at_arrival) noexcept;

/// Decides, from the first frames of a stream, which timestamp source and domain to trust, then converts.
class SensorClockResolver {
 public:
  struct Decision {
    SensorClockDomain domain = SensorClockDomain::kUnknown;  ///< Domain of the timestamp actually used.
    bool use_tsc = false;      ///< true: getSensorSofTimestampTsc - offset_ns; false: getSensorTimestamp.
    bool raw_mono_ambiguous = false;  ///< MONOTONIC and MONOTONIC_RAW were both plausible (assumed RAW).
    std::string description;   ///< e.g. "argus getSensorTimestamp in monotonic_raw (tsc cross-check available)".
  };

  explicit SensorClockResolver(int frames_needed = 5, int64_t max_latency_ns = kMaxPlausibleCameraLatencyNs,
                               int64_t max_future_ns = kMaxFutureCameraStampNs);

  /// Adds one frame's raw stamps. `tsc_sof_ns` / `tsc_offset_ns`: std::nullopt if the extension or sysfs node is
  /// unavailable. Ignored once decided.
  void add(uint64_t sensor_ts_ns, std::optional<uint64_t> tsc_sof_ns, std::optional<int64_t> tsc_offset_ns,
           const ClockAnchor& at_arrival);

  /// True after `frames_needed` frames were added.
  bool ready() const noexcept { return static_cast<int>(samples_.size()) >= frames_needed_; }

  /// The decision (computed on first call after ready()). kFailedPrecondition before ready(); kDeviceError with
  /// the observed latencies per domain if no source is plausible.
  Result<Decision> decide();

  /// First-data time in MONOTONIC_RAW under the decision; kInvalidTime before a successful decide().
  TimeNs to_raw_ns(uint64_t sensor_ts_ns, std::optional<uint64_t> tsc_sof_ns, std::optional<int64_t> tsc_offset_ns,
                   const ClockAnchor& at_arrival) const noexcept;

 private:
  struct Sample {
    uint64_t sensor_ts;
    std::optional<uint64_t> tsc;
    std::optional<int64_t> offset;
    ClockAnchor arrival;
  };
  int frames_needed_;
  int64_t max_latency_ns_;
  int64_t max_future_ns_;
  std::vector<Sample> samples_;
  std::optional<Decision> decision_;
};

}  // namespace rig
