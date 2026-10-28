// SPDX-License-Identifier: MIT
/// @file rig/drivers/camera/probe_report.hpp
/// @brief ec-cam-probe analysis (PLAN T2 / hardware-day test H2): exact capture statistics over every delivered
/// frame and PASS/FAIL checks. In rigcommon so it is unit-tested; the app only does I/O. Owned by T2.
/// Threading: ProbeAnalyzer is not thread-safe (one per camera, fed by the thread that calls read()).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rig/drivers/camera_backend.hpp"
#include "rig/types.hpp"

namespace rig {

/// H2 acceptance (IMPLEMENTATION_PLAN section 5): main 60 fps / uw 30 fps; p99 jitter < 2 ms; 0 unexplained drops;
/// exposure and gain locked; focus status known.
struct ProbeCriteria {
  double expected_fps = 0.0;            ///< Configured fps.
  double fps_rel_tolerance = 0.005;     ///< |measured / expected - 1| allowed.
  double max_jitter_p99_ns = 2e6;       ///< p99 of |SOF interval - 1/fps|.
  uint64_t max_drops = 0;               ///< drops_total() allowed.
  int64_t expected_exposure_ns = 0;     ///< Locked exposure from the config; 0 = AE free (lock check skipped).
  double expected_gain = 0.0;           ///< Locked analog gain; 0 = free.
  double lock_rel_tolerance = 0.05;     ///< Readback mean within this of the request (sensor line quantization).
  double lock_spread_rel = 0.01;        ///< (max - min) / request allowed (constant = locked).
  bool require_focus_known = true;      ///< Focus status must not be unknown/failed.
  double max_latency_p99_ns = 200e6;    ///< t_arrival - t_sof p99 (timestamp-domain sanity).
};

struct ProbeCheck {
  std::string name;
  bool pass = false;
  std::string detail;
};

struct ProbeReport {
  std::string camera_id;
  uint64_t frames = 0;
  double duration_s = 0.0;              ///< First to last SOF [s].
  double fps = kNaN;                    ///< (frames - 1) / duration.
  SummaryStats interval_ns;             ///< SOF-to-SOF.
  SummaryStats jitter_ns;               ///< |interval - 1/fps|.
  SummaryStats latency_ns;              ///< t_arrival - t_sof.
  SummaryStats mid_minus_sof_ns;        ///< t_mid - t_sof (= exposure/2 + readout/2 on Argus).
  uint64_t seq_missing = 0;             ///< Missing seq numbers among delivered frames.
  uint64_t negative_latency = 0;        ///< Frames with t_arrival < t_sof (impossible: wrong domain).
  int64_t exposure_min_ns = -1, exposure_max_ns = -1;
  double exposure_mean_ns = kNaN;
  double gain_min = kNaN, gain_max = kNaN, gain_mean = kNaN;
  double digital_gain_min = kNaN, digital_gain_max = kNaN;
  int64_t readout_ns = -1;              ///< Last frame's readout.
  CameraStats stats;                    ///< Backend stats at the end (drops, focus, time source, cross-check).
  std::vector<ProbeCheck> checks;
  bool pass() const;
};

class ProbeAnalyzer {
 public:
  explicit ProbeAnalyzer(std::string camera_id, double nominal_fps);
  void add(const ImageFrame& f);
  /// Computes the report and evaluates `criteria` (criteria.expected_fps 0 -> nominal_fps).
  ProbeReport finish(const CameraStats& final_stats, ProbeCriteria criteria) const;
  uint64_t frames() const noexcept { return static_cast<uint64_t>(sof_.size()); }

 private:
  std::string id_;
  double fps_;
  std::vector<TimeNs> sof_;
  std::vector<double> latency_, mid_sof_;
  std::vector<uint64_t> seq_;
  std::vector<int64_t> exposure_;
  std::vector<double> gain_, dgain_;
  int64_t readout_ = -1;
};

/// Multi-line human-readable report (ms units) with PASS/FAIL per check.
std::string format_probe_report(const ProbeReport& r);

/// Variance of the Laplacian of the luma/gray plane in the central 50 % crop (focus-sweep sharpness score).
/// NaN for formats without a luma plane.
double frame_sharpness(const ImageFrame& f);

}  // namespace rig
