// SPDX-License-Identifier: MIT
// ec-cam-probe analysis (rig/drivers/camera/probe_report.hpp).
#include "rig/drivers/camera/probe_report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>

#include <opencv2/imgproc.hpp>

#include "rig/drivers/camera/frame_tracker.hpp"

namespace rig {

bool ProbeReport::pass() const {
  return !checks.empty() && std::all_of(checks.begin(), checks.end(), [](const ProbeCheck& c) { return c.pass; });
}

ProbeAnalyzer::ProbeAnalyzer(std::string camera_id, double nominal_fps) : id_(std::move(camera_id)), fps_(nominal_fps) {}

void ProbeAnalyzer::add(const ImageFrame& f) {
  sof_.push_back(f.t_sof_ns);
  latency_.push_back(static_cast<double>(f.t_arrival_ns - f.t_sof_ns));
  mid_sof_.push_back(static_cast<double>(f.t_mid_ns - f.t_sof_ns));
  seq_.push_back(f.seq);
  if (f.meta.exposure_ns >= 0) exposure_.push_back(f.meta.exposure_ns);
  if (!std::isnan(f.meta.analog_gain)) gain_.push_back(f.meta.analog_gain);
  if (!std::isnan(f.meta.digital_gain)) dgain_.push_back(f.meta.digital_gain);
  readout_ = f.meta.readout_ns;
}

namespace {
std::string ms(double ns) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.3f ms", ns * 1e-6);
  return b;
}
}  // namespace

ProbeReport ProbeAnalyzer::finish(const CameraStats& st, ProbeCriteria c) const {
  ProbeReport r;
  r.camera_id = id_;
  r.stats = st;
  r.frames = sof_.size();
  if (!(c.expected_fps > 0)) c.expected_fps = fps_;
  const double period = c.expected_fps > 0 ? 1e9 / c.expected_fps : kNaN;
  std::vector<double> iv, jit;
  for (std::size_t i = 1; i < sof_.size(); ++i) {
    const double d = static_cast<double>(sof_[i] - sof_[i - 1]);
    iv.push_back(d);
    jit.push_back(std::fabs(d - period));
    if (seq_[i] > seq_[i - 1] + 1) r.seq_missing += seq_[i] - seq_[i - 1] - 1;
  }
  r.interval_ns = summarize(iv);
  r.jitter_ns = summarize(jit);
  r.latency_ns = summarize(latency_);
  r.mid_minus_sof_ns = summarize(mid_sof_);
  r.negative_latency = static_cast<uint64_t>(std::count_if(latency_.begin(), latency_.end(), [](double v) { return v < 0; }));
  if (sof_.size() >= 2) {
    r.duration_s = static_cast<double>(sof_.back() - sof_.front()) * 1e-9;
    r.fps = r.duration_s > 0 ? static_cast<double>(sof_.size() - 1) / r.duration_s : kNaN;
  }
  if (!exposure_.empty()) {
    r.exposure_min_ns = *std::min_element(exposure_.begin(), exposure_.end());
    r.exposure_max_ns = *std::max_element(exposure_.begin(), exposure_.end());
    r.exposure_mean_ns = std::accumulate(exposure_.begin(), exposure_.end(), 0.0) / static_cast<double>(exposure_.size());
  }
  if (!gain_.empty()) {
    r.gain_min = *std::min_element(gain_.begin(), gain_.end());
    r.gain_max = *std::max_element(gain_.begin(), gain_.end());
    r.gain_mean = std::accumulate(gain_.begin(), gain_.end(), 0.0) / static_cast<double>(gain_.size());
  }
  if (!dgain_.empty()) {
    r.digital_gain_min = *std::min_element(dgain_.begin(), dgain_.end());
    r.digital_gain_max = *std::max_element(dgain_.begin(), dgain_.end());
  }
  r.readout_ns = readout_;

  char b[256];
  auto add = [&](const std::string& name, bool pass, const std::string& detail) {
    r.checks.push_back(ProbeCheck{name, pass, detail});
  };
  // fps
  const bool fps_ok = !std::isnan(r.fps) && std::fabs(r.fps / c.expected_fps - 1.0) <= c.fps_rel_tolerance;
  std::snprintf(b, sizeof(b), "%.4f fps measured vs %.4g expected (tolerance %.2f %%)", r.fps, c.expected_fps,
                c.fps_rel_tolerance * 100);
  add("fps", fps_ok, b);
  add("jitter_p99", r.jitter_ns.count > 0 && r.jitter_ns.p99 < c.max_jitter_p99_ns,
      "p50 " + ms(r.jitter_ns.p50) + ", p99 " + ms(r.jitter_ns.p99) + ", max " + ms(r.jitter_ns.max) + " (limit p99 " +
          ms(c.max_jitter_p99_ns) + ")");
  std::snprintf(b, sizeof(b), "total %llu (seq gap %llu, time gap %llu, backend queue %llu, error %llu); errors %llu",
                static_cast<unsigned long long>(st.drops_total()), static_cast<unsigned long long>(st.drops_seq_gap),
                static_cast<unsigned long long>(st.drops_time_gap), static_cast<unsigned long long>(st.drops_backend_queue),
                static_cast<unsigned long long>(st.drops_error), static_cast<unsigned long long>(st.errors));
  add("drops", st.drops_total() <= c.max_drops && r.seq_missing <= c.max_drops, b);
  add("latency", r.negative_latency == 0 && r.latency_ns.count > 0 && r.latency_ns.p99 < c.max_latency_p99_ns,
      "t_arrival - t_sof p50 " + ms(r.latency_ns.p50) + ", p99 " + ms(r.latency_ns.p99) + ", max " +
          ms(r.latency_ns.max) + ", negative " + std::to_string(r.negative_latency) + " (time source: " +
          st.time_source + ")");
  if (c.expected_exposure_ns > 0) {
    const double req = static_cast<double>(c.expected_exposure_ns);
    const bool ok = !exposure_.empty() &&
                    static_cast<double>(r.exposure_max_ns - r.exposure_min_ns) <= c.lock_spread_rel * req &&
                    std::fabs(r.exposure_mean_ns / req - 1.0) <= c.lock_rel_tolerance;
    std::snprintf(b, sizeof(b), "readback %lld..%lld ns (mean %.0f) vs locked %lld ns", static_cast<long long>(r.exposure_min_ns),
                  static_cast<long long>(r.exposure_max_ns), r.exposure_mean_ns, static_cast<long long>(c.expected_exposure_ns));
    add("exposure_locked", ok, exposure_.empty() ? std::string("no exposure readback (gst/mock backend?)") : std::string(b));
  }
  if (c.expected_gain > 0) {
    const bool ok = !gain_.empty() && (r.gain_max - r.gain_min) <= c.lock_spread_rel * c.expected_gain &&
                    std::fabs(r.gain_mean / c.expected_gain - 1.0) <= c.lock_rel_tolerance;
    std::snprintf(b, sizeof(b), "readback %.4g..%.4g (mean %.4g) vs locked %.4g; ISP digital gain %.4g..%.4g", r.gain_min,
                  r.gain_max, r.gain_mean, c.expected_gain, r.digital_gain_min, r.digital_gain_max);
    add("gain_locked", ok, gain_.empty() ? std::string("no gain readback (gst/mock backend?)") : std::string(b));
  }
  if (c.require_focus_known) {
    const FocusStatus fs = st.focus_status;
    add("focus", fs != FocusStatus::kUnknown && fs != FocusStatus::kFailed,
        std::string(to_string(fs)) + " position " + std::to_string(st.focus_position) + ": " + st.focus_detail);
  }
  return r;
}

std::string format_probe_report(const ProbeReport& r) {
  std::ostringstream os;
  char b[256];
  os << "camera " << r.camera_id << ": " << (r.pass() ? "PASS" : "FAIL") << "\n";
  std::snprintf(b, sizeof(b), "  frames %llu over %.2f s -> %.4f fps; mode %d; time source %s\n",
                static_cast<unsigned long long>(r.frames), r.duration_s, r.fps, r.stats.sensor_mode,
                r.stats.time_source.c_str());
  os << b;
  std::snprintf(b, sizeof(b), "  interval p50 %.3f ms, p99 %.3f ms, min %.3f, max %.3f ms\n", r.interval_ns.p50 * 1e-6,
                r.interval_ns.p99 * 1e-6, r.interval_ns.min * 1e-6, r.interval_ns.max * 1e-6);
  os << b;
  std::snprintf(b, sizeof(b), "  t_mid - t_sof %.3f ms (exposure/2 + readout/2); readout %lld ns\n",
                r.mid_minus_sof_ns.p50 * 1e-6, static_cast<long long>(r.readout_ns));
  os << b;
  if (r.stats.tsc_cross_check_ns.count > 0) {
    std::snprintf(b, sizeof(b), "  TSC cross-check (TSC SOF - offset_ns) - sensor_ts: p50 %.3f us, min %.3f, max %.3f us\n",
                  r.stats.tsc_cross_check_ns.p50 * 1e-3, r.stats.tsc_cross_check_ns.min * 1e-3,
                  r.stats.tsc_cross_check_ns.max * 1e-3);
    os << b;
  }
  for (const ProbeCheck& c : r.checks) os << "  [" << (c.pass ? "PASS" : "FAIL") << "] " << c.name << ": " << c.detail << "\n";
  return os.str();
}

double frame_sharpness(const ImageFrame& f) {
  if (!f.valid() || (f.fmt != PixelFormat::kNV12 && f.fmt != PixelFormat::kGRAY8)) return kNaN;
  const Plane& p = f.planes[0];
  const cv::Mat y(f.height, f.width, CV_8UC1, const_cast<uint8_t*>(f.plane_data(0)), p.stride);
  const cv::Rect roi(f.width / 4, f.height / 4, std::max(1, f.width / 2), std::max(1, f.height / 2));
  cv::Mat lap;
  cv::Laplacian(y(roi), lap, CV_64F);
  cv::Scalar mean, sd;
  cv::meanStdDev(lap, mean, sd);
  return sd[0] * sd[0];
}

}  // namespace rig
