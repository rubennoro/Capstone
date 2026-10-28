// SPDX-License-Identifier: MIT
// ec-live (L1): pure model + renderer. See live_view.hpp.
#include "live_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <memory>

#include <opencv2/imgproc.hpp>

#include "rig/imageprep/imageprep.hpp"

namespace eclive {
namespace {

constexpr double kRadToDeg = 57.29577951308232;

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof(buf), f, ap);
  va_end(ap);
  return buf;
}

cv::Scalar color_of(Severity s) {
  switch (s) {
    case Severity::kOk: return {235, 235, 235};
    case Severity::kInfo: return {200, 200, 120};
    case Severity::kWarn: return {0, 210, 255};
    case Severity::kBad: return {60, 60, 255};
  }
  return {255, 255, 255};
}

constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;

/// Text lines on a darkened box at `org` (top-left). Returns the box.
cv::Rect draw_text_box(cv::Mat& img, const std::vector<TextLine>& lines, cv::Point org, double scale, int line_h,
                       int max_w) {
  if (lines.empty()) return {};
  int w = 0;
  for (const auto& l : lines) {
    int base = 0;
    w = std::max(w, cv::getTextSize(l.text, kFont, scale, 1, &base).width);
  }
  cv::Rect box(org.x, org.y, std::min(w + 12, max_w), static_cast<int>(lines.size()) * line_h + 8);
  box &= cv::Rect(0, 0, img.cols, img.rows);
  if (box.area() > 0) {
    cv::Mat roi = img(box);
    roi.convertTo(roi, -1, 0.35, 0);  // darken for contrast
  }
  int y = org.y + line_h;
  for (const auto& l : lines) {
    cv::putText(img, l.text, {org.x + 6, y}, kFont, scale, color_of(l.severity), 1, cv::LINE_AA);
    y += line_h;
  }
  return box;
}

void draw_centered(cv::Mat& img, const cv::Rect& r, const std::vector<std::string>& lines, double scale,
                   cv::Scalar color, int thickness) {
  int base = 0;
  const int lh = cv::getTextSize("Ag", kFont, scale, thickness, &base).height + 12;
  int y = r.y + r.height / 2 - static_cast<int>(lines.size()) * lh / 2 + lh / 2;
  for (const auto& s : lines) {
    const cv::Size ts = cv::getTextSize(s, kFont, scale, thickness, &base);
    const cv::Point p(r.x + std::max(4, (r.width - ts.width) / 2), y + ts.height / 2);
    cv::putText(img, s, p + cv::Point(2, 2), kFont, scale, {0, 0, 0}, thickness + 2, cv::LINE_AA);
    cv::putText(img, s, p, kFont, scale, color, thickness, cv::LINE_AA);
    y += lh;
  }
}

/// One rolling plot of three axes (x red, y green, z blue).
void draw_plot(cv::Mat& img, const cv::Rect& r, const std::deque<ImuPoint>& pts, bool gyro, TimeNs now_ns,
               TimeNs span_ns, double min_range, const char* unit) {
  if (r.width < 20 || r.height < 20) return;
  cv::rectangle(img, r, {90, 90, 90}, 1);
  double max_abs = 0;
  for (const auto& p : pts)
    for (int i = 0; i < 3; ++i) max_abs = std::max(max_abs, std::abs(static_cast<double>(gyro ? p.gyro_dps[i] : p.acc[i])));
  const double range = plot_range(max_abs, min_range);
  const double mid = r.y + r.height / 2.0;
  const double half = (r.height - 4) / 2.0;
  cv::line(img, {r.x, static_cast<int>(mid)}, {r.x + r.width - 1, static_cast<int>(mid)}, {70, 70, 70}, 1);
  // 1 s grid.
  for (int s = 1; s * kSecondNs < span_ns; ++s) {
    const int x = r.x + r.width - 1 - static_cast<int>(static_cast<double>(s) * kSecondNs / span_ns * (r.width - 1));
    cv::line(img, {x, r.y + 1}, {x, r.y + r.height - 2}, {55, 55, 55}, 1);
  }
  const cv::Scalar axis_color[3] = {{80, 80, 255}, {80, 220, 80}, {255, 150, 60}};
  if (pts.size() >= 2) {
    // At most ~2 points per pixel column: plenty for a 5 s window and cheap at 833 Hz.
    const std::size_t step = std::max<std::size_t>(1, pts.size() / static_cast<std::size_t>(2 * r.width));
    for (int a = 0; a < 3; ++a) {
      std::vector<cv::Point> poly;
      poly.reserve(pts.size() / step + 1);
      for (std::size_t i = 0; i < pts.size(); i += step) {
        const ImuPoint& p = pts[i];
        const double age = static_cast<double>(now_ns - p.t_ns);
        if (age > static_cast<double>(span_ns)) continue;
        const double v = gyro ? p.gyro_dps[a] : p.acc[a];
        const int x = r.x + r.width - 1 - static_cast<int>(std::max(0.0, age) / static_cast<double>(span_ns) * (r.width - 1));
        const int y = static_cast<int>(mid - std::clamp(v / range, -1.0, 1.0) * half);
        poly.emplace_back(x, y);
      }
      if (poly.size() >= 2) cv::polylines(img, poly, false, axis_color[a], 1, cv::LINE_AA);
    }
  }
  const std::string label = fmt("%s  +/-%g %s   last 5 s   ", gyro ? "gyro" : "accel", range, unit);
  int base = 0;
  cv::Point p(r.x + 6, r.y + 16);
  cv::putText(img, label, p, kFont, 0.45, {220, 220, 220}, 1, cv::LINE_AA);
  p.x += cv::getTextSize(label, kFont, 0.45, 1, &base).width;
  const char* axis_name[3] = {"x", "y", "z"};
  for (int a = 0; a < 3; ++a) {
    cv::putText(img, axis_name[a], p, kFont, 0.5, axis_color[a], 2, cv::LINE_AA);
    p.x += 18;
  }
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------------

void apply_mock_overrides(rig::RigConfig& config) {
  for (auto& kv : config.cameras) kv.second.driver.backend = rig::CameraBackendKind::kMock;
  rig::ImuConfig& d = config.imu.driver;
  d.backend = rig::ImuBackendKind::kSim;
  d.sim.motion = true;
  if (d.sim.accel_noise_std_mps2 <= 0) d.sim.accel_noise_std_mps2 = 0.02;
  if (d.sim.gyro_noise_std_rps <= 0) d.sim.gyro_noise_std_rps = 0.002;
}

std::vector<std::string> display_order(const rig::RigConfig& config) {
  std::vector<std::string> ids;
  for (const char* role : {"main", "ultrawide"})
    for (const auto& kv : config.cameras)
      if (kv.second.role == role) ids.push_back(kv.first);
  for (const auto& kv : config.cameras)
    if (std::find(ids.begin(), ids.end(), kv.first) == ids.end()) ids.push_back(kv.first);
  return ids;
}

void RateMeter::add(TimeNs t_ns, uint64_t count) {
  if (!pts_.empty() && (t_ns < pts_.back().first || count < pts_.back().second)) pts_.clear();  // reset
  pts_.emplace_back(t_ns, count);
  // Keep exactly one point at or before t - window, so the rate covers (about) the full window.
  while (pts_.size() > 2 && pts_[1].first <= t_ns - window_ns_) pts_.pop_front();
}

double RateMeter::rate_hz() const {
  if (pts_.size() < 2) return 0.0;
  const auto& a = pts_.front();
  const auto& b = pts_.back();
  if (b.first <= a.first) return 0.0;
  return static_cast<double>(b.second - a.second) * 1e9 / static_cast<double>(b.first - a.first);
}

void ImuHistory::add(const rig::ImuSample& s) {
  latest_ = s;
  has_latest_ = true;
  if (!std::isnan(s.temp_c)) temp_c_ = s.temp_c;
  if (s.t_ns == rig::kInvalidTime) return;
  if (!pts_.empty() && s.t_ns <= pts_.back().t_ns) return;  // keep strictly increasing
  ImuPoint p;
  p.t_ns = s.t_ns;
  for (int i = 0; i < 3; ++i) {
    p.acc[i] = static_cast<float>(s.acc_mps2[i]);
    p.gyro_dps[i] = static_cast<float>(s.gyro_rps[i] * kRadToDeg);
  }
  pts_.push_back(p);
}

void ImuHistory::prune(TimeNs now_ns) {
  while (!pts_.empty() && pts_.front().t_ns < now_ns - span_ns_) pts_.pop_front();
}

// ------------------------------------------------------------------------------------------------------------------

cv::Rect fit_rect(cv::Size src, const cv::Rect& box) {
  if (src.width <= 0 || src.height <= 0 || box.width <= 0 || box.height <= 0) return {box.x, box.y, 0, 0};
  const double s = std::min(static_cast<double>(box.width) / src.width, static_cast<double>(box.height) / src.height);
  const int w = std::max(1, static_cast<int>(std::floor(src.width * s)));
  const int h = std::max(1, static_cast<int>(std::floor(src.height * s)));
  return {box.x + (box.width - w) / 2, box.y + (box.height - h) / 2, w, h};
}

Layout compute_layout(const RenderOptions& opt, const std::vector<cv::Size>& camera_sizes, bool with_imu) {
  Layout L;
  const int W = std::max(320, opt.width);
  const int n = static_cast<int>(camera_sizes.size());
  int row_h = 0;
  if (n > 0) {
    const int col_w = (W - opt.gap * (n - 1)) / n;
    // Row height: the tallest camera at column width, capped.
    for (const cv::Size& s : camera_sizes) {
      if (s.width <= 0 || s.height <= 0) continue;
      row_h = std::max(row_h, static_cast<int>(std::floor(static_cast<double>(s.height) * col_w / s.width)));
    }
    if (row_h <= 0) row_h = col_w * 3 / 4;
    row_h = std::min(row_h, std::max(120, opt.max_camera_height));
    for (int i = 0; i < n; ++i) {
      const cv::Rect tile(i * (col_w + opt.gap), 0, col_w, row_h);
      L.tiles.push_back(tile);
      L.images.push_back(fit_rect(camera_sizes[static_cast<std::size_t>(i)], tile));
    }
  }
  int h = row_h;
  if (with_imu) {
    const int y = n > 0 ? row_h + opt.gap : 0;
    L.imu = cv::Rect(0, y, W, std::max(120, opt.imu_panel_height));
    h = L.imu.y + L.imu.height;
  }
  L.size = cv::Size(W, std::max(h, 120));
  return L;
}

std::vector<TextLine> camera_overlay_lines(const CameraView& c, TimeNs now_ns) {
  std::vector<TextLine> v;
  v.push_back({fmt("%s (%s)  %dx%d@%g  [%s]", c.id.c_str(), c.role.c_str(), c.width, c.height, c.fps,
                   c.backend.c_str()),
               Severity::kOk});
  if (!c.started) return v;
  const uint64_t drops = c.has_stats ? c.stats.drops_total() : 0;
  const bool slow = c.fps > 0 && c.measured_fps < 0.9 * c.fps;
  v.push_back({fmt("fps %.1f (1 s)   frames %llu   drops %llu", c.measured_fps,
                   static_cast<unsigned long long>(c.frames_seen), static_cast<unsigned long long>(drops)),
               drops > 0 || slow ? Severity::kWarn : Severity::kOk});
  if (c.has_stats && drops > 0)
    v.push_back({fmt("  seq %llu  time %llu  queue %llu  error %llu",
                     static_cast<unsigned long long>(c.stats.drops_seq_gap),
                     static_cast<unsigned long long>(c.stats.drops_time_gap),
                     static_cast<unsigned long long>(c.stats.drops_backend_queue),
                     static_cast<unsigned long long>(c.stats.drops_error)),
                 Severity::kWarn});
  if (!c.has_frame) return v;
  const rig::FrameMeta& m = c.meta;
  std::string exp = m.exposure_ns >= 0 ? fmt("exp %.2f ms", m.exposure_ns / 1e6) : std::string("exp ?");
  std::string gain = !std::isnan(m.analog_gain) ? fmt("gain %.2f", m.analog_gain) : std::string("gain ?");
  std::string dg = !std::isnan(m.digital_gain) ? fmt("  dgain %.2f", m.digital_gain) : std::string();
  v.push_back({exp + "   " + gain + dg, Severity::kOk});
  if (c.role == "main" || (m.focus_status != rig::FocusStatus::kNotApplicable &&
                           m.focus_status != rig::FocusStatus::kUnknown)) {
    const bool bad = m.focus_status == rig::FocusStatus::kFailed;
    v.push_back({fmt("focus %s%s", rig::to_string(m.focus_status),
                     m.focus_position >= 0 ? fmt(" @%d", m.focus_position).c_str() : ""),
                 bad ? Severity::kBad : Severity::kOk});
  }
  const double age_ms = c.t_mid_ns != rig::kInvalidTime ? static_cast<double>(now_ns - c.t_mid_ns) / 1e6 : rig::kNaN;
  const Severity age_sev = std::isnan(age_ms) || age_ms > 250 || age_ms < -5 ? Severity::kWarn : Severity::kOk;
  v.push_back({fmt("range %s   age %s", rig::to_string(m.color_range),
                   std::isnan(age_ms) ? "?" : fmt("%.0f ms", age_ms).c_str()),
               m.color_range == rig::ColorRange::kUnknown ? Severity::kWarn : age_sev});
  if (!c.convert_error.empty()) v.push_back({"convert: " + c.convert_error, Severity::kBad});
  return v;
}

std::string stall_text(const CameraView& c, TimeNs now_ns, double stall_s) {
  if (!c.started) return "";
  const TimeNs ref = c.last_rx_ns != rig::kInvalidTime ? c.last_rx_ns : c.started_ns;
  if (ref == rig::kInvalidTime) return "";
  const double dt = static_cast<double>(now_ns - ref) / 1e9;
  if (dt <= stall_s) return "";
  return fmt("NO FRAMES for %.1f s", dt);
}

std::vector<TextLine> imu_panel_lines(const ImuView& u, const ImuHistory& h) {
  std::vector<TextLine> v;
  v.push_back({fmt("IMU %s  [%s]  ODR %g Hz", u.id.c_str(), u.backend.c_str(), u.odr_hz), Severity::kOk});
  if (!u.started) {
    v.push_back({"FAILED TO START", Severity::kBad});
    for (const auto& s : wrap_text(u.error, 70)) v.push_back({s, Severity::kBad});
    return v;
  }
  if (!h.has_sample()) {
    v.push_back({"waiting for samples...", Severity::kWarn});
  } else {
    const rig::ImuSample& s = h.latest();
    const double an = s.acc_mps2.norm();
    v.push_back({fmt("acc  x %+7.2f  y %+7.2f  z %+7.2f   |a| %6.3f m/s2", s.acc_mps2.x(), s.acc_mps2.y(),
                     s.acc_mps2.z(), an),
                 Severity::kOk});
    v.push_back({fmt("gyro x %+7.1f  y %+7.1f  z %+7.1f   deg/s", s.gyro_rps.x() * kRadToDeg,
                     s.gyro_rps.y() * kRadToDeg, s.gyro_rps.z() * kRadToDeg),
                 Severity::kOk});
  }
  const float t = h.temperature_c();
  const bool slow = u.odr_hz > 0 && u.measured_rate_hz < 0.95 * u.odr_hz;
  v.push_back({fmt("temp %s   rate %.1f Hz (1 s)", std::isnan(t) ? "?" : fmt("%.1f C", t).c_str(), u.measured_rate_hz),
               slow ? Severity::kWarn : Severity::kOk});
  if (u.has_stats) {
    const rig::ImuStats& st = u.stats;
    const bool ovr = st.fifo_overruns > 0 || st.queue_overruns > 0;
    v.push_back({fmt("FIFO max %d/%d words   overruns fifo %llu  queue %llu   gaps %llu", st.fifo_level_max,
                     st.fifo_watermark_words, static_cast<unsigned long long>(st.fifo_overruns),
                     static_cast<unsigned long long>(st.queue_overruns), static_cast<unsigned long long>(st.gaps)),
                 ovr ? Severity::kBad : (st.gaps > 0 ? Severity::kWarn : Severity::kOk)});
    const rig::SummaryStats& r = st.clock.residual_ns;
    const double rmax = r.count > 0 ? std::max(std::abs(r.min), std::abs(r.max)) / 1e3 : rig::kNaN;
    const double rp99 = r.count > 0 ? std::abs(r.p99) / 1e3 : rig::kNaN;
    v.push_back({fmt("I2C errors %llu (retries %llu)   clock fit resid p99 %s  max %s",
                     static_cast<unsigned long long>(st.i2c_errors), static_cast<unsigned long long>(st.i2c_retries),
                     std::isnan(rp99) ? "?" : fmt("%.0f us", rp99).c_str(),
                     std::isnan(rmax) ? "?" : fmt("%.0f us", rmax).c_str()),
                 st.i2c_errors > 0 || (!std::isnan(rp99) && rp99 > 100) ? Severity::kWarn : Severity::kOk});
  }
  if (!u.error.empty()) {
    for (const auto& s : wrap_text("ERROR: " + u.error, 70)) v.push_back({s, Severity::kBad});
  }
  return v;
}

std::string status_line(const std::vector<CameraView>& cams, const ImuView& u, const ImuHistory& h, TimeNs now_ns,
                        TimeNs t0_ns, double stall_s) {
  std::string s = fmt("[%5.0fs]", static_cast<double>(now_ns - t0_ns) / 1e9);
  for (const auto& c : cams) {
    s += " | " + c.id + " ";
    if (!c.started) {
      s += "FAILED";
      continue;
    }
    if (!c.error.empty()) {
      s += "ERROR ";
    }
    const std::string st = stall_text(c, now_ns, stall_s);
    if (!st.empty()) {
      s += st;
      continue;
    }
    const double age_ms = c.has_frame && c.t_mid_ns != rig::kInvalidTime
                              ? static_cast<double>(now_ns - c.t_mid_ns) / 1e6
                              : rig::kNaN;
    s += fmt("%.1ffps drops %llu age %s", c.measured_fps,
             static_cast<unsigned long long>(c.has_stats ? c.stats.drops_total() : 0),
             std::isnan(age_ms) ? "?" : fmt("%.0fms", age_ms).c_str());
  }
  if (u.enabled) {
    s += " | " + u.id + " ";
    if (!u.started) {
      s += "FAILED";
    } else {
      if (!u.error.empty()) s += "ERROR ";
      s += fmt("%.1fHz", u.measured_rate_hz);
      if (h.has_sample()) s += fmt(" |a| %.2f", h.latest().acc_mps2.norm());
      if (u.has_stats)
        s += fmt(" ovr %llu i2c %llu", static_cast<unsigned long long>(u.stats.fifo_overruns + u.stats.queue_overruns),
                 static_cast<unsigned long long>(u.stats.i2c_errors));
    }
  }
  return s;
}

double plot_range(double max_abs, double min_range) {
  const double want = std::max(std::isnan(max_abs) ? 0.0 : max_abs * 1.1, min_range);
  if (!(want > 0)) return 1.0;
  const double e = std::pow(10.0, std::floor(std::log10(want)));
  for (double m : {1.0, 2.0, 5.0, 10.0})
    if (m * e >= want * (1 - 1e-12)) return m * e;
  return 10.0 * e;
}

const char* camera_backend_name(rig::CameraBackendKind k) {
  switch (k) {
    case rig::CameraBackendKind::kArgus: return "argus";
    case rig::CameraBackendKind::kGst: return "gst";
    case rig::CameraBackendKind::kMock: return "mock";
    case rig::CameraBackendKind::kReplay: return "replay";
  }
  return "?";
}

const char* imu_backend_name(rig::ImuBackendKind k) {
  switch (k) {
    case rig::ImuBackendKind::kLsm6dsoxI2c: return "lsm6dsox_i2c";
    case rig::ImuBackendKind::kSim: return "sim";
    case rig::ImuBackendKind::kReplay: return "replay";
  }
  return "?";
}

std::string failure_hint(bool is_camera, const std::string& backend, const rig::Status& e) {
  const std::string& m = e.message();
  auto has = [&m](const char* s) { return m.find(s) != std::string::npos; };
  if (e.code() == rig::StatusCode::kUnimplemented)
    return backend == "argus" ? "this build has no Argus backend: rebuild with -DRIG_WITH_ARGUS=ON (laptop: use --mock)"
                              : "backend '" + backend + "' is not compiled into this build";
  if (is_camera) {
    if (backend == "argus" || backend == "gst") {
      if (has("CameraProvider") || has("daemon"))
        return "is nvargus-daemon running? sudo systemctl restart nvargus-daemon; check dmesg for the sensor";
      if (has("sensor") || has("device"))
        return "is the camera detected? ls /dev/video*; sudo dmesg | grep -i -E 'imx519|imx219'; check sensor_id";
      if (e.code() == rig::StatusCode::kNotFound)
        return "mode not offered by the driver: ec-cam-probe --config <yaml> --list";
      return "see ec-cam-probe --config <yaml> --camera <id> for details";
    }
    return "";
  }
  if (backend == "lsm6dsox_i2c") {
    if (has("Permission") || has("EACCES") || has("permission"))
      return "no access to /dev/i2c-N: run with sudo or add the user to the i2c group";
    if (e.code() == rig::StatusCode::kNotFound || has("No such") || has("no such") || has("ENOENT"))
      return "I2C bus missing: ls /dev/i2c-*; check imu.driver.i2c_bus";
    if (has("WHO_AM_I") || e.code() == rig::StatusCode::kDeviceError)
      return "chip not answering as expected: sudo i2cdetect -y -r 7 (expect 6a); INT1 must be unconnected";
    return "run ec-imu-probe --config <yaml> for the register-level checks";
  }
  return "";
}

std::vector<std::string> wrap_text(const std::string& text, std::size_t max_chars) {
  std::vector<std::string> out;
  if (max_chars == 0) max_chars = 1;
  std::string rest = text;
  while (rest.size() > max_chars) {
    std::size_t cut = rest.rfind(' ', max_chars);
    if (cut == std::string::npos || cut == 0) cut = max_chars;
    out.push_back(rest.substr(0, cut));
    rest = rest.substr(cut);
    while (!rest.empty() && rest.front() == ' ') rest.erase(rest.begin());
  }
  if (!rest.empty() || out.empty()) out.push_back(rest);
  return out;
}

cv::Mat render_mosaic(const Layout& L, const std::vector<CameraView>& cams, const ImuView& u, const ImuHistory& h,
                      const RenderOptions& opt, TimeNs now_ns) {
  cv::Mat img(L.size, CV_8UC3, cv::Scalar(24, 24, 24));
  for (std::size_t i = 0; i < cams.size() && i < L.tiles.size(); ++i) {
    const CameraView& c = cams[i];
    const cv::Rect tile = L.tiles[i] & cv::Rect(0, 0, img.cols, img.rows);
    const cv::Rect ir = L.images[i] & tile;
    img(tile).setTo(cv::Scalar(0, 0, 0));
    if (c.has_frame && !c.image.empty() && ir.area() > 0) {
      cv::Mat dst = img(ir);
      if (c.image.size() == ir.size() && c.image.type() == CV_8UC3) {
        c.image.copyTo(dst);
      } else {
        cv::Mat tmp;
        cv::resize(c.image, tmp, ir.size(), 0, 0, cv::INTER_AREA);
        if (tmp.type() == CV_8UC3) tmp.copyTo(dst);
      }
    }
    draw_text_box(img, camera_overlay_lines(c, now_ns), {tile.x + 6, tile.y + 6}, 0.52, 21, tile.width - 12);
    const int max_chars = std::max(20, tile.width / 13);
    if (!c.started) {
      std::vector<std::string> lines{c.id + ": FAILED TO START"};
      for (auto& s : wrap_text(c.error, static_cast<std::size_t>(max_chars))) lines.push_back(s);
      draw_centered(img, tile, {lines.front()}, 0.9, {60, 60, 255}, 2);
      std::vector<TextLine> tl;
      for (std::size_t k = 1; k < lines.size(); ++k) tl.push_back({lines[k], Severity::kBad});
      draw_text_box(img, tl, {tile.x + 6, tile.y + tile.height / 2 + 30}, 0.5, 20, tile.width - 12);
    } else {
      const std::string st = stall_text(c, now_ns, opt.stall_s);
      if (!st.empty()) draw_centered(img, tile, {st}, 1.1, {60, 60, 255}, 2);
      if (!c.error.empty()) {
        std::vector<TextLine> tl;
        for (auto& s : wrap_text("ERROR: " + c.error, static_cast<std::size_t>(max_chars))) tl.push_back({s, Severity::kBad});
        draw_text_box(img, tl, {tile.x + 6, tile.y + tile.height - 8 - 20 * static_cast<int>(tl.size()) - 8}, 0.5, 20,
                      tile.width - 12);
      }
    }
  }
  if (L.imu.area() > 0) {
    const cv::Rect r = L.imu & cv::Rect(0, 0, img.cols, img.rows);
    img(r).setTo(cv::Scalar(36, 30, 30));
    cv::rectangle(img, r, {80, 80, 80}, 1);
    const int text_w = std::min(600, r.width / 2);
    int y = r.y + 4;
    const std::vector<TextLine> lines = imu_panel_lines(u, h);
    for (const auto& l : lines) {
      y += 21;
      cv::putText(img, l.text, {r.x + 10, y}, kFont, 0.5, color_of(l.severity), 1, cv::LINE_AA);
    }
    if (u.started) {
      const int px = r.x + text_w + 10;
      const int pw = r.x + r.width - 8 - px;
      const int ph = (r.height - 3 * 6) / 2;
      draw_plot(img, {px, r.y + 6, pw, ph}, h.points(), false, now_ns, h.span_ns(), 12.0, "m/s2");
      draw_plot(img, {px, r.y + 12 + ph, pw, ph}, h.points(), true, now_ns, h.span_ns(), 20.0, "deg/s");
    }
  }
  return img;
}

rig::Result<cv::Mat> frame_to_display(const rig::ImageFrame& f, cv::Size target) {
  if (!f.valid()) return rig::invalid_argument("frame_to_display: invalid frame");
  const bool halve = f.fmt == rig::PixelFormat::kNV12 && target.width > 0 && target.width * 2 <= f.width &&
                     f.width % 4 == 0 && f.height % 4 == 0 && f.width >= 16 && f.height >= 16;
  rig::Result<cv::Mat> bgr = cv::Mat();
  if (halve) {
    const int W = f.width, H = f.height, W2 = W / 2, H2 = H / 2;
    auto* d = const_cast<uint8_t*>(f.data.get());  // cv::Mat wants non-const; read-only use
    const cv::Mat y(H, W, CV_8UC1, d + f.planes[0].offset, f.planes[0].stride);
    const cv::Mat uv(H / 2, W / 2, CV_8UC2, d + f.planes[1].offset, f.planes[1].stride);
    const std::size_t ny = static_cast<std::size_t>(W2) * H2, nuv = static_cast<std::size_t>(W2) * (H2 / 2);
    std::shared_ptr<uint8_t[]> buf(new uint8_t[ny + nuv]);
    cv::Mat y2(H2, W2, CV_8UC1, buf.get(), static_cast<std::size_t>(W2));
    cv::Mat uv2(H2 / 2, W2 / 2, CV_8UC2, buf.get() + ny, static_cast<std::size_t>(W2));
    cv::resize(y, y2, y2.size(), 0, 0, cv::INTER_AREA);
    cv::resize(uv, uv2, uv2.size(), 0, 0, cv::INTER_AREA);
    rig::ImageFrame g = f;  // timestamps + meta (color_range) kept
    g.width = W2;
    g.height = H2;
    g.planes[0] = {0, static_cast<std::size_t>(W2), W2, H2};
    g.planes[1] = {ny, static_cast<std::size_t>(W2), W2 / 2, H2 / 2};
    g.data = std::shared_ptr<const uint8_t[]>(buf);
    g.size = ny + nuv;
    bgr = rig::frame_to_bgr(g);
  } else {
    bgr = rig::frame_to_bgr(f);
  }
  if (!bgr.ok()) return bgr.status();
  if (target.width <= 0 || target.height <= 0 || bgr->size() == target) return bgr;
  cv::Mat out;
  cv::resize(*bgr, out, target, 0, 0, cv::INTER_AREA);
  return out;
}

}  // namespace eclive
