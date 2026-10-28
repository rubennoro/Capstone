// SPDX-License-Identifier: MIT
// ReplayCamera: frames from a package raw/cam_<id>/ directory or any EuRoC-style camera folder (PLAN T2).
//
// Input layout (PIPELINE 6.1; EuRoC MAV dataset format):
// - <path>/data.csv (layout::kCamData): optional '#' header line(s), then rows "<t_ns>,<filename>" (t_ns = image time
//   = t_mid in our packages, the EuRoC timestamp in EuRoC folders). Rows must be strictly increasing in time.
// - images: <path>/<filename>, else <path>/<layout::kFramesDir>/<filename> (our packages), else <path>/data/<filename>
//   (EuRoC).
//   Any OpenCV-readable 8-bit gray/BGR/BGRA image (jpg, png).
// - optional <path>/frame_meta.csv: a header row naming the columns (leading '#' allowed), joined to data.csv by
//   the time column `t_ns` (or `t_mid_ns`). Recognized columns, all optional: t_sof_ns, t_arrival_ns, seq,
//   exposure_ns, analog_gain, digital_gain, readout_ns, sensor_mode, focus_position, focus_status (FocusStatus
//   strings). Unknown columns are ignored. [Contract proposal for T5's RawWriter frame_meta.csv: T2.WORKLOG.]
// Replay semantics:
// - t_mid_ns = the data.csv time; t_sof_ns = frame_meta t_sof_ns if present, else t_mid - exposure/2 - readout/2 if
//   both are known, else t_mid. seq = frame_meta seq if present (recorded drops replay as seq gaps), else the row
//   index. Loops continue seq and time monotonically.
// - retime: timestamps are shifted so the first frame is at now_raw_ns() at start(); otherwise the originals.
// - realtime: frame i is delivered at start + (t_i - t_0) (+ loop offsets); else immediately on read().
// - Images are delivered at their stored size (packages store main at storage.width x height, not the capture size;
//   a different configured width x height is logged once and ignored).
// - End of data (loop off): read() returns std::nullopt and status() becomes kOutOfRange "end of replay".
// - A missing/corrupt image or a malformed CSV is fatal (kDataLoss / kInvalidArgument).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "backends.hpp"
#include "camera_internal.hpp"
#include "rig/drivers/camera/sensor_modes.hpp"
#include "rig/package/layout.hpp"
#include "rig/time/clock.hpp"

namespace rig {
namespace cam {
namespace {

namespace fs = std::filesystem;

/// EuRoC MAV camera folder: mav0/cam0/data.csv + mav0/cam0/data/<t>.png (EuRoC dataset format).
constexpr const char* kEurocDataDir = "data";

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::string cell;
  std::istringstream is(line);
  while (std::getline(is, cell, ',')) out.push_back(trim(cell));
  if (!line.empty() && line.back() == ',') out.emplace_back();
  return out;
}

bool parse_i64(const std::string& s, int64_t* v) {
  if (s.empty()) return false;
  try {
    std::size_t pos = 0;
    const long long x = std::stoll(s, &pos);
    if (pos != s.size()) return false;
    *v = x;
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_f64(const std::string& s, double* v) {
  if (s.empty()) return false;
  try {
    std::size_t pos = 0;
    const double x = std::stod(s, &pos);
    if (pos != s.size()) return false;
    *v = x;
    return true;
  } catch (...) {
    return false;
  }
}

FocusStatus parse_focus_status(const std::string& s) {
  for (auto st : {FocusStatus::kUnknown, FocusStatus::kNotApplicable, FocusStatus::kLockedVcmI2c,
                  FocusStatus::kLockedV4l2, FocusStatus::kFixedAtPowerUp, FocusStatus::kFailed})
    if (s == to_string(st)) return st;
  return FocusStatus::kUnknown;
}

struct Row {
  TimeNs t_mid = kInvalidTime;
  std::string file;
  TimeNs t_sof = kInvalidTime;
  int64_t seq = -1;
  FrameMeta meta;
};

class ReplayCamera final : public CameraBase {
 public:
  explicit ReplayCamera(const CameraConfig& c) : CameraBase(c) {}
  ~ReplayCamera() override { (void)ReplayCamera::stop(); }

  Result<std::vector<SensorMode>> modes() const override {
    if (is_known_sensor(config_.sensor)) return known_sensor_modes(config_.sensor);
    SensorMode m;
    m.index = config_.sensor_mode;
    m.width = config_.width;
    m.height = config_.height;
    m.min_fps = 0.0;
    m.max_fps = config_.fps;
    m.description = "replay of " + config_.replay.path;
    return std::vector<SensorMode>{m};
  }

  Status start() override {
    if (running_.load()) return failed_precondition("camera " + config_.id + ": already started");
    const std::string ctx = "cameras." + config_.id + ".driver.replay.path";
    if (config_.pixel_format == PixelFormat::kJPEG || config_.pixel_format == PixelFormat::kUnknown)
      return invalid_argument("cameras." + config_.id + ".driver.pixel_format: unsupported for replay");
    RIG_RETURN_IF_ERROR(load_index(ctx));
    // Nominal period: from the data (median interval) if possible, else the configured fps.
    std::vector<int64_t> dts;
    for (std::size_t i = 1; i < rows_.size(); ++i) dts.push_back(rows_[i].t_mid - rows_[i - 1].t_mid);
    if (!dts.empty()) {
      std::nth_element(dts.begin(), dts.begin() + static_cast<std::ptrdiff_t>(dts.size() / 2), dts.end());
      period_ns_ = dts[dts.size() / 2];
    } else {
      period_ns_ = config_.fps > 0 ? static_cast<int64_t>(1e9 / config_.fps) : 33'333'333;
    }
    const double fps = config_.fps > 0 ? config_.fps : 1e9 / static_cast<double>(period_ns_);
    max_seq_ = 0;
    for (const Row& r : rows_) max_seq_ = std::max<int64_t>(max_seq_, r.seq);
    span_ns_ = rows_.back().t_mid - rows_.front().t_mid;
    begin_stream(fps, config_.sensor_mode, "replay");
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = false;
    }
    idx_ = 0;
    loop_ = 0;
    pending_.reset();
    warned_size_ = false;
    start_raw_ = now_raw_ns();
    offset_ns_ = config_.replay.retime ? start_raw_ - rows_.front().t_mid : 0;
    running_.store(true);
    spdlog::info("camera {}: replay of {} frames from {} (realtime={}, loop={}, retime={})", config_.id, rows_.size(),
                 config_.replay.path, config_.replay.realtime, config_.replay.loop, config_.replay.retime);
    return Status::Ok();
  }

  Status stop() override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    running_.store(false);
    return Status::Ok();
  }

  std::optional<ImageFrame> read(std::chrono::nanoseconds timeout) override {
    if (!running_.load() || fatal()) return std::nullopt;
    const TimeNs deadline = now_raw_ns() + timeout.count();
    for (;;) {
      if (idx_ >= rows_.size()) {
        if (!config_.replay.loop) {
          set_fatal(out_of_range("camera " + config_.id + ": end of replay (" + std::to_string(rows_.size()) +
                                 " frames from " + config_.replay.path + ")"));
          return std::nullopt;
        }
        idx_ = 0;
        ++loop_;
      }
      const Row& r = rows_[idx_];
      const TimeNs loop_shift = static_cast<TimeNs>(loop_) * (span_ns_ + period_ns_);
      if (!pending_) {  // decode before pacing so the decode time does not delay delivery
        Result<ImageFrame> fr = load_frame(r);
        if (!fr.ok()) {
          set_fatal(fr.status());
          return std::nullopt;
        }
        pending_ = std::move(fr).value();
      }
      if (config_.replay.realtime) {
        const TimeNs due = start_raw_ + (r.t_mid - rows_.front().t_mid) + loop_shift;
        if (due > deadline) {
          sleep_until(deadline);
          return std::nullopt;
        }
        if (!sleep_until(due)) return std::nullopt;
      } else {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_) return std::nullopt;
      }
      ImageFrame f = std::move(*pending_);
      pending_.reset();
      f.t_mid_ns = r.t_mid + offset_ns_ + loop_shift;
      f.t_sof_ns = r.t_sof + offset_ns_ + loop_shift;
      f.t_arrival_ns = now_raw_ns();
      const uint64_t seq = static_cast<uint64_t>(loop_) * static_cast<uint64_t>(max_seq_ + 1) + static_cast<uint64_t>(r.seq);
      ++idx_;
      if (!track(f, seq)) continue;
      // Replay keeps the recorded focus metadata (track() stamps the backend's own, which is unknown here).
      f.meta.focus_status = r.meta.focus_status;
      f.meta.focus_position = r.meta.focus_position;
      deliver(f);
      return f;
    }
  }

 private:
  Status load_index(const std::string& ctx) {
    rows_.clear();
    dir_ = config_.replay.path;
    if (dir_.empty()) return invalid_argument(ctx + ": empty");
    const fs::path csv = fs::path(dir_) / layout::kCamData;
    std::ifstream in(csv);
    if (!in) return not_found(ctx + ": cannot open " + csv.string());
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
      ++lineno;
      const std::string t = trim(line);
      if (t.empty() || t[0] == '#') continue;
      const auto cells = split_csv(t);
      Row r;
      if (cells.size() < 2 || !parse_i64(cells[0], &r.t_mid) || cells[1].empty()) {
        if (rows_.empty() && lineno == 1) continue;  // tolerate a header without '#'
        return invalid_argument(csv.string() + ":" + std::to_string(lineno) + ": expected '<t_ns>,<filename>', got '" +
                                t + "'");
      }
      if (!rows_.empty() && r.t_mid <= rows_.back().t_mid)
        return invalid_argument(csv.string() + ":" + std::to_string(lineno) + ": timestamps not strictly increasing");
      r.file = cells[1];
      r.seq = static_cast<int64_t>(rows_.size());
      rows_.push_back(std::move(r));
    }
    if (rows_.empty()) return invalid_argument(ctx + ": " + csv.string() + " has no frames");
    RIG_RETURN_IF_ERROR(load_meta());
    for (Row& r : rows_) {
      if (r.t_sof == kInvalidTime) {
        r.t_sof = (r.meta.exposure_ns >= 0 && r.meta.readout_ns >= 0)
                      ? r.t_mid - r.meta.exposure_ns / 2 - r.meta.readout_ns / 2
                      : r.t_mid;
      }
    }
    for (std::size_t i = 1; i < rows_.size(); ++i) {
      if (rows_[i].seq <= rows_[i - 1].seq)
        return invalid_argument(dir_ + "/frame_meta.csv: seq not strictly increasing at t_ns " +
                                std::to_string(rows_[i].t_mid));
    }
    return Status::Ok();
  }

  Status load_meta() {
    const fs::path p = fs::path(dir_) / layout::kFrameMeta;
    std::ifstream in(p);
    if (!in) return Status::Ok();  // optional
    std::string line;
    std::map<std::string, std::size_t> col;
    int lineno = 0;
    std::map<TimeNs, std::size_t> by_t;
    for (std::size_t i = 0; i < rows_.size(); ++i) by_t[rows_[i].t_mid] = i;
    while (std::getline(in, line)) {
      ++lineno;
      std::string t = trim(line);
      if (t.empty()) continue;
      if (col.empty()) {
        if (t[0] == '#') t = trim(t.substr(1));
        const auto names = split_csv(t);
        for (std::size_t i = 0; i < names.size(); ++i) col[names[i]] = i;
        if (!col.count("t_ns") && !col.count("t_mid_ns"))
          return invalid_argument(p.string() + ":1: header needs a t_ns (or t_mid_ns) column");
        continue;
      }
      if (t[0] == '#') continue;
      const auto cells = split_csv(t);
      auto get = [&](const char* name) -> const std::string* {
        auto it = col.find(name);
        if (it == col.end() || it->second >= cells.size()) return nullptr;
        return &cells[it->second];
      };
      const std::string* tk = get("t_ns");
      if (tk == nullptr) tk = get("t_mid_ns");
      int64_t tm = 0;
      if (tk == nullptr || !parse_i64(*tk, &tm))
        return invalid_argument(p.string() + ":" + std::to_string(lineno) + ": bad time column");
      auto it = by_t.find(tm);
      if (it == by_t.end()) continue;  // meta for a frame not in data.csv: ignore
      Row& r = rows_[it->second];
      int64_t i64 = 0;
      double f64 = 0.0;
      if (const auto* s = get("t_sof_ns"); s && parse_i64(*s, &i64)) r.t_sof = i64;
      if (const auto* s = get("seq"); s && parse_i64(*s, &i64)) r.seq = i64;
      if (const auto* s = get("exposure_ns"); s && parse_i64(*s, &i64)) r.meta.exposure_ns = i64;
      if (const auto* s = get("readout_ns"); s && parse_i64(*s, &i64)) r.meta.readout_ns = i64;
      if (const auto* s = get("sensor_mode"); s && parse_i64(*s, &i64)) r.meta.sensor_mode = static_cast<int>(i64);
      if (const auto* s = get("focus_position"); s && parse_i64(*s, &i64)) r.meta.focus_position = static_cast<int>(i64);
      if (const auto* s = get("analog_gain"); s && parse_f64(*s, &f64)) r.meta.analog_gain = f64;
      if (const auto* s = get("digital_gain"); s && parse_f64(*s, &f64)) r.meta.digital_gain = f64;
      if (const auto* s = get("focus_status")) r.meta.focus_status = parse_focus_status(*s);
    }
    return Status::Ok();
  }

  Result<ImageFrame> load_frame(const Row& r) {
    // <dir>/<file>, then the package frame dir (rig/package/layout.hpp kFramesDir: T3 owns the name, R1 F14), then
    // the EuRoC "data/" folder (identical once kFramesDir is "data").
    fs::path img = fs::path(dir_) / r.file;
    std::error_code ec;
    if (!fs::exists(img, ec)) img = fs::path(dir_) / layout::kFramesDir / r.file;
    if (!fs::exists(img, ec)) img = fs::path(dir_) / kEurocDataDir / r.file;
    if (!fs::exists(img, ec))
      return data_loss("camera " + config_.id + ": replay image '" + r.file + "' not found in " + dir_ +
                       " (also tried " + layout::kFramesDir + "/ and " + kEurocDataDir + "/)");
    cv::Mat m;
    try {
      m = cv::imread(img.string(), cv::IMREAD_UNCHANGED);
    } catch (const cv::Exception& e) {
      return data_loss(img.string() + ": " + e.what());
    }
    if (m.empty()) return data_loss(img.string() + ": cannot decode (corrupt or unsupported)");
    if (m.depth() != CV_8U) return data_loss(img.string() + ": only 8-bit images are supported");
    if (m.channels() == 4) cv::cvtColor(m, m, cv::COLOR_BGRA2BGR);
    if (m.channels() != 1 && m.channels() != 3) return data_loss(img.string() + ": unsupported channel count");
    if (!warned_size_ && config_.width > 0 && config_.height > 0 && (m.cols != config_.width || m.rows != config_.height)) {
      spdlog::warn("camera {}: replay images are {}x{}, config says {}x{}: delivering the stored size", config_.id,
                   m.cols, m.rows, config_.width, config_.height);
      warned_size_ = true;
    }
    ImageFrame f;
    uint8_t* dst = allocate_frame(pool_, config_.pixel_format, m.cols, m.rows, f);
    if (dst == nullptr)
      return data_loss(img.string() + ": " + std::to_string(m.cols) + "x" + std::to_string(m.rows) +
                       " cannot be delivered as " + to_string(config_.pixel_format) + " (NV12 needs even sizes)");
    convert_into_frame(m.data, m.channels(), m.step, f, dst);
    f.meta = r.meta;
    f.meta.color_range = ColorRange::kFull;  // re-encoded here as JFIF full range (convert_into_frame)
    return f;
  }

  bool sleep_until(TimeNs t) {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      if (stop_) return false;
      const TimeNs now = now_raw_ns();
      if (now >= t) return true;
      cv_.wait_for(lk, std::chrono::nanoseconds(t - now));
    }
  }

  BufferPool pool_;
  std::string dir_;
  std::vector<Row> rows_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::atomic<bool> running_{false};
  std::size_t idx_ = 0;
  uint64_t loop_ = 0;
  int64_t max_seq_ = 0;
  int64_t period_ns_ = 0;
  int64_t span_ns_ = 0;
  TimeNs start_raw_ = 0;
  TimeNs offset_ns_ = 0;
  bool warned_size_ = false;
  std::optional<ImageFrame> pending_;  ///< Decoded rows_[idx_] waiting for its delivery time.
};

}  // namespace

std::unique_ptr<CameraBackend> make_replay_camera(const CameraConfig& config) {
  return std::make_unique<ReplayCamera>(config);
}

}  // namespace cam
}  // namespace rig
