// SPDX-License-Identifier: MIT
// ImuReplay: replays an EuRoC-style imu0/data.csv (package raw/imu0/data.csv has the same columns, PIPELINE 6.1):
//   #timestamp [ns], w_x, w_y, w_z [rad/s], a_x, a_y, a_z [m/s^2]
// chip_ticks = -1 (chip.csv is not needed for replay). Optional real-time pacing and retiming to now_raw_ns().
#include "rig/drivers/imu/imu_replay.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#include "rig/time/clock.hpp"

namespace rig {

namespace {

class ImuReplay final : public ImuBackend {
 public:
  ImuReplay(ImuConfig c, std::vector<ImuSample> s) : cfg_(std::move(c)), all_(std::move(s)) {}
  ~ImuReplay() override { (void)stop(); }

  const ImuConfig& config() const override { return cfg_; }

  Status start() override {
    std::lock_guard<std::mutex> lk(mu_);
    if (running_) return failed_precondition("replay " + cfg_.replay.path + ": already running");
    running_ = true;
    next_ = 0;
    delivered_ = 0;
    last_t_ = kInvalidTime;
    t0_wall_ = now_raw_ns();
    shift_ = (cfg_.replay.retime && !all_.empty()) ? t0_wall_ - all_.front().t_ns : 0;
    return Status::Ok();
  }

  Status stop() override {
    std::lock_guard<std::mutex> lk(mu_);
    running_ = false;
    cv_.notify_all();
    return Status::Ok();
  }

  std::size_t read(std::vector<ImuSample>& out, std::chrono::nanoseconds timeout) override {
    std::unique_lock<std::mutex> lk(mu_);
    const TimeNs deadline = now_raw_ns() + timeout.count();
    for (;;) {
      if (!running_ || next_ >= all_.size()) return 0;
      std::size_t end = next_;
      if (!cfg_.replay.realtime) {
        end = std::min(all_.size(), next_ + 1000);
      } else {
        // Map sample time to wall time through the same shift as retime (or the first-sample anchor).
        const TimeNs base = cfg_.replay.retime ? shift_ : t0_wall_ - all_.front().t_ns;
        const TimeNs now = now_raw_ns();
        while (end < all_.size() && all_[end].t_ns + base <= now) ++end;
      }
      if (end > next_) {
        for (std::size_t i = next_; i < end; ++i) {
          ImuSample s = all_[i];
          s.t_ns += shift_;
          out.push_back(s);
          last_t_ = s.t_ns;
        }
        const std::size_t n = end - next_;
        next_ = end;
        delivered_ += n;
        return n;
      }
      const TimeNs now = now_raw_ns();
      if (now >= deadline) return 0;
      cv_.wait_for(lk, std::chrono::nanoseconds(std::min<int64_t>(deadline - now, 2000000)));
    }
  }

  std::size_t read_brackets(std::vector<ClockBracket>& /*out*/) override { return 0; }

  ImuStats stats() const override {
    std::lock_guard<std::mutex> lk(mu_);
    ImuStats s;
    s.samples = delivered_;
    s.last_t_ns = last_t_;
    if (all_.size() >= 2)
      s.rate_hz_estimate = static_cast<double>(all_.size() - 1) * 1e9 /
                           static_cast<double>(all_.back().t_ns - all_.front().t_ns);
    return s;
  }
  Status status() const override { return Status::Ok(); }

 private:
  const ImuConfig cfg_;
  const std::vector<ImuSample> all_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool running_ = false;
  std::size_t next_ = 0;
  uint64_t delivered_ = 0;
  TimeNs last_t_ = kInvalidTime;
  TimeNs t0_wall_ = 0;
  TimeNs shift_ = 0;
};

}  // namespace

Result<std::vector<ImuSample>> load_euroc_imu_csv(const std::string& path) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path p(path);
  if (fs::is_directory(p, ec)) p /= "data.csv";
  std::ifstream in(p);
  if (!in) return not_found("imu replay: cannot open " + p.string());
  std::vector<ImuSample> v;
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream ss(line);
    long long t = 0;
    double w[3], a[3];
    if (!(ss >> t >> w[0] >> w[1] >> w[2] >> a[0] >> a[1] >> a[2]))
      return invalid_argument("imu replay: " + p.string() + ":" + std::to_string(lineno) +
                              ": expected 7 columns (t_ns, w_xyz, a_xyz)");
    ImuSample s;
    s.t_ns = static_cast<TimeNs>(t);
    s.gyro_rps = Eigen::Vector3d(w[0], w[1], w[2]);
    s.acc_mps2 = Eigen::Vector3d(a[0], a[1], a[2]);
    if (!v.empty() && s.t_ns <= v.back().t_ns)
      return invalid_argument("imu replay: " + p.string() + ":" + std::to_string(lineno) +
                              ": timestamps not strictly increasing");
    v.push_back(s);
  }
  if (v.empty()) return invalid_argument("imu replay: " + p.string() + ": no samples");
  return v;
}

Result<std::unique_ptr<ImuBackend>> make_imu_replay(const ImuConfig& c) {
  RIG_ASSIGN_OR_RETURN(std::vector<ImuSample> v, load_euroc_imu_csv(c.replay.path));
  return std::unique_ptr<ImuBackend>(std::make_unique<ImuReplay>(c, std::move(v)));
}

}  // namespace rig
