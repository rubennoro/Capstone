// SPDX-License-Identifier: MIT
// CaptureService (rig/capture/capture_service.hpp; PIPELINE 5).
//
// One thread per sensor. Each applies its ThreadConfig, then calls backend->start() on itself (so backend-internal
// threads inherit the placement; R1 F1), then loops. Camera threads loop read(timeout) -> publish; the IMU thread loops read(batch) ->
// publish each sample, then drains clock brackets -> bracket hub. Thread placement from ThreadConfig (best effort,
// logged, counted). Steady state (SteadyStateConfig):
// - camera: `min_frames` consecutive frame intervals with |dt * fps - 1| <= rate_tolerance (dt from t_sof_ns, or
//   t_mid_ns when the backend has no SOF), so any drop or stall restarts the count;
// - IMU: the sample rate over the last imu_window_s within rate_tolerance of odr_hz (only once that much data exists).
// is_steady() latches. The first fatal backend error (read returned nothing and status() is not OK) is kept for
// health() and ends that sensor's thread; other sensors keep running (the live checks decide what to do).
#include "rig/capture/capture_service.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <set>
#include <thread>

#include <spdlog/spdlog.h>

#include "rig/capture/thread_config.hpp"
#include "rig/time/clock.hpp"

namespace rig {
namespace {

constexpr TimeNs kSecond = 1000000000LL;

/// Rate over a sliding window of timestamps.
struct RateWindow {
  std::deque<TimeNs> t;
  TimeNs span_ns = kSecond;
  void add(TimeNs x) {
    t.push_back(x);
    while (!t.empty() && t.front() < x - span_ns) t.pop_front();
  }
  double rate_hz() const {
    if (t.size() < 2 || t.back() <= t.front()) return 0.0;
    return static_cast<double>(t.size() - 1) * 1e9 / static_cast<double>(t.back() - t.front());
  }
};

class CaptureServiceImpl final : public CaptureService {
 public:
  struct Cam {
    std::string id;
    std::unique_ptr<CameraBackend> be;
    std::unique_ptr<StreamHub<ImageFrame>> hub;
    ThreadConfig thread;
    double fps = 0;
    std::thread th;
    mutable std::mutex mu;
    uint64_t frames = 0;
    TimeNs last_t = kInvalidTime;
    TimeNs prev_t = kInvalidTime;
    int consecutive = 0;
    RateWindow fps_window;
    std::atomic<bool> steady{false};
  };
  struct Imu {
    std::unique_ptr<ImuBackend> be;
    std::unique_ptr<StreamHub<ImuSample>> hub;
    std::unique_ptr<StreamHub<ClockBracket>> brackets;
    ThreadConfig thread;
    double odr = 0;
    std::thread th;
    mutable std::mutex mu;
    uint64_t samples = 0;
    TimeNs first_t = kInvalidTime;
    RateWindow window;
    std::atomic<bool> steady{false};
  };

  CaptureServiceImpl(CaptureOptions o, std::vector<std::unique_ptr<Cam>> cams, std::unique_ptr<Imu> imu)
      : opt_(std::move(o)), cams_(std::move(cams)), imu_(std::move(imu)) {
    if (imu_) imu_->window.span_ns = static_cast<TimeNs>(opt_.capture.steady_state.imu_window_s * 1e9);
  }
  ~CaptureServiceImpl() override { (void)stop(); }

  Status start() override {
    std::lock_guard<std::mutex> lk(life_mu_);
    if (started_ || stopped_)
      return failed_precondition("CaptureService::start: already started or stopped (not restartable)");
    started_ = true;
    // Each sensor thread applies its ThreadConfig FIRST and then calls backend->start() itself, so every thread
    // the backend creates in start() (T1's IMU poll thread, the Argus acquire thread) inherits the affinity,
    // SCHED_FIFO priority / nice value (pthread_create(3) inherits affinity; PTHREAD_INHERIT_SCHED is the default
    // for policy and priority; the nice value is per-thread and inherited). The Status of start() comes back
    // through a promise. Backends start in parallel; on any failure the ones that started are stopped.
    // Construct spdlog's registry here, before any sensor thread exists: the backends' and our threads log from
    // their first frame on, and a concurrent first use of the (uninstrumented) libspdlog static is reported by TSan.
    (void)spdlog::default_logger_raw();
    std::vector<std::future<Status>> results;
    std::vector<std::string> names;
    for (auto& c : cams_) {
      Cam* cp = c.get();
      auto prom = std::make_shared<std::promise<Status>>();
      results.push_back(prom->get_future());
      names.push_back("camera " + c->id + " start");
      cp->th = std::thread([this, cp, prom] { camera_thread(*cp, *prom); });
    }
    if (imu_) {
      auto prom = std::make_shared<std::promise<Status>>();
      results.push_back(prom->get_future());
      names.push_back("imu " + imu_->be->config().id + " start");
      imu_->th = std::thread([this, prom] { imu_thread(*imu_, *prom); });
    }
    Status first;
    std::vector<bool> ok(results.size(), false);
    for (std::size_t i = 0; i < results.size(); ++i) {
      const Status s = results[i].get();
      ok[i] = s.ok();
      if (!s.ok() && first.ok()) first = s.with_context(names[i]);
    }
    if (!first.ok()) {
      stop_flag_.store(true, std::memory_order_release);
      for (std::size_t i = 0; i < cams_.size(); ++i)
        if (ok[i]) (void)cams_[i]->be->stop();
      if (imu_ && ok.back()) (void)imu_->be->stop();
      for (auto& c : cams_)
        if (c->th.joinable()) c->th.join();
      if (imu_ && imu_->th.joinable()) imu_->th.join();
      close_hubs();
      stopped_ = true;
      return first;
    }
    t_start_ = now_raw_ns();
    return Status::Ok();
  }

  Status stop() override {
    std::lock_guard<std::mutex> lk(life_mu_);
    if (stopped_) return Status::Ok();
    stopped_ = true;
    stop_flag_.store(true, std::memory_order_release);
    Status first;
    if (started_) {
      for (auto& c : cams_) {
        Status s = c->be->stop();  // unblocks a concurrent read()
        if (!s.ok() && first.ok()) first = s.with_context("camera " + c->id + " stop");
      }
      if (imu_) {
        Status s = imu_->be->stop();
        if (!s.ok() && first.ok()) first = s.with_context("imu stop");
      }
      for (auto& c : cams_)
        if (c->th.joinable()) c->th.join();
      if (imu_ && imu_->th.joinable()) imu_->th.join();
    }
    close_hubs();
    return first;
  }

  bool is_steady() const override {
    if (steady_latched_.load(std::memory_order_acquire)) return true;
    for (const auto& c : cams_)
      if (!c->steady.load(std::memory_order_acquire)) return false;
    if (imu_ && !imu_->steady.load(std::memory_order_acquire)) return false;
    steady_latched_.store(true, std::memory_order_release);
    return true;
  }

  Status wait_steady(std::chrono::nanoseconds timeout) override {
    const TimeNs deadline = now_raw_ns() + timeout.count();
    while (true) {
      Status h = health();
      if (!h.ok()) return h;
      if (is_steady()) return Status::Ok();
      if (now_raw_ns() >= deadline) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::string not_steady;
    for (const auto& c : cams_) {
      if (!c->steady.load()) {
        std::lock_guard<std::mutex> lk(c->mu);
        not_steady += (not_steady.empty() ? "" : ", ") + c->hub->stream_name() + " (" + std::to_string(c->frames) +
                      " frames, " + std::to_string(c->consecutive) + "/" +
                      std::to_string(opt_.capture.steady_state.min_frames) + " in tolerance)";
      }
    }
    if (imu_ && !imu_->steady.load()) {
      std::lock_guard<std::mutex> lk(imu_->mu);
      char buf[96];
      std::snprintf(buf, sizeof(buf), " (%.1f Hz over the window, nominal %.1f Hz)", imu_->window.rate_hz(), imu_->odr);
      not_steady += (not_steady.empty() ? "" : ", ") + imu_->hub->stream_name() + buf;
    }
    return timeout_error("capture not steady after " + std::to_string(timeout.count() / 1000000) + " ms: " + not_steady);
  }

  std::vector<std::string> camera_ids() const override {
    std::vector<std::string> v;
    for (const auto& c : cams_) v.push_back(c->id);
    return v;
  }
  StreamHub<ImageFrame>* camera_hub(const std::string& id) override {
    for (auto& c : cams_)
      if (c->id == id) return c->hub.get();
    return nullptr;
  }
  StreamHub<ImuSample>* imu_hub() override { return imu_ ? imu_->hub.get() : nullptr; }
  StreamHub<ClockBracket>* imu_bracket_hub() override { return imu_ ? imu_->brackets.get() : nullptr; }

  Result<CameraStats> camera_stats(const std::string& id) const override {
    for (const auto& c : cams_)
      if (c->id == id) return c->be->stats();
    return not_found("no camera '" + id + "'");
  }
  Result<ImuStats> imu_stats() const override {
    if (!imu_) return not_found("no IMU");
    return imu_->be->stats();
  }

  CaptureStatus status() const override {
    CaptureStatus s{};
    s.t_ns = now_raw_ns();
    s.heartbeat_seq = heartbeat_.fetch_add(1, std::memory_order_relaxed) + 1;
    s.n_cameras = static_cast<uint32_t>(std::min<std::size_t>(cams_.size(), kMaxStatusCameras));
    s.steady = is_steady() ? 1 : 0;
    s.healthy = health().ok() ? 1 : 0;
    s.thread_config_failures = static_cast<uint16_t>(thread_config_failures_.load());
    for (std::size_t i = 0; i < s.n_cameras; ++i) {
      const Cam& c = *cams_[i];
      CaptureStreamStatus& o = s.cameras[i];
      std::strncpy(o.id, c.id.c_str(), sizeof(o.id) - 1);
      {
        std::lock_guard<std::mutex> lk(c.mu);
        o.fps = c.fps_window.rate_hz();
        o.frames = c.frames;
        o.last_t_ns = c.last_t;
      }
      o.drops = c.be->stats().drops_total();
      for (const SubscriberStats& ss : c.hub->stats())
        if (ss.policy == DeliveryPolicy::kLossless) o.lossless_overflows += ss.overflows;
    }
    s.imu_last_t_ns = kInvalidTime;
    if (imu_) {
      const ImuStats st = imu_->be->stats();
      {
        std::lock_guard<std::mutex> lk(imu_->mu);
        s.imu_rate_hz = imu_->window.rate_hz();
        s.imu_samples = imu_->samples;
      }
      s.imu_fifo_overruns = st.fifo_overruns;
      s.imu_queue_overruns = st.queue_overruns;
      s.imu_gaps = st.gaps;
      s.imu_max_gap_ns = st.max_gap_ns;
      s.imu_fifo_level_max = st.fifo_level_max;
      s.imu_last_t_ns = st.last_t_ns;
    }
    return s;
  }

  Status health() const override {
    std::lock_guard<std::mutex> lk(err_mu_);
    return fatal_;
  }

 private:
  void set_fatal(const Status& s) {
    std::lock_guard<std::mutex> lk(err_mu_);
    if (fatal_.ok()) {
      fatal_ = s;
      spdlog::error("capture: {}", s.to_string());
    }
  }

  void apply_placement(const ThreadConfig& tc, const std::string& name) {
    Status s = apply_thread_config(tc, name);
    if (!s.ok()) {
      thread_config_failures_.fetch_add(1);
      spdlog::warn("capture: thread placement not applied (continuing at default priority): {}", s.message());
    }
  }

  void camera_thread(Cam& c, std::promise<Status>& started) {
    apply_placement(c.thread, "cap_" + c.id);
    Status s = c.be->start();
    const bool ok = s.ok();
    started.set_value(std::move(s));
    if (ok) camera_loop(c);
  }

  void imu_thread(Imu& m, std::promise<Status>& started) {
    apply_placement(m.thread, "cap_" + m.be->config().id);
    Status s = m.be->start();
    const bool ok = s.ok();
    started.set_value(std::move(s));
    if (ok) imu_loop(m);
  }

  void camera_loop(Cam& c) {
    const auto timeout = std::chrono::milliseconds(opt_.capture.read_timeout_ms);
    const SteadyStateConfig& ss = opt_.capture.steady_state;
    while (!stop_flag_.load(std::memory_order_acquire)) {
      std::optional<ImageFrame> f = c.be->read(timeout);
      if (!f) {
        if (stop_flag_.load(std::memory_order_acquire)) break;
        Status st = c.be->status();
        if (!st.ok()) {
          set_fatal(st.with_context("camera " + c.id));
          break;
        }
        continue;
      }
      const TimeNs t = f->t_sof_ns != kInvalidTime ? f->t_sof_ns : f->t_mid_ns;
      {
        std::lock_guard<std::mutex> lk(c.mu);
        ++c.frames;
        if (t != kInvalidTime) {
          if (c.prev_t != kInvalidTime && c.fps > 0) {
            const double ratio = static_cast<double>(t - c.prev_t) * c.fps / 1e9;
            c.consecutive = std::abs(ratio - 1.0) <= ss.rate_tolerance ? c.consecutive + 1 : 0;
            if (c.consecutive >= ss.min_frames) c.steady.store(true, std::memory_order_release);
          }
          c.prev_t = t;
          c.fps_window.add(t);
        }
        c.last_t = f->t_mid_ns;
      }
      c.hub->publish(*f);
    }
  }

  void imu_loop(Imu& m) {
    const auto timeout = std::chrono::milliseconds(opt_.capture.read_timeout_ms);
    const SteadyStateConfig& ss = opt_.capture.steady_state;
    const TimeNs window_ns = static_cast<TimeNs>(ss.imu_window_s * 1e9);
    std::vector<ImuSample> batch;
    std::vector<ClockBracket> br;
    while (!stop_flag_.load(std::memory_order_acquire)) {
      batch.clear();
      const std::size_t n = m.be->read(batch, timeout);
      br.clear();
      m.be->read_brackets(br);
      for (const ClockBracket& b : br) m.brackets->publish(b);
      if (n == 0) {
        if (stop_flag_.load(std::memory_order_acquire)) break;
        Status st = m.be->status();
        if (!st.ok()) {
          set_fatal(st.with_context("imu " + m.be->config().id));
          break;
        }
        continue;
      }
      {
        std::lock_guard<std::mutex> lk(m.mu);
        for (const ImuSample& s : batch) {
          ++m.samples;
          if (s.t_ns == kInvalidTime) continue;
          if (m.first_t == kInvalidTime) m.first_t = s.t_ns;
          m.window.add(s.t_ns);
        }
        if (!m.window.t.empty() && m.window.t.back() - m.first_t >= window_ns && m.odr > 0 &&
            std::abs(m.window.rate_hz() / m.odr - 1.0) <= ss.rate_tolerance)
          m.steady.store(true, std::memory_order_release);
      }
      for (const ImuSample& s : batch) m.hub->publish(s);
    }
  }

  void close_hubs() {
    for (auto& c : cams_) c->hub->close();
    if (imu_) {
      imu_->hub->close();
      imu_->brackets->close();
    }
  }

  CaptureOptions opt_;
  std::vector<std::unique_ptr<Cam>> cams_;
  std::unique_ptr<Imu> imu_;
  std::mutex life_mu_;
  bool started_ = false, stopped_ = false;
  std::atomic<bool> stop_flag_{false};
  TimeNs t_start_ = kInvalidTime;
  mutable std::mutex err_mu_;
  Status fatal_;
  mutable std::atomic<uint64_t> heartbeat_{0};
  mutable std::atomic<bool> steady_latched_{false};
  std::atomic<int> thread_config_failures_{0};
};

}  // namespace

Result<std::unique_ptr<CaptureService>> CaptureService::create(CaptureOptions options,
                                                              std::vector<std::unique_ptr<CameraBackend>> cameras,
                                                              std::unique_ptr<ImuBackend> imu) {
  if (cameras.empty() && !imu) return invalid_argument("CaptureService: no cameras and no IMU");
  std::vector<std::unique_ptr<CaptureServiceImpl::Cam>> cams;
  std::set<std::string> ids;
  for (auto& be : cameras) {
    if (!be) return invalid_argument("CaptureService: null camera backend");
    const std::string id = be->config().id;
    if (id.empty()) return invalid_argument("CaptureService: camera backend without an id");
    if (!ids.insert(id).second) return invalid_argument("CaptureService: duplicate camera id '" + id + "'");
    auto c = std::make_unique<CaptureServiceImpl::Cam>();
    c->id = id;
    c->fps = be->config().fps;
    auto it = options.camera_threads.find(id);
    if (it != options.camera_threads.end()) c->thread = it->second;
    c->hub = std::make_unique<StreamHub<ImageFrame>>("cam_" + id);
    c->be = std::move(be);
    cams.push_back(std::move(c));
  }
  std::unique_ptr<CaptureServiceImpl::Imu> im;
  if (imu) {
    im = std::make_unique<CaptureServiceImpl::Imu>();
    const std::string id = imu->config().id;
    im->odr = imu->config().odr_hz;
    im->thread = options.imu_thread;
    im->hub = std::make_unique<StreamHub<ImuSample>>(id);
    im->brackets = std::make_unique<StreamHub<ClockBracket>>(id + "_brackets");
    im->be = std::move(imu);
  }
  return std::unique_ptr<CaptureService>(new CaptureServiceImpl(std::move(options), std::move(cams), std::move(im)));
}

Result<std::unique_ptr<CaptureService>> CaptureService::create(const RigConfig& config) {
  CaptureOptions o;
  o.capture = config.capture;
  std::vector<std::unique_ptr<CameraBackend>> cams;
  for (const auto& kv : config.cameras) {
    CameraConfig cc = kv.second.driver;
    cc.id = kv.first;
    auto be = make_camera(cc);
    if (!be.ok()) return be.status().with_context("camera " + kv.first);
    cams.push_back(std::move(be).value());
    o.camera_threads[kv.first] = kv.second.thread;
  }
  auto imu = make_imu(config.imu.driver);
  if (!imu.ok()) return imu.status().with_context("imu " + config.imu.driver.id);
  o.imu_thread = config.imu.thread;
  return create(std::move(o), std::move(cams), std::move(imu).value());
}

}  // namespace rig
