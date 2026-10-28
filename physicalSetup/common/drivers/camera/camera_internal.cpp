// SPDX-License-Identifier: MIT
// Shared camera backend machinery (camera_internal.hpp).
#include "camera_internal.hpp"

#include <algorithm>
#include <cstring>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace rig {
namespace cam {

// ----------------------------------------------------------------------------------------------------------------
// BufferPool
// ----------------------------------------------------------------------------------------------------------------
BufferPool::State::~State() {
  for (uint8_t* p : free) delete[] p;
}

BufferPool::BufferPool(std::size_t max_free) : state_(std::make_shared<State>()) { state_->max_free = max_free; }

std::shared_ptr<uint8_t[]> BufferPool::acquire(std::size_t n) {
  uint8_t* p = nullptr;
  {
    std::lock_guard<std::mutex> lk(state_->mu);
    if (state_->size != n) {  // size change (restart with another mode): drop the old buffers
      for (uint8_t* q : state_->free) delete[] q;
      state_->free.clear();
      state_->size = n;
    }
    if (!state_->free.empty()) {
      p = state_->free.back();
      state_->free.pop_back();
    } else {
      ++state_->allocations;
    }
  }
  if (p == nullptr) p = new uint8_t[n];
  std::shared_ptr<State> st = state_;
  return std::shared_ptr<uint8_t[]>(p, [st, n](uint8_t* q) {
    std::lock_guard<std::mutex> lk(st->mu);
    if (st->size == n && st->free.size() < st->max_free) {
      st->free.push_back(q);
    } else {
      delete[] q;
    }
  });
}

std::size_t BufferPool::allocations() const {
  std::lock_guard<std::mutex> lk(state_->mu);
  return state_->allocations;
}

// ----------------------------------------------------------------------------------------------------------------
// FrameQueue
// ----------------------------------------------------------------------------------------------------------------
std::size_t FrameQueue::push(ImageFrame f) {
  std::size_t dropped = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_) return 0;  // shutting down: discarded, not a drop
    while (q_.size() >= capacity_) {
      q_.pop_front();
      ++dropped;
    }
    q_.push_back(std::move(f));
  }
  cv_.notify_one();
  return dropped;
}

std::optional<ImageFrame> FrameQueue::pop(std::chrono::nanoseconds timeout) {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait_for(lk, timeout, [&] { return closed_ || !q_.empty(); });
  if (closed_ || q_.empty()) return std::nullopt;
  ImageFrame f = std::move(q_.front());
  q_.pop_front();
  return f;
}

void FrameQueue::close() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    q_.clear();
  }
  cv_.notify_all();
}

void FrameQueue::reopen(std::size_t capacity) {
  std::lock_guard<std::mutex> lk(mu_);
  closed_ = false;
  q_.clear();
  capacity_ = capacity < 1 ? 1 : capacity;
}

void FrameQueue::clear() {
  std::lock_guard<std::mutex> lk(mu_);
  q_.clear();
}

std::size_t FrameQueue::size() const {
  std::lock_guard<std::mutex> lk(mu_);
  return q_.size();
}

// ----------------------------------------------------------------------------------------------------------------
// Frame layout and conversion
// ----------------------------------------------------------------------------------------------------------------
std::size_t packed_frame_bytes(PixelFormat fmt, int w, int h) {
  if (w <= 0 || h <= 0) return 0;
  const auto W = static_cast<std::size_t>(w), H = static_cast<std::size_t>(h);
  switch (fmt) {
    case PixelFormat::kNV12: return (w % 2 || h % 2) ? 0 : W * H * 3 / 2;
    case PixelFormat::kGRAY8: return W * H;
    case PixelFormat::kBGR8:
    case PixelFormat::kRGB8: return W * H * 3;
    case PixelFormat::kYUYV: return (w % 2) ? 0 : W * H * 2;
    case PixelFormat::kJPEG:
    case PixelFormat::kUnknown: return 0;
  }
  return 0;
}

uint8_t* allocate_frame(BufferPool& pool, PixelFormat fmt, int w, int h, ImageFrame& f) {
  const std::size_t n = packed_frame_bytes(fmt, w, h);
  if (n == 0) return nullptr;
  std::shared_ptr<uint8_t[]> buf = pool.acquire(n);
  uint8_t* p = buf.get();
  f.data = std::move(buf);
  f.size = n;
  f.width = w;
  f.height = h;
  f.fmt = fmt;
  f.planes = {};
  const auto W = static_cast<std::size_t>(w), H = static_cast<std::size_t>(h);
  switch (fmt) {
    case PixelFormat::kNV12:
      f.planes[0] = Plane{0, W, w, h};
      f.planes[1] = Plane{W * H, W, w / 2, h / 2};
      break;
    case PixelFormat::kGRAY8: f.planes[0] = Plane{0, W, w, h}; break;
    case PixelFormat::kBGR8:
    case PixelFormat::kRGB8: f.planes[0] = Plane{0, W * 3, w, h}; break;
    case PixelFormat::kYUYV: f.planes[0] = Plane{0, W * 2, w, h}; break;
    default: break;
  }
  return p;
}

void convert_into_frame(const uint8_t* src, int channels, std::size_t stride, ImageFrame& f, uint8_t* dst) {
  const int w = f.width, h = f.height;
  const cv::Mat in(h, w, channels == 1 ? CV_8UC1 : CV_8UC3, const_cast<uint8_t*>(src), stride);
  const auto W = static_cast<std::size_t>(w), H = static_cast<std::size_t>(h);
  switch (f.fmt) {
    case PixelFormat::kGRAY8: {
      cv::Mat out(h, w, CV_8UC1, dst, W);
      if (channels == 1) in.copyTo(out); else cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
      break;
    }
    case PixelFormat::kBGR8:
    case PixelFormat::kRGB8: {
      cv::Mat out(h, w, CV_8UC3, dst, W * 3);
      if (channels == 1) {
        cv::cvtColor(in, out, cv::COLOR_GRAY2BGR);
      } else if (f.fmt == PixelFormat::kBGR8) {
        in.copyTo(out);
      } else {
        cv::cvtColor(in, out, cv::COLOR_BGR2RGB);
      }
      break;
    }
    case PixelFormat::kNV12: {
      cv::Mat y(h, w, CV_8UC1, dst, W);
      uint8_t* uv = dst + W * H;
      if (channels == 1) {
        in.copyTo(y);
        std::memset(uv, 128, W * H / 2);
      } else {
        // Full-range (JFIF) YCbCr: OpenCV's COLOR_BGR2YCrCb is the JPEG conversion (Y = 0.299 R + 0.587 G +
        // 0.114 B, Cb/Cr centred on 128, no 16..235 offset), unlike COLOR_BGR2YUV_I420, which is BT.601 video
        // range. Chroma is 2x2-averaged (INTER_AREA) and interleaved as NV12 (U = Cb first, then V = Cr).
        cv::Mat ycrcb;
        cv::cvtColor(in, ycrcb, cv::COLOR_BGR2YCrCb);
        cv::Mat ch[3];
        cv::split(ycrcb, ch);
        ch[0].copyTo(y);
        cv::Mat cr, cb;
        cv::resize(ch[1], cr, cv::Size(w / 2, h / 2), 0, 0, cv::INTER_AREA);
        cv::resize(ch[2], cb, cv::Size(w / 2, h / 2), 0, 0, cv::INTER_AREA);
        for (int r = 0; r < h / 2; ++r) {
          const uint8_t* pb = cb.ptr<uint8_t>(r);
          const uint8_t* pr = cr.ptr<uint8_t>(r);
          uint8_t* o = uv + static_cast<std::size_t>(r) * W;
          for (int x = 0; x < w / 2; ++x) {
            o[2 * x] = pb[x];
            o[2 * x + 1] = pr[x];
          }
        }
      }
      break;
    }
    case PixelFormat::kYUYV: {
      cv::Mat gray;
      if (channels == 1) gray = in; else cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);
      // Luma only (chroma neutral): mock/replay convenience, never a capture path.
      for (int r = 0; r < h; ++r) {
        const uint8_t* g = gray.ptr<uint8_t>(r);
        uint8_t* o = dst + static_cast<std::size_t>(r) * W * 2;
        for (int x = 0; x < w; ++x) {
          o[2 * x] = g[x];
          o[2 * x + 1] = 128;
        }
      }
      break;
    }
    default: break;
  }
  f.meta.color_range = ColorRange::kFull;  // gray copied as Y, colour via JFIF YCbCr, RGB/BGR are full range
}

// ----------------------------------------------------------------------------------------------------------------
// CameraBase
// ----------------------------------------------------------------------------------------------------------------
CameraBase::CameraBase(CameraConfig config)
    : config_(std::move(config)), queue_(static_cast<std::size_t>(std::max(1, config_.buffer_count))) {}

std::optional<ImageFrame> CameraBase::latest() const {
  std::lock_guard<std::mutex> lk(mu_);
  return latest_;
}

CameraStats CameraBase::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  CameraStats s = stats_;
  if (tracker_) {
    tracker_->fill(s);
    s.drops_error += tracker_->rejected();
    s.timestamp_anomalies += tracker_->rejected();
  }
  s.tsc_cross_check_ns = summarize(std::vector<double>(tsc_diffs_.begin(), tsc_diffs_.end()));
  s.focus_status = focus_.status;
  s.focus_position = focus_.position;
  s.focus_detail = focus_.detail;
  return s;
}

Status CameraBase::status() const {
  std::lock_guard<std::mutex> lk(mu_);
  return status_;
}

void CameraBase::begin_stream(double fps, int sensor_mode, std::string time_source) {
  std::lock_guard<std::mutex> lk(mu_);
  stats_ = CameraStats{};
  stats_.sensor_mode = sensor_mode;
  stats_.time_source = std::move(time_source);
  tracker_ = std::make_unique<FrameTracker>(fps, config_.drop_gap_factor);
  latest_.reset();
  status_ = Status::Ok();
  seq_base_.reset();
  own_counter_ = 0;
  rebase_pending_ = true;
  tsc_diffs_.clear();
  logged_drop_events_ = 0;
  if (config_.focus.mode == FocusMode::kNone) {
    focus_ = FocusResult{FocusStatus::kNotApplicable, -1, false, "fixed-focus lens (focus.mode none)"};
  } else {
    focus_ = FocusResult{FocusStatus::kUnknown, -1, false, "focus not applied yet"};
  }
}

void CameraBase::rebase() {
  std::lock_guard<std::mutex> lk(mu_);
  rebase_pending_ = true;
  seq_base_.reset();
  own_counter_ = 0;
  if (tracker_) tracker_->reset();
  const int mode = stats_.sensor_mode;
  std::string ts = stats_.time_source;
  stats_ = CameraStats{};
  stats_.sensor_mode = mode;
  stats_.time_source = std::move(ts);
  tsc_diffs_.clear();
  latest_.reset();
  queue_.clear();
}

bool CameraBase::track(ImageFrame& f, std::optional<uint64_t> driver_seq) {
  std::lock_guard<std::mutex> lk(mu_);
  return track_locked(f, driver_seq);
}

bool CameraBase::track_locked(ImageFrame& f, std::optional<uint64_t> driver_seq) {
  if (!tracker_) return false;
  DropEvent ev;
  const bool ok = tracker_->observe(driver_seq, f.t_sof_ns, f.t_arrival_ns, &ev);
  if (ev.any()) {
    ++logged_drop_events_;
    if (logged_drop_events_ <= 20 || logged_drop_events_ % 100 == 0)
      spdlog::warn("camera {}: {} (event #{})", config_.id, ev.reason, logged_drop_events_);
  }
  if (!ok) return false;
  if (rebase_pending_) {
    seq_base_ = driver_seq;
    rebase_pending_ = false;
  }
  if (driver_seq && seq_base_) {
    f.seq = *driver_seq - *seq_base_;
  } else {
    f.seq = own_counter_;
  }
  own_counter_ = f.seq + 1;
  if (f.t_arrival_ns != kInvalidTime && f.t_arrival_ns < f.t_sof_ns) ++stats_.timestamp_anomalies;
  f.camera_id = config_.id;
  f.meta.focus_status = focus_.status;
  f.meta.focus_position = focus_.position;
  if (f.meta.sensor_mode < 0) f.meta.sensor_mode = stats_.sensor_mode;
  stats_.exposure_ns = f.meta.exposure_ns;
  stats_.analog_gain = f.meta.analog_gain;
  return true;
}

void CameraBase::produce(ImageFrame f, std::optional<uint64_t> driver_seq) {
  // Track and push under mu_ so that rebase() (which clears the queue under mu_) can never be interleaved between
  // the seq assignment and the push (a stale pre-rebase frame would otherwise enter the queue). Lock order:
  // mu_ -> queue mutex; FrameQueue::pop() takes only the queue mutex.
  std::lock_guard<std::mutex> lk(mu_);
  if (!track_locked(f, driver_seq)) return;
  stats_.drops_backend_queue += queue_.push(std::move(f));
}

void CameraBase::deliver(const ImageFrame& f) {
  std::lock_guard<std::mutex> lk(mu_);
  ++stats_.frames;
  latest_ = f;
}

void CameraBase::set_fatal(Status s) {
  std::lock_guard<std::mutex> lk(mu_);
  if (status_.ok() && !s.ok()) {
    spdlog::error("camera {}: {}", config_.id, s.to_string());
    status_ = std::move(s);
  }
}

bool CameraBase::fatal() const {
  std::lock_guard<std::mutex> lk(mu_);
  return !status_.ok();
}

void CameraBase::count_error(const std::string& why) {
  std::lock_guard<std::mutex> lk(mu_);
  ++stats_.errors;
  if (stats_.errors <= 20 || stats_.errors % 100 == 0)
    spdlog::warn("camera {}: error #{}: {}", config_.id, stats_.errors, why);
}

void CameraBase::count_timestamp_anomaly(const std::string& why) {
  std::lock_guard<std::mutex> lk(mu_);
  ++stats_.timestamp_anomalies;
  if (stats_.timestamp_anomalies <= 20 || stats_.timestamp_anomalies % 1000 == 0)
    spdlog::warn("camera {}: timestamp anomaly #{}: {}", config_.id, stats_.timestamp_anomalies, why);
}

void CameraBase::set_focus(const FocusResult& r) {
  std::lock_guard<std::mutex> lk(mu_);
  focus_ = r;
}

void CameraBase::set_tsc_cross_check(double diff_ns) {
  std::lock_guard<std::mutex> lk(mu_);
  tsc_diffs_.push_back(diff_ns);
  if (tsc_diffs_.size() > 256) tsc_diffs_.pop_front();
}

void CameraBase::set_time_source(std::string s) {
  std::lock_guard<std::mutex> lk(mu_);
  stats_.time_source = std::move(s);
}

}  // namespace cam
}  // namespace rig
