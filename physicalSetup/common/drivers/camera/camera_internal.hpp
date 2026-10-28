// SPDX-License-Identifier: MIT
// Private building blocks shared by the camera backends: BufferPool, FrameQueue, frame allocation, CameraBase.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "rig/drivers/camera/frame_tracker.hpp"
#include "rig/drivers/camera/imx519_focus.hpp"
#include "rig/drivers/camera_backend.hpp"
#include "rig/types.hpp"

namespace rig {
namespace cam {

/// Recycles pixel buffers of one size. A buffer handed out keeps the pool state alive through its deleter, so
/// frames may outlive the backend. Thread-safe.
class BufferPool {
 public:
  explicit BufferPool(std::size_t max_free = 16);
  /// Writable buffer of `n` bytes (recycled when the size matches, else freshly allocated).
  std::shared_ptr<uint8_t[]> acquire(std::size_t n);
  std::size_t allocations() const;  ///< Buffers allocated so far (diagnostics/tests).

 private:
  struct State {
    std::mutex mu;
    std::vector<uint8_t*> free;
    std::size_t size = 0;
    std::size_t max_free = 16;
    std::size_t allocations = 0;
    ~State();
  };
  std::shared_ptr<State> state_;
};

/// Bounded FIFO of frames between a producer thread and read(). push() drops the OLDEST frame when full and
/// returns how many were dropped. pop() returns std::nullopt on timeout or once closed. Thread-safe.
class FrameQueue {
 public:
  explicit FrameQueue(std::size_t capacity) : capacity_(capacity < 1 ? 1 : capacity) {}
  std::size_t push(ImageFrame f);
  std::optional<ImageFrame> pop(std::chrono::nanoseconds timeout);
  void close();
  void reopen(std::size_t capacity);
  void clear();
  std::size_t size() const;

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<ImageFrame> q_;
  std::size_t capacity_;
  bool closed_ = false;
};

/// Bytes needed for a tightly packed frame of `fmt` (0 for unsupported: JPEG/unknown, or odd NV12 sizes).
std::size_t packed_frame_bytes(PixelFormat fmt, int width, int height);

/// Fills width/height/fmt/planes/size of `f` for tight packing (NV12: Y stride W, UV at W*H with stride W) and
/// returns a writable buffer from `pool` (also stored in f.data). nullptr if the format/size is unsupported.
uint8_t* allocate_frame(BufferPool& pool, PixelFormat fmt, int width, int height, ImageFrame& f);

/// Converts an 8-bit gray or BGR image (OpenCV layout, `channels` 1 or 3, row stride `stride`) into `f`'s format.
/// `f` must already be allocated with allocate_frame(). YCbCr output is FULL range (JFIF): gray is copied as Y with
/// neutral chroma, colour uses OpenCV COLOR_BGR2YCrCb; sets f.meta.color_range = kFull (R1 F16).
void convert_into_frame(const uint8_t* src, int channels, std::size_t stride, ImageFrame& f, uint8_t* dst);

/// Common state of every backend: config, stats, latest frame, fatal status, focus result, drop tracking.
class CameraBase : public CameraBackend {
 public:
  explicit CameraBase(CameraConfig config);

  const CameraConfig& config() const override { return config_; }
  std::optional<ImageFrame> latest() const override;
  CameraStats stats() const override;
  Status status() const override;

 protected:
  /// Resets counters and the tracker for a new stream at `fps` (call in start()).
  void begin_stream(double fps, int sensor_mode, std::string time_source);
  /// Producer side: tracks drops on `f` (t_sof/t_arrival set) and assigns f.seq from `driver_seq` relative to the
  /// first frame after begin_stream()/rebase(). Returns false if the frame must be discarded (non-monotonic). Also
  /// stamps focus and records exposure/gain readback.
  bool track(ImageFrame& f, std::optional<uint64_t> driver_seq);
  bool track_locked(ImageFrame& f, std::optional<uint64_t> driver_seq);  ///< track() with mu_ held.
  /// Producer side for queued backends: track(), then push into `queue_` (oldest dropped -> drops_backend_queue).
  void produce(ImageFrame f, std::optional<uint64_t> driver_seq);
  /// Consumer side: frame handed to the caller of read() (counts it, sets latest()).
  void deliver(const ImageFrame& f);
  /// Restart counting after the settle phase: the next frame gets seq 0 and the queue is emptied.
  void rebase();
  void set_fatal(Status s);
  bool fatal() const;
  void count_error(const std::string& why);
  /// A plausible-but-odd stamp (e.g. sensor first-data time up to kMaxFutureCameraStampNs after arrival, R1 F3).
  void count_timestamp_anomaly(const std::string& why);
  void set_focus(const FocusResult& r);
  void set_tsc_cross_check(double diff_ns);
  void set_time_source(std::string s);
  void set_resolved_mode(int index) { config_.sensor_mode = index; }

  CameraConfig config_;
  FrameQueue queue_;

 private:
  mutable std::mutex mu_;
  CameraStats stats_;
  std::unique_ptr<FrameTracker> tracker_;
  std::optional<ImageFrame> latest_;
  Status status_;
  std::optional<uint64_t> seq_base_;
  uint64_t own_counter_ = 0;
  bool rebase_pending_ = true;
  FocusResult focus_;
  std::deque<double> tsc_diffs_;
  uint64_t logged_drop_events_ = 0;
};

}  // namespace cam
}  // namespace rig
