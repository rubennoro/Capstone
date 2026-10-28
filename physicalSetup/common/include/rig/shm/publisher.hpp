// SPDX-License-Identifier: MIT
/// @file rig/shm/publisher.hpp
/// @brief StreamHub -> shm ring bridges ("ShmPublisher" in PIPELINE 5): frames (native or downscaled per consumer,
/// PLAN T3 "frame rings carry a downscaled or native frame") and IMU samples. T3 addition (own module).
///
/// Each publisher subscribes to a hub with kDropOk (a slow shm path never affects lossless consumers), runs one
/// thread that pops, (optionally) downscales, encodes into the ring with the zero-copy begin_write()/commit(), and
/// stops when the hub closes or stop() is called. Ring names: "<shm.prefix>.<stream>" (ring.hpp), e.g.
/// "/rig.cam_uw", "/rig.imu0".
/// Threading: start/stop/stats from the owning thread; stats() is thread-safe.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "rig/capture/stream_hub.hpp"
#include "rig/shm/messages.hpp"
#include "rig/shm/ring.hpp"
#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

/// Pixel bytes of a tightly packed width x height image in `fmt` (what encode_frame() writes after the header).
/// 0 for kJPEG / kUnknown (variable size) or a non-positive size.
std::size_t frame_payload_bytes(int width, int height, PixelFormat fmt);

/// Output size for a downscale to `target_width` keeping the aspect ratio, both dimensions even (NV12 chroma).
/// Returns the input size when target_width <= 0 or >= width.
void downscaled_size(int width, int height, int target_width, int& out_width, int& out_height);

/// Resizes NV12 / GRAY8 / BGR8 / RGB8 frames with cv::INTER_AREA (NV12: Y and interleaved UV planes separately).
/// Returns `frame` itself (shared pixels, no copy) when no downscale is needed. kInvalidArgument for other formats.
Result<ImageFrame> downscale_frame(const ImageFrame& frame, int target_width);

struct ShmFramePublisherOptions {
  std::string shm_name;         ///< e.g. "/rig.cam_uw".
  std::string subscriber_name = "shm";  ///< Hub subscriber name.
  int capture_width = 0;        ///< Size and format of the hub's frames (fixes the slot size).
  int capture_height = 0;
  PixelFormat fmt = PixelFormat::kNV12;
  int out_width = 0;            ///< Published width; 0 = native (shm.viz_frame_width).
  std::size_t n_slots = 8;      ///< shm.frame_ring_slots.
  std::size_t queue = 2;        ///< DropOk queue depth between the hub and the publisher thread.
};

struct ShmPublisherStats {
  uint64_t published = 0;   ///< Messages committed to the ring.
  uint64_t errors = 0;      ///< Frames that could not be downscaled/encoded (wrong size/format).
  SubscriberStats queue;    ///< The DropOk hub subscription (dropped = frames skipped because the ring path lagged).
};

class ShmFramePublisher {
 public:
  /// Creates the ring (kAlreadyExists if a live writer owns the name), subscribes, starts the thread.
  static Result<std::unique_ptr<ShmFramePublisher>> start(StreamHub<ImageFrame>& hub,
                                                          const ShmFramePublisherOptions& options);
  ~ShmFramePublisher();  ///< stop(); the ring is unlinked.
  ShmFramePublisher(const ShmFramePublisher&) = delete;
  ShmFramePublisher& operator=(const ShmFramePublisher&) = delete;
  void stop();
  ShmPublisherStats stats() const;
  int out_width() const;
  int out_height() const;

 private:
  ShmFramePublisher();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class ShmImuPublisher {
 public:
  /// Ring of one ShmImuSample per slot (`n_slots` = shm.imu_ring_slots).
  static Result<std::unique_ptr<ShmImuPublisher>> start(StreamHub<ImuSample>& hub, const std::string& shm_name,
                                                        std::size_t n_slots, std::size_t queue = 4096,
                                                        const std::string& subscriber_name = "shm");
  ~ShmImuPublisher();
  ShmImuPublisher(const ShmImuPublisher&) = delete;
  ShmImuPublisher& operator=(const ShmImuPublisher&) = delete;
  void stop();
  ShmPublisherStats stats() const;

 private:
  ShmImuPublisher();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rig
