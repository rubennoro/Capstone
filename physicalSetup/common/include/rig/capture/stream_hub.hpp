// SPDX-License-Identifier: MIT
/// @file rig/capture/stream_hub.hpp
/// @brief In-process fan-out of one stream to many bounded subscriber queues (PIPELINE 5 "Producers never
/// block"). FROZEN CONTRACT (T0); implemented by T3 in common/capture/ with explicit instantiations for
/// ImageFrame, ImuSample and ClockBracket (other T are not available).
///
/// Policies:
/// - `kLossless` (RawWriter): on a full queue the NEW item is rejected (queue order stays gap-free up to the
///   overflow), `overflows` is incremented and `StreamHub::lossless_overflow()` latches true. T5 turns that into an
///   abort (PIPELINE 6.2 "Raw writer queue overflow").
/// - `kDropOk` (viewers, shm publisher, live checks that only need recent data): on a full queue the OLDEST item
///   is discarded and `dropped` is incremented.
/// publish() never blocks on a consumer: it takes each queue's mutex only for the push (bounded, O(1)).
///
/// Threading: publish() from one producer thread (calls from several threads are safe but unordered).
/// subscribe()/unsubscribe()/stats() from any thread. A StreamSubscription is popped by one consumer thread;
/// its stats() is thread-safe.
/// Ownership: the hub and each subscription share the queue (shared_ptr); either may be destroyed first.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rig/status.hpp"
#include "rig/time/clock.hpp"
#include "rig/types.hpp"

namespace rig {

enum class DeliveryPolicy : uint8_t {
  kLossless = 0,  ///< See file comment.
  kDropOk = 1,
};

constexpr const char* to_string(DeliveryPolicy p) noexcept {
  return p == DeliveryPolicy::kLossless ? "lossless" : "drop_ok";
}

struct SubscriberOptions {
  std::string name;                             ///< Unique per hub, e.g. "raw_writer", "viz", "live_checks".
  DeliveryPolicy policy = DeliveryPolicy::kDropOk;
  std::size_t capacity = 16;                    ///< Max queued items (>= 1).
};

struct SubscriberStats {
  std::string name;
  DeliveryPolicy policy = DeliveryPolicy::kDropOk;
  std::size_t capacity = 0;
  uint64_t offered = 0;    ///< Items published while subscribed.
  uint64_t delivered = 0;  ///< Items popped by the consumer.
  uint64_t dropped = 0;    ///< kDropOk: oldest items discarded.
  uint64_t overflows = 0;  ///< kLossless: new items rejected because the queue was full (data lost).
  std::size_t depth = 0;   ///< Current queue length.
  std::size_t max_depth = 0;  ///< High-water mark.
};

template <class T>
class StreamSubscription {
 public:
  ~StreamSubscription();
  StreamSubscription(const StreamSubscription&) = delete;
  StreamSubscription& operator=(const StreamSubscription&) = delete;

  /// Pops the oldest item into `out`, waiting up to `timeout`. False on timeout, or when the hub is closed and the
  /// queue is empty.
  bool pop(T& out, std::chrono::nanoseconds timeout);
  /// Non-blocking pop.
  bool try_pop(T& out);
  /// Appends every queued item to `out` (waiting up to `timeout` for the first). Returns the number appended.
  std::size_t pop_all(std::vector<T>& out, std::chrono::nanoseconds timeout);

  const std::string& name() const;
  SubscriberStats stats() const;
  /// True after the hub was closed or this subscription was unsubscribed (remaining items can still be popped).
  bool closed() const;

  struct Queue;  // defined by T3
  explicit StreamSubscription(std::shared_ptr<Queue> q);

 private:
  std::shared_ptr<Queue> q_;
};

template <class T>
class StreamHub {
 public:
  /// @param stream_name for logs and stats, e.g. "cam_main", "imu0".
  explicit StreamHub(std::string stream_name);
  ~StreamHub();
  StreamHub(const StreamHub&) = delete;
  StreamHub& operator=(const StreamHub&) = delete;

  /// Registers a subscriber. Subscribe before CaptureService::start() so lossless consumers miss nothing.
  /// kAlreadyExists for a duplicate name, kInvalidArgument for capacity 0, kFailedPrecondition after close().
  Result<std::shared_ptr<StreamSubscription<T>>> subscribe(const SubscriberOptions& options);
  /// Removes a subscriber by name (its queue is closed; queued items stay poppable). kNotFound if absent.
  Status unsubscribe(const std::string& name);

  /// Offers `item` to every subscriber per its policy. Never blocks on consumers.
  void publish(const T& item);
  /// Closes the hub: wakes all consumers; further publish() calls are ignored.
  void close();

  const std::string& stream_name() const;
  uint64_t published() const;                  ///< Items passed to publish().
  std::vector<SubscriberStats> stats() const;  ///< One entry per current subscriber.
  /// Latched true once any kLossless subscriber overflowed (never resets).
  bool lossless_overflow() const;

 private:
  struct Impl;  // defined by T3
  std::unique_ptr<Impl> impl_;
};

extern template class StreamSubscription<ImageFrame>;
extern template class StreamSubscription<ImuSample>;
extern template class StreamSubscription<ClockBracket>;
extern template class StreamHub<ImageFrame>;
extern template class StreamHub<ImuSample>;
extern template class StreamHub<ClockBracket>;

}  // namespace rig
