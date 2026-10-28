// SPDX-License-Identifier: MIT
// T3: StreamHub semantics and a multi-threaded stress test (run under TSan by check.sh --sanitize thread).
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "rig/capture/stream_hub.hpp"

namespace rig {
namespace {

using namespace std::chrono_literals;

ImuSample sample(int64_t i) {
  ImuSample s;
  s.t_ns = i;
  s.chip_ticks = i;
  return s;
}

TEST(StreamHub, SubscribeValidation) {
  StreamHub<ImuSample> hub("imu0");
  EXPECT_EQ(hub.stream_name(), "imu0");
  EXPECT_EQ(hub.subscribe({"a", DeliveryPolicy::kDropOk, 0}).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(hub.subscribe({"", DeliveryPolicy::kDropOk, 1}).status().code(), StatusCode::kInvalidArgument);
  auto a = hub.subscribe({"a", DeliveryPolicy::kDropOk, 4});
  ASSERT_TRUE(a.ok());
  EXPECT_EQ(hub.subscribe({"a", DeliveryPolicy::kLossless, 4}).status().code(), StatusCode::kAlreadyExists);
  EXPECT_EQ(hub.unsubscribe("zzz").code(), StatusCode::kNotFound);
  hub.close();
  EXPECT_EQ(hub.subscribe({"b", DeliveryPolicy::kDropOk, 4}).status().code(), StatusCode::kFailedPrecondition);
}

TEST(StreamHub, LosslessRejectsNewestAndLatches) {
  StreamHub<ImuSample> hub("imu0");
  auto sub = hub.subscribe({"raw_writer", DeliveryPolicy::kLossless, 3}).value();
  for (int i = 0; i < 5; ++i) hub.publish(sample(i));
  EXPECT_TRUE(hub.lossless_overflow());
  SubscriberStats st = sub->stats();
  EXPECT_EQ(st.offered, 5u);
  EXPECT_EQ(st.overflows, 2u);
  EXPECT_EQ(st.dropped, 0u);
  EXPECT_EQ(st.depth, 3u);
  EXPECT_EQ(st.max_depth, 3u);
  ImuSample s;
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(sub->try_pop(s));
    EXPECT_EQ(s.t_ns, i);  // gap-free prefix: the oldest are kept
  }
  EXPECT_FALSE(sub->try_pop(s));
  hub.publish(sample(9));
  ASSERT_TRUE(sub->try_pop(s));
  EXPECT_EQ(s.t_ns, 9);
  EXPECT_TRUE(hub.lossless_overflow());  // sticky
  EXPECT_EQ(hub.published(), 6u);
}

TEST(StreamHub, DropOkDropsOldest) {
  StreamHub<ImuSample> hub("imu0");
  auto sub = hub.subscribe({"viz", DeliveryPolicy::kDropOk, 2}).value();
  for (int i = 0; i < 5; ++i) hub.publish(sample(i));
  EXPECT_FALSE(hub.lossless_overflow());
  EXPECT_EQ(sub->stats().dropped, 3u);
  std::vector<ImuSample> all;
  EXPECT_EQ(sub->pop_all(all, 0ns), 2u);
  EXPECT_EQ(all[0].t_ns, 3);
  EXPECT_EQ(all[1].t_ns, 4);
  EXPECT_EQ(sub->stats().delivered, 2u);
}

TEST(StreamHub, PopTimesOutAndCloseWakes) {
  StreamHub<ImuSample> hub("imu0");
  auto sub = hub.subscribe({"c", DeliveryPolicy::kLossless, 8}).value();
  ImuSample s;
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_FALSE(sub->pop(s, 30ms));
  EXPECT_GE(std::chrono::steady_clock::now() - t0, 25ms);
  hub.publish(sample(1));
  std::thread closer([&] {
    std::this_thread::sleep_for(20ms);
    hub.close();
  });
  ASSERT_TRUE(sub->pop(s, 5s));  // queued item first
  const auto t1 = std::chrono::steady_clock::now();
  EXPECT_FALSE(sub->pop(s, 10s));  // woken by close, not by the timeout
  EXPECT_LT(std::chrono::steady_clock::now() - t1, 5s);
  closer.join();
  EXPECT_TRUE(sub->closed());
  hub.publish(sample(2));  // ignored
  EXPECT_FALSE(sub->try_pop(s));
}

TEST(StreamHub, UnsubscribeKeepsQueuedItemsAndDestroyedSubscriptionIsSkipped) {
  StreamHub<ImuSample> hub("imu0");
  auto a = hub.subscribe({"a", DeliveryPolicy::kLossless, 4}).value();
  {
    auto forgotten = hub.subscribe({"forgotten", DeliveryPolicy::kLossless, 1}).value();
  }
  hub.publish(sample(1));
  hub.publish(sample(2));
  EXPECT_FALSE(hub.lossless_overflow());  // the destroyed lossless subscriber must not latch
  ASSERT_EQ(hub.stats().size(), 1u);
  EXPECT_TRUE(hub.subscribe({"forgotten", DeliveryPolicy::kDropOk, 1}).ok());  // name reusable
  ASSERT_TRUE(hub.unsubscribe("a").ok());
  EXPECT_TRUE(a->closed());
  hub.publish(sample(3));
  std::vector<ImuSample> got;
  EXPECT_EQ(a->pop_all(got, 0ns), 2u);
  EXPECT_EQ(a->stats().offered, 2u);
}

TEST(StreamHub, SubscriptionOutlivesHub) {
  std::shared_ptr<StreamSubscription<ImageFrame>> sub;
  {
    StreamHub<ImageFrame> hub("cam_uw");
    sub = hub.subscribe({"x", DeliveryPolicy::kDropOk, 2}).value();
    ImageFrame f;
    f.camera_id = "uw";
    f.seq = 7;
    hub.publish(f);
  }
  EXPECT_TRUE(sub->closed());
  ImageFrame f;
  ASSERT_TRUE(sub->try_pop(f));
  EXPECT_EQ(f.seq, 7u);
}

TEST(StreamHub, ClockBracketInstantiation) {
  StreamHub<ClockBracket> hub("imu0_brackets");
  auto sub = hub.subscribe({"w", DeliveryPolicy::kLossless, 2}).value();
  hub.publish(ClockBracket{1, 2, 3});
  ClockBracket b;
  ASSERT_TRUE(sub->pop(b, 1ms));
  EXPECT_EQ(b.chip_ticks, 2);
}

// One publisher, several consumers with both policies, subscribers joining and leaving, a stats poller.
// Invariants: lossless consumers that never overflowed see 0..N-1 in order; every consumer sees strictly increasing
// values; offered == delivered + dropped + overflows + depth for every snapshot; the publisher never blocks long.
TEST(StreamHub, StressManyConsumersBothPolicies) {
  StreamHub<ImuSample> hub("stress");
  constexpr int64_t kN = 200000;
  auto lossless = hub.subscribe({"lossless", DeliveryPolicy::kLossless, static_cast<std::size_t>(kN)}).value();
  auto lossless_small = hub.subscribe({"lossless_small", DeliveryPolicy::kLossless, 4}).value();
  std::vector<std::shared_ptr<StreamSubscription<ImuSample>>> dropok;
  for (int i = 0; i < 3; ++i)
    dropok.push_back(hub.subscribe({"drop" + std::to_string(i), DeliveryPolicy::kDropOk, 16}).value());

  std::atomic<bool> done{false};
  std::atomic<int> failures{0};
  auto consume = [&](std::shared_ptr<StreamSubscription<ImuSample>> sub, bool expect_all) {
    int64_t last = -1;
    int64_t count = 0;
    std::vector<ImuSample> batch;
    while (true) {
      batch.clear();
      const std::size_t n = sub->pop_all(batch, 2ms);
      for (std::size_t i = 0; i < n; ++i) {
        if (batch[i].t_ns <= last) failures++;
        if (expect_all && batch[i].t_ns != last + 1) failures++;
        last = batch[i].t_ns;
        ++count;
      }
      if (n == 0 && sub->closed()) break;
    }
    if (expect_all && count != kN) failures++;
  };
  std::vector<std::thread> threads;
  threads.emplace_back(consume, lossless, true);
  threads.emplace_back(consume, lossless_small, false);
  for (auto& d : dropok) threads.emplace_back(consume, d, false);

  // Churn: subscribe/unsubscribe/destroy while publishing.
  threads.emplace_back([&] {
    int k = 0;
    while (!done.load()) {
      const std::string name = "churn" + std::to_string(k++ % 4);
      auto s = hub.subscribe({name, k % 2 ? DeliveryPolicy::kDropOk : DeliveryPolicy::kLossless, 8});
      if (s.ok()) {
        ImuSample x;
        (void)s.value()->try_pop(x);
        if (k % 3 == 0) (void)hub.unsubscribe(name);
      }
      std::this_thread::yield();
    }
  });
  // Stats poller.
  threads.emplace_back([&] {
    while (!done.load()) {
      for (const SubscriberStats& s : hub.stats()) {
        if (s.offered != s.delivered + s.dropped + s.overflows + s.depth) failures++;
        if (s.depth > s.capacity) failures++;
      }
      std::this_thread::sleep_for(100us);
    }
  });

  auto max_publish = std::chrono::nanoseconds(0);
  for (int64_t i = 0; i < kN; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    hub.publish(sample(i));
    max_publish = std::max(max_publish, std::chrono::steady_clock::now() - t0);
  }
  done = true;
  hub.close();
  for (auto& t : threads) t.join();
  EXPECT_EQ(failures.load(), 0);
  EXPECT_FALSE(lossless->stats().overflows);
  EXPECT_EQ(lossless->stats().delivered, static_cast<uint64_t>(kN));
  // lossless_small almost surely overflowed (capacity 4 vs. a tight publish loop); the latch must reflect it.
  if (lossless_small->stats().overflows > 0) {
    EXPECT_TRUE(hub.lossless_overflow());
  }
  for (auto& d : dropok) {
    const SubscriberStats s = d->stats();
    EXPECT_EQ(s.offered, static_cast<uint64_t>(kN));
    EXPECT_EQ(s.offered, s.delivered + s.dropped + s.depth);
    EXPECT_EQ(s.overflows, 0u);
  }
  RecordProperty("max_publish_us", static_cast<int>(max_publish.count() / 1000));
}

}  // namespace
}  // namespace rig
