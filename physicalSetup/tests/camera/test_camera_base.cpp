// SPDX-License-Identifier: MIT
// CameraBase / FrameQueue / BufferPool under concurrency, without GStreamer or Argus: exactly the machinery the gst
// appsink callback and the Argus acquisition thread use (produce() on a producer thread, read() on the consumer,
// stats()/latest() from observers, rebase() from start()). Run under TSan by check.sh --sanitize thread.
#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "../../common/drivers/camera/camera_internal.hpp"  // private header of the camera module (T2)
#include "rig/time/clock.hpp"

namespace rig {
namespace {

using namespace std::chrono_literals;

/// A queued backend driven by a test thread instead of a driver.
class FakeQueuedCamera final : public cam::CameraBase {
 public:
  explicit FakeQueuedCamera(CameraConfig c) : CameraBase(std::move(c)) {}
  Status start() override {
    begin_stream(config_.fps, 0, "fake");
    queue_.reopen(static_cast<std::size_t>(config_.buffer_count));
    return Status::Ok();
  }
  Status stop() override {
    queue_.close();
    return Status::Ok();
  }
  std::optional<ImageFrame> read(std::chrono::nanoseconds timeout) override {
    auto f = queue_.pop(timeout);
    if (f) deliver(*f);
    return f;
  }
  Result<std::vector<SensorMode>> modes() const override { return std::vector<SensorMode>{}; }

  /// Producer side, as in GstCamera::handle_sample / ArgusCamera::process. The first pixel carries driver_seq.
  void push(uint64_t driver_seq, TimeNs sof) {
    ImageFrame f;
    uint8_t* p = cam::allocate_frame(pool_, PixelFormat::kGRAY8, 16, 8, f);
    p[0] = static_cast<uint8_t>(driver_seq & 0xFF);
    f.t_sof_ns = sof;
    f.t_mid_ns = sof;
    f.t_arrival_ns = sof + 1000;
    produce(std::move(f), driver_seq);
  }
  void do_rebase() { rebase(); }
  void fail() { set_fatal(device_error("boom")); }
  void anomaly() { count_timestamp_anomaly("test"); }
  std::size_t pool_allocations() const { return pool_.allocations(); }

 private:
  cam::BufferPool pool_;
};

CameraConfig cfg(int buffers) {
  CameraConfig c;
  c.id = "fake";
  c.fps = 1000.0;
  c.buffer_count = buffers;
  return c;
}

constexpr TimeNs kT0 = 1'000'000'000;
constexpr TimeNs kP = 1'000'000;  // 1 kHz

TEST(CameraBase, ProducerConsumerObserver) {
  FakeQueuedCamera cam(cfg(4096));
  ASSERT_TRUE(cam.start().ok());
  constexpr uint64_t kN = 3000;
  std::atomic<bool> done{false};
  std::thread producer([&] {
    for (uint64_t k = 0; k < kN; ++k) cam.push(1000 + k, kT0 + static_cast<TimeNs>(k) * kP);
  });
  std::thread observer([&] {
    while (!done) {  // polls (real callers: 1-10 Hz); a busy spin on an unfair mutex can starve the producer
      (void)cam.stats();
      (void)cam.latest();
      (void)cam.status();
      std::this_thread::sleep_for(100us);
    }
  });
  std::vector<uint64_t> seqs;
  while (seqs.size() < kN) {
    auto f = cam.read(10s);
    ASSERT_TRUE(f.has_value()) << "after " << seqs.size();
    EXPECT_EQ(f->plane_data(0)[0], static_cast<uint8_t>((1000 + f->seq) & 0xFF));
    EXPECT_EQ(f->camera_id, "fake");
    seqs.push_back(f->seq);
  }
  producer.join();
  done = true;
  observer.join();
  for (uint64_t i = 0; i < kN; ++i) ASSERT_EQ(seqs[i], i);  // seq rebased to the first driver number, in order
  const CameraStats s = cam.stats();
  EXPECT_EQ(s.frames, kN);
  EXPECT_EQ(s.drops_total(), 0u);
  EXPECT_EQ(cam.latest()->seq, kN - 1);
}

TEST(CameraBase, OverflowDropsOldestAndSeqShowsTheGap) {
  FakeQueuedCamera cam(cfg(3));
  ASSERT_TRUE(cam.start().ok());
  for (uint64_t k = 0; k < 10; ++k) cam.push(k, kT0 + static_cast<TimeNs>(k) * kP);
  std::vector<uint64_t> seqs;
  while (auto f = cam.read(10ms)) seqs.push_back(f->seq);
  EXPECT_EQ(seqs, (std::vector<uint64_t>{7, 8, 9}));
  const CameraStats s = cam.stats();
  EXPECT_EQ(s.drops_backend_queue, 7u);
  EXPECT_EQ(s.drops_seq_gap, 0u);  // the driver did not skip: no double counting
  EXPECT_EQ(s.drops_total(), 7u);
}

TEST(CameraBase, DriverGapsAndNonMonotonicFrames) {
  FakeQueuedCamera cam(cfg(16));
  ASSERT_TRUE(cam.start().ok());
  cam.push(5, kT0);
  cam.push(6, kT0 + kP);
  cam.push(9, kT0 + 4 * kP);      // 2 lost in the driver
  cam.push(10, kT0 + 4 * kP);     // same SOF: rejected
  cam.push(11, kT0 + 8 * kP);     // 3 slots of SOF gap, seq says 1 missing -> 2 unexplained
  std::vector<uint64_t> seqs;
  while (auto f = cam.read(10ms)) seqs.push_back(f->seq);
  EXPECT_EQ(seqs, (std::vector<uint64_t>{0, 1, 4, 6}));
  const CameraStats s = cam.stats();
  EXPECT_EQ(s.drops_seq_gap, 3u);
  EXPECT_EQ(s.drops_time_gap, 2u);
  EXPECT_EQ(s.drops_error, 1u);
  EXPECT_EQ(s.timestamp_anomalies, 1u);
}

TEST(CameraBase, RebaseWhileProducingNeverLeaksStaleFrames) {
  FakeQueuedCamera cam(cfg(8));
  ASSERT_TRUE(cam.start().ok());
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> k{0};
  std::thread producer([&] {
    while (!stop) {
      const uint64_t i = k++;
      cam.push(100 + i, kT0 + static_cast<TimeNs>(i) * kP);
    }
  });
  std::this_thread::sleep_for(20ms);
  cam.do_rebase();  // as start() does after the settle phase
  // Every frame delivered after the rebase must carry a post-rebase seq. The producer runs flat out, so the queue
  // may legitimately drop the oldest post-rebase frames before the first read; those are counted, so the first
  // delivered seq can never exceed the queue drops. A stale pre-rebase frame (seq in the thousands) would.
  auto first = cam.read(1s);
  ASSERT_TRUE(first);
  EXPECT_LE(first->seq, cam.stats().drops_backend_queue);
  uint64_t last = first->seq;
  for (int i = 0; i < 200; ++i) {
    auto f = cam.read(1s);
    ASSERT_TRUE(f);
    EXPECT_GT(f->seq, last);
    last = f->seq;
  }
  stop = true;
  producer.join();
  EXPECT_EQ(cam.stats().drops_seq_gap, 0u);
}

TEST(CameraBase, StopUnblocksAndFatalIsVisible) {
  FakeQueuedCamera cam(cfg(4));
  ASSERT_TRUE(cam.start().ok());
  std::thread reader([&] { EXPECT_FALSE(cam.read(10s).has_value()); });
  std::this_thread::sleep_for(20ms);
  const TimeNs t0 = now_raw_ns();
  cam.fail();
  ASSERT_TRUE(cam.stop().ok());
  reader.join();
  EXPECT_LT(now_raw_ns() - t0, 500'000'000);
  EXPECT_EQ(cam.status().code(), StatusCode::kDeviceError);
  cam.push(1, kT0);  // after close: discarded silently, not counted as a drop
  EXPECT_EQ(cam.stats().drops_backend_queue, 0u);
}

TEST(CameraBase, TimestampAnomaliesCounted) {
  FakeQueuedCamera cam(cfg(4));
  ASSERT_TRUE(cam.start().ok());
  cam.anomaly();
  cam.anomaly();
  EXPECT_EQ(cam.stats().timestamp_anomalies, 2u);
  EXPECT_EQ(cam.stats().drops_total(), 0u);  // accepted frames, not drops
}

TEST(BufferPool, RecyclesAndOutlivesPool) {
  std::shared_ptr<uint8_t[]> keep;
  {
    cam::BufferPool pool(2);
    { auto a = pool.acquire(100); }
    { auto b = pool.acquire(100); }
    EXPECT_EQ(pool.allocations(), 1u);
    keep = pool.acquire(100);
    auto c = pool.acquire(50);  // size change drops the free list
    EXPECT_EQ(pool.allocations(), 2u);
  }
  keep[0] = 7;  // the buffer outlives its pool (state kept alive by the deleter)
  keep.reset();
}

}  // namespace
}  // namespace rig
