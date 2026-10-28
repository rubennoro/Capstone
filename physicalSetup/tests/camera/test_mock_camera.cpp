// SPDX-License-Identifier: MIT
// MockCamera timing, drops, stalls, patterns and lifecycle (it drives T5's tests, so it must be exact).
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "rig/drivers/camera_backend.hpp"
#include "rig/time/clock.hpp"

namespace rig {
namespace {

using namespace std::chrono_literals;

CameraConfig mock_cfg(double fps = 200.0) {
  CameraConfig c;
  c.id = "m";
  c.backend = CameraBackendKind::kMock;
  c.width = 64;
  c.height = 48;
  c.fps = fps;
  c.exposure_ns = 1'000'000;
  c.analog_gain = 2.0;
  // Deep simulated driver FIFO (1.28 s at 200 fps): a test thread descheduled under a loaded parallel ctest/sanitizer
  // run still gets every frame (late, true stamps), so the exact-sequence assertions stay deterministic. The FIFO
  // overflow behaviour itself is tested with buffer_count = 4 (BehindRealTimeDropsBeyondFifo).
  c.buffer_count = 256;
  return c;
}

std::unique_ptr<CameraBackend> make(const CameraConfig& c) {
  auto r = make_camera(c);
  EXPECT_TRUE(r.ok()) << r.status().to_string();
  return std::move(r).value();
}

std::vector<ImageFrame> grab(CameraBackend& cam, int n) {
  std::vector<ImageFrame> out;
  for (int i = 0; i < n; ++i) {
    auto f = cam.read(1s);
    if (!f) break;
    out.push_back(*f);
  }
  return out;
}

TEST(MockCamera, SteadyTimingAndMeta) {
  auto cam = make(mock_cfg());
  EXPECT_FALSE(cam->read(1ms).has_value());  // not started
  ASSERT_TRUE(cam->start().ok());
  EXPECT_EQ(cam->start().code(), StatusCode::kFailedPrecondition);
  const auto fr = grab(*cam, 40);
  ASSERT_EQ(fr.size(), 40u);
  for (std::size_t i = 0; i < fr.size(); ++i) {
    const ImageFrame& f = fr[i];
    EXPECT_TRUE(f.valid());
    EXPECT_EQ(f.camera_id, "m");
    EXPECT_EQ(f.seq, i);
    EXPECT_EQ(f.fmt, PixelFormat::kNV12);
    EXPECT_EQ(f.size, 64u * 48u * 3u / 2u);
    EXPECT_EQ(f.planes[1].offset, 64u * 48u);
    EXPECT_EQ(f.meta.exposure_ns, 1'000'000);
    EXPECT_DOUBLE_EQ(f.meta.analog_gain, 2.0);
    EXPECT_EQ(f.meta.readout_ns, 2'500'000);  // period/2 at 200 fps
    EXPECT_EQ(f.meta.color_range, ColorRange::kFull);
    EXPECT_EQ(f.t_mid_ns, mid_exposure_time(f.t_sof_ns, f.meta.exposure_ns, f.meta.readout_ns));
    EXPECT_GE(f.t_arrival_ns, f.t_sof_ns + 1'000'000 + 2'500'000 + 1'000'000);
    if (i > 0) {
      EXPECT_NEAR(static_cast<double>(f.t_sof_ns - fr[i - 1].t_sof_ns), 5e6, 1.0);  // exact ideal grid
    }
  }
  const CameraStats s = cam->stats();
  EXPECT_EQ(s.frames, 40u);
  EXPECT_EQ(s.drops_total(), 0u);
  EXPECT_NEAR(s.fps_estimate, 200.0, 0.5);
  EXPECT_EQ(s.exposure_ns, 1'000'000);
  EXPECT_EQ(s.time_source, "mock");
  EXPECT_EQ(s.focus_status, FocusStatus::kNotApplicable);
  ASSERT_TRUE(cam->latest().has_value());
  EXPECT_EQ(cam->latest()->seq, 39u);
  EXPECT_TRUE(cam->status().ok());
  EXPECT_TRUE(cam->stop().ok());
  EXPECT_TRUE(cam->stop().ok());  // idempotent
  EXPECT_FALSE(cam->read(1ms).has_value());
}

TEST(MockCamera, PacedInRealTime) {
  auto cam = make(mock_cfg(100.0));
  ASSERT_TRUE(cam->start().ok());
  const TimeNs t0 = now_raw_ns();
  const auto fr = grab(*cam, 21);
  const double elapsed = static_cast<double>(now_raw_ns() - t0);
  ASSERT_EQ(fr.size(), 21u);
  EXPECT_GT(elapsed, 0.19e9);  // 20 periods at least
  EXPECT_LT(elapsed, 0.40e9);
}

TEST(MockCamera, DropEveryN) {
  CameraConfig c = mock_cfg();
  c.mock.drop_every_n = 5;  // frame indices 4, 9, 14, ... are lost
  auto cam = make(c);
  ASSERT_TRUE(cam->start().ok());
  const auto fr = grab(*cam, 12);
  ASSERT_EQ(fr.size(), 12u);
  const uint64_t expect[] = {0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13};
  for (int i = 0; i < 12; ++i) EXPECT_EQ(fr[static_cast<std::size_t>(i)].seq, expect[i]);
  EXPECT_NEAR(static_cast<double>(fr[4].t_sof_ns - fr[3].t_sof_ns), 10e6, 1.0);
  const CameraStats s = cam->stats();
  EXPECT_EQ(s.drops_seq_gap, 2u);
  EXPECT_EQ(s.drops_time_gap, 0u);  // explained by the seq gap
  EXPECT_EQ(s.frames, 12u);
}

// R1 F12: a consumer that falls behind real time loses frames beyond the FIFO depth (counted), and never receives
// stale nominal stamps for frames that a real sensor would have overwritten.
TEST(MockCamera, BehindRealTimeDropsBeyondFifo) {
  CameraConfig c = mock_cfg();  // 200 fps
  c.buffer_count = 4;           // FIFO window 20 ms
  auto cam = make(c);
  ASSERT_TRUE(cam->start().ok());
  auto a = cam->read(1s);
  ASSERT_TRUE(a);
  std::this_thread::sleep_for(200ms);  // 40 periods behind
  auto b = cam->read(1s);
  ASSERT_TRUE(b);
  const uint64_t lost = b->seq - a->seq - 1;
  EXPECT_GT(lost, 25u);
  EXPECT_LT(lost, 45u);
  EXPECT_LT(b->t_arrival_ns - b->t_sof_ns, 45'000'000);  // at most ~FIFO window + nominal latency
  EXPECT_EQ(cam->stats().drops_seq_gap, lost);
  // It catches up: the next frames are consecutive.
  auto c1 = cam->read(1s), c2 = cam->read(1s);
  ASSERT_TRUE(c1 && c2);
  EXPECT_EQ(c1->seq, b->seq + 1);
  EXPECT_EQ(c2->seq, b->seq + 2);
}

TEST(MockCamera, JitterBoundedAndDeterministic) {
  CameraConfig c = mock_cfg();
  c.mock.jitter_ns = 1'000'000;
  c.mock.seed = 7;
  auto run = [&](uint64_t seed) {
    CameraConfig cc = c;
    cc.mock.seed = seed;
    auto cam = make(cc);
    EXPECT_TRUE(cam->start().ok());
    std::vector<int64_t> d;
    const auto fr = grab(*cam, 20);
    for (std::size_t i = 1; i < fr.size(); ++i) d.push_back(fr[i].t_sof_ns - fr[i - 1].t_sof_ns);
    return d;
  };
  const auto a = run(7), b = run(7), other = run(8);
  ASSERT_EQ(a.size(), 19u);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, other);
  for (int64_t d : a) {
    EXPECT_GE(d, 5'000'000 - 2'000'000);
    EXPECT_LE(d, 5'000'000 + 2'000'000);
  }
}

TEST(MockCamera, StallThenResume) {
  CameraConfig c = mock_cfg();
  c.mock.stall_after_frames = 5;
  c.mock.stall_ns = 50'000'000;  // 10 periods
  auto cam = make(c);
  ASSERT_TRUE(cam->start().ok());
  const auto fr = grab(*cam, 8);
  ASSERT_EQ(fr.size(), 8u);
  EXPECT_EQ(fr[5].seq, 5u);  // no seq gap: a driver stall
  EXPECT_NEAR(static_cast<double>(fr[5].t_sof_ns - fr[4].t_sof_ns), 55e6, 1.0);
  const CameraStats s = cam->stats();
  EXPECT_EQ(s.drops_seq_gap, 0u);
  EXPECT_EQ(s.drops_time_gap, 10u);
}

TEST(MockCamera, StallForeverTimesOut) {
  CameraConfig c = mock_cfg();
  c.mock.stall_after_frames = 3;
  auto cam = make(c);
  ASSERT_TRUE(cam->start().ok());
  EXPECT_EQ(grab(*cam, 3).size(), 3u);
  const TimeNs t0 = now_raw_ns();
  EXPECT_FALSE(cam->read(50ms).has_value());
  EXPECT_GE(now_raw_ns() - t0, 45'000'000);
  EXPECT_TRUE(cam->status().ok());
}

TEST(MockCamera, ReadTimesOutBetweenFrames) {
  auto cam = make(mock_cfg(10.0));
  ASSERT_TRUE(cam->start().ok());
  EXPECT_FALSE(cam->read(1ms).has_value());  // first frame is ~100 ms + latency away
  auto f = cam->read(1s);
  ASSERT_TRUE(f.has_value());
  EXPECT_EQ(f->seq, 0u);  // the timed-out attempt did not consume a frame
}

TEST(MockCamera, StopUnblocksRead) {
  auto cam = make(mock_cfg(1.0));
  ASSERT_TRUE(cam->start().ok());
  std::atomic<bool> returned{false};
  std::thread t([&] {
    EXPECT_FALSE(cam->read(10s).has_value());
    returned = true;
  });
  std::this_thread::sleep_for(50ms);
  const TimeNs t0 = now_raw_ns();
  ASSERT_TRUE(cam->stop().ok());
  t.join();
  EXPECT_TRUE(returned.load());
  EXPECT_LT(now_raw_ns() - t0, 500'000'000);
}

TEST(MockCamera, RestartStartsSeqAtZero) {
  auto cam = make(mock_cfg());
  ASSERT_TRUE(cam->start().ok());
  EXPECT_EQ(grab(*cam, 5).size(), 5u);
  ASSERT_TRUE(cam->stop().ok());
  ASSERT_TRUE(cam->start().ok());
  auto f = cam->read(1s);
  ASSERT_TRUE(f);
  EXPECT_EQ(f->seq, 0u);
  EXPECT_EQ(cam->stats().frames, 1u);
}

TEST(MockCamera, CounterPatternEncodesFrameIndex) {
  CameraConfig c = mock_cfg();
  c.mock.pattern = "counter";
  c.mock.drop_every_n = 3;
  auto cam = make(c);
  ASSERT_TRUE(cam->start().ok());
  for (const ImageFrame& f : grab(*cam, 6)) {
    uint64_t k = 0;
    for (int b = 0; b < 8; ++b) k |= static_cast<uint64_t>(f.plane_data(0)[b]) << (8 * b);
    EXPECT_EQ(k, f.seq);
    EXPECT_EQ(f.plane_data(1)[0], 128);  // gray: neutral chroma
  }
}

TEST(MockCamera, PatternsAndFormats) {
  for (const char* pat : {"checker", "gradient", "noise"}) {
    for (PixelFormat fmt : {PixelFormat::kNV12, PixelFormat::kGRAY8, PixelFormat::kBGR8, PixelFormat::kRGB8,
                            PixelFormat::kYUYV}) {
      CameraConfig c = mock_cfg();
      c.mock.pattern = pat;
      c.pixel_format = fmt;
      auto cam = make(c);
      ASSERT_TRUE(cam->start().ok()) << pat;
      auto a = cam->read(1s), b = cam->read(1s);
      ASSERT_TRUE(a && b);
      EXPECT_EQ(a->fmt, fmt);
      EXPECT_EQ(a->planes[0].width, 64);
      // Consecutive frames differ (moving pattern / fresh noise).
      EXPECT_NE(std::vector<uint8_t>(a->data.get(), a->data.get() + a->size),
                std::vector<uint8_t>(b->data.get(), b->data.get() + b->size))
          << pat << " " << to_string(fmt);
    }
  }
  // Same seed -> identical noise frames.
  CameraConfig c = mock_cfg();
  c.mock.pattern = "noise";
  auto x = make(c), y = make(c);
  ASSERT_TRUE(x->start().ok() && y->start().ok());
  auto fx = x->read(1s), fy = y->read(1s);
  ASSERT_TRUE(fx && fy);
  EXPECT_EQ(std::vector<uint8_t>(fx->data.get(), fx->data.get() + fx->size),
            std::vector<uint8_t>(fy->data.get(), fy->data.get() + fy->size));
}

TEST(MockCamera, UsesSensorTable) {
  CameraConfig c = mock_cfg(30.0);
  c.sensor = "imx219";
  c.width = 1640;
  c.height = 1232;
  c.exposure_ns = 4'000'000;
  auto cam = make(c);
  auto modes = cam->modes();
  ASSERT_TRUE(modes.ok());
  EXPECT_EQ(modes->size(), 5u);
  ASSERT_TRUE(cam->start().ok());
  EXPECT_EQ(cam->config().sensor_mode, 3);
  auto f = cam->read(1s);
  ASSERT_TRUE(f);
  EXPECT_EQ(f->meta.sensor_mode, 3);
  EXPECT_EQ(cam->stats().sensor_mode, 3);

  c.width = 1280;
  c.height = 720;
  c.fps = 60;
  auto crop = make(c);
  const Status s = crop->start();
  EXPECT_EQ(s.code(), StatusCode::kNotFound);
  EXPECT_NE(s.message().find("cropped mode on imx219"), std::string::npos) << s.message();
}

TEST(MockCamera, InvalidConfigs) {
  struct Case {
    void (*m)(CameraConfig&);
    StatusCode code;
  };
  const Case cases[] = {
      {[](CameraConfig& c) { c.mock.jitter_ns = 3'000'000; }, StatusCode::kInvalidArgument},  // >= period/2
      {[](CameraConfig& c) { c.mock.drop_every_n = 1; }, StatusCode::kInvalidArgument},
      {[](CameraConfig& c) { c.mock.pattern = "zebra"; }, StatusCode::kInvalidArgument},
      {[](CameraConfig& c) { c.width = 63; }, StatusCode::kInvalidArgument},  // odd NV12
      {[](CameraConfig& c) { c.pixel_format = PixelFormat::kJPEG; }, StatusCode::kInvalidArgument},
      {[](CameraConfig& c) { c.exposure_ns = 6'000'000; }, StatusCode::kInvalidArgument},  // > frame duration
  };
  for (const Case& t : cases) {
    CameraConfig c = mock_cfg();
    t.m(c);
    auto cam = make(c);
    EXPECT_EQ(cam->start().code(), t.code);
  }
}

TEST(MockCamera, FocusRecordedAsUnknown) {
  CameraConfig c = mock_cfg();
  c.focus.mode = FocusMode::kFixed;
  c.focus.position = 512;
  c.focus.i2c_bus = 9;
  auto cam = make(c);
  ASSERT_TRUE(cam->start().ok());
  auto f = cam->read(1s);
  ASSERT_TRUE(f);
  EXPECT_EQ(f->meta.focus_status, FocusStatus::kUnknown);
  EXPECT_EQ(f->meta.focus_position, 512);
}

TEST(MockCamera, ConcurrentStatsWhileReading) {
  auto cam = make(mock_cfg());
  ASSERT_TRUE(cam->start().ok());
  std::atomic<bool> done{false};
  std::thread observer([&] {
    while (!done) {
      (void)cam->stats();
      (void)cam->latest();
      (void)cam->status();
      std::this_thread::sleep_for(1ms);
    }
  });
  EXPECT_EQ(grab(*cam, 30).size(), 30u);
  done = true;
  observer.join();
}

}  // namespace
}  // namespace rig
