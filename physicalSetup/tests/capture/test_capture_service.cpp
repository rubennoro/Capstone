// SPDX-License-Identifier: MIT
// T3: CaptureService with fake backends: 60 s of (accelerated) capture with lossless consumers and zero drops,
// steady-state detection, start rollback, fatal errors, thread placement failures, status snapshot.
#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <functional>
#include <thread>

#include "fake_backends.hpp"
#include "rig/capture/capture_service.hpp"
#include "rig/capture/thread_config.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/time/clock.hpp"

namespace rig {
namespace {

using namespace std::chrono_literals;
using testing::FakeCamera;
using testing::FakeCameraOptions;
using testing::FakeImu;
using testing::FakeImuOptions;

FakeCameraOptions cam(const std::string& id, double fps = 60.0, int w = 64, int h = 48, double accel = 1.0) {
  FakeCameraOptions o;
  o.id = id;
  o.fps = fps;
  o.width = w;
  o.height = h;
  o.accel = accel;
  return o;
}

FakeImuOptions imu_opt(double odr, double accel) {
  FakeImuOptions o;
  o.odr_hz = odr;
  o.accel = accel;
  return o;
}

CaptureOptions default_options() {
  CaptureOptions o;  // SteadyStateConfig defaults: 30 frames, 10 %, 1 s IMU window
  return o;
}

std::vector<std::unique_ptr<CameraBackend>> cams(std::vector<FakeCameraOptions> os,
                                                 std::vector<FakeCamera*>* raw = nullptr) {
  std::vector<std::unique_ptr<CameraBackend>> v;
  for (auto& o : os) {
    auto c = std::make_unique<FakeCamera>(o);
    if (raw) raw->push_back(c.get());
    v.push_back(std::move(c));
  }
  return v;
}

/// Lossless consumer thread that checks in-order, gap-free delivery and counts.
template <class T>
struct Consumer {
  std::shared_ptr<StreamSubscription<T>> sub;
  std::atomic<uint64_t> count{0};
  std::atomic<int> errors{0};
  std::thread th;
  template <class Key>
  void run(Key key) {
    th = std::thread([this, key] {
      std::vector<T> batch;
      int64_t last = -1;
      while (true) {
        batch.clear();
        const std::size_t n = sub->pop_all(batch, 20ms);
        for (const T& x : batch) {
          const int64_t k = key(x);
          if (k != last + 1) errors++;
          last = k;
        }
        count += n;
        if (n == 0 && sub->closed()) break;
      }
    });
  }
};

TEST(CaptureService, SixtySecondsAcceleratedZeroLosslessDrops) {
  constexpr double kAccel = 10.0;  // 60 s of virtual capture in ~6 s
  constexpr int64_t kMain = 60 * 60, kUw = 30 * 60, kImu = 833 * 60;
  FakeCameraOptions main = cam("main", 60.0, 160, 120, kAccel);
  main.stall_at = kMain;
  FakeCameraOptions uw = cam("uw", 30.0, 164, 124, kAccel);
  uw.stall_at = kUw;
  FakeImuOptions im = imu_opt(833.0, kAccel);
  im.max_samples = kImu;
  CaptureOptions o = default_options();
  o.capture.read_timeout_ms = 50;
  auto svc = CaptureService::create(o, cams({main, uw}), std::make_unique<FakeImu>(im)).value();
  EXPECT_EQ(svc->camera_ids(), (std::vector<std::string>{"main", "uw"}));
  ASSERT_NE(svc->camera_hub("main"), nullptr);
  EXPECT_EQ(svc->camera_hub("side"), nullptr);
  EXPECT_EQ(svc->camera_hub("uw")->stream_name(), "cam_uw");

  // Subscribe before start (RawWriter-sized lossless queues from the example rig.yaml) plus a slow viewer.
  Consumer<ImageFrame> cm, cu;
  Consumer<ImuSample> ci;
  Consumer<ClockBracket> cb;
  cm.sub = svc->camera_hub("main")->subscribe({"raw_writer", DeliveryPolicy::kLossless, 120}).value();
  cu.sub = svc->camera_hub("uw")->subscribe({"raw_writer", DeliveryPolicy::kLossless, 60}).value();
  ci.sub = svc->imu_hub()->subscribe({"raw_writer", DeliveryPolicy::kLossless, 16384}).value();
  cb.sub = svc->imu_bracket_hub()->subscribe({"raw_writer", DeliveryPolicy::kLossless, 1024}).value();
  auto viewer = svc->camera_hub("uw")->subscribe({"viz", DeliveryPolicy::kDropOk, 2}).value();
  cm.run([](const ImageFrame& f) { return static_cast<int64_t>(f.seq); });
  cu.run([](const ImageFrame& f) { return static_cast<int64_t>(f.seq); });
  ci.run([](const ImuSample& s) { return s.chip_ticks / 48; });
  cb.run([](const ClockBracket& b) { return b.chip_ticks; });
  std::atomic<bool> viewer_done{false};
  std::thread vt([&] {
    ImageFrame f;
    while (!viewer_done) {
      if (viewer->pop(f, 10ms)) std::this_thread::sleep_for(30ms);  // slow: must only drop, never block
    }
  });

  EXPECT_FALSE(svc->is_steady());
  ASSERT_TRUE(svc->start().ok());
  EXPECT_EQ(svc->start().code(), StatusCode::kFailedPrecondition);
  // Steady needs 30 in-tolerance uw intervals (1 s virtual = 0.1 s real) and 1 s of IMU.
  ASSERT_TRUE(svc->wait_steady(10s).ok()) << svc->wait_steady(0s).to_string();
  EXPECT_TRUE(svc->is_steady());
  const CaptureStatus st = svc->status();
  EXPECT_EQ(st.n_cameras, 2u);
  EXPECT_EQ(st.steady, 1);
  EXPECT_EQ(st.healthy, 1);
  EXPECT_STREQ(st.cameras[0].id, "main");
  EXPECT_NEAR(st.cameras[0].fps, 60.0, 1.0);  // over the last virtual second
  EXPECT_GT(svc->status().heartbeat_seq, st.heartbeat_seq);

  const TimeNs deadline = now_raw_ns() + 60 * 1000000000LL;  // generous for sanitizer builds
  while ((cm.count < kMain || cu.count < kUw || ci.count < kImu) && now_raw_ns() < deadline)
    std::this_thread::sleep_for(20ms);
  const CaptureStatus end = svc->status();
  ASSERT_TRUE(svc->stop().ok());
  ASSERT_TRUE(svc->stop().ok());  // idempotent
  for (auto* c : {&cm, &cu}) c->th.join();
  ci.th.join();
  cb.th.join();
  viewer_done = true;
  vt.join();

  EXPECT_EQ(cm.count.load(), static_cast<uint64_t>(kMain));
  EXPECT_EQ(cu.count.load(), static_cast<uint64_t>(kUw));
  EXPECT_EQ(ci.count.load(), static_cast<uint64_t>(kImu));
  EXPECT_GE(cb.count.load(), 599u);
  EXPECT_EQ(cm.errors + cu.errors + ci.errors + cb.errors, 0);
  for (const char* id : {"main", "uw"}) {
    EXPECT_FALSE(svc->camera_hub(id)->lossless_overflow()) << id;
    for (const SubscriberStats& s : svc->camera_hub(id)->stats())
      if (s.policy == DeliveryPolicy::kLossless) {
        EXPECT_EQ(s.overflows, 0u) << id;
      }
  }
  EXPECT_FALSE(svc->imu_hub()->lossless_overflow());
  EXPECT_FALSE(svc->imu_bracket_hub()->lossless_overflow());
  EXPECT_EQ(end.cameras[1].frames, static_cast<uint64_t>(kUw));
  EXPECT_EQ(end.cameras[0].lossless_overflows, 0u);
  EXPECT_EQ(end.imu_samples, static_cast<uint64_t>(kImu));
  EXPECT_NEAR(end.imu_rate_hz, 833.0, 2.0);
  EXPECT_GT(viewer->stats().dropped, 0u);  // the slow viewer dropped instead of stalling capture
  EXPECT_TRUE(cm.sub->closed());           // stop() closed the hubs
  EXPECT_TRUE(svc->health().ok());
  EXPECT_EQ(svc->camera_stats("main").value().frames, static_cast<uint64_t>(kMain));
  EXPECT_EQ(svc->camera_stats("x").status().code(), StatusCode::kNotFound);
  EXPECT_EQ(svc->imu_stats().value().samples, static_cast<uint64_t>(kImu));
}

TEST(CaptureService, CreateValidation) {
  EXPECT_EQ(CaptureService::create(default_options(), {}, nullptr).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(CaptureService::create(default_options(), cams({cam("a"), cam("a")}), nullptr).status().code(),
            StatusCode::kInvalidArgument);
  auto cams_only = CaptureService::create(default_options(), cams({cam("main")}), nullptr).value();
  EXPECT_EQ(cams_only->imu_hub(), nullptr);
  EXPECT_EQ(cams_only->imu_bracket_hub(), nullptr);
  EXPECT_EQ(cams_only->imu_stats().status().code(), StatusCode::kNotFound);
  EXPECT_TRUE(cams_only->stop().ok());  // stop without start
  EXPECT_EQ(cams_only->start().code(), StatusCode::kFailedPrecondition);  // not restartable
}

TEST(CaptureService, StartFailureStopsWhatWasStarted) {
  std::vector<FakeCamera*> raw;
  FakeCameraOptions bad = cam("uw");
  bad.start_error = device_error("argus: no camera on CSI port 1");
  auto svc = CaptureService::create(default_options(), cams({cam("main"), bad}, &raw), nullptr).value();
  auto sub = svc->camera_hub("main")->subscribe({"w", DeliveryPolicy::kLossless, 4}).value();
  const Status s = svc->start();
  EXPECT_EQ(s.code(), StatusCode::kDeviceError);
  EXPECT_NE(s.message().find("camera uw start"), std::string::npos) << s.to_string();
  EXPECT_EQ(raw[0]->start_calls, 1);
  EXPECT_EQ(raw[0]->stop_calls, 1);
  EXPECT_TRUE(sub->closed());
  EXPECT_TRUE(svc->stop().ok());

  FakeImuOptions bi = imu_opt(833.0, 1.0);
  bi.start_error = device_error("WHO_AM_I 0x00 != 0x6c");
  std::vector<FakeCamera*> raw2;
  auto svc2 = CaptureService::create(default_options(), cams({cam("main")}, &raw2), std::make_unique<FakeImu>(bi)).value();
  EXPECT_EQ(svc2->start().code(), StatusCode::kDeviceError);
  EXPECT_EQ(raw2[0]->stop_calls, 1);
}

TEST(CaptureService, FatalBackendErrorIsReportedAndOthersKeepRunning) {
  FakeCameraOptions bad = cam("uw", 30.0, 64, 48, 10.0);
  bad.fatal_after = 5;
  auto svc = CaptureService::create(default_options(), cams({cam("main", 60.0, 64, 48, 10.0), bad}), nullptr).value();
  auto sub = svc->camera_hub("main")->subscribe({"w", DeliveryPolicy::kDropOk, 4}).value();
  ASSERT_TRUE(svc->start().ok());
  const Status w = svc->wait_steady(5s);
  EXPECT_EQ(w.code(), StatusCode::kDeviceError) << w.to_string();
  EXPECT_NE(w.message().find("camera uw"), std::string::npos);
  EXPECT_EQ(svc->health().code(), StatusCode::kDeviceError);
  EXPECT_EQ(svc->status().healthy, 0);
  const uint64_t before = sub->stats().offered;
  std::this_thread::sleep_for(100ms);
  EXPECT_GT(sub->stats().offered, before);  // main still captures
  EXPECT_TRUE(svc->stop().ok());
}

TEST(CaptureService, WaitSteadyTimeoutNamesTheStream) {
  FakeCameraOptions stalled = cam("uw", 30.0, 64, 48, 1.0);
  stalled.stall_at = 3;
  FakeCameraOptions dropping = cam("main", 60.0, 64, 48, 4.0);
  dropping.drop_every = 10;  // never 30 consecutive in-tolerance intervals
  auto svc = CaptureService::create(default_options(), cams({dropping, stalled}), nullptr).value();
  ASSERT_TRUE(svc->start().ok());
  const Status s = svc->wait_steady(400ms);
  EXPECT_EQ(s.code(), StatusCode::kTimeout);
  EXPECT_NE(s.message().find("cam_uw (3 frames"), std::string::npos) << s.to_string();
  EXPECT_NE(s.message().find("cam_main"), std::string::npos) << s.to_string();
  EXPECT_FALSE(svc->is_steady());
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_TRUE(svc->stop().ok());  // unblocks the stalled read promptly
  EXPECT_LT(std::chrono::steady_clock::now() - t0, 2s);
}

TEST(CaptureService, ThreadPlacementFailureIsCountedNotFatal) {
  CaptureOptions o = default_options();
  ThreadConfig impossible;
  impossible.cpus = {1000};  // no such CPU
  impossible.priority = 99;  // SCHED_FIFO: needs CAP_SYS_NICE (absent in tests)
  o.camera_threads["main"] = impossible;
  auto svc = CaptureService::create(o, cams({cam("main", 60.0, 64, 48, 10.0)}), nullptr).value();
  ASSERT_TRUE(svc->start().ok());
  ASSERT_TRUE(svc->wait_steady(5s).ok());
  EXPECT_GE(svc->status().thread_config_failures, 1);
  EXPECT_TRUE(svc->health().ok());
  EXPECT_TRUE(svc->stop().ok());

  // On a scratch thread: pinning the gtest main thread would leak into every later test's threads.
  std::thread([&] {
    ThreadConfig ok;
    ok.cpus = {0};
    EXPECT_TRUE(apply_thread_config(ok, "t3test").ok());
    const Status bad = apply_thread_config(impossible, "t3test");
    EXPECT_FALSE(bad.ok());
    EXPECT_NE(bad.message().find("affinity {1000}"), std::string::npos) << bad.to_string();
  }).join();
}

// R1 F1: placement is applied before backend->start(), on the thread that calls it, so threads the backend creates
// inherit it. Affinity {0} always exists; nice +5 needs no privilege.
TEST(CaptureService, BackendStartRunsWithTheConfiguredPlacement) {
  CaptureOptions o = default_options();
  ThreadConfig tc;
  tc.cpus = {0};
  tc.nice = 5;
  o.camera_threads["main"] = tc;
  std::vector<FakeCamera*> raw;
  auto svc = CaptureService::create(o, cams({cam("main", 60.0, 64, 48, 10.0), cam("uw", 30.0, 64, 48, 10.0)}, &raw),
                                    nullptr)
                 .value();
  ASSERT_TRUE(svc->start().ok());
  EXPECT_EQ(raw[0]->start_affinity, std::vector<int>{0});
  EXPECT_EQ(raw[0]->child_affinity, std::vector<int>{0});
  EXPECT_EQ(raw[0]->start_nice, 5);
  EXPECT_EQ(raw[0]->child_nice, 5);
  if (std::thread::hardware_concurrency() > 1) {
    EXPECT_GT(raw[1]->start_affinity.size(), 1u);  // the unpinned camera is not affected
  }
  EXPECT_EQ(raw[1]->start_nice, 0);
  EXPECT_EQ(svc->status().thread_config_failures, 0);
  EXPECT_TRUE(svc->stop().ok());
}

TEST(CaptureService, FromRigConfigWithMockBackends) {
  // Exercises make_camera()/make_imu() (T2 mock + T1 sim) when they are built; skipped otherwise.
  auto cfg = load_rig_config(std::string(RIG_SOURCE_DIR) +
                             "/episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml");
  ASSERT_TRUE(cfg.ok()) << cfg.status().to_string();
  RigConfig c = *cfg;
  for (auto& kv : c.cameras) {
    kv.second.driver.backend = CameraBackendKind::kMock;
    kv.second.thread = ThreadConfig{};
  }
  c.imu.driver.backend = ImuBackendKind::kSim;
  c.imu.thread = ThreadConfig{};
  auto svc = CaptureService::create(c);
  if (!svc.ok() && svc.status().code() == StatusCode::kUnimplemented) GTEST_SKIP() << svc.status().to_string();
  ASSERT_TRUE(svc.ok()) << svc.status().to_string();
  auto sub = (*svc)->camera_hub("uw")->subscribe({"w", DeliveryPolicy::kLossless, 120}).value();
  // Drain the lossless subscriber as a real consumer (writer) would; undrained, a slow (TSan) wait overflows it.
  std::atomic<bool> draining{true};
  std::atomic<uint64_t> drained{0};
  std::thread drainer([&] {
    ImageFrame f;
    while (draining.load()) {
      if (sub->pop(f, 20ms)) drained.fetch_add(1);
    }
  });
  auto stop_drainer = [&] {
    draining = false;
    if (drainer.joinable()) drainer.join();
  };
  struct Guard {
    std::function<void()> f;
    ~Guard() { f(); }
  } guard{stop_drainer};  // also on ASSERT/SKIP returns
  Status s = (*svc)->start();
  if (s.code() == StatusCode::kUnimplemented) GTEST_SKIP() << s.to_string();
  ASSERT_TRUE(s.ok()) << s.to_string();
  const Status w = (*svc)->wait_steady(10s);
#if defined(__SANITIZE_THREAD__)
  // TSan slows T2's full-size NV12 synthesis below the nominal rate on a loaded machine, so steady state (a rate
  // criterion) is not asserted there; the data path (delivery, zero overflows) still is.
  if (!w.ok()) std::printf("note (TSan): %s\n", w.to_string().c_str());
#else
  EXPECT_TRUE(w.ok()) << w.to_string();
#endif
  EXPECT_TRUE((*svc)->stop().ok());
  stop_drainer();
  EXPECT_GT(drained.load(), 0u);
  EXPECT_EQ(sub->stats().overflows, 0u);
}

}  // namespace
}  // namespace rig
