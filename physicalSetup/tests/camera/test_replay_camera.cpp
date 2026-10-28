// SPDX-License-Identifier: MIT
// ReplayCamera on EuRoC-style and package-style folders.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <thread>

#include <opencv2/imgcodecs.hpp>

#include "rig/drivers/camera_backend.hpp"
#include "rig/package/layout.hpp"
#include "rig/time/clock.hpp"

// ThreadSanitizer: OpenCV's imgcodecs loads libgdal, whose driver registration takes two internal mutexes in both
// orders on one thread (reported as lock-order-inversion inside libgdal, no rig frames). Third-party; suppressed.
extern "C" const char* __tsan_default_suppressions() { return "deadlock:libgdal.so\n"; }

namespace rig {
namespace {

using namespace std::chrono_literals;
namespace fs = std::filesystem;

class ReplayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = fs::path(RIG_TEST_TMP_ROOT) /
            ("t2_replay_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::remove_all(root_);
    fs::create_directories(root_);
  }
  void TearDown() override { fs::remove_all(root_); }

  /// Writes n frames, gray value = 10 * i, at t0 + i * dt, into <dir>/<sub>/<t>.png, data.csv names "<t>.png".
  fs::path write_folder(const std::string& name, const std::string& sub, int n, int64_t t0, int64_t dt,
                        const std::string& ext = ".png") {
    const fs::path d = root_ / name;
    fs::create_directories(d / sub);
    std::ofstream csv(d / layout::kCamData);
    csv << layout::kCamDataHeader << "\n";
    for (int i = 0; i < n; ++i) {
      const int64_t t = t0 + i * dt;
      cv::Mat img(24, 32, CV_8UC1, cv::Scalar(10 * i));
      cv::imwrite((d / sub / (std::to_string(t) + ext)).string(), img);
      csv << t << "," << t << ext << "\n";
    }
    return d;
  }

  CameraConfig cfg(const fs::path& p, bool realtime) {
    CameraConfig c;
    c.id = "r";
    c.backend = CameraBackendKind::kReplay;
    c.replay.path = p.string();
    c.replay.realtime = realtime;
    c.replay.retime = false;
    c.fps = 50;
    return c;
  }

  std::unique_ptr<CameraBackend> open(const CameraConfig& c) {
    auto r = make_camera(c);
    EXPECT_TRUE(r.ok()) << r.status().to_string();
    auto cam = std::move(r).value();
    const Status s = cam->start();
    EXPECT_TRUE(s.ok()) << s.to_string();
    return cam;
  }

  fs::path root_;
};

TEST_F(ReplayTest, EurocFolderAsFastAsPossible) {
  const fs::path d = write_folder("euroc", "data", 5, 1'000'000'000, 20'000'000);
  auto cam = open(cfg(d, false));
  for (int i = 0; i < 5; ++i) {
    auto f = cam->read(1s);
    ASSERT_TRUE(f) << i << " " << cam->status().to_string();
    EXPECT_EQ(f->seq, static_cast<uint64_t>(i));
    EXPECT_EQ(f->t_mid_ns, 1'000'000'000 + i * 20'000'000);
    EXPECT_EQ(f->t_sof_ns, f->t_mid_ns);  // no meta: t_sof = t_mid
    EXPECT_EQ(f->width, 32);
    EXPECT_EQ(f->height, 24);
    EXPECT_EQ(f->fmt, PixelFormat::kNV12);
    EXPECT_EQ(f->plane_data(0)[0], 10 * i);
    EXPECT_EQ(f->plane_data(1)[0], 128);
  }
  EXPECT_FALSE(cam->read(10ms).has_value());
  EXPECT_EQ(cam->status().code(), StatusCode::kOutOfRange);
  EXPECT_NE(cam->status().message().find("end of replay"), std::string::npos);
  EXPECT_EQ(cam->stats().frames, 5u);
  EXPECT_EQ(cam->stats().drops_total(), 0u);
}

TEST_F(ReplayTest, PackageFolderWithMetaAndRecordedDrops) {
  // Package layout exactly as the RawWriter writes it (rig/package/layout.hpp; the dir name is T3's constant).
  const fs::path d = write_folder("pkg", layout::kFramesDir, 4, 5'000'000'000, 16'666'667, ".jpg");
  {
    std::ofstream m(d / layout::kFrameMeta);
    m << layout::kFrameMetaHeader << ",extra\n";  // an unknown extra column is ignored
    const int seqs[] = {0, 1, 3, 4};  // seq 2 was dropped while recording
    for (int i = 0; i < 4; ++i) {
      const int64_t t = 5'000'000'000 + i * 16'666'667;
      // t_ns,seq,t_sof_ns,t_arrival_ns,exposure_ns,analog_gain,digital_gain,readout_ns,sensor_mode,focus_position,
      // focus_status
      m << t << "," << seqs[i] << "," << (t - 9'000'000) << "," << (t + 20'000'000) << ",8000000,2.5,1.0,10000000,2,512,"
        << to_string(FocusStatus::kLockedVcmI2c) << ",zzz\n";
    }
  }
  CameraConfig c = cfg(d, false);
  c.fps = 60;
  c.pixel_format = PixelFormat::kGRAY8;
  auto cam = open(c);
  std::vector<ImageFrame> fr;
  while (auto f = cam->read(1s)) fr.push_back(*f);
  ASSERT_EQ(fr.size(), 4u);
  EXPECT_EQ(fr[2].seq, 3u);
  EXPECT_EQ(fr[1].t_sof_ns, 5'000'000'000 + 16'666'667 - 9'000'000);
  EXPECT_EQ(fr[0].meta.exposure_ns, 8'000'000);
  EXPECT_DOUBLE_EQ(fr[0].meta.analog_gain, 2.5);
  EXPECT_EQ(fr[0].meta.readout_ns, 10'000'000);
  EXPECT_EQ(fr[0].meta.focus_status, FocusStatus::kLockedVcmI2c);
  EXPECT_EQ(fr[0].meta.focus_position, 512);
  EXPECT_EQ(fr[0].fmt, PixelFormat::kGRAY8);
  EXPECT_EQ(fr[0].meta.sensor_mode, 2);
  EXPECT_DOUBLE_EQ(fr[0].meta.digital_gain, 1.0);
  EXPECT_EQ(fr[0].meta.color_range, ColorRange::kFull);
  EXPECT_NEAR(fr[1].plane_data(0)[0], 10, 2);  // jpeg
  EXPECT_EQ(cam->stats().drops_seq_gap, 1u);
}

// NV12 from a colour image is JFIF full range (R1 F16), so a JPEG encoded straight from it decodes correctly.
TEST_F(ReplayTest, ColourIsFullRangeYCbCr) {
  const fs::path d = root_ / "colour";
  fs::create_directories(d);
  cv::imwrite((d / "1.png").string(), cv::Mat(24, 32, CV_8UC3, cv::Scalar(0, 0, 255)));  // pure red (BGR)
  std::ofstream(d / layout::kCamData) << layout::kCamDataHeader << "\n1,1.png\n";
  auto cam = open(cfg(d, false));
  auto f = cam->read(1s);
  ASSERT_TRUE(f) << cam->status().to_string();
  EXPECT_EQ(f->meta.color_range, ColorRange::kFull);
  // JFIF: Y = 0.299 R = 76.2, Cb = 128 - 0.1687 R = 85.0, Cr = 128 + 0.5 R = 255.5 -> 255. (BT.601 video range
  // would give Y 82, Cb 90, Cr 240.)
  EXPECT_NEAR(f->plane_data(0)[0], 76, 1);
  EXPECT_NEAR(f->plane_data(1)[0], 85, 1);
  EXPECT_NEAR(f->plane_data(1)[1], 255, 1);
}

TEST_F(ReplayTest, TsofFromExposureAndReadoutWhenNotRecorded) {
  const fs::path d = write_folder("meta2", "", 2, 1'000'000'000, 33'333'333);
  std::ofstream(d / layout::kFrameMeta) << "t_ns,exposure_ns,readout_ns\n1000000000,4000000,20000000\n";
  auto cam = open(cfg(d, false));
  auto f = cam->read(1s);
  ASSERT_TRUE(f);
  EXPECT_EQ(f->t_sof_ns, 1'000'000'000 - 2'000'000 - 10'000'000);
  auto g = cam->read(1s);
  ASSERT_TRUE(g);
  EXPECT_EQ(g->t_sof_ns, g->t_mid_ns);
}

TEST_F(ReplayTest, RealtimeRetimeAndLoop) {
  const fs::path d = write_folder("rt", "data", 3, 7'000'000'000, 20'000'000);
  CameraConfig c = cfg(d, true);
  c.replay.retime = true;
  c.replay.loop = true;
  const TimeNs before = now_raw_ns();
  auto cam = open(c);
  std::vector<ImageFrame> fr;
  for (int i = 0; i < 7; ++i) {
    auto f = cam->read(1s);
    ASSERT_TRUE(f) << cam->status().to_string();
    fr.push_back(*f);
  }
  const TimeNs elapsed = now_raw_ns() - before;
  // Retimed: first frame at start(); loop period = span (40 ms) + median dt (20 ms) = 60 ms.
  EXPECT_GE(fr[0].t_mid_ns, before);
  EXPECT_LT(fr[0].t_mid_ns - before, 50'000'000);
  for (std::size_t i = 1; i < fr.size(); ++i) {
    EXPECT_EQ(fr[i].t_mid_ns - fr[i - 1].t_mid_ns, 20'000'000) << i;
    EXPECT_EQ(fr[i].seq, i);
    EXPECT_GE(fr[i].t_arrival_ns + 2'000'000, fr[i].t_mid_ns);  // paced: not delivered early
  }
  EXPECT_GE(elapsed, 110'000'000);  // 6 intervals of 20 ms
  EXPECT_EQ(fr[3].plane_data(0)[0], 0);  // looped back to frame 0
  EXPECT_TRUE(cam->status().ok());
}

TEST_F(ReplayTest, RealtimeReadTimesOutWithoutSkipping) {
  const fs::path d = write_folder("slow", "data", 2, 0, 300'000'000);
  auto cam = open(cfg(d, true));
  auto f0 = cam->read(1s);
  ASSERT_TRUE(f0);
  EXPECT_FALSE(cam->read(20ms).has_value());
  auto f1 = cam->read(1s);
  ASSERT_TRUE(f1);
  EXPECT_EQ(f1->seq, 1u);
}

TEST_F(ReplayTest, Errors) {
  // Missing folder.
  {
    auto cam = make_camera(cfg(root_ / "nope", false)).value();
    EXPECT_EQ(cam->start().code(), StatusCode::kNotFound);
  }
  // Non-increasing timestamps.
  {
    const fs::path d = root_ / "bad";
    fs::create_directories(d);
    std::ofstream(d / "data.csv") << "#t,f\n10,a.png\n10,b.png\n";
    auto cam = make_camera(cfg(d, false)).value();
    const Status s = cam->start();
    EXPECT_EQ(s.code(), StatusCode::kInvalidArgument);
    EXPECT_NE(s.message().find("strictly increasing"), std::string::npos);
  }
  // Malformed row.
  {
    const fs::path d = root_ / "bad2";
    fs::create_directories(d);
    std::ofstream(d / "data.csv") << "#t,f\n10,a.png\nxyz\n";
    auto cam = make_camera(cfg(d, false)).value();
    EXPECT_EQ(cam->start().code(), StatusCode::kInvalidArgument);
  }
  // Missing and corrupt images are fatal at read time.
  {
    const fs::path d = write_folder("corrupt", "data", 2, 0, 1000);
    std::ofstream(d / "data" / "1000.png") << "not a png";
    auto cam = open(cfg(d, false));
    EXPECT_TRUE(cam->read(1s).has_value());
    EXPECT_FALSE(cam->read(1s).has_value());
    EXPECT_EQ(cam->status().code(), StatusCode::kDataLoss);
    EXPECT_NE(cam->status().message().find("cannot decode"), std::string::npos) << cam->status().message();
    EXPECT_FALSE(cam->read(1s).has_value());
  }
  {
    const fs::path d = write_folder("missing", "data", 1, 0, 1000);
    std::ofstream(d / "data.csv", std::ios::app) << "5000,ghost.png\n";
    auto cam = open(cfg(d, false));
    EXPECT_TRUE(cam->read(1s).has_value());
    EXPECT_FALSE(cam->read(1s).has_value());
    EXPECT_NE(cam->status().message().find("not found"), std::string::npos);
  }
}

TEST_F(ReplayTest, StopUnblocksPacedRead) {
  const fs::path d = write_folder("block", "data", 2, 0, 5'000'000'000);
  auto cam = open(cfg(d, true));
  ASSERT_TRUE(cam->read(1s));
  std::thread t([&] { EXPECT_FALSE(cam->read(10s).has_value()); });
  std::this_thread::sleep_for(30ms);
  const TimeNs t0 = now_raw_ns();
  ASSERT_TRUE(cam->stop().ok());
  t.join();
  EXPECT_LT(now_raw_ns() - t0, 500'000'000);
}

}  // namespace
}  // namespace rig
