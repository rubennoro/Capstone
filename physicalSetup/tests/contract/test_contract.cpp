// SPDX-License-Identifier: MIT
// T0 contract tests: the inline parts of the frozen headers and the defaults that encode iPhone parity.
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "rig/capture/capture_service.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/drivers/camera_backend.hpp"
#include "rig/imageprep/imageprep.hpp"
#include "rig/package/package.hpp"
#include "rig/shm/messages.hpp"
#include "rig/shm/status_channel.hpp"
#include "rig/status.hpp"
#include "rig/time/clock.hpp"
#include "rig/version.hpp"

namespace rig {
namespace {

Result<int> parse_positive(int v) {
  if (v <= 0) return invalid_argument("value: must be > 0, got " + std::to_string(v));
  return v;
}

Status twice(int v, int* out) {
  RIG_ASSIGN_OR_RETURN(int x, parse_positive(v));
  RIG_RETURN_IF_ERROR(x > 100 ? out_of_range("too big") : Status::Ok());
  *out = 2 * x;
  return Status::Ok();
}

TEST(Status, DefaultIsOkAndCodesHaveNames) {
  Status s;
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(s.to_string(), "ok");
  Status e = device_error("/dev/i2c-7 addr 0x6a: NACK");
  EXPECT_FALSE(e.ok());
  EXPECT_EQ(e.code(), StatusCode::kDeviceError);
  EXPECT_EQ(e.to_string(), "device_error: /dev/i2c-7 addr 0x6a: NACK");
  EXPECT_EQ(e.with_context("imu0").message(), "imu0: /dev/i2c-7 addr 0x6a: NACK");
  EXPECT_TRUE(Status(StatusCode::kOk, "ignored").message().empty());
}

TEST(Result, ValueErrorAndMacros) {
  Result<int> ok = parse_positive(3);
  ASSERT_TRUE(ok.ok());
  EXPECT_EQ(*ok, 3);
  Result<int> bad = parse_positive(-1);
  EXPECT_FALSE(bad.ok());
  EXPECT_EQ(bad.status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(bad.value_or(7), 7);

  int out = 0;
  EXPECT_TRUE(twice(4, &out).ok());
  EXPECT_EQ(out, 8);
  EXPECT_EQ(twice(-4, &out).code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(twice(400, &out).code(), StatusCode::kOutOfRange);

  // Constructing from an OK status is a bug and must not produce an "ok" result without a value.
  Result<int> from_ok{Status::Ok()};
  EXPECT_FALSE(from_ok.ok());
  EXPECT_EQ(from_ok.status().code(), StatusCode::kInternal);
}

struct Base {
  virtual ~Base() = default;
};
struct Derived : Base {};

TEST(Result, ConvertingConstructorForUniquePtr) {
  Result<std::unique_ptr<Base>> r = std::make_unique<Derived>();
  EXPECT_TRUE(r.ok());
  Result<std::string> s = "abc";
  EXPECT_EQ(*s, "abc");
}

TEST(Span, ViewsContainers) {
  std::vector<int> v{1, 2, 3, 4};
  Span<int> s(v);
  Span<const int> cs = s;
  EXPECT_EQ(cs.size(), 4u);
  EXPECT_EQ(cs[2], 3);
  EXPECT_EQ(s.subspan(1, 2).size(), 2u);
  EXPECT_EQ(s.subspan(3).size(), 1u);
  EXPECT_EQ(s.subspan(9).size(), 0u);
  const std::vector<int>& cv = v;
  Span<const int> from_const(cv);
  EXPECT_EQ(from_const.data(), v.data());
  int arr[3] = {0, 0, 0};
  EXPECT_EQ(Span<int>(arr).size_bytes(), sizeof(arr));
}

TEST(Clock, MonotonicRawAdvancesAndAnchorIsTight) {
  const TimeNs a = now_raw_ns();
  const TimeNs b = now_raw_ns();
  EXPECT_GE(b, a);
  EXPECT_GT(now_unix_ns(), 1700000000LL * 1000000000LL);  // after 2023
  const ClockAnchor anc = sample_clock_anchor(5);
  EXPECT_GE(anc.bracket_ns, 0);
  EXPECT_LT(anc.bracket_ns, 1000000);  // < 1 ms
  EXPECT_NE(anc.raw_ns, kInvalidTime);
}

TEST(Camera, MidExposureTime) {
  EXPECT_EQ(mid_exposure_time(1000, 400, 200), 1300);
  EXPECT_EQ(mid_exposure_time(1000, -1, 200), 1000);
  EXPECT_EQ(mid_exposure_time(kInvalidTime, 1, 1), kInvalidTime);
  CameraStats st;
  st.drops_seq_gap = 1;
  st.drops_time_gap = 2;
  st.drops_backend_queue = 3;
  st.drops_error = 4;
  EXPECT_EQ(st.drops_total(), 10u);
  EXPECT_EQ(num_planes(PixelFormat::kNV12), 2);
  EXPECT_STREQ(to_string(PixelFormat::kNV12), "nv12");
}

TEST(ImagePrep, LetterboxGeometryMatchesCvUtil) {
  // cv_util.py:13-17 with Python floor division.
  LetterboxGeometry g = letterbox_geometry(1920, 1080);
  EXPECT_EQ(g.size, 1920);
  EXPECT_EQ(g.top, 420);
  EXPECT_EQ(g.bottom, 420);
  EXPECT_EQ(g.left, 0);
  g = letterbox_geometry(1920, 1440);  // iPhone main
  EXPECT_EQ(g.top, 240);
  g = letterbox_geometry(1281, 720);  // odd padding: extra row goes to the bottom
  EXPECT_EQ(g.top, 280);
  EXPECT_EQ(g.bottom, 281);
  g = letterbox_geometry(480, 640);  // portrait: pad left/right
  EXPECT_EQ(g.left, 80);
  EXPECT_EQ(g.right, 80);
}

TEST(Config, LiveCheckDefaultsAreIphoneParity) {
  LiveChecksConfig c;
  EXPECT_DOUBLE_EQ(c.main_fps_min, 31.0);        // ViewController.swift:1819
  EXPECT_DOUBLE_EQ(c.main_fps_grace_s, 1.0);     // ViewController.swift:1819
  EXPECT_DOUBLE_EQ(c.pose_speed_max_mps, 7.0);   // ViewController.swift:305
  EXPECT_TRUE(c.uw_require_frames);              // DemonstrationData.swift:574
  EXPECT_DOUBLE_EQ(c.slam_heartbeat_timeout_s, 0.5);  // PIPELINE 6.2
  EXPECT_DOUBLE_EQ(c.uw_fps_min, 27.0);
  EXPECT_DOUBLE_EQ(c.imu_gap_max_ms, 20.0);
  EXPECT_DOUBLE_EQ(c.imu_rate_min_ratio, 0.95);
  EXPECT_DOUBLE_EQ(c.disk_min_free_gb, 5.0);
  EXPECT_DOUBLE_EQ(c.tags_unseen_warn_s, 0.5);
  RigConfig rc;
  EXPECT_EQ(rc.schema_version, kRigConfigSchemaVersion);
  EXPECT_EQ(rc.imageprep.out_size, 224);
}

TEST(Config, ImuDefaultsAreTheRigSettings) {
  ImuConfig c;
  EXPECT_EQ(c.i2c_bus, 7);
  EXPECT_EQ(c.i2c_addr, 0x6A);
  EXPECT_EQ(c.expected_who_am_i, 0x6C);
  EXPECT_DOUBLE_EQ(c.odr_hz, 833.0);
  EXPECT_EQ(c.accel_fs_g, 8);
  EXPECT_EQ(c.gyro_fs_dps, 2000);
  EXPECT_EQ(c.poll_period_ms, 10);
  EXPECT_DOUBLE_EQ(c.clock.nominal_tick_ns, 25000.0);
}

TEST(Package, NameFormatMatchesIphumi) {
  PackageName n;
  n.iso = "2026-10-01T21-13-57.489Z";
  n.randomizer = "75494";
  n.session = "firstreal";
  n.kind = PackageKind::kDemonstration;
  n.side = Side::kLeft;
  // A real iPhone package name from data/raw/2026-10-01/.
  EXPECT_EQ(n.str(), "2026-10-01T21-13-57.489Z_75494_firstreal_demonstration_left");
  n.kind = PackageKind::kGripperCalibration;
  n.side = Side::kRight;
  EXPECT_EQ(n.str(), "2026-10-01T21-13-57.489Z_75494_firstreal_grippercalibration_right");
  // No component may contain '_' (group.py:36-37 splits on '_' and needs exactly 5 parts).
  EXPECT_EQ(std::string(to_string(PackageKind::kRigCalibration)).find('_'), std::string::npos);
}

TEST(Shm, PodLayoutsAndImuRoundTrip) {
  EXPECT_EQ(sizeof(ShmImuSample), 72u);
  EXPECT_EQ(sizeof(ShmSlotHeader), 64u);
  EXPECT_EQ(alignof(ShmRingHeader), 64u);
  ImuSample s;
  s.t_ns = 123;
  s.chip_ticks = 456;
  s.gyro_rps = Eigen::Vector3d(0.1, -0.2, 0.3);
  s.acc_mps2 = Eigen::Vector3d(0.0, 0.0, kStandardGravity);
  s.temp_c = 31.5f;
  s.flags = kImuFlagAccSaturated;
  const ImuSample r = from_shm(to_shm(s));
  EXPECT_EQ(r.t_ns, s.t_ns);
  EXPECT_EQ(r.chip_ticks, s.chip_ticks);
  EXPECT_EQ(r.gyro_rps, s.gyro_rps);
  EXPECT_EQ(r.acc_mps2, s.acc_mps2);
  EXPECT_EQ(r.temp_c, s.temp_c);
  EXPECT_EQ(r.flags, s.flags);
}

TEST(Shm, SlamStatusDefaults) {
  const SlamStatus s = make_slam_status();
  EXPECT_EQ(s.tracking_state, static_cast<uint8_t>(TrackingState::kNotStarted));
  EXPECT_EQ(s.q[0], 1.0);
  EXPECT_EQ(s.t_ns, kInvalidTime);
  EXPECT_STREQ(to_string(AbortReason::kPoseJump), "pose_jump");
  EXPECT_STREQ(to_string(RecorderState::kArming), "arming");
}

TEST(Version, Generated) { EXPECT_NE(std::string(kSoftwareVersion).find("physicalSetup"), std::string::npos); }

}  // namespace
}  // namespace rig
