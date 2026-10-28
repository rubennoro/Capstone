// SPDX-License-Identifier: MIT
// ec-cam-probe analysis and H2 checks.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <map>
#include <memory>

#include "rig/drivers/camera/probe_report.hpp"

namespace rig {
namespace {

ImageFrame frame(uint64_t seq, TimeNs sof, int64_t exposure = 8'000'000, double gain = 2.0) {
  ImageFrame f;
  f.seq = seq;
  f.t_sof_ns = sof;
  f.t_mid_ns = sof + exposure / 2 + 5'000'000;
  f.t_arrival_ns = sof + 30'000'000;
  f.meta.exposure_ns = exposure;
  f.meta.analog_gain = gain;
  f.meta.digital_gain = 1.0;
  f.meta.readout_ns = 10'000'000;
  return f;
}

CameraStats ok_stats() {
  CameraStats s;
  s.focus_status = FocusStatus::kLockedVcmI2c;
  s.focus_position = 512;
  s.time_source = "argus:test";
  return s;
}

ProbeCriteria crit() {
  ProbeCriteria c;
  c.expected_fps = 60;
  c.expected_exposure_ns = 8'000'000;
  c.expected_gain = 2.0;
  return c;
}

TEST(ProbeReport, AllPass) {
  ProbeAnalyzer a("main", 60);
  for (uint64_t k = 0; k < 600; ++k) {
    const TimeNs jitter = (k % 2) ? 300'000 : -300'000;
    a.add(frame(k, 1'000'000'000 + static_cast<TimeNs>(k) * 16'666'667 + jitter));
  }
  const ProbeReport r = a.finish(ok_stats(), crit());
  EXPECT_TRUE(r.pass()) << format_probe_report(r);
  EXPECT_EQ(r.frames, 600u);
  EXPECT_NEAR(r.fps, 60.0, 0.01);
  EXPECT_NEAR(r.jitter_ns.p99, 600'000, 1000);
  EXPECT_NEAR(r.latency_ns.p50, 30e6, 1);
  EXPECT_NEAR(r.mid_minus_sof_ns.p50, 9e6, 1);
  EXPECT_EQ(r.exposure_min_ns, 8'000'000);
  const std::string txt = format_probe_report(r);
  EXPECT_NE(txt.find("camera main: PASS"), std::string::npos) << txt;
  EXPECT_NE(txt.find("[PASS] exposure_locked"), std::string::npos);
}

TEST(ProbeReport, Failures) {
  ProbeAnalyzer a("uw", 30);
  for (uint64_t k = 0; k < 100; ++k) {
    if (k == 50) continue;  // a dropped frame (seq gap)
    a.add(frame(k, static_cast<TimeNs>(k) * 33'333'333 + ((k % 10 == 0) ? 4'000'000 : 0),
                k < 20 ? 3'000'000 : 4'000'000, 4.0));  // exposure not locked for the first 20 frames
  }
  CameraStats s = ok_stats();
  s.drops_seq_gap = 1;
  s.focus_status = FocusStatus::kUnknown;
  ProbeCriteria c = crit();
  c.expected_fps = 30;
  c.expected_exposure_ns = 4'000'000;
  c.expected_gain = 4.0;
  const ProbeReport r = a.finish(s, c);
  EXPECT_FALSE(r.pass());
  EXPECT_EQ(r.seq_missing, 1u);
  std::map<std::string, bool> by;
  for (const auto& ch : r.checks) by[ch.name] = ch.pass;
  EXPECT_FALSE(by.at("fps"));  // a missing frame over 3.3 s lowers the rate by 1 %
  EXPECT_FALSE(by.at("jitter_p99"));
  EXPECT_FALSE(by.at("drops"));
  EXPECT_FALSE(by.at("exposure_locked"));
  EXPECT_TRUE(by.at("gain_locked"));
  EXPECT_FALSE(by.at("focus"));
  EXPECT_TRUE(by.at("latency"));
}

TEST(ProbeReport, NegativeLatencyFails) {
  ProbeAnalyzer a("x", 30);
  for (uint64_t k = 0; k < 10; ++k) {
    ImageFrame f = frame(k, static_cast<TimeNs>(k) * 33'333'333);
    f.t_arrival_ns = f.t_sof_ns - 1;
    a.add(f);
  }
  ProbeCriteria c;
  c.expected_fps = 30;
  c.require_focus_known = false;
  const ProbeReport r = a.finish(ok_stats(), c);
  EXPECT_FALSE(r.pass());
  EXPECT_EQ(r.negative_latency, 10u);
}

TEST(ProbeReport, Sharpness) {
  const int w = 64, h = 48;
  auto make = [&](bool checker) {
    std::shared_ptr<uint8_t[]> buf(new uint8_t[w * h * 3 / 2]);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) buf[y * w + x] = checker ? (((x / 2 + y / 2) & 1) ? 250 : 5) : 128;
    std::memset(buf.get() + w * h, 128, w * h / 2);
    ImageFrame f;
    f.width = w;
    f.height = h;
    f.fmt = PixelFormat::kNV12;
    f.planes[0] = Plane{0, static_cast<std::size_t>(w), w, h};
    f.planes[1] = Plane{static_cast<std::size_t>(w * h), static_cast<std::size_t>(w), w / 2, h / 2};
    f.data = buf;
    f.size = static_cast<std::size_t>(w * h * 3 / 2);
    f.t_mid_ns = 1;
    return f;
  };
  EXPECT_GT(frame_sharpness(make(true)), 1000.0);
  EXPECT_NEAR(frame_sharpness(make(false)), 0.0, 1e-9);
  EXPECT_TRUE(std::isnan(frame_sharpness(ImageFrame{})));
}

}  // namespace
}  // namespace rig
