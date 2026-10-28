// SPDX-License-Identifier: MIT
// ec-live (L1): the program end to end on the laptop (mock cameras + simulated IMU): headless snapshot, the
// no-display fallback, partial start failures (+ --strict), a stalled camera, IMU-only, SIGINT, and no /dev/shm
// segments left behind.
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../../episode_collection/apps/ec-live/live_view.hpp"
#include "rig/config/rig_config.hpp"

namespace fs = std::filesystem;

namespace {

const char* kRigYaml = RIG_SOURCE_DIR "/episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml";

std::string binary() { return (fs::path(RIG_TEST_TMP_ROOT) / ".." / "episode_collection" / "ec-live").string(); }

struct RunResult {
  int code = -1;
  std::string out;
};

/// Runs a shell command (stdout + stderr captured), bounded by `timeout 60`.
RunResult run(const std::string& args, const std::string& env_prefix = "") {
  const std::string cmd = env_prefix + " timeout -s INT 60 " + binary() + " " + args + " 2>&1";
  RunResult r;
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return r;
  std::array<char, 4096> buf{};
  while (fgets(buf.data(), buf.size(), p)) r.out += buf.data();
  const int st = pclose(p);
  r.code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
  return r;
}

/// Segments a rig program with the example config would create: "<shm.prefix>.<stream>" = /dev/shm/rig.*. (Other
/// test areas use their own prefixes, e.g. rigtest.*, and may run concurrently under ctest -j.)
std::set<std::string> rig_shm_segments() {
  std::set<std::string> s;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator("/dev/shm", ec))
    if (e.path().filename().string().rfind("rig.", 0) == 0) s.insert(e.path().filename().string());
  return s;
}

fs::path tmp_dir(const std::string& name) {
  const fs::path d = fs::path(RIG_TEST_TMP_ROOT) / ("live_" + name + "_" + std::to_string(getpid()));
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}

/// Writes a variant of the example rig.yaml.
template <class F>
std::string write_config(const fs::path& dir, F&& edit) {
  auto rc = rig::load_rig_config(kRigYaml);
  EXPECT_TRUE(rc.ok()) << rc.status().to_string();
  rig::RigConfig c = *rc;
  edit(c);
  const fs::path p = dir / "rig.yaml";
  std::ofstream(p) << rig::to_yaml(c);
  return p.string();
}

void mock_cameras(rig::RigConfig& c) {
  for (auto& kv : c.cameras) kv.second.driver.backend = rig::CameraBackendKind::kMock;
}

double region_mean(const cv::Mat& img, const cv::Rect& r) {
  const cv::Scalar m = cv::mean(img(r));
  return (m[0] + m[1] + m[2]) / 3;
}

class EcLive : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(fs::exists(binary())) << binary() << " not built";
    shm_before_ = rig_shm_segments();
  }
  void TearDown() override {
    // Each test runs in its own process (gtest_discover_tests), so this removes exactly this test's dirs.
    std::error_code ec;
    const std::string suffix = "_" + std::to_string(getpid());
    for (const auto& e : fs::directory_iterator(RIG_TEST_TMP_ROOT, ec)) {
      const std::string n = e.path().filename().string();
      if (n.rfind("live_", 0) == 0 && n.size() > suffix.size() &&
          n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0)
        fs::remove_all(e.path(), ec);
    }
    const std::set<std::string> after = rig_shm_segments();
    for (const auto& s : after) EXPECT_TRUE(shm_before_.count(s)) << "leftover /dev/shm/" << s;
  }
  std::set<std::string> shm_before_;
};

}  // namespace

TEST_F(EcLive, MockHeadlessSnapshot) {
  const fs::path d = tmp_dir("snap");
  const std::string png = (d / "snap.png").string();
  const RunResult r = run(std::string("--config ") + kRigYaml + " --mock --headless-snapshot " + png + " --seconds 3");
  ASSERT_EQ(r.code, 0) << r.out;
  EXPECT_NE(r.out.find("[ OK ] imu imu0 (sim)"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("[ OK ] camera main (mock"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("[ OK ] camera uw (mock"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("| main "), std::string::npos) << r.out;  // status line once per second
  EXPECT_NE(r.out.find("headless snapshot"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("stopped cleanly"), std::string::npos) << r.out;
  const cv::Mat img = cv::imread(png, cv::IMREAD_COLOR);
  ASSERT_FALSE(img.empty()) << png;
  const eclive::Layout L = eclive::compute_layout(eclive::RenderOptions{}, {{1920, 1080}, {1640, 1232}}, true);
  ASSERT_EQ(img.size(), L.size);
  ASSERT_EQ(L.images.size(), 2u);
  for (int i = 0; i < 2; ++i) EXPECT_GT(region_mean(img, L.images[i]), 40.0) << "tile " << i << " is black";
  cv::Scalar m, sd;
  cv::meanStdDev(img(L.imu), m, sd);
  EXPECT_GT(m[0], 10.0);
  EXPECT_GT(sd[0] + sd[1] + sd[2], 10.0) << "IMU panel is a flat colour";
}

TEST_F(EcLive, ImuOnlyAndOneCamera) {
  const fs::path d = tmp_dir("subsets");
  {
    const std::string png = (d / "imu.png").string();
    const RunResult r = run(std::string("--config ") + kRigYaml + " --mock --no-cameras --headless-snapshot " + png +
                      " --seconds 2");
    ASSERT_EQ(r.code, 0) << r.out;
    EXPECT_EQ(r.out.find("camera main"), std::string::npos) << r.out;
    const cv::Mat img = cv::imread(png);
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(img.size(), eclive::compute_layout(eclive::RenderOptions{}, {}, true).size);
  }
  {
    const std::string png = (d / "uw.png").string();
    const RunResult r = run(std::string("--config ") + kRigYaml + " --mock --camera uw --no-imu --width 800 "
                      "--headless-snapshot " + png + " --seconds 2");
    ASSERT_EQ(r.code, 0) << r.out;
    EXPECT_EQ(r.out.find("imu imu0"), std::string::npos) << r.out;
    const cv::Mat img = cv::imread(png);
    ASSERT_FALSE(img.empty());
    eclive::RenderOptions o;
    o.width = 800;
    const eclive::Layout L = eclive::compute_layout(o, {{1640, 1232}}, false);
    ASSERT_EQ(img.size(), L.size);
    EXPECT_GT(region_mean(img, L.images[0]), 40.0);
  }
  EXPECT_EQ(run(std::string("--config ") + kRigYaml + " --camera nope --no-window --seconds 1").code, 2);
  EXPECT_EQ(run(std::string("--config ") + kRigYaml + " --no-cameras --no-imu --no-window").code, 2);
}

TEST_F(EcLive, NoDisplayFallsBackToTerminal) {
  const RunResult r = run(std::string("--config ") + kRigYaml + " --mock --seconds 2.5", "env -u DISPLAY -u WAYLAND_DISPLAY");
  ASSERT_EQ(r.code, 0) << r.out;
  EXPECT_NE(r.out.find("no display: $DISPLAY is not set"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("terminal-only mode"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("[    1s] | main "), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("imu0 "), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("stopped cleanly"), std::string::npos) << r.out;
}

TEST_F(EcLive, PartialFailureContinuesAndStrictExits) {
  const fs::path d = tmp_dir("partial");
  // Cameras mock; the IMU on an I2C bus that does not exist.
  const std::string cfg = write_config(d, [](rig::RigConfig& c) {
    mock_cameras(c);
    c.imu.driver.backend = rig::ImuBackendKind::kLsm6dsoxI2c;
    c.imu.driver.i2c_bus = 99;
  });
  const std::string png = (d / "p.png").string();
  RunResult r = run("--config " + cfg + " --headless-snapshot " + png + " --seconds 2");
  ASSERT_EQ(r.code, 0) << r.out;
  EXPECT_NE(r.out.find("[FAIL] imu imu0 (lsm6dsox_i2c)"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("i2c-99"), std::string::npos) << r.out;  // the Status message names the bus
  EXPECT_NE(r.out.find("hint:"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("continuing with the sensors that started"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("imu0 FAILED"), std::string::npos) << r.out;  // in the status line
  EXPECT_TRUE(fs::exists(png));

  r = run("--config " + cfg + " --strict --no-window --seconds 2");
  EXPECT_EQ(r.code, 1) << r.out;
  EXPECT_NE(r.out.find("--strict: stopping"), std::string::npos) << r.out;

  // Nothing at all starts: exit 2 with every reason printed.
  const fs::path d2 = tmp_dir("none");
  const std::string none = write_config(d2, [](rig::RigConfig& c) {
    for (auto& kv : c.cameras) {
      kv.second.driver.backend = rig::CameraBackendKind::kReplay;
      kv.second.driver.replay.path = "/nonexistent/ec-live-test";
    }
    c.imu.driver.backend = rig::ImuBackendKind::kLsm6dsoxI2c;
    c.imu.driver.i2c_bus = 99;
  });
  r = run("--config " + none + " --no-window --seconds 1");
  EXPECT_EQ(r.code, 2) << r.out;
  EXPECT_NE(r.out.find("[FAIL] camera main (replay"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("[FAIL] camera uw (replay"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("nothing started"), std::string::npos) << r.out;
}

TEST_F(EcLive, StalledCameraShowsNoFrames) {
  const fs::path d = tmp_dir("stall");
  const std::string cfg = write_config(d, [](rig::RigConfig& c) {
    mock_cameras(c);
    c.imu.driver.backend = rig::ImuBackendKind::kSim;
    c.cameras.at("main").driver.mock.stall_after_frames = 5;
    c.cameras.at("main").driver.mock.stall_ns = 0;  // forever
  });
  const RunResult r = run("--config " + cfg + " --no-window --seconds 3.5");
  ASSERT_EQ(r.code, 0) << r.out;
  EXPECT_NE(r.out.find("main NO FRAMES for"), std::string::npos) << r.out;
  EXPECT_NE(r.out.find("| uw "), std::string::npos) << r.out;
}

TEST_F(EcLive, SigintStopsCleanly) {
  // Started in the background, interrupted after ~2 s; the exit status of ec-live itself is echoed.
  const std::string cmd = "sh -c '" + binary() + " --config " + kRigYaml +
                          " --mock --no-window & p=$!; sleep 2; kill -INT $p; wait $p; echo EXIT=$?' 2>&1";
  std::string out;
  FILE* p = popen(("timeout 60 " + cmd).c_str(), "r");
  ASSERT_NE(p, nullptr);
  std::array<char, 4096> buf{};
  while (fgets(buf.data(), buf.size(), p)) out += buf.data();
  pclose(p);
  EXPECT_NE(out.find("signal received, stopping"), std::string::npos) << out;
  EXPECT_NE(out.find("stopped cleanly"), std::string::npos) << out;
  EXPECT_NE(out.find("EXIT=0"), std::string::npos) << out;
}
