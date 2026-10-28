// SPDX-License-Identifier: MIT
// Imx519Focus: AK7375 register sequence (Arducam Focuser.py), fallback chain, sysfs detection (fake trees).
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

#include "rig/drivers/camera/imx519_focus.hpp"

namespace rig {
namespace {

namespace fs = std::filesystem;

/// Register-level fake of the VCM on one bus. Records writes; reads return the register file.
struct FakeVcm {
  bool present = true;      // NACK everything when false
  bool readable = true;     // reads NACK when false
  std::map<uint8_t, uint8_t> regs;
  std::vector<std::pair<uint8_t, uint8_t>> writes;
};

class FakeBus final : public I2cBus {
 public:
  FakeBus(FakeVcm* vcm, uint8_t addr) : vcm_(vcm), addr_(addr) {}
  Status write_read(uint8_t a, const uint8_t* w, std::size_t wn, uint8_t* r, std::size_t rn) override {
    if (a != addr_ || !vcm_->present || !vcm_->readable) return device_error("NACK");
    if (wn != 1) return invalid_argument("bad");
    for (std::size_t i = 0; i < rn; ++i) r[i] = vcm_->regs[static_cast<uint8_t>(w[0] + i)];
    return Status::Ok();
  }
  Status write(uint8_t a, const uint8_t* w, std::size_t wn) override {
    if (a != addr_ || !vcm_->present) return device_error("NACK (EREMOTEIO)");
    if (wn != 2) return invalid_argument("bad");
    vcm_->regs[w[0]] = w[1];
    vcm_->writes.emplace_back(w[0], w[1]);
    return Status::Ok();
  }
  std::string name() const override { return "fake"; }

 private:
  FakeVcm* vcm_;
  uint8_t addr_;
};

struct FakeEnv final : public FocusEnvironment {
  FakeVcm* vcm = nullptr;
  bool bound = false;
  bool bus_exists = true;
  std::vector<std::string> nodes;
  std::map<std::string, V4l2ControlRange> controls;  // nodes with focus_absolute
  std::map<std::string, int> values;
  bool v4l2_set_fails = false;

  bool i2c_address_kernel_bound(int, int, std::string* d) override {
    if (bound && d) *d = "sysfs driver ak7375";
    return bound;
  }
  Result<std::unique_ptr<I2cBus>> open_i2c_bus(int bus) override {
    if (!bus_exists) return not_found("/dev/i2c-" + std::to_string(bus) + " missing");
    return std::unique_ptr<I2cBus>(std::make_unique<FakeBus>(vcm, 0x0C));
  }
  std::vector<std::string> v4l2_focus_candidates(const FocusConfig&) override { return nodes; }
  Result<V4l2ControlRange> v4l2_query_focus(const std::string& n) override {
    auto it = controls.find(n);
    if (it == controls.end()) return not_found(n + ": no focus_absolute");
    return it->second;
  }
  Status v4l2_set_focus(const std::string& n, int v) override {
    if (v4l2_set_fails) return device_error("EINVAL");
    values[n] = v;
    return Status::Ok();
  }
  Result<int> v4l2_get_focus(const std::string& n) override {
    auto it = values.find(n);
    if (it == values.end()) return device_error("no value");
    return it->second;
  }
};

FocusConfig fixed(int pos = 512) {
  FocusConfig c;
  c.mode = FocusMode::kFixed;
  c.position = pos;
  c.i2c_bus = 9;
  c.i2c_addr = 0x0C;
  c.allow_v4l2_fallback = true;
  return c;
}

TEST(Focus, Ak7375SequenceMatchesFocuserPy) {
  // Focuser.py: value <<= 4; reg0 = value >> 8; reg1 = value & 0xFF; preceded by init reg2 = 0.
  for (int dac : {0, 1, 512, 2047, 4095}) {
    const auto s = ak7375_write_sequence(dac);
    const int v = dac << 4;
    EXPECT_EQ(s[0].reg, 0x02);
    EXPECT_EQ(s[0].value, 0x00);
    EXPECT_EQ(s[1].reg, 0x00);
    EXPECT_EQ(s[1].value, (v >> 8) & 0xFF);
    EXPECT_EQ(s[2].reg, 0x01);
    EXPECT_EQ(s[2].value, v & 0xFF);
  }
  const auto s = ak7375_write_sequence(512);  // 0x200 << 4 = 0x2000
  EXPECT_EQ(s[1].value, 0x20);
  EXPECT_EQ(s[2].value, 0x00);
  EXPECT_EQ(ak7375_write_sequence(5000)[1].value, 0xFF);  // clamped to 4095 -> 0xFFF0
  EXPECT_EQ(ak7375_write_sequence(5000)[2].value, 0xF0);
  EXPECT_EQ(ak7375_write_sequence(-3)[1].value, 0x00);
}

TEST(Focus, FocuserScale) {
  EXPECT_EQ(focuser_value_to_dac(0), 0);
  EXPECT_EQ(focuser_value_to_dac(1000), 4095);
  EXPECT_EQ(focuser_value_to_dac(500), 2047);  // int(0.5 * 4095) = 2047
  EXPECT_EQ(focuser_value_to_dac(-5), 0);
  EXPECT_EQ(focuser_value_to_dac(2000), 4095);
}

TEST(Focus, V4l2Scaling) {
  EXPECT_EQ(dac_to_v4l2_value(512, {0, 4095, 1, 0}), 512);
  EXPECT_EQ(dac_to_v4l2_value(4095, {0, 1023, 1, 0}), 1023);
  EXPECT_EQ(dac_to_v4l2_value(2048, {0, 1023, 1, 0}), 512);
  EXPECT_EQ(dac_to_v4l2_value(2048, {0, 1000, 10, 0}), 500);
  EXPECT_EQ(dac_to_v4l2_value(100, {5, 5, 1, 0}), 5);
}

TEST(Focus, Validate) {
  EXPECT_TRUE(Imx519Focus::validate(FocusConfig{}).ok());
  EXPECT_FALSE(Imx519Focus::validate(fixed(4096)).ok());
  EXPECT_FALSE(Imx519Focus::validate(fixed(-1)).ok());
  FocusConfig c = fixed();
  c.i2c_bus = -1;
  c.allow_v4l2_fallback = false;
  EXPECT_FALSE(Imx519Focus::validate(c).ok());
  c = fixed();
  c.i2c_addr = 0x80;
  EXPECT_FALSE(Imx519Focus::validate(c).ok());
}

TEST(Focus, NoneIsNotApplicable) {
  auto env = std::make_unique<FakeEnv>();
  Imx519Focus f(FocusConfig{}, std::move(env));
  EXPECT_EQ(f.apply().status, FocusStatus::kNotApplicable);
}

TEST(Focus, VcmI2cWriteWithReadback) {
  FakeVcm vcm;
  auto env = std::make_unique<FakeEnv>();
  env->vcm = &vcm;
  Imx519Focus f(fixed(1000), std::move(env));
  const FocusResult r = f.apply();
  EXPECT_EQ(r.status, FocusStatus::kLockedVcmI2c);
  EXPECT_EQ(r.position, 1000);
  EXPECT_TRUE(r.readback_verified);
  ASSERT_EQ(vcm.writes.size(), 3u);
  // Order matters: control first (Focuser.py init), then MSB, then LSB. 1000 << 4 = 0x3E80.
  EXPECT_EQ(vcm.writes[0], std::make_pair(uint8_t{0x02}, uint8_t{0x00}));
  EXPECT_EQ(vcm.writes[1], std::make_pair(uint8_t{0x00}, uint8_t{0x3E}));
  EXPECT_EQ(vcm.writes[2], std::make_pair(uint8_t{0x01}, uint8_t{0x80}));
  EXPECT_NE(r.detail.find("readback 0x3e80 matches"), std::string::npos) << r.detail;
}

TEST(Focus, VcmWithoutReadSupportStillLocks) {
  FakeVcm vcm;
  vcm.readable = false;
  auto env = std::make_unique<FakeEnv>();
  env->vcm = &vcm;
  const FocusResult r = Imx519Focus(fixed(), std::move(env)).apply();
  EXPECT_EQ(r.status, FocusStatus::kLockedVcmI2c);
  EXPECT_FALSE(r.readback_verified);
  EXPECT_NE(r.detail.find("readback unavailable"), std::string::npos);
}

TEST(Focus, KernelBoundUsesV4l2) {
  FakeVcm vcm;
  auto env = std::make_unique<FakeEnv>();
  FakeEnv* e = env.get();
  env->vcm = &vcm;
  env->bound = true;
  env->nodes = {"/dev/v4l-subdev0", "/dev/v4l-subdev1"};
  env->controls["/dev/v4l-subdev1"] = V4l2ControlRange{0, 1023, 1, 0};
  Imx519Focus focus(fixed(2048), std::move(env));  // keeps `e` alive
  const FocusResult r = focus.apply();
  EXPECT_EQ(r.status, FocusStatus::kLockedV4l2);
  EXPECT_EQ(r.position, 512);
  EXPECT_TRUE(r.readback_verified);
  EXPECT_EQ(e->values["/dev/v4l-subdev1"], 512);
  EXPECT_TRUE(vcm.writes.empty());  // never touch a kernel-owned address
  EXPECT_NE(r.detail.find("kernel-bound (UU"), std::string::npos) << r.detail;
}

TEST(Focus, KernelBoundNoV4l2IsFixedAtPowerUp) {
  auto env = std::make_unique<FakeEnv>();
  env->bound = true;
  env->nodes = {"/dev/v4l-subdev0"};
  const FocusResult r = Imx519Focus(fixed(), std::move(env)).apply();
  EXPECT_EQ(r.status, FocusStatus::kFixedAtPowerUp);
  EXPECT_EQ(r.position, -1);
  EXPECT_NE(r.detail.find("fixed-at-power-up"), std::string::npos);
  EXPECT_NE(r.detail.find("no focus_absolute control on /dev/v4l-subdev0"), std::string::npos) << r.detail;
}

TEST(Focus, KernelBoundFallbackDisabled) {
  auto env = std::make_unique<FakeEnv>();
  env->bound = true;
  FocusConfig c = fixed();
  c.allow_v4l2_fallback = false;
  const FocusResult r = Imx519Focus(c, std::move(env)).apply();
  EXPECT_EQ(r.status, FocusStatus::kFixedAtPowerUp);
  EXPECT_NE(r.detail.find("V4L2 fallback disabled"), std::string::npos);
}

TEST(Focus, MissingBusFallsBack) {
  auto env = std::make_unique<FakeEnv>();
  env->bus_exists = false;
  const FocusResult r = Imx519Focus(fixed(), std::move(env)).apply();
  EXPECT_EQ(r.status, FocusStatus::kFixedAtPowerUp);
  EXPECT_NE(r.detail.find("cannot open bus"), std::string::npos);
}

TEST(Focus, NackIsFailedUnlessV4l2Works) {
  FakeVcm vcm;
  vcm.present = false;
  {
    auto env = std::make_unique<FakeEnv>();
    env->vcm = &vcm;
    const FocusResult r = Imx519Focus(fixed(), std::move(env)).apply();
    EXPECT_EQ(r.status, FocusStatus::kFailed);
    EXPECT_NE(r.detail.find("write failed"), std::string::npos);
  }
  {
    auto env = std::make_unique<FakeEnv>();
    env->vcm = &vcm;
    env->nodes = {"/dev/video0"};
    env->controls["/dev/video0"] = V4l2ControlRange{0, 4095, 1, 0};
    EXPECT_EQ(Imx519Focus(fixed(), std::move(env)).apply().status, FocusStatus::kLockedV4l2);
  }
  {
    auto env = std::make_unique<FakeEnv>();
    env->vcm = &vcm;
    env->nodes = {"/dev/video0"};
    env->controls["/dev/video0"] = V4l2ControlRange{0, 4095, 1, 0};
    env->v4l2_set_fails = true;
    EXPECT_EQ(Imx519Focus(fixed(), std::move(env)).apply().status, FocusStatus::kFailed);
  }
}

class SysfsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Unique per test: ctest runs every gtest case as its own process, in parallel.
    root_ = fs::path(RIG_TEST_TMP_ROOT) /
            ("t2_focus_sysroot_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::remove_all(root_);
  }
  void TearDown() override { fs::remove_all(root_); }
  void v4l(const std::string& node, const std::string& name) {
    fs::create_directories(root_ / "sys/class/video4linux" / node);
    std::ofstream(root_ / "sys/class/video4linux" / node / "name") << name << "\n";
  }
  fs::path root_;
};

TEST_F(SysfsTest, KernelBindingFromSysfs) {
  EXPECT_EQ(i2c_sysfs_driver(root_.string(), 9, 0x0C), "");
  const fs::path dev = root_ / "sys/bus/i2c/devices/9-000c";
  fs::create_directories(dev);
  EXPECT_EQ(i2c_sysfs_driver(root_.string(), 9, 0x0C), "");  // device without driver: not bound
  fs::create_directories(root_ / "sys/bus/i2c/drivers/ak7375");
  fs::create_symlink("../../../../bus/i2c/drivers/ak7375", dev / "driver");
  EXPECT_EQ(i2c_sysfs_driver(root_.string(), 9, 0x0C), "ak7375");
  EXPECT_EQ(i2c_sysfs_driver(root_.string(), 10, 0x0C), "");
  auto env = make_linux_focus_environment(root_.string());
  std::string why;
  EXPECT_TRUE(env->i2c_address_kernel_bound(9, 0x0C, &why));
  EXPECT_NE(why.find("ak7375"), std::string::npos);
  EXPECT_FALSE(env->i2c_address_kernel_bound(10, 0x0C, &why));
}

TEST_F(SysfsTest, V4l2CandidatesOrdered) {
  v4l("video0", "vi-output, imx519 9-001a");
  v4l("video1", "vi-output, imx219 10-0010");
  v4l("v4l-subdev0", "imx519 9-001a");
  v4l("v4l-subdev1", "ak7375 9-000c");
  v4l("v4l-subdev2", "imx219 10-0010");
  v4l("v4l-subdev3", "imx519 10-001a");
  const auto c = find_v4l2_focus_candidates(root_.string(), 9, 0x0C);
  const std::string d = root_.string() + "/dev/";
  ASSERT_EQ(c.size(), 3u);
  EXPECT_EQ(c[0], d + "v4l-subdev1");  // lens first
  EXPECT_EQ(c[1], d + "v4l-subdev0");  // then the sensor subdev on bus 9
  EXPECT_EQ(c[2], d + "video0");       // then the video node
  EXPECT_TRUE(find_v4l2_focus_candidates((root_ / "none").string(), 9, 0x0C).empty());
  // Explicit subdev overrides discovery.
  auto env = make_linux_focus_environment(root_.string());
  FocusConfig fc = fixed();
  fc.v4l2_subdev = "/dev/v4l-subdev7";
  EXPECT_EQ(env->v4l2_focus_candidates(fc), std::vector<std::string>{"/dev/v4l-subdev7"});
  // A node that does not exist has no control.
  EXPECT_EQ(env->v4l2_query_focus(d + "v4l-subdev1").status().code(), StatusCode::kNotFound);
}

}  // namespace
}  // namespace rig
