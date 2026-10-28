// SPDX-License-Identifier: MIT
/// @file rig/drivers/camera/imx519_focus.hpp
/// @brief IMX519 lens focus lock (PLAN T2 `Imx519Focus`): one fixed VCM position written at start, never autofocus.
/// Owned by T2 (implementation in common/drivers/focus/).
///
/// ## VCM protocol (cited)
/// Arducam Jetson_IMX519_Focus_Example, Focuser.py (commit f0d9b29, MIT;
/// https://github.com/ArduCAM/Jetson_IMX519_Focus_Example/blob/f0d9b29afd25ba270d2a4e459715eecfba44df90/Focuser.py):
/// - :39 `CHIP_I2C_ADDR = 0x0C` (7-bit);
/// - :29-30 init(): `i2cset -y <bus> 0x0c 0x02 0x00` (control register 0x02 = 0x00: active mode);
/// - :55 user value v in 0..1000 (:66-70) -> DAC = int(v / 1000.0 * 4095), i.e. a 12-bit code 0..4095;
/// - :32-35 write(): value <<= 4; reg 0x00 = value >> 8; reg 0x01 = value & 0xFF (MSB first).
/// This is the AK7375 register layout (Linux drivers/media/i2c/ak7375.c: AK7375_REG_POSITION 0x00 written as a
/// 16-bit big-endian value << 4, AK7375_REG_CONT 0x02, AK7375_MODE_ACTIVE 0x00, max position 4095).
/// `FocusConfig::position` is that 12-bit DAC code (a Focuser.py value v corresponds to DAC int(v/1000*4095)). Which
/// end of travel is near/far is not documented by Arducam: choose the position on day H with
/// `ec-cam-probe --focus-sweep` (BRINGUP 10).
///
/// ## Power
/// The VCM is powered with the sensor (the module regulators are on only while the sensor streams), so the
/// backends call apply() AFTER the first frame and let it settle (capture_plan.hpp kSettleNs); a write while the
/// sensor is off NACKs or is lost. Arducam's FocuserExample.py likewise starts the camera before focusing.
///
/// ## Fallback chain (PLAN T2; Arducam forum 8962: 0x0C shows `UU` on JP 6.2 / L4T 36.4.3)
/// 1. mode none -> kNotApplicable.
/// 2. If the address is kernel-bound (sysfs <bus>-000c/driver exists, or ioctl(I2C_SLAVE) -> EBUSY, which is what
///    i2cdetect prints as `UU`), userspace must not write it: go to 4.
/// 3. Otherwise write the sequence through I2cBus; read back 0x00/0x01 (verification is best effort: some VCMs
///    do not support reads). Success -> kLockedVcmI2c. A write error -> step 4, else kFailed.
/// 4. If allow_v4l2_fallback: V4L2_CID_FOCUS_ABSOLUTE on `v4l2_subdev` or the auto-detected lens/sensor subdev
///    or video node -> kLockedV4l2 (value scaled linearly if the control's range is not 0..4095).
/// 5. Otherwise kFixedAtPowerUp after a bound address / missing bus, kFailed after a failed write.
/// The result never fails start(): it is recorded per frame (FrameMeta) and in stats; ec-cam-probe / ec-doctor
/// report it.
/// Threading: not thread-safe; used once from the backend's start().
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rig/drivers/camera_backend.hpp"
#include "rig/drivers/i2c_bus.hpp"
#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

inline constexpr int kVcmDacMax = 4095;  ///< 12-bit (Focuser.py:55, ak7375.c).

/// One 8-bit register write.
struct VcmRegWrite {
  uint8_t reg;
  uint8_t value;
};

/// The write sequence for `dac` (clamped to 0..4095): {0x02,0x00}, {0x00, dac>>4}, {0x01, (dac&0xF)<<4}.
std::array<VcmRegWrite, 3> ak7375_write_sequence(int dac) noexcept;

/// Focuser.py user scale (0..1000) -> DAC code, exactly int(v / 1000.0 * 4095) with v clamped to 0..1000.
int focuser_value_to_dac(int value) noexcept;

/// Range of a V4L2 integer control.
struct V4l2ControlRange {
  int minimum = 0;
  int maximum = 0;
  int step = 1;
  int default_value = 0;
};

/// Maps a DAC code onto a V4L2 control range (identity for 0..4095; otherwise linear and step-aligned).
int dac_to_v4l2_value(int dac, const V4l2ControlRange& r) noexcept;

/// Operating-system access used by the focus logic; replaced by a fake in tests.
class FocusEnvironment {
 public:
  virtual ~FocusEnvironment() = default;
  /// True if a kernel driver owns bus/addr (`UU`). `detail` receives e.g. "sysfs 9-000c/driver -> ak7375".
  virtual bool i2c_address_kernel_bound(int bus, int addr7, std::string* detail) = 0;
  virtual Result<std::unique_ptr<I2cBus>> open_i2c_bus(int bus) = 0;
  /// Device nodes to try for V4L2 focus_absolute, best first.
  virtual std::vector<std::string> v4l2_focus_candidates(const FocusConfig& config) = 0;
  /// kNotFound if the node has no focus_absolute control.
  virtual Result<V4l2ControlRange> v4l2_query_focus(const std::string& node) = 0;
  virtual Status v4l2_set_focus(const std::string& node, int value) = 0;
  virtual Result<int> v4l2_get_focus(const std::string& node) = 0;
};

/// Linux implementation. `sysroot` ("" = real system) prefixes /sys and /dev paths for tests; the I2C_SLAVE EBUSY
/// probe is done only on the real system (sysroot empty).
std::unique_ptr<FocusEnvironment> make_linux_focus_environment(const std::string& sysroot = "");

/// Pure sysfs helpers (testable on fake trees).
/// Driver bound to <sysroot>/sys/bus/i2c/devices/<bus>-00<addr>, or "" if none.
std::string i2c_sysfs_driver(const std::string& sysroot, int bus, int addr7);
/// V4L2 nodes whose /sys/class/video4linux/<node>/name matches the camera on `bus`: lens subdevs ("ak7375", or an
/// i2c client "<bus>-000c") first, then the imx519 sensor subdev ("imx519 <bus>-001a"), then video nodes
/// ("vi-output, imx519 <bus>-001a"). Returns "<sysroot>/dev/<node>" paths.
std::vector<std::string> find_v4l2_focus_candidates(const std::string& sysroot, int bus, int addr7);

/// Outcome of a focus attempt.
struct FocusResult {
  FocusStatus status = FocusStatus::kUnknown;
  int position = -1;              ///< Value written (DAC code, or the V4L2 value); -1 if none.
  bool readback_verified = false; ///< VCM regs / V4L2 control read back equal to the write.
  std::string detail;             ///< One line for logs, events.csv and ec-cam-probe.
};

class Imx519Focus {
 public:
  /// `env` = nullptr -> make_linux_focus_environment().
  explicit Imx519Focus(FocusConfig config, std::unique_ptr<FocusEnvironment> env = nullptr);
  ~Imx519Focus();
  Imx519Focus(const Imx519Focus&) = delete;
  Imx519Focus& operator=(const Imx519Focus&) = delete;

  /// Runs the fallback chain (file comment) once and returns the result (also kept in result()).
  FocusResult apply();
  const FocusResult& result() const noexcept { return result_; }

  /// kInvalidArgument for mode fixed with position outside 0..4095 or i2c_bus < 0, or i2c_addr > 0x7F.
  static Status validate(const FocusConfig& config);

 private:
  FocusConfig config_;
  std::unique_ptr<FocusEnvironment> env_;
  FocusResult result_;
};

}  // namespace rig
