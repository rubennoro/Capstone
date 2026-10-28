// SPDX-License-Identifier: MIT
// IMX519 focus lock: fallback chain and AK7375 register sequence (rig/drivers/camera/imx519_focus.hpp, sources there).
#include "rig/drivers/camera/imx519_focus.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <spdlog/spdlog.h>

namespace rig {

std::array<VcmRegWrite, 3> ak7375_write_sequence(int dac) noexcept {
  const int d = std::clamp(dac, 0, kVcmDacMax);
  const int v = d << 4;  // Focuser.py:33 `value <<= 4`
  return {{
      {0x02, 0x00},                                     // Focuser.py:30 init: control = active
      {0x00, static_cast<uint8_t>((v >> 8) & 0xFF)},    // Focuser.py:34 MSB
      {0x01, static_cast<uint8_t>(v & 0xFF)},           // Focuser.py:35 LSB
  }};
}

int focuser_value_to_dac(int value) noexcept {
  const int v = std::clamp(value, 0, 1000);  // Focuser.py:66-70 MIN 0 / MAX 1000; :51-52 negative -> 0
  return static_cast<int>(v / 1000.0 * 4095);  // Focuser.py:55 (int() truncates toward zero)
}

int dac_to_v4l2_value(int dac, const V4l2ControlRange& r) noexcept {
  const int d = std::clamp(dac, 0, kVcmDacMax);
  if (r.minimum == 0 && r.maximum == kVcmDacMax) return d;
  if (r.maximum <= r.minimum) return r.minimum;
  const double span = static_cast<double>(r.maximum) - static_cast<double>(r.minimum);
  double v = static_cast<double>(r.minimum) + span * static_cast<double>(d) / kVcmDacMax;
  const int step = r.step > 0 ? r.step : 1;
  v = r.minimum + std::round((v - r.minimum) / step) * step;
  return static_cast<int>(std::clamp(v, static_cast<double>(r.minimum), static_cast<double>(r.maximum)));
}

Status Imx519Focus::validate(const FocusConfig& c) {
  if (c.i2c_addr < 0 || c.i2c_addr > 0x7F) return invalid_argument("i2c_addr: must be a 7-bit address (0..0x7F)");
  if (c.mode == FocusMode::kNone) return Status::Ok();
  if (c.position < 0 || c.position > kVcmDacMax)
    return invalid_argument("position: " + std::to_string(c.position) + " outside the 12-bit VCM DAC range 0..4095");
  if (c.i2c_bus < 0 && !c.allow_v4l2_fallback)
    return invalid_argument("i2c_bus: required for mode fixed unless allow_v4l2_fallback is true");
  return Status::Ok();
}

Imx519Focus::Imx519Focus(FocusConfig config, std::unique_ptr<FocusEnvironment> env)
    : config_(std::move(config)), env_(env ? std::move(env) : make_linux_focus_environment()) {}

Imx519Focus::~Imx519Focus() = default;

FocusResult Imx519Focus::apply() {
  FocusResult r;
  if (config_.mode == FocusMode::kNone) {
    r.status = FocusStatus::kNotApplicable;
    r.detail = "fixed-focus lens (focus.mode none)";
    result_ = r;
    return r;
  }
  if (Status s = validate(config_); !s.ok()) {
    r.status = FocusStatus::kFailed;
    r.detail = "invalid focus config: " + s.message();
    result_ = r;
    return r;
  }
  char where[64];
  std::snprintf(where, sizeof(where), "i2c-%d addr 0x%02x", config_.i2c_bus, config_.i2c_addr);
  const auto addr = static_cast<uint8_t>(config_.i2c_addr);
  bool write_failed = false;
  std::string why;

  // Step 2/3: userspace VCM write unless kernel-bound.
  if (config_.i2c_bus < 0) {
    why = "no i2c_bus configured";
  } else if (std::string bound; env_->i2c_address_kernel_bound(config_.i2c_bus, config_.i2c_addr, &bound)) {
    why = std::string(where) + " is kernel-bound (UU: " + bound + ")";
  } else {
    Result<std::unique_ptr<I2cBus>> bus = env_->open_i2c_bus(config_.i2c_bus);
    if (!bus.ok()) {
      why = std::string(where) + ": cannot open bus: " + bus.status().message();
    } else {
      Status ws;
      for (const VcmRegWrite& w : ak7375_write_sequence(config_.position)) {
        ws = (*bus)->write_reg(addr, w.reg, w.value);
        if (!ws.ok()) break;
      }
      if (ws.ok()) {
        r.status = FocusStatus::kLockedVcmI2c;
        r.position = config_.position;
        const auto seq = ak7375_write_sequence(config_.position);
        uint8_t hi = 0, lo = 0;
        const Status rs1 = (*bus)->read_regs(addr, 0x00, &hi, 1);
        const Status rs2 = rs1.ok() ? (*bus)->read_regs(addr, 0x01, &lo, 1) : rs1;
        char buf[200];
        if (rs2.ok()) {
          r.readback_verified = hi == seq[1].value && lo == seq[2].value;
          std::snprintf(buf, sizeof(buf), "VCM DAC %d written at %s; readback 0x%02x%02x %s", config_.position, where,
                        hi, lo, r.readback_verified ? "matches" : "DIFFERS (VCM may not support reads; check focus on H)");
        } else {
          std::snprintf(buf, sizeof(buf), "VCM DAC %d written at %s; readback unavailable (%s)", config_.position,
                        where, rs2.message().c_str());
        }
        r.detail = buf;
        result_ = r;
        spdlog::info("focus: {}", r.detail);
        return r;
      }
      write_failed = true;
      why = std::string(where) + ": write failed: " + ws.message();
    }
  }

  // Step 4: V4L2 focus_absolute.
  if (config_.allow_v4l2_fallback) {
    std::string tried;
    for (const std::string& node : env_->v4l2_focus_candidates(config_)) {
      Result<V4l2ControlRange> range = env_->v4l2_query_focus(node);
      if (!range.ok()) {
        tried += (tried.empty() ? "" : ", ") + node;
        continue;
      }
      const int value = dac_to_v4l2_value(config_.position, *range);
      if (Status s = env_->v4l2_set_focus(node, value); !s.ok()) {
        r.status = FocusStatus::kFailed;
        r.detail = why + "; V4L2 focus_absolute=" + std::to_string(value) + " on " + node + " failed: " + s.message();
        result_ = r;
        spdlog::error("focus: {}", r.detail);
        return r;
      }
      Result<int> back = env_->v4l2_get_focus(node);
      r.status = FocusStatus::kLockedV4l2;
      r.position = value;
      r.readback_verified = back.ok() && *back == value;
      r.detail = why + "; V4L2 focus_absolute=" + std::to_string(value) + " (DAC " + std::to_string(config_.position) +
                 ", range " + std::to_string(range->minimum) + ".." + std::to_string(range->maximum) + ") on " + node +
                 (r.readback_verified ? ", readback matches" : ", readback unavailable or different");
      result_ = r;
      spdlog::info("focus: {}", r.detail);
      return r;
    }
    why += tried.empty() ? "; no V4L2 focus node found" : "; no focus_absolute control on " + tried;
  } else {
    why += "; V4L2 fallback disabled";
  }

  // Step 5.
  r.status = write_failed ? FocusStatus::kFailed : FocusStatus::kFixedAtPowerUp;
  r.position = -1;
  r.detail = why + (write_failed ? " -> focus FAILED" : " -> focus: fixed-at-power-up");
  result_ = r;
  spdlog::warn("focus: {}", r.detail);
  return r;
}

}  // namespace rig
