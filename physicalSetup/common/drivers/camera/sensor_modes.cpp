// SPDX-License-Identifier: MIT
// Sensor-mode tables and select_sensor_mode() (rig/drivers/camera/sensor_modes.hpp, rig/drivers/camera_backend.hpp).
// Sources for every number are in sensor_modes.hpp.
#include "rig/drivers/camera/sensor_modes.hpp"

#include <cmath>
#include <cstdio>

namespace rig {
namespace {

constexpr int64_t kExpMinNs = 13000;       // Argus mode dumps: "Exposure Range min 13000, max 683709000".
constexpr int64_t kExpMaxNs = 683709000;

SensorMode make_mode(int index, int w, int h, double max_fps, double min_fps, double max_gain, bool full_fov,
                     const char* note) {
  SensorMode m;
  m.index = index;
  m.width = w;
  m.height = h;
  m.min_fps = min_fps;
  m.max_fps = max_fps;
  m.min_exposure_ns = kExpMinNs;
  m.max_exposure_ns = kExpMaxNs;
  m.min_analog_gain = 1.0;
  m.max_analog_gain = max_gain;
  m.bit_depth = 10;
  m.full_fov = full_fov;
  m.description = note;
  return m;
}

// Minimum fps = 1e9 / max frame duration. The dumps do not list the max frame duration; the exposure maximum
// (683.709 ms) bounds it from below, so 1e9 / 683709000 = 1.4626 fps is used as the nominal minimum.
constexpr double kMinFps = 1e9 / static_cast<double>(kExpMaxNs);

}  // namespace

const std::vector<SensorMode>& known_sensor_modes(std::string_view sensor) {
  // imx219: NVIDIA L4T imx219 driver mode indices 0..4 (forum 246754 dump: FR 21 / 28.000001 / 29.999999 /
  // 29.999999 / 59.999999, analog gain 1..10.625). The table stores the dump's fps (not rounded) so the
  // tolerance logic is exercised exactly as on hardware.
  static const std::vector<SensorMode> kImx219 = {
      make_mode(0, 3280, 2464, 21.000000, kMinFps, 10.625, true, "imx219 full array"),
      make_mode(1, 3280, 1848, 28.000001, kMinFps, 10.625, false, "imx219 16:9 crop (full width)"),
      make_mode(2, 1920, 1080, 29.999999, kMinFps, 10.625, false, "imx219 1080p crop"),
      make_mode(3, 1640, 1232, 29.999999, kMinFps, 10.625, true, "imx219 2x2 binned full array"),
      make_mode(4, 1280, 720, 59.999999, kMinFps, 10.625, false, "imx219 720p crop"),
  };
  // imx519: Arducam Jetson driver on Orin Nano (forum 8481 dump: 9 / 17 / 59.999999 / 120 fps, gain 1..16;
  // forum 8962 / other driver builds report 10 and 21 fps for modes 0 and 1: runtime values are authoritative).
  static const std::vector<SensorMode> kImx519 = {
      make_mode(0, 4656, 3496, 9.0, kMinFps, 16.0, true, "imx519 full array"),
      make_mode(1, 3840, 2160, 17.0, kMinFps, 16.0, true, "imx519 16:9 field (reference)"),
      make_mode(2, 1920, 1080, 59.999999, kMinFps, 16.0, true, "imx519 2x2 binned 16:9 field"),
      make_mode(3, 1280, 720, 120.0, kMinFps, 16.0, false, "imx519 2x2 binned 2560x1440 crop (narrower FOV)"),
  };
  static const std::vector<SensorMode> kNone;
  if (sensor == "imx219") return kImx219;
  if (sensor == "imx519") return kImx519;
  return kNone;
}

bool is_known_sensor(std::string_view sensor) { return sensor == "imx219" || sensor == "imx519"; }

void annotate_sensor_modes(std::string_view sensor, std::vector<SensorMode>& modes) {
  if (!is_known_sensor(sensor)) return;
  const auto& table = known_sensor_modes(sensor);
  for (SensorMode& m : modes) {
    bool found = false;
    for (const SensorMode& t : table) {
      if (t.width == m.width && t.height == m.height) {
        m.full_fov = t.full_fov;
        if (m.description.empty()) m.description = t.description;
        found = true;
        break;
      }
    }
    if (!found) {
      m.full_fov = false;
      m.description += m.description.empty() ? "not in the T2 table (treated as cropped)"
                                             : " [not in the T2 table (treated as cropped)]";
    }
  }
}

bool mode_supports_fps(const SensorMode& mode, double fps) {
  if (!(fps > 0.0) || std::isnan(mode.max_fps)) return false;
  const double lo = std::isnan(mode.min_fps) ? 0.0 : mode.min_fps;
  return fps >= lo * (1.0 - kFpsRelTolerance) && fps <= mode.max_fps * (1.0 + kFpsRelTolerance);
}

int64_t frame_duration_ns(double fps) {
  if (!(fps > 0.0)) return 0;
  return static_cast<int64_t>(std::llround(1e9 / fps));
}

std::string format_sensor_mode(const SensorMode& m) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%d: %dx%d @ %.6g fps (%s; exposure %lld..%lld ns, gain %.4g..%.4g)", m.index,
                m.width, m.height, m.max_fps, m.full_fov ? "full FOV" : "cropped",
                static_cast<long long>(m.min_exposure_ns), static_cast<long long>(m.max_exposure_ns),
                m.min_analog_gain, m.max_analog_gain);
  std::string s = buf;
  if (!m.description.empty()) s += " [" + m.description + "]";
  return s;
}

std::string format_sensor_modes(const std::vector<SensorMode>& modes) {
  std::string s;
  for (const SensorMode& m : modes) {
    if (!s.empty()) s += "; ";
    s += format_sensor_mode(m);
  }
  return s.empty() ? std::string("(no modes)") : s;
}

Result<SensorMode> select_sensor_mode(const std::vector<SensorMode>& modes, const CameraConfig& config) {
  const std::string ctx = "cameras." + config.id + ".driver";
  if (config.sensor_mode >= 0) {
    for (const SensorMode& m : modes) {
      if (m.index == config.sensor_mode) {
        if (m.width != config.width || m.height != config.height) {
          return invalid_argument(ctx + ": sensor_mode " + std::to_string(config.sensor_mode) + " is " +
                                  std::to_string(m.width) + "x" + std::to_string(m.height) + ", but width x height is " +
                                  std::to_string(config.width) + "x" + std::to_string(config.height) +
                                  "; available modes: " + format_sensor_modes(modes));
        }
        if (!mode_supports_fps(m, config.fps)) {
          return invalid_argument(ctx + ": sensor_mode " + std::to_string(config.sensor_mode) + " cannot run at " +
                                  std::to_string(config.fps) + " fps; available modes: " + format_sensor_modes(modes));
        }
        return m;
      }
    }
    return not_found(ctx + ": sensor_mode " + std::to_string(config.sensor_mode) +
                     " does not exist; available modes: " + format_sensor_modes(modes));
  }
  const SensorMode* cropped_match = nullptr;
  for (const SensorMode& m : modes) {
    if (m.width != config.width || m.height != config.height || !mode_supports_fps(m, config.fps)) continue;
    if (m.full_fov) return m;  // first full-FOV match in driver order
    if (cropped_match == nullptr) cropped_match = &m;
  }
  char req[96];
  std::snprintf(req, sizeof(req), "%dx%d@%g", config.width, config.height, config.fps);
  if (cropped_match != nullptr) {
    return not_found(ctx + ": " + req + " is a cropped mode on " + (config.sensor.empty() ? "this sensor" : config.sensor) +
                     " (mode " + std::to_string(cropped_match->index) +
                     "); set sensor_mode to force it. Available modes: " + format_sensor_modes(modes));
  }
  return not_found(ctx + ": no sensor mode " + req + "; available modes: " + format_sensor_modes(modes));
}

}  // namespace rig
