// SPDX-License-Identifier: MIT
/// @file rig/drivers/camera/sensor_modes.hpp
/// @brief Published sensor-mode tables (IMX519, IMX219 on Jetson L4T r36) and mode helpers. Owned by T2.
///
/// The Argus backend enumerates the real modes from the driver at start(); these tables are used (a) by the mock and
/// GStreamer backends, which cannot enumerate, (b) to annotate driver modes with `full_fov` (Argus does not report
/// crops), and (c) by tests. Runtime Argus values are authoritative; the tables are checked on day H (BRINGUP 10).
///
/// `full_fov` definition (per sensor "reference field"; no crop relative to it):
/// - imx219: the full 3280x2464 pixel array. Full FOV: 3280x2464@21, 1640x1232@30 (2x2 binned). Crops: 3280x1848@28,
///   1920x1080@30, 1280x720@60 (PIPELINE 3; sensor_rig picamera2_logic.py:99-104).
/// - imx519: the 3840x2160 16:9 field (crop 416,720 of the 4656x3496 array; upstream imx519.c supported_modes,
///   https://lkml.iu.edu/hypermail/linux/kernel/2309.1/00390.html). The main camera's design FOV (PIPELINE 2, ~66x38
///   deg at 16:9) is this field. Full FOV: 4656x3496 (the whole array, a superset), 3840x2160, 1920x1080 (2x2 binned
///   from the 3840x2160 field). Crop: 1280x720@120, a 2x2 bin of a 2560x1440 crop (1056,1090) = 2/3 of the 1080p
///   field's width, i.e. a NARROWER FOV.
/// Threading: pure functions / immutable tables.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "rig/drivers/camera_backend.hpp"

namespace rig {

/// Relative tolerance on fps range checks. Drivers report e.g. 1080p60 as 59.999999 fps (frame duration 16666667 ns,
/// NVIDIA forum 246754 mode dump), so an exact comparison would reject the nominal rate.
inline constexpr double kFpsRelTolerance = 1e-4;

/// Nominal Jetson mode table of `sensor` ("imx519" | "imx219"); empty for anything else.
///
/// - imx219 (NVIDIA L4T imx219 driver; Argus dump https://forums.developer.nvidia.com/t/jetson-orin-imx219-csi-camera/
///   246754): index 0 3280x2464@21, 1 3280x1848@28, 2 1920x1080@30, 3 1640x1232@30, 4 1280x720@60; analog gain
///   1..10.625, exposure 13 us..683.709 ms, RAW10.
/// - imx519 (Arducam Jetson driver on Orin Nano, JP 6.x; https://forum.arducam.com/t/imx519-and-jetson-orin-nano/8481
///   and .../8962): index 0 4656x3496@9 (some driver versions report 10), 1 3840x2160@17 (or 21), 2 1920x1080@60,
///   3 1280x720@120; analog gain 1..16, exposure 13 us..683.709 ms, RAW10.
const std::vector<SensorMode>& known_sensor_modes(std::string_view sensor);

/// True for the sensors that have a table ("imx519", "imx219").
bool is_known_sensor(std::string_view sensor);

/// Copies `full_fov` (and a table note into an empty `description`) from the table onto driver-reported modes with
/// the same width x height. Modes of a known sensor that are not in the table get full_fov = false and a
/// "not in the T2 table" note (they can still be forced by index). No-op for an unknown sensor.
void annotate_sensor_modes(std::string_view sensor, std::vector<SensorMode>& modes);

/// True if `fps` lies in [min_fps, max_fps] with kFpsRelTolerance.
bool mode_supports_fps(const SensorMode& mode, double fps);

/// Frame duration for `fps` [ns], rounded to the nearest ns (60 -> 16666667). 0 for fps <= 0.
int64_t frame_duration_ns(double fps);

/// "3: 1640x1232 @ 30 fps (full FOV)" style one-line description.
std::string format_sensor_mode(const SensorMode& mode);
/// All modes, separated by "; ".
std::string format_sensor_modes(const std::vector<SensorMode>& modes);

}  // namespace rig
