// SPDX-License-Identifier: MIT
// Private: YAML spellings of every enum in rig.yaml, shared by the loader and to_yaml() so the two can never
// disagree. Spellings are the ones documented in the frozen headers (rig/types.hpp, drivers/*.hpp,
// camera_model/model_params.hpp, time/clock.hpp, config/rig_config.hpp).
#pragma once

#include <utility>
#include <vector>

#include "rig/camera_model/model_params.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/drivers/camera_backend.hpp"
#include "rig/drivers/imu_backend.hpp"
#include "rig/time/clock.hpp"
#include "rig/types.hpp"

namespace rig {
namespace cfg {

template <class E>
using EnumTable = std::vector<std::pair<const char*, E>>;

inline const EnumTable<Provenance>& provenance_table() {
  static const EnumTable<Provenance> t = {{"nominal", Provenance::kNominal},
                                          {"cad", Provenance::kCad},
                                          {"kalibr", Provenance::kKalibr},
                                          {"measured", Provenance::kMeasured},
                                          {"manual", Provenance::kManual}};
  return t;
}

/// Formats delivered to consumers (rig.yaml driver.pixel_format). kJPEG/kUnknown are not capture formats.
inline const EnumTable<PixelFormat>& pixel_format_table() {
  static const EnumTable<PixelFormat> t = {{"nv12", PixelFormat::kNV12},
                                           {"gray8", PixelFormat::kGRAY8},
                                           {"bgr8", PixelFormat::kBGR8},
                                           {"rgb8", PixelFormat::kRGB8},
                                           {"yuyv", PixelFormat::kYUYV}};
  return t;
}

inline const EnumTable<CameraBackendKind>& camera_backend_table() {
  static const EnumTable<CameraBackendKind> t = {{"argus", CameraBackendKind::kArgus},
                                                 {"gst", CameraBackendKind::kGst},
                                                 {"mock", CameraBackendKind::kMock},
                                                 {"replay", CameraBackendKind::kReplay}};
  return t;
}

inline const EnumTable<FocusMode>& focus_mode_table() {
  static const EnumTable<FocusMode> t = {{"none", FocusMode::kNone}, {"fixed", FocusMode::kFixed}};
  return t;
}

inline const EnumTable<CameraModelKind>& camera_model_table() {
  static const EnumTable<CameraModelKind> t = {{"pinhole-radtan", CameraModelKind::kPinholeRadtan},
                                               {"kannala-brandt4", CameraModelKind::kKannalaBrandt4},
                                               {"double-sphere", CameraModelKind::kDoubleSphere},
                                               {"eucm", CameraModelKind::kEucm},
                                               {"pinhole", CameraModelKind::kVirtualPinhole}};
  return t;
}

inline const EnumTable<ImuBackendKind>& imu_backend_table() {
  static const EnumTable<ImuBackendKind> t = {{"lsm6dsox_i2c", ImuBackendKind::kLsm6dsoxI2c},
                                              {"sim", ImuBackendKind::kSim},
                                              {"replay", ImuBackendKind::kReplay}};
  return t;
}

inline const EnumTable<ClockFitMethod>& clock_fit_table() {
  static const EnumTable<ClockFitMethod> t = {{"lower_envelope", ClockFitMethod::kLowerEnvelope},
                                              {"robust_linear", ClockFitMethod::kRobustLinear}};
  return t;
}

inline const EnumTable<SlamBackendKind>& slam_backend_table() {
  static const EnumTable<SlamBackendKind> t = {{"orbslam3", SlamBackendKind::kOrbSlam3},
                                               {"mock", SlamBackendKind::kMock}};
  return t;
}

inline const EnumTable<VizMode>& viz_mode_table() {
  static const EnumTable<VizMode> t = {{"none", VizMode::kNone},
                                       {"cameras", VizMode::kCameras},
                                       {"slam", VizMode::kSlam},
                                       {"both", VizMode::kBoth}};
  return t;
}

/// Plain string enums (validated as strings, stored as std::string in the structs).
inline const std::vector<const char*>& camera_roles() {
  static const std::vector<const char*> t = {"main", "ultrawide"};
  return t;
}
inline const std::vector<const char*>& camera_sensors() {
  static const std::vector<const char*> t = {"imx519", "imx219"};
  return t;
}
/// Argus AwbMode names (Argus/Types.h AWB_MODE_*), lowercase, as documented in the example rig.yaml; T2 maps them.
inline const std::vector<const char*>& awb_modes() {
  static const std::vector<const char*> t = {"off",      "auto",            "incandescent", "fluorescent",
                                             "daylight", "cloudy_daylight", "twilight",     "shade"};
  return t;
}
inline const std::vector<const char*>& mock_patterns() {
  static const std::vector<const char*> t = {"checker", "gradient", "noise", "counter"};
  return t;
}

/// Name of `v` in `t` ("?" if absent: only for corrupt values in to_yaml).
template <class E>
const char* name_of(const EnumTable<E>& t, E v) {
  for (const auto& kv : t)
    if (kv.second == v) return kv.first;
  return "?";
}

}  // namespace cfg
}  // namespace rig
