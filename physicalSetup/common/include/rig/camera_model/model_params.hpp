// SPDX-License-Identifier: MIT
/// @file rig/camera_model/model_params.hpp
/// @brief Dependency-light description of a camera model (used by the config, the SLAM settings writer and
/// camera_model.hpp). FROZEN CONTRACT (T0).
#pragma once

#include <string>
#include <vector>

#include "rig/types.hpp"

namespace rig {

/// Projection model. YAML strings in parentheses; parameter layout in CameraModelParams.
enum class CameraModelKind : uint8_t {
  kPinholeRadtan = 0,   ///< ("pinhole-radtan") distortion = [k1, k2, p1, p2] (OpenCV / Kalibr radtan order).
  kKannalaBrandt4 = 1,  ///< ("kannala-brandt4") equidistant fisheye, distortion = [k1, k2, k3, k4]
                        ///< (OpenCV fisheye, Kalibr `equidistant`, ORB-SLAM3 `KannalaBrandt8`).
  kDoubleSphere = 2,    ///< ("double-sphere") distortion = [xi, alpha] (Usenko et al. 2018; Kalibr `ds`).
  kEucm = 3,            ///< ("eucm") distortion = [alpha, beta] (Khomutenko et al. 2016; Kalibr `eucm`).
  kVirtualPinhole = 4,  ///< ("pinhole") ideal pinhole, distortion = [] (virtual target cameras).
};

constexpr const char* to_string(CameraModelKind k) noexcept {
  switch (k) {
    case CameraModelKind::kPinholeRadtan: return "pinhole-radtan";
    case CameraModelKind::kKannalaBrandt4: return "kannala-brandt4";
    case CameraModelKind::kDoubleSphere: return "double-sphere";
    case CameraModelKind::kEucm: return "eucm";
    case CameraModelKind::kVirtualPinhole: return "pinhole";
  }
  return "unknown";
}

/// Number of distortion parameters each kind requires.
constexpr int num_distortion_params(CameraModelKind k) noexcept {
  switch (k) {
    case CameraModelKind::kPinholeRadtan: return 4;
    case CameraModelKind::kKannalaBrandt4: return 4;
    case CameraModelKind::kDoubleSphere: return 2;
    case CameraModelKind::kEucm: return 2;
    case CameraModelKind::kVirtualPinhole: return 0;
  }
  return 0;
}

/// Model parameters at a given image size. Mirrors rig.yaml `cameras.<id>.model` and `virtual_cameras.<id>.model`.
struct CameraModelParams {
  CameraModelKind kind = CameraModelKind::kVirtualPinhole;
  int width = 0;                    ///< Image size the intrinsics refer to [px].
  int height = 0;
  double fx = kNaN;                 ///< Focal length x [px].
  double fy = kNaN;                 ///< Focal length y [px].
  double cx = kNaN;                 ///< Principal point x [px], OpenCV pixel-centre convention (0,0 = centre of the
                                    ///< top-left pixel).
  double cy = kNaN;                 ///< Principal point y [px].
  std::vector<double> distortion;   ///< Layout per CameraModelKind; size == num_distortion_params(kind).
  double fov_max_deg = 0.0;         ///< Full field of view beyond which unproject/project report invalid; 0 = none.
  Provenance provenance = Provenance::kNominal;
  std::string note;                 ///< Free text: source of the numbers.
};

}  // namespace rig
