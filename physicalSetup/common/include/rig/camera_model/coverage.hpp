// SPDX-License-Identifier: MIT
/// @file rig/camera_model/coverage.hpp
/// @brief Coverage report of a physical -> virtual remap (T4, new header in its own module; not part of the T0
/// frozen set). Used by ImagePrep::create (min_coverage), ec-doctor / calibration checks and T7 diagnostics.
///
/// A virtual pixel is "covered" when its ray is valid in both models and lands inside the physical image's
/// pixel-centre range [0, W-1] x [0, H-1] (bilinear-safe; sensor_rig/sensor_rig/camera_model.py:139-143, 573-581).
/// This is the same rule build_remap uses for RemapTable::valid_mask, so report.coverage == table.coverage.
#pragma once

#include <Eigen/Core>

#include "rig/camera_model/camera_model.hpp"
#include "rig/status.hpp"

namespace rig {

struct CoverageReport {
  int dst_width = 0;               ///< Virtual image size [px].
  int dst_height = 0;
  long n_total = 0;                ///< dst_width * dst_height.
  long n_valid = 0;                ///< Covered virtual pixels.
  long n_model_invalid = 0;        ///< Virtual pixels whose ray is invalid in a model (FoV limit, domain).
  long n_outside = 0;              ///< Valid rays that land outside the physical image.
  double coverage = 0.0;           ///< n_valid / n_total [0..1].
  /// Smallest distance [px] from any covered sample position to the physical pixel-centre border; a measure of
  /// slack (negative never happens for covered pixels). NaN if nothing is covered.
  double min_margin_px = kNaN;
  /// Largest angle [deg] between the physical optical axis and any covered ray (how much of the fisheye FoV the
  /// virtual view uses). NaN if nothing is covered.
  double max_ray_angle_deg = kNaN;
  /// Bounding box of the sampled physical positions of covered pixels [px] (NaN if nothing is covered).
  double src_u_min = kNaN, src_u_max = kNaN, src_v_min = kNaN, src_v_max = kNaN;
};

/// Computes the report in float64 for every virtual pixel centre (same geometry as build_remap).
Result<CoverageReport> coverage_report(const CameraModel& phys, const VirtualPinhole& virt,
                                       const Eigen::Matrix3d& R_phys_virt = Eigen::Matrix3d::Identity());

}  // namespace rig
