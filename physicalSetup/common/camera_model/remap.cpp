// SPDX-License-Identifier: MIT
// Physical -> virtual remap tables and coverage (T4). Port of sensor_rig/sensor_rig/camera_model.py:478-581
// (RemapTable, _map_rays, _table, build_remap, coverage_mask). The maps are bit-identical to the Python ones on the
// golden cases (tests/camera_model/test_remap_golden.cpp), so cv::remap output is too.
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include <Eigen/LU>
#include <opencv2/imgproc.hpp>

#include "rig/camera_model/camera_model.hpp"
#include "rig/camera_model/coverage.hpp"

namespace rig {
namespace {

// camera_model.py:36: map coordinate for "no source sample" (outside even the bilinear footprint), so
// BORDER_CONSTANT paints the pixel 0.
constexpr float kInvalidMap = -16.0f;

// camera_model.py:139-143: pixel-centre range [0, W-1] x [0, H-1] (bilinear-safe).
inline bool in_image_bilinear(const CameraModel& m, const Eigen::Vector2d& uv) {
  return uv.x() >= 0 && uv.x() <= m.width() - 1 && uv.y() >= 0 && uv.y() <= m.height() - 1;
}

// camera_model.py:524-530 (_map_rays) for one destination pixel. Returns model validity; *ray_phys and *uv_src set
// when valid.
inline bool map_pixel(const CameraModel& phys, const VirtualPinhole& virt, const Eigen::Matrix3d& R, bool identity,
                      double u, double v, Eigen::Vector3d* ray_phys, Eigen::Vector2d* uv_src) {
  Eigen::Vector3d ray;
  if (!virt.unproject(Eigen::Vector2d(u, v), &ray)) return false;
  // camera_model.py:527-528: rays @ R.T, skipped for the identity (camera_model.py:516).
  if (!identity) ray = R * ray;
  *ray_phys = ray;
  return phys.project(ray, uv_src);
}

Status check_rotation(const Eigen::Matrix3d& R) {
  if (!R.allFinite()) return invalid_argument("build_remap: R_phys_virt must be finite");
  const double orth = (R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
  if (orth > 1e-6 || std::abs(R.determinant() - 1.0) > 1e-6) {
    return invalid_argument("build_remap: R_phys_virt must be a rotation (orthonormal, det +1)");
  }
  return Status::Ok();
}

}  // namespace

Status RemapTable::apply(const cv::Mat& src, cv::Mat& dst, int interpolation) const {
  if (map1.empty() || map1.size() != dst_size) return failed_precondition("RemapTable::apply: table is empty");
  if (src.empty()) return invalid_argument("RemapTable::apply: empty source image");
  if (src.size() != src_size) {
    return invalid_argument("RemapTable::apply: expected a " + std::to_string(src_size.width) + "x" +
                            std::to_string(src_size.height) + " source, got " + std::to_string(src.cols) + "x" +
                            std::to_string(src.rows));
  }
  if (src.depth() != CV_8U) return invalid_argument("RemapTable::apply: source must be 8-bit (camera_model.py:499)");
  try {
    // cv::remap cannot run in place.
    if (dst.data != nullptr && dst.data == src.data) {
      cv::Mat tmp;
      cv::remap(src, tmp, map1, map2, interpolation, cv::BORDER_CONSTANT, cv::Scalar());
      dst = tmp;
    } else {
      cv::remap(src, dst, map1, map2, interpolation, cv::BORDER_CONSTANT, cv::Scalar());
    }
  } catch (const cv::Exception& e) {
    return internal_error(std::string("RemapTable::apply: cv::remap: ") + e.what());
  }
  return Status::Ok();
}

// camera_model.py:533-554 (_table / build_remap).
Result<RemapTable> build_remap(const CameraModel& phys, const VirtualPinhole& virt, const Eigen::Matrix3d& R_phys_virt,
                               bool fixed_point) {
  RIG_RETURN_IF_ERROR(check_rotation(R_phys_virt));
  const bool identity = R_phys_virt == Eigen::Matrix3d::Identity();
  const int W = virt.width(), H = virt.height();
  cv::Mat mx(H, W, CV_32FC1), my(H, W, CV_32FC1), valid(H, W, CV_8UC1);
  long n_valid = 0;
  for (int i = 0; i < H; ++i) {
    float* px = mx.ptr<float>(i);
    float* py = my.ptr<float>(i);
    uint8_t* pv = valid.ptr<uint8_t>(i);
    for (int j = 0; j < W; ++j) {
      Eigen::Vector3d ray;
      Eigen::Vector2d uv;
      const bool ok = map_pixel(phys, virt, R_phys_virt, identity, static_cast<double>(j), static_cast<double>(i),
                                &ray, &uv) &&
                      in_image_bilinear(phys, uv);
      px[j] = ok ? static_cast<float>(uv.x()) : kInvalidMap;
      py[j] = ok ? static_cast<float>(uv.y()) : kInvalidMap;
      pv[j] = ok ? 255 : 0;
      n_valid += ok ? 1 : 0;
    }
  }
  RemapTable t;
  t.src_size = cv::Size(phys.width(), phys.height());
  t.dst_size = cv::Size(W, H);
  t.valid_mask = valid;
  t.coverage = static_cast<double>(n_valid) / (static_cast<double>(W) * static_cast<double>(H));
  t.fixed_point = fixed_point;
  if (fixed_point) {
    try {
      cv::convertMaps(mx, my, t.map1, t.map2, CV_16SC2, false);  // camera_model.py:539-540
    } catch (const cv::Exception& e) {
      return internal_error(std::string("build_remap: cv::convertMaps: ") + e.what());
    }
  } else {
    t.map1 = mx;
    t.map2 = my;
  }
  return t;
}

Result<CoverageReport> coverage_report(const CameraModel& phys, const VirtualPinhole& virt,
                                       const Eigen::Matrix3d& R_phys_virt) {
  RIG_RETURN_IF_ERROR(check_rotation(R_phys_virt));
  const bool identity = R_phys_virt == Eigen::Matrix3d::Identity();
  CoverageReport r;
  r.dst_width = virt.width();
  r.dst_height = virt.height();
  r.n_total = static_cast<long>(r.dst_width) * r.dst_height;
  double margin = std::numeric_limits<double>::infinity(), max_angle = 0.0;
  double umin = margin, vmin = margin, umax = -margin, vmax = -margin;
  const double W1 = phys.width() - 1, H1 = phys.height() - 1;
  for (int i = 0; i < r.dst_height; ++i) {
    for (int j = 0; j < r.dst_width; ++j) {
      Eigen::Vector3d ray;
      Eigen::Vector2d uv;
      if (!map_pixel(phys, virt, R_phys_virt, identity, j, i, &ray, &uv)) {
        ++r.n_model_invalid;
        continue;
      }
      if (!in_image_bilinear(phys, uv)) {
        ++r.n_outside;
        continue;
      }
      ++r.n_valid;
      margin = std::min({margin, uv.x(), uv.y(), W1 - uv.x(), H1 - uv.y()});
      max_angle = std::max(max_angle, std::atan2(std::hypot(ray.x(), ray.y()), ray.z()));
      umin = std::min(umin, uv.x());
      umax = std::max(umax, uv.x());
      vmin = std::min(vmin, uv.y());
      vmax = std::max(vmax, uv.y());
    }
  }
  r.coverage = static_cast<double>(r.n_valid) / static_cast<double>(r.n_total);
  if (r.n_valid > 0) {
    r.min_margin_px = margin;
    r.max_ray_angle_deg = max_angle * 180.0 / 3.14159265358979323846;
    r.src_u_min = umin;
    r.src_u_max = umax;
    r.src_v_min = vmin;
    r.src_v_max = vmax;
  }
  return r;
}

}  // namespace rig
