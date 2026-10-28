// SPDX-License-Identifier: MIT
/// @file rig/camera_model/camera_model.hpp
/// @brief Camera projection models and physical -> virtual remap tables. FROZEN CONTRACT (T0); declarations only,
/// implemented by T4 in common/camera_model/ (ported from sensor_rig/sensor_rig/camera_model.py with cited lines).
///
/// Conventions: camera frame = OpenCV optical; pixel coordinates = OpenCV (0,0 at the centre of the top-left pixel).
/// Threading: models and RemapTable are immutable after construction; const methods are thread-safe.
/// Private members of the classes below belong to T4 and may change; the public API is the contract.
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "rig/camera_model/model_params.hpp"
#include "rig/status.hpp"

namespace rig {

/// Abstract projection model.
class CameraModel {
 public:
  virtual ~CameraModel() = default;

  CameraModelKind kind() const noexcept { return params_.kind; }
  int width() const noexcept { return params_.width; }
  int height() const noexcept { return params_.height; }
  const CameraModelParams& params() const noexcept { return params_; }

  /// Projects a point in the camera frame [m, any scale] to pixels. Returns false (uv unspecified) if the point is
  /// not projectable (behind the camera for pinhole models, beyond fov_max_deg, or outside the model's valid
  /// domain). @param J optional d(uv)/d(p) Jacobian.
  virtual bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv,
                       Eigen::Matrix<double, 2, 3>* J = nullptr) const = 0;

  /// Unprojects a pixel to a unit-norm ray in the camera frame. Returns false if the pixel has no valid ray.
  virtual bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const = 0;

  /// The same model for an image uniformly scaled to width x height (intrinsics scaled; distortion unchanged).
  virtual std::unique_ptr<CameraModel> scaled(int width, int height) const = 0;

  /// True if uv lies inside [-0.5, width - 0.5] x [-0.5, height - 0.5].
  bool in_image(const Eigen::Vector2d& uv) const noexcept {
    return uv.x() >= -0.5 && uv.y() >= -0.5 && uv.x() <= params_.width - 0.5 && uv.y() <= params_.height - 0.5;
  }

 protected:
  explicit CameraModel(CameraModelParams params) : params_(std::move(params)) {}
  CameraModelParams params_;
};

/// Pinhole + radial-tangential distortion [k1, k2, p1, p2].
class PinholeRadtan final : public CameraModel {
 public:
  static Result<std::unique_ptr<PinholeRadtan>> create(const CameraModelParams& params);
  bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv, Eigen::Matrix<double, 2, 3>* J = nullptr) const override;
  bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const override;
  std::unique_ptr<CameraModel> scaled(int width, int height) const override;

 private:
  PinholeRadtan(CameraModelParams params, double r_valid) : CameraModel(std::move(params)), r_valid_(r_valid) {}
  double r_valid_;  ///< Largest undistorted radius x/z on the monotonic branch (camera_model.py:177-180).
};

/// Equidistant Kannala-Brandt fisheye with 4 coefficients (ORB-SLAM3 KannalaBrandt8 / OpenCV fisheye).
class KannalaBrandt4 final : public CameraModel {
 public:
  static Result<std::unique_ptr<KannalaBrandt4>> create(const CameraModelParams& params);
  bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv, Eigen::Matrix<double, 2, 3>* J = nullptr) const override;
  bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const override;
  std::unique_ptr<CameraModel> scaled(int width, int height) const override;

 private:
  KannalaBrandt4(CameraModelParams params, double theta_valid)
      : CameraModel(std::move(params)), theta_valid_(theta_valid) {}
  double theta_valid_;  ///< Largest incidence angle [rad] on the monotonic branch (camera_model.py:286-288).
};

/// Double Sphere [xi, alpha].
class DoubleSphere final : public CameraModel {
 public:
  static Result<std::unique_ptr<DoubleSphere>> create(const CameraModelParams& params);
  bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv, Eigen::Matrix<double, 2, 3>* J = nullptr) const override;
  bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const override;
  std::unique_ptr<CameraModel> scaled(int width, int height) const override;

 private:
  explicit DoubleSphere(CameraModelParams params) : CameraModel(std::move(params)) {}
};

/// Extended Unified Camera Model [alpha, beta].
class Eucm final : public CameraModel {
 public:
  static Result<std::unique_ptr<Eucm>> create(const CameraModelParams& params);
  bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv, Eigen::Matrix<double, 2, 3>* J = nullptr) const override;
  bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const override;
  std::unique_ptr<CameraModel> scaled(int width, int height) const override;

 private:
  explicit Eucm(CameraModelParams params) : CameraModel(std::move(params)) {}
};

/// Ideal pinhole used as the remap target (e.g. the virtual iPhone ultrawide 640x480, fx 246.6; PIPELINE 2).
class VirtualPinhole final : public CameraModel {
 public:
  static Result<std::unique_ptr<VirtualPinhole>> create(const CameraModelParams& params);
  bool project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv, Eigen::Matrix<double, 2, 3>* J = nullptr) const override;
  bool unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const override;
  std::unique_ptr<CameraModel> scaled(int width, int height) const override;
  /// 3x3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
  Eigen::Matrix3d K() const;

 private:
  explicit VirtualPinhole(CameraModelParams params) : CameraModel(std::move(params)) {}
};

/// Creates the model for `params.kind`. kInvalidArgument on wrong distortion size, non-positive size/focal, or
/// non-finite values.
Result<std::unique_ptr<CameraModel>> make_camera_model(const CameraModelParams& params);

/// Precomputed physical -> virtual resampling maps (dst pixel -> src pixel), applied with cv::remap.
struct RemapTable {
  cv::Mat map1;            ///< CV_16SC2 (fixed point) or CV_32FC1 (x) in virtual (dst) size.
  cv::Mat map2;            ///< CV_16UC1 (fixed-point fractions) or CV_32FC1 (y).
  cv::Mat valid_mask;      ///< CV_8UC1 dst-size: 255 where the dst pixel's ray lands inside the source image.
  cv::Size src_size;       ///< Physical image size the maps expect.
  cv::Size dst_size;       ///< Virtual image size produced.
  double coverage = 0.0;   ///< Fraction of dst pixels with valid_mask = 255 [0..1].
  bool fixed_point = true; ///< Map representation (cv::convertMaps CV_16SC2).

  /// cv::remap(src, dst, map1, map2, interpolation, BORDER_CONSTANT, 0). kInvalidArgument if src.size() !=
  /// src_size. Bit-exactness versus the Python reference is the T4 golden test.
  Status apply(const cv::Mat& src, cv::Mat& dst, int interpolation = cv::INTER_LINEAR) const;
};

/// Builds maps so that virtual pixel u_v samples the physical image at phys.project(R_phys_virt * virt.unproject(u_v)).
/// @param R_phys_virt rotation of the virtual camera in the physical camera frame (identity = same optical axis).
Result<RemapTable> build_remap(const CameraModel& phys, const VirtualPinhole& virt,
                               const Eigen::Matrix3d& R_phys_virt = Eigen::Matrix3d::Identity(),
                               bool fixed_point = true);

}  // namespace rig
