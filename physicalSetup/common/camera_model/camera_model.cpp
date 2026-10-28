// SPDX-License-Identifier: MIT
// Camera projection models (T4). Contract: rig/camera_model/camera_model.hpp.
//
// PinholeRadtan, KannalaBrandt4 and VirtualPinhole are line-by-line scalar ports of
// sensor_rig/sensor_rig/camera_model.py (cited per function) and are checked against it to 1e-9
// (tests/camera_model/test_python_parity.cpp). DoubleSphere follows Usenko, Demmel, Cremers, "The Double Sphere
// Camera Model", 3DV 2018, eqs. (40)-(47); Eucm follows Khomutenko, Garcia, Martinet, "An Enhanced Unified Camera
// Model", RA-L 2016, in the form used by Usenko et al. 2018 eqs. (24)-(31) (= Kalibr `ds` / `eucm`).
//
// Deliberate differences to camera_model.py (documented in T4.WORKLOG):
// - VirtualPinhole::project does not require the pixel to be inside the image (camera_model.py:372 does); callers
//   use in_image(). build_remap/coverage apply the image check themselves with the bilinear-safe [0, W-1] rule.
// - fov_max_deg (not in the Python models) caps the valid domain: radtan r <= tan(fov/2), KB4 theta <= fov/2;
//   0 = the Python defaults (radtan r cap tan(89 deg), camera_model.py:37; KB4 theta_max_deg 89,
//   camera_model.py:273).
#include "rig/camera_model/camera_model.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "camera_model/model_math.hpp"

namespace rig {
namespace {

using camera_model_detail::deg2rad;
using camera_model_detail::first_nonpositive;
using camera_model_detail::monotone_inverse;

std::string ctx(const CameraModelParams& p) { return std::string("camera model '") + to_string(p.kind) + "'"; }

// camera_model.py:112-124 (_check_size_and_intrinsics) + distortion size/finiteness + fov range.
Status validate_common(const CameraModelParams& p, double fov_upper_exclusive) {
  if (p.width <= 0 || p.height <= 0) {
    return invalid_argument(ctx(p) + ": width/height must be > 0, got " + std::to_string(p.width) + "x" +
                            std::to_string(p.height));
  }
  if (!std::isfinite(p.fx) || !std::isfinite(p.fy) || !std::isfinite(p.cx) || !std::isfinite(p.cy)) {
    return invalid_argument(ctx(p) + ": fx, fy, cx, cy must be finite");
  }
  if (p.fx <= 0 || p.fy <= 0) return invalid_argument(ctx(p) + ": focal lengths must be > 0");
  const int n = num_distortion_params(p.kind);
  if (static_cast<int>(p.distortion.size()) != n) {
    return invalid_argument(ctx(p) + ": distortion must have " + std::to_string(n) + " values, got " +
                            std::to_string(p.distortion.size()));
  }
  for (double d : p.distortion) {
    if (!std::isfinite(d)) return invalid_argument(ctx(p) + ": distortion values must be finite");
  }
  if (!std::isfinite(p.fov_max_deg) || p.fov_max_deg < 0.0 || p.fov_max_deg >= fov_upper_exclusive) {
    return invalid_argument(ctx(p) + ": fov_max_deg must be in [0, " + std::to_string(fov_upper_exclusive) +
                            "), got " + std::to_string(p.fov_max_deg));
  }
  return Status::Ok();
}

// Uniform rescale with the cv2.resize half-pixel convention (camera_model.py:42-52). nullopt-like: false if the
// target size is not a uniform scale of the source.
bool scale_params(const CameraModelParams& p, int width, int height, CameraModelParams* out) {
  if (width <= 0 || height <= 0) return false;
  if (static_cast<long long>(width) * p.height != static_cast<long long>(height) * p.width) return false;
  const double s = static_cast<double>(width) / static_cast<double>(p.width);
  *out = p;
  out->width = width;
  out->height = height;
  out->fx = s * p.fx;
  out->fy = s * p.fy;
  out->cx = s * (p.cx + 0.5) - 0.5;
  out->cy = s * (p.cy + 0.5) - 0.5;
  return true;
}

// Angle check for models whose domain is not otherwise limited by fov_max_deg.
bool within_fov(const CameraModelParams& p, double x, double y, double z) {
  if (p.fov_max_deg <= 0.0) return true;
  return std::atan2(std::hypot(x, y), z) <= 0.5 * deg2rad(p.fov_max_deg);
}

bool all_finite(const Eigen::Vector3d& v) { return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z()); }

void set_nan(Eigen::Vector2d* uv) {
  if (uv) uv->setConstant(kNaN);
}
void set_nan(Eigen::Vector3d* r) {
  if (r) r->setConstant(kNaN);
}

}  // namespace

// ================================================================================================ PinholeRadtan

Result<std::unique_ptr<PinholeRadtan>> PinholeRadtan::create(const CameraModelParams& params) {
  if (params.kind != CameraModelKind::kPinholeRadtan) return invalid_argument(ctx(params) + ": wrong kind");
  RIG_RETURN_IF_ERROR(validate_common(params, 180.0));
  const double k1 = params.distortion[0], k2 = params.distortion[1];
  // camera_model.py:37 (_RADTAN_R_CAP = tan(89 deg)) and :178-180; fov_max_deg tightens the cap.
  double cap = std::tan(deg2rad(89.0));
  if (params.fov_max_deg > 0.0) cap = std::min(cap, std::tan(0.5 * deg2rad(params.fov_max_deg)));
  const double r_valid =
      first_nonpositive([&](double r) { return 1.0 + 3.0 * k1 * (r * r) + 5.0 * k2 * std::pow(r, 4); }, cap);
  return std::unique_ptr<PinholeRadtan>(new PinholeRadtan(params, r_valid));
}

namespace {
// camera_model.py:186-191 (_distort).
inline void radtan_distort(double k1, double k2, double p1, double p2, double x, double y, double* xd, double* yd) {
  const double r2 = x * x + y * y;
  const double rho = 1.0 + k1 * r2 + k2 * r2 * r2;
  *xd = x * rho + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  *yd = y * rho + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
}
}  // namespace

// camera_model.py:193-203.
bool PinholeRadtan::project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv, Eigen::Matrix<double, 2, 3>* J) const {
  const auto& d = params_.distortion;
  const double k1 = d[0], k2 = d[1], p1 = d[2], p2 = d[3];
  const double Z = p_c.z();
  bool ok = Z > 1e-9;
  const double Zs = ok ? Z : 1.0;
  const double x = p_c.x() / Zs, y = p_c.y() / Zs;
  ok = ok && (x * x + y * y <= r_valid_ * r_valid_);
  if (!ok) {
    set_nan(uv);
    return false;
  }
  double xd, yd;
  radtan_distort(k1, k2, p1, p2, x, y, &xd, &yd);
  if (uv) *uv = Eigen::Vector2d(params_.fx * xd + params_.cx, params_.fy * yd + params_.cy);
  if (J) {
    // d(xd,yd)/d(x,y): the same terms as the Newton step of camera_model.py:222-227.
    const double r2 = x * x + y * y;
    const double rho = 1.0 + k1 * r2 + k2 * r2 * r2;
    const double a = 2.0 * k1 + 4.0 * k2 * r2;
    const double j11 = rho + a * x * x + 2.0 * p1 * y + 6.0 * p2 * x;
    const double j12 = a * x * y + 2.0 * p1 * x + 2.0 * p2 * y;
    const double j22 = rho + a * y * y + 6.0 * p1 * y + 2.0 * p2 * x;
    Eigen::Matrix<double, 2, 3> Jn;
    Jn << 1.0 / Z, 0.0, -x / Z, 0.0, 1.0 / Z, -y / Z;
    Eigen::Matrix2d Jd;
    Jd << params_.fx * j11, params_.fx * j12, params_.fy * j12, params_.fy * j22;
    *J = Jd * Jn;
  }
  return true;
}

// camera_model.py:205-242.
bool PinholeRadtan::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const {
  const auto& d = params_.distortion;
  const double k1 = d[0], k2 = d[1], p1 = d[2], p2 = d[3];
  const double xd = (uv.x() - params_.cx) / params_.fx;
  const double yd = (uv.y() - params_.cy) / params_.fy;
  const double rd = std::hypot(xd, yd);
  double r = 0.0;
  bool ok = monotone_inverse([&](double t) { return t * (1.0 + k1 * t * t + k2 * std::pow(t, 4)); },
                             [&](double t) { return 1.0 + 3.0 * k1 * t * t + 5.0 * k2 * std::pow(t, 4); }, rd,
                             r_valid_, &r);
  const double s = rd > 1e-15 ? r / std::max(rd, 1e-300) : 1.0;
  double x = xd * s, y = yd * s;
  if (p1 != 0.0 || p2 != 0.0) {
    for (int it = 0; it < 20; ++it) {
      double fx_, fy_;
      radtan_distort(k1, k2, p1, p2, x, y, &fx_, &fy_);
      const double ex = fx_ - xd, ey = fy_ - yd;
      if (std::abs(ex) + std::abs(ey) < 1e-15) break;
      const double r2 = x * x + y * y;
      const double rho = 1.0 + k1 * r2 + k2 * r2 * r2;
      const double a = 2.0 * k1 + 4.0 * k2 * r2;
      const double j11 = rho + a * x * x + 2.0 * p1 * y + 6.0 * p2 * x;
      const double j12 = a * x * y + 2.0 * p1 * x + 2.0 * p2 * y;
      const double j22 = rho + a * y * y + 6.0 * p1 * y + 2.0 * p2 * x;
      double det = j11 * j22 - j12 * j12;
      det = std::abs(det) > 1e-300 ? det : 1e-300;
      x = x - (j22 * ex - j12 * ey) / det;
      y = y - (j11 * ey - j12 * ex) / det;
      const double rr = std::hypot(x, y);
      if (rr > r_valid_) {
        x = x * r_valid_ / std::max(rr, 1e-300);
        y = y * r_valid_ / std::max(rr, 1e-300);
      }
    }
  }
  double fx_, fy_;
  radtan_distort(k1, k2, p1, p2, x, y, &fx_, &fy_);
  ok = ok && (std::hypot(fx_ - xd, fy_ - yd) < 1e-10) && (x * x + y * y <= r_valid_ * r_valid_ * (1 + 1e-12));
  ok = ok && std::isfinite(x) && std::isfinite(y);
  if (!ok) {
    set_nan(ray);
    return false;
  }
  if (ray) {
    const double n = std::sqrt(x * x + y * y + 1.0);
    *ray = Eigen::Vector3d(x / n, y / n, 1.0 / n);
  }
  return true;
}

std::unique_ptr<CameraModel> PinholeRadtan::scaled(int width, int height) const {
  CameraModelParams p;
  if (!scale_params(params_, width, height, &p)) return nullptr;
  auto r = create(p);
  return r.ok() ? std::move(r).value() : nullptr;
}

// =============================================================================================== KannalaBrandt4

Result<std::unique_ptr<KannalaBrandt4>> KannalaBrandt4::create(const CameraModelParams& params) {
  if (params.kind != CameraModelKind::kKannalaBrandt4) return invalid_argument(ctx(params) + ": wrong kind");
  RIG_RETURN_IF_ERROR(validate_common(params, 180.0));
  const auto& d = params.distortion;
  const double k1 = d[0], k2 = d[1], k3 = d[2], k4 = d[3];
  // camera_model.py:273 (theta_max_deg default 89) and :286-288. fov_max_deg is the full FoV: theta_max = fov / 2.
  const double theta_max_deg = params.fov_max_deg > 0.0 ? 0.5 * params.fov_max_deg : 89.0;
  auto g = [&](double t) {
    return 1 + 3 * k1 * std::pow(t, 2) + 5 * k2 * std::pow(t, 4) + 7 * k3 * std::pow(t, 6) + 9 * k4 * std::pow(t, 8);
  };
  const double theta_valid = first_nonpositive(g, deg2rad(theta_max_deg));
  return std::unique_ptr<KannalaBrandt4>(new KannalaBrandt4(params, theta_valid));
}

namespace {
// camera_model.py:294-300.
inline double kb_theta_d(const std::vector<double>& k, double t) {
  const double t2 = t * t;
  return t * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
}
inline double kb_dtheta_d(const std::vector<double>& k, double t) {
  const double t2 = t * t;
  return 1.0 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
}
}  // namespace

// camera_model.py:302-312.
bool KannalaBrandt4::project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv, Eigen::Matrix<double, 2, 3>* J) const {
  const auto& k = params_.distortion;
  const double X = p_c.x(), Y = p_c.y(), Z = p_c.z();
  const double r = std::hypot(X, Y);
  const double theta = std::atan2(r, Z);
  if (!(Z > 0 && theta <= theta_valid_)) {
    set_nan(uv);
    return false;
  }
  const double td = kb_theta_d(k, theta);
  const double s = r > 1e-12 ? td / std::max(r, 1e-300) : 0.0;
  if (uv) *uv = Eigen::Vector2d(params_.fx * X * s + params_.cx, params_.fy * Y * s + params_.cy);
  if (J) {
    if (r < 1e-9) {
      // On the axis td/r -> 1/Z: the pinhole Jacobian is the limit.
      *J << params_.fx / Z, 0.0, -params_.fx * X / (Z * Z), 0.0, params_.fy / Z, -params_.fy * Y / (Z * Z);
    } else {
      const double rho2 = r * r + Z * Z;
      const double dtd = kb_dtheta_d(k, theta);
      const double sr = td / r;
      // d(theta)/dX = Z X / (r rho2); ds/dX = (dtd * dtheta/dX * r - td * X / r) / r^2.
      const double dth_dX = Z * X / (r * rho2), dth_dY = Z * Y / (r * rho2), dth_dZ = -r / rho2;
      const double ds_dX = (dtd * dth_dX * r - td * X / r) / (r * r);
      const double ds_dY = (dtd * dth_dY * r - td * Y / r) / (r * r);
      const double ds_dZ = dtd * dth_dZ / r;
      *J << params_.fx * (sr + X * ds_dX), params_.fx * X * ds_dY, params_.fx * X * ds_dZ,  //
          params_.fy * Y * ds_dX, params_.fy * (sr + Y * ds_dY), params_.fy * Y * ds_dZ;
    }
  }
  return true;
}

// camera_model.py:314-325.
bool KannalaBrandt4::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const {
  const auto& k = params_.distortion;
  const double xd = (uv.x() - params_.cx) / params_.fx;
  const double yd = (uv.y() - params_.cy) / params_.fy;
  const double td = std::hypot(xd, yd);
  double theta = 0.0;
  bool ok = monotone_inverse([&](double t) { return kb_theta_d(k, t); }, [&](double t) { return kb_dtheta_d(k, t); },
                             td, theta_valid_, &theta);
  ok = ok && std::abs(kb_theta_d(k, theta) - td) < 1e-12;
  if (!ok) {
    set_nan(ray);
    return false;
  }
  const double s = td > 1e-15 ? std::sin(theta) / std::max(td, 1e-300) : 1.0;
  Eigen::Vector3d v(xd * s, yd * s, std::cos(theta));
  const double n = std::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
  v /= n;
  if (!all_finite(v)) {
    set_nan(ray);
    return false;
  }
  if (ray) *ray = v;
  return true;
}

std::unique_ptr<CameraModel> KannalaBrandt4::scaled(int width, int height) const {
  CameraModelParams p;
  if (!scale_params(params_, width, height, &p)) return nullptr;
  auto r = create(p);
  return r.ok() ? std::move(r).value() : nullptr;
}

// ================================================================================================= DoubleSphere

Result<std::unique_ptr<DoubleSphere>> DoubleSphere::create(const CameraModelParams& params) {
  if (params.kind != CameraModelKind::kDoubleSphere) return invalid_argument(ctx(params) + ": wrong kind");
  RIG_RETURN_IF_ERROR(validate_common(params, 360.0));
  const double xi = params.distortion[0], alpha = params.distortion[1];
  // Usenko et al. 2018 sec. 3.6: alpha in [0, 1]; xi in (-1, 1) in practice (Kalibr / basalt).
  if (!(alpha >= 0.0 && alpha <= 1.0)) return invalid_argument(ctx(params) + ": alpha must be in [0, 1]");
  if (!(xi > -1.0 && xi < 1.0)) return invalid_argument(ctx(params) + ": xi must be in (-1, 1)");
  return std::unique_ptr<DoubleSphere>(new DoubleSphere(params));
}

// Usenko 2018 eqs. (40)-(45).
bool DoubleSphere::project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv, Eigen::Matrix<double, 2, 3>* J) const {
  const double xi = params_.distortion[0], alpha = params_.distortion[1];
  const double x = p_c.x(), y = p_c.y(), z = p_c.z();
  const double r2 = x * x + y * y;
  const double d1 = std::sqrt(r2 + z * z);
  const double zp = xi * d1 + z;
  const double d2 = std::sqrt(r2 + zp * zp);
  const double den = alpha * d2 + (1.0 - alpha) * zp;
  const double w1 = alpha <= 0.5 ? alpha / (1.0 - alpha) : (1.0 - alpha) / alpha;
  const double w2 = (w1 + xi) / std::sqrt(2.0 * w1 * xi + xi * xi + 1.0);
  if (!(d1 > 0.0 && z > -w2 * d1 && den > 0.0 && within_fov(params_, x, y, z))) {
    set_nan(uv);
    return false;
  }
  if (uv) *uv = Eigen::Vector2d(params_.fx * x / den + params_.cx, params_.fy * y / den + params_.cy);
  if (J) {
    const Eigen::Vector3d p(x, y, z);
    const Eigen::RowVector3d dd1 = p.transpose() / d1;
    const Eigen::RowVector3d dzp = xi * dd1 + Eigen::RowVector3d(0, 0, 1);
    const Eigen::RowVector3d dd2 = (Eigen::RowVector3d(x, y, 0) + zp * dzp) / d2;
    const Eigen::RowVector3d dden = alpha * dd2 + (1.0 - alpha) * dzp;
    const double den2 = den * den;
    J->row(0) = params_.fx * (Eigen::RowVector3d(den, 0, 0) - x * dden) / den2;
    J->row(1) = params_.fy * (Eigen::RowVector3d(0, den, 0) - y * dden) / den2;
  }
  return true;
}

// Usenko 2018 eqs. (46)-(47).
bool DoubleSphere::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const {
  const double xi = params_.distortion[0], alpha = params_.distortion[1];
  const double mx = (uv.x() - params_.cx) / params_.fx;
  const double my = (uv.y() - params_.cy) / params_.fy;
  const double r2 = mx * mx + my * my;
  if (alpha > 0.5 && r2 > 1.0 / (2.0 * alpha - 1.0)) {
    set_nan(ray);
    return false;
  }
  const double mz = (1.0 - alpha * alpha * r2) / (alpha * std::sqrt(1.0 - (2.0 * alpha - 1.0) * r2) + 1.0 - alpha);
  const double mz2 = mz * mz;
  const double disc = mz2 + (1.0 - xi * xi) * r2;
  if (!(disc >= 0.0)) {
    set_nan(ray);
    return false;
  }
  const double k = (mz * xi + std::sqrt(disc)) / (mz2 + r2);
  Eigen::Vector3d v(k * mx, k * my, k * mz - xi);
  const double n = v.norm();
  if (!(n > 0.0) || !all_finite(v)) {
    set_nan(ray);
    return false;
  }
  v /= n;
  if (!within_fov(params_, v.x(), v.y(), v.z())) {
    set_nan(ray);
    return false;
  }
  if (ray) *ray = v;
  return true;
}

std::unique_ptr<CameraModel> DoubleSphere::scaled(int width, int height) const {
  CameraModelParams p;
  if (!scale_params(params_, width, height, &p)) return nullptr;
  auto r = create(p);
  return r.ok() ? std::move(r).value() : nullptr;
}

// ========================================================================================================= Eucm

Result<std::unique_ptr<Eucm>> Eucm::create(const CameraModelParams& params) {
  if (params.kind != CameraModelKind::kEucm) return invalid_argument(ctx(params) + ": wrong kind");
  RIG_RETURN_IF_ERROR(validate_common(params, 360.0));
  const double alpha = params.distortion[0], beta = params.distortion[1];
  if (!(alpha >= 0.0 && alpha <= 1.0)) return invalid_argument(ctx(params) + ": alpha must be in [0, 1]");
  if (!(beta > 0.0)) return invalid_argument(ctx(params) + ": beta must be > 0");
  return std::unique_ptr<Eucm>(new Eucm(params));
}

// Usenko 2018 eqs. (24)-(27). Valid set: den > 0, and for alpha > 0.5 the point must lie on the branch the
// unprojection returns, z / den >= (alpha - 1) / (2 alpha - 1) (for beta = 1 this is the UCM rule
// z > -((1 - alpha) / alpha) |p|, eq. (22)).
bool Eucm::project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv, Eigen::Matrix<double, 2, 3>* J) const {
  const double alpha = params_.distortion[0], beta = params_.distortion[1];
  const double x = p_c.x(), y = p_c.y(), z = p_c.z();
  const double d = std::sqrt(beta * (x * x + y * y) + z * z);
  const double den = alpha * d + (1.0 - alpha) * z;
  bool ok = den > 0.0 && within_fov(params_, x, y, z);
  if (ok && alpha > 0.5) ok = z / den >= (alpha - 1.0) / (2.0 * alpha - 1.0);
  if (!ok) {
    set_nan(uv);
    return false;
  }
  if (uv) *uv = Eigen::Vector2d(params_.fx * x / den + params_.cx, params_.fy * y / den + params_.cy);
  if (J) {
    const Eigen::RowVector3d dd = Eigen::RowVector3d(beta * x, beta * y, z) / d;
    const Eigen::RowVector3d dden = alpha * dd + Eigen::RowVector3d(0, 0, 1.0 - alpha);
    const double den2 = den * den;
    J->row(0) = params_.fx * (Eigen::RowVector3d(den, 0, 0) - x * dden) / den2;
    J->row(1) = params_.fy * (Eigen::RowVector3d(0, den, 0) - y * dden) / den2;
  }
  return true;
}

// Usenko 2018 eqs. (28)-(31).
bool Eucm::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const {
  const double alpha = params_.distortion[0], beta = params_.distortion[1];
  const double mx = (uv.x() - params_.cx) / params_.fx;
  const double my = (uv.y() - params_.cy) / params_.fy;
  const double r2 = mx * mx + my * my;
  if (alpha > 0.5 && r2 > 1.0 / (beta * (2.0 * alpha - 1.0))) {
    set_nan(ray);
    return false;
  }
  const double mz =
      (1.0 - beta * alpha * alpha * r2) / (alpha * std::sqrt(1.0 - (2.0 * alpha - 1.0) * beta * r2) + 1.0 - alpha);
  Eigen::Vector3d v(mx, my, mz);
  const double n = v.norm();
  if (!(n > 0.0) || !all_finite(v)) {
    set_nan(ray);
    return false;
  }
  v /= n;
  if (!within_fov(params_, v.x(), v.y(), v.z())) {
    set_nan(ray);
    return false;
  }
  if (ray) *ray = v;
  return true;
}

std::unique_ptr<CameraModel> Eucm::scaled(int width, int height) const {
  CameraModelParams p;
  if (!scale_params(params_, width, height, &p)) return nullptr;
  auto r = create(p);
  return r.ok() ? std::move(r).value() : nullptr;
}

// =============================================================================================== VirtualPinhole

Result<std::unique_ptr<VirtualPinhole>> VirtualPinhole::create(const CameraModelParams& params) {
  if (params.kind != CameraModelKind::kVirtualPinhole) return invalid_argument(ctx(params) + ": wrong kind");
  RIG_RETURN_IF_ERROR(validate_common(params, 180.0));
  return std::unique_ptr<VirtualPinhole>(new VirtualPinhole(params));
}

// camera_model.py:365-372 (without the in-image condition; see the file comment).
bool VirtualPinhole::project(const Eigen::Vector3d& p_c, Eigen::Vector2d* uv, Eigen::Matrix<double, 2, 3>* J) const {
  const double Z = p_c.z();
  if (!(Z > 1e-9) || !within_fov(params_, p_c.x(), p_c.y(), Z)) {
    set_nan(uv);
    return false;
  }
  if (uv) *uv = Eigen::Vector2d(params_.fx * p_c.x() / Z + params_.cx, params_.fy * p_c.y() / Z + params_.cy);
  if (J) {
    *J << params_.fx / Z, 0.0, -params_.fx * p_c.x() / (Z * Z), 0.0, params_.fy / Z, -params_.fy * p_c.y() / (Z * Z);
  }
  return true;
}

// camera_model.py:374-379.
bool VirtualPinhole::unproject(const Eigen::Vector2d& uv, Eigen::Vector3d* ray) const {
  const double x = (uv.x() - params_.cx) / params_.fx;
  const double y = (uv.y() - params_.cy) / params_.fy;
  const double n = std::sqrt(x * x + y * y + 1.0);
  Eigen::Vector3d v(x / n, y / n, 1.0 / n);
  if (!all_finite(v) || !within_fov(params_, v.x(), v.y(), v.z())) {
    set_nan(ray);
    return false;
  }
  if (ray) *ray = v;
  return true;
}

std::unique_ptr<CameraModel> VirtualPinhole::scaled(int width, int height) const {
  CameraModelParams p;
  if (!scale_params(params_, width, height, &p)) return nullptr;
  auto r = create(p);
  return r.ok() ? std::move(r).value() : nullptr;
}

Eigen::Matrix3d VirtualPinhole::K() const {
  Eigen::Matrix3d k;
  k << params_.fx, 0.0, params_.cx, 0.0, params_.fy, params_.cy, 0.0, 0.0, 1.0;
  return k;
}

// ===================================================================================================== factory

Result<std::unique_ptr<CameraModel>> make_camera_model(const CameraModelParams& params) {
  switch (params.kind) {
    case CameraModelKind::kPinholeRadtan: {
      RIG_ASSIGN_OR_RETURN(auto m, PinholeRadtan::create(params));
      return std::unique_ptr<CameraModel>(std::move(m));
    }
    case CameraModelKind::kKannalaBrandt4: {
      RIG_ASSIGN_OR_RETURN(auto m, KannalaBrandt4::create(params));
      return std::unique_ptr<CameraModel>(std::move(m));
    }
    case CameraModelKind::kDoubleSphere: {
      RIG_ASSIGN_OR_RETURN(auto m, DoubleSphere::create(params));
      return std::unique_ptr<CameraModel>(std::move(m));
    }
    case CameraModelKind::kEucm: {
      RIG_ASSIGN_OR_RETURN(auto m, Eucm::create(params));
      return std::unique_ptr<CameraModel>(std::move(m));
    }
    case CameraModelKind::kVirtualPinhole: {
      RIG_ASSIGN_OR_RETURN(auto m, VirtualPinhole::create(params));
      return std::unique_ptr<CameraModel>(std::move(m));
    }
  }
  return invalid_argument("make_camera_model: unknown kind " + std::to_string(static_cast<int>(params.kind)));
}

}  // namespace rig
