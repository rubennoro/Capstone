// SPDX-License-Identifier: MIT
// T4: model round trips, Jacobians (vs central differences), DS/EUCM identities, validation, scaling.
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "rig/camera_model/camera_model.hpp"

namespace rig {
namespace {

CameraModelParams P(CameraModelKind k, int w, int h, double fx, double fy, double cx, double cy,
                    std::vector<double> d, double fov = 0.0) {
  CameraModelParams p;
  p.kind = k;
  p.width = w;
  p.height = h;
  p.fx = fx;
  p.fy = fy;
  p.cx = cx;
  p.cy = cy;
  p.distortion = std::move(d);
  p.fov_max_deg = fov;
  return p;
}

std::vector<CameraModelParams> all_models() {
  return {
      P(CameraModelKind::kPinholeRadtan, 1920, 1080, 1450.3, 1452.1, 951.2, 545.7, {-0.12, 0.31, 8e-4, -5e-4}),
      P(CameraModelKind::kKannalaBrandt4, 1640, 1232, 735.0, 735.0, 819.5, 615.5, {0, 0, 0, 0}, 160.0),
      P(CameraModelKind::kKannalaBrandt4, 1640, 1232, 712.4, 713.9, 822.1, 611.3, {0.031, -0.012, 0.004, -0.0011}),
      // Kalibr-like values for a ~160 deg lens (Usenko 2018 Table 1 orders of magnitude).
      P(CameraModelKind::kDoubleSphere, 1640, 1232, 420.0, 421.0, 820.3, 614.9, {-0.21, 0.59}, 170.0),
      P(CameraModelKind::kEucm, 1640, 1232, 560.0, 561.0, 818.7, 616.2, {0.62, 1.07}, 170.0),
      P(CameraModelKind::kVirtualPinhole, 640, 480, 246.64919, 246.64919, 320.16495, 239.53679, {}),
  };
}

TEST(CameraModels, RoundTripAllKinds) {
  std::mt19937_64 rng(1);
  for (const auto& p : all_models()) {
    SCOPED_TRACE(to_string(p.kind));
    auto m = make_camera_model(p);
    ASSERT_TRUE(m.ok()) << m.status().to_string();
    std::uniform_real_distribution<double> U(0, p.width - 1), V(0, p.height - 1);
    int n_ok = 0;
    for (int i = 0; i < 2000; ++i) {
      const Eigen::Vector2d uv(U(rng), V(rng));
      Eigen::Vector3d ray;
      if (!(*m)->unproject(uv, &ray)) continue;
      ++n_ok;
      EXPECT_NEAR(ray.norm(), 1.0, 1e-12);
      Eigen::Vector2d back;
      ASSERT_TRUE((*m)->project(ray * 3.7, &back)) << uv.transpose();
      EXPECT_LT((back - uv).norm(), 1e-8) << uv.transpose();
    }
    EXPECT_GT(n_ok, 1000);
  }
}

TEST(CameraModels, JacobiansMatchCentralDifferences) {
  std::mt19937_64 rng(2);
  std::uniform_real_distribution<double> A(-1.0, 1.0);
  for (const auto& p : all_models()) {
    SCOPED_TRACE(to_string(p.kind));
    auto m = make_camera_model(p);
    ASSERT_TRUE(m.ok());
    int n = 0;
    for (int i = 0; i < 500; ++i) {
      const Eigen::Vector3d pt(A(rng), A(rng), 0.3 + std::abs(A(rng)));
      Eigen::Vector2d uv;
      Eigen::Matrix<double, 2, 3> J;
      if (!(*m)->project(pt, &uv, &J)) continue;
      Eigen::Matrix<double, 2, 3> Jn;
      bool all = true;
      for (int k = 0; k < 3; ++k) {
        const double h = 1e-6;
        Eigen::Vector3d a = pt, b = pt;
        a[k] += h;
        b[k] -= h;
        Eigen::Vector2d ua, ub;
        all = all && (*m)->project(a, &ua) && (*m)->project(b, &ub);
        Jn.col(k) = (ua - ub) / (2 * h);
      }
      if (!all) continue;
      ++n;
      EXPECT_LT((J - Jn).cwiseAbs().maxCoeff(), 1e-4 * std::max(1.0, Jn.cwiseAbs().maxCoeff())) << pt.transpose();
    }
    EXPECT_GT(n, 100);
  }
  // On-axis KB4 Jacobian (r -> 0 branch).
  auto kb = make_camera_model(all_models()[1]);
  Eigen::Vector2d uv;
  Eigen::Matrix<double, 2, 3> J;
  ASSERT_TRUE((*kb)->project(Eigen::Vector3d(0, 0, 2), &uv, &J));
  EXPECT_NEAR(J(0, 0), 735.0 / 2, 1e-9);
  EXPECT_NEAR(J(1, 1), 735.0 / 2, 1e-9);
}

TEST(CameraModels, DoubleSphereXi0EqualsEucmBeta1EqualsUcm) {
  // DS with xi = 0 and EUCM with beta = 1 are both the unified camera model (Usenko 2018 sec. 3.3-3.5).
  auto ds = make_camera_model(P(CameraModelKind::kDoubleSphere, 800, 600, 300, 301, 400, 300, {0.0, 0.57}));
  auto eu = make_camera_model(P(CameraModelKind::kEucm, 800, 600, 300, 301, 400, 300, {0.57, 1.0}));
  ASSERT_TRUE(ds.ok() && eu.ok());
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> A(-2.0, 2.0);
  for (int i = 0; i < 1000; ++i) {
    const Eigen::Vector3d p(A(rng), A(rng), A(rng));
    Eigen::Vector2d a, b;
    const bool oa = (*ds)->project(p, &a), ob = (*eu)->project(p, &b);
    ASSERT_EQ(oa, ob) << p.transpose();
    if (oa) {
      EXPECT_LT((a - b).norm(), 1e-9);
    }
  }
  // alpha = 0 -> plain pinhole.
  auto d0 = make_camera_model(P(CameraModelKind::kDoubleSphere, 800, 600, 300, 300, 400, 300, {0.0, 0.0}));
  Eigen::Vector2d uv;
  ASSERT_TRUE((*d0)->project(Eigen::Vector3d(0.2, -0.1, 1.0), &uv));
  EXPECT_NEAR(uv.x(), 460.0, 1e-9);
  EXPECT_NEAR(uv.y(), 270.0, 1e-9);
}

TEST(CameraModels, FovLimitsAndDomain) {
  auto kb = make_camera_model(all_models()[1]);  // fov 160
  Eigen::Vector2d uv;
  const double t79 = 79.9 * M_PI / 180, t81 = 80.1 * M_PI / 180;
  EXPECT_TRUE((*kb)->project(Eigen::Vector3d(std::sin(t79), 0, std::cos(t79)), &uv));
  EXPECT_FALSE((*kb)->project(Eigen::Vector3d(std::sin(t81), 0, std::cos(t81)), &uv));
  EXPECT_TRUE(std::isnan(uv.x()));
  EXPECT_FALSE((*kb)->project(Eigen::Vector3d(0, 0, -1), &uv));
  auto vp = make_camera_model(all_models()[5]);
  EXPECT_FALSE((*vp)->project(Eigen::Vector3d(0, 0, 0), &uv));
  EXPECT_TRUE((*vp)->project(Eigen::Vector3d(10, 0, 1), &uv));  // outside the image is still projectable
  EXPECT_FALSE((*vp)->in_image(uv));
  auto ds = make_camera_model(all_models()[3]);
  EXPECT_FALSE((*ds)->project(Eigen::Vector3d(0, 0, -1), &uv));
  auto eu = make_camera_model(all_models()[4]);
  EXPECT_FALSE((*eu)->project(Eigen::Vector3d(0, 0, -1), &uv));
  Eigen::Vector3d ray;
  EXPECT_FALSE((*eu)->unproject(Eigen::Vector2d(1e6, 1e6), &ray));
  EXPECT_FALSE((*ds)->unproject(Eigen::Vector2d(1e6, 1e6), &ray));
  EXPECT_FALSE((*kb)->unproject(Eigen::Vector2d(1e5, 0), &ray));
}

TEST(CameraModels, ValidationErrors) {
  auto bad = all_models()[0];
  bad.distortion.pop_back();
  EXPECT_EQ(make_camera_model(bad).status().code(), StatusCode::kInvalidArgument);
  bad = all_models()[0];
  bad.fx = -1;
  EXPECT_FALSE(make_camera_model(bad).ok());
  bad = all_models()[0];
  bad.cx = kNaN;
  EXPECT_FALSE(make_camera_model(bad).ok());
  bad = all_models()[0];
  bad.width = 0;
  EXPECT_FALSE(make_camera_model(bad).ok());
  bad = all_models()[1];
  bad.fov_max_deg = 180.0;
  EXPECT_FALSE(make_camera_model(bad).ok());
  bad = all_models()[3];
  bad.distortion = {1.5, 0.5};
  EXPECT_FALSE(make_camera_model(bad).ok());
  bad = all_models()[4];
  bad.distortion = {0.5, 0.0};
  EXPECT_FALSE(make_camera_model(bad).ok());
  bad = all_models()[2];
  bad.distortion[1] = kNaN;
  EXPECT_FALSE(make_camera_model(bad).ok());
  EXPECT_FALSE(KannalaBrandt4::create(all_models()[0]).ok());  // wrong kind
}

TEST(CameraModels, ScaledUsesHalfPixelConvention) {
  for (const auto& p : all_models()) {
    auto m = make_camera_model(p);
    ASSERT_TRUE(m.ok());
    auto s = (*m)->scaled(p.width / 2, p.height / 2);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->kind(), p.kind);
    EXPECT_DOUBLE_EQ(s->params().fx, 0.5 * p.fx);
    EXPECT_DOUBLE_EQ(s->params().cx, 0.5 * (p.cx + 0.5) - 0.5);  // camera_model.py:42-44
    EXPECT_EQ(s->params().distortion, p.distortion);
    EXPECT_EQ((*m)->scaled(p.width / 2, p.height), nullptr);  // non-uniform
    EXPECT_EQ((*m)->scaled(0, 0), nullptr);
  }
  auto v = VirtualPinhole::create(all_models()[5]);
  ASSERT_TRUE(v.ok());
  const Eigen::Matrix3d K = (*v)->K();
  EXPECT_EQ(K(0, 0), 246.64919);
  EXPECT_EQ(K(1, 2), 239.53679);
  EXPECT_EQ(K(2, 2), 1.0);
}

}  // namespace
}  // namespace rig
