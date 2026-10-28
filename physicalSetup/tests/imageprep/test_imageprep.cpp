// SPDX-License-Identifier: MIT
// T4: ImagePrep from a RigConfig, frame_to_bgr formats, error paths.
#include <cstring>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "golden_util.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/imageprep/imageprep.hpp"

namespace rig {
namespace {

CameraModelParams nominal_uw() {
  return test::parse_model_line("1 1640 1232 735.0 735.0 819.5 615.5 160.0 4 0.0 0.0 0.0 0.0");
}
CameraModelParams nominal_main() {
  return test::parse_model_line("0 1920 1080 1478.0 1478.0 959.5 539.5 0.0 4 0.0 0.0 0.0 0.0");
}

RigConfig rig() {
  RigConfig c;
  c.cameras["main"].role = "main";
  c.cameras["main"].model = nominal_main();
  c.cameras["uw"].role = "ultrawide";
  c.cameras["uw"].model = nominal_uw();
  VirtualCameraConfig v;
  v.model = test::iphone_uw_virtual();
  v.min_coverage = 1.0;
  c.virtual_cameras["uw"] = v;
  return c;
}

bool same(const cv::Mat& a, const cv::Mat& b) {
  return a.size() == b.size() && a.type() == b.type() && cv::norm(a, b, cv::NORM_INF) == 0.0;
}

TEST(ImagePrep, CreatesModesAndPreparesLikeTheParts) {
  auto prep = ImagePrep::create(rig());
  ASSERT_TRUE(prep.ok()) << prep.status().to_string();
  EXPECT_EQ((*prep)->out_size(), 224);
  EXPECT_EQ(*(*prep)->mode("main"), PrepMode::kLetterbox);
  EXPECT_EQ(*(*prep)->mode("uw"), PrepMode::kRemapLetterbox);
  EXPECT_EQ((*prep)->mode("x").status().code(), StatusCode::kNotFound);
  EXPECT_EQ((*prep)->remap("main").status().code(), StatusCode::kNotFound);
  auto table = (*prep)->remap("uw");
  ASSERT_TRUE(table.ok());
  EXPECT_EQ((*table)->coverage, 1.0);
  EXPECT_EQ((*table)->src_size, cv::Size(1640, 1232));
  EXPECT_EQ((*table)->dst_size, cv::Size(640, 480));

  const cv::Mat uw = test::synth_image(1640, 1232);
  auto out = (*prep)->prepare("uw", uw, true);
  ASSERT_TRUE(out.ok()) << out.status().to_string();
  cv::Mat virt;
  ASSERT_TRUE((*table)->apply(uw, virt).ok());
  EXPECT_TRUE(same(*out, *letterbox_square_resize(virt, 224, true)));

  const cv::Mat main = test::synth_image(1280, 720);  // stored size; letterbox accepts any size
  auto m = (*prep)->prepare("main", main, false);
  ASSERT_TRUE(m.ok());
  EXPECT_TRUE(same(*m, *letterbox_square_resize(main, 224, false)));
  EXPECT_EQ((*prep)->prepare("uw", main).status().code(), StatusCode::kInvalidArgument);  // wrong size for remap
  EXPECT_EQ((*prep)->prepare("nope", main).status().code(), StatusCode::kNotFound);
}

// Header contract: ImagePrep is immutable after create() and const-thread-safe (checked under TSan).
TEST(ImagePrep, ConcurrentPrepareIsDeterministic) {
  auto prep = ImagePrep::create(rig());
  ASSERT_TRUE(prep.ok());
  const cv::Mat uw = test::synth_image(1640, 1232);
  const cv::Mat main = test::synth_image(1920, 1080);
  const cv::Mat ref_uw = *(*prep)->prepare("uw", uw, true);
  const cv::Mat ref_main = *(*prep)->prepare("main", main, false);
  std::vector<int> ok(4, 0);
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      bool good = true;
      for (int i = 0; i < 3; ++i) {
        auto a = (*prep)->prepare("uw", uw, true);
        auto b = (*prep)->prepare("main", main, false);
        good = good && a.ok() && b.ok() && cv::norm(*a, ref_uw, cv::NORM_INF) == 0 &&
               cv::norm(*b, ref_main, cv::NORM_INF) == 0;
      }
      ok[static_cast<size_t>(t)] = good ? 1 : 0;
    });
  }
  for (auto& th : threads) th.join();
  for (int v : ok) EXPECT_EQ(v, 1);
}

TEST(ImagePrep, SwapThenLetterboxEqualsLetterboxThenSwap) {
  const cv::Mat src = cv::imread(test::imageprep_data_dir() + "/main_crop_211x158.png", cv::IMREAD_COLOR);
  ASSERT_FALSE(src.empty());
  cv::Mat rgb;
  cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
  EXPECT_TRUE(same(*letterbox_square_resize(src, 224, true), *letterbox_square_resize(rgb, 224, false)));
}

TEST(ImagePrep, RejectsLowCoverageAndBadConfig) {
  RigConfig c = rig();
  c.virtual_cameras["main"].model = test::iphone_uw_virtual();  // the 66 deg main cannot cover 105 deg
  c.virtual_cameras["main"].min_coverage = 0.9;
  auto p = ImagePrep::create(c);
  EXPECT_EQ(p.status().code(), StatusCode::kFailedPrecondition) << p.status().to_string();
  c.virtual_cameras["main"].min_coverage = 0.0;
  EXPECT_TRUE(ImagePrep::create(c).ok());

  RigConfig d = rig();
  d.virtual_cameras["ghost"].model = test::iphone_uw_virtual();
  EXPECT_EQ(ImagePrep::create(d).status().code(), StatusCode::kInvalidArgument);
  RigConfig e = rig();
  e.virtual_cameras["uw"].model.kind = CameraModelKind::kKannalaBrandt4;
  EXPECT_EQ(ImagePrep::create(e).status().code(), StatusCode::kInvalidArgument);
  RigConfig g = rig();
  g.imageprep.out_size = 0;
  EXPECT_EQ(ImagePrep::create(g).status().code(), StatusCode::kInvalidArgument);
}

TEST(ImagePrep, LetterboxErrors) {
  EXPECT_EQ(letterbox_square_resize(cv::Mat()).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(letterbox_square_resize(cv::Mat(4, 4, CV_32FC3)).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(letterbox_square_resize(cv::Mat(4, 4, CV_8UC3), 0).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(letterbox_square_resize(cv::Mat(4, 4, CV_8UC1), 8, true).status().code(), StatusCode::kInvalidArgument);
  // Non-continuous ROI input works and equals the cloned input.
  const cv::Mat big = test::synth_image(300, 200);
  const cv::Mat roi = big(cv::Rect(10, 20, 150, 100));
  EXPECT_TRUE(same(*letterbox_square_resize(roi, 64), *letterbox_square_resize(roi.clone(), 64)));
}

// Builds a frame over a copy of `bytes` with the given planes.
ImageFrame make_frame(PixelFormat fmt, int w, int h, const std::vector<uint8_t>& bytes, std::vector<Plane> planes) {
  ImageFrame f;
  f.fmt = fmt;
  f.width = w;
  f.height = h;
  std::shared_ptr<uint8_t[]> buf(new uint8_t[bytes.size()]);
  std::memcpy(buf.get(), bytes.data(), bytes.size());
  f.data = buf;
  f.size = bytes.size();
  for (size_t i = 0; i < planes.size(); ++i) f.planes[i] = planes[i];
  f.t_mid_ns = 1;
  return f;
}

TEST(FrameToBgr, Nv12WithStrideMatchesContiguousCvtColor) {
  const int W = 64, H = 48, stride = 80;
  cv::Mat yuv(H * 3 / 2, W, CV_8UC1);
  cv::randu(yuv, 0, 256);
  cv::Mat ref;
  cv::cvtColor(yuv, ref, cv::COLOR_YUV2BGR_NV12);
  std::vector<uint8_t> bytes(static_cast<size_t>(stride) * H * 3 / 2 + 7, 0xEE);
  for (int r = 0; r < H * 3 / 2; ++r) std::memcpy(&bytes[static_cast<size_t>(r) * stride], yuv.ptr(r), W);
  const auto f = make_frame(PixelFormat::kNV12, W, H, bytes,
                            {Plane{0, stride, W, H}, Plane{static_cast<size_t>(stride) * H, stride, W / 2, H / 2}});
  auto bgr = frame_to_bgr(f);
  ASSERT_TRUE(bgr.ok()) << bgr.status().to_string();
  EXPECT_TRUE(same(*bgr, ref));
}

TEST(FrameToBgr, PackedFormatsAndJpeg) {
  const cv::Mat img = test::synth_image(32, 16);
  std::vector<uint8_t> bgr_bytes(img.data, img.data + img.total() * 3);
  auto b = frame_to_bgr(make_frame(PixelFormat::kBGR8, 32, 16, bgr_bytes, {Plane{0, 96, 32, 16}}));
  ASSERT_TRUE(b.ok());
  EXPECT_TRUE(same(*b, img));
  cv::Mat rgb;
  cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
  std::vector<uint8_t> rgb_bytes(rgb.data, rgb.data + rgb.total() * 3);
  auto r = frame_to_bgr(make_frame(PixelFormat::kRGB8, 32, 16, rgb_bytes, {Plane{0, 0, 32, 16}}));
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(same(*r, img));
  std::vector<uint8_t> gray(32 * 16, 77);
  auto g = frame_to_bgr(make_frame(PixelFormat::kGRAY8, 32, 16, gray, {Plane{0, 32, 32, 16}}));
  ASSERT_TRUE(g.ok());
  EXPECT_EQ(g->type(), CV_8UC3);
  EXPECT_EQ(g->at<cv::Vec3b>(3, 3), cv::Vec3b(77, 77, 77));
  std::vector<uint8_t> yuyv(32 * 16 * 2, 128);
  auto y = frame_to_bgr(make_frame(PixelFormat::kYUYV, 32, 16, yuyv, {Plane{0, 64, 32, 16}}));
  ASSERT_TRUE(y.ok());
  std::vector<uint8_t> jpg;
  ASSERT_TRUE(cv::imencode(".jpg", img, jpg));
  auto j = frame_to_bgr(make_frame(PixelFormat::kJPEG, 32, 16, jpg, {Plane{0, 0, 32, 16}}));
  ASSERT_TRUE(j.ok()) << j.status().to_string();
  EXPECT_TRUE(same(*j, cv::imdecode(jpg, cv::IMREAD_COLOR)));
  EXPECT_FALSE(frame_to_bgr(make_frame(PixelFormat::kJPEG, 33, 16, jpg, {Plane{}})).ok());  // size mismatch
}

TEST(FrameToBgr, RejectsShortBuffersAndUnknown) {
  std::vector<uint8_t> small(100, 0);
  EXPECT_EQ(frame_to_bgr(make_frame(PixelFormat::kBGR8, 32, 16, small, {Plane{0, 96, 32, 16}})).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(frame_to_bgr(make_frame(PixelFormat::kBGR8, 32, 1, small, {Plane{0, 50, 32, 1}})).status().code(),
            StatusCode::kInvalidArgument);  // stride < 3 * W
  EXPECT_FALSE(frame_to_bgr(make_frame(PixelFormat::kUnknown, 4, 4, small, {})).ok());
  EXPECT_FALSE(frame_to_bgr(make_frame(PixelFormat::kNV12, 5, 4, small, {})).ok());
  EXPECT_FALSE(frame_to_bgr(ImageFrame{}).ok());
  std::vector<uint8_t> junk(64, 1);
  EXPECT_FALSE(frame_to_bgr(make_frame(PixelFormat::kJPEG, 4, 4, junk, {Plane{}})).ok());
}

}  // namespace
}  // namespace rig
