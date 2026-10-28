// SPDX-License-Identifier: MIT
// T4 / R1 F16: frame_to_bgr honours FrameMeta::color_range.
//  - full: bit-exact with cv::imdecode when the NV12 planes are the JPEG's own decoded YCbCr (libjpeg raw_data_out),
//    i.e. chroma upsampling + colour tables are libjpeg's; against a JPEG encoded from the same NV12 with libjpeg
//    raw_data_in (what the recorder does) only the DCT loss remains, within the bounds asserted below;
//  - limited: unchanged (OpenCV COLOR_YUV2BGR_NV12), unknown: limited + counted.
#include <cstdio>
#include <cstring>
#include <vector>

#include <jpeglib.h>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "golden_util.hpp"
#include "rig/imageprep/color.hpp"
#include "rig/imageprep/imageprep.hpp"

namespace rig {
namespace {

struct Nv12 {
  int w = 0, h = 0;
  std::vector<uint8_t> bytes;  // Y (w*h) then interleaved CbCr (w * h/2)
  cv::Mat y() { return cv::Mat(h, w, CV_8UC1, bytes.data()); }
  cv::Mat uv() { return cv::Mat(h / 2, w / 2, CV_8UC2, bytes.data() + static_cast<size_t>(w) * h); }
};

ImageFrame to_frame(const Nv12& n, ColorRange range) {
  ImageFrame f;
  f.camera_id = "uw";
  f.fmt = PixelFormat::kNV12;
  f.width = n.w;
  f.height = n.h;
  std::shared_ptr<uint8_t[]> buf(new uint8_t[n.bytes.size()]);
  std::memcpy(buf.get(), n.bytes.data(), n.bytes.size());
  f.data = buf;
  f.size = n.bytes.size();
  f.planes[0] = Plane{0, static_cast<size_t>(n.w), n.w, n.h};
  f.planes[1] = Plane{static_cast<size_t>(n.w) * n.h, static_cast<size_t>(n.w), n.w / 2, n.h / 2};
  f.t_mid_ns = 1;
  f.meta.color_range = range;
  return f;
}

// libjpeg raw_data_out decode of a 4:2:0 JPEG: the decoder's own Y/Cb/Cr planes (after IDCT, before upsampling and
// colour conversion). Sizes must be multiples of 16.
Nv12 decode_raw_planes(const std::vector<uint8_t>& jpg) {
  jpeg_decompress_struct d;
  jpeg_error_mgr err;
  d.err = jpeg_std_error(&err);
  jpeg_create_decompress(&d);
  jpeg_mem_src(&d, jpg.data(), static_cast<unsigned long>(jpg.size()));
  jpeg_read_header(&d, TRUE);
  d.raw_data_out = TRUE;
  d.out_color_space = JCS_YCbCr;
  jpeg_start_decompress(&d);
  Nv12 n;
  n.w = static_cast<int>(d.image_width);
  n.h = static_cast<int>(d.image_height);
  EXPECT_EQ(d.comp_info[0].h_samp_factor, 2);
  EXPECT_EQ(d.comp_info[0].v_samp_factor, 2);
  EXPECT_EQ(n.w % 16, 0);
  EXPECT_EQ(n.h % 16, 0);
  cv::Mat Y(n.h, n.w, CV_8UC1), Cb(n.h / 2, n.w / 2, CV_8UC1), Cr(n.h / 2, n.w / 2, CV_8UC1);
  while (d.output_scanline < d.output_height) {
    const int r = static_cast<int>(d.output_scanline);
    JSAMPROW yr[16], cbr[8], crr[8];
    for (int i = 0; i < 16; ++i) yr[i] = Y.ptr<uint8_t>(r + i);
    for (int i = 0; i < 8; ++i) {
      cbr[i] = Cb.ptr<uint8_t>(r / 2 + i);
      crr[i] = Cr.ptr<uint8_t>(r / 2 + i);
    }
    JSAMPARRAY planes[3] = {yr, cbr, crr};
    jpeg_read_raw_data(&d, planes, 16);
  }
  jpeg_finish_decompress(&d);
  jpeg_destroy_decompress(&d);
  n.bytes.resize(static_cast<size_t>(n.w) * n.h * 3 / 2);
  std::memcpy(n.bytes.data(), Y.data, static_cast<size_t>(n.w) * n.h);
  cv::Mat uv = n.uv();
  cv::Mat ch[2] = {Cb, Cr};
  cv::merge(ch, 2, uv);
  return n;
}

// The recorder path: libjpeg raw_data_in from the NV12 planes, 4:2:0, JFIF (full range), ISLOW DCT.
std::vector<uint8_t> encode_raw_planes(Nv12& n, int quality) {
  jpeg_compress_struct c;
  jpeg_error_mgr err;
  c.err = jpeg_std_error(&err);
  jpeg_create_compress(&c);
  unsigned char* buf = nullptr;
  unsigned long size = 0;
  jpeg_mem_dest(&c, &buf, &size);
  c.image_width = static_cast<JDIMENSION>(n.w);
  c.image_height = static_cast<JDIMENSION>(n.h);
  c.input_components = 3;
  c.in_color_space = JCS_YCbCr;
  jpeg_set_defaults(&c);
  jpeg_set_quality(&c, quality, TRUE);
  c.raw_data_in = TRUE;
  c.dct_method = JDCT_ISLOW;
  c.comp_info[0].h_samp_factor = c.comp_info[0].v_samp_factor = 2;
  for (int i = 1; i < 3; ++i) c.comp_info[i].h_samp_factor = c.comp_info[i].v_samp_factor = 1;
  jpeg_start_compress(&c, TRUE);
  cv::Mat ch[2];
  cv::split(n.uv(), ch);
  cv::Mat Y = n.y();
  while (c.next_scanline < c.image_height) {
    const int r = static_cast<int>(c.next_scanline);
    JSAMPROW yr[16], cbr[8], crr[8];
    for (int i = 0; i < 16; ++i) yr[i] = Y.ptr<uint8_t>(r + i);
    for (int i = 0; i < 8; ++i) {
      cbr[i] = ch[0].ptr<uint8_t>(r / 2 + i);
      crr[i] = ch[1].ptr<uint8_t>(r / 2 + i);
    }
    JSAMPARRAY planes[3] = {yr, cbr, crr};
    jpeg_write_raw_data(&c, planes, 16);
  }
  jpeg_finish_compress(&c);
  std::vector<uint8_t> out(buf, buf + size);
  free(buf);  // allocated by jpeg_mem_dest with malloc
  jpeg_destroy_compress(&c);
  return out;
}

// Test images whose YCbCr planes come from a real JPEG decode (real iPhone texture).
std::vector<std::pair<std::string, Nv12>> real_planes() {
  std::vector<std::pair<std::string, Nv12>> out;
  for (const char* name : {"main_crop_448x336", "uw_crop_320x240"}) {
    const cv::Mat bgr = cv::imread(test::imageprep_data_dir() + "/" + name + ".png", cv::IMREAD_COLOR);
    EXPECT_FALSE(bgr.empty());
    std::vector<uint8_t> jpg;
    EXPECT_TRUE(cv::imencode(".jpg", bgr, jpg, {cv::IMWRITE_JPEG_QUALITY, 90}));  // libjpeg default 4:2:0
    out.emplace_back(name, decode_raw_planes(jpg));
  }
  return out;
}

// "ISP-like" full-range NV12 of a real image that never went through a JPEG: OpenCV BGR2YCrCb (JFIF-style full
// range) for Y, CbCr averaged over 2x2 (INTER_AREA). Used for the recorder-path bound.
std::vector<std::pair<std::string, Nv12>> isp_like_planes() {
  std::vector<std::pair<std::string, Nv12>> out;
  for (const char* name : {"main_crop_448x336", "uw_crop_320x240"}) {
    const cv::Mat bgr = cv::imread(test::imageprep_data_dir() + "/" + name + ".png", cv::IMREAD_COLOR);
    cv::Mat ycrcb, ch[3], cr2, cb2;
    cv::cvtColor(bgr, ycrcb, cv::COLOR_BGR2YCrCb);
    cv::split(ycrcb, ch);
    cv::resize(ch[1], cr2, cv::Size(bgr.cols / 2, bgr.rows / 2), 0, 0, cv::INTER_AREA);
    cv::resize(ch[2], cb2, cv::Size(bgr.cols / 2, bgr.rows / 2), 0, 0, cv::INTER_AREA);
    Nv12 n;
    n.w = bgr.cols;
    n.h = bgr.rows;
    n.bytes.resize(static_cast<size_t>(n.w) * n.h * 3 / 2);
    std::memcpy(n.bytes.data(), ch[0].data, static_cast<size_t>(n.w) * n.h);
    cv::Mat uv = n.uv();
    cv::Mat m[2] = {cb2, cr2};
    cv::merge(m, 2, uv);
    out.emplace_back(name, n);
  }
  return out;
}

struct Diff {
  double max = 0, mean = 0;
};
Diff diff(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat d;
  cv::absdiff(a, b, d);
  Diff r;
  r.max = cv::norm(d, cv::NORM_INF);
  const cv::Scalar m = cv::mean(d);
  r.mean = (m[0] + m[1] + m[2]) / 3.0;
  return r;
}

TEST(ColorRange, FullRangeIsBitExactWithLibjpegDecodeOfTheSamePlanes) {
  for (auto& [name, n] : real_planes()) {
    SCOPED_TRACE(name);
    // decode_raw_planes(jpg) are exactly the planes imdecode(jpg) upsamples and converts.
    const cv::Mat bgr = cv::imread(test::imageprep_data_dir() + "/" + name + ".png", cv::IMREAD_COLOR);
    std::vector<uint8_t> jpg;
    ASSERT_TRUE(cv::imencode(".jpg", bgr, jpg, {cv::IMWRITE_JPEG_QUALITY, 90}));
    Nv12 planes = decode_raw_planes(jpg);
    auto live = frame_to_bgr(to_frame(planes, ColorRange::kFull));
    ASSERT_TRUE(live.ok()) << live.status().to_string();
    const cv::Mat ref = cv::imdecode(jpg, cv::IMREAD_COLOR);
    EXPECT_EQ(diff(*live, ref).max, 0.0);
  }
}

// Recorder path. Bounds asserted (measured 2026-10-27 with libjpeg-turbo 2.1.2 on ISP-like planes of two real
// crops: q100 max 3 / mean 0.15; q90 max 19 / mean 1.2 LSB):
//   q100: max <= 4, mean <= 0.3 LSB;  q90 (recorder default): max <= 32, mean <= 2.0 LSB.
// All of it is the encoder's DCT quantisation (the colour path itself is bit-exact, previous test). The limited
// conversion of the same full-range planes is off by ~11 LSB mean (max ~50): the error R1 F16 removes.
TEST(ColorRange, FullRangeMatchesRecorderJpegWithinDctBound) {
  for (auto& [name, n] : isp_like_planes()) {
    SCOPED_TRACE(name);
    auto live = frame_to_bgr(to_frame(n, ColorRange::kFull));
    auto limited = frame_to_bgr(to_frame(n, ColorRange::kLimited));
    ASSERT_TRUE(live.ok() && limited.ok());
    const cv::Mat j100 = cv::imdecode(encode_raw_planes(n, 100), cv::IMREAD_COLOR);
    const cv::Mat j90 = cv::imdecode(encode_raw_planes(n, 90), cv::IMREAD_COLOR);
    const Diff d100 = diff(*live, j100), d90 = diff(*live, j90), dlim = diff(*limited, j90);
    std::printf("%-18s full vs jpeg q100: max %.0f mean %.3f | q90: max %.0f mean %.3f | limited vs q90: max %.0f "
                "mean %.3f\n",
                name.c_str(), d100.max, d100.mean, d90.max, d90.mean, dlim.max, dlim.mean);
    EXPECT_LE(d100.max, 4.0);
    EXPECT_LE(d100.mean, 0.3);
    EXPECT_LE(d90.max, 32.0);
    EXPECT_LE(d90.mean, 2.0);
    EXPECT_GT(dlim.mean, 3.0 * d90.mean);
  }
}

TEST(ColorRange, LimitedUnchangedAndUnknownCounted) {
  auto planes = real_planes();
  Nv12& n = planes[0].second;
  cv::Mat ref;
  cv::cvtColorTwoPlane(n.y(), n.uv(), ref, cv::COLOR_YUV2BGR_NV12);
  const uint64_t before = frame_to_bgr_unknown_range_count();
  auto lim = frame_to_bgr(to_frame(n, ColorRange::kLimited));
  ASSERT_TRUE(lim.ok());
  EXPECT_EQ(diff(*lim, ref).max, 0.0);
  EXPECT_EQ(frame_to_bgr_unknown_range_count(), before);
  ASSERT_TRUE(frame_to_bgr(to_frame(n, ColorRange::kFull)).ok());
  EXPECT_EQ(frame_to_bgr_unknown_range_count(), before);
  auto unk = frame_to_bgr(to_frame(n, ColorRange::kUnknown));
  ASSERT_TRUE(unk.ok());
  EXPECT_EQ(diff(*unk, ref).max, 0.0);
  EXPECT_EQ(frame_to_bgr_unknown_range_count(), before + 1);
}

TEST(ColorRange, FullRangeGrayAndPrimaries) {
  // Neutral chroma: full range maps Y unchanged (0 -> 0, 255 -> 255); limited stretches 16..235.
  Nv12 n;
  n.w = 16;
  n.h = 4;
  n.bytes.assign(16 * 4 * 3 / 2, 128);
  for (int x = 0; x < 16; ++x)
    for (int r = 0; r < 4; ++r) n.bytes[static_cast<size_t>(r * 16 + x)] = static_cast<uint8_t>(x * 17);
  auto full = frame_to_bgr(to_frame(n, ColorRange::kFull));
  ASSERT_TRUE(full.ok());
  for (int x = 0; x < 16; ++x) EXPECT_EQ(full->at<cv::Vec3b>(1, x), cv::Vec3b(x * 17, x * 17, x * 17)) << x;
  // YUYV full with Cb = 255 (max blue): B = Y + 226 clamped (jdcolor.c: (FIX(1.772) * 127 + 2^15) >> 16 = 225).
  cv::Mat yuyv(2, 4, CV_8UC2);
  for (int r = 0; r < 2; ++r) {
    uint8_t* p = yuyv.ptr<uint8_t>(r);
    const uint8_t v[8] = {10, 255, 10, 128, 20, 128, 20, 128};
    std::memcpy(p, v, 8);
  }
  cv::Mat out;
  ASSERT_TRUE(yuyv_full_range_to_bgr(yuyv, out).ok());
  EXPECT_EQ(out.at<cv::Vec3b>(0, 0)[0], 235);  // 10 + 225
  EXPECT_EQ(out.at<cv::Vec3b>(0, 0)[2], 10);   // Cr = 128 -> R = Y
  EXPECT_EQ(out.at<cv::Vec3b>(0, 2), cv::Vec3b(20, 20, 20));
  cv::Mat bad;
  EXPECT_FALSE(nv12_full_range_to_bgr(cv::Mat(4, 6, CV_8UC1), cv::Mat(2, 2, CV_8UC2), bad).ok());
}

}  // namespace
}  // namespace rig
