// SPDX-License-Identifier: MIT
// Full-range YCbCr -> BGR matching libjpeg(-turbo) decoding (T4, R1 F16). Contract: rig/imageprep/color.hpp.
//
// Sources (libjpeg-turbo 2.1.x, the libjpeg.so.8 that Ubuntu 22.04 OpenCV and our encoder link; IJG-derived):
// - jdcolor.c build_ycc_rgb_table / ycc_rgb_convert: SCALEBITS 16, FIX(x) = x * 2^16 + 0.5,
//     R = Y + ((FIX(1.40200) * (Cr-128) + 2^15) >> 16)
//     G = Y + ((-FIX(0.34414) * (Cb-128) + 2^15 - FIX(0.71414) * (Cr-128)) >> 16)
//     B = Y + ((FIX(1.77200) * (Cb-128) + 2^15) >> 16), clamped to [0, 255].
// - jdsample.c h2v2_fancy_upsample: colsum = 3 * nearer_row + further_row; even output column
//   (3 * colsum[j] + colsum[j-1] + 8) >> 4, odd (3 * colsum[j] + colsum[j+1] + 7) >> 4; first column
//   (4 * colsum + 8) >> 4, last column (4 * colsum + 7) >> 4.
// - jdmainct.c (make_funny_pointers / set_bottom_pointers): above the first and below the last chroma row the
//   edge row is replicated.
#include "rig/imageprep/color.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <vector>

namespace rig {
namespace {

struct YccTables {
  std::array<int, 256> cr_r{}, cb_b{}, cr_g{}, cb_g{};
  YccTables() {
    constexpr int kScale = 16;
    constexpr int32_t kHalf = int32_t{1} << (kScale - 1);
    auto fix = [](double x) { return static_cast<int32_t>(x * (1 << 16) + 0.5); };
    for (int i = 0; i < 256; ++i) {
      const int32_t x = i - 128;
      cr_r[static_cast<size_t>(i)] = static_cast<int>((fix(1.40200) * x + kHalf) >> kScale);
      cb_b[static_cast<size_t>(i)] = static_cast<int>((fix(1.77200) * x + kHalf) >> kScale);
      cr_g[static_cast<size_t>(i)] = -fix(0.71414) * x;
      cb_g[static_cast<size_t>(i)] = -fix(0.34414) * x + kHalf;
    }
  }
};

const YccTables& tables() {
  static const YccTables t;
  return t;
}

inline uint8_t clamp8(int v) { return static_cast<uint8_t>(std::min(255, std::max(0, v))); }

inline void ycc_to_bgr(const YccTables& t, int y, int cb, int cr, uint8_t* out) {
  out[0] = clamp8(y + t.cb_b[static_cast<size_t>(cb)]);
  out[1] = clamp8(y + ((t.cb_g[static_cast<size_t>(cb)] + t.cr_g[static_cast<size_t>(cr)]) >> 16));
  out[2] = clamp8(y + t.cr_r[static_cast<size_t>(cr)]);
}

std::atomic<uint64_t> g_unknown_range{0};

}  // namespace

uint64_t frame_to_bgr_unknown_range_count() noexcept { return g_unknown_range.load(std::memory_order_relaxed); }

namespace detail {
// Called by frame_to_bgr (imageprep.cpp); returns the new count.
uint64_t note_unknown_color_range() noexcept { return g_unknown_range.fetch_add(1, std::memory_order_relaxed) + 1; }
}  // namespace detail

Status nv12_full_range_to_bgr(const cv::Mat& y, const cv::Mat& uv, cv::Mat& bgr) {
  if (y.type() != CV_8UC1 || uv.type() != CV_8UC2) return invalid_argument("nv12_full_range_to_bgr: bad plane types");
  const int W = y.cols, H = y.rows;
  if (W < 4 || H < 4 || W % 2 || H % 2) return invalid_argument("nv12_full_range_to_bgr: W, H must be even and >= 4");
  if (uv.cols != W / 2 || uv.rows != H / 2) return invalid_argument("nv12_full_range_to_bgr: UV plane size mismatch");
  const int cw = W / 2, ch = H / 2;
  const YccTables& t = tables();
  cv::Mat out(H, W, CV_8UC3);
  std::vector<int> sum_cb(static_cast<size_t>(cw)), sum_cr(static_cast<size_t>(cw));
  std::vector<int> up_cb(static_cast<size_t>(W)), up_cr(static_cast<size_t>(W));
  for (int r = 0; r < H; ++r) {
    const int c0 = r / 2;
    const int c1 = std::clamp((r % 2 == 0) ? c0 - 1 : c0 + 1, 0, ch - 1);  // jdmainct.c edge replication
    const uint8_t* p0 = uv.ptr<uint8_t>(c0);
    const uint8_t* p1 = uv.ptr<uint8_t>(c1);
    for (int j = 0; j < cw; ++j) {
      sum_cb[static_cast<size_t>(j)] = 3 * p0[2 * j] + p1[2 * j];
      sum_cr[static_cast<size_t>(j)] = 3 * p0[2 * j + 1] + p1[2 * j + 1];
    }
    auto horiz = [cw](const std::vector<int>& s, std::vector<int>& o) {
      o[0] = (s[0] * 4 + 8) >> 4;
      o[1] = (s[0] * 3 + s[1] + 7) >> 4;
      for (int j = 1; j < cw - 1; ++j) {
        const size_t k = static_cast<size_t>(j);
        o[2 * k] = (s[k] * 3 + s[k - 1] + 8) >> 4;
        o[2 * k + 1] = (s[k] * 3 + s[k + 1] + 7) >> 4;
      }
      const size_t l = static_cast<size_t>(cw - 1);
      o[2 * l] = (s[l] * 3 + s[l - 1] + 8) >> 4;
      o[2 * l + 1] = (s[l] * 4 + 7) >> 4;
    };
    horiz(sum_cb, up_cb);
    horiz(sum_cr, up_cr);
    const uint8_t* py = y.ptr<uint8_t>(r);
    uint8_t* po = out.ptr<uint8_t>(r);
    for (int x = 0; x < W; ++x) {
      ycc_to_bgr(t, py[x], up_cb[static_cast<size_t>(x)], up_cr[static_cast<size_t>(x)], po + 3 * x);
    }
  }
  bgr = out;
  return Status::Ok();
}

Status yuyv_full_range_to_bgr(const cv::Mat& yuyv, cv::Mat& bgr) {
  if (yuyv.type() != CV_8UC2 || yuyv.cols % 2 || yuyv.empty()) {
    return invalid_argument("yuyv_full_range_to_bgr: expected CV_8UC2 with an even width");
  }
  const YccTables& t = tables();
  cv::Mat out(yuyv.rows, yuyv.cols, CV_8UC3);
  for (int r = 0; r < yuyv.rows; ++r) {
    const uint8_t* p = yuyv.ptr<uint8_t>(r);  // Y0 U Y1 V
    uint8_t* o = out.ptr<uint8_t>(r);
    for (int x = 0; x < yuyv.cols; x += 2) {
      const int cb = p[2 * x + 1], cr = p[2 * x + 3];
      ycc_to_bgr(t, p[2 * x], cb, cr, o + 3 * x);
      ycc_to_bgr(t, p[2 * x + 2], cb, cr, o + 3 * x + 3);
    }
  }
  bgr = out;
  return Status::Ok();
}

}  // namespace rig
