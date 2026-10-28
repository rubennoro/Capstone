// SPDX-License-Identifier: MIT
/// @file rig/imageprep/color.hpp
/// @brief YCbCr -> BGR conversions used by frame_to_bgr (T4, new header in its own module; R1 finding F16).
///
/// Why: Argus delivers full-range NV12 (NV12_ER) and the recorder JPEG-encodes those planes as they are (JFIF =
/// full range), so training images are "libjpeg decode of the NV12". For a live frame to give the same BGR, the
/// full-range conversion below reproduces what libjpeg(-turbo) does when decoding a 4:2:0 JPEG with its defaults
/// (as cv::imdecode does):
///   - chroma: h2v2 "fancy" (triangle) upsampling, libjpeg-turbo jdsample.c h2v2_fancy_upsample, with the first and
///     last chroma rows replicated (jdmainct.c context rows);
///   - colour: the integer JFIF tables of jdcolor.c build_ycc_rgb_table (SCALEBITS 16).
/// It is bit-exact with cv::imdecode on JPEG-decoded planes (tests/imageprep/test_color.cpp); against a JPEG made from
/// the same NV12 only the DCT quantisation of the encoder remains.
#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

#include "rig/status.hpp"

namespace rig {

/// Full-range (JFIF) NV12 -> BGR as libjpeg decodes 4:2:0. @param y CV_8UC1 W x H (W, H even, >= 4);
/// @param uv CV_8UC2 W/2 x H/2 interleaved Cb, Cr. Strided (ROI) inputs are fine. @param bgr CV_8UC3 W x H output.
Status nv12_full_range_to_bgr(const cv::Mat& y, const cv::Mat& uv, cv::Mat& bgr);

/// Full-range (JFIF) YUYV (4:2:2 packed, CV_8UC2 W x H) -> BGR with the jdcolor.c tables and sample replication.
Status yuyv_full_range_to_bgr(const cv::Mat& yuyv, cv::Mat& bgr);

/// Number of NV12/YUYV frames frame_to_bgr has converted with FrameMeta::color_range unknown (treated as limited,
/// the previous behaviour). Process-wide, monotonic; ec-doctor / LiveChecks may report it. The first one also logs a
/// warning.
uint64_t frame_to_bgr_unknown_range_count() noexcept;

}  // namespace rig
