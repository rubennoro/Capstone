// SPDX-License-Identifier: MIT
// T4 test helpers shared by the tests/imageprep area (golden data written by gen_golden.py).
#pragma once

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "rig/camera_model/model_params.hpp"

namespace rig::test {

/// FNV-1a 64 over the continuous bytes of `m` (gen_golden.py fnv1a64), as 16 lowercase hex digits.
std::string fnv1a64_hex(const cv::Mat& m);
/// Deterministic pattern (gen_golden.py synth_image): ((x*7 + y*13 + c*85 + ((x*y) >> 5)) & 255), CV_8UC3.
cv::Mat synth_image(int width, int height);
/// physicalSetup/tests/imageprep/data
std::string imageprep_data_dir();
/// $RIG_T4_EXTENDED_DIR or "" (extended real-frame set made by `gen_golden.py extended`, scratch only).
std::string extended_dir();
/// Parses "kind w h fx fy cx cy fov n d1..dn" (gen_golden.py model line).
CameraModelParams parse_model_line(const std::string& line);
/// The virtual iPhone ultrawide (firstreal session JSON ultrawideCameraIntrinsics).
CameraModelParams iphone_uw_virtual();
std::vector<std::string> split(const std::string& s, const std::string& sep);

}  // namespace rig::test
