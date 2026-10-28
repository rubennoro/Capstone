// SPDX-License-Identifier: MIT
/// @file rig/imageprep/imageprep.hpp
/// @brief Policy-input image preparation shared by the offline `images` stage and deployment (PIPELINE 2 "Image
/// views", 7.2 stage 6). FROZEN CONTRACT (T0); declarations only, implemented by T4 in common/imageprep/.
///
/// Parity target: iPhUMI `iPhUMI/python_package/iphumi/common/cv_util.py:5-34`
/// (`get_image_transform_with_border`, mode "rgb"): pad to a square canvas with BORDER_CONSTANT value 0, the image
/// centred with top = (size - ih) // 2, bottom = size - ih - top, left = (size - iw) // 2, right = size - iw - left
/// (size = max(iw, ih)), then cv2.resize to (out, out) with INTER_AREA. Channel order is unchanged unless the
/// caller asks for a swap (the Python `bgr_to_rgb` flag). Output must be bit-exact with Python cv2 (T4 golden test).
/// Threading: free functions are thread-safe; ImagePrep is immutable after create() and const-thread-safe.
#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "rig/camera_model/camera_model.hpp"
#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

struct RigConfig;  // rig/config/rig_config.hpp

/// Square-canvas padding computed exactly as cv_util.py:13-17.
struct LetterboxGeometry {
  int size = 0;    ///< max(iw, ih).
  int top = 0;     ///< (size - ih) / 2 (floor).
  int bottom = 0;  ///< size - ih - top.
  int left = 0;    ///< (size - iw) / 2 (floor).
  int right = 0;   ///< size - iw - left.
};

/// Pure arithmetic (implemented inline: every consumer needs the identical rounding).
constexpr LetterboxGeometry letterbox_geometry(int iw, int ih) noexcept {
  LetterboxGeometry g;
  g.size = iw > ih ? iw : ih;
  g.top = (g.size - ih) / 2;
  g.bottom = g.size - ih - g.top;
  g.left = (g.size - iw) / 2;
  g.right = g.size - iw - g.left;
  return g;
}

/// Letterbox to square (black, centred) and resize to out_size x out_size with INTER_AREA (cv_util.py:5-34).
/// @param src CV_8UC3, any size (CV_8UC1 also accepted). @param swap_rb swap channels 0 and 2 after the resize
/// (Python bgr_to_rgb). kInvalidArgument for empty input or unsupported type.
Result<cv::Mat> letterbox_square_resize(const cv::Mat& src, int out_size = 224, bool swap_rb = false);

/// Converts an ImageFrame (NV12 / GRAY8 / BGR8 / RGB8 / JPEG) to a CV_8UC3 BGR image (copy).
Result<cv::Mat> frame_to_bgr(const ImageFrame& frame);

/// What happens to each camera's frames before the policy sees them (PIPELINE 2, D5).
enum class PrepMode : uint8_t {
  kLetterbox = 0,        ///< ("letterbox") main: native view, letterbox -> out_size.
  kRemapLetterbox = 1,   ///< ("remap_letterbox") uw: remap to the virtual camera, then letterbox -> out_size.
};

/// Prepared 224 images for one rig: built once from the config, then applied per frame.
class ImagePrep {
 public:
  /// Builds the remap tables for every camera with a virtual_cameras entry (uw -> virtual iPhone ultrawide).
  /// Fails if a remap's coverage is below that entry's min_coverage.
  static Result<std::unique_ptr<ImagePrep>> create(const RigConfig& config);
  virtual ~ImagePrep() = default;

  /// Output side length (config imageprep.out_size, default 224).
  virtual int out_size() const = 0;
  /// Mode applied to `camera_id`; kNotFound if the camera is unknown.
  virtual Result<PrepMode> mode(const std::string& camera_id) const = 0;
  /// The physical -> virtual remap for `camera_id` (kNotFound if that camera is letterbox-only).
  virtual Result<const RemapTable*> remap(const std::string& camera_id) const = 0;
  /// BGR image of the physical camera at its native size -> out_size x out_size BGR (RGB if swap_rb).
  virtual Result<cv::Mat> prepare(const std::string& camera_id, const cv::Mat& bgr, bool swap_rb = false) const = 0;
};

}  // namespace rig
