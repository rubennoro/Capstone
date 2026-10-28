// SPDX-License-Identifier: MIT
// Policy-input image preparation (T4). Contract: rig/imageprep/imageprep.hpp; PIPELINE.md section 2 "Image views".
//
// letterbox_square_resize is the exact OpenCV call sequence of iPhUMI
// iPhUMI/python_package/iphumi/common/cv_util.py:13-34 (mode "rgb"): copyMakeBorder(top, bottom, left, right,
// BORDER_CONSTANT, value [0, 0, 0]) then resize((out, out), INTER_AREA), then the optional channel reversal
// (cv_util.py:32-33). It is checked bit-exact against Python cv2 4.12 (iPhUMI environment.yml) on real iPhone frames
// by tests/imageprep/test_golden.cpp.
//
// Colour: iPhUMI decodes its mp4s with PyAV `to_ndarray(format="rgb24")` and stores the letterboxed RGB without a
// swap (gen_replay_buffer_zarr.py:577-581, 599-603). Rig images are BGR (cv::imdecode / frame_to_bgr), so producing
// iPhUMI-layout (RGB) policy images from them needs swap_rb = true. Because copyMakeBorder and INTER_AREA act on
// each channel independently, "swap then letterbox" and "letterbox then swap" give identical bytes.
#include "rig/imageprep/imageprep.hpp"

#include <map>
#include <string>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <spdlog/spdlog.h>

#include "rig/config/rig_config.hpp"
#include "rig/imageprep/color.hpp"

namespace rig {

Result<cv::Mat> letterbox_square_resize(const cv::Mat& src, int out_size, bool swap_rb) {
  if (src.empty()) return invalid_argument("letterbox_square_resize: empty image");
  if (src.type() != CV_8UC3 && src.type() != CV_8UC1) {
    return invalid_argument("letterbox_square_resize: expected CV_8UC3 or CV_8UC1, got type " +
                            std::to_string(src.type()));
  }
  if (out_size <= 0) return invalid_argument("letterbox_square_resize: out_size must be > 0");
  if (swap_rb && src.channels() != 3) return invalid_argument("letterbox_square_resize: swap_rb needs 3 channels");
  const LetterboxGeometry g = letterbox_geometry(src.cols, src.rows);  // cv_util.py:13-17
  try {
    cv::Mat padded, out;
    // cv_util.py:22-30. value=[0, 0, 0] -> Scalar(0, 0, 0, 0). BORDER_ISOLATED: for a ROI src, copyMakeBorder would
    // otherwise pad with the parent image's pixels outside the ROI; a numpy view passed to cv2 never carries its
    // parent, so Python always pads with zeros.
    cv::copyMakeBorder(src, padded, g.top, g.bottom, g.left, g.right, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED,
                       cv::Scalar(0, 0, 0));
    // cv_util.py:31 (INTER_AREA for mode "rgb", cv_util.py:8).
    cv::resize(padded, out, cv::Size(out_size, out_size), 0, 0, cv::INTER_AREA);
    if (swap_rb) cv::cvtColor(out, out, cv::COLOR_BGR2RGB);  // cv_util.py:32-33 (a pure channel reversal)
    return out;
  } catch (const cv::Exception& e) {
    return internal_error(std::string("letterbox_square_resize: ") + e.what());
  }
}

namespace detail {
uint64_t note_unknown_color_range() noexcept;  // color.cpp
}  // namespace detail

namespace {
// R1 F16: kFull -> JFIF conversion; kLimited and kUnknown -> BT.601 limited (unknown is counted, warned once).
bool full_range(const ImageFrame& frame) {
  switch (frame.meta.color_range) {
    case ColorRange::kFull: return true;
    case ColorRange::kLimited: return false;
    case ColorRange::kUnknown: break;
  }
  if (detail::note_unknown_color_range() == 1) {
    spdlog::warn("frame_to_bgr: camera '{}' frame {} has color_range unknown; converting as BT.601 limited range "
                 "(set FrameMeta::color_range; count: rig::frame_to_bgr_unknown_range_count())",
                 frame.camera_id, frame.seq);
  }
  return false;
}
}  // namespace

Result<cv::Mat> frame_to_bgr(const ImageFrame& frame) {
  if (frame.data == nullptr || frame.size == 0) return invalid_argument("frame_to_bgr: empty frame");
  if (frame.width <= 0 || frame.height <= 0) return invalid_argument("frame_to_bgr: bad frame size");
  const int W = frame.width, H = frame.height;
  // Checks that plane i (rows x bytes_per_row) lies inside the buffer and returns a non-owning Mat over it.
  auto plane_mat = [&](int i, int rows, int cols, int type, int bytes_per_row, cv::Mat* m) -> Status {
    const Plane& p = frame.planes[static_cast<std::size_t>(i)];
    const std::size_t stride = p.stride != 0 ? p.stride : static_cast<std::size_t>(bytes_per_row);
    if (stride < static_cast<std::size_t>(bytes_per_row)) {
      return invalid_argument("frame_to_bgr: plane " + std::to_string(i) + " stride too small");
    }
    const std::size_t need = p.offset + stride * static_cast<std::size_t>(rows - 1) + bytes_per_row;
    if (need > frame.size) return invalid_argument("frame_to_bgr: plane " + std::to_string(i) + " exceeds the buffer");
    *m = cv::Mat(rows, cols, type, const_cast<uint8_t*>(frame.data.get() + p.offset), stride);
    return Status::Ok();
  };
  try {
    cv::Mat out;
    switch (frame.fmt) {
      case PixelFormat::kNV12: {
        // NV12 = Y plane + interleaved CbCr at half resolution (types.hpp). Range per FrameMeta::color_range
        // (R1 F16): full -> libjpeg-identical JFIF conversion (rig/imageprep/color.hpp), so a live frame equals the
        // decode of a JPEG encoded from the same planes up to the DCT loss; limited -> OpenCV COLOR_YUV2BGR_NV12
        // (BT.601 video range, unchanged); unknown -> limited, counted and warned once.
        if (W % 2 != 0 || H % 2 != 0) return invalid_argument("frame_to_bgr: NV12 needs even width and height");
        cv::Mat y, uv;
        RIG_RETURN_IF_ERROR(plane_mat(0, H, W, CV_8UC1, W, &y));
        RIG_RETURN_IF_ERROR(plane_mat(1, H / 2, W / 2, CV_8UC2, W, &uv));
        if (full_range(frame)) {
          RIG_RETURN_IF_ERROR(nv12_full_range_to_bgr(y, uv, out).with_context("frame_to_bgr"));
        } else {
          cv::cvtColorTwoPlane(y, uv, out, cv::COLOR_YUV2BGR_NV12);
        }
        break;
      }
      case PixelFormat::kGRAY8: {
        cv::Mat g;
        RIG_RETURN_IF_ERROR(plane_mat(0, H, W, CV_8UC1, W, &g));
        cv::cvtColor(g, out, cv::COLOR_GRAY2BGR);
        break;
      }
      case PixelFormat::kBGR8: {
        cv::Mat m;
        RIG_RETURN_IF_ERROR(plane_mat(0, H, W, CV_8UC3, 3 * W, &m));
        out = m.clone();
        break;
      }
      case PixelFormat::kRGB8: {
        cv::Mat m;
        RIG_RETURN_IF_ERROR(plane_mat(0, H, W, CV_8UC3, 3 * W, &m));
        cv::cvtColor(m, out, cv::COLOR_RGB2BGR);
        break;
      }
      case PixelFormat::kYUYV: {
        if (W % 2 != 0) return invalid_argument("frame_to_bgr: YUYV needs an even width");
        cv::Mat m;
        RIG_RETURN_IF_ERROR(plane_mat(0, H, W, CV_8UC2, 2 * W, &m));
        if (full_range(frame)) {
          RIG_RETURN_IF_ERROR(yuyv_full_range_to_bgr(m, out).with_context("frame_to_bgr"));
        } else {
          cv::cvtColor(m, out, cv::COLOR_YUV2BGR_YUYV);
        }
        break;
      }
      case PixelFormat::kJPEG: {
        const Plane& p = frame.planes[0];
        if (p.offset >= frame.size) return invalid_argument("frame_to_bgr: JPEG plane offset exceeds the buffer");
        const cv::Mat bytes(1, static_cast<int>(frame.size - p.offset), CV_8UC1,
                            const_cast<uint8_t*>(frame.data.get() + p.offset));
        out = cv::imdecode(bytes, cv::IMREAD_COLOR);
        if (out.empty()) return invalid_argument("frame_to_bgr: JPEG decode failed");
        if (out.cols != W || out.rows != H) {
          return invalid_argument("frame_to_bgr: JPEG decodes to " + std::to_string(out.cols) + "x" +
                                  std::to_string(out.rows) + ", frame says " + std::to_string(W) + "x" +
                                  std::to_string(H));
        }
        break;
      }
      case PixelFormat::kUnknown:
        return invalid_argument("frame_to_bgr: unknown pixel format");
    }
    return out;
  } catch (const cv::Exception& e) {
    return internal_error(std::string("frame_to_bgr: ") + e.what());
  }
}

namespace {

class ImagePrepImpl final : public ImagePrep {
 public:
  struct Camera {
    PrepMode mode = PrepMode::kLetterbox;
    RemapTable remap;
  };

  ImagePrepImpl(int out_size, std::map<std::string, Camera> cams) : out_size_(out_size), cams_(std::move(cams)) {}

  int out_size() const override { return out_size_; }

  Result<PrepMode> mode(const std::string& camera_id) const override {
    auto it = cams_.find(camera_id);
    if (it == cams_.end()) return not_found("ImagePrep: unknown camera '" + camera_id + "'");
    return it->second.mode;
  }

  Result<const RemapTable*> remap(const std::string& camera_id) const override {
    auto it = cams_.find(camera_id);
    if (it == cams_.end()) return not_found("ImagePrep: unknown camera '" + camera_id + "'");
    if (it->second.mode != PrepMode::kRemapLetterbox) {
      return not_found("ImagePrep: camera '" + camera_id + "' is letterbox-only (no remap)");
    }
    return &it->second.remap;
  }

  Result<cv::Mat> prepare(const std::string& camera_id, const cv::Mat& bgr, bool swap_rb) const override {
    auto it = cams_.find(camera_id);
    if (it == cams_.end()) return not_found("ImagePrep: unknown camera '" + camera_id + "'");
    if (it->second.mode == PrepMode::kLetterbox) {
      auto r = letterbox_square_resize(bgr, out_size_, swap_rb);
      if (!r.ok()) return r.status().with_context("ImagePrep '" + camera_id + "'");
      return r;
    }
    cv::Mat virt;
    RIG_RETURN_IF_ERROR(it->second.remap.apply(bgr, virt).with_context("ImagePrep '" + camera_id + "'"));
    auto r = letterbox_square_resize(virt, out_size_, swap_rb);
    if (!r.ok()) return r.status().with_context("ImagePrep '" + camera_id + "'");
    return r;
  }

 private:
  int out_size_;
  std::map<std::string, Camera> cams_;
};

}  // namespace

Result<std::unique_ptr<ImagePrep>> ImagePrep::create(const RigConfig& config) {
  if (config.imageprep.out_size <= 0) return invalid_argument("imageprep.out_size: must be > 0");
  std::map<std::string, ImagePrepImpl::Camera> cams;
  for (const auto& [id, cam] : config.cameras) cams[id] = ImagePrepImpl::Camera{};
  for (const auto& [id, vc] : config.virtual_cameras) {
    const std::string where = "virtual_cameras." + id;
    auto cam_it = config.cameras.find(id);
    if (cam_it == config.cameras.end()) return invalid_argument(where + ": no camera '" + id + "' in cameras");
    if (vc.model.kind != CameraModelKind::kVirtualPinhole) return invalid_argument(where + ".model.kind: must be pinhole");
    auto phys = make_camera_model(cam_it->second.model);
    if (!phys.ok()) return phys.status().with_context("cameras." + id + ".model");
    auto virt = VirtualPinhole::create(vc.model);
    if (!virt.ok()) return virt.status().with_context(where + ".model");
    auto table = build_remap(**phys, **virt, vc.R_phys_virt, true);
    if (!table.ok()) return table.status().with_context(where);
    // min_coverage 1.0 means every virtual pixel; compare on pixel counts to avoid a rounding false failure.
    const double n = static_cast<double>(table->dst_size.area());
    if (table->coverage * n + 0.5 < vc.min_coverage * n) {
      return failed_precondition(where + ": remap coverage " + std::to_string(table->coverage) +
                                 " is below min_coverage " + std::to_string(vc.min_coverage));
    }
    cams[id] = ImagePrepImpl::Camera{PrepMode::kRemapLetterbox, std::move(table).value()};
  }
  return std::unique_ptr<ImagePrep>(new ImagePrepImpl(config.imageprep.out_size, std::move(cams)));
}

}  // namespace rig
