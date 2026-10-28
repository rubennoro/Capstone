// SPDX-License-Identifier: MIT
// ArgusCamera: the primary Jetson backend (PLAN T2). Built only with RIG_WITH_ARGUS=ON.
// Compile-checked on the laptop against the L4T r36.4.3 nvidia-l4t-jetson-multimedia-api headers; first run on H.
//
// API references (Jetson Multimedia API r36.4.3 headers; doc comments quoted where they define behaviour):
// - CameraProvider::create / ICameraProvider::getCameraDevices / createCaptureSession (Argus/CameraProvider.h).
// - ICameraProperties::getAllSensorModes; ISensorMode::getResolution, getFrameDurationRange, getExposureTimeRange,
//   getAnalogGainRange, getInputBitDepth (Argus/CameraDevice.h:125-280).
// - ISourceSettings::setSensorMode / setFrameDurationRange ("If frame range is out of bounds of the current sensor
//   mode, the capture's frame duration will be as close as possible to the range specified", Settings.h:131) /
//   setExposureTimeRange / setGainRange; IAutoControlSettings::setAeLock / setAwbLock / setAwbMode /
//   setIspDigitalGainRange / setAeAntibandingMode; IDenoiseSettings::setDenoiseMode; IEdgeEnhanceSettings::
//   setEdgeEnhanceMode (Argus/Settings.h).
// - IEGLOutputStreamSettings::setPixelFormat(PIXEL_FMT_YCbCr_420_888) / setResolution / setMetadataEnable /
//   setMode(EGL_STREAM_MODE_FIFO): "every producer frame is made available to the consumer ... If the fifo is full
//   Argus will stall" (Argus/EGLStream.h:156-176) / setFifoLength.
// - EGLStream::FrameConsumer::create + IFrameConsumer::acquireFrame(timeout, &status): "If NULL is returned and the
//   status code is STATUS_DISCONNECTED, the producer has disconnected" (EGLStream/FrameConsumer.h:88-106);
//   IFrame::getNumber; IArgusCaptureMetadata::getMetadata -> ICaptureMetadata (sensor timestamp, exposure, analog
//   gain, ISP digital gain, frame readout time; Argus/CaptureMetadata.h:356-397).
// - Ext::ISensorTimestampTsc::getSensorSofTimestampTsc / getSensorEofTimestampTsc (Argus/Ext/SensorTimestampTsc.h).
// - EGLStream::NV::IImageNativeBuffer::createNvBuffer(size, NVBUF_COLOR_FORMAT_NV12_ER, NVBUF_LAYOUT_PITCH) once, then
//   copyToNvBuffer(fd) per frame (a VIC blit, i.e. NvBufSurfTransform inside libargus; ImageNativeBuffer.h:70-108);
//   NvBufSurfaceFromFd / NvBufSurfaceMap(READ) / NvBufSurfaceSyncForCpu / NvBufSurfaceDestroy (nvbufsurface.h).
// The same sequence as the Multimedia API samples argus/samples/userAutoExposure (per-frame ICaptureMetadata) and
// samples/09_argus_camera_jpeg (FrameConsumer -> IImageNativeBuffer -> NvBufSurface), and sensor_rig's (never built)
// native/argus_sof/argus_sof.cpp, whose t_mid convention was wrong (see sensor_clock.hpp).
//
// Pipeline per camera: an acquisition thread drains the EGLStream FIFO as fast as possible (so Argus never stalls
// unless the CPU copy cannot keep up, which then shows as SOF time gaps), stamps t_arrival = now_raw_ns() right
// after acquireFrame, converts the timestamp (SensorClockResolver), copies NV12 into a pooled CPU buffer, and
// queues the frame for read() (bounded; oldest dropped -> drops_backend_queue; the gap is visible in seq).
//
// Image time: t_sof_ns = argus_start_of_exposure(raw(getSensorTimestamp), exposure) = start of row-0 exposure;
// t_mid_ns = mid_exposure_time(t_sof, exposure, readout) = ts - exposure/2 + readout/2 (sensor_clock.hpp).
#include <Argus/Argus.h>
#include <Argus/Ext/SensorTimestampTsc.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/NV/ImageNativeBuffer.h>
#include <nvbufsurface.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "../backends.hpp"
#include "../camera_internal.hpp"
#include "rig/drivers/camera/capture_plan.hpp"
#include "rig/drivers/camera/imx519_focus.hpp"
#include "rig/drivers/camera/sensor_clock.hpp"
#include "rig/drivers/camera/sensor_modes.hpp"
#include "rig/time/clock.hpp"

namespace rig {
namespace cam {
namespace {

using Argus::interface_cast;
using Argus::UniqueObj;

const char* argus_status_name(Argus::Status s) {
  switch (s) {
    case Argus::STATUS_OK: return "OK";
    case Argus::STATUS_INVALID_PARAMS: return "INVALID_PARAMS";
    case Argus::STATUS_INVALID_SETTINGS: return "INVALID_SETTINGS";
    case Argus::STATUS_UNAVAILABLE: return "UNAVAILABLE";
    case Argus::STATUS_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
    case Argus::STATUS_UNIMPLEMENTED: return "UNIMPLEMENTED";
    case Argus::STATUS_TIMEOUT: return "TIMEOUT";
    case Argus::STATUS_CANCELLED: return "CANCELLED";
    case Argus::STATUS_DISCONNECTED: return "DISCONNECTED";
    case Argus::STATUS_END_OF_STREAM: return "END_OF_STREAM";
    default: return "UNKNOWN";
  }
}

Status argus_check(Argus::Status s, const std::string& what) {
  if (s == Argus::STATUS_OK) return Status::Ok();
  return device_error(what + ": Argus status " + argus_status_name(s));
}

// ---------------------------------------------------------------------------------------------------------------
// One CameraProvider per process, shared by all cameras (the samples, e.g. argus/samples/multiSensor, use a single
// provider for several devices; creating several in one process is not documented as supported).
// ---------------------------------------------------------------------------------------------------------------
std::mutex g_provider_mu;
std::weak_ptr<Argus::CameraProvider> g_provider;

Result<std::shared_ptr<Argus::CameraProvider>> shared_provider() {
  std::lock_guard<std::mutex> lk(g_provider_mu);
  if (auto p = g_provider.lock()) return p;
  Argus::Status st = Argus::STATUS_OK;
  Argus::CameraProvider* raw = Argus::CameraProvider::create(&st);
  if (raw == nullptr || interface_cast<Argus::ICameraProvider>(raw) == nullptr) {
    if (raw != nullptr) raw->destroy();
    return unavailable(std::string("Argus CameraProvider::create failed (") + argus_status_name(st) +
                       "): is nvargus-daemon running? (sudo systemctl restart nvargus-daemon)");
  }
  std::shared_ptr<Argus::CameraProvider> p(raw, [](Argus::CameraProvider* q) { q->destroy(); });
  g_provider = p;
  auto* ip = interface_cast<Argus::ICameraProvider>(raw);
  spdlog::info("argus: provider {} {}", ip->getVendor(), ip->getVersion());
  return p;
}

Result<Argus::CameraDevice*> device_of(Argus::CameraProvider* prov, int sensor_id) {
  auto* ip = interface_cast<Argus::ICameraProvider>(prov);
  std::vector<Argus::CameraDevice*> devs;
  RIG_RETURN_IF_ERROR(argus_check(ip->getCameraDevices(&devs), "getCameraDevices"));
  if (sensor_id < 0 || sensor_id >= static_cast<int>(devs.size()))
    return not_found("Argus sensor_id " + std::to_string(sensor_id) + " not found: " + std::to_string(devs.size()) +
                     " camera(s) enumerated (check the device tree overlay and `dmesg | grep -i -E 'imx|tegra-camrtc'`)");
  return devs[static_cast<std::size_t>(sensor_id)];
}

struct ModeList {
  std::vector<SensorMode> modes;
  std::vector<Argus::SensorMode*> handles;
};

Result<ModeList> enumerate(Argus::CameraDevice* dev, const std::string& sensor) {
  auto* props = interface_cast<Argus::ICameraProperties>(dev);
  if (props == nullptr) return device_error("ICameraProperties unavailable");
  ModeList out;
  RIG_RETURN_IF_ERROR(argus_check(props->getAllSensorModes(&out.handles), "getAllSensorModes"));
  for (std::size_t i = 0; i < out.handles.size(); ++i) {
    auto* sm = interface_cast<Argus::ISensorMode>(out.handles[i]);
    if (sm == nullptr) return device_error("ISensorMode unavailable for mode " + std::to_string(i));
    SensorMode m;
    m.index = static_cast<int>(i);
    const Argus::Size2D<uint32_t> res = sm->getResolution();
    m.width = static_cast<int>(res.width());
    m.height = static_cast<int>(res.height());
    const Argus::Range<uint64_t> fd = sm->getFrameDurationRange();
    m.max_fps = fd.min() > 0 ? 1e9 / static_cast<double>(fd.min()) : kNaN;
    m.min_fps = fd.max() > 0 ? 1e9 / static_cast<double>(fd.max()) : kNaN;
    const Argus::Range<uint64_t> ex = sm->getExposureTimeRange();
    m.min_exposure_ns = static_cast<int64_t>(ex.min());
    m.max_exposure_ns = static_cast<int64_t>(ex.max());
    const Argus::Range<float> g = sm->getAnalogGainRange();
    m.min_analog_gain = g.min();
    m.max_analog_gain = g.max();
    m.bit_depth = static_cast<int>(sm->getInputBitDepth());
    m.description = props->getModelName();
    out.modes.push_back(m);
  }
  annotate_sensor_modes(sensor, out.modes);
  return out;
}

const Argus::AwbMode& awb_of(int idx) {
  switch (idx) {
    case 0: return Argus::AWB_MODE_OFF;
    case 2: return Argus::AWB_MODE_INCANDESCENT;
    case 3: return Argus::AWB_MODE_FLUORESCENT;
    case 4: return Argus::AWB_MODE_WARM_FLUORESCENT;
    case 5: return Argus::AWB_MODE_DAYLIGHT;
    case 6: return Argus::AWB_MODE_CLOUDY_DAYLIGHT;
    case 7: return Argus::AWB_MODE_TWILIGHT;
    case 8: return Argus::AWB_MODE_SHADE;
    case 9: return Argus::AWB_MODE_MANUAL;
    default: return Argus::AWB_MODE_AUTO;
  }
}

class ArgusCamera final : public CameraBase {
 public:
  explicit ArgusCamera(const CameraConfig& c) : CameraBase(c) {}
  ~ArgusCamera() override { (void)ArgusCamera::stop(); }

  Result<std::vector<SensorMode>> modes() const override {
    if (running_.load()) return modes_cache_;
    return argus_list_modes(config_.sensor_id, config_.sensor);
  }

  Status start() override {
    if (running_.load()) return failed_precondition("camera " + config_.id + ": already started");
    Status s = start_impl();
    if (!s.ok()) {
      (void)stop();
      return s.with_context("camera " + config_.id + " (argus sensor_id " + std::to_string(config_.sensor_id) + ")");
    }
    return s;
  }

  Status stop() override {
    stopping_.store(true);
    if (session_) {
      auto* is = interface_cast<Argus::ICaptureSession>(session_);
      if (is != nullptr) {
        is->stopRepeat();
        is->waitForIdle(2'000'000'000ULL);
      }
    }
    if (thread_.joinable()) thread_.join();
    if (surf_ != nullptr) {
      NvBufSurfaceUnMap(surf_, 0, -1);
      NvBufSurfaceDestroy(surf_);  // owner of dmabuf_ (ImageNativeBuffer.h:72-74)
      surf_ = nullptr;
      dmabuf_ = -1;
    }
    consumer_.reset();
    stream_.reset();
    request_.reset();
    session_.reset();
    provider_.reset();
    running_.store(false);
    queue_.close();
    return Status::Ok();
  }

  std::optional<ImageFrame> read(std::chrono::nanoseconds timeout) override {
    std::optional<ImageFrame> f = queue_.pop(timeout);
    if (f) deliver(*f);
    return f;
  }

 private:
  Status start_impl() {
    const std::string ctx = "cameras." + config_.id + ".driver";
    if (config_.pixel_format != PixelFormat::kNV12 && config_.pixel_format != PixelFormat::kGRAY8 &&
        config_.pixel_format != PixelFormat::kBGR8 && config_.pixel_format != PixelFormat::kRGB8)
      return invalid_argument(ctx + ".pixel_format: argus delivers nv12 | gray8 | bgr8 | rgb8");
    if (packed_frame_bytes(PixelFormat::kNV12, config_.width, config_.height) == 0)
      return invalid_argument(ctx + ": NV12 needs even width and height");
    stopping_.store(false);
    RIG_ASSIGN_OR_RETURN(provider_, shared_provider());
    RIG_ASSIGN_OR_RETURN(Argus::CameraDevice * dev, device_of(provider_.get(), config_.sensor_id));
    RIG_ASSIGN_OR_RETURN(ModeList ml, enumerate(dev, config_.sensor));
    modes_cache_ = ml.modes;
    spdlog::info("camera {}: argus modes: {}", config_.id, format_sensor_modes(ml.modes));
    RIG_ASSIGN_OR_RETURN(SensorMode mode, select_sensor_mode(ml.modes, config_));
    RIG_ASSIGN_OR_RETURN(CapturePlan plan, build_capture_plan(config_, mode));
    plan_ = plan;
    set_resolved_mode(mode.index);
    Argus::SensorMode* mode_handle = ml.handles[static_cast<std::size_t>(mode.index)];

    auto* ip = interface_cast<Argus::ICameraProvider>(provider_.get());
    tsc_supported_ = ip->supportsExtension(Argus::EXT_SENSOR_TIMESTAMP_TSC);
    Result<int64_t> off = read_tsc_offset_ns();
    tsc_offset_ = off.ok() ? std::optional<int64_t>(*off) : std::nullopt;
    spdlog::info("camera {}: Ext::SensorTimestampTsc {}; clocksource0/offset_ns {}", config_.id,
                 tsc_supported_ ? "supported" : "NOT supported",
                 tsc_offset_ ? std::to_string(*tsc_offset_) : off.status().message());

    Argus::Status st = Argus::STATUS_OK;
    session_.reset(ip->createCaptureSession(dev, &st));
    auto* is = interface_cast<Argus::ICaptureSession>(session_);
    if (is == nullptr)
      return unavailable(std::string("createCaptureSession failed (") + argus_status_name(st) +
                         "): camera busy (another process?) or nvargus-daemon unhealthy");

    UniqueObj<Argus::OutputStreamSettings> ss(is->createOutputStreamSettings(Argus::STREAM_TYPE_EGL, &st));
    auto* ies = interface_cast<Argus::IEGLOutputStreamSettings>(ss);
    if (ies == nullptr) return device_error("createOutputStreamSettings(EGL) failed");
    RIG_RETURN_IF_ERROR(argus_check(ies->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888), "setPixelFormat"));
    RIG_RETURN_IF_ERROR(argus_check(
        ies->setResolution(Argus::Size2D<uint32_t>(static_cast<uint32_t>(config_.width),
                                                   static_cast<uint32_t>(config_.height))),
        "setResolution"));
    RIG_RETURN_IF_ERROR(argus_check(ies->setMetadataEnable(true), "setMetadataEnable"));
    RIG_RETURN_IF_ERROR(argus_check(ies->setMode(Argus::EGL_STREAM_MODE_FIFO), "setMode(FIFO)"));
    RIG_RETURN_IF_ERROR(argus_check(ies->setFifoLength(static_cast<uint32_t>(plan.fifo_length)), "setFifoLength"));
    stream_.reset(is->createOutputStream(ss.get(), &st));
    if (!stream_) return device_error(std::string("createOutputStream failed (") + argus_status_name(st) + ")");
    consumer_.reset(EGLStream::FrameConsumer::create(stream_.get(), &st));
    iconsumer_ = interface_cast<EGLStream::IFrameConsumer>(consumer_);
    if (iconsumer_ == nullptr) return device_error(std::string("FrameConsumer::create failed (") + argus_status_name(st) + ")");

    request_.reset(is->createRequest(Argus::CAPTURE_INTENT_PREVIEW, &st));
    auto* ir = interface_cast<Argus::IRequest>(request_);
    if (ir == nullptr) return device_error("createRequest failed");
    RIG_RETURN_IF_ERROR(argus_check(ir->enableOutputStream(stream_.get()), "enableOutputStream"));
    RIG_RETURN_IF_ERROR(apply_controls(mode_handle, /*lock_now=*/false));

    begin_stream(config_.fps, mode.index, "argus:pending");
    queue_.reopen(static_cast<std::size_t>(plan.fifo_length));
    resolver_ = SensorClockResolver(kMinSettleFrames);
    {
      std::lock_guard<std::mutex> lk(ev_mu_);
      acquired_ = 0;
      resolved_ = false;
      fatal_flag_ = false;
      eof_logged_ = false;
    }
    running_.store(true);
    thread_ = std::thread([this] { acquire_loop(); });
    RIG_RETURN_IF_ERROR(argus_check(is->repeat(request_.get()), "repeat (start streaming)"));

    // Settle: first frame -> focus (sensor and VCM powered now) -> timestamp domain -> AE/AWB -> locks.
    RIG_RETURN_IF_ERROR(wait_for([&] { return acquired_ > 0; }, 5'000'000'000LL, "first frame"));
    bool focus_applied = false;
    if (config_.focus.mode == FocusMode::kFixed) {
      Imx519Focus focus(config_.focus);
      set_focus(focus.apply());
      focus_applied = true;
    }
    RIG_RETURN_IF_ERROR(wait_for([&] { return resolved_; }, 5'000'000'000LL, "timestamp-domain classification"));
    if (fatal()) return status();
    if (plan.needs_settle() || focus_applied) {
      const TimeNs until = now_raw_ns() + kSettleNs;
      while (now_raw_ns() < until) {
        if (!queue_.pop(std::chrono::seconds(2)) || fatal())
          return unavailable("frames stopped during settle" + (fatal() ? ": " + status().message() : std::string()));
      }
    }
    if (plan.needs_settle()) {
      RIG_RETURN_IF_ERROR(apply_controls(mode_handle, /*lock_now=*/true));
      RIG_RETURN_IF_ERROR(argus_check(is->repeat(request_.get()), "repeat (locked request)"));
      for (int i = 0; i < kMinSettleFrames; ++i)  // frames already in flight carry the unlocked request
        if (!queue_.pop(std::chrono::seconds(2))) return unavailable("frames stopped after locking AE/AWB");
      spdlog::info("camera {}: AE lock {} / AWB lock {} applied after {} ms settle", config_.id,
                   plan.ae_lock_requested, plan.awb_lock_requested, kSettleNs / 1'000'000);
    }
    rebase();
    spdlog::info("camera {}: argus streaming {}x{} @ {} fps, mode {}, frame duration {} ns, exposure {}..{} ns, "
                 "gain {}..{}, isp gain {}, denoise {}, edge {}",
                 config_.id, config_.width, config_.height, config_.fps, mode.index, plan.frame_duration_ns,
                 plan.exposure_min_ns, plan.exposure_max_ns, plan.gain_min, plan.gain_max, plan.isp_gain_min,
                 plan.denoise, plan.edge_enhance);
    return Status::Ok();
  }

  /// Applies the plan to request_. lock_now = false: deferred locks are off (first request); true: all locks.
  Status apply_controls(Argus::SensorMode* mode, bool lock_now) {
    auto* ir = interface_cast<Argus::IRequest>(request_);
    auto* src = interface_cast<Argus::ISourceSettings>(ir->getSourceSettings());
    auto* ac = interface_cast<Argus::IAutoControlSettings>(ir->getAutoControlSettings());
    auto* dn = interface_cast<Argus::IDenoiseSettings>(request_);
    auto* ee = interface_cast<Argus::IEdgeEnhanceSettings>(request_);
    if (src == nullptr || ac == nullptr) return device_error("ISourceSettings / IAutoControlSettings unavailable");
    const CapturePlan& p = plan_;
    RIG_RETURN_IF_ERROR(argus_check(src->setSensorMode(mode), "setSensorMode"));
    RIG_RETURN_IF_ERROR(argus_check(
        src->setFrameDurationRange(Argus::Range<uint64_t>(static_cast<uint64_t>(p.frame_duration_ns))),
        "setFrameDurationRange"));
    RIG_RETURN_IF_ERROR(argus_check(src->setExposureTimeRange(Argus::Range<uint64_t>(
                                        static_cast<uint64_t>(p.exposure_min_ns), static_cast<uint64_t>(p.exposure_max_ns))),
                                    "setExposureTimeRange"));
    RIG_RETURN_IF_ERROR(argus_check(
        src->setGainRange(Argus::Range<float>(static_cast<float>(p.gain_min), static_cast<float>(p.gain_max))),
        "setGainRange"));
    RIG_RETURN_IF_ERROR(argus_check(ac->setAeAntibandingMode(Argus::AE_ANTIBANDING_MODE_OFF), "setAeAntibandingMode"));
    RIG_RETURN_IF_ERROR(argus_check(ac->setAwbMode(awb_of(p.wbmode)), "setAwbMode"));
    RIG_RETURN_IF_ERROR(argus_check(ac->setIspDigitalGainRange(Argus::Range<float>(
                                        static_cast<float>(p.isp_gain_min), static_cast<float>(p.isp_gain_max))),
                                    "setIspDigitalGainRange"));
    const bool ae = p.ae_lock_requested && (lock_now || !p.ae_lock_deferred);
    const bool awb = p.awb_lock_requested && (lock_now || !p.awb_lock_deferred);
    RIG_RETURN_IF_ERROR(argus_check(ac->setAeLock(ae), "setAeLock"));
    RIG_RETURN_IF_ERROR(argus_check(ac->setAwbLock(awb), "setAwbLock"));
    if (dn != nullptr) {
      RIG_RETURN_IF_ERROR(argus_check(
          dn->setDenoiseMode(p.denoise ? Argus::DENOISE_MODE_FAST : Argus::DENOISE_MODE_OFF), "setDenoiseMode"));
    } else if (!p.denoise) {
      spdlog::warn("camera {}: IDenoiseSettings unavailable; ISP denoise may be on", config_.id);
    }
    if (ee != nullptr) {
      RIG_RETURN_IF_ERROR(argus_check(
          ee->setEdgeEnhanceMode(p.edge_enhance ? Argus::EDGE_ENHANCE_MODE_FAST : Argus::EDGE_ENHANCE_MODE_OFF),
          "setEdgeEnhanceMode"));
    } else if (!p.edge_enhance) {
      spdlog::warn("camera {}: IEdgeEnhanceSettings unavailable; edge enhancement may be on", config_.id);
    }
    return Status::Ok();
  }

  /// Colour format of our CPU-side NvBuffer (R1 F16). nvbufsurface.h: NVBUF_COLOR_FORMAT_NV12_ER = "BT.601 colorspace
  /// - Y/CbCr ER 4:2:0 multi-planar" (extended = full range, JFIF), so NV12 / GRAY8 frames can be JPEG-encoded
  /// directly. For BGR/RGB output the blit targets NVBUF_COLOR_FORMAT_NV12 (BT.601 video range), because OpenCV's
  /// COLOR_YUV2BGR_NV12 expects video range; the result is full-range RGB either way.
  NvBufSurfaceColorFormat nvbuf_format() const {
    return (config_.pixel_format == PixelFormat::kBGR8 || config_.pixel_format == PixelFormat::kRGB8)
               ? NVBUF_COLOR_FORMAT_NV12
               : NVBUF_COLOR_FORMAT_NV12_ER;
  }

  template <class Pred>
  Status wait_for(Pred pred, int64_t timeout_ns, const char* what) {
    std::unique_lock<std::mutex> lk(ev_mu_);
    const bool ok = ev_cv_.wait_for(lk, std::chrono::nanoseconds(timeout_ns), [&] { return pred() || fatal_flag_; });
    if (fatal_flag_) return status().ok() ? device_error(std::string(what) + ": stream failed") : status();
    if (!ok) return unavailable(std::string("timeout waiting for ") + what);
    return Status::Ok();
  }

  void notify(bool fatal_event) {
    {
      std::lock_guard<std::mutex> lk(ev_mu_);
      if (fatal_event) fatal_flag_ = true;
    }
    ev_cv_.notify_all();
  }

  void acquire_loop() {
    TimeNs last_frame = now_raw_ns();
    while (!stopping_.load()) {
      Argus::Status st = Argus::STATUS_OK;
      UniqueObj<EGLStream::Frame> frame(iconsumer_->acquireFrame(100'000'000ULL, &st));
      if (!frame) {
        if (stopping_.load()) break;
        if (st == Argus::STATUS_TIMEOUT) {
          if (now_raw_ns() - last_frame > 2'000'000'000) {
            count_error("no frame for 2 s (sensor or nvargus-daemon stalled)");
            last_frame = now_raw_ns();
          }
          continue;
        }
        if (st == Argus::STATUS_DISCONNECTED || st == Argus::STATUS_END_OF_STREAM) {
          set_fatal(device_error(std::string("camera ") + config_.id + ": EGLStream producer disconnected (" +
                                 argus_status_name(st) + "): nvargus-daemon died or the sensor was lost"));
          notify(true);
          queue_.close();
          break;
        }
        count_error(std::string("acquireFrame: ") + argus_status_name(st));
        continue;
      }
      const TimeNs t_arrival = now_raw_ns();
      const ClockAnchor anchor = sample_clock_anchor(1);
      last_frame = t_arrival;
      process(frame, t_arrival, anchor);
    }
  }

  void process(UniqueObj<EGLStream::Frame>& frame, TimeNs t_arrival, const ClockAnchor& anchor) {
    auto* ifr = interface_cast<EGLStream::IFrame>(frame);
    auto* iam = interface_cast<EGLStream::IArgusCaptureMetadata>(frame);
    Argus::CaptureMetadata* cm = iam != nullptr ? iam->getMetadata() : nullptr;
    auto* md = interface_cast<Argus::ICaptureMetadata>(cm);
    if (ifr == nullptr || md == nullptr) {
      count_error("frame without IFrame / capture metadata (setMetadataEnable)");
      return;
    }
    const uint64_t number = ifr->getNumber();
    const uint64_t sensor_ts = md->getSensorTimestamp();
    const auto exposure = static_cast<int64_t>(md->getSensorExposureTime());
    const double again = md->getSensorAnalogGain();
    const double dgain = md->getIspDigitalGain();
    const auto readout = static_cast<int64_t>(md->getFrameReadoutTime());
    std::optional<uint64_t> tsc_sof, tsc_eof;
    if (tsc_supported_) {
      if (auto* its = interface_cast<Argus::Ext::ISensorTimestampTsc>(cm)) {
        tsc_sof = its->getSensorSofTimestampTsc();
        tsc_eof = its->getSensorEofTimestampTsc();
      }
    }
    // Copy the image out of the Argus buffer (VIC blit into our pitch-linear NV12 NvBuffer), then release the
    // Argus frame immediately so the FIFO slot returns to the producer.
    auto* inb = interface_cast<EGLStream::NV::IImageNativeBuffer>(ifr->getImage());
    if (inb == nullptr) {
      count_error("IImageNativeBuffer unavailable");
      return;
    }
    if (dmabuf_ < 0) {
      Argus::Status st = Argus::STATUS_OK;
      dmabuf_ = inb->createNvBuffer(Argus::Size2D<uint32_t>(static_cast<uint32_t>(config_.width),
                                                            static_cast<uint32_t>(config_.height)),
                                    nvbuf_format(), NVBUF_LAYOUT_PITCH, EGLStream::NV::ROTATION_0, &st);
      if (dmabuf_ < 0 || NvBufSurfaceFromFd(dmabuf_, reinterpret_cast<void**>(&surf_)) != 0 || surf_ == nullptr ||
          NvBufSurfaceMap(surf_, 0, -1, NVBUF_MAP_READ) != 0) {
        set_fatal(device_error(std::string("createNvBuffer/NvBufSurfaceMap failed (") + argus_status_name(st) + ")"));
        notify(true);
        return;
      }
    } else if (Argus::Status st = inb->copyToNvBuffer(dmabuf_); st != Argus::STATUS_OK) {
      count_error(std::string("copyToNvBuffer: ") + argus_status_name(st));
      return;
    }
    frame.reset();

    if (!resolver_.ready()) {
      resolver_.add(sensor_ts, tsc_sof, tsc_offset_, anchor);
      if (resolver_.ready()) {
        Result<SensorClockResolver::Decision> d = resolver_.decide();
        if (!d.ok()) {
          set_fatal(d.status());
          notify(true);
          return;
        }
        set_time_source("argus:" + d->description);
        spdlog::info("camera {}: timestamps: {}", config_.id, d->description);
        std::lock_guard<std::mutex> lk(ev_mu_);
        resolved_ = true;
      }
      {
        std::lock_guard<std::mutex> lk(ev_mu_);
        ++acquired_;
      }
      ev_cv_.notify_all();
      return;  // classification frames are never delivered
    }
    const TimeNs first_data = resolver_.to_raw_ns(sensor_ts, tsc_sof, tsc_offset_, anchor);
    if (first_data != kInvalidTime && first_data > t_arrival)  // accepted up to kMaxFutureCameraStampNs (R1 F3)
      count_timestamp_anomaly("sensor first-data stamp " + std::to_string(first_data - t_arrival) +
                              " ns after arrival");
    if (tsc_sof && tsc_offset_) {
      // Cross-check: VI hardware SOF (TSC - offset_ns) minus getSensorTimestamp in its own classified domain.
      // ~0 when both are the same SOF event in MONOTONIC_RAW; a constant offset reveals a domain/offset mismatch.
      const TimeNs tsc_raw = sensor_clock_to_raw_ns(*tsc_sof, SensorClockDomain::kTscOffset, anchor, tsc_offset_);
      const TimeNs ts_raw = sensor_clock_to_raw_ns(sensor_ts, classify_sensor_clock(sensor_ts, anchor), anchor);
      if (tsc_raw != kInvalidTime && ts_raw != kInvalidTime) set_tsc_cross_check(static_cast<double>(tsc_raw - ts_raw));
    }
    if (tsc_sof && tsc_eof && !eof_logged_) {  // EOF - SOF ~ readout + one line: an H cross-check of readout
      spdlog::info("camera {}: TSC EOF - SOF = {} ns, getFrameReadoutTime = {} ns", config_.id,
                   static_cast<long long>(*tsc_eof - *tsc_sof), static_cast<long long>(readout));
      eof_logged_ = true;
    }

    ImageFrame f;
    uint8_t* dst = allocate_frame(pool_, config_.pixel_format, config_.width, config_.height, f);
    if (dst == nullptr) {
      count_error("frame allocation failed");
      return;
    }
    NvBufSurfaceSyncForCpu(surf_, 0, -1);
    const NvBufSurfaceParams& sp = surf_->surfaceList[0];
    const auto W = static_cast<std::size_t>(config_.width), H = static_cast<std::size_t>(config_.height);
    uint8_t* nv12 = dst;
    if (config_.pixel_format != PixelFormat::kNV12) {
      nv12_tmp_.resize(W * H * 3 / 2);
      nv12 = nv12_tmp_.data();
    }
    for (int plane = 0; plane < 2; ++plane) {  // Y rows, then interleaved UV rows; strip the pitch padding
      const auto* src = static_cast<const uint8_t*>(sp.mappedAddr.addr[plane]);
      const std::size_t pitch = sp.planeParams.pitch[plane];
      const std::size_t rows = plane == 0 ? H : H / 2;
      uint8_t* out = nv12 + (plane == 0 ? 0 : W * H);
      for (std::size_t r = 0; r < rows; ++r) std::memcpy(out + r * W, src + r * pitch, W);
    }
    if (config_.pixel_format == PixelFormat::kGRAY8) {
      std::memcpy(dst, nv12, W * H);
    } else if (config_.pixel_format == PixelFormat::kBGR8 || config_.pixel_format == PixelFormat::kRGB8) {
      const cv::Mat in(config_.height * 3 / 2, config_.width, CV_8UC1, nv12);
      cv::Mat out(config_.height, config_.width, CV_8UC3, dst);
      cv::cvtColor(in, out, config_.pixel_format == PixelFormat::kBGR8 ? cv::COLOR_YUV2BGR_NV12 : cv::COLOR_YUV2RGB_NV12);
    }
    f.t_sof_ns = argus_start_of_exposure(first_data, exposure);
    f.t_mid_ns = mid_exposure_time(f.t_sof_ns, exposure, readout);
    f.t_arrival_ns = t_arrival;
    f.meta.exposure_ns = exposure;
    f.meta.analog_gain = again;
    f.meta.digital_gain = dgain;
    f.meta.readout_ns = readout;
    f.meta.sensor_mode = config_.sensor_mode;
    f.meta.color_range = ColorRange::kFull;  // NV12_ER / its Y plane, or RGB converted from limited NV12
    produce(std::move(f), number);
    {
      std::lock_guard<std::mutex> lk(ev_mu_);
      ++acquired_;
    }
    ev_cv_.notify_all();
  }

  BufferPool pool_;
  CapturePlan plan_;
  std::vector<SensorMode> modes_cache_;
  std::shared_ptr<Argus::CameraProvider> provider_;
  UniqueObj<Argus::CaptureSession> session_;
  UniqueObj<Argus::OutputStream> stream_;
  UniqueObj<EGLStream::FrameConsumer> consumer_;
  EGLStream::IFrameConsumer* iconsumer_ = nullptr;
  UniqueObj<Argus::Request> request_;
  int dmabuf_ = -1;
  NvBufSurface* surf_ = nullptr;
  std::vector<uint8_t> nv12_tmp_;
  bool tsc_supported_ = false;
  bool eof_logged_ = false;
  std::optional<int64_t> tsc_offset_;
  SensorClockResolver resolver_{kMinSettleFrames};
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stopping_{false};
  std::mutex ev_mu_;
  std::condition_variable ev_cv_;
  uint64_t acquired_ = 0;
  bool resolved_ = false;
  bool fatal_flag_ = false;
};

}  // namespace

Result<std::vector<SensorMode>> argus_list_modes(int sensor_id, const std::string& sensor) {
  RIG_ASSIGN_OR_RETURN(std::shared_ptr<Argus::CameraProvider> prov, shared_provider());
  RIG_ASSIGN_OR_RETURN(Argus::CameraDevice * dev, device_of(prov.get(), sensor_id));
  RIG_ASSIGN_OR_RETURN(ModeList ml, enumerate(dev, sensor));
  return ml.modes;
}

std::unique_ptr<CameraBackend> make_argus_camera(const CameraConfig& config) {
  return std::make_unique<ArgusCamera>(config);
}

}  // namespace cam
}  // namespace rig
