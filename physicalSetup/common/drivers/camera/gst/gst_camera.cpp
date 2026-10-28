// SPDX-License-Identifier: MIT
// GstCamera: GStreamer fallback backend (PLAN T2). nvarguscamerasrc ! nvvidconv ! appsink on the Jetson, or
// `driver.gst.source_override` (e.g. videotestsrc) on the laptop; the appsink code path is the same in both.
//
// Timestamps (sensor_clock.hpp gst_clock_time_to_raw_ns): the pipeline clock is forced to a GstSystemClock with
// clock-type MONOTONIC, so buffer PTS (running time, via the sample's segment) + element base time = CLOCK_MONOTONIC
// capture time; it is converted to MONOTONIC_RAW with a clock anchor sampled at arrival. For nvarguscamerasrc this
// stamp is made inside the source, roughly 30 ms after mid-exposure and load-dependent (sensor_rig
// argus_logic.py:11-16): the GStreamer backend is the bring-up fallback, never the recording backend. Exposure,
// gain and readout are not available per frame (FrameMeta stays "unknown"), so t_mid = t_sof = that PTS time.
// Colour range: FrameMeta::color_range is taken from the negotiated caps' colorimetry (color_range_of()).
// Threading: frames arrive on the appsink streaming thread (new_sample callback) and are queued for read(); a bus
// thread watches for ERROR / EOS. stop() sets the pipeline to NULL, which joins the streaming threads.
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#include "../backends.hpp"
#include "../camera_internal.hpp"
#include "rig/drivers/camera/capture_plan.hpp"
#include "rig/drivers/camera/imx519_focus.hpp"
#include "rig/drivers/camera/sensor_clock.hpp"
#include "rig/drivers/camera/sensor_modes.hpp"
#include "rig/time/clock.hpp"

// ThreadSanitizer annotations (R1 F11). GStreamer hands frames to our appsink callback on pooled GLib streaming
// threads; the ordering "start() set-up -> set_state(PLAYING) -> callback" and "callback -> set_state(NULL) returns"
// is real but goes through GLib's uninstrumented primitives (GLib's pthread_create and mutexes are ignored by the
// test's called_from_lib suppressions), so TSan cannot see it. These annotations declare exactly those two edges
// (they are not memory accesses), keeping every other access of our code checked. No-ops outside TSan builds.
#if defined(__SANITIZE_THREAD__)
#include <sanitizer/tsan_interface.h>
#define RIG_TSAN_RELEASE(addr) __tsan_release(addr)
#define RIG_TSAN_ACQUIRE(addr) __tsan_acquire(addr)
#else
#define RIG_TSAN_RELEASE(addr) ((void)(addr))
#define RIG_TSAN_ACQUIRE(addr) ((void)(addr))
#endif

namespace rig {
namespace cam {
namespace {

Status gst_init_once() {
  static std::once_flag once;
  static std::string error;
  std::call_once(once, [] {
    GError* err = nullptr;
    if (!gst_init_check(nullptr, nullptr, &err)) {
      error = err ? err->message : "gst_init_check failed";
      if (err) g_error_free(err);
    }
  });
  if (!error.empty()) return unavailable("gstreamer: " + error);
  return Status::Ok();
}

std::string take_bus_error(GstElement* pipeline) {
  std::string out;
  GstBus* bus = gst_element_get_bus(pipeline);
  if (bus == nullptr) return out;
  while (GstMessage* m = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR))) {
    GError* err = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_error(m, &err, &dbg);
    if (!out.empty()) out += "; ";
    out += std::string(GST_OBJECT_NAME(GST_MESSAGE_SRC(m))) + ": " + (err ? err->message : "?");
    if (dbg) out += std::string(" (") + dbg + ")";
    if (err) g_error_free(err);
    g_free(dbg);
    gst_message_unref(m);
  }
  gst_object_unref(bus);
  return out;
}

/// Row bytes of plane i of a tightly packed frame (camera_internal.cpp allocate_frame layout).
std::size_t packed_row_bytes(PixelFormat f, int width) {
  switch (f) {
    case PixelFormat::kNV12: return static_cast<std::size_t>(width);  // Y: W samples; UV: W/2 pairs x 2 bytes
    case PixelFormat::kGRAY8: return static_cast<std::size_t>(width);
    case PixelFormat::kBGR8:
    case PixelFormat::kRGB8: return static_cast<std::size_t>(width) * 3;
    case PixelFormat::kYUYV: return static_cast<std::size_t>(width) * 2;
    default: return 0;
  }
}

/// YCbCr range of a negotiated GStreamer frame (R1 F16): whatever the producer states in the caps colorimetry
/// (GstVideoInfo.colorimetry.range). videoconvert/videotestsrc NV12 is BT.601 video range (16..235) by default;
/// nvvidconv's NV12 range is taken from its caps and checked on day H (BRINGUP 10.3). RGB/BGR are always full.
ColorRange color_range_of(const GstVideoInfo& info, PixelFormat fmt) {
  if (fmt == PixelFormat::kBGR8 || fmt == PixelFormat::kRGB8) return ColorRange::kFull;
  switch (GST_VIDEO_INFO_COLORIMETRY(&info).range) {
    case GST_VIDEO_COLOR_RANGE_0_255: return ColorRange::kFull;
    case GST_VIDEO_COLOR_RANGE_16_235: return ColorRange::kLimited;
    default: return ColorRange::kUnknown;
  }
}

class GstCamera final : public CameraBase {
 public:
  explicit GstCamera(const CameraConfig& c) : CameraBase(c) {}
  ~GstCamera() override { (void)GstCamera::stop(); }

  Result<std::vector<SensorMode>> modes() const override {
    // nvarguscamerasrc cannot enumerate modes for us: the published table is used (indices = the Jetson driver's
    // sensor-mode numbers, printed as GST_ARGUS lines at pipeline start; checked on H, BRINGUP 10).
    if (is_known_sensor(config_.sensor)) return known_sensor_modes(config_.sensor);
    SensorMode m;
    m.index = -1;
    m.width = config_.width;
    m.height = config_.height;
    m.min_fps = 1.0;
    m.max_fps = config_.fps;
    m.min_exposure_ns = 13000;
    m.max_exposure_ns = 1'000'000'000;
    m.min_analog_gain = 1.0;
    m.max_analog_gain = 16.0;
    m.full_fov = true;
    m.description = config_.gst.source_override.empty() ? "unknown sensor: Argus picks the mode"
                                                        : "gst source_override";
    return std::vector<SensorMode>{m};
  }

  Status start() override {
    if (pipeline_ != nullptr) return failed_precondition("camera " + config_.id + ": already started");
    const std::string ctx = "cameras." + config_.id + ".driver";
    RIG_RETURN_IF_ERROR(gst_init_once());
    if (gst_format_name(config_.pixel_format).empty())
      return invalid_argument(ctx + ".pixel_format: " + std::string(to_string(config_.pixel_format)) +
                              " not supported by the gst backend");
    if (packed_frame_bytes(config_.pixel_format, config_.width, config_.height) == 0)
      return invalid_argument(ctx + ": " + std::to_string(config_.width) + "x" + std::to_string(config_.height) +
                              " invalid for " + to_string(config_.pixel_format));
    RIG_ASSIGN_OR_RETURN(std::vector<SensorMode> ms, modes());
    RIG_ASSIGN_OR_RETURN(SensorMode mode, select_sensor_mode(ms, config_));
    RIG_ASSIGN_OR_RETURN(CapturePlan plan, build_capture_plan(config_, mode));
    set_resolved_mode(mode.index);
    const std::string desc = build_gst_pipeline(config_, plan);
    spdlog::info("camera {}: gst pipeline: {}", config_.id, desc);

    GError* err = nullptr;
    pipeline_ = gst_parse_launch(desc.c_str(), &err);
    if (err != nullptr) {
      const std::string msg = err->message;
      g_error_free(err);
      release();
      return invalid_argument(ctx + ".gst: cannot parse pipeline '" + desc + "': " + msg);
    }
    if (pipeline_ == nullptr) return internal_error(ctx + ".gst: gst_parse_launch returned null");
    sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "rig_sink");
    src_ = gst_bin_get_by_name(GST_BIN(pipeline_), "rig_src");  // nullptr with source_override
    if (sink_ == nullptr) {
      release();
      return internal_error(ctx + ".gst: appsink not found in pipeline");
    }
    // Pipeline clock = MONOTONIC system clock (so base_time + running_time is CLOCK_MONOTONIC).
    GstClock* clk = GST_CLOCK(g_object_new(GST_TYPE_SYSTEM_CLOCK, "clock-type", GST_CLOCK_TYPE_MONOTONIC, nullptr));
    gst_pipeline_use_clock(GST_PIPELINE(pipeline_), clk);
    gst_object_unref(clk);

    GstAppSinkCallbacks cb{};
    cb.new_sample = &GstCamera::on_new_sample;
    cb.eos = &GstCamera::on_eos;
    gst_app_sink_set_callbacks(GST_APP_SINK(sink_), &cb, this, nullptr);

    begin_stream(config_.fps, mode.index, "gst:pts(monotonic->raw)");
    queue_.reopen(static_cast<std::size_t>(plan.fifo_length));
    stopping_.store(false);
    // Publish everything set up above to the streaming thread (release; acquired in handle_sample). The pipeline's
    // own ordering goes through GLib's uninstrumented primitives (and pooled streaming threads), so this explicit
    // edge is what makes the hand-over provable under TSan (R1 F11).
    RIG_TSAN_RELEASE(&tsan_sync_);
    armed_.store(true, std::memory_order_release);

    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE ||
        gst_element_get_state(pipeline_, nullptr, nullptr, 5 * GST_SECOND) == GST_STATE_CHANGE_FAILURE) {
      const std::string e = take_bus_error(pipeline_);
      release();
      return unavailable("camera " + config_.id + ": gst pipeline failed to start: " + (e.empty() ? "unknown" : e) +
                         (config_.gst.source_override.empty() ? " (is nvargus-daemon running? sudo systemctl status "
                                                                "nvargus-daemon)"
                                                              : ""));
    }
    bus_thread_ = std::thread([this] { bus_loop(); });

    // Settle: first frame, focus (sensor powered now), AE/AWB convergence, deferred locks, latency sanity check.
    Status s = settle(plan);
    if (!s.ok()) {
      (void)stop();
      return s;
    }
    rebase();
    return Status::Ok();
  }

  Status stop() override {
    stopping_.store(true);
    armed_.store(false, std::memory_order_release);  // later samples are discarded unprocessed
    if (pipeline_ != nullptr) gst_element_set_state(pipeline_, GST_STATE_NULL);  // joins streaming threads
    // Pair with the callbacks' release decrements, so their last accesses happen-before the teardown below.
    while (inflight_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    RIG_TSAN_ACQUIRE(&tsan_sync_);
    if (bus_thread_.joinable()) bus_thread_.join();
    release();
    queue_.close();
    return Status::Ok();
  }

  std::optional<ImageFrame> read(std::chrono::nanoseconds timeout) override {
    std::optional<ImageFrame> f = queue_.pop(timeout);
    if (f) deliver(*f);
    return f;
  }

 private:
  Status settle(const CapturePlan& plan) {
    std::optional<ImageFrame> first = queue_.pop(std::chrono::seconds(5));
    if (!first) {
      const std::string e = fatal() ? status().message() : "no frame within 5 s";
      return unavailable("camera " + config_.id + ": gst: " + e);
    }
    std::vector<int64_t> lat{first->t_arrival_ns - first->t_sof_ns};
    bool focus_applied = false;
    if (config_.focus.mode == FocusMode::kFixed) {
      if (config_.gst.source_override.empty()) {
        Imx519Focus focus(config_.focus);
        set_focus(focus.apply());
        focus_applied = true;
      } else {
        set_focus(FocusResult{FocusStatus::kUnknown, -1, false, "gst source_override: focus not applied"});
      }
    }
    // AE/AWB only exist in nvarguscamerasrc (src_); a source_override has nothing to converge.
    const bool timed = (src_ != nullptr && plan.needs_settle()) || focus_applied;
    const TimeNs until = now_raw_ns() + (timed ? kSettleNs : 0);
    int n = 1;
    while (n < kMinSettleFrames || now_raw_ns() < until) {
      std::optional<ImageFrame> f = queue_.pop(std::chrono::seconds(2));
      if (!f) return unavailable("camera " + config_.id + ": gst: frames stopped during settle" +
                                 (fatal() ? ": " + status().message() : std::string()));
      lat.push_back(f->t_arrival_ns - f->t_sof_ns);
      ++n;
    }
    if (src_ != nullptr) {
      if (plan.ae_lock_deferred) g_object_set(src_, "aelock", TRUE, nullptr);
      if (plan.awb_lock_deferred) g_object_set(src_, "awblock", TRUE, nullptr);
      if (plan.ae_lock_deferred || plan.awb_lock_deferred)
        spdlog::info("camera {}: deferred AE/AWB lock applied after {} settle frames", config_.id, n);
    }
    std::sort(lat.begin(), lat.end());
    const int64_t med = lat[lat.size() / 2];
    if (med < -5'000'000 || med > kMaxPlausibleCameraLatencyNs)
      return device_error("camera " + config_.id + ": gst PTS domain unexpected: median arrival - PTS = " +
                          std::to_string(med) + " ns (expected 0..1 s after MONOTONIC->RAW conversion)");
    spdlog::info("camera {}: gst started, {} settle frames, median latency {:.2f} ms", config_.id, n, med * 1e-6);
    return Status::Ok();
  }

  void release() {
    if (src_ != nullptr) gst_object_unref(src_);
    if (sink_ != nullptr) gst_object_unref(sink_);
    if (pipeline_ != nullptr) {
      gst_element_set_state(pipeline_, GST_STATE_NULL);
      gst_object_unref(pipeline_);
    }
    src_ = sink_ = pipeline_ = nullptr;
  }

  void bus_loop() {
    GstBus* bus = gst_element_get_bus(pipeline_);
    while (!stopping_.load()) {
      GstMessage* m = gst_bus_timed_pop_filtered(
          bus, 100 * GST_MSECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_WARNING));
      if (m == nullptr) continue;
      if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_WARNING) {
        GError* e = nullptr;
        gst_message_parse_warning(m, &e, nullptr);
        count_error(std::string("gst warning: ") + (e ? e->message : "?"));
        if (e) g_error_free(e);
      } else if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_ERROR) {
        GError* e = nullptr;
        gchar* dbg = nullptr;
        gst_message_parse_error(m, &e, &dbg);
        set_fatal(device_error("camera " + config_.id + ": gst error from " + GST_OBJECT_NAME(GST_MESSAGE_SRC(m)) +
                               ": " + (e ? e->message : "?") + (dbg ? std::string(" (") + dbg + ")" : "")));
        if (e) g_error_free(e);
        g_free(dbg);
        queue_.close();  // unblock read(): status() now carries the error
      } else {
        set_fatal(data_loss("camera " + config_.id + ": gst end of stream"));
        queue_.close();
      }
      gst_message_unref(m);
    }
    gst_object_unref(bus);
  }

  static void on_eos(GstAppSink*, gpointer) {}  // reported through the bus (EOS message)

  static GstFlowReturn on_new_sample(GstAppSink* sink, gpointer user) {
    static_cast<GstCamera*>(user)->handle_sample(sink);
    return GST_FLOW_OK;
  }

  void handle_sample(GstAppSink* sink) {
    const TimeNs t_arrival = now_raw_ns();
    RIG_TSAN_ACQUIRE(&tsan_sync_);
    inflight_.fetch_add(1, std::memory_order_acq_rel);
    struct InflightGuard {
      std::atomic<int>& n;
      char* sync;
      ~InflightGuard() {
        RIG_TSAN_RELEASE(sync);
        n.fetch_sub(1, std::memory_order_release);
      }
    } inflight_guard{inflight_, &tsan_sync_};
    if (!armed_.load(std::memory_order_acquire)) {  // starting or stopping: drop the sample unprocessed
      if (GstSample* s = gst_app_sink_pull_sample(sink)) gst_sample_unref(s);
      return;
    }
    const ClockAnchor anchor = sample_clock_anchor(1);
    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (sample == nullptr) return;
    struct SampleGuard {
      GstSample* s;
      ~SampleGuard() { gst_sample_unref(s); }
    } guard{sample};
    GstBuffer* buf = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    const GstSegment* seg = gst_sample_get_segment(sample);
    GstVideoInfo info;
    if (buf == nullptr || caps == nullptr || !gst_video_info_from_caps(&info, caps)) {
      count_error("gst sample without buffer/caps");
      return;
    }
    if (GST_VIDEO_INFO_WIDTH(&info) != config_.width || GST_VIDEO_INFO_HEIGHT(&info) != config_.height) {
      count_error("gst caps size " + std::to_string(GST_VIDEO_INFO_WIDTH(&info)) + "x" +
                  std::to_string(GST_VIDEO_INFO_HEIGHT(&info)) + " != configured");
      return;
    }
    const GstClockTime pts = GST_BUFFER_PTS(buf);
    if (!GST_CLOCK_TIME_IS_VALID(pts)) {
      count_error("gst buffer without PTS");
      return;
    }
    const GstClockTime running =
        seg != nullptr ? gst_segment_to_running_time(seg, GST_FORMAT_TIME, pts) : pts;
    // The bin distributes its base time to every child on PLAYING, so the appsink's equals the pipeline's; using the
    // element we were called for avoids touching pipeline_ from the streaming thread.
    const GstClockTime base = gst_element_get_base_time(GST_ELEMENT(sink));
    if (!GST_CLOCK_TIME_IS_VALID(running) || !GST_CLOCK_TIME_IS_VALID(base)) {
      count_error("gst running/base time invalid");
      return;
    }
    ImageFrame f;
    uint8_t* dst = allocate_frame(pool_, config_.pixel_format, config_.width, config_.height, f);
    GstVideoFrame vf;
    if (dst == nullptr || !gst_video_frame_map(&vf, &info, buf, GST_MAP_READ)) {
      count_error("gst frame map failed");
      return;
    }
    const int nplanes = num_planes(config_.pixel_format);
    for (int p = 0; p < nplanes && p < static_cast<int>(GST_VIDEO_FRAME_N_PLANES(&vf)); ++p) {
      const auto* src = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&vf, p));
      const auto sstride = static_cast<std::size_t>(GST_VIDEO_FRAME_PLANE_STRIDE(&vf, p));
      const Plane& pl = f.planes[static_cast<std::size_t>(p)];
      const std::size_t row = std::min(packed_row_bytes(config_.pixel_format, config_.width), sstride);
      for (int r = 0; r < pl.height; ++r)
        std::memcpy(dst + pl.offset + static_cast<std::size_t>(r) * pl.stride, src + static_cast<std::size_t>(r) * sstride,
                    row);
    }
    gst_video_frame_unmap(&vf);
    f.t_sof_ns = gst_clock_time_to_raw_ns(base, running, anchor);
    f.t_mid_ns = f.t_sof_ns;  // exposure/readout unknown on this path (mid_exposure_time() would return t_sof)
    f.t_arrival_ns = t_arrival;
    f.meta.sensor_mode = config_.sensor_mode;
    f.meta.color_range = color_range_of(info, config_.pixel_format);
    std::optional<uint64_t> dseq;
    if (GST_BUFFER_OFFSET(buf) != GST_BUFFER_OFFSET_NONE) dseq = GST_BUFFER_OFFSET(buf);
    produce(std::move(f), dseq);
  }

  BufferPool pool_;
  GstElement* pipeline_ = nullptr;
  GstElement* sink_ = nullptr;
  GstElement* src_ = nullptr;
  std::thread bus_thread_;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> armed_{false};  ///< start() -> streaming thread hand-over (release/acquire).
  std::atomic<int> inflight_{0};    ///< handle_sample() calls in progress.
  char tsan_sync_ = 0;              ///< Address for the TSan happens-before annotations (never read or written).
};

}  // namespace

std::unique_ptr<CameraBackend> make_gst_camera(const CameraConfig& config) {
  return std::make_unique<GstCamera>(config);
}

}  // namespace cam
}  // namespace rig
