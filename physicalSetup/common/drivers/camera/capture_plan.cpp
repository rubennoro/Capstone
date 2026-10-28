// SPDX-License-Identifier: MIT
// Config + mode -> sensor/ISP control plan and GStreamer launch strings (rig/drivers/camera/capture_plan.hpp).
#include "rig/drivers/camera/capture_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

#include "rig/drivers/camera/sensor_modes.hpp"

namespace rig {

int awb_mode_index(const std::string& name) {
  // Order = nvarguscamerasrc wbmode enum = Argus AWB_MODE_* order in Argus/Types.h:209-218.
  static const char* const kNames[] = {"off",      "auto",          "incandescent", "fluorescent", "warm_fluorescent",
                                       "daylight", "cloudy_daylight", "twilight",   "shade",       "manual"};
  for (int i = 0; i < static_cast<int>(sizeof(kNames) / sizeof(kNames[0])); ++i)
    if (name == kNames[i]) return i;
  return -1;
}

namespace {
std::string fmt_num(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return buf;
}
}  // namespace

Result<CapturePlan> build_capture_plan(const CameraConfig& c, const SensorMode& mode) {
  const std::string ctx = "cameras." + c.id + ".driver";
  CapturePlan p;
  p.mode = mode;
  if (!mode_supports_fps(mode, c.fps))
    return invalid_argument(ctx + ".fps: " + fmt_num(c.fps) + " outside mode " + format_sensor_mode(mode));
  // Frame duration [d, d], clamped into the mode's range (Argus would clamp too: Settings.h:131-133).
  int64_t d = frame_duration_ns(c.fps);
  const int64_t d_min = frame_duration_ns(mode.max_fps);
  const int64_t d_max = std::isnan(mode.min_fps) || mode.min_fps <= 0 ? d : frame_duration_ns(mode.min_fps);
  d = std::clamp(d, std::min(d_min, d_max), std::max(d_min, d_max));
  p.frame_duration_ns = d;

  if (c.exposure_ns < 0) return invalid_argument(ctx + ".exposure_ns: must be >= 0 (0 = AE free)");
  if (c.exposure_ns > 0) {
    if (c.exposure_ns > d)
      return invalid_argument(ctx + ".exposure_ns: " + std::to_string(c.exposure_ns) +
                              " ns exceeds the frame duration " + std::to_string(d) + " ns at " + fmt_num(c.fps) + " fps");
    if ((mode.min_exposure_ns >= 0 && c.exposure_ns < mode.min_exposure_ns) ||
        (mode.max_exposure_ns >= 0 && c.exposure_ns > mode.max_exposure_ns))
      return invalid_argument(ctx + ".exposure_ns: " + std::to_string(c.exposure_ns) + " outside the mode range " +
                              std::to_string(mode.min_exposure_ns) + ".." + std::to_string(mode.max_exposure_ns));
    p.exposure_min_ns = p.exposure_max_ns = c.exposure_ns;
    p.exposure_locked = true;
  } else {
    p.exposure_min_ns = mode.min_exposure_ns >= 0 ? mode.min_exposure_ns : 0;
    p.exposure_max_ns = mode.max_exposure_ns >= 0 ? std::min(mode.max_exposure_ns, d) : d;
  }

  if (c.analog_gain < 0.0 || std::isnan(c.analog_gain))
    return invalid_argument(ctx + ".analog_gain: must be >= 0 (0 = AE free)");
  if (c.analog_gain > 0.0) {
    const double eps = 1e-6;
    if ((!std::isnan(mode.min_analog_gain) && c.analog_gain < mode.min_analog_gain - eps) ||
        (!std::isnan(mode.max_analog_gain) && c.analog_gain > mode.max_analog_gain + eps))
      return invalid_argument(ctx + ".analog_gain: " + fmt_num(c.analog_gain) + " outside the mode range " +
                              fmt_num(mode.min_analog_gain) + ".." + fmt_num(mode.max_analog_gain));
    p.gain_min = p.gain_max = c.analog_gain;
    p.gain_locked = true;
  } else {
    p.gain_min = std::isnan(mode.min_analog_gain) ? 1.0 : mode.min_analog_gain;
    p.gain_max = std::isnan(mode.max_analog_gain) ? 16.0 : mode.max_analog_gain;
  }

  if (!(c.isp_digital_gain >= 1.0))
    return invalid_argument(ctx + ".isp_digital_gain: " + fmt_num(c.isp_digital_gain) + " must be >= 1");
  p.isp_gain_min = p.isp_gain_max = c.isp_digital_gain;

  p.wbmode = awb_mode_index(c.awb_mode);
  if (p.wbmode < 0)
    return invalid_argument(ctx + ".awb_mode: unknown '" + c.awb_mode +
                            "' (off | auto | incandescent | fluorescent | warm_fluorescent | daylight | "
                            "cloudy_daylight | twilight | shade | manual)");
  p.awb_mode = c.awb_mode;
  p.ae_lock_requested = c.ae_lock;
  p.awb_lock_requested = c.awb_lock;
  p.ae_lock_deferred = c.ae_lock && !(p.exposure_locked && p.gain_locked);
  p.awb_lock_deferred = c.awb_lock && c.awb_mode != "off";
  p.denoise = c.denoise;
  p.edge_enhance = c.edge_enhance;
  if (c.buffer_count < 1) return invalid_argument(ctx + ".buffer_count: must be >= 1");
  p.fifo_length = c.buffer_count;
  return p;
}

std::string gst_format_name(PixelFormat f) {
  switch (f) {
    case PixelFormat::kNV12: return "NV12";
    case PixelFormat::kGRAY8: return "GRAY8";
    case PixelFormat::kBGR8: return "BGR";
    case PixelFormat::kRGB8: return "RGB";
    case PixelFormat::kYUYV: return "YUY2";
    case PixelFormat::kJPEG:
    case PixelFormat::kUnknown: return "";
  }
  return "";
}

std::string gst_framerate(double fps) {
  // Common NTSC-style rates first, then integer rates, then a 1/1000 approximation.
  const double r = std::round(fps);
  if (std::fabs(fps - r) < 1e-9) return std::to_string(static_cast<long long>(r)) + "/1";
  const double ntsc = fps * 1001.0 / 1000.0;
  if (std::fabs(ntsc - std::round(ntsc)) < 1e-6)
    return std::to_string(static_cast<long long>(std::round(ntsc) * 1000)) + "/1001";
  return std::to_string(static_cast<long long>(std::llround(fps * 1000.0))) + "/1000";
}

std::string build_gst_pipeline(const CameraConfig& c, const CapturePlan& p, const std::string& sink_name) {
  const std::string fmt = gst_format_name(c.pixel_format);
  const std::string rate = gst_framerate(c.fps);
  std::ostringstream os;
  if (c.gst.source_override.empty()) {
    // nvarguscamerasrc properties (gst-inspect-1.0 nvarguscamerasrc, L4T r36): sensor-id, sensor-mode,
    // exposuretimerange / gainrange / ispdigitalgainrange ("low high" strings), aelock, awblock, wbmode,
    // tnr-mode / ee-mode (0 off, 1 fast, 2 high quality), aeantibanding (0 off).
    os << "nvarguscamerasrc name=rig_src sensor-id=" << c.sensor_id;
    if (p.mode.index >= 0) os << " sensor-mode=" << p.mode.index;
    os << " exposuretimerange=\"" << p.exposure_min_ns << " " << p.exposure_max_ns << "\""
       << " gainrange=\"" << fmt_num(p.gain_min) << " " << fmt_num(p.gain_max) << "\""
       << " ispdigitalgainrange=\"" << fmt_num(p.isp_gain_min) << " " << fmt_num(p.isp_gain_max) << "\""
       << " aelock=" << ((p.ae_lock_requested && !p.ae_lock_deferred) ? "true" : "false")
       << " awblock=" << ((p.awb_lock_requested && !p.awb_lock_deferred) ? "true" : "false")
       << " wbmode=" << p.wbmode << " tnr-mode=" << (p.denoise ? 1 : 0) << " ee-mode=" << (p.edge_enhance ? 1 : 0)
       << " aeantibanding=0";
    os << " ! video/x-raw(memory:NVMM),width=" << c.width << ",height=" << c.height << ",framerate=" << rate
       << ",format=NV12 ! nvvidconv ! video/x-raw,format=NV12";
    if (fmt != "NV12") os << " ! videoconvert ! video/x-raw,format=" << fmt;
  } else {
    os << c.gst.source_override << " ! videoconvert ! videoscale ! video/x-raw,format=" << fmt << ",width=" << c.width
       << ",height=" << c.height << ",framerate=" << rate;
  }
  os << " ! appsink name=" << sink_name << " sync=false max-buffers=" << p.fifo_length
     << " drop=false emit-signals=false";
  return os.str();
}

}  // namespace rig
