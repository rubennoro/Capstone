// SPDX-License-Identifier: MIT
// rig.yaml loader + validator (rig/config/rig_config.hpp). Every key typed and range-checked, unknown keys rejected,
// cross-field checks (model size == capture size, sensor-mode legality, roles, live-check thresholds vs. fps ...).
// Never throws: yaml-cpp parse errors become kInvalidArgument with the source line.
#include <cmath>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

#include <Eigen/Dense>

#include "enum_tables.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/drivers/camera/sensor_modes.hpp"
#include "yaml_reader.hpp"

namespace rig {
namespace {

using cfg::closed;
using cfg::DRange;
using cfg::IRange;
using cfg::Issues;
using cfg::MapReader;
using cfg::open_lo;

constexpr int64_t kI64Max = std::numeric_limits<int64_t>::max();
constexpr int64_t kI64Min = std::numeric_limits<int64_t>::min();

bool in_list(const std::string& v, const std::vector<const char*>& list) {
  for (const char* s : list)
    if (v == s) return true;
  return false;
}

std::string join(const std::vector<const char*>& list) {
  std::string s;
  for (const char* x : list) {
    if (!s.empty()) s += " | ";
    s += x;
  }
  return s;
}

void str_enum(MapReader& m, const std::string& key, std::string& out, const std::vector<const char*>& allowed,
              bool required) {
  std::string v = out;
  if (!m.str(key, v, required)) return;
  if (!in_list(v, allowed)) {
    m.error(key, "'" + v + "' is not one of " + join(allowed));
    return;
  }
  out = v;
}

/// T2 full_fov: no crop relative to the sensor's reference field (main: the 16:9 field).
bool covers_full_field(const SensorMode& m, const std::vector<SensorMode>&) { return m.full_fov; }

/// The whole pixel array: full_fov and the aspect ratio of the largest (full-resolution) mode.
bool covers_full_array(const SensorMode& m, const std::vector<SensorMode>& all) {
  const SensorMode* big = nullptr;
  for (const auto& x : all)
    if (!big || int64_t{x.width} * x.height > int64_t{big->width} * big->height) big = &x;
  return m.full_fov && big && int64_t{m.width} * big->height == int64_t{m.height} * big->width;
}

/// "3280x2464@21, 1640x1232@30" for the modes passing `pred` (all if null), in table order.
std::string describe_modes(const std::vector<SensorMode>& modes,
                           bool (*pred)(const SensorMode&, const std::vector<SensorMode>&)) {
  std::string s;
  for (const auto& md : modes) {
    if (pred && !pred(md, modes)) continue;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%dx%d@%g", md.width, md.height, md.max_fps);
    s += (s.empty() ? "" : ", ") + std::string(buf);
  }
  return s;
}

// ---------------------------------------------------------------------------------------------------------------
// Leaf structs
// ---------------------------------------------------------------------------------------------------------------

void read_thread(MapReader m, ThreadConfig& t) {
  if (!m.ok_map()) return;
  // CPU ids: 0..1023 (CPU_SETSIZE); whether they exist on this machine is checked when applied (logged, not fatal).
  if (m.i32_list("cpus", t.cpus, IRange{0, 1023})) {
    std::set<int> u(t.cpus.begin(), t.cpus.end());
    if (u.size() != t.cpus.size()) m.error("cpus", "duplicate CPU id");
  }
  m.i32("priority", t.priority, IRange{0, 99});  // sched(7): SCHED_FIFO 1..99
  m.i32("nice", t.nice, IRange{-20, 19});        // setpriority(2)
  if (t.priority > 0 && t.nice != 0) m.error("nice", "nice applies to SCHED_OTHER only; set it to 0 when priority > 0");
  m.finish();
}

void check_rotation(MapReader& m, const std::string& key, const Eigen::Matrix3d& R) {
  // Orthonormal within 1e-6 (9+ significant digits in the file) and a proper rotation (det +1, not a reflection).
  const double ortho = (R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
  const double det = R.determinant();
  if (!(ortho <= 1e-6)) {
    m.error(key, "not orthonormal (max |R^T R - I| = " + cfg::fmt_double(ortho) + " > 1e-6)");
  } else if (!(std::abs(det - 1.0) <= 1e-6)) {
    m.error(key, "determinant is " + cfg::fmt_double(det) + ", must be +1 (a reflection is not a rotation)");
  }
}

void read_extrinsics(MapReader m, Extrinsics& e) {
  if (!m.ok_map()) return;
  if (m.mat3("rotation", e.rotation)) check_rotation(m, "rotation", e.rotation);
  if (m.vec3("translation_m", e.translation_m)) {
    if (e.translation_m.cwiseAbs().maxCoeff() > 10.0) m.error("translation_m", "|t| component > 10 m: wrong unit?");
  }
  m.enumeration("provenance", e.provenance, cfg::provenance_table());
  m.str("note", e.note);
  m.finish();
}

void read_model(MapReader m, CameraModelParams& p) {
  if (!m.ok_map()) return;
  m.enumeration("kind", p.kind, cfg::camera_model_table(), true);
  m.i32("width", p.width, IRange{1, 16384}, true);
  m.i32("height", p.height, IRange{1, 16384}, true);
  m.f64("fx", p.fx, open_lo(0, 1e5), true);
  m.f64("fy", p.fy, open_lo(0, 1e5), true);
  m.f64("cx", p.cx, closed(-1e5, 1e5), true);
  m.f64("cy", p.cy, closed(-1e5, 1e5), true);
  if (std::isfinite(p.cx) && p.width > 0 && !(p.cx >= 0 && p.cx <= p.width))
    m.error("cx", cfg::fmt_double(p.cx) + " is outside the image [0, " + std::to_string(p.width) + "]");
  if (std::isfinite(p.cy) && p.height > 0 && !(p.cy >= 0 && p.cy <= p.height))
    m.error("cy", cfg::fmt_double(p.cy) + " is outside the image [0, " + std::to_string(p.height) + "]");
  if (m.f64_list("distortion", p.distortion, true)) {
    const std::size_t need = static_cast<std::size_t>(num_distortion_params(p.kind));
    if (p.distortion.size() != need) {
      m.error("distortion", std::string(to_string(p.kind)) + " needs " + std::to_string(need) +
                                " parameters, got " + std::to_string(p.distortion.size()));
    } else if (p.kind == CameraModelKind::kDoubleSphere && !(p.distortion[1] >= 0 && p.distortion[1] <= 1)) {
      m.error("distortion", "double-sphere alpha (element 1) must be in [0, 1]");  // Usenko et al. 2018, eq. 43
    } else if (p.kind == CameraModelKind::kEucm && !(p.distortion[0] >= 0 && p.distortion[0] <= 1 && p.distortion[1] > 0)) {
      m.error("distortion", "eucm needs alpha in [0, 1] and beta > 0");  // Khomutenko et al. 2016
    }
  }
  m.f64("fov_max_deg", p.fov_max_deg, DRange{0, 360, false, true});
  m.enumeration("provenance", p.provenance, cfg::provenance_table());
  m.str("note", p.note);
  m.finish();
}

void read_focus(MapReader m, FocusConfig& f) {
  if (!m.ok_map()) return;
  m.enumeration("mode", f.mode, cfg::focus_mode_table());
  // Lens position: -1 = n/a; VCM DAC codes up to 12 bits (T2 cites the exact range from Arducam Focuser.py).
  m.i32("position", f.position, IRange{-1, 4095});
  m.i32("i2c_bus", f.i2c_bus, IRange{-1, 255});
  m.i32("i2c_addr", f.i2c_addr, IRange{0x03, 0x77});  // 7-bit non-reserved addresses (I2C-bus spec UM10204 3.1.12)
  m.boolean("allow_v4l2_fallback", f.allow_v4l2_fallback);
  m.str("v4l2_subdev", f.v4l2_subdev);
  if (f.mode == FocusMode::kFixed) {
    if (f.position < 0) m.error("position", "focus mode fixed needs a lens position >= 0");
    if (f.i2c_bus < 0 && !f.allow_v4l2_fallback)
      m.error("i2c_bus", "focus mode fixed needs an i2c_bus (or allow_v4l2_fallback: true)");
  }
  m.finish();
}

void read_driver(MapReader m, CameraConfig& c) {
  if (!m.ok_map()) return;
  m.enumeration("backend", c.backend, cfg::camera_backend_table(), true);
  str_enum(m, "sensor", c.sensor, cfg::camera_sensors(), true);
  m.i32("sensor_id", c.sensor_id, IRange{0, 15});
  m.i32("width", c.width, IRange{16, 16384}, true);
  m.i32("height", c.height, IRange{16, 16384}, true);
  m.f64("fps", c.fps, open_lo(0, 240), true);
  m.i32("sensor_mode", c.sensor_mode, IRange{-1, 63});
  m.enumeration("pixel_format", c.pixel_format, cfg::pixel_format_table());
  if (c.pixel_format == PixelFormat::kNV12 && (c.width % 2 != 0 || c.height % 2 != 0))
    m.error("pixel_format", "nv12 needs an even width and height");
  if (m.i64("exposure_ns", c.exposure_ns, IRange{0, 1000000000}) && c.exposure_ns > 0 && c.fps > 0) {
    const double frame_ns = 1e9 / c.fps;
    if (static_cast<double>(c.exposure_ns) > frame_ns)
      m.error("exposure_ns", std::to_string(c.exposure_ns) + " ns is longer than the frame duration " +
                                 cfg::fmt_double(frame_ns) + " ns at " + cfg::fmt_double(c.fps) + " fps");
  }
  m.f64("analog_gain", c.analog_gain, closed(0, 64));
  if (c.analog_gain > 0 && c.analog_gain < 1) m.error("analog_gain", "must be 0 (free) or >= 1");
  m.f64("isp_digital_gain", c.isp_digital_gain, closed(1, 256));
  m.boolean("ae_lock", c.ae_lock);
  m.boolean("awb_lock", c.awb_lock);
  str_enum(m, "awb_mode", c.awb_mode, cfg::awb_modes(), false);
  m.boolean("denoise", c.denoise);
  m.boolean("edge_enhance", c.edge_enhance);
  m.i32("buffer_count", c.buffer_count, IRange{2, 32});
  m.f64("drop_gap_factor", c.drop_gap_factor, DRange{1.0, 10.0, true, false});
  read_focus(m.sub("focus"), c.focus);
  {
    MapReader g = m.sub("gst");
    if (g.ok_map()) {
      g.str("source_override", c.gst.source_override);
      g.finish();
    }
  }
  {
    MapReader k = m.sub("mock");
    if (k.ok_map()) {
      str_enum(k, "pattern", c.mock.pattern, cfg::mock_patterns(), false);
      if (k.i64("jitter_ns", c.mock.jitter_ns, IRange{0, 1000000000}) && c.fps > 0 &&
          static_cast<double>(c.mock.jitter_ns) >= 0.5e9 / c.fps)
        k.error("jitter_ns", "must be < half the frame period (frames would reorder)");
      if (k.i32("drop_every_n", c.mock.drop_every_n, IRange{0, 1000000000}) && c.mock.drop_every_n == 1)
        k.error("drop_every_n", "1 would drop every frame; use 0 (never) or >= 2");
      k.i32("stall_after_frames", c.mock.stall_after_frames, IRange{-1, 1000000000});
      k.i64("stall_ns", c.mock.stall_ns, IRange{0, kI64Max});
      k.u64("seed", c.mock.seed);
      k.finish();
    }
  }
  {
    MapReader r = m.sub("replay");
    if (r.ok_map()) {
      r.str("path", c.replay.path);
      r.boolean("realtime", c.replay.realtime);
      r.boolean("loop", c.replay.loop);
      r.boolean("retime", c.replay.retime);
      r.finish();
    }
  }
  if (c.backend == CameraBackendKind::kReplay && c.replay.path.empty())
    m.error("replay", "backend replay needs replay.path");

  // Sensor-mode legality for the real backends (mock/replay synthesize any size).
  if ((c.backend == CameraBackendKind::kArgus || c.backend == CameraBackendKind::kGst) && c.width > 0 &&
      c.height > 0 && c.fps > 0 && !c.sensor.empty()) {
    // T2's table is the single source (rig/drivers/camera/sensor_modes.hpp; R1 F15).
    const std::vector<SensorMode>& modes = known_sensor_modes(c.sensor);
    const SensorMode* hit = nullptr;
    for (const auto& md : modes)
      if (md.width == c.width && md.height == c.height) hit = &md;
    char req[64];
    std::snprintf(req, sizeof(req), "%dx%d@%g", c.width, c.height, c.fps);
    if (!hit) {
      m.error_here(std::string(req) + " is not a mode of " + c.sensor + "; modes are " + describe_modes(modes, nullptr));
    } else if (!mode_supports_fps(*hit, c.fps)) {
      m.error_here(std::string(req) + ": " + c.sensor + " " + std::to_string(c.width) + "x" +
                   std::to_string(c.height) + " runs at " + cfg::fmt_double(hit->min_fps) + ".." +
                   cfg::fmt_double(hit->max_fps) + " fps");
    }
  }
  m.finish();
}

void read_storage(MapReader m, CameraStorageConfig& s) {
  if (!m.ok_map()) return;
  m.i32("width", s.width, IRange{0, 16384});
  m.i32("height", s.height, IRange{0, 16384});
  m.i32("jpeg_quality", s.jpeg_quality, IRange{1, 100});
  m.i32("encoder_threads", s.encoder_threads, IRange{1, 16});
  m.size("queue_frames", s.queue_frames, IRange{1, 100000});
  m.finish();
}

const std::regex& id_regex() {
  static const std::regex r("[a-z0-9_]+");
  return r;
}

void read_camera(MapReader m, RigCameraConfig& c) {
  if (!m.ok_map()) return;
  str_enum(m, "role", c.role, cfg::camera_roles(), true);
  read_driver(m.sub("driver", true), c.driver);
  read_model(m.sub("model", true), c.model);
  read_extrinsics(m.sub("T_imu_cam"), c.T_imu_cam);
  {
    MapReader t = m.sub("time");
    if (t.ok_map()) {
      t.i64("offset_ns", c.time.offset_ns, IRange{-1000000000, 1000000000});  // |offset| <= 1 s
      t.enumeration("provenance", c.time.provenance, cfg::provenance_table());
      t.finish();
    }
  }
  read_storage(m.sub("storage"), c.storage);
  read_thread(m.sub("thread"), c.thread);

  // Cross-field checks inside one camera.
  const CameraConfig& d = c.driver;
  if (d.width > 0 && c.model.width > 0 && (c.model.width != d.width || c.model.height != d.height)) {
    m.error("model", "size " + std::to_string(c.model.width) + "x" + std::to_string(c.model.height) +
                         " != driver capture size " + std::to_string(d.width) + "x" + std::to_string(d.height) +
                         " (intrinsics refer to the captured image)");
  }
  if (c.model.kind == CameraModelKind::kVirtualPinhole)
    m.error("model", "kind 'pinhole' is for virtual cameras; physical cameras need a distortion model");
  const CameraStorageConfig& s = c.storage;
  if ((s.width == 0) != (s.height == 0)) {
    m.error("storage", "width and height must both be 0 (native) or both > 0");
  } else if (s.width > 0 && d.width > 0 && d.height > 0) {
    if (s.width > d.width || s.height > d.height)
      m.error("storage", "stored size " + std::to_string(s.width) + "x" + std::to_string(s.height) +
                             " is larger than the capture " + std::to_string(d.width) + "x" + std::to_string(d.height));
    if (int64_t{s.width} * d.height != int64_t{s.height} * d.width)
      m.error("storage", "stored size " + std::to_string(s.width) + "x" + std::to_string(s.height) +
                             " changes the capture aspect ratio " + std::to_string(d.width) + "x" +
                             std::to_string(d.height));
  }
  // Field-of-view rule (R1 F15), real backends only:
  // - ultrawide: the mode must cover the full sensor array (SLAM + tag camera; PIPELINE 3);
  // - main: the mode must cover the full 16:9 field (T2 `full_fov`; e.g. IMX519 1280x720 is a 2/3 crop),
  //   unless sensor_mode forces a mode index.
  if ((d.backend == CameraBackendKind::kArgus || d.backend == CameraBackendKind::kGst) &&
      (c.role == "ultrawide" || (c.role == "main" && d.sensor_mode < 0))) {
    const std::vector<SensorMode>& modes = known_sensor_modes(d.sensor);
    const auto ok = c.role == "ultrawide" ? covers_full_array : covers_full_field;
    for (const auto& md : modes) {
      if (md.width != d.width || md.height != d.height || ok(md, modes)) continue;
      char req[64];
      std::snprintf(req, sizeof(req), "%dx%d@%g", d.width, d.height, d.fps);
      if (c.role == "ultrawide") {
        m.error("driver", std::string(req) + " is a cropped mode on " + d.sensor + "; full-FOV modes are " +
                              describe_modes(modes, covers_full_array));
      } else {
        m.error("driver", std::string(req) + " is a crop of the full field on " + d.sensor +
                              " (narrower FOV); full-field modes are " + describe_modes(modes, covers_full_field) +
                              " (set sensor_mode to force it)");
      }
    }
  }
  m.finish();
}

void read_imu(MapReader m, RigImuConfig& imu) {
  if (!m.ok_map()) return;
  {
    MapReader d = m.sub("driver", true);
    ImuConfig& c = imu.driver;
    if (d.ok_map()) {
      if (d.str("id", c.id) && !std::regex_match(c.id, id_regex())) d.error("id", "'" + c.id + "' must match [a-z0-9_]+");
      d.enumeration("backend", c.backend, cfg::imu_backend_table(), true);
      d.i32("i2c_bus", c.i2c_bus, IRange{0, 255});
      d.i32("i2c_addr", c.i2c_addr, IRange{0x03, 0x77});
      d.i32("expected_who_am_i", c.expected_who_am_i, IRange{0, 0xFF});
      // ODR table: DS12814 CTRL1_XL / CTRL2_G ODR_[3:0] (high-performance mode values).
      static const double kOdrs[] = {12.5, 26, 52, 104, 208, 416, 833, 1666, 3332, 6667};
      if (d.f64("odr_hz", c.odr_hz, open_lo(0, 10000))) {
        bool ok = false;
        for (double o : kOdrs) ok = ok || std::abs(o - c.odr_hz) < 1e-9;
        if (!ok) d.error("odr_hz", cfg::fmt_double(c.odr_hz) + " is not one of 12.5, 26, 52, 104, 208, 416, 833, 1666, 3332, 6667");
      }
      if (d.i32("accel_fs_g", c.accel_fs_g, IRange{1, 16}) &&
          !(c.accel_fs_g == 2 || c.accel_fs_g == 4 || c.accel_fs_g == 8 || c.accel_fs_g == 16))
        d.error("accel_fs_g", std::to_string(c.accel_fs_g) + " is not one of 2, 4, 8, 16");  // DS12814 FS_XL
      if (d.i32("gyro_fs_dps", c.gyro_fs_dps, IRange{1, 4000}) &&
          !(c.gyro_fs_dps == 125 || c.gyro_fs_dps == 250 || c.gyro_fs_dps == 500 || c.gyro_fs_dps == 1000 ||
            c.gyro_fs_dps == 2000))
        d.error("gyro_fs_dps", std::to_string(c.gyro_fs_dps) + " is not one of 125, 250, 500, 1000, 2000");
      d.i32("fifo_watermark_words", c.fifo_watermark_words, IRange{1, 511});  // FIFO = 512 words (DS12814 9.x)
      d.i32("poll_period_ms", c.poll_period_ms, IRange{1, 1000});
      d.i32("bracket_period_ms", c.bracket_period_ms, IRange{1, 10000});
      d.i32("i2c_retries", c.i2c_retries, IRange{0, 10});
      d.i32("i2c_error_budget", c.i2c_error_budget, IRange{0, 100000});
      d.f64("gap_factor", c.gap_factor, DRange{1.0, 100.0, true, false});
      d.size("queue_capacity", c.queue_capacity, IRange{16, 10000000});
      // T1 FIFO options (rig/drivers/imu_backend.hpp). DS12814 FIFO_CTRL4.DEC_TS_BATCH: timestamp every 1/8/32
      // batches; FIFO_CTRL4.ODR_T_BATCH: temperature 1.6/12.5/52 Hz (0 = off).
      if (d.i32("timestamp_decimation", c.timestamp_decimation, IRange{1, 32}) &&
          !(c.timestamp_decimation == 1 || c.timestamp_decimation == 8 || c.timestamp_decimation == 32))
        d.error("timestamp_decimation", std::to_string(c.timestamp_decimation) + " is not one of 1, 8, 32");
      if (d.f64("temperature_batch_hz", c.temperature_batch_hz, closed(0, 52))) {
        bool ok = false;
        for (double t : {0.0, 1.6, 12.5, 52.0}) ok = ok || std::abs(t - c.temperature_batch_hz) < 1e-9;
        if (!ok) d.error("temperature_batch_hz", cfg::fmt_double(c.temperature_batch_hz) + " is not one of 0, 1.6, 12.5, 52");
      }
      d.i32("fifo_burst_words", c.fifo_burst_words, IRange{1, 256});  // T1 driver limit (R1 F19)
      d.boolean("pair_by_tag_counter", c.pair_by_tag_counter);
      // FIFO words per poll: gyro + accel (2 per ODR period) + one timestamp word per `timestamp_decimation`
      // ODR periods + temperature words at temperature_batch_hz. The FIFO (512 words) must stay below 50 %
      // between two polls (R1 F8).
      const double words_per_odr = 2.0 + 1.0 / std::max(1, c.timestamp_decimation);
      const double words_per_poll =
          (words_per_odr * c.odr_hz + c.temperature_batch_hz) * c.poll_period_ms / 1000.0;
      if (words_per_poll * 2.0 > 512.0)
        d.error("poll_period_ms", std::to_string(c.poll_period_ms) + " ms at " + cfg::fmt_double(c.odr_hz) +
                                      " Hz fills " + cfg::fmt_double(words_per_poll) +
                                      " of 512 FIFO words per poll (need < 50 %)");
      {
        MapReader k = d.sub("clock");
        if (k.ok_map()) {
          k.f64("nominal_tick_ns", c.clock.nominal_tick_ns, open_lo(0, 1e9));
          k.f64("max_slope_deviation", c.clock.max_slope_deviation, DRange{0, 0.5, true, false});
          k.i32("window_brackets", c.clock.window_brackets, IRange{2, 1000000});
          k.i64("max_bracket_width_ns", c.clock.max_bracket_width_ns, IRange{1, 1000000000});
          k.enumeration("method", c.clock.method, cfg::clock_fit_table());
          k.i64("outlier_gate_ns", c.clock.outlier_gate_ns, IRange{0, 1000000000});  // 0 = gate off
          k.finish();
        }
      }
      {
        MapReader s = d.sub("sim");
        if (s.ok_map()) {
          s.f64("chip_clock_drift_ppm", c.sim.chip_clock_drift_ppm, closed(-100000, 100000));
          s.i64("chip_clock_offset_ticks", c.sim.chip_clock_offset_ticks, IRange{0, kI64Max});
          s.f64("accel_noise_std_mps2", c.sim.accel_noise_std_mps2, closed(0, 100));
          s.f64("gyro_noise_std_rps", c.sim.gyro_noise_std_rps, closed(0, 100));
          s.boolean("motion", c.sim.motion);
          s.f64("i2c_error_rate", c.sim.i2c_error_rate, closed(0, 1));
          s.u64("seed", c.sim.seed);
          s.finish();
        }
      }
      {
        MapReader r = d.sub("replay");
        if (r.ok_map()) {
          r.str("path", c.replay.path);
          r.boolean("realtime", c.replay.realtime);
          r.boolean("retime", c.replay.retime);
          r.finish();
        }
      }
      if (c.backend == ImuBackendKind::kReplay && c.replay.path.empty())
        d.error("replay", "backend replay needs replay.path");
      d.finish();
    }
  }
  {
    MapReader n = m.sub("noise");
    if (n.ok_map()) {
      // Continuous-time densities: REQUIRED (ORB-SLAM3 / Kalibr settings are written from them).
      n.f64("accel_noise_density", imu.noise.accel_noise_density, open_lo(0, 10), true);
      n.f64("accel_random_walk", imu.noise.accel_random_walk, open_lo(0, 10), true);
      n.f64("gyro_noise_density", imu.noise.gyro_noise_density, open_lo(0, 10), true);
      n.f64("gyro_random_walk", imu.noise.gyro_random_walk, open_lo(0, 10), true);
      n.enumeration("provenance", imu.noise.provenance, cfg::provenance_table());
      n.str("note", imu.noise.note);
      n.finish();
    } else if (!n.present()) {
      m.error("noise", "missing required key (IMU noise densities feed the SLAM settings)");
    }
  }
  read_thread(m.sub("thread"), imu.thread);
  m.size("queue_samples", imu.queue_samples, IRange{1, 100000000});
  m.finish();
}

void read_virtual(MapReader m, VirtualCameraConfig& v) {
  if (!m.ok_map()) return;
  read_model(m.sub("model", true), v.model);
  if (v.model.kind != CameraModelKind::kVirtualPinhole) m.error("model", "virtual camera model kind must be 'pinhole'");
  if (m.mat3("R_phys_virt", v.R_phys_virt)) check_rotation(m, "R_phys_virt", v.R_phys_virt);
  m.f64("min_coverage", v.min_coverage, closed(0, 1));
  m.finish();
}

const std::regex& shm_prefix_regex() {
  static const std::regex r("/[A-Za-z0-9._-]+");
  return r;
}

// ---------------------------------------------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------------------------------------------

void read_root(MapReader& root, RigConfig& c) {
  if (root.i32("schema_version", c.schema_version, IRange{0, 1000000}, true) &&
      c.schema_version != kRigConfigSchemaVersion) {
    root.error("schema_version", std::to_string(c.schema_version) + " is not supported (this build reads " +
                                     std::to_string(kRigConfigSchemaVersion) + ")");
  }
  if (root.str("rig_name", c.rig_name, true) && c.rig_name.empty()) root.error("rig_name", "must not be empty");
  root.str("notes", c.notes);

  {
    MapReader cams = root.sub("cameras", true);
    if (cams.ok_map()) {
      for (const std::string& id : cams.keys()) {
        if (!std::regex_match(id, id_regex()) || id.size() > 15) {
          // ShmFrameHeader::camera_id / CaptureStreamStatus::id are char[16]; dirs are raw/cam_<id>.
          cams.error(id, "camera id must match [a-z0-9_]+ and be <= 15 characters");
          cams.take(id);
          continue;
        }
        RigCameraConfig cam;
        cam.driver.id = id;
        read_camera(cams.sub(id), cam);
        c.cameras[id] = std::move(cam);
      }
      cams.finish();
    }
    int n_main = 0, n_uw = 0;
    for (const auto& kv : c.cameras) {
      n_main += kv.second.role == "main";
      n_uw += kv.second.role == "ultrawide";
    }
    if (cams.ok_map() && (n_main != 1 || n_uw != 1))
      cams.error_here("need exactly one camera with role main and one with role ultrawide (got " +
                      std::to_string(n_main) + " main, " + std::to_string(n_uw) + " ultrawide)");
    std::map<int, std::string> ids;
    for (const auto& kv : c.cameras) {
      const CameraConfig& d = kv.second.driver;
      if (d.backend != CameraBackendKind::kArgus && d.backend != CameraBackendKind::kGst) continue;
      auto [it, ok] = ids.emplace(d.sensor_id, kv.first);
      if (!ok)
        cams.error(kv.first + ".driver.sensor_id", "sensor_id " + std::to_string(d.sensor_id) +
                                                       " is also used by camera '" + it->second + "'");
    }
  }

  read_imu(root.sub("imu", true), c.imu);

  {
    MapReader vc = root.sub("virtual_cameras");
    if (vc.ok_map()) {
      for (const std::string& id : vc.keys()) {
        VirtualCameraConfig v;
        read_virtual(vc.sub(id), v);
        if (c.cameras.count(id) == 0) vc.error(id, "no physical camera '" + id + "' in cameras");
        c.virtual_cameras[id] = std::move(v);
      }
      vc.finish();
    }
  }
  {
    MapReader m = root.sub("imageprep");
    if (m.ok_map()) {
      m.i32("out_size", c.imageprep.out_size, IRange{16, 4096});
      m.finish();
    }
  }
  {
    MapReader m = root.sub("capture");
    if (m.ok_map()) {
      MapReader s = m.sub("steady_state");
      if (s.ok_map()) {
        s.i32("min_frames", c.capture.steady_state.min_frames, IRange{1, 100000});
        s.f64("rate_tolerance", c.capture.steady_state.rate_tolerance, DRange{0, 1, true, true});
        s.f64("imu_window_s", c.capture.steady_state.imu_window_s, open_lo(0, 60));
        s.finish();
      }
      m.i64("read_timeout_ms", c.capture.read_timeout_ms, IRange{1, 10000});
      m.finish();
    }
  }
  {
    MapReader m = root.sub("live_checks");
    LiveChecksConfig& l = c.live_checks;
    if (m.ok_map()) {
      m.str("main_camera", l.main_camera);
      m.f64("main_fps_min", l.main_fps_min, open_lo(0, 1000));
      m.f64("main_fps_grace_s", l.main_fps_grace_s, closed(0, 60));
      m.f64("pose_speed_max_mps", l.pose_speed_max_mps, open_lo(0, 1000));
      m.boolean("uw_require_frames", l.uw_require_frames);
      m.str("uw_camera", l.uw_camera);
      m.f64("uw_fps_min", l.uw_fps_min, closed(0, 1000));
      m.f64("uw_fps_window_s", l.uw_fps_window_s, open_lo(0, 60));
      m.boolean("slam_abort_on_lost", l.slam_abort_on_lost);
      m.boolean("slam_abort_on_reset", l.slam_abort_on_reset);
      m.boolean("slam_abort_on_map_change", l.slam_abort_on_map_change);
      m.f64("slam_heartbeat_timeout_s", l.slam_heartbeat_timeout_s, open_lo(0, 60));
      m.f64("imu_gap_max_ms", l.imu_gap_max_ms, open_lo(0, 10000));
      m.f64("imu_rate_min_ratio", l.imu_rate_min_ratio, closed(0, 1));
      m.f64("imu_rate_window_s", l.imu_rate_window_s, open_lo(0, 60));
      m.boolean("imu_abort_on_overrun", l.imu_abort_on_overrun);
      m.boolean("writer_abort_on_overflow", l.writer_abort_on_overflow);
      m.f64("disk_min_free_gb", l.disk_min_free_gb, closed(0, 100000));
      m.f64("thermal_warn_c", l.thermal_warn_c, open_lo(0, 150));
      m.boolean("thermal_abort_on_throttle", l.thermal_abort_on_throttle);
      if (m.str_list("thermal_zones", l.thermal_zones)) {
        for (const auto& z : l.thermal_zones)
          if (z.empty()) m.error("thermal_zones", "empty zone name");
      }
      m.f64("tags_unseen_warn_s", l.tags_unseen_warn_s, open_lo(0, 3600));
      m.finish();
    }
    {
      // Cross-checks against the cameras (also when the section is omitted and defaults apply).
      auto cam = c.cameras.find(l.main_camera);
      if (cam == c.cameras.end()) {
        m.error("main_camera", "no camera '" + l.main_camera + "' in cameras");
      } else if (cam->second.driver.fps > 0 && l.main_fps_min >= cam->second.driver.fps) {
        m.error("main_fps_min", cfg::fmt_double(l.main_fps_min) + " >= the camera's " +
                                    cfg::fmt_double(cam->second.driver.fps) + " fps: every frame would abort");
      }
      cam = c.cameras.find(l.uw_camera);
      if (cam == c.cameras.end()) {
        m.error("uw_camera", "no camera '" + l.uw_camera + "' in cameras");
      } else if (cam->second.driver.fps > 0 && l.uw_fps_min >= cam->second.driver.fps) {
        m.error("uw_fps_min", cfg::fmt_double(l.uw_fps_min) + " >= the camera's " +
                                  cfg::fmt_double(cam->second.driver.fps) + " fps: every window would abort");
      }
      if (c.imu.driver.odr_hz > 0 && l.imu_gap_max_ms * 1e-3 * c.imu.driver.odr_hz < 1.0)
        m.error("imu_gap_max_ms", "shorter than one IMU sample period at " + cfg::fmt_double(c.imu.driver.odr_hz) + " Hz");
    }
  }
  {
    MapReader m = root.sub("record");
    if (m.ok_map()) {
      m.f64("arming_timeout_s", c.record.arming_timeout_s, open_lo(0, 3600));
      m.boolean("require_gripper_calibration", c.record.require_gripper_calibration);
      m.f64("stop_flush_timeout_s", c.record.stop_flush_timeout_s, open_lo(0, 3600));
      m.finish();
    }
  }
  {
    MapReader m = root.sub("slam_live");
    SlamLiveConfig& s = c.slam_live;
    if (m.ok_map()) {
      m.boolean("enabled", s.enabled);
      m.enumeration("backend", s.backend, cfg::slam_backend_table());
      m.str("camera", s.camera);
      m.str("vocabulary_path", s.vocabulary_path);
      m.i32("input_width", s.input_width, IRange{16, 16384});
      m.i32("input_height", s.input_height, IRange{16, 16384});
      m.i32("n_features", s.n_features, IRange{100, 20000});
      m.i32("frame_decimation", s.frame_decimation, IRange{1, 100});
      m.boolean("viewer", s.viewer);
      m.f64("viewer_refresh_hz", s.viewer_refresh_hz, open_lo(0, 240));
      read_thread(m.sub("thread"), s.thread);
      MapReader k = m.sub("mock");
      if (k.ok_map()) {
        k.str("trajectory_tum", s.mock.trajectory_tum);
        k.f64("init_delay_s", s.mock.init_delay_s, closed(0, 3600));
        k.finish();
      }
      m.finish();
    }
    {
      auto cam = c.cameras.find(s.camera);
      if (cam == c.cameras.end()) {
        m.error("camera", "no camera '" + s.camera + "' in cameras");
      } else if (s.input_width > cam->second.driver.width || s.input_height > cam->second.driver.height) {
        m.error("input_width", "input " + std::to_string(s.input_width) + "x" + std::to_string(s.input_height) +
                                   " is larger than the camera capture");
      }
      if (s.enabled && s.backend == SlamBackendKind::kOrbSlam3 && s.vocabulary_path.empty())
        m.error("vocabulary_path", "backend orbslam3 needs the ORB vocabulary path");
    }
  }
  {
    MapReader m = root.sub("viz");
    if (m.ok_map()) {
      m.enumeration("mode", c.viz.mode, cfg::viz_mode_table());
      m.f64("refresh_hz", c.viz.refresh_hz, open_lo(0, 240));
      m.i32("mosaic_width", c.viz.mosaic_width, IRange{160, 7680});
      m.boolean("tag_overlay", c.viz.tag_overlay);
      m.f64("tag_detect_hz", c.viz.tag_detect_hz, open_lo(0, 240));
      m.finish();
    }
  }
  {
    MapReader m = root.sub("shm");
    if (m.ok_map()) {
      if (m.str("prefix", c.shm.prefix) && (!std::regex_match(c.shm.prefix, shm_prefix_regex()) || c.shm.prefix.size() > 100))
        m.error("prefix", "'" + c.shm.prefix + "' must be '/' + [A-Za-z0-9._-]+ (POSIX shm name, <= 100 chars)");
      m.size("frame_ring_slots", c.shm.frame_ring_slots, IRange{2, 1024});
      m.i32("viz_frame_width", c.shm.viz_frame_width, IRange{0, 16384});
      if (c.shm.viz_frame_width != 0 && (c.shm.viz_frame_width < 16 || c.shm.viz_frame_width % 2 != 0))
        m.error("viz_frame_width", "must be 0 (native) or an even width >= 16");
      m.size("imu_ring_slots", c.shm.imu_ring_slots, IRange{2, 10000000});
      m.finish();
    }
  }
  {
    MapReader m = root.sub("storage");
    if (m.ok_map()) {
      if (m.str("data_root", c.storage.data_root) && c.storage.data_root.empty()) m.error("data_root", "must not be empty");
      m.boolean("fsync_on_stop", c.storage.fsync_on_stop);
      m.finish();
    }
  }
  {
    MapReader m = root.sub("calibration");
    if (m.ok_map()) {
      m.enumeration("provenance", c.calibration.provenance, cfg::provenance_table());
      m.str("source", c.calibration.source);
      static const std::regex kDate("[0-9]{4}-[0-9]{2}-[0-9]{2}");
      if (m.str("date", c.calibration.date) && !c.calibration.date.empty() &&
          !std::regex_match(c.calibration.date, kDate))
        m.error("date", "'" + c.calibration.date + "' must be \"\" or YYYY-MM-DD");
      read_extrinsics(m.sub("T_uw_tcp"), c.calibration.T_uw_tcp);
      m.finish();
    }
  }
  // R1 F18: the IMU poll thread needs a core of its own. A camera backend's internal threads (Argus NV12 copy)
  // inherit its capture thread's affinity, so sharing a core delayed IMU polls by ~50 ms in R1's run.
  {
    const std::set<int> imu_cpus(c.imu.thread.cpus.begin(), c.imu.thread.cpus.end());
    auto clash = [&](const std::string& who, const std::vector<int>& cpus) {
      for (int cpu : cpus)
        if (imu_cpus.count(cpu))
          root.error("imu.thread.cpus", "CPU " + std::to_string(cpu) + " is shared with " + who +
                                            " (the IMU poll thread needs its own core)");
    };
    for (const auto& kv : c.cameras) clash("cameras." + kv.first + ".thread.cpus", kv.second.thread.cpus);
    if (c.slam_live.enabled) clash("slam_live.thread.cpus", c.slam_live.thread.cpus);
  }
  root.finish();
}

}  // namespace

Result<RigConfig> parse_rig_config(const std::string& yaml_text, const std::string& source_name) {
  YAML::Node doc;
  try {
    doc = YAML::Load(yaml_text);
  } catch (const YAML::Exception& e) {
    return invalid_argument("<root>: YAML syntax error: " + e.msg + " (" + source_name + ":" +
                            std::to_string(e.mark.line + 1) + ":" + std::to_string(e.mark.column + 1) + ")");
  }
  Issues issues(source_name);
  RigConfig c;
  try {
    MapReader root(issues, doc, "", 0);
    if (root.ok_map()) read_root(root, c);
  } catch (const std::exception& e) {  // yaml-cpp accessors on malformed nodes; should not happen
    issues.add("", -1, std::string("internal YAML access error: ") + e.what());
  }
  if (!issues.empty()) return invalid_argument(issues.summary());
  c.source_path = source_name;
  return c;
}

Result<RigConfig> load_rig_config(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return not_found(path + ": cannot open rig config");
  std::ostringstream ss;
  ss << in.rdbuf();
  if (in.bad()) return io_error(path + ": read error");
  return parse_rig_config(ss.str(), path);
}

}  // namespace rig
