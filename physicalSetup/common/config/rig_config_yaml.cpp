// SPDX-License-Identifier: MIT
// Canonical YAML (to_yaml) and the content hash of a RigConfig (rig/config/rig_config.hpp).
//
// Hand-written emitter (not YAML::Emitter) so the canonical text, and therefore content_hash(), does not depend on the
// yaml-cpp version installed on the laptop or the Jetson. Rules: keys in struct order, 2-space indent, every string
// double-quoted with JSON-style escapes, doubles as the shortest %.17g form that round-trips (".nan"/".inf" for
// non-finite values, which the loader rejects), lists in flow style.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "enum_tables.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/package/sha256.hpp"

namespace rig {
namespace {

std::string q(const std::string& s) {
  std::string o = "\"";
  for (unsigned char ch : s) {
    switch (ch) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      case '\r': o += "\\r"; break;
      default:
        if (ch < 0x20 || ch == 0x7F) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\x%02X", ch);
          o += b;
        } else {
          o += static_cast<char>(ch);
        }
    }
  }
  return o + "\"";
}

std::string d(double v) {
  if (std::isnan(v)) return ".nan";
  if (std::isinf(v)) return v > 0 ? ".inf" : "-.inf";
  char buf[40];
  for (int prec = 1; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
    if (std::strtod(buf, nullptr) == v) break;
  }
  return buf;
}

std::string b(bool v) { return v ? "true" : "false"; }

class Out {
 public:
  void kv(int ind, const std::string& key, const std::string& value) {
    s_ += std::string(static_cast<std::size_t>(ind) * 2, ' ') + key + ": " + value + "\n";
  }
  void open(int ind, const std::string& key) { s_ += std::string(static_cast<std::size_t>(ind) * 2, ' ') + key + ":\n"; }
  std::string s_;
};

std::string vec3(const Eigen::Vector3d& v) { return "[" + d(v[0]) + ", " + d(v[1]) + ", " + d(v[2]) + "]"; }
std::string mat3(const Eigen::Matrix3d& m) {
  std::string s = "[";
  for (int r = 0; r < 3; ++r) {
    if (r) s += ", ";
    s += "[" + d(m(r, 0)) + ", " + d(m(r, 1)) + ", " + d(m(r, 2)) + "]";
  }
  return s + "]";
}
std::string dlist(const std::vector<double>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + d(v[i]);
  return s + "]";
}
std::string ilist(const std::vector<int>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
  return s + "]";
}
std::string slist(const std::vector<std::string>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + q(v[i]);
  return s + "]";
}

void thread(Out& o, int i, const ThreadConfig& t) {
  o.kv(i, "cpus", ilist(t.cpus));
  o.kv(i, "priority", std::to_string(t.priority));
  o.kv(i, "nice", std::to_string(t.nice));
}

void extr(Out& o, int i, const Extrinsics& e) {
  o.kv(i, "rotation", mat3(e.rotation));
  o.kv(i, "translation_m", vec3(e.translation_m));
  o.kv(i, "provenance", to_string(e.provenance));
  o.kv(i, "note", q(e.note));
}

void model(Out& o, int i, const CameraModelParams& p) {
  o.kv(i, "kind", cfg::name_of(cfg::camera_model_table(), p.kind));
  o.kv(i, "width", std::to_string(p.width));
  o.kv(i, "height", std::to_string(p.height));
  o.kv(i, "fx", d(p.fx));
  o.kv(i, "fy", d(p.fy));
  o.kv(i, "cx", d(p.cx));
  o.kv(i, "cy", d(p.cy));
  o.kv(i, "distortion", dlist(p.distortion));
  o.kv(i, "fov_max_deg", d(p.fov_max_deg));
  o.kv(i, "provenance", to_string(p.provenance));
  o.kv(i, "note", q(p.note));
}

void driver(Out& o, int i, const CameraConfig& c) {
  o.kv(i, "backend", cfg::name_of(cfg::camera_backend_table(), c.backend));
  o.kv(i, "sensor", q(c.sensor));
  o.kv(i, "sensor_id", std::to_string(c.sensor_id));
  o.kv(i, "width", std::to_string(c.width));
  o.kv(i, "height", std::to_string(c.height));
  o.kv(i, "fps", d(c.fps));
  o.kv(i, "sensor_mode", std::to_string(c.sensor_mode));
  o.kv(i, "pixel_format", cfg::name_of(cfg::pixel_format_table(), c.pixel_format));
  o.kv(i, "exposure_ns", std::to_string(c.exposure_ns));
  o.kv(i, "analog_gain", d(c.analog_gain));
  o.kv(i, "isp_digital_gain", d(c.isp_digital_gain));
  o.kv(i, "ae_lock", b(c.ae_lock));
  o.kv(i, "awb_lock", b(c.awb_lock));
  o.kv(i, "awb_mode", q(c.awb_mode));
  o.kv(i, "denoise", b(c.denoise));
  o.kv(i, "edge_enhance", b(c.edge_enhance));
  o.kv(i, "buffer_count", std::to_string(c.buffer_count));
  o.kv(i, "drop_gap_factor", d(c.drop_gap_factor));
  o.open(i, "focus");
  o.kv(i + 1, "mode", cfg::name_of(cfg::focus_mode_table(), c.focus.mode));
  o.kv(i + 1, "position", std::to_string(c.focus.position));
  o.kv(i + 1, "i2c_bus", std::to_string(c.focus.i2c_bus));
  o.kv(i + 1, "i2c_addr", std::to_string(c.focus.i2c_addr));
  o.kv(i + 1, "allow_v4l2_fallback", b(c.focus.allow_v4l2_fallback));
  o.kv(i + 1, "v4l2_subdev", q(c.focus.v4l2_subdev));
  o.open(i, "gst");
  o.kv(i + 1, "source_override", q(c.gst.source_override));
  o.open(i, "mock");
  o.kv(i + 1, "pattern", q(c.mock.pattern));
  o.kv(i + 1, "jitter_ns", std::to_string(c.mock.jitter_ns));
  o.kv(i + 1, "drop_every_n", std::to_string(c.mock.drop_every_n));
  o.kv(i + 1, "stall_after_frames", std::to_string(c.mock.stall_after_frames));
  o.kv(i + 1, "stall_ns", std::to_string(c.mock.stall_ns));
  o.kv(i + 1, "seed", std::to_string(c.mock.seed));
  o.open(i, "replay");
  o.kv(i + 1, "path", q(c.replay.path));
  o.kv(i + 1, "realtime", b(c.replay.realtime));
  o.kv(i + 1, "loop", b(c.replay.loop));
  o.kv(i + 1, "retime", b(c.replay.retime));
}

}  // namespace

std::string to_yaml(const RigConfig& c) {
  Out o;
  o.kv(0, "schema_version", std::to_string(c.schema_version));
  o.kv(0, "rig_name", q(c.rig_name));
  o.kv(0, "notes", q(c.notes));
  if (c.cameras.empty()) {
    o.kv(0, "cameras", "{}");
  } else {
    o.open(0, "cameras");
    for (const auto& kv : c.cameras) {
      const RigCameraConfig& cam = kv.second;
      o.open(1, q(kv.first));
      o.kv(2, "role", q(cam.role));
      o.open(2, "driver");
      driver(o, 3, cam.driver);
      o.open(2, "model");
      model(o, 3, cam.model);
      o.open(2, "T_imu_cam");
      extr(o, 3, cam.T_imu_cam);
      o.open(2, "time");
      o.kv(3, "offset_ns", std::to_string(cam.time.offset_ns));
      o.kv(3, "provenance", to_string(cam.time.provenance));
      o.open(2, "storage");
      o.kv(3, "width", std::to_string(cam.storage.width));
      o.kv(3, "height", std::to_string(cam.storage.height));
      o.kv(3, "jpeg_quality", std::to_string(cam.storage.jpeg_quality));
      o.kv(3, "encoder_threads", std::to_string(cam.storage.encoder_threads));
      o.kv(3, "queue_frames", std::to_string(cam.storage.queue_frames));
      o.open(2, "thread");
      thread(o, 3, cam.thread);
    }
  }
  const ImuConfig& im = c.imu.driver;
  o.open(0, "imu");
  o.open(1, "driver");
  o.kv(2, "id", q(im.id));
  o.kv(2, "backend", cfg::name_of(cfg::imu_backend_table(), im.backend));
  o.kv(2, "i2c_bus", std::to_string(im.i2c_bus));
  o.kv(2, "i2c_addr", std::to_string(im.i2c_addr));
  o.kv(2, "expected_who_am_i", std::to_string(im.expected_who_am_i));
  o.kv(2, "odr_hz", d(im.odr_hz));
  o.kv(2, "accel_fs_g", std::to_string(im.accel_fs_g));
  o.kv(2, "gyro_fs_dps", std::to_string(im.gyro_fs_dps));
  o.kv(2, "fifo_watermark_words", std::to_string(im.fifo_watermark_words));
  o.kv(2, "poll_period_ms", std::to_string(im.poll_period_ms));
  o.kv(2, "bracket_period_ms", std::to_string(im.bracket_period_ms));
  o.kv(2, "i2c_retries", std::to_string(im.i2c_retries));
  o.kv(2, "i2c_error_budget", std::to_string(im.i2c_error_budget));
  o.kv(2, "gap_factor", d(im.gap_factor));
  o.kv(2, "queue_capacity", std::to_string(im.queue_capacity));
  o.kv(2, "timestamp_decimation", std::to_string(im.timestamp_decimation));
  o.kv(2, "temperature_batch_hz", d(im.temperature_batch_hz));
  o.kv(2, "fifo_burst_words", std::to_string(im.fifo_burst_words));
  o.kv(2, "pair_by_tag_counter", b(im.pair_by_tag_counter));
  o.open(2, "clock");
  o.kv(3, "nominal_tick_ns", d(im.clock.nominal_tick_ns));
  o.kv(3, "max_slope_deviation", d(im.clock.max_slope_deviation));
  o.kv(3, "window_brackets", std::to_string(im.clock.window_brackets));
  o.kv(3, "max_bracket_width_ns", std::to_string(im.clock.max_bracket_width_ns));
  o.kv(3, "method", cfg::name_of(cfg::clock_fit_table(), im.clock.method));
  o.kv(3, "outlier_gate_ns", std::to_string(im.clock.outlier_gate_ns));
  o.open(2, "sim");
  o.kv(3, "chip_clock_drift_ppm", d(im.sim.chip_clock_drift_ppm));
  o.kv(3, "chip_clock_offset_ticks", std::to_string(im.sim.chip_clock_offset_ticks));
  o.kv(3, "accel_noise_std_mps2", d(im.sim.accel_noise_std_mps2));
  o.kv(3, "gyro_noise_std_rps", d(im.sim.gyro_noise_std_rps));
  o.kv(3, "motion", b(im.sim.motion));
  o.kv(3, "i2c_error_rate", d(im.sim.i2c_error_rate));
  o.kv(3, "seed", std::to_string(im.sim.seed));
  o.open(2, "replay");
  o.kv(3, "path", q(im.replay.path));
  o.kv(3, "realtime", b(im.replay.realtime));
  o.kv(3, "retime", b(im.replay.retime));
  o.open(1, "noise");
  o.kv(2, "accel_noise_density", d(c.imu.noise.accel_noise_density));
  o.kv(2, "accel_random_walk", d(c.imu.noise.accel_random_walk));
  o.kv(2, "gyro_noise_density", d(c.imu.noise.gyro_noise_density));
  o.kv(2, "gyro_random_walk", d(c.imu.noise.gyro_random_walk));
  o.kv(2, "provenance", to_string(c.imu.noise.provenance));
  o.kv(2, "note", q(c.imu.noise.note));
  o.open(1, "thread");
  thread(o, 2, c.imu.thread);
  o.kv(1, "queue_samples", std::to_string(c.imu.queue_samples));

  if (c.virtual_cameras.empty()) {
    o.kv(0, "virtual_cameras", "{}");
  } else {
    o.open(0, "virtual_cameras");
    for (const auto& kv : c.virtual_cameras) {
      o.open(1, q(kv.first));
      o.open(2, "model");
      model(o, 3, kv.second.model);
      o.kv(2, "R_phys_virt", mat3(kv.second.R_phys_virt));
      o.kv(2, "min_coverage", d(kv.second.min_coverage));
    }
  }
  o.open(0, "imageprep");
  o.kv(1, "out_size", std::to_string(c.imageprep.out_size));
  o.open(0, "capture");
  o.open(1, "steady_state");
  o.kv(2, "min_frames", std::to_string(c.capture.steady_state.min_frames));
  o.kv(2, "rate_tolerance", d(c.capture.steady_state.rate_tolerance));
  o.kv(2, "imu_window_s", d(c.capture.steady_state.imu_window_s));
  o.kv(1, "read_timeout_ms", std::to_string(c.capture.read_timeout_ms));
  const LiveChecksConfig& l = c.live_checks;
  o.open(0, "live_checks");
  o.kv(1, "main_camera", q(l.main_camera));
  o.kv(1, "main_fps_min", d(l.main_fps_min));
  o.kv(1, "main_fps_grace_s", d(l.main_fps_grace_s));
  o.kv(1, "pose_speed_max_mps", d(l.pose_speed_max_mps));
  o.kv(1, "uw_require_frames", b(l.uw_require_frames));
  o.kv(1, "uw_camera", q(l.uw_camera));
  o.kv(1, "uw_fps_min", d(l.uw_fps_min));
  o.kv(1, "uw_fps_window_s", d(l.uw_fps_window_s));
  o.kv(1, "slam_abort_on_lost", b(l.slam_abort_on_lost));
  o.kv(1, "slam_abort_on_reset", b(l.slam_abort_on_reset));
  o.kv(1, "slam_abort_on_map_change", b(l.slam_abort_on_map_change));
  o.kv(1, "slam_heartbeat_timeout_s", d(l.slam_heartbeat_timeout_s));
  o.kv(1, "imu_gap_max_ms", d(l.imu_gap_max_ms));
  o.kv(1, "imu_rate_min_ratio", d(l.imu_rate_min_ratio));
  o.kv(1, "imu_rate_window_s", d(l.imu_rate_window_s));
  o.kv(1, "imu_abort_on_overrun", b(l.imu_abort_on_overrun));
  o.kv(1, "writer_abort_on_overflow", b(l.writer_abort_on_overflow));
  o.kv(1, "disk_min_free_gb", d(l.disk_min_free_gb));
  o.kv(1, "thermal_warn_c", d(l.thermal_warn_c));
  o.kv(1, "thermal_abort_on_throttle", b(l.thermal_abort_on_throttle));
  o.kv(1, "thermal_zones", slist(l.thermal_zones));
  o.kv(1, "tags_unseen_warn_s", d(l.tags_unseen_warn_s));
  o.open(0, "record");
  o.kv(1, "arming_timeout_s", d(c.record.arming_timeout_s));
  o.kv(1, "require_gripper_calibration", b(c.record.require_gripper_calibration));
  o.kv(1, "stop_flush_timeout_s", d(c.record.stop_flush_timeout_s));
  const SlamLiveConfig& s = c.slam_live;
  o.open(0, "slam_live");
  o.kv(1, "enabled", b(s.enabled));
  o.kv(1, "backend", cfg::name_of(cfg::slam_backend_table(), s.backend));
  o.kv(1, "camera", q(s.camera));
  o.kv(1, "vocabulary_path", q(s.vocabulary_path));
  o.kv(1, "input_width", std::to_string(s.input_width));
  o.kv(1, "input_height", std::to_string(s.input_height));
  o.kv(1, "n_features", std::to_string(s.n_features));
  o.kv(1, "frame_decimation", std::to_string(s.frame_decimation));
  o.kv(1, "viewer", b(s.viewer));
  o.kv(1, "viewer_refresh_hz", d(s.viewer_refresh_hz));
  o.open(1, "thread");
  thread(o, 2, s.thread);
  o.open(1, "mock");
  o.kv(2, "trajectory_tum", q(s.mock.trajectory_tum));
  o.kv(2, "init_delay_s", d(s.mock.init_delay_s));
  o.open(0, "viz");
  o.kv(1, "mode", cfg::name_of(cfg::viz_mode_table(), c.viz.mode));
  o.kv(1, "refresh_hz", d(c.viz.refresh_hz));
  o.kv(1, "mosaic_width", std::to_string(c.viz.mosaic_width));
  o.kv(1, "tag_overlay", b(c.viz.tag_overlay));
  o.kv(1, "tag_detect_hz", d(c.viz.tag_detect_hz));
  o.open(0, "shm");
  o.kv(1, "prefix", q(c.shm.prefix));
  o.kv(1, "frame_ring_slots", std::to_string(c.shm.frame_ring_slots));
  o.kv(1, "viz_frame_width", std::to_string(c.shm.viz_frame_width));
  o.kv(1, "imu_ring_slots", std::to_string(c.shm.imu_ring_slots));
  o.open(0, "storage");
  o.kv(1, "data_root", q(c.storage.data_root));
  o.kv(1, "fsync_on_stop", b(c.storage.fsync_on_stop));
  o.open(0, "calibration");
  o.kv(1, "provenance", to_string(c.calibration.provenance));
  o.kv(1, "source", q(c.calibration.source));
  o.kv(1, "date", q(c.calibration.date));
  o.open(1, "T_uw_tcp");
  extr(o, 2, c.calibration.T_uw_tcp);
  return o.s_;
}

std::string content_hash(const RigConfig& config) { return sha256_hex(to_yaml(config)); }

}  // namespace rig
