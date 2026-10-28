// SPDX-License-Identifier: MIT
// ec-cam-probe: camera bring-up probe for hardware day (PLAN T2, test H2; BRINGUP step 10).
//
//   ec-cam-probe --config rig.yaml [--camera main --camera uw] --list
//   ec-cam-probe --config rig.yaml --seconds 300 [--together] [--save-frames 3] [--out DIR] [--json FILE]
//   ec-cam-probe --config rig.yaml --camera main --focus-sweep 0:4095:256
//   ec-cam-probe --backend gst --gst-source "videotestsrc is-live=true" --width 640 --height 480 --fps 30 --seconds 2
//
// Per camera: lists the sensor modes, captures N seconds and reports fps, SOF interval jitter (p50/p99/max), drops
// (seq / time / backend queue / error), t_arrival - t_sof latency, exposure and gain readback (confirms the locks),
// the focus status and the timestamp source (with the Argus TSC cross-check), then PASS/FAIL against H2. Saves
// sample frames as JPEG. Exit code: 0 all PASS, 1 a check failed, 2 setup error.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <CLI11.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "rig/config/rig_config.hpp"
#include "rig/drivers/camera/imx519_focus.hpp"
#include "rig/drivers/camera/probe_report.hpp"
#include "rig/drivers/camera/sensor_modes.hpp"
#include "rig/drivers/camera_backend.hpp"
#include "rig/time/clock.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

// Sanitizer builds only (weak hooks, no effect otherwise): libglib allocates process-lifetime state in a library
// constructor run by the loader; see tests/camera/gst/test_gst_camera.cpp.
extern "C" const char* __lsan_default_suppressions() { return "leak:ld-linux\n"; }
extern "C" const char* __asan_default_options() { return "fast_unwind_on_malloc=0"; }

namespace {

bool save_frame(const rig::ImageFrame& f, const fs::path& path) {
  if (!f.valid()) return false;
  cv::Mat bgr;
  auto* d = const_cast<uint8_t*>(f.data.get());  // cv::Mat wants non-const; read-only use
  switch (f.fmt) {
    case rig::PixelFormat::kNV12:
      cv::cvtColor(cv::Mat(f.height * 3 / 2, f.width, CV_8UC1, d), bgr, cv::COLOR_YUV2BGR_NV12);
      break;
    case rig::PixelFormat::kGRAY8: bgr = cv::Mat(f.height, f.width, CV_8UC1, d, f.planes[0].stride); break;
    case rig::PixelFormat::kBGR8: bgr = cv::Mat(f.height, f.width, CV_8UC3, d, f.planes[0].stride); break;
    case rig::PixelFormat::kRGB8:
      cv::cvtColor(cv::Mat(f.height, f.width, CV_8UC3, d, f.planes[0].stride), bgr, cv::COLOR_RGB2BGR);
      break;
    case rig::PixelFormat::kYUYV:
      cv::cvtColor(cv::Mat(f.height, f.width, CV_8UC2, d, f.planes[0].stride), bgr, cv::COLOR_YUV2BGR_YUYV);
      break;
    default: return false;
  }
  return cv::imwrite(path.string(), bgr);
}

nlohmann::json stats_json(const rig::SummaryStats& s) {
  return {{"count", s.count}, {"min", s.min}, {"max", s.max}, {"mean", s.mean}, {"p50", s.p50}, {"p99", s.p99}};
}

nlohmann::json report_json(const rig::ProbeReport& r) {
  nlohmann::json checks = nlohmann::json::array();
  for (const auto& c : r.checks) checks.push_back({{"name", c.name}, {"pass", c.pass}, {"detail", c.detail}});
  const rig::CameraStats& s = r.stats;
  return {{"camera", r.camera_id},
          {"pass", r.pass()},
          {"frames", r.frames},
          {"duration_s", r.duration_s},
          {"fps", r.fps},
          {"interval_ns", stats_json(r.interval_ns)},
          {"jitter_ns", stats_json(r.jitter_ns)},
          {"latency_ns", stats_json(r.latency_ns)},
          {"mid_minus_sof_ns", stats_json(r.mid_minus_sof_ns)},
          {"exposure_ns", {{"min", r.exposure_min_ns}, {"max", r.exposure_max_ns}, {"mean", r.exposure_mean_ns}}},
          {"analog_gain", {{"min", r.gain_min}, {"max", r.gain_max}, {"mean", r.gain_mean}}},
          {"digital_gain", {{"min", r.digital_gain_min}, {"max", r.digital_gain_max}}},
          {"readout_ns", r.readout_ns},
          {"drops", {{"seq_gap", s.drops_seq_gap}, {"time_gap", s.drops_time_gap}, {"backend_queue", s.drops_backend_queue},
                     {"error", s.drops_error}, {"errors", s.errors}, {"timestamp_anomalies", s.timestamp_anomalies}}},
          {"sensor_mode", s.sensor_mode},
          {"time_source", s.time_source},
          {"tsc_cross_check_ns", stats_json(s.tsc_cross_check_ns)},
          {"focus", {{"status", rig::to_string(s.focus_status)}, {"position", s.focus_position}, {"detail", s.focus_detail}}},
          {"checks", checks}};
}

struct Job {
  rig::CameraConfig cfg;
  std::unique_ptr<rig::CameraBackend> cam;
  rig::ProbeReport report;
  bool ok = false;
  std::string error;
};

void capture(Job& j, double seconds, int save_frames, const fs::path& out) {
  rig::Status s = j.cam->start();
  if (!s.ok()) {
    j.error = s.to_string();
    return;
  }
  rig::ProbeAnalyzer an(j.cfg.id, j.cfg.fps);
  const rig::TimeNs end = rig::now_raw_ns() + static_cast<rig::TimeNs>(seconds * 1e9);
  const uint64_t expect = static_cast<uint64_t>(seconds * j.cfg.fps);
  uint64_t next_save = 0;
  int saved = 0;
  while (rig::now_raw_ns() < end) {
    auto f = j.cam->read(500ms);
    if (!f) {
      if (!j.cam->status().ok()) break;
      continue;
    }
    an.add(*f);
    if (saved < save_frames && an.frames() > next_save) {
      const fs::path p = out / (j.cfg.id + "_" + std::to_string(f->seq) + ".jpg");
      if (save_frame(*f, p)) spdlog::info("{}: saved {}", j.cfg.id, p.string());
      ++saved;
      next_save = save_frames > 1 ? (expect * static_cast<uint64_t>(saved)) / static_cast<uint64_t>(save_frames) : expect;
    }
  }
  const rig::CameraStats st = j.cam->stats();
  const rig::Status final_status = j.cam->status();
  (void)j.cam->stop();
  rig::ProbeCriteria c;
  c.expected_fps = j.cfg.fps;
  c.expected_exposure_ns = j.cfg.exposure_ns;
  c.expected_gain = j.cfg.analog_gain;
  if (j.cfg.backend == rig::CameraBackendKind::kGst || j.cfg.backend == rig::CameraBackendKind::kMock) {
    c.expected_exposure_ns = 0;  // no per-frame readback on these backends
    c.expected_gain = 0;
  }
  c.require_focus_known = j.cfg.backend != rig::CameraBackendKind::kMock;
  j.report = an.finish(st, c);
  if (!final_status.ok()) j.report.checks.push_back({"status", false, final_status.to_string()});
  j.ok = true;
}

int focus_sweep(Job& j, const std::string& spec) {
  int lo = 0, hi = 0, step = 0;
  if (std::sscanf(spec.c_str(), "%d:%d:%d", &lo, &hi, &step) != 3 || step <= 0 || lo < 0 || hi > rig::kVcmDacMax || lo > hi) {
    std::fprintf(stderr, "--focus-sweep: expected lo:hi:step within 0..4095\n");
    return 2;
  }
  // Only a real camera has a lens to move (R1 F10): mock / replay / a gst source_override would only touch the
  // laptop's I2C buses.
  const bool real = j.cfg.backend == rig::CameraBackendKind::kArgus ||
                    (j.cfg.backend == rig::CameraBackendKind::kGst && j.cfg.gst.source_override.empty());
  if (!real) {
    std::fprintf(stderr, "--focus-sweep: camera %s is not a real camera (needs backend argus, or gst without "
                         "source_override)\n", j.cfg.id.c_str());
    return 2;
  }
  if (rig::Status s = j.cam->start(); !s.ok()) {
    std::fprintf(stderr, "%s: %s\n", j.cfg.id.c_str(), s.to_string().c_str());
    return 2;
  }
  std::printf("%s focus sweep (DAC code -> Laplacian variance of the centre crop):\n", j.cfg.id.c_str());
  int best = -1;
  double best_score = -1;
  int effective = 0;
  std::string last_detail;
  for (int pos = lo; pos <= hi; pos += step) {
    rig::FocusConfig fc = j.cfg.focus;
    fc.mode = rig::FocusMode::kFixed;
    fc.position = pos;
    const rig::FocusResult r = rig::Imx519Focus(fc).apply();
    const rig::TimeNs settle = rig::now_raw_ns() + 200'000'000;  // VCM settle
    double sum = 0;
    int n = 0;
    while (n < 5) {
      auto f = j.cam->read(1s);
      if (!f) break;
      if (f->t_sof_ns < settle) continue;
      const double sc = rig::frame_sharpness(*f);
      if (!std::isnan(sc)) sum += sc, ++n;
    }
    const double score = n > 0 ? sum / n : -1;
    const bool took_effect = r.status == rig::FocusStatus::kLockedVcmI2c || r.status == rig::FocusStatus::kLockedV4l2;
    std::printf("  %4d  %12.2f  (%s)\n", pos, score, rig::to_string(r.status));
    last_detail = r.detail;
    if (!took_effect) continue;  // a step that did not move the lens says nothing about this position
    ++effective;
    if (score > best_score) best_score = score, best = pos;
  }
  (void)j.cam->stop();
  if (effective == 0 || best < 0) {
    std::printf("focus sweep FAILED: no step moved the lens (%s)\n", last_detail.c_str());
    return 1;
  }
  std::printf("best position: %d of %d effective steps (set cameras.%s.driver.focus.position)\n", best, effective,
              j.cfg.id.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"ec-cam-probe: camera bring-up probe (modes, fps, jitter, drops, latency, locks, focus)"};
  std::string config_path, backend = "argus", sensor, gst_source, out_dir, json_path, sweep, id = "cam";
  std::vector<std::string> cameras;
  int sensor_id = 0, width = 0, height = 0, save_frames = 3, focus_position = -1;
  double fps = 0, seconds = 10, gain = 0;
  int64_t exposure_ns = 0;
  bool list = false, together = false;
  app.add_option("--config", config_path, "rig.yaml (cameras.<id>.driver is used)");
  app.add_option("--camera", cameras, "Camera id(s) from the config (default: all)");
  app.add_option("--backend", backend, "Ad-hoc camera without --config: argus | gst | mock")
      ->check(CLI::IsMember({"argus", "gst", "mock"}));
  app.add_option("--id", id, "Ad-hoc camera id");
  app.add_option("--sensor", sensor, "Ad-hoc: imx519 | imx219 | '' (mode table)");
  app.add_option("--sensor-id", sensor_id, "Ad-hoc: Argus sensor index / nvarguscamerasrc sensor-id");
  app.add_option("--width", width, "Ad-hoc capture width");
  app.add_option("--height", height, "Ad-hoc capture height");
  app.add_option("--fps", fps, "Ad-hoc frame rate");
  app.add_option("--exposure-ns", exposure_ns, "Ad-hoc locked exposure [ns] (0 = AE free)");
  app.add_option("--gain", gain, "Ad-hoc locked analog gain (0 = free)");
  app.add_option("--gst-source", gst_source, "Ad-hoc gst source_override (e.g. 'videotestsrc is-live=true')");
  app.add_flag("--list", list, "List sensor modes and exit");
  app.add_option("--seconds", seconds, "Capture duration per camera [s]")->check(CLI::PositiveNumber);
  app.add_flag("--together", together, "Capture all cameras at the same time (default: one after another)");
  app.add_option("--save-frames", save_frames, "Sample frames to save per camera")->check(CLI::NonNegativeNumber);
  app.add_option("--out", out_dir, "Output directory for sample frames (default ./ec-cam-probe-<unix time>)");
  app.add_option("--json", json_path, "Write the reports as JSON");
  app.add_option("--focus-position", focus_position, "Override focus.position (DAC 0..4095) for this run");
  app.add_option("--focus-sweep", sweep, "lo:hi:step DAC sweep with a sharpness score (single camera)");
  CLI11_PARSE(app, argc, argv);

  std::vector<Job> jobs;
  if (!config_path.empty()) {
    rig::Result<rig::RigConfig> rc = rig::load_rig_config(config_path);
    if (!rc.ok()) {
      std::fprintf(stderr, "%s\n", rc.status().to_string().c_str());
      return 2;
    }
    for (const auto& [cid, cam] : rc->cameras) {
      if (!cameras.empty() && std::find(cameras.begin(), cameras.end(), cid) == cameras.end()) continue;
      Job j;
      j.cfg = cam.driver;
      j.cfg.id = cid;
      jobs.push_back(std::move(j));
    }
    for (const std::string& c : cameras) {
      if (!rc->cameras.count(c)) {
        std::fprintf(stderr, "--camera %s: not in %s\n", c.c_str(), config_path.c_str());
        return 2;
      }
    }
  } else {
    Job j;
    j.cfg.id = id;
    j.cfg.backend = backend == "argus" ? rig::CameraBackendKind::kArgus
                    : backend == "gst" ? rig::CameraBackendKind::kGst
                                       : rig::CameraBackendKind::kMock;
    j.cfg.sensor = sensor;
    j.cfg.sensor_id = sensor_id;
    j.cfg.width = width;
    j.cfg.height = height;
    j.cfg.fps = fps;
    j.cfg.exposure_ns = exposure_ns;
    j.cfg.analog_gain = gain;
    j.cfg.gst.source_override = gst_source;
    jobs.push_back(std::move(j));
  }
  if (jobs.empty()) {
    std::fprintf(stderr, "no cameras selected\n");
    return 2;
  }
  for (Job& j : jobs) {
    if (focus_position >= 0) {
      j.cfg.focus.position = focus_position;
      if (j.cfg.focus.mode == rig::FocusMode::kNone) j.cfg.focus.mode = rig::FocusMode::kFixed;
    }
    auto c = rig::make_camera(j.cfg);
    if (!c.ok()) {
      std::fprintf(stderr, "%s\n", c.status().to_string().c_str());
      return 2;
    }
    j.cam = std::move(c).value();
  }

  if (list) {
    for (Job& j : jobs) {
      auto m = j.cam->modes();
      std::printf("camera %s (%s, sensor '%s', sensor_id %d):\n", j.cfg.id.c_str(),
                  j.cfg.backend == rig::CameraBackendKind::kArgus ? "argus: driver-reported" : "published table",
                  j.cfg.sensor.c_str(), j.cfg.sensor_id);
      if (!m.ok()) {
        std::printf("  error: %s\n", m.status().to_string().c_str());
        continue;
      }
      for (const auto& mode : *m) std::printf("  %s\n", rig::format_sensor_mode(mode).c_str());
      auto sel = rig::select_sensor_mode(*m, j.cfg);
      std::printf("  configured %dx%d@%g -> %s\n", j.cfg.width, j.cfg.height, j.cfg.fps,
                  sel.ok() ? rig::format_sensor_mode(*sel).c_str() : sel.status().message().c_str());
    }
    return 0;
  }
  if (!sweep.empty()) {
    if (jobs.size() != 1) {
      std::fprintf(stderr, "--focus-sweep needs exactly one --camera\n");
      return 2;
    }
    return focus_sweep(jobs[0], sweep);
  }

  const fs::path out = out_dir.empty() ? fs::path("ec-cam-probe-" + std::to_string(rig::now_unix_ns() / 1000000000))
                                       : fs::path(out_dir);
  std::error_code ec;
  if (save_frames > 0) fs::create_directories(out, ec);
  if (together) {
    std::vector<std::thread> threads;
    for (Job& j : jobs) threads.emplace_back([&j, seconds, save_frames, &out] { capture(j, seconds, save_frames, out); });
    for (auto& t : threads) t.join();
  } else {
    for (Job& j : jobs) capture(j, seconds, save_frames, out);
  }
  bool all_pass = true;
  nlohmann::json js = nlohmann::json::array();
  for (Job& j : jobs) {
    if (!j.ok) {
      std::printf("camera %s: FAIL (start): %s\n", j.cfg.id.c_str(), j.error.c_str());
      all_pass = false;
      js.push_back({{"camera", j.cfg.id}, {"pass", false}, {"error", j.error}});
      continue;
    }
    std::printf("%s", rig::format_probe_report(j.report).c_str());
    all_pass = all_pass && j.report.pass();
    js.push_back(report_json(j.report));
  }
  if (!json_path.empty()) {
    std::ofstream f(json_path);
    f << js.dump(2) << "\n";
  }
  std::printf("ec-cam-probe: %s\n", all_pass ? "ALL PASS" : "FAIL");
  return all_pass ? 0 : 1;
}
