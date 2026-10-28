// SPDX-License-Identifier: MIT
// ec-live (L1): live view of both cameras plus the IMU, for hardware bring-up.
//
//   ec-live --config rig.yaml                       # all sensors in the config
//   ec-live --config rig.yaml --mock                # laptop: mock cameras + simulated IMU
//   ec-live --config rig.yaml --no-cameras          # IMU only
//   ec-live --config rig.yaml --camera uw --no-imu  # one camera
//   ec-live --config rig.yaml --mock --headless-snapshot out.png --seconds 3   # tests
//
// Each sensor runs in its own CaptureService (T3), so one that fails to start does not stop the others. ec-live
// subscribes to the hubs with kDropOk only (it can never cause a lossless overflow or backpressure); the main
// thread drains them at --display-hz, keeps the newest frame of each camera, converts it with
// imageprep::frame_to_bgr (honours FrameMeta::color_range), and draws the mosaic + overlays + IMU strip.
// A status line goes to stdout once per second, so it is usable over SSH without a display.
//
// Keys: q / Esc quit, s save a PNG snapshot. SIGINT / SIGTERM stop cleanly (a second one exits at once).
// Exit codes: 0 ok; 1 --strict and a sensor failed; 2 setup error (config, CLI, nothing started, snapshot write).
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <CLI11.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include "live_view.hpp"
#include "rig/capture/capture_service.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/time/clock.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;
using eclive::CameraView;
using eclive::ImuView;
using rig::TimeNs;

// Sanitizer builds only (weak hooks, no effect otherwise): libglib allocates process-lifetime state in a library
// constructor run by the loader (same as ec-cam-probe; tests/camera/gst/test_gst_camera.cpp).
// libtbb: the system OpenCV's thread pool leaves worker allocations at exit (tests/imageprep/golden_util.cpp).
// GTK/GLib/fontconfig: process-lifetime state of the highgui window. TSan: cv::imwrite registers codecs through
// libgdal, whose internal mutexes report a lock-order inversion (third party; golden_util.cpp).
extern "C" const char* __lsan_default_suppressions() {
  return "leak:ld-linux\nleak:libtbb.so\nleak:libgtk\nleak:libgdk\nleak:libglib\nleak:libgobject\nleak:libfontconfig\n";
}
extern "C" const char* __asan_default_options() { return "fast_unwind_on_malloc=0"; }
extern "C" const char* __tsan_default_suppressions() { return "deadlock:libgdal.so\n"; }

namespace {

constexpr const char* kWindow = "ec-live";
constexpr const char* kSubscriber = "ec_live";

std::atomic<int> g_signals{0};
static_assert(std::atomic<int>::is_always_lock_free, "signal flag must be lock-free");

extern "C" void on_signal(int) {
  if (g_signals.fetch_add(1) >= 1) {
    const char msg[] = "\nec-live: second signal, exiting immediately\n";
    ssize_t r = write(STDERR_FILENO, msg, sizeof(msg) - 1);
    (void)r;
    _exit(130);
  }
}

void install_signals() {
  struct sigaction sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
}

bool stop_requested() { return g_signals.load() > 0; }

/// Can we open a highgui window? Checks $DISPLAY, then tries in a forked child, because some GTK builds call
/// exit() instead of failing when the display cannot be opened. Called before any sensor thread exists.
bool display_usable(std::string* why) {
  const char* d = std::getenv("DISPLAY");
  const char* w = std::getenv("WAYLAND_DISPLAY");
  if ((d == nullptr || *d == '\0') && (w == nullptr || *w == '\0')) {
    *why = "$DISPLAY is not set (SSH session?). Run ec-live in a terminal on the Jetson's desktop, or "
           "`export DISPLAY=:0` if that desktop is logged in as you";
    return false;
  }
  std::fflush(stdout);
  std::fflush(stderr);
  const pid_t pid = fork();
  if (pid < 0) return true;  // cannot probe; try in-process (guarded by try/catch)
  if (pid == 0) {
    int rc = 0;
    try {
      cv::namedWindow("ec-live-probe", cv::WINDOW_AUTOSIZE);
      cv::waitKey(1);
      cv::destroyAllWindows();
      cv::waitKey(1);
    } catch (...) {
      rc = 3;
    }
    _exit(rc);
  }
  int status = 0;
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (true) {
    const pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) break;
    if (r < 0) return true;
    if (std::chrono::steady_clock::now() > deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      *why = std::string("opening a window on DISPLAY=") + (d ? d : "") + " timed out";
      return false;
    }
    std::this_thread::sleep_for(20ms);
  }
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return true;
  *why = std::string("cannot open a window on DISPLAY=") + (d ? d : "") +
         (WIFEXITED(status) && WEXITSTATUS(status) == 3 ? " (OpenCV has no GUI backend)"
                                                        : " (display refused; is the desktop session yours? "
                                                          "try `xhost +local:` on it)");
  return false;
}

std::string timestamp_name() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
  return buf;
}

struct CamSensor {
  std::string id;
  std::unique_ptr<rig::CaptureService> svc;
  std::shared_ptr<rig::StreamSubscription<rig::ImageFrame>> sub;
  rig::StreamHub<rig::ImageFrame>* hub = nullptr;
  eclive::RateMeter rate;
  bool runtime_error_reported = false;
};

struct ImuSensor {
  std::unique_ptr<rig::CaptureService> svc;
  std::shared_ptr<rig::StreamSubscription<rig::ImuSample>> sub;
  rig::StreamHub<rig::ImuSample>* hub = nullptr;
  eclive::RateMeter rate;
  bool runtime_error_reported = false;
};

void report_failure(const std::string& what, bool is_camera, const std::string& backend, const rig::Status& s) {
  std::printf("[FAIL] %s: %s\n", what.c_str(), s.to_string().c_str());
  const std::string hint = eclive::failure_hint(is_camera, backend, s);
  if (!hint.empty()) std::printf("       hint: %s\n", hint.c_str());
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"ec-live: live view of the rig cameras + IMU (bring-up)"};
  std::string config_path, out_dir = ".", snapshot_path;
  std::vector<std::string> camera_ids;
  bool mock = false, no_imu = false, no_cameras = false, no_window = false, strict = false;
  double display_hz = 15.0, seconds = 0.0, stall_s = 1.0;
  int width = 1600;
  app.add_option("--config", config_path, "rig.yaml")->required()->check(CLI::ExistingFile);
  app.add_option("--camera", camera_ids, "Camera id(s) to show (default: all in the config)");
  app.add_flag("--no-imu", no_imu, "Do not start the IMU");
  app.add_flag("--no-cameras", no_cameras, "Do not start any camera (IMU only)");
  app.add_flag("--mock", mock, "Mock cameras + simulated IMU (sizes and fps from the config; laptop)");
  app.add_option("--width", width, "Mosaic width [px]")->check(CLI::Range(320, 7680));
  app.add_option("--display-hz", display_hz, "Display refresh [Hz]")->check(CLI::Range(1.0, 60.0));
  app.add_option("--stall-s", stall_s, "Show NO FRAMES after this long without a frame [s]")
      ->check(CLI::Range(0.1, 60.0));
  app.add_flag("--no-window", no_window, "Terminal status only (no window)");
  app.add_flag("--strict", strict, "Exit 1 if any sensor fails (default: continue with the rest)");
  app.add_option("--out", out_dir, "Directory for 's' snapshots (default: current dir)");
  app.add_option("--seconds", seconds, "Stop after this long [s] (0 = until q / Ctrl-C)")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--headless-snapshot", snapshot_path,
                 "Run --seconds (default 3) without a window, write the final mosaic PNG to PATH, exit");
  CLI11_PARSE(app, argc, argv);
  if (no_cameras && !camera_ids.empty()) {
    std::fprintf(stderr, "ec-live: --no-cameras and --camera exclude each other\n");
    return 2;
  }
  if (no_cameras && no_imu) {
    std::fprintf(stderr, "ec-live: --no-cameras and --no-imu leave nothing to show\n");
    return 2;
  }
  const bool headless = !snapshot_path.empty();
  if (headless && seconds <= 0) seconds = 3.0;

  rig::Result<rig::RigConfig> rc = rig::load_rig_config(config_path);
  if (!rc.ok()) {
    std::fprintf(stderr, "ec-live: %s\n", rc.status().to_string().c_str());
    return 2;
  }
  rig::RigConfig cfg = std::move(rc).value();
  if (mock) eclive::apply_mock_overrides(cfg);

  std::vector<std::string> order;
  if (!no_cameras) {
    for (const std::string& id : camera_ids)
      if (!cfg.cameras.count(id)) {
        std::fprintf(stderr, "ec-live: --camera %s: not in %s (have:", id.c_str(), config_path.c_str());
        for (const auto& kv : cfg.cameras) std::fprintf(stderr, " %s", kv.first.c_str());
        std::fprintf(stderr, ")\n");
        return 2;
      }
    for (const std::string& id : eclive::display_order(cfg))
      if (camera_ids.empty() || std::find(camera_ids.begin(), camera_ids.end(), id) != camera_ids.end())
        order.push_back(id);
  }

  // Window decision first: the display probe forks, so it must run before any sensor thread exists.
  bool window = false;
  if (!no_window && !headless) {
    std::string why;
    window = display_usable(&why);
    if (!window) std::printf("ec-live: no display: %s.\nec-live: continuing in terminal-only mode (status line "
                             "once per second; Ctrl-C to quit).\n", why.c_str());
  }
  install_signals();
  cv::setNumThreads(0);  // sequential: conversions stay on this thread; the capture cores stay free

  std::printf("ec-live: %s (%s)%s, cameras: %s, imu: %s\n", config_path.c_str(), cfg.rig_name.c_str(),
              mock ? " --mock" : "", order.empty() ? "none" : [&] {
                std::string s;
                for (const auto& id : order) s += (s.empty() ? "" : ", ") + id;
                return s;
              }().c_str(), no_imu ? "off" : cfg.imu.driver.id.c_str());
  std::fflush(stdout);

  // ---- Start each sensor in its own CaptureService (IMU first: the least risky).
  std::vector<std::string> failed;
  ImuView imu_view;
  ImuSensor imu;
  imu_view.enabled = !no_imu;
  imu_view.id = cfg.imu.driver.id;
  imu_view.backend = eclive::imu_backend_name(cfg.imu.driver.backend);
  imu_view.odr_hz = cfg.imu.driver.odr_hz;
  if (!no_imu) {
    const std::string what = "imu " + imu_view.id + " (" + imu_view.backend + ")";
    rig::Status st;
    auto be = rig::make_imu(cfg.imu.driver);
    if (!be.ok()) {
      st = be.status();
    } else {
      rig::CaptureOptions o;
      o.capture = cfg.capture;
      o.imu_thread = cfg.imu.thread;
      auto svc = rig::CaptureService::create(std::move(o), {}, std::move(be).value());
      if (!svc.ok()) {
        st = svc.status();
      } else {
        imu.svc = std::move(svc).value();
        imu.hub = imu.svc->imu_hub();
        // DropOk, ~10 s at 833 Hz: the render loop drains it every tick.
        auto sub = imu.hub->subscribe({kSubscriber, rig::DeliveryPolicy::kDropOk, 8192});
        if (!sub.ok()) {
          st = sub.status();
        } else {
          imu.sub = std::move(sub).value();
          st = imu.svc->start();
        }
      }
    }
    if (st.ok()) {
      imu_view.started = true;
      imu_view.started_ns = rig::now_raw_ns();
      std::printf("[ OK ] %s started, ODR %g Hz\n", what.c_str(), imu_view.odr_hz);
    } else {
      imu_view.error = st.to_string();
      report_failure(what, false, imu_view.backend, st);
      failed.push_back(what);
      imu.sub.reset();
      imu.svc.reset();
      imu.hub = nullptr;
    }
  }

  std::vector<CameraView> cams(order.size());
  std::vector<CamSensor> cam_sensors(order.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    const rig::RigCameraConfig& rcam = cfg.cameras.at(order[i]);
    CameraView& v = cams[i];
    CamSensor& s = cam_sensors[i];
    v.id = s.id = order[i];
    v.role = rcam.role;
    v.backend = eclive::camera_backend_name(rcam.driver.backend);
    v.width = rcam.driver.width;
    v.height = rcam.driver.height;
    v.fps = rcam.driver.fps;
    const std::string what = "camera " + v.id + " (" + v.backend + " " + std::to_string(v.width) + "x" +
                             std::to_string(v.height) + "@" + std::to_string(static_cast<int>(v.fps)) + ")";
    rig::CameraConfig cc = rcam.driver;
    cc.id = v.id;
    rig::Status st;
    auto be = rig::make_camera(cc);
    if (!be.ok()) {
      st = be.status();
    } else {
      rig::CaptureOptions o;
      o.capture = cfg.capture;
      o.camera_threads[v.id] = rcam.thread;
      std::vector<std::unique_ptr<rig::CameraBackend>> list;
      list.push_back(std::move(be).value());
      auto svc = rig::CaptureService::create(std::move(o), std::move(list), nullptr);
      if (!svc.ok()) {
        st = svc.status();
      } else {
        s.svc = std::move(svc).value();
        s.hub = s.svc->camera_hub(v.id);
        auto sub = s.hub->subscribe({kSubscriber, rig::DeliveryPolicy::kDropOk, 2});
        if (!sub.ok()) {
          st = sub.status();
        } else {
          s.sub = std::move(sub).value();
          st = s.svc->start();
        }
      }
    }
    if (st.ok()) {
      v.started = true;
      v.started_ns = rig::now_raw_ns();
      std::printf("[ OK ] %s started\n", what.c_str());
    } else {
      v.error = st.to_string();
      report_failure(what, true, v.backend, st);
      failed.push_back(what);
      s.sub.reset();
      s.svc.reset();
      s.hub = nullptr;
    }
    std::fflush(stdout);
  }

  auto stop_all = [&] {
    for (auto& s : cam_sensors)
      if (s.svc) {
        rig::Status st = s.svc->stop();
        if (!st.ok()) std::printf("ec-live: camera %s stop: %s\n", s.id.c_str(), st.to_string().c_str());
      }
    if (imu.svc) {
      rig::Status st = imu.svc->stop();
      if (!st.ok()) std::printf("ec-live: imu stop: %s\n", st.to_string().c_str());
    }
    for (auto& s : cam_sensors) s.sub.reset(), s.svc.reset();
    imu.sub.reset();
    imu.svc.reset();
  };

  const int started_count = static_cast<int>(std::count_if(cams.begin(), cams.end(), [](const CameraView& c) {
                              return c.started;
                            })) + (imu_view.started ? 1 : 0);
  if (!failed.empty()) {
    std::printf("ec-live: %zu sensor(s) failed to start:", failed.size());
    for (const auto& f : failed) std::printf(" [%s]", f.c_str());
    std::printf("\n");
    if (strict) {
      std::printf("ec-live: --strict: stopping.\n");
      stop_all();
      return 1;
    }
    if (started_count == 0) {
      std::printf("ec-live: nothing started; exiting.\n");
      return 2;
    }
    std::printf("ec-live: continuing with the sensors that started (use --strict to exit instead).\n");
  }
  std::fflush(stdout);

  // ---- Render loop.
  eclive::RenderOptions ropt;
  ropt.width = width;
  ropt.stall_s = stall_s;
  std::vector<cv::Size> sizes;
  for (const auto& c : cams) sizes.emplace_back(c.width, c.height);
  const eclive::Layout layout = eclive::compute_layout(ropt, sizes, !no_imu);
  const bool need_images = window || headless;
  eclive::ImuHistory hist;

  if (window) {
    try {
      cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
      std::printf("ec-live: window %dx%d. Keys: q / Esc quit, s snapshot PNG to %s\n", layout.size.width,
                  layout.size.height, out_dir.c_str());
    } catch (const cv::Exception& e) {
      std::printf("ec-live: cannot open the window (%s); continuing in terminal-only mode.\n", e.what());
      window = false;
    }
  }
  std::fflush(stdout);

  const TimeNs t0 = rig::now_raw_ns();
  const TimeNs period = static_cast<TimeNs>(1e9 / display_hz);
  TimeNs next_status = t0 + eclive::kSecondNs;
  bool window_seen = false;
  bool runtime_error = false;
  int snapshots = 0;
  std::vector<rig::ImuSample> batch;
  cv::Mat mosaic;
  double work_ms_max = 0;  // drain + convert + render per tick, max over the last status second

  auto save_snapshot = [&](const cv::Mat& img) {
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    const fs::path p = fs::path(out_dir) / ("ec-live-" + timestamp_name() + "-" + std::to_string(++snapshots) + ".png");
    bool ok = false;
    try {
      ok = cv::imwrite(p.string(), img);
    } catch (const cv::Exception&) {
      ok = false;
    }
    std::printf("ec-live: snapshot %s %s\n", p.string().c_str(), ok ? "saved" : "FAILED to write");
    std::fflush(stdout);
  };

  while (!stop_requested()) {
    const TimeNs tick = rig::now_raw_ns();
    for (std::size_t i = 0; i < cams.size(); ++i) {
      CameraView& v = cams[i];
      CamSensor& s = cam_sensors[i];
      if (!s.svc) continue;
      rig::ImageFrame f;
      bool got = false;
      while (s.sub->try_pop(f)) got = true;  // keep the newest
      const TimeNs now = rig::now_raw_ns();
      if (got) {
        v.has_frame = true;
        v.seq = f.seq;
        v.t_mid_ns = f.t_mid_ns;
        v.meta = f.meta;
        v.last_rx_ns = now;
        if (need_images) {
          const cv::Size target = i < layout.images.size() ? layout.images[i].size() : cv::Size();
          auto img = eclive::frame_to_display(f, target);
          if (img.ok()) {
            v.image = std::move(img).value();
            v.convert_error.clear();
          } else {
            v.convert_error = img.status().message();
          }
        }
        f = rig::ImageFrame{};  // release the buffer now
      }
      v.frames_seen = s.hub->published();
      s.rate.add(now, v.frames_seen);
      v.measured_fps = s.rate.rate_hz();
      if (auto st = s.svc->camera_stats(v.id); st.ok()) {
        v.stats = *st;
        v.has_stats = true;
      }
      if (rig::Status h = s.svc->health(); !h.ok()) {
        v.error = h.to_string();
        runtime_error = true;
        if (!s.runtime_error_reported) {
          s.runtime_error_reported = true;
          std::printf("ec-live: camera %s ERROR: %s\n", v.id.c_str(), v.error.c_str());
        }
      }
    }
    if (imu.svc) {
      batch.clear();
      imu.sub->pop_all(batch, std::chrono::nanoseconds(0));
      for (const auto& smp : batch) hist.add(smp);
      const TimeNs now = rig::now_raw_ns();
      if (!batch.empty()) imu_view.last_rx_ns = now;
      hist.prune(now);
      imu.rate.add(now, imu.hub->published());
      imu_view.measured_rate_hz = imu.rate.rate_hz();
      if (auto st = imu.svc->imu_stats(); st.ok()) {
        imu_view.stats = *st;
        imu_view.has_stats = true;
      }
      if (rig::Status h = imu.svc->health(); !h.ok()) {
        imu_view.error = h.to_string();
        runtime_error = true;
        if (!imu.runtime_error_reported) {
          imu.runtime_error_reported = true;
          std::printf("ec-live: imu ERROR: %s\n", imu_view.error.c_str());
        }
      }
    }

    const TimeNs now = rig::now_raw_ns();
    if (now >= next_status) {
      std::string line = eclive::status_line(cams, imu_view, hist, now, t0, stall_s);
      if (need_images) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), " | ui %.0fms", work_ms_max);
        line += buf;
      }
      work_ms_max = 0;
      std::printf("%s\n", line.c_str());
      std::fflush(stdout);
      next_status += eclive::kSecondNs;
      if (next_status < now) next_status = now + eclive::kSecondNs;
    }
    if (seconds > 0 && static_cast<double>(now - t0) / 1e9 >= seconds) break;

    if (window) {
      mosaic = eclive::render_mosaic(layout, cams, imu_view, hist, ropt, now);
      int key = -1;
      try {
        cv::imshow(kWindow, mosaic);
        work_ms_max = std::max(work_ms_max, static_cast<double>(rig::now_raw_ns() - tick) / 1e6);
        const TimeNs left = tick + period - rig::now_raw_ns();
        key = cv::waitKey(std::max(1, static_cast<int>(left / 1000000)));
        const double vis = cv::getWindowProperty(kWindow, cv::WND_PROP_VISIBLE);
        if (vis >= 1) window_seen = true;
        if (window_seen && vis < 1) {
          std::printf("ec-live: window closed\n");
          break;
        }
      } catch (const cv::Exception& e) {
        std::printf("ec-live: window error (%s); continuing in terminal-only mode.\n", e.what());
        window = false;
      }
      key &= 0xFF;
      if (key == 'q' || key == 'Q' || key == 27) break;
      if (key == 's' || key == 'S') save_snapshot(mosaic);
    } else {
      work_ms_max = std::max(work_ms_max, static_cast<double>(rig::now_raw_ns() - tick) / 1e6);
      const TimeNs left = tick + period - rig::now_raw_ns();
      if (left > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(left));
    }
  }
  if (stop_requested()) std::printf("ec-live: signal received, stopping\n");

  int rc_exit = 0;
  if (headless) {
    const TimeNs now = rig::now_raw_ns();
    mosaic = eclive::render_mosaic(layout, cams, imu_view, hist, ropt, now);
    bool ok = false;
    try {
      const fs::path p(snapshot_path);
      if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
      }
      ok = cv::imwrite(snapshot_path, mosaic);
    } catch (const cv::Exception& e) {
      std::printf("ec-live: %s\n", e.what());
    }
    std::printf("ec-live: headless snapshot %s (%dx%d) %s\n", snapshot_path.c_str(), mosaic.cols, mosaic.rows,
                ok ? "written" : "FAILED to write");
    if (!ok) rc_exit = 2;
  }
  if (window) {
    try {
      cv::destroyAllWindows();
      cv::waitKey(1);
    } catch (const cv::Exception&) {
    }
  }
  stop_all();
  std::printf("ec-live: stopped cleanly\n");
  if (rc_exit == 0 && strict && runtime_error) rc_exit = 1;
  return rc_exit;
}
