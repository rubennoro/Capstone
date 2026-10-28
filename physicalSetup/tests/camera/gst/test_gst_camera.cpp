// SPDX-License-Identifier: MIT
// GstCamera through the real appsink code path, with videotestsrc substituted via gst.source_override
// (PLAN T2: "the GStreamer path runs on the laptop").
#include <gtest/gtest.h>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <sys/wait.h>

#include <atomic>
#include <cstdlib>
#include <thread>

#include "rig/drivers/camera_backend.hpp"
#include "rig/time/clock.hpp"

// LeakSanitizer: libglib-2.0 allocates 16 KiB in a library constructor run by the dynamic loader at load time
// (stack: g_malloc <- libglib <- ld-linux; process-lifetime state, never freed by design). Suppress only leaks whose
// stack goes through the loader, so leaks from our own g_* calls (stacks through rigcommon) still fail the test.
// The hook is a weak symbol read by the sanitizer runtime; harmless in non-ASan builds.
// The default fast unwinder stops at g_malloc (glib has no frame pointers), so full (slow) unwinding is enabled for
// this binary; ASAN_OPTIONS from check.sh does not set fast_unwind_on_malloc and therefore keeps this default.
extern "C" const char* __lsan_default_suppressions() { return "leak:ld-linux\n"; }
extern "C" const char* __asan_default_options() { return "fast_unwind_on_malloc=0"; }
// ThreadSanitizer (R1 F11): GLib/GStreamer are not instrumented, so their atomics and lock-free queues (GstAtomicQueue
// in GstBus, GstMiniObject refcounts) are invisible and TSan reports their own malloc/free pairs across threads as
// races. `called_from_lib:` makes TSan ignore only the interceptors (malloc/free/...) invoked from inside these
// libraries; every access made by our instrumented code (GstCamera::handle_sample on the streaming thread, the bus
// thread, read()) is still checked. GstTsan.InjectedRaceInAppsinkCallbackIsReported proves that.
// One template per module: TSan requires every called_from_lib template to match exactly one library (it aborts on
// an ambiguous one), and a GMutex locked in libglib but unlocked from a plugin must be ignored on both sides, or TSan
// reports "unlock of an unlocked mutex". The list = the libraries the test binary links (ldd / LD_DEBUG=files)
// plus the plugins these pipelines load (coreelements for capsfilter, videotestsrc, videoconvert, videoscale, app).
extern "C" const char* __tsan_default_suppressions() {
  return "called_from_lib:libglib-2.0.so\n"
         "called_from_lib:libgobject-2.0.so\n"
         "called_from_lib:libgmodule-2.0.so\n"
         "called_from_lib:libgstreamer-1.0.so\n"
         "called_from_lib:libgstbase-1.0.so\n"
         "called_from_lib:libgstapp-1.0.so\n"
         "called_from_lib:libgstvideo-1.0.so\n"
         "called_from_lib:libgstaudio-1.0.so\n"
         "called_from_lib:libgsttag-1.0.so\n"
         "called_from_lib:libgstpbutils-1.0.so\n"
         "called_from_lib:libgstriff-1.0.so\n"
         "called_from_lib:liborc-0.4.so\n"
         "called_from_lib:libgstcoreelements.so\n"
         "called_from_lib:libgstvideotestsrc.so\n"
         "called_from_lib:libgstvideoconvert.so\n"
         "called_from_lib:libgstvideoscale.so\n"
         "called_from_lib:/libgstapp.so\n";
}

namespace rig {
namespace {

using namespace std::chrono_literals;

CameraConfig gst_cfg(const std::string& src, PixelFormat fmt = PixelFormat::kNV12, double fps = 60.0) {
  CameraConfig c;
  c.id = "g";
  c.backend = CameraBackendKind::kGst;
  c.width = 320;
  c.height = 240;
  c.fps = fps;
  c.pixel_format = fmt;
  c.gst.source_override = src;
  return c;
}

TEST(GstCamera, VideotestsrcNv12) {
  auto cam = make_camera(gst_cfg("videotestsrc is-live=true pattern=white")).value();
  const Status s = cam->start();
  ASSERT_TRUE(s.ok()) << s.to_string();
  std::vector<ImageFrame> fr;
  for (int i = 0; i < 30; ++i) {
    auto f = cam->read(2s);
    ASSERT_TRUE(f) << cam->status().to_string();
    fr.push_back(*f);
  }
  for (std::size_t i = 0; i < fr.size(); ++i) {
    const ImageFrame& f = fr[i];
    EXPECT_TRUE(f.valid());
    EXPECT_EQ(f.fmt, PixelFormat::kNV12);
    EXPECT_EQ(f.width, 320);
    EXPECT_EQ(f.size, 320u * 240u * 3u / 2u);
    EXPECT_EQ(f.seq, i);  // GST_BUFFER_OFFSET from videotestsrc, rebased after the settle frames
    EXPECT_EQ(f.t_mid_ns, f.t_sof_ns);
    // PTS (MONOTONIC) -> RAW conversion: the frame time is just before its arrival.
    const int64_t lat = f.t_arrival_ns - f.t_sof_ns;
    EXPECT_GT(lat, -2'000'000) << i;
    EXPECT_LT(lat, 100'000'000) << i;
    if (i > 0) {
      EXPECT_NEAR(static_cast<double>(f.t_sof_ns - fr[i - 1].t_sof_ns), 1e9 / 60, 0.5e6);
    }
    // videotestsrc ! videoconvert NV12 is BT.601 video range by default: white = Y 235, stated in the caps.
    EXPECT_EQ(f.meta.color_range, ColorRange::kLimited);
    EXPECT_GE(f.plane_data(0)[160 * 320 + 160], 230);  // white -> Y 235
    EXPECT_NEAR(f.plane_data(1)[0], 128, 2);
  }
  const CameraStats st = cam->stats();
  EXPECT_EQ(st.frames, 30u);
  EXPECT_EQ(st.drops_seq_gap, 0u);
  EXPECT_EQ(st.time_source, "gst:pts(monotonic->raw)");
  EXPECT_NEAR(st.fps_estimate, 60.0, 1.0);
  EXPECT_EQ(st.latency_ns.count, 30u);
  EXPECT_TRUE(cam->stop().ok());
  EXPECT_TRUE(cam->stop().ok());
  EXPECT_FALSE(cam->read(10ms).has_value());
  // Restartable.
  ASSERT_TRUE(cam->start().ok());
  auto f = cam->read(2s);
  ASSERT_TRUE(f);
  EXPECT_EQ(f->seq, 0u);
  EXPECT_TRUE(cam->stop().ok());
}

TEST(GstCamera, Gray8AndBgr) {
  for (PixelFormat fmt : {PixelFormat::kGRAY8, PixelFormat::kBGR8}) {
    auto cam = make_camera(gst_cfg("videotestsrc is-live=true pattern=black", fmt, 30.0)).value();
    ASSERT_TRUE(cam->start().ok());
    auto f = cam->read(2s);
    ASSERT_TRUE(f);
    EXPECT_EQ(f->fmt, fmt);
    EXPECT_EQ(f->planes[0].stride, fmt == PixelFormat::kGRAY8 ? 320u : 960u);
    EXPECT_LE(f->plane_data(0)[1000], 20);
    EXPECT_TRUE(cam->stop().ok());
  }
}

TEST(GstCamera, StopUnblocksRead) {
  auto cam = make_camera(gst_cfg("videotestsrc is-live=true", PixelFormat::kNV12, 30.0)).value();
  ASSERT_TRUE(cam->start().ok());
  std::atomic<int> got{0};
  std::thread reader([&] {
    while (cam->read(5s)) ++got;
  });
  std::this_thread::sleep_for(200ms);
  const TimeNs t0 = now_raw_ns();
  ASSERT_TRUE(cam->stop().ok());
  reader.join();
  EXPECT_LT(now_raw_ns() - t0, 1'000'000'000);
  EXPECT_GT(got.load(), 2);
}

TEST(GstCamera, SlowConsumerDropsOldestAndCounts) {
  CameraConfig c = gst_cfg("videotestsrc is-live=true", PixelFormat::kNV12, 100.0);
  c.buffer_count = 2;
  auto cam = make_camera(c).value();
  ASSERT_TRUE(cam->start().ok());
  std::this_thread::sleep_for(300ms);  // ~30 frames arrive, queue holds 2
  auto f = cam->read(1s);
  ASSERT_TRUE(f);
  const CameraStats s = cam->stats();
  EXPECT_GT(s.drops_backend_queue, 10u);
  EXPECT_GT(f->seq, 10u);  // the queue dropped the oldest: the delivered frame is recent, seq shows the gap
  EXPECT_TRUE(cam->stop().ok());
}

TEST(GstCamera, Errors) {
  auto bad = make_camera(gst_cfg("nosuchelement_xyz")).value();
  const Status s = bad->start();
  EXPECT_EQ(s.code(), StatusCode::kInvalidArgument);
  EXPECT_NE(s.message().find("cannot parse pipeline"), std::string::npos) << s.message();

  CameraConfig j = gst_cfg("videotestsrc", PixelFormat::kJPEG);
  EXPECT_EQ(make_camera(j).value()->start().code(), StatusCode::kInvalidArgument);

  // A source that ends: EOS becomes a fatal status and read() returns nullopt.
  auto eos = make_camera(gst_cfg("videotestsrc is-live=true num-buffers=8", PixelFormat::kNV12, 60.0)).value();
  ASSERT_TRUE(eos->start().ok());
  int n = 0;
  while (eos->read(2s)) ++n;
  EXPECT_LE(n, 8);
  EXPECT_EQ(eos->status().code(), StatusCode::kDataLoss);
  EXPECT_NE(eos->status().message().find("end of stream"), std::string::npos);
  EXPECT_TRUE(eos->stop().ok());
}

#if defined(__SANITIZE_THREAD__)
// The injected race: written on the GStreamer streaming thread inside an appsink callback (the same position as
// GstCamera::handle_sample, whose stack also runs through libgstapp/libgstbase), read on the test thread without
// synchronization. With the old `race:libgst*` suppressions this report was hidden; it must now abort the child.
int g_injected_race = 0;

GstFlowReturn racy_new_sample(GstAppSink* sink, gpointer) {
  GstSample* s = gst_app_sink_pull_sample(sink);
  if (s != nullptr) gst_sample_unref(s);
  ++g_injected_race;  // unsynchronized write (deliberate)
  return GST_FLOW_OK;
}

void run_injected_race() {
  gst_init(nullptr, nullptr);
  GError* err = nullptr;
  GstElement* p = gst_parse_launch(
      "videotestsrc is-live=true num-buffers=30 ! video/x-raw,width=64,height=48,framerate=100/1 ! "
      "appsink name=s sync=false emit-signals=false", &err);
  if (p == nullptr) std::exit(3);
  GstElement* sink = gst_bin_get_by_name(GST_BIN(p), "s");
  GstAppSinkCallbacks cb{};
  cb.new_sample = &racy_new_sample;
  gst_app_sink_set_callbacks(GST_APP_SINK(sink), &cb, nullptr, nullptr);
  gst_element_set_state(p, GST_STATE_PLAYING);
  int seen = 0;
  for (int i = 0; i < 300; ++i) {
    seen += g_injected_race;  // unsynchronized read (deliberate)
    std::this_thread::sleep_for(1ms);
  }
  gst_element_set_state(p, GST_STATE_NULL);
  std::exit(seen >= 0 ? 0 : 1);  // reached only if TSan did not halt
}

TEST(GstTsan, InjectedRaceInAppsinkCallbackIsReported) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  // TSan exits non-zero (halt_on_error=1 from check.sh, else its default exitcode 66 at exit).
  EXPECT_EXIT(run_injected_race(), [](int st) { return !(WIFEXITED(st) && WEXITSTATUS(st) == 0); },
              "ThreadSanitizer: data race");
}
#endif

}  // namespace
}  // namespace rig
