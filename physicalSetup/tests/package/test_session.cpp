// SPDX-License-Identifier: MIT
// T3: Session / Package directories, discard, validate_layout.
#include <gtest/gtest.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>

#include "rig/config/rig_config.hpp"
#include "rig/package/atomic_file.hpp"
#include "rig/package/csv.hpp"
#include "rig/package/layout.hpp"
#include "rig/package/package.hpp"
#include "rig/time/clock.hpp"

namespace fs = std::filesystem;

namespace rig {
namespace {

const std::string kExample =
    std::string(RIG_SOURCE_DIR) + "/episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml";

fs::path fresh_root(const std::string& name) {
  const fs::path d = fs::path(RIG_TEST_TMP_ROOT) / ("sess_" + name + "_" + std::to_string(getpid()));
  fs::remove_all(d);
  return d;
}

SessionInfo info(const std::string& session) {
  SessionInfo i;
  i.session = session;
  i.task = "block_raise";
  i.operator_name = "ruben";
  return i;
}

void write_text(const fs::path& p, const std::string& s) { std::ofstream(p) << s; }

/// Fills a package with the complete recorded layout (what T5's RawWriter produces).
void fill_recorded_layout(Package& p) {
  write_text(p.events_path(), std::string(layout::kEventsHeader) + "\n1,arm,\n2,rec_start,\n3,rec_stop,\n");
  write_text(p.live_checks_path(), "{}\n");
  for (const char* cam : {"main", "uw"}) {
    write_text(p.camera_dir(cam) / layout::kCamData, std::string(layout::kCamDataHeader) + "\n");
    write_text(p.camera_dir(cam) / layout::kFrameMeta, std::string(layout::kFrameMetaHeader) + "\n");
    write_text(p.camera_dir(cam) / layout::kDrops, std::string(layout::kDropsHeader) + "\n");
  }
  write_text(p.imu_dir() / layout::kImuData, std::string(layout::kImuDataHeader) + "\n");
  write_text(p.imu_dir() / layout::kChip, std::string(layout::kChipHeader) + "\n");
  write_text(p.imu_dir() / layout::kClockFit, "{}\n");
  p.manifest().recorded = true;
  p.manifest().recorded_unix_ns = now_unix_ns();
  ASSERT_TRUE(p.save_manifest().ok());
}

TEST(Session, CreateOpenAndFreeze) {
  const fs::path root = fresh_root("create");
  auto s = Session::create(root, info("block-raise-2026-10-30"), kExample);
  ASSERT_TRUE(s.ok()) << s.status().to_string();
  EXPECT_EQ(s->dir(), root / "block-raise-2026-10-30");
  EXPECT_EQ(read_file(s->rig_yaml_path()).value(), read_file(kExample).value());  // verbatim copy
  EXPECT_EQ(s->info().rig_config_hash, content_hash(load_rig_config(kExample).value()));
  EXPECT_NE(s->info().created_unix_ns, kInvalidTime);
  EXPECT_NE(s->info().anchor_raw_ns, kInvalidTime);
  EXPECT_FALSE(s->info().software_version.empty());
  for (const fs::path& d : {s->calibration_dir(), s->episodes_dir(), s->aborted_dir()}) EXPECT_TRUE(fs::is_directory(d));
  // Nothing but the session (no leftover temp dir) in the data root.
  int n = 0;
  for (const auto& e : fs::directory_iterator(root)) n += e.path().filename().string()[0] != '.' ? 1 : 100;
  EXPECT_EQ(n, 1);

  auto o = Session::open(s->dir());
  ASSERT_TRUE(o.ok()) << o.status().to_string();
  EXPECT_EQ(o->info().operator_name, "ruben");
  EXPECT_EQ(o->info().rig_config_hash, s->info().rig_config_hash);

  EXPECT_EQ(Session::create(root, info("block-raise-2026-10-30"), kExample).status().code(), StatusCode::kAlreadyExists);
  EXPECT_EQ(Session::create(root, info("block_raise"), kExample).status().code(), StatusCode::kInvalidArgument);
  SessionInfo wrong = info("other");
  wrong.rig_config_hash = "deadbeef";
  EXPECT_EQ(Session::create(root, wrong, kExample).status().code(), StatusCode::kInvalidArgument);
  EXPECT_FALSE(fs::exists(root / "other"));
  EXPECT_EQ(Session::create(root, info("missing"), root / "nope.yaml").status().code(), StatusCode::kNotFound);

  // Editing the frozen rig.yaml (a value, not a comment) is detected on open.
  std::string y = read_file(s->rig_yaml_path()).value();
  write_text(s->rig_yaml_path(), y + "\n# a comment does not matter\n");
  EXPECT_TRUE(Session::open(s->dir()).ok());
  const std::string from = "out_size: 224";
  y.replace(y.find(from), from.size(), "out_size: 256");
  write_text(s->rig_yaml_path(), y);
  EXPECT_EQ(Session::open(s->dir()).status().code(), StatusCode::kDataLoss);
}

TEST(Session, InvalidConfigIsRefused) {
  const fs::path root = fresh_root("badcfg");
  fs::create_directories(root);
  std::string y = read_file(kExample).value();
  y += "\nunknown_top_key: 1\n";
  write_text(root / "bad.yaml", y);
  auto s = Session::create(root, info("x"), root / "bad.yaml");
  EXPECT_EQ(s.status().code(), StatusCode::kInvalidArgument);
  EXPECT_NE(s.status().message().find("unknown_top_key"), std::string::npos);
}

TEST(Package, CreateLayoutListAndOpen) {
  const fs::path root = fresh_root("pkg");
  auto s = Session::create(root, info("firstreal"), kExample).value();
  const TimeNs t = 1790889237489000000LL;
  auto p = s.create_package(PackageKind::kDemonstration, Side::kRight, t, 1);
  ASSERT_TRUE(p.ok()) << p.status().to_string();
  EXPECT_EQ(p->dir().parent_path(), s.episodes_dir());
  EXPECT_EQ(p->name().str().rfind("2026-10-01T21-13-57.489Z_", 0), 0u);
  EXPECT_TRUE(fs::is_directory(p->camera_dir("main") / layout::kFramesDir));
  EXPECT_TRUE(fs::is_directory(p->camera_dir("uw") / layout::kFramesDir));
  EXPECT_TRUE(fs::is_directory(p->imu_dir()));
  EXPECT_FALSE(p->manifest().recorded);
  EXPECT_EQ(p->manifest().task, "block_raise");
  EXPECT_EQ(p->manifest().rig_config_hash, s.info().rig_config_hash);
  EXPECT_EQ(derive_state(p->manifest()), PackageState::kRecording);
  EXPECT_TRUE(p->validate_layout().ok()) << p->validate_layout().to_string();  // skeleton is valid while recording
  // Same name again (same time + seed) is refused, never overwritten.
  EXPECT_EQ(s.create_package(PackageKind::kDemonstration, Side::kRight, t, 1).status().code(), StatusCode::kAlreadyExists);

  auto cal = s.create_package(PackageKind::kGripperCalibration, Side::kRight, t + 1000000, 2).value();
  EXPECT_EQ(cal.dir().parent_path(), s.calibration_dir());
  auto rigcal = s.create_package(PackageKind::kRigCalibration, Side::kRight, t + 2000000, 3).value();
  EXPECT_EQ(rigcal.dir().parent_path(), s.calibration_dir());
  auto p2 = s.create_package(PackageKind::kDemonstration, Side::kRight, t + 3000000, 4).value();
  fs::create_directory(s.episodes_dir() / "not-a-package");
  fs::create_directory(s.episodes_dir() / "2026-10-01T21-13-59.000Z_11111_firstreal_demonstration_right");  // no manifest

  auto demos = s.list_packages(PackageKind::kDemonstration).value();
  ASSERT_EQ(demos.size(), 2u);
  EXPECT_EQ(demos[0], p->dir());
  EXPECT_EQ(demos[1], p2.dir());
  EXPECT_EQ(s.list_packages(PackageKind::kGripperCalibration).value().size(), 1u);
  EXPECT_EQ(s.list_packages(PackageKind::kRigCalibration).value().size(), 1u);

  auto o = Package::open(p->dir());
  ASSERT_TRUE(o.ok()) << o.status().to_string();
  EXPECT_EQ(o->name().str(), p->name().str());
  o->manifest().stages["validate_raw"] = StageRecord{};
  ASSERT_TRUE(o->save_manifest().ok());
  EXPECT_EQ(Package::open(p->dir()).value().manifest().stages.size(), 1u);
  EXPECT_FALSE(Package::open(s.episodes_dir() / "not-a-package").ok());

  // A manifest that does not match its directory is rejected.
  fs::rename(p2.dir(), s.episodes_dir() / "2026-10-01T21-14-00.000Z_22222_firstreal_demonstration_right");
  EXPECT_EQ(Package::open(s.episodes_dir() / "2026-10-01T21-14-00.000Z_22222_firstreal_demonstration_right")
                .status()
                .code(),
            StatusCode::kDataLoss);
}

TEST(Package, ValidateLayoutRecorded) {
  const fs::path root = fresh_root("validate");
  auto s = Session::create(root, info("firstreal"), kExample).value();
  auto p = s.create_package(PackageKind::kDemonstration, Side::kRight, 1790889237489000000LL, 9).value();
  fill_recorded_layout(p);
  ASSERT_TRUE(p.validate_layout().ok()) << p.validate_layout().to_string();
  // Optional files are checked when present.
  write_text(p.live_slam_path(), std::string(layout::kLiveSlamHeader) + "\n");
  EXPECT_TRUE(p.validate_layout().ok());
  // A crash leftover of an atomic write is ignored.
  write_text(p.dir() / ".manifest.json.tmp.1.2", "{");
  EXPECT_TRUE(p.validate_layout().ok());

  auto expect_bad = [&](StatusCode code, const std::string& needle) {
    const Status st = p.validate_layout();
    EXPECT_EQ(st.code(), code) << st.to_string();
    EXPECT_NE(st.message().find(needle), std::string::npos) << st.to_string();
  };
  write_text(p.camera_dir("uw") / layout::kFrameMeta, "t_ns,seq\n");
  expect_bad(StatusCode::kDataLoss, "raw/cam_uw/frame_meta.csv: header");
  write_text(p.camera_dir("uw") / layout::kFrameMeta, std::string(layout::kFrameMetaHeader) + "\n");
  fs::remove(p.imu_dir() / layout::kClockFit);
  expect_bad(StatusCode::kNotFound, "raw/imu0/clock_fit.json");
  write_text(p.imu_dir() / layout::kClockFit, "{}");
  fs::remove(p.live_checks_path());
  expect_bad(StatusCode::kNotFound, "raw/live_checks.json");
  write_text(p.live_checks_path(), "{}");
  write_text(p.dir() / "stray.txt", "x");
  expect_bad(StatusCode::kDataLoss, "stray.txt: unexpected entry");
  fs::remove(p.dir() / "stray.txt");
  fs::remove_all(p.camera_dir("uw") / layout::kFramesDir);
  expect_bad(StatusCode::kNotFound, "raw/cam_uw/data");
  fs::create_directory(p.camera_dir("uw") / layout::kFramesDir);
  EXPECT_TRUE(p.validate_layout().ok());
  write_text(p.dir() / "manifest.json", "{");
  expect_bad(StatusCode::kDataLoss, "manifest.json");
}

TEST(Package, DiscardKeepsLogsAndDeletesTheRest) {
  const fs::path root = fresh_root("discard");
  auto s = Session::create(root, info("firstreal"), kExample).value();
  auto p = s.create_package(PackageKind::kDemonstration, Side::kRight, 1790889237489000000LL, 5).value();
  {
    auto log = EventLog::open(p.events_path()).value();  // an open writer must not prevent the discard
    ASSERT_TRUE(log->append(1, event_type::kArm, "").ok());
    ASSERT_TRUE(log->append(2, event_type::kAbort, "pose_jump").ok());
    write_text(p.live_checks_path(), "{\"pose_jump\": \"fail\"}\n");
    write_text(p.camera_dir("uw") / layout::kFramesDir / "1.jpg", "jpeg");
    const std::string name = p.name().str();
    ASSERT_TRUE(p.discard(s.aborted_dir()).ok());
    EXPECT_FALSE(fs::exists(s.episodes_dir() / name));
    EXPECT_TRUE(fs::is_regular_file(s.aborted_dir() / name / "events.csv"));
    EXPECT_TRUE(fs::is_regular_file(s.aborted_dir() / name / "live_checks.json"));
    int entries = 0;
    for (const auto& e : fs::directory_iterator(s.episodes_dir())) {
      (void)e;
      ++entries;
    }
    EXPECT_EQ(entries, 0);  // no trash left behind
  }
  auto t = read_csv(s.aborted_dir() / p.name().str() / "events.csv").value();
  ASSERT_EQ(t.rows.size(), 2u);
  EXPECT_EQ(t.rows[1][2], "pose_jump");
  // Discard with no logs (aborted right after creation) still works.
  auto q = s.create_package(PackageKind::kDemonstration, Side::kRight, 1790889237489000000LL, 6).value();
  ASSERT_TRUE(q.discard(s.aborted_dir()).ok());
  EXPECT_TRUE(fs::is_directory(s.aborted_dir() / q.name().str()));
  EXPECT_EQ(q.discard(s.aborted_dir()).code(), StatusCode::kNotFound);
}

}  // namespace
}  // namespace rig
