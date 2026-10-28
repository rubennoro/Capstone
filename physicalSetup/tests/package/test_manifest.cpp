// SPDX-License-Identifier: MIT
// T3: manifest JSON + derived state, atomic writes under simulated crashes (forked child _exit at every step),
// EventLog, CSV writer/reader.
#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "rig/package/atomic_file.hpp"
#include "rig/package/csv.hpp"
#include "rig/package/layout.hpp"
#include "rig/package/package.hpp"
// Private serializer (the same text write_manifest_atomic writes).
#include "../../common/package/manifest_json.hpp"

namespace fs = std::filesystem;

namespace rig {
namespace {

fs::path fresh_dir(const std::string& name) {
  const fs::path d = fs::path(RIG_TEST_TMP_ROOT) / ("pkg_" + name + "_" + std::to_string(getpid()));
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}

Manifest sample_manifest() {
  Manifest m;
  m.name = "2026-10-01T21-13-57.489Z_75494_firstreal_demonstration_right";
  m.kind = PackageKind::kDemonstration;
  m.side = Side::kRight;
  m.session = "firstreal";
  m.task = "block_raise";
  m.recorded = true;
  m.created_unix_ns = 1790889237489000000LL;
  m.rig_config_hash = std::string(64, 'a');
  m.software_version = "physicalSetup 0.1.0 (test)";
  StageRecord r;
  r.impl = "orbslam3_mi";
  r.inputs_hash = std::string(64, 'b');
  r.outputs = {"stages/slam/traj.csv", "stages/slam/result.json"};
  r.tool_version = "x";
  r.t_start_unix_ns = 5;
  r.log = "stages/slam/log.txt";
  m.stages["validate_raw"] = StageRecord{};
  m.stages["slam"] = r;
  return m;
}

TEST(Manifest, RoundTripAndNullTimes) {
  const fs::path d = fresh_dir("rt");
  const Manifest m = sample_manifest();
  ASSERT_TRUE(write_manifest_atomic(d / "manifest.json", m).ok());
  auto r = read_manifest(d / "manifest.json");
  ASSERT_TRUE(r.ok()) << r.status().to_string();
  EXPECT_EQ(r->name, m.name);
  EXPECT_EQ(r->recorded_unix_ns, kInvalidTime);  // written as null
  EXPECT_EQ(r->stages.at("slam").outputs, m.stages.at("slam").outputs);
  EXPECT_EQ(r->stages.at("slam").t_end_unix_ns, kInvalidTime);
  EXPECT_EQ(r->stages.at("slam").t_start_unix_ns, 5);
  const std::string text = read_file(d / "manifest.json").value();
  EXPECT_NE(text.find("\"recorded_unix_ns\": null"), std::string::npos);
  EXPECT_NE(text.find("\"state\": \"slam_done\""), std::string::npos);
  EXPECT_EQ(read_manifest(d / "none.json").status().code(), StatusCode::kNotFound);
}

TEST(Manifest, RejectsCorruptOrForeignFiles) {
  const fs::path d = fresh_dir("bad");
  auto put = [&](const std::string& s) {
    std::ofstream(d / "m.json") << s;
    return read_manifest(d / "m.json").status();
  };
  EXPECT_EQ(put("{").code(), StatusCode::kDataLoss);
  EXPECT_EQ(put("[]").code(), StatusCode::kDataLoss);
  std::string good = read_file([&] {
                       (void)write_manifest_atomic(d / "g.json", sample_manifest());
                       return d / "g.json";
                     }())
                         .value();
  auto mutate = [&](const std::string& from, const std::string& to) {
    std::string s = good;
    const auto p = s.find(from);
    EXPECT_NE(p, std::string::npos) << from;
    s.replace(p, from.size(), to);
    return put(s);
  };
  EXPECT_EQ(mutate("\"schema_version\": 1", "\"schema_version\": 2").code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(mutate("\"kind\": \"demonstration\"", "\"kind\": \"demo\"").code(), StatusCode::kDataLoss);
  EXPECT_EQ(mutate("\"recorded\": true", "\"recorded\": 1").code(), StatusCode::kDataLoss);
  EXPECT_EQ(mutate("\"task\":", "\"tusk\":").code(), StatusCode::kDataLoss);
  EXPECT_EQ(mutate("\"status\": \"done\"", "\"status\": \"ok\"").code(), StatusCode::kDataLoss);
  EXPECT_EQ(mutate("\"created_unix_ns\": 1790889237489000000", "\"created_unix_ns\": \"x\"").code(),
            StatusCode::kDataLoss);
}

TEST(Manifest, DerivedStateFollowsPipelineOrder) {
  Manifest m;
  EXPECT_EQ(derive_state(m), PackageState::kRecording);
  m.recorded = true;
  EXPECT_EQ(derive_state(m), PackageState::kRecorded);
  StageRecord done, failed, skipped;
  failed.status = StageStatus::kFailed;
  skipped.status = StageStatus::kSkipped;
  m.stages["validate_raw"] = failed;
  EXPECT_EQ(derive_state(m), PackageState::kInvalid);
  m.stages["validate_raw"] = done;
  EXPECT_EQ(derive_state(m), PackageState::kRawOk);
  m.stages["slam"] = failed;  // re-runnable failure: state stays
  EXPECT_EQ(derive_state(m), PackageState::kRawOk);
  m.stages["slam"] = done;
  EXPECT_EQ(derive_state(m), PackageState::kSlamDone);
  m.stages["traj_fit"] = done;
  EXPECT_EQ(derive_state(m), PackageState::kTrajDone);
  m.stages["pose"] = done;
  EXPECT_EQ(derive_state(m), PackageState::kPosed);
  m.stages["width"] = skipped;
  EXPECT_EQ(derive_state(m), PackageState::kWidthDone);
  m.stages["export"] = done;  // export without validate does not count
  EXPECT_EQ(derive_state(m), PackageState::kWidthDone);
  m.stages["validate"] = failed;
  EXPECT_EQ(derive_state(m), PackageState::kInvalid);
  m.stages["validate"] = done;
  EXPECT_EQ(derive_state(m), PackageState::kExported);
  m.stages.erase("export");
  EXPECT_EQ(derive_state(m), PackageState::kValid);
}

// ---------------------------------------------------------------------------------------------------------------
// Atomic write: crash at every step in a forked child (real _exit, no cleanup), then check what a reader sees.
// ---------------------------------------------------------------------------------------------------------------

TEST(AtomicWrite, CrashAtEveryStepLeavesOldOrNewNeverPartial) {
  const fs::path d = fresh_dir("crash");
  const fs::path target = d / "manifest.json";
  Manifest old_m = sample_manifest();
  Manifest new_m = old_m;
  new_m.task = std::string(5000, 'x');  // large: a half write is clearly partial
  for (int with_old = 0; with_old < 2; ++with_old) {
    for (int step = 0; step <= 4; ++step) {
      fs::remove_all(d);
      fs::create_directories(d);
      if (with_old) {
        ASSERT_TRUE(write_manifest_atomic(target, old_m).ok());
      }
      const std::string text = manifest_to_json_text(new_m);
      const pid_t pid2 = fork();
      ASSERT_GE(pid2, 0);
      if (pid2 == 0) {
        AtomicWriteHook hook = [step](AtomicWriteStep s) {
          if (static_cast<int>(s) == step) _exit(0);  // crash here: no cleanup, no unwinding
          return true;
        };
        (void)write_file_atomic(target, text, hook);
        _exit(1);
      }
      int st = 0;
      waitpid(pid2, &st, 0);
      ASSERT_TRUE(WIFEXITED(st) && WEXITSTATUS(st) == 0) << "step " << step;
      auto r = read_manifest(target);
      if (step <= 3) {  // before rename: the old file (or none)
        if (with_old) {
          ASSERT_TRUE(r.ok()) << "step " << step << ": " << r.status().to_string();
          EXPECT_EQ(r->task, old_m.task) << "step " << step;
        } else {
          EXPECT_EQ(r.status().code(), StatusCode::kNotFound) << "step " << step;
        }
      } else {  // renamed, dir not yet fsynced: the new file (the rename is visible to this host)
        ASSERT_TRUE(r.ok()) << r.status().to_string();
        EXPECT_EQ(r->task, new_m.task);
      }
      for (const auto& e : fs::directory_iterator(d)) {
        const std::string n = e.path().filename().string();
        EXPECT_TRUE(n == "manifest.json" || is_atomic_temp_name(n)) << n;
      }
    }
  }
}

TEST(AtomicWrite, HookCancelAndOverwrite) {
  const fs::path d = fresh_dir("hook");
  const fs::path f = d / "x.json";
  ASSERT_TRUE(write_file_atomic(f, "one").ok());
  EXPECT_EQ(write_file_atomic(f, "two", [](AtomicWriteStep s) { return s != AtomicWriteStep::kRename; }).code(),
            StatusCode::kCancelled);
  EXPECT_EQ(read_file(f).value(), "one");
  ASSERT_TRUE(write_file_atomic(f, "three").ok());
  EXPECT_EQ(read_file(f).value(), "three");
  EXPECT_FALSE(write_file_atomic(d / "nodir" / "x", "a").ok());
  EXPECT_TRUE(is_atomic_temp_name(".manifest.json.tmp.123.4"));
  EXPECT_FALSE(is_atomic_temp_name("manifest.json"));
}

// ---------------------------------------------------------------------------------------------------------------
// EventLog
// ---------------------------------------------------------------------------------------------------------------

TEST(EventLog, HeaderQuotingAppendAndThreads) {
  const fs::path d = fresh_dir("events");
  const fs::path p = d / "events.csv";
  {
    auto log = EventLog::open(p).value();
    EXPECT_EQ(log->path(), p);
    ASSERT_TRUE(log->append(100, event_type::kArm, "").ok());
    ASSERT_TRUE(log->append(200, event_type::kWarning, "thermal, 86.5 C \"hot\"\nline2").ok());
    EXPECT_EQ(log->append(kInvalidTime, "info", "x").code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(log->append(1, "Bad Type", "x").code(), StatusCode::kInvalidArgument);
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t)
      ts.emplace_back([&, t] {
        for (int i = 0; i < 250; ++i) ASSERT_TRUE(log->append(1000 + t * 1000 + i, "info", "t" + std::to_string(t)).ok());
      });
    for (auto& t : ts) t.join();
    ASSERT_TRUE(log->sync().ok());
  }
  {
    auto log = EventLog::open(p).value();  // reopen appends, no second header
    ASSERT_TRUE(log->append(9999, event_type::kRecStop, "done").ok());
  }
  auto t = read_csv(p).value();
  EXPECT_EQ(t.header, (std::vector<std::string>{"t_ns", "type", "detail"}));
  ASSERT_EQ(t.rows.size(), 2u + 1000u + 1u);
  EXPECT_EQ(t.rows[1][2], "thermal, 86.5 C \"hot\"\nline2");
  EXPECT_EQ(t.rows.back()[1], "rec_stop");
  for (const auto& r : t.rows) EXPECT_EQ(r.size(), 3u);

  // A file with a foreign header is refused; a torn last line is terminated before appending.
  std::ofstream(d / "foreign.csv") << "a,b\n";
  EXPECT_EQ(EventLog::open(d / "foreign.csv").status().code(), StatusCode::kDataLoss);
  std::ofstream(d / "torn.csv") << "t_ns,type,detail\n1,info,par";
  {
    auto log = EventLog::open(d / "torn.csv").value();
    ASSERT_TRUE(log->append(2, "info", "next").ok());
  }
  auto torn = read_csv(d / "torn.csv").value();
  ASSERT_EQ(torn.rows.size(), 2u);
  EXPECT_EQ(torn.rows[1][2], "next");
}

// ---------------------------------------------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------------------------------------------

TEST(Csv, WriterReaderRoundTrip) {
  const fs::path d = fresh_dir("csv");
  auto w = CsvWriter::create(d / "data.csv", layout::kImuDataHeader, 64).value();  // tiny buffer: many flushes
  for (int i = 0; i < 1000; ++i) {
    w->field(static_cast<int64_t>(1000000000000LL + i)).field(0.1 * i).field(-1e-300).field(std::nan(""));
    w->field(static_cast<uint64_t>(i)).field("a,b").field(std::string_view("plain"));
    ASSERT_TRUE(w->end_row().ok());
  }
  EXPECT_EQ(w->rows(), 1000u);
  ASSERT_TRUE(w->sync().ok());
  ASSERT_TRUE(w->close().ok());
  ASSERT_TRUE(w->close().ok());  // idempotent
  EXPECT_EQ(read_first_line(d / "data.csv").value(), layout::kImuDataHeader);
  auto t = read_csv(d / "data.csv").value();
  EXPECT_EQ(t.header.size(), 7u);
  ASSERT_EQ(t.rows.size(), 1000u);
  EXPECT_EQ(t.rows[7][0], "1000000000007");
  EXPECT_DOUBLE_EQ(std::stod(t.rows[7][1]), 0.1 * 7);  // %.17g round-trips
  EXPECT_EQ(t.rows[7][3], "nan");
  EXPECT_EQ(t.rows[7][5], "a,b");
  EXPECT_EQ(parse_csv("a,b\n\"x\"\"y\",\"1\r\n2\"\r\n").value().rows[0][0], "x\"y");
  EXPECT_EQ(parse_csv("a\n\"open").status().code(), StatusCode::kDataLoss);
  EXPECT_EQ(csv_quote("plain"), "plain");
  EXPECT_EQ(csv_quote("q\""), "\"q\"\"\"");
}

}  // namespace
}  // namespace rig
