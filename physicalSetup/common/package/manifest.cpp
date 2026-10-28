// SPDX-License-Identifier: MIT
// manifest.json (rig/package/package.hpp; PIPELINE 7.1) and events.csv (EventLog).
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <mutex>
#include <regex>

#include <nlohmann/json.hpp>

#include "rig/package/atomic_file.hpp"
#include "rig/package/csv.hpp"
#include "rig/package/layout.hpp"
#include "rig/package/package.hpp"
#include "manifest_json.hpp"

namespace rig {

using nlohmann::json;

namespace pkgjson {

json time_to_json(TimeNs t) { return t == kInvalidTime ? json(nullptr) : json(t); }

Status time_from_json(const json& j, const char* key, TimeNs& out, const std::string& ctx) {
  auto it = j.find(key);
  if (it == j.end()) return data_loss(ctx + "." + key + ": missing");
  if (it->is_null()) {
    out = kInvalidTime;
    return Status::Ok();
  }
  if (!it->is_number_integer()) return data_loss(ctx + "." + key + ": expected an integer ns time or null");
  out = it->get<int64_t>();
  return Status::Ok();
}

Status str_from_json(const json& j, const char* key, std::string& out, const std::string& ctx) {
  auto it = j.find(key);
  if (it == j.end()) return data_loss(ctx + "." + key + ": missing");
  if (!it->is_string()) return data_loss(ctx + "." + key + ": expected a string");
  out = it->get<std::string>();
  return Status::Ok();
}

Status check_keys(const json& j, std::initializer_list<const char*> allowed, const std::string& ctx) {
  if (!j.is_object()) return data_loss(ctx + ": expected an object");
  for (auto it = j.begin(); it != j.end(); ++it) {
    bool ok = false;
    for (const char* a : allowed) ok = ok || it.key() == a;
    if (!ok) return data_loss(ctx + "." + it.key() + ": unknown key");
  }
  return Status::Ok();
}

}  // namespace pkgjson

using namespace pkgjson;

namespace {

bool stage_passed(const Manifest& m, const char* name) {
  auto it = m.stages.find(name);
  return it != m.stages.end() && (it->second.status == StageStatus::kDone || it->second.status == StageStatus::kSkipped);
}
bool stage_failed(const Manifest& m, const char* name) {
  auto it = m.stages.find(name);
  return it != m.stages.end() && it->second.status == StageStatus::kFailed;
}

template <class E, std::size_t N>
bool enum_from(const std::string& s, const E (&values)[N], E& out) {
  for (E v : values) {
    if (s == to_string(v)) {
      out = v;
      return true;
    }
  }
  return false;
}

}  // namespace

PackageState derive_state(const Manifest& m) {
  // PIPELINE 7.1: recorded -> raw_ok -> slam_done -> traj_done -> posed -> width_done -> valid|invalid -> exported.
  // A stage counts as passed when done or skipped (e.g. width for a rig calibration). A failed validate_raw or
  // validate means the episode is invalid; other failures leave the state where it was (re-runnable).
  if (!m.recorded) return PackageState::kRecording;
  if (stage_failed(m, "validate_raw")) return PackageState::kInvalid;
  if (!stage_passed(m, "validate_raw")) return PackageState::kRecorded;
  if (!stage_passed(m, "slam")) return PackageState::kRawOk;
  if (!stage_passed(m, "traj_fit")) return PackageState::kSlamDone;
  if (!stage_passed(m, "pose")) return PackageState::kTrajDone;
  if (!stage_passed(m, "width")) return PackageState::kPosed;
  if (stage_failed(m, "validate")) return PackageState::kInvalid;
  if (!stage_passed(m, "validate")) return PackageState::kWidthDone;
  if (stage_passed(m, "export")) return PackageState::kExported;
  return PackageState::kValid;
}

std::string manifest_to_json_text(const Manifest& m) {
  json j;
  j["schema_version"] = m.schema_version;
  j["name"] = m.name;
  j["kind"] = to_string(m.kind);
  j["side"] = to_string(m.side);
  j["session"] = m.session;
  j["task"] = m.task;
  j["recorded"] = m.recorded;
  j["created_unix_ns"] = time_to_json(m.created_unix_ns);
  j["recorded_unix_ns"] = time_to_json(m.recorded_unix_ns);
  j["rig_config_hash"] = m.rig_config_hash;
  j["software_version"] = m.software_version;
  j["state"] = to_string(derive_state(m));  // informational; recomputed on read
  json stages = json::object();
  for (const auto& kv : m.stages) {
    const StageRecord& r = kv.second;
    stages[kv.first] = {{"status", to_string(r.status)},
                        {"impl", r.impl},
                        {"inputs_hash", r.inputs_hash},
                        {"outputs", r.outputs},
                        {"tool_version", r.tool_version},
                        {"t_start_unix_ns", time_to_json(r.t_start_unix_ns)},
                        {"t_end_unix_ns", time_to_json(r.t_end_unix_ns)},
                        {"log", r.log},
                        {"message", r.message}};
  }
  j["stages"] = stages;
  return j.dump(2) + "\n";
}

Result<Manifest> manifest_from_json_text(const std::string& text, const std::string& ctx) {
  json j;
  try {
    j = json::parse(text);
  } catch (const json::exception& e) {
    return data_loss(ctx + ": not valid JSON: " + e.what());
  }
  RIG_RETURN_IF_ERROR(check_keys(j,
                                 {"schema_version", "name", "kind", "side", "session", "task", "recorded",
                                  "created_unix_ns", "recorded_unix_ns", "rig_config_hash", "software_version",
                                  "state", "stages"},
                                 ctx));
  Manifest m;
  auto sv = j.find("schema_version");
  if (sv == j.end() || !sv->is_number_integer()) return data_loss(ctx + ".schema_version: missing or not an integer");
  m.schema_version = sv->get<int>();
  if (m.schema_version != kManifestSchemaVersion)
    return failed_precondition(ctx + ": manifest schema " + std::to_string(m.schema_version) + ", this build reads " +
                               std::to_string(kManifestSchemaVersion));
  RIG_RETURN_IF_ERROR(str_from_json(j, "name", m.name, ctx));
  std::string kind, side;
  RIG_RETURN_IF_ERROR(str_from_json(j, "kind", kind, ctx));
  static const PackageKind kKinds[] = {PackageKind::kDemonstration, PackageKind::kGripperCalibration,
                                       PackageKind::kRigCalibration};
  if (!enum_from(kind, kKinds, m.kind)) return data_loss(ctx + ".kind: '" + kind + "' unknown");
  RIG_RETURN_IF_ERROR(str_from_json(j, "side", side, ctx));
  static const Side kSides[] = {Side::kLeft, Side::kRight, Side::kHead};
  if (!enum_from(side, kSides, m.side)) return data_loss(ctx + ".side: '" + side + "' unknown");
  RIG_RETURN_IF_ERROR(str_from_json(j, "session", m.session, ctx));
  RIG_RETURN_IF_ERROR(str_from_json(j, "task", m.task, ctx));
  auto rec = j.find("recorded");
  if (rec == j.end() || !rec->is_boolean()) return data_loss(ctx + ".recorded: missing or not a boolean");
  m.recorded = rec->get<bool>();
  RIG_RETURN_IF_ERROR(time_from_json(j, "created_unix_ns", m.created_unix_ns, ctx));
  RIG_RETURN_IF_ERROR(time_from_json(j, "recorded_unix_ns", m.recorded_unix_ns, ctx));
  RIG_RETURN_IF_ERROR(str_from_json(j, "rig_config_hash", m.rig_config_hash, ctx));
  RIG_RETURN_IF_ERROR(str_from_json(j, "software_version", m.software_version, ctx));
  auto st = j.find("stages");
  if (st == j.end() || !st->is_object()) return data_loss(ctx + ".stages: missing or not an object");
  static const StageStatus kStatuses[] = {StageStatus::kDone, StageStatus::kFailed, StageStatus::kSkipped};
  for (auto it = st->begin(); it != st->end(); ++it) {
    const std::string sctx = ctx + ".stages." + it.key();
    const json& s = it.value();
    RIG_RETURN_IF_ERROR(check_keys(s,
                                   {"status", "impl", "inputs_hash", "outputs", "tool_version", "t_start_unix_ns",
                                    "t_end_unix_ns", "log", "message"},
                                   sctx));
    StageRecord r;
    std::string status;
    RIG_RETURN_IF_ERROR(str_from_json(s, "status", status, sctx));
    if (!enum_from(status, kStatuses, r.status)) return data_loss(sctx + ".status: '" + status + "' unknown");
    RIG_RETURN_IF_ERROR(str_from_json(s, "impl", r.impl, sctx));
    RIG_RETURN_IF_ERROR(str_from_json(s, "inputs_hash", r.inputs_hash, sctx));
    auto outs = s.find("outputs");
    if (outs == s.end() || !outs->is_array()) return data_loss(sctx + ".outputs: missing or not a list");
    for (const json& o : *outs) {
      if (!o.is_string()) return data_loss(sctx + ".outputs: not a string");
      r.outputs.push_back(o.get<std::string>());
    }
    RIG_RETURN_IF_ERROR(str_from_json(s, "tool_version", r.tool_version, sctx));
    RIG_RETURN_IF_ERROR(time_from_json(s, "t_start_unix_ns", r.t_start_unix_ns, sctx));
    RIG_RETURN_IF_ERROR(time_from_json(s, "t_end_unix_ns", r.t_end_unix_ns, sctx));
    RIG_RETURN_IF_ERROR(str_from_json(s, "log", r.log, sctx));
    RIG_RETURN_IF_ERROR(str_from_json(s, "message", r.message, sctx));
    m.stages[it.key()] = std::move(r);
  }
  return m;
}

Result<Manifest> read_manifest(const std::filesystem::path& path) {
  RIG_ASSIGN_OR_RETURN(std::string text, read_file(path));
  return manifest_from_json_text(text, path.string());
}

Status write_manifest_atomic(const std::filesystem::path& path, const Manifest& manifest) {
  return write_file_atomic(path, manifest_to_json_text(manifest));
}

// ---------------------------------------------------------------------------------------------------------------
// EventLog
// ---------------------------------------------------------------------------------------------------------------

namespace {

class EventLogImpl final : public EventLog {
 public:
  EventLogImpl(std::filesystem::path p, int fd) : path_(std::move(p)), fd_(fd) {}
  ~EventLogImpl() override {
    if (fd_ >= 0) ::close(fd_);
  }

  Status append(TimeNs t_ns, std::string_view type, std::string_view detail) override {
    static const std::regex kType("[a-z0-9_]+");
    if (t_ns == kInvalidTime) return invalid_argument(path_.string() + ": event without a timestamp");
    if (!std::regex_match(type.begin(), type.end(), kType))
      return invalid_argument(path_.string() + ": event type '" + std::string(type) + "' must match [a-z0-9_]+");
    // One write(2) per line on an O_APPEND fd: lines never interleave and nothing sits in a user-space buffer.
    const std::string line = std::to_string(t_ns) + "," + std::string(type) + "," + csv_quote(detail) + "\n";
    std::lock_guard<std::mutex> lk(mu_);
    std::size_t off = 0;
    while (off < line.size()) {
      const ssize_t w = ::write(fd_, line.data() + off, line.size() - off);
      if (w < 0) {
        if (errno == EINTR) continue;
        return io_error(path_.string() + ": write: " + std::strerror(errno));
      }
      off += static_cast<std::size_t>(w);
    }
    return Status::Ok();
  }

  Status sync() override {
    std::lock_guard<std::mutex> lk(mu_);
    if (::fsync(fd_) != 0) return io_error(path_.string() + ": fsync: " + std::strerror(errno));
    return Status::Ok();
  }

  const std::filesystem::path& path() const override { return path_; }

 private:
  std::filesystem::path path_;
  int fd_;
  std::mutex mu_;
};

}  // namespace

Result<std::unique_ptr<EventLog>> EventLog::open(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  if (fd < 0) return io_error(path.string() + ": open: " + std::strerror(errno));
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    ::close(fd);
    return io_error(path.string() + ": fstat: " + std::strerror(errno));
  }
  auto log = std::make_unique<EventLogImpl>(path, fd);
  std::string prefix;
  if (st.st_size == 0) {
    prefix = std::string(layout::kEventsHeader) + "\n";
  } else {
    auto first = read_first_line(path);
    if (!first.ok()) return first.status();
    if (*first != layout::kEventsHeader)
      return data_loss(path.string() + ": header '" + *first + "' != '" + layout::kEventsHeader + "'");
    // A crash mid-line (cannot happen with single writes, but e.g. a full disk) must not glue two events.
    char last = '\n';
    if (pread(fd, &last, 1, st.st_size - 1) == 1 && last != '\n') prefix = "\n";
  }
  if (!prefix.empty() && ::write(fd, prefix.data(), prefix.size()) != static_cast<ssize_t>(prefix.size()))
    return io_error(path.string() + ": write header: " + std::strerror(errno));
  return std::unique_ptr<EventLog>(std::move(log));
}

}  // namespace rig
