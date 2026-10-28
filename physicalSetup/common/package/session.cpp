// SPDX-License-Identifier: MIT
// Session and Package directories (rig/package/package.hpp; layout PIPELINE 6.1).
//
// Crash safety:
// - Session::create builds the whole tree in a hidden sibling ".session-<name>.creating.<pid>" and publishes it with
//   one renameat2(RENAME_NOREPLACE): a session directory either exists complete or not at all.
// - Session::create_package claims the name with mkdir (fails if it exists), creates raw/..., and writes
//   manifest.json LAST (atomically): a directory without a manifest is an interrupted creation (Package::open and
//   list_packages ignore it; validate_layout reports it).
// - Package::discard: (1) moves events.csv + live_checks.json into aborted/<name>/ (fsync), (2) renames the package
//   dir to a hidden ".trash-<name>" (so it never looks like a package again), (3) deletes it. A crash between steps
//   leaves either the logs safe in aborted/ or an inert hidden directory.
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <system_error>

#include <nlohmann/json.hpp>

#include "manifest_json.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/package/atomic_file.hpp"
#include "rig/package/csv.hpp"
#include "rig/package/layout.hpp"
#include "rig/package/package.hpp"
#include "rig/time/clock.hpp"
#include "rig/version.hpp"

namespace fs = std::filesystem;

namespace rig {

using nlohmann::json;
using namespace pkgjson;

namespace {

constexpr int kSessionSchemaVersion = 1;

Status mkdir_one(const fs::path& p) {
  std::error_code ec;
  if (!fs::create_directory(p, ec)) {
    if (ec) return io_error(p.string() + ": mkdir: " + ec.message());
    return already_exists(p.string() + ": already exists");
  }
  return Status::Ok();
}

Status rename_noreplace(const fs::path& from, const fs::path& to) {
  if (::renameat2(AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), RENAME_NOREPLACE) == 0) return Status::Ok();
  const int e = errno;
  if (e == EEXIST || e == ENOTEMPTY) return already_exists(to.string() + ": already exists");
  if (e == EINVAL || e == ENOSYS) {  // file system without RENAME_NOREPLACE: check-then-rename
    std::error_code ec;
    if (fs::exists(to, ec)) return already_exists(to.string() + ": already exists");
    if (::rename(from.c_str(), to.c_str()) == 0) return Status::Ok();
  }
  return io_error(from.string() + " -> " + to.string() + ": rename: " + std::strerror(e));
}

/// rename, or copy + fsync + unlink across file systems.
Status move_file(const fs::path& from, const fs::path& to) {
  if (::rename(from.c_str(), to.c_str()) == 0) return Status::Ok();
  if (errno != EXDEV) return io_error(from.string() + " -> " + to.string() + ": rename: " + std::strerror(errno));
  RIG_ASSIGN_OR_RETURN(std::string data, read_file(from));
  RIG_RETURN_IF_ERROR(write_file_atomic(to, data));
  std::error_code ec;
  fs::remove(from, ec);
  return Status::Ok();
}

std::string session_to_json_text(const SessionInfo& s) {
  json j;
  j["schema_version"] = kSessionSchemaVersion;
  j["session"] = s.session;
  j["task"] = s.task;
  j["operator"] = s.operator_name;
  j["created_unix_ns"] = time_to_json(s.created_unix_ns);
  j["rig_config_hash"] = s.rig_config_hash;
  j["software_version"] = s.software_version;
  j["anchor_raw_ns"] = time_to_json(s.anchor_raw_ns);
  j["anchor_unix_ns"] = time_to_json(s.anchor_unix_ns);
  return j.dump(2) + "\n";
}

Result<SessionInfo> session_from_json_text(const std::string& text, const std::string& ctx) {
  json j;
  try {
    j = json::parse(text);
  } catch (const json::exception& e) {
    return data_loss(ctx + ": not valid JSON: " + e.what());
  }
  RIG_RETURN_IF_ERROR(check_keys(j,
                                 {"schema_version", "session", "task", "operator", "created_unix_ns",
                                  "rig_config_hash", "software_version", "anchor_raw_ns", "anchor_unix_ns"},
                                 ctx));
  auto sv = j.find("schema_version");
  if (sv == j.end() || !sv->is_number_integer() || sv->get<int>() != kSessionSchemaVersion)
    return failed_precondition(ctx + ": unsupported session.json schema_version");
  SessionInfo s;
  RIG_RETURN_IF_ERROR(str_from_json(j, "session", s.session, ctx));
  RIG_RETURN_IF_ERROR(str_from_json(j, "task", s.task, ctx));
  RIG_RETURN_IF_ERROR(str_from_json(j, "operator", s.operator_name, ctx));
  RIG_RETURN_IF_ERROR(time_from_json(j, "created_unix_ns", s.created_unix_ns, ctx));
  RIG_RETURN_IF_ERROR(str_from_json(j, "rig_config_hash", s.rig_config_hash, ctx));
  RIG_RETURN_IF_ERROR(str_from_json(j, "software_version", s.software_version, ctx));
  RIG_RETURN_IF_ERROR(time_from_json(j, "anchor_raw_ns", s.anchor_raw_ns, ctx));
  RIG_RETURN_IF_ERROR(time_from_json(j, "anchor_unix_ns", s.anchor_unix_ns, ctx));
  return s;
}

bool is_calibration_kind(PackageKind k) {
  return k == PackageKind::kGripperCalibration || k == PackageKind::kRigCalibration;
}

Status check_header(const fs::path& file, const char* expected, const fs::path& pkg) {
  const std::string rel = fs::relative(file, pkg).string();
  auto line = read_first_line(file);
  if (!line.ok()) {
    if (line.status().code() == StatusCode::kNotFound) return not_found(rel + ": missing");
    return data_loss(rel + ": " + line.status().message());
  }
  if (*line != expected) return data_loss(rel + ": header '" + *line + "', expected '" + expected + "'");
  return Status::Ok();
}

Status require_dir(const fs::path& p, const fs::path& pkg) {
  std::error_code ec;
  if (!fs::is_directory(p, ec)) return not_found(fs::relative(p, pkg).string() + ": missing directory");
  return Status::Ok();
}

Status require_file(const fs::path& p, const fs::path& pkg) {
  std::error_code ec;
  if (!fs::is_regular_file(p, ec)) return not_found(fs::relative(p, pkg).string() + ": missing file");
  return Status::Ok();
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Package
// ---------------------------------------------------------------------------------------------------------------

Result<Package> Package::open(const fs::path& dir) {
  Package p;
  p.dir_ = dir;
  const std::string base = dir.filename().empty() ? dir.parent_path().filename().string() : dir.filename().string();
  RIG_ASSIGN_OR_RETURN(p.name_, parse_package_name(base));
  RIG_ASSIGN_OR_RETURN(p.manifest_, read_manifest(dir / layout::kManifest));
  const Manifest& m = p.manifest_;
  if (m.name != base || m.kind != p.name_.kind || m.side != p.name_.side || m.session != p.name_.session)
    return data_loss((dir / layout::kManifest).string() + ": name/kind/side/session do not match the directory name '" +
                     base + "'");
  return p;
}

Status Package::save_manifest() const { return write_manifest_atomic(dir_ / layout::kManifest, manifest_); }

Status Package::discard(const fs::path& aborted_dir) {
  std::error_code ec;
  if (!fs::is_directory(dir_, ec)) return not_found(dir_.string() + ": package directory missing");
  fs::create_directories(aborted_dir, ec);
  if (ec) return io_error(aborted_dir.string() + ": mkdir: " + ec.message());
  const fs::path keep = aborted_dir / name_.str();
  fs::create_directory(keep, ec);
  if (ec) return io_error(keep.string() + ": mkdir: " + ec.message());
  for (const fs::path& f : {events_path(), live_checks_path()}) {
    if (fs::exists(f, ec)) RIG_RETURN_IF_ERROR(move_file(f, keep / f.filename()));
  }
  RIG_RETURN_IF_ERROR(fsync_path(keep));
  RIG_RETURN_IF_ERROR(fsync_path(aborted_dir));
  const fs::path trash = dir_.parent_path() / (".trash-" + name_.str());
  fs::remove_all(trash, ec);
  if (::rename(dir_.c_str(), trash.c_str()) != 0)
    return io_error(dir_.string() + ": rename to trash: " + std::strerror(errno));
  RIG_RETURN_IF_ERROR(fsync_path(dir_.parent_path()));
  fs::remove_all(trash, ec);
  if (ec) return io_error(trash.string() + ": delete: " + ec.message());
  return fsync_path(dir_.parent_path());
}

Status Package::validate_layout() const {
  const fs::path& d = dir_;
  RIG_RETURN_IF_ERROR(require_dir(d, d.parent_path()));
  RIG_RETURN_IF_ERROR(require_file(d / layout::kManifest, d));
  auto m = read_manifest(d / layout::kManifest);
  if (!m.ok()) return data_loss(std::string(layout::kManifest) + ": " + m.status().message());
  if (m->name != d.filename().string()) return data_loss(std::string(layout::kManifest) + ": name != directory name");
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(d, ec)) {
    const std::string n = e.path().filename().string();
    if (n == layout::kManifest || n == layout::kRawDir || n == layout::kStagesDir || n == layout::kExportDir ||
        is_atomic_temp_name(n))
      continue;
    return data_loss(n + ": unexpected entry in the package directory");
  }
  const fs::path raw = d / layout::kRawDir;
  RIG_RETURN_IF_ERROR(require_dir(raw, d));
  if (!m->recorded) return Status::Ok();  // still recording: only the skeleton is guaranteed

  RIG_RETURN_IF_ERROR(check_header(raw / layout::kEvents, layout::kEventsHeader, d));
  if (m->kind == PackageKind::kDemonstration) RIG_RETURN_IF_ERROR(require_file(raw / layout::kLiveChecks, d));
  if (fs::exists(raw / layout::kLiveSlam, ec)) RIG_RETURN_IF_ERROR(check_header(raw / layout::kLiveSlam, layout::kLiveSlamHeader, d));
  int n_cam = 0, n_imu = 0;
  std::vector<fs::path> subdirs;
  for (const auto& e : fs::directory_iterator(raw, ec))
    if (e.is_directory()) subdirs.push_back(e.path());
  std::sort(subdirs.begin(), subdirs.end());
  for (const fs::path& sd : subdirs) {
    const std::string n = sd.filename().string();
    if (n.rfind(layout::kCameraDirPrefix, 0) == 0) {
      ++n_cam;
      RIG_RETURN_IF_ERROR(check_header(sd / layout::kCamData, layout::kCamDataHeader, d));
      RIG_RETURN_IF_ERROR(check_header(sd / layout::kFrameMeta, layout::kFrameMetaHeader, d));
      RIG_RETURN_IF_ERROR(check_header(sd / layout::kDrops, layout::kDropsHeader, d));
      RIG_RETURN_IF_ERROR(require_dir(sd / layout::kFramesDir, d));
    } else {
      ++n_imu;
      RIG_RETURN_IF_ERROR(check_header(sd / layout::kImuData, layout::kImuDataHeader, d));
      RIG_RETURN_IF_ERROR(check_header(sd / layout::kChip, layout::kChipHeader, d));
      RIG_RETURN_IF_ERROR(require_file(sd / layout::kClockFit, d));
      if (fs::exists(sd / layout::kSampleMeta, ec))
        RIG_RETURN_IF_ERROR(check_header(sd / layout::kSampleMeta, layout::kSampleMetaHeader, d));
    }
  }
  if (n_cam == 0) return not_found(std::string(layout::kRawDir) + "/cam_*: no camera directory");
  if (n_imu == 0 && m->kind != PackageKind::kGripperCalibration)
    return not_found(std::string(layout::kRawDir) + "/<imu>: no IMU directory");
  return Status::Ok();
}

// ---------------------------------------------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------------------------------------------

Result<Session> Session::create(const fs::path& data_root, const SessionInfo& info_in, const fs::path& rig_yaml_path) {
  RIG_RETURN_IF_ERROR(validate_session_name(info_in.session));
  RIG_ASSIGN_OR_RETURN(std::string yaml_bytes, read_file(rig_yaml_path));
  auto cfg = parse_rig_config(yaml_bytes, rig_yaml_path.string());
  if (!cfg.ok()) return cfg.status();
  SessionInfo info = info_in;
  const std::string hash = content_hash(*cfg);
  if (!info.rig_config_hash.empty() && info.rig_config_hash != hash)
    return invalid_argument("session " + info.session + ": rig_config_hash " + info.rig_config_hash +
                            " != content_hash(rig.yaml) " + hash);
  info.rig_config_hash = hash;
  if (info.software_version.empty()) info.software_version = kSoftwareVersion;
  if (info.created_unix_ns == kInvalidTime || info.anchor_raw_ns == kInvalidTime) {
    const ClockAnchor a = sample_clock_anchor();
    if (info.created_unix_ns == kInvalidTime) info.created_unix_ns = a.unix_ns;
    if (info.anchor_raw_ns == kInvalidTime) {
      info.anchor_raw_ns = a.raw_ns;
      info.anchor_unix_ns = a.unix_ns;
    }
  }

  std::error_code ec;
  fs::create_directories(data_root, ec);
  if (ec) return io_error(data_root.string() + ": mkdir: " + ec.message());
  const fs::path final_dir = data_root / info.session;
  if (fs::exists(final_dir, ec)) return already_exists(final_dir.string() + ": session already exists");
  const fs::path tmp = data_root / (".session-" + info.session + ".creating." + std::to_string(::getpid()));
  fs::remove_all(tmp, ec);
  auto fail = [&](Status s) {
    std::error_code e2;
    fs::remove_all(tmp, e2);
    return s;
  };
  if (Status s = mkdir_one(tmp); !s.ok()) return fail(s);
  for (const char* sub : {"calibration", "episodes", "aborted"})
    if (Status s = mkdir_one(tmp / sub); !s.ok()) return fail(s);
  if (Status s = write_file_atomic(tmp / layout::kRigYaml, yaml_bytes); !s.ok()) return fail(s);
  if (Status s = write_file_atomic(tmp / layout::kSessionJson, session_to_json_text(info)); !s.ok()) return fail(s);
  for (const char* sub : {"calibration", "episodes", "aborted"})
    if (Status s = fsync_path(tmp / sub); !s.ok()) return fail(s);
  if (Status s = fsync_path(tmp); !s.ok()) return fail(s);
  if (Status s = rename_noreplace(tmp, final_dir); !s.ok()) return fail(s);
  RIG_RETURN_IF_ERROR(fsync_path(data_root));

  Session sess;
  sess.dir_ = final_dir;
  sess.info_ = info;
  for (const auto& kv : cfg->cameras) sess.camera_ids_.push_back(kv.first);
  sess.imu_id_ = cfg->imu.driver.id;
  return sess;
}

Result<Session> Session::open(const fs::path& session_dir) {
  RIG_ASSIGN_OR_RETURN(std::string text, read_file(session_dir / layout::kSessionJson));
  RIG_ASSIGN_OR_RETURN(SessionInfo info, session_from_json_text(text, (session_dir / layout::kSessionJson).string()));
  const std::string base =
      session_dir.filename().empty() ? session_dir.parent_path().filename().string() : session_dir.filename().string();
  if (info.session != base)
    return data_loss((session_dir / layout::kSessionJson).string() + ": session '" + info.session +
                     "' != directory name '" + base + "'");
  auto cfg = load_rig_config((session_dir / layout::kRigYaml).string());
  if (!cfg.ok()) return cfg.status().with_context("session " + base + " frozen rig.yaml");
  const std::string hash = content_hash(*cfg);
  if (hash != info.rig_config_hash)
    return data_loss("session " + base + ": rig.yaml content hash " + hash + " != session.json " +
                     info.rig_config_hash + " (the frozen config was modified)");
  Session s;
  s.dir_ = session_dir;
  s.info_ = info;
  for (const auto& kv : cfg->cameras) s.camera_ids_.push_back(kv.first);
  s.imu_id_ = cfg->imu.driver.id;
  return s;
}

Result<Package> Session::create_package(PackageKind kind, Side side, TimeNs unix_ns, uint64_t seed) {
  RIG_ASSIGN_OR_RETURN(PackageName name, make_package_name(info_.session, kind, side, unix_ns, seed));
  const fs::path parent = is_calibration_kind(kind) ? calibration_dir() : episodes_dir();
  const fs::path dir = parent / name.str();
  RIG_RETURN_IF_ERROR(mkdir_one(dir));  // claims the name
  Package p;
  p.dir_ = dir;
  p.name_ = name;
  auto fail = [&](Status s) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    return s;
  };
  if (Status s = mkdir_one(p.raw_dir()); !s.ok()) return fail(s);
  for (const std::string& id : camera_ids_) {
    if (Status s = mkdir_one(p.camera_dir(id)); !s.ok()) return fail(s);
    if (Status s = mkdir_one(p.camera_dir(id) / layout::kFramesDir); !s.ok()) return fail(s);
  }
  if (Status s = mkdir_one(p.imu_dir(imu_id_)); !s.ok()) return fail(s);
  Manifest& m = p.manifest_;
  m.name = name.str();
  m.kind = kind;
  m.side = side;
  m.session = info_.session;
  m.task = info_.task;
  m.recorded = false;
  m.created_unix_ns = unix_ns;
  m.rig_config_hash = info_.rig_config_hash;
  m.software_version = kSoftwareVersion;
  if (Status s = p.save_manifest(); !s.ok()) return fail(s);  // manifest last (see file comment)
  if (Status s = fsync_path(p.raw_dir()); !s.ok()) return fail(s);
  RIG_RETURN_IF_ERROR(fsync_path(parent));
  return p;
}

Result<std::vector<fs::path>> Session::list_packages(PackageKind kind) const {
  const fs::path parent = is_calibration_kind(kind) ? calibration_dir() : episodes_dir();
  std::vector<fs::path> out;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(parent, ec)) {
    if (!e.is_directory()) continue;
    auto n = parse_package_name(e.path().filename().string());
    if (!n.ok() || n->kind != kind || n->session != info_.session) continue;
    if (!fs::exists(e.path() / layout::kManifest)) continue;  // interrupted creation
    out.push_back(e.path());
  }
  if (ec) return io_error(parent.string() + ": list: " + ec.message());
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace rig
