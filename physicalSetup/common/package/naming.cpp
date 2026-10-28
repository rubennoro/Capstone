// SPDX-License-Identifier: MIT
// iPhUMI-identical package naming (rig/package/package.hpp; sources cited there):
//   <ISO>_<5 digits>_<session>_<kind>_<side>
#include <ctime>
#include <regex>

#include "rig/package/package.hpp"

namespace rig {
namespace {

const std::regex& session_re() {
  static const std::regex r("[A-Za-z0-9-]{1,64}");
  return r;
}
// ISO8601DateFormatter [.withInternetDateTime, .withFractionalSeconds] -> "2026-10-01T21:13:57.489Z"
// (DateManager.swift:13-17), then ':' -> '-' (ViewController.swift:1336).
const std::regex& iso_re() {
  static const std::regex r("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}\\.[0-9]{3}Z");
  return r;
}

uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

}  // namespace

Status validate_session_name(std::string_view session) {
  if (!std::regex_match(session.begin(), session.end(), session_re()))
    return invalid_argument("session name '" + std::string(session) +
                            "': must match [A-Za-z0-9-]+ (1..64 chars; no '_': iPhUMI group.py:36-37 splits package "
                            "names on '_')");
  return Status::Ok();
}

std::string iso_name_timestamp(TimeNs unix_ns) {
  // Floor division so pre-1970 times (never used, but defined) stay consistent.
  int64_t sec = unix_ns / 1000000000LL;
  int64_t rem = unix_ns % 1000000000LL;
  if (rem < 0) {
    rem += 1000000000LL;
    sec -= 1;
  }
  const std::time_t t = static_cast<std::time_t>(sec);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[96];  // generous: -Wformat-truncation cannot bound the int fields
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d-%02d-%02d.%03dZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(rem / 1000000));
  return buf;
}

std::string random_five_digits(uint64_t seed) {
  // ViewController.swift:1097-1102: five times Int.random(in: 1...9).
  std::string s;
  uint64_t x = seed;
  for (int i = 0; i < 5; ++i) s.push_back(static_cast<char>('1' + splitmix64(x) % 9));
  return s;
}

Result<PackageName> make_package_name(std::string_view session, PackageKind kind, Side side, TimeNs unix_ns,
                                      uint64_t seed) {
  RIG_RETURN_IF_ERROR(validate_session_name(session));
  if (unix_ns == kInvalidTime) return invalid_argument("make_package_name: invalid time");
  PackageName n;
  n.iso = iso_name_timestamp(unix_ns);
  n.randomizer = random_five_digits(seed);
  n.session = std::string(session);
  n.kind = kind;
  n.side = side;
  return n;
}

Result<PackageName> parse_package_name(std::string_view name) {
  // group.py:36-38: name.split('_'), exactly 5 parts.
  std::vector<std::string> parts;
  std::size_t start = 0;
  while (true) {
    const std::size_t p = name.find('_', start);
    parts.emplace_back(name.substr(start, p == std::string_view::npos ? std::string_view::npos : p - start));
    if (p == std::string_view::npos) break;
    start = p + 1;
  }
  const std::string n(name);
  if (parts.size() != 5)
    return invalid_argument("package name '" + n + "': " + std::to_string(parts.size()) +
                            " '_'-separated parts, need 5 (<ISO>_<5 digits>_<session>_<kind>_<side>)");
  PackageName out;
  if (!std::regex_match(parts[0], iso_re()))
    return invalid_argument("package name '" + n + "': bad timestamp '" + parts[0] + "' (YYYY-MM-DDTHH-MM-SS.mmmZ)");
  static const std::regex kDigits("[1-9]{5}");
  if (!std::regex_match(parts[1], kDigits))
    return invalid_argument("package name '" + n + "': randomizer '" + parts[1] + "' must be 5 digits 1-9");
  RIG_RETURN_IF_ERROR(validate_session_name(parts[2]).with_context("package name '" + n + "'"));
  bool kind_ok = false;
  for (PackageKind k : {PackageKind::kDemonstration, PackageKind::kGripperCalibration, PackageKind::kRigCalibration}) {
    if (parts[3] == to_string(k)) {
      out.kind = k;
      kind_ok = true;
    }
  }
  if (!kind_ok)
    return invalid_argument("package name '" + n + "': kind '" + parts[3] +
                            "' is not demonstration | grippercalibration | rigcalibration");
  bool side_ok = false;
  for (Side s : {Side::kLeft, Side::kRight, Side::kHead}) {
    if (parts[4] == to_string(s)) {
      out.side = s;
      side_ok = true;
    }
  }
  if (!side_ok) return invalid_argument("package name '" + n + "': side '" + parts[4] + "' is not left | right | head");
  out.iso = parts[0];
  out.randomizer = parts[1];
  out.session = parts[2];
  return out;
}

}  // namespace rig
