// SPDX-License-Identifier: MIT
// Private helpers of the rig.yaml loader (common/config/). Strict typed access to a yaml-cpp mapping with the YAML
// path in every message, unknown-key and duplicate-key detection. Never throws (yaml-cpp exceptions are caught).
#pragma once

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

namespace rig {
namespace cfg {

/// Collected validation errors ("<yaml.path>: <message> (<source>:<line>)").
class Issues {
 public:
  explicit Issues(std::string source) : source_(std::move(source)) {}
  /// `mark_line` is 0-based (yaml-cpp Mark::line); < 0 = unknown.
  void add(const std::string& path, int mark_line, const std::string& message);
  bool empty() const { return errors_.empty(); }
  const std::vector<std::string>& errors() const { return errors_; }
  /// All errors in one message: the first one, then "; N more: ..." (each still path-first).
  std::string summary() const;
  const std::string& source() const { return source_; }

 private:
  std::string source_;
  std::vector<std::string> errors_;
};

/// Closed or open interval for range checks.
struct DRange {
  double lo;
  double hi;
  bool lo_open = false;
  bool hi_open = false;
  bool contains(double v) const;
  std::string str() const;
};
inline DRange closed(double lo, double hi) { return DRange{lo, hi, false, false}; }
inline DRange open_lo(double lo, double hi) { return DRange{lo, hi, true, false}; }  ///< (lo, hi]
inline DRange open_both(double lo, double hi) { return DRange{lo, hi, true, true}; }

struct IRange {
  int64_t lo;
  int64_t hi;
  bool contains(int64_t v) const { return v >= lo && v <= hi; }
  std::string str() const;
};

/// Formats a double for messages (shortest of %g with up to 17 digits that round-trips).
std::string fmt_double(double v);

/// One YAML mapping at `path`. A missing node (key absent) is allowed: every read leaves the default, and required
/// reads report "missing required key". Call finish() to report unknown keys.
class MapReader {
 public:
  MapReader(Issues& issues, YAML::Node node, std::string path, int parent_line);

  bool present() const { return present_; }
  bool ok_map() const { return is_map_; }
  const std::string& path() const { return path_; }
  int line() const { return line_; }
  /// "<path>.<key>" (or "<key>" at the root).
  std::string child_path(const std::string& key) const;
  /// True if `key` is present (marks it as known).
  bool has(const std::string& key);
  /// Marks `key` known and returns its node (undefined if absent).
  YAML::Node take(const std::string& key);
  /// Line of a child (or of this map if absent), 0-based.
  int child_line(const std::string& key);
  /// Reports keys that were never taken. Idempotent.
  void finish();
  /// Keys of this mapping in document order (for maps keyed by id).
  std::vector<std::string> keys() const;
  Issues& issues() { return issues_; }
  void error(const std::string& key, const std::string& message);  ///< At child_path(key).
  void error_here(const std::string& message);                     ///< At path().

  /// Typed reads. Return true if the key was present and valid (out written); false otherwise (out untouched).
  bool i64(const std::string& key, int64_t& out, IRange range, bool required = false);
  bool i32(const std::string& key, int& out, IRange range, bool required = false);
  bool u64(const std::string& key, uint64_t& out, bool required = false);
  bool size(const std::string& key, std::size_t& out, IRange range, bool required = false);
  bool f64(const std::string& key, double& out, DRange range, bool required = false);
  bool boolean(const std::string& key, bool& out, bool required = false);
  bool str(const std::string& key, std::string& out, bool required = false);
  template <class E>
  bool enumeration(const std::string& key, E& out, const std::vector<std::pair<const char*, E>>& table,
                   bool required = false);
  bool vec3(const std::string& key, Eigen::Vector3d& out, bool required = false);
  bool mat3(const std::string& key, Eigen::Matrix3d& out, bool required = false);
  bool f64_list(const std::string& key, std::vector<double>& out, bool required = false);
  bool i32_list(const std::string& key, std::vector<int>& out, IRange range, bool required = false);
  bool str_list(const std::string& key, std::vector<std::string>& out, bool required = false);
  /// Sub-mapping (absent -> a reader with present() == false; required -> also an error).
  MapReader sub(const std::string& key, bool required = false);

 private:
  bool fetch_scalar(const std::string& key, bool required, YAML::Node& n, std::string& text, bool& quoted);
  void missing(const std::string& key);

  Issues& issues_;
  YAML::Node node_;
  std::string path_;
  int line_ = -1;
  bool present_ = false;
  bool is_map_ = false;
  bool finished_ = false;
  std::set<std::string> known_;
  std::vector<std::pair<std::string, int>> keys_;  ///< (key, line), document order.
};

/// Scalar parsers on raw text (exposed for tests). Return false on a syntax error.
bool parse_int_text(const std::string& text, int64_t& out);   ///< [+-]digits or [+-]0x hex; overflow = false.
bool parse_double_text(const std::string& text, double& out);  ///< finite decimal / exponent; ints allowed.
bool parse_bool_text(const std::string& text, bool& out);      ///< true|false|True|False|TRUE|FALSE.

template <class E>
bool MapReader::enumeration(const std::string& key, E& out, const std::vector<std::pair<const char*, E>>& table,
                            bool required) {
  YAML::Node n;
  std::string text;
  bool quoted = false;
  if (!fetch_scalar(key, required, n, text, quoted)) return false;
  for (const auto& kv : table) {
    if (text == kv.first) {
      out = kv.second;
      return true;
    }
  }
  std::string allowed;
  for (const auto& kv : table) {
    if (!allowed.empty()) allowed += " | ";
    allowed += kv.first;
  }
  error(key, "'" + text + "' is not one of " + allowed);
  return false;
}

}  // namespace cfg
}  // namespace rig
