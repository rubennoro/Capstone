// SPDX-License-Identifier: MIT
// Private helpers of the rig.yaml loader; see yaml_reader.hpp.
#include "yaml_reader.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>

namespace rig {
namespace cfg {
namespace {

// yaml-cpp: plain (unquoted) scalars carry the non-specific tag "?", quoted / block scalars "!" (yaml-cpp
// src/nodebuilder.cpp, YAML 1.2 spec 6.9.1). Numbers and booleans must be plain so that `fps: "60"` is a type error.
bool is_quoted(const YAML::Node& n) { return n.Tag() == "!"; }

const char* node_type_name(const YAML::Node& n) {
  switch (n.Type()) {
    case YAML::NodeType::Undefined: return "nothing";
    case YAML::NodeType::Null: return "null";
    case YAML::NodeType::Scalar: return "a scalar";
    case YAML::NodeType::Sequence: return "a list";
    case YAML::NodeType::Map: return "a mapping";
  }
  return "?";
}

}  // namespace

void Issues::add(const std::string& path, int mark_line, const std::string& message) {
  std::string e = (path.empty() ? std::string("<root>") : path) + ": " + message;
  if (mark_line >= 0) e += " (" + source_ + ":" + std::to_string(mark_line + 1) + ")";
  errors_.push_back(std::move(e));
}

std::string Issues::summary() const {
  if (errors_.empty()) return std::string();
  std::string s = errors_.front();
  if (errors_.size() > 1) {
    s += "; " + std::to_string(errors_.size() - 1) + " more: ";
    for (std::size_t i = 1; i < errors_.size(); ++i) {
      if (i > 1) s += "; ";
      s += errors_[i];
    }
  }
  return s;
}

bool DRange::contains(double v) const {
  if (!std::isfinite(v)) return false;
  if (lo_open ? !(v > lo) : !(v >= lo)) return false;
  if (hi_open ? !(v < hi) : !(v <= hi)) return false;
  return true;
}

std::string fmt_double(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  char buf[40];
  for (int prec = 6; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
    if (std::strtod(buf, nullptr) == v) break;
  }
  return buf;
}

std::string DRange::str() const {
  return std::string(lo_open ? "(" : "[") + fmt_double(lo) + ", " + fmt_double(hi) + (hi_open ? ")" : "]");
}

std::string IRange::str() const { return "[" + std::to_string(lo) + ", " + std::to_string(hi) + "]"; }

bool parse_int_text(const std::string& text, int64_t& out) {
  if (text.empty()) return false;
  std::size_t i = 0;
  bool neg = false;
  if (text[0] == '+' || text[0] == '-') {
    neg = text[0] == '-';
    i = 1;
  }
  int base = 10;
  if (text.size() >= i + 2 && text[i] == '0' && (text[i + 1] == 'x' || text[i + 1] == 'X')) {
    base = 16;
    i += 2;
  }
  if (i >= text.size()) return false;
  for (std::size_t j = i; j < text.size(); ++j) {
    const char c = text[j];
    const bool ok = base == 10 ? (c >= '0' && c <= '9')
                               : ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
    if (!ok) return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long mag = std::strtoull(text.c_str() + i, &end, base);
  if (errno == ERANGE || end != text.c_str() + text.size()) return false;
  constexpr unsigned long long kMaxPos = static_cast<unsigned long long>(std::numeric_limits<int64_t>::max());
  if (neg) {
    if (mag > kMaxPos + 1ull) return false;
    out = mag == kMaxPos + 1ull ? std::numeric_limits<int64_t>::min() : -static_cast<int64_t>(mag);
  } else {
    if (mag > kMaxPos) return false;
    out = static_cast<int64_t>(mag);
  }
  return true;
}

bool parse_double_text(const std::string& text, double& out) {
  if (text.empty()) return false;
  // Only [+-]digits[.digits][e[+-]digits] (also ".5", "5."): rejects inf/nan/hex floats that strtod accepts.
  std::size_t i = 0;
  if (text[i] == '+' || text[i] == '-') ++i;
  std::size_t digits = 0;
  while (i < text.size() && text[i] >= '0' && text[i] <= '9') ++i, ++digits;
  if (i < text.size() && text[i] == '.') {
    ++i;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') ++i, ++digits;
  }
  if (digits == 0) return false;
  if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
    ++i;
    if (i < text.size() && (text[i] == '+' || text[i] == '-')) ++i;
    std::size_t exp_digits = 0;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') ++i, ++exp_digits;
    if (exp_digits == 0) return false;
  }
  if (i != text.size()) return false;
  errno = 0;
  char* end = nullptr;
  const double v = std::strtod(text.c_str(), &end);
  if (end != text.c_str() + text.size() || !std::isfinite(v)) return false;
  out = v;
  return true;
}

bool parse_bool_text(const std::string& text, bool& out) {
  if (text == "true" || text == "True" || text == "TRUE") {
    out = true;
    return true;
  }
  if (text == "false" || text == "False" || text == "FALSE") {
    out = false;
    return true;
  }
  return false;
}

MapReader::MapReader(Issues& issues, YAML::Node node, std::string path, int parent_line)
    : issues_(issues), node_(std::move(node)), path_(std::move(path)), line_(parent_line) {
  present_ = node_.IsDefined();
  if (!present_) return;
  if (node_.Mark().line >= 0) line_ = node_.Mark().line;
  if (!node_.IsMap()) {
    issues_.add(path_, line_, std::string("expected a mapping, got ") + node_type_name(node_));
    return;
  }
  is_map_ = true;
  std::map<std::string, int> seen;
  for (auto it = node_.begin(); it != node_.end(); ++it) {
    const YAML::Node k = it->first;
    const int kl = k.Mark().line;
    if (!k.IsScalar()) {
      issues_.add(path_, kl, "mapping keys must be scalars");
      continue;
    }
    const std::string key = k.Scalar();
    auto [pos, inserted] = seen.emplace(key, kl);
    if (!inserted) {
      issues_.add(child_path(key), kl,
                  "duplicate key (first at " + issues_.source() + ":" + std::to_string(pos->second + 1) + ")");
      continue;
    }
    keys_.emplace_back(key, kl);
  }
}

std::string MapReader::child_path(const std::string& key) const { return path_.empty() ? key : path_ + "." + key; }

bool MapReader::has(const std::string& key) {
  known_.insert(key);
  if (!is_map_) return false;
  for (const auto& kv : keys_)
    if (kv.first == key) return true;
  return false;
}

YAML::Node MapReader::take(const std::string& key) {
  known_.insert(key);
  if (!is_map_) return YAML::Node(YAML::NodeType::Undefined);
  for (auto it = node_.begin(); it != node_.end(); ++it) {
    if (it->first.IsScalar() && it->first.Scalar() == key) return it->second;
  }
  return YAML::Node(YAML::NodeType::Undefined);
}

int MapReader::child_line(const std::string& key) {
  for (const auto& kv : keys_) {
    if (kv.first == key) {
      const YAML::Node n = take(key);
      return n.IsDefined() && n.Mark().line >= 0 ? n.Mark().line : kv.second;
    }
  }
  return line_;
}

void MapReader::finish() {
  if (finished_ || !is_map_) return;
  finished_ = true;
  std::string known_list;
  for (const auto& k : known_) {
    if (!known_list.empty()) known_list += ", ";
    known_list += k;
  }
  for (const auto& kv : keys_) {
    if (known_.count(kv.first) == 0) {
      issues_.add(child_path(kv.first), kv.second, "unknown key (allowed here: " + known_list + ")");
    }
  }
}

std::vector<std::string> MapReader::keys() const {
  std::vector<std::string> out;
  for (const auto& kv : keys_) out.push_back(kv.first);
  return out;
}

void MapReader::error(const std::string& key, const std::string& message) {
  issues_.add(child_path(key), child_line(key), message);
}

void MapReader::error_here(const std::string& message) { issues_.add(path_, line_, message); }

void MapReader::missing(const std::string& key) {
  // Only inside a mapping that exists: an absent parent is reported once by whoever required it.
  if (is_map_) issues_.add(child_path(key), line_, "missing required key");
}

bool MapReader::fetch_scalar(const std::string& key, bool required, YAML::Node& n, std::string& text, bool& quoted) {
  n = take(key);
  if (!n.IsDefined()) {
    if (required) missing(key);
    return false;
  }
  if (!n.IsScalar()) {
    error(key, std::string("expected a scalar, got ") + node_type_name(n) + (n.IsNull() ? " (write \"\" for an empty string)" : ""));
    return false;
  }
  text = n.Scalar();
  quoted = is_quoted(n);
  return true;
}

bool MapReader::i64(const std::string& key, int64_t& out, IRange range, bool required) {
  YAML::Node n;
  std::string text;
  bool quoted = false;
  if (!fetch_scalar(key, required, n, text, quoted)) return false;
  int64_t v = 0;
  if (quoted || !parse_int_text(text, v)) {
    error(key, "expected an integer (decimal or 0x hex), got " + std::string(quoted ? "the quoted string " : "") +
                   "'" + text + "'");
    return false;
  }
  if (!range.contains(v)) {
    error(key, text + " is out of range " + range.str());
    return false;
  }
  out = v;
  return true;
}

bool MapReader::i32(const std::string& key, int& out, IRange range, bool required) {
  int64_t v = 0;
  if (range.lo < std::numeric_limits<int>::min()) range.lo = std::numeric_limits<int>::min();
  if (range.hi > std::numeric_limits<int>::max()) range.hi = std::numeric_limits<int>::max();
  if (!i64(key, v, range, required)) return false;
  out = static_cast<int>(v);
  return true;
}

bool MapReader::u64(const std::string& key, uint64_t& out, bool required) {
  YAML::Node n;
  std::string text;
  bool quoted = false;
  if (!fetch_scalar(key, required, n, text, quoted)) return false;
  bool ok = !quoted && !text.empty() && text[0] != '-' && text[0] != '+';
  unsigned long long v = 0;
  if (ok) {
    const bool hex = text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X');
    for (std::size_t j = hex ? 2 : 0; j < text.size() && ok; ++j) {
      const char c = text[j];
      ok = hex ? ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) : (c >= '0' && c <= '9');
    }
    if (ok) {
      errno = 0;
      char* end = nullptr;
      v = std::strtoull(text.c_str() + (hex ? 2 : 0), &end, hex ? 16 : 10);
      ok = errno != ERANGE && end == text.c_str() + text.size();
    }
  }
  if (!ok) {
    error(key, "expected a non-negative integer, got '" + text + "'");
    return false;
  }
  out = static_cast<uint64_t>(v);
  return true;
}

bool MapReader::size(const std::string& key, std::size_t& out, IRange range, bool required) {
  int64_t v = 0;
  if (range.lo < 0) range.lo = 0;
  if (!i64(key, v, range, required)) return false;
  out = static_cast<std::size_t>(v);
  return true;
}

bool MapReader::f64(const std::string& key, double& out, DRange range, bool required) {
  YAML::Node n;
  std::string text;
  bool quoted = false;
  if (!fetch_scalar(key, required, n, text, quoted)) return false;
  double v = 0;
  if (quoted || !parse_double_text(text, v)) {
    error(key, "expected a finite number, got " + std::string(quoted ? "the quoted string " : "") + "'" + text + "'");
    return false;
  }
  if (!range.contains(v)) {
    error(key, text + " is out of range " + range.str());
    return false;
  }
  out = v;
  return true;
}

bool MapReader::boolean(const std::string& key, bool& out, bool required) {
  YAML::Node n;
  std::string text;
  bool quoted = false;
  if (!fetch_scalar(key, required, n, text, quoted)) return false;
  bool v = false;
  if (quoted || !parse_bool_text(text, v)) {
    error(key, "expected true or false, got '" + text + "'");
    return false;
  }
  out = v;
  return true;
}

bool MapReader::str(const std::string& key, std::string& out, bool required) {
  YAML::Node n;
  std::string text;
  bool quoted = false;
  if (!fetch_scalar(key, required, n, text, quoted)) return false;
  out = text;
  return true;
}

namespace {
bool scalar_double(const YAML::Node& n, double& v) {
  return n.IsScalar() && !is_quoted(n) && parse_double_text(n.Scalar(), v);
}
}  // namespace

bool MapReader::vec3(const std::string& key, Eigen::Vector3d& out, bool required) {
  const YAML::Node n = take(key);
  if (!n.IsDefined()) {
    if (required) missing(key);
    return false;
  }
  if (!n.IsSequence() || n.size() != 3) {
    error(key, "expected a list of 3 numbers");
    return false;
  }
  Eigen::Vector3d v;
  for (std::size_t i = 0; i < 3; ++i) {
    if (!scalar_double(n[i], v[static_cast<Eigen::Index>(i)])) {
      error(key, "element " + std::to_string(i) + " is not a finite number");
      return false;
    }
  }
  out = v;
  return true;
}

bool MapReader::mat3(const std::string& key, Eigen::Matrix3d& out, bool required) {
  const YAML::Node n = take(key);
  if (!n.IsDefined()) {
    if (required) missing(key);
    return false;
  }
  if (!n.IsSequence() || n.size() != 3) {
    error(key, "expected 3 rows of 3 numbers ([[r00, r01, r02], [..], [..]])");
    return false;
  }
  Eigen::Matrix3d m;
  for (std::size_t r = 0; r < 3; ++r) {
    const YAML::Node row = n[r];
    if (!row.IsSequence() || row.size() != 3) {
      error(key, "row " + std::to_string(r) + " is not a list of 3 numbers");
      return false;
    }
    for (std::size_t c = 0; c < 3; ++c) {
      double v = 0;
      if (!scalar_double(row[c], v)) {
        error(key, "element [" + std::to_string(r) + "][" + std::to_string(c) + "] is not a finite number");
        return false;
      }
      m(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = v;
    }
  }
  out = m;
  return true;
}

bool MapReader::f64_list(const std::string& key, std::vector<double>& out, bool required) {
  const YAML::Node n = take(key);
  if (!n.IsDefined()) {
    if (required) missing(key);
    return false;
  }
  if (!n.IsSequence()) {
    error(key, std::string("expected a list of numbers, got ") + node_type_name(n));
    return false;
  }
  std::vector<double> v(n.size());
  for (std::size_t i = 0; i < n.size(); ++i) {
    if (!scalar_double(n[i], v[i])) {
      error(key, "element " + std::to_string(i) + " is not a finite number");
      return false;
    }
  }
  out = std::move(v);
  return true;
}

bool MapReader::i32_list(const std::string& key, std::vector<int>& out, IRange range, bool required) {
  const YAML::Node n = take(key);
  if (!n.IsDefined()) {
    if (required) missing(key);
    return false;
  }
  if (!n.IsSequence()) {
    error(key, std::string("expected a list of integers, got ") + node_type_name(n));
    return false;
  }
  std::vector<int> v;
  for (std::size_t i = 0; i < n.size(); ++i) {
    int64_t x = 0;
    if (!n[i].IsScalar() || is_quoted(n[i]) || !parse_int_text(n[i].Scalar(), x)) {
      error(key, "element " + std::to_string(i) + " is not an integer");
      return false;
    }
    if (!range.contains(x)) {
      error(key, "element " + std::to_string(i) + " = " + std::to_string(x) + " is out of range " + range.str());
      return false;
    }
    v.push_back(static_cast<int>(x));
  }
  out = std::move(v);
  return true;
}

bool MapReader::str_list(const std::string& key, std::vector<std::string>& out, bool required) {
  const YAML::Node n = take(key);
  if (!n.IsDefined()) {
    if (required) missing(key);
    return false;
  }
  if (!n.IsSequence()) {
    error(key, std::string("expected a list of strings, got ") + node_type_name(n));
    return false;
  }
  std::vector<std::string> v;
  for (std::size_t i = 0; i < n.size(); ++i) {
    if (!n[i].IsScalar()) {
      error(key, "element " + std::to_string(i) + " is not a string");
      return false;
    }
    v.push_back(n[i].Scalar());
  }
  out = std::move(v);
  return true;
}

MapReader MapReader::sub(const std::string& key, bool required) {
  YAML::Node n = take(key);
  if (!n.IsDefined() && required) missing(key);
  return MapReader(issues_, n, child_path(key), child_line(key));
}

}  // namespace cfg
}  // namespace rig
