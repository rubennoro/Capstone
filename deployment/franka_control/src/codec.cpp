// The C++ half of contracts/contracts/codec.py + message.py + fields.py + c2_state.py + c3_commands.py + common.py.
// The decoder mirrors the Python validator's check order exactly, so every broken example file fails here with the
// same field and problem (test_codec.cpp proves it against contracts/examples/).
#include "franka_control/codec.h"

#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

#include <Eigen/Dense>
#include <msgpack.hpp>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the wire format is little-endian; this code memcpy's it");

namespace franka_control {
namespace {

using msgpack::object;
namespace mt = msgpack::type;

// Every schema registered in contracts/contracts/registry.py (all at v1). A known name with another version is
// schema_version; an unknown name is unknown_schema, exactly as Python's lookup().
const std::map<std::string, int>& knownSchemas() {
  static const std::map<std::string, int> known = {
      {"c0.frame", 1},       {"c0.time_sync", 1},  {"c0.time_sync.reply", 1}, {"c1.get_frames", 1},
      {"c1.get_health", 1},  {"c1.frames", 1},     {"c1.health", 1},          {"c2.state", 1},
      {"c3.set_waypoints", 1}, {"c3.set_waypoints.reply", 1}, {"c3.heartbeat", 1}, {"c3.start", 1},
      {"c3.stop", 1},        {"c3.home", 1},       {"c3.get_state", 1},       {"c4.config", 1},
      {"c4.reset", 1},       {"c4.prompt", 1},     {"c4.infer", 1},           {"c4.config.reply", 1},
      {"c4.chunk", 1},       {"ack", 1},           {"error", 1}};
  return known;
}

const char* kModes[] = {"idle", "running", "holding", "stopped", "homing", "fault"};

ContractError fail(const std::string& problem, const std::string& detail, const std::string& field = "") {
  return ContractError("?", field, problem, detail);
}

const char* pyType(const object& o) {
  switch (o.type) {
    case mt::NIL: return "NoneType";
    case mt::BOOLEAN: return "bool";
    case mt::POSITIVE_INTEGER:
    case mt::NEGATIVE_INTEGER: return "int";
    case mt::FLOAT32:
    case mt::FLOAT64: return "float";
    case mt::STR: return "str";
    case mt::BIN: return "bytes";
    case mt::ARRAY: return "list";
    case mt::MAP: return "dict";
    default: return "ExtType";
  }
}

std::string asString(const object& o) { return std::string(o.via.str.ptr, o.via.str.size); }

std::string repr(const std::string& s) { return "'" + s + "'"; }

std::string numText(double v) {
  std::ostringstream ss;
  ss << v;
  return ss.str();
}

bool validUtf8(const char* p, size_t n) {
  size_t i = 0;
  const auto* s = reinterpret_cast<const unsigned char*>(p);
  while (i < n) {
    const unsigned char c = s[i];
    size_t extra = 0;
    uint32_t cp = 0;
    if (c < 0x80) {
      ++i;
      continue;
    } else if ((c & 0xE0) == 0xC0) {
      extra = 1;
      cp = c & 0x1F;
    } else if ((c & 0xF0) == 0xE0) {
      extra = 2;
      cp = c & 0x0F;
    } else if ((c & 0xF8) == 0xF0) {
      extra = 3;
      cp = c & 0x07;
    } else {
      return false;
    }
    for (size_t k = 1; k <= extra; ++k) {
      if (i + k >= n || (s[i + k] & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (s[i + k] & 0x3F);
    }
    if ((extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) || (extra == 3 && cp < 0x10000) || cp > 0x10FFFF ||
        (cp >= 0xD800 && cp <= 0xDFFF)) {
      return false;
    }
    i += extra + 1;
  }
  return true;
}

// What msgpack.unpackb(raw=False, strict_map_key=True) refuses beyond the byte syntax: map keys other than str or
// bin, and str that is not UTF-8.
void checkWire(const object& o) {
  if (o.type == mt::STR && !validUtf8(o.via.str.ptr, o.via.str.size)) {
    throw fail("not_msgpack", "not msgpack (UnicodeDecodeError: invalid utf-8 in a str)");
  }
  if (o.type == mt::ARRAY) {
    for (uint32_t i = 0; i < o.via.array.size; ++i) checkWire(o.via.array.ptr[i]);
  } else if (o.type == mt::MAP) {
    for (uint32_t i = 0; i < o.via.map.size; ++i) {
      const object& k = o.via.map.ptr[i].key;
      if (k.type != mt::STR && k.type != mt::BIN) {
        throw fail("not_msgpack", std::string("not msgpack (ValueError: ") + pyType(k) +
                                      " is not allowed for map key when strict_map_key=True)");
      }
      checkWire(k);
      checkWire(o.via.map.ptr[i].val);
    }
  }
}

// A msgpack map as Python's dict sees it: keys in order of first appearance, the last value wins.
struct Entry {
  std::string key;        // str(key) for the error texts
  bool is_str = true;
  const object* val = nullptr;
};

std::vector<Entry> entriesOf(const object& map) {
  std::vector<Entry> out;
  for (uint32_t i = 0; i < map.via.map.size; ++i) {
    const object& k = map.via.map.ptr[i].key;
    Entry e;
    e.is_str = k.type == mt::STR;
    e.key = e.is_str ? asString(k) : "b'" + std::string(k.via.bin.ptr, k.via.bin.size) + "'";
    e.val = &map.via.map.ptr[i].val;
    bool replaced = false;
    for (auto& existing : out) {
      if (existing.key == e.key && existing.is_str == e.is_str) {
        existing.val = e.val;
        replaced = true;
        break;
      }
    }
    if (!replaced) out.push_back(e);
  }
  return out;
}

const object* findStr(const std::vector<Entry>& entries, const std::string& key) {
  for (const auto& e : entries) {
    if (e.is_str && e.key == key) return e.val;
  }
  return nullptr;
}

// Missing keys (field order), then unknown keys (map order), then each field through field() (Python's _build).
class Fields {
 public:
  Fields(const std::vector<Entry>& entries, std::string label, std::vector<std::string> names, bool with_schema)
      : entries_(entries), label_(std::move(label)) {
    for (const auto& n : names) {
      if (!findStr(entries_, n)) throw ContractError(label_, n, "missing", "field is missing");
    }
    std::set<std::string> allowed(names.begin(), names.end());
    if (with_schema) allowed.insert("schema");
    for (const auto& e : entries_) {
      if (!e.is_str || !allowed.count(e.key)) throw ContractError(label_, e.key, "unknown_field", "unknown field");
    }
  }

  template <class F>
  auto field(const std::string& name, F read) const {
    try {
      return read(*findStr(entries_, name));
    } catch (const ContractError& e) {
      throw e.within(label_, name);
    }
  }

  ContractError value(const std::string& field, const std::string& detail) const {
    return ContractError(label_, field, "value", detail);
  }

 private:
  const std::vector<Entry>& entries_;
  std::string label_;
};

int64_t readInt64(const object& o) {
  if (o.type == mt::NEGATIVE_INTEGER) return o.via.i64;
  if (o.type == mt::POSITIVE_INTEGER) {
    if (o.via.u64 > static_cast<uint64_t>(INT64_MAX)) {
      throw fail("value", std::to_string(o.via.u64) + " is out of the int64 range");
    }
    return static_cast<int64_t>(o.via.u64);
  }
  throw fail("type", std::string("expected int64, got ") + pyType(o));
}

uint64_t readUint64(const object& o) {
  if (o.type == mt::POSITIVE_INTEGER) return o.via.u64;
  if (o.type == mt::NEGATIVE_INTEGER) throw fail("value", std::to_string(o.via.i64) + " is out of the uint64 range");
  throw fail("type", std::string("expected uint64, got ") + pyType(o));
}

double readFloat64(const object& o, std::optional<double> min = std::nullopt) {
  if (o.type != mt::FLOAT64 && o.type != mt::FLOAT32) {
    throw fail("type", std::string("expected float64, got ") + pyType(o));
  }
  const double v = o.via.f64;
  if (std::isnan(v)) throw fail("value", "nan is not a number");
  if (min && v < *min) throw fail("value", numText(v) + " is below the minimum " + numText(*min));
  return v;
}

bool readBool(const object& o) {
  if (o.type != mt::BOOLEAN) throw fail("type", std::string("expected bool, got ") + pyType(o));
  return o.via.boolean;
}

std::string readStr(const object& o) {
  if (o.type != mt::STR) throw fail("type", std::string("expected str, got ") + pyType(o));
  return asString(o);
}

struct ArrayView {
  std::string dtype;
  std::vector<uint64_t> shape;
  const char* data = nullptr;
  size_t bytes = 0;
};

size_t itemSize(const std::string& dtype) {
  if (dtype == "uint8") return 1;
  if (dtype == "float32") return 4;
  return 8;   // int64, uint64, float64
}

std::string shapeText(const std::vector<int64_t>& shape) {   // -1 = any length ("N"), as Python's _shape_text
  std::string s = "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += shape[i] < 0 ? "N" : std::to_string(shape[i]);
  }
  if (shape.size() == 1) s += ",";
  return s + ")";
}

std::string shapeText(const std::vector<uint64_t>& shape) {
  std::vector<int64_t> s(shape.begin(), shape.end());
  return shapeText(s);
}

// contracts.codec.unpack_array, check for check.
ArrayView unpackArray(const object& o) {
  static const char* kKeys[] = {"__nd__", "dtype", "shape", "data"};
  if (o.type != mt::MAP) throw fail("type", std::string("expected an array map, got ") + pyType(o));
  const std::vector<Entry> entries = entriesOf(o);
  bool keys_ok = entries.size() == 4;
  for (const char* k : kKeys) keys_ok = keys_ok && findStr(entries, k) != nullptr;
  if (!keys_ok) {
    std::string got;
    for (size_t i = 0; i < entries.size(); ++i) got += (i ? ", " : "") + entries[i].key;
    throw fail("type", "an array map has exactly the keys __nd__, dtype, shape, data; got " + got);
  }
  const object& nd = *findStr(entries, "__nd__");
  if (nd.type != mt::BOOLEAN || !nd.via.boolean) throw fail("type", "an array map needs __nd__: true");
  const object& dtype = *findStr(entries, "dtype");
  ArrayView a;
  if (dtype.type == mt::STR) a.dtype = asString(dtype);
  static const std::set<std::string> kDtypes = {"uint8", "int64", "uint64", "float32", "float64"};
  if (dtype.type != mt::STR || !kDtypes.count(a.dtype)) {
    throw fail("dtype", "array dtype " + (dtype.type == mt::STR ? repr(a.dtype) : std::string(pyType(dtype))) +
                            " is not one of uint8, int64, uint64, float32, float64");
  }
  const object& shape = *findStr(entries, "shape");
  bool shape_ok = shape.type == mt::ARRAY;
  if (shape_ok) {
    for (uint32_t i = 0; i < shape.via.array.size; ++i) {
      const object& d = shape.via.array.ptr[i];
      if (d.type != mt::POSITIVE_INTEGER) {
        shape_ok = false;
        break;
      }
      a.shape.push_back(d.via.u64);
    }
  }
  if (!shape_ok) throw fail("shape", "array shape must be a list of non-negative integers");
  const object& data = *findStr(entries, "data");
  if (data.type != mt::BIN) throw fail("type", std::string("array data must be bin, got ") + pyType(data));
  long double count = 1;
  for (uint64_t d : a.shape) count *= static_cast<long double>(d);
  const long double need = count * static_cast<long double>(itemSize(a.dtype));
  if (static_cast<long double>(data.via.bin.size) != need) {
    throw fail("shape", "array data has " + std::to_string(data.via.bin.size) + " bytes; shape " +
                            shapeText(a.shape) + " " + a.dtype + " needs " +
                            std::to_string(static_cast<unsigned long long>(need)));
  }
  a.data = data.via.bin.ptr;
  a.bytes = data.via.bin.size;
  return a;
}

// ArraySpec.check: dtype, then shape (-1 = any length).
ArrayView readArray(const object& o, const std::string& dtype, const std::vector<int64_t>& shape) {
  ArrayView a = unpackArray(o);
  if (a.dtype != dtype) throw fail("dtype", "dtype " + a.dtype + ", expected " + dtype);
  bool ok = a.shape.size() == shape.size();
  for (size_t i = 0; ok && i < shape.size(); ++i) {
    ok = shape[i] < 0 || a.shape[i] == static_cast<uint64_t>(shape[i]);
  }
  if (!ok) throw fail("shape", "shape " + shapeText(a.shape) + ", expected " + shapeText(shape));
  return a;
}

template <size_t N>
std::array<double, N> fixedDoubles(const object& o, const std::vector<int64_t>& shape) {
  const ArrayView a = readArray(o, "float64", shape);
  std::array<double, N> out{};
  std::memcpy(out.data(), a.data, N * sizeof(double));
  return out;
}

bool allFinite(const double* v, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) return false;
  }
  return true;
}

// c2_state.rigid_problem.
std::optional<std::string> rigidProblem(const std::array<double, 16>& m, double tol) {
  if (!allFinite(m.data(), 16)) return std::string("not finite");
  if (!(m[12] == 0.0 && m[13] == 0.0 && m[14] == 0.0 && m[15] == 1.0)) {
    return "bottom row [" + numText(m[12]) + ", " + numText(m[13]) + ", " + numText(m[14]) + ", " + numText(m[15]) +
           "], expected [0, 0, 0, 1] (column-major matrix sent without transposing?)";
  }
  Eigen::Matrix3d R;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) R(r, c) = m[static_cast<size_t>(4 * r + c)];
  }
  const double err = (R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
  if (err > tol) return "rotation not orthonormal: max |R^T R - I| = " + numText(err) + " > " + numText(tol);
  if (R.determinant() <= 0.0) return std::string("rotation has determinant <= 0 (a reflection)");
  return std::nullopt;
}

msg::GateCounters readGate(const object& o) {
  if (o.type != mt::MAP) throw fail("type", std::string("expected a map (GateCounters), got ") + pyType(o));
  const std::vector<Entry> entries = entriesOf(o);
  const Fields f(entries, "GateCounters", {"accepted", "shortened", "pulled_back", "rejected"}, false);
  msg::GateCounters g;
  g.accepted = f.field("accepted", readUint64);
  g.shortened = f.field("shortened", readUint64);
  g.pulled_back = f.field("pulled_back", readUint64);
  g.rejected = f.field("rejected", readUint64);
  return g;
}

msg::ArmState decodeArmState(const std::vector<Entry>& entries, double rotation_tol) {
  const Fields f(entries, msg::ArmState::kSchema,
                 {"t_ns", "seq", "O_T_TCP", "O_T_TCP_target", "q", "dq", "gripper_width_m", "mode", "error", "gate",
                  "loop_late"},
                 true);
  msg::ArmState s;
  s.t_ns = f.field("t_ns", readInt64);
  s.seq = f.field("seq", readUint64);
  s.O_T_TCP = f.field("O_T_TCP", [](const object& o) { return fixedDoubles<16>(o, {4, 4}); });
  s.O_T_TCP_target = f.field("O_T_TCP_target", [](const object& o) { return fixedDoubles<16>(o, {4, 4}); });
  s.q = f.field("q", [](const object& o) { return fixedDoubles<7>(o, {7}); });
  s.dq = f.field("dq", [](const object& o) { return fixedDoubles<7>(o, {7}); });
  s.gripper_width_m = f.field("gripper_width_m", [](const object& o) { return readFloat64(o, 0.0); });
  s.mode = f.field("mode", [](const object& o) {
    std::string m = readStr(o);
    for (const char* k : kModes) {
      if (m == k) return m;
    }
    throw fail("value", repr(m) + " is not one of idle, running, holding, stopped, homing, fault");
  });
  s.error = f.field("error", readStr);
  s.gate = f.field("gate", readGate);
  s.loop_late = f.field("loop_late", readUint64);
  // check_values
  if (s.gate.shortened > s.gate.accepted) {
    throw f.value("gate.shortened",
                  std::to_string(s.gate.shortened) + " > accepted " + std::to_string(s.gate.accepted));
  }
  if (s.gate.pulled_back > s.gate.accepted) {
    throw f.value("gate.pulled_back",
                  std::to_string(s.gate.pulled_back) + " > accepted " + std::to_string(s.gate.accepted));
  }
  if (!allFinite(s.O_T_TCP.data(), 16)) throw f.value("O_T_TCP", "not finite");
  if (!allFinite(s.O_T_TCP_target.data(), 16)) throw f.value("O_T_TCP_target", "not finite");
  if (!allFinite(s.q.data(), 7)) throw f.value("q", "not finite");
  if (!allFinite(s.dq.data(), 7)) throw f.value("dq", "not finite");
  if (!std::isfinite(s.gripper_width_m)) throw f.value("gripper_width_m", "not finite");
  if (auto p = rigidProblem(s.O_T_TCP, rotation_tol)) throw f.value("O_T_TCP", "not a rigid transform: " + *p);
  if (auto p = rigidProblem(s.O_T_TCP_target, rotation_tol)) {
    throw f.value("O_T_TCP_target", "not a rigid transform: " + *p);
  }
  return s;
}

msg::SetWaypoints decodeSetWaypoints(const std::vector<Entry>& entries, const ContractLimits& limits) {
  const Fields f(entries, msg::SetWaypoints::kSchema, {"chunk_id", "t_ns", "O_T_TCP", "gripper_width_m"}, true);
  msg::SetWaypoints w;
  w.chunk_id = f.field("chunk_id", readUint64);
  const ArrayView t = f.field("t_ns", [](const object& o) { return readArray(o, "int64", {-1}); });
  const ArrayView poses = f.field("O_T_TCP", [](const object& o) { return readArray(o, "float64", {-1, 4, 4}); });
  const ArrayView widths = f.field("gripper_width_m", [](const object& o) { return readArray(o, "float64", {-1}); });
  w.t_ns.resize(t.shape[0]);
  if (!w.t_ns.empty()) std::memcpy(w.t_ns.data(), t.data, t.bytes);
  w.O_T_TCP.resize(poses.shape[0]);
  if (!w.O_T_TCP.empty()) std::memcpy(w.O_T_TCP.data(), poses.data, poses.bytes);
  w.gripper_width_m.resize(widths.shape[0]);
  if (!w.gripper_width_m.empty()) std::memcpy(w.gripper_width_m.data(), widths.data, widths.bytes);
  // check_values
  const size_t n = w.t_ns.size();
  if (n < 1 || n > limits.max_waypoints) {
    throw f.value("t_ns", std::to_string(n) + " waypoints, expected 1 to " + std::to_string(limits.max_waypoints));
  }
  if (w.O_T_TCP.size() != n) {
    throw f.value("O_T_TCP", std::to_string(w.O_T_TCP.size()) + " poses for " + std::to_string(n) + " times");
  }
  if (w.gripper_width_m.size() != n) {
    throw f.value("gripper_width_m",
                  std::to_string(w.gripper_width_m.size()) + " widths for " + std::to_string(n) + " times");
  }
  for (size_t i = 1; i < n; ++i) {
    if (w.t_ns[i] <= w.t_ns[i - 1]) throw f.value("t_ns", "times not strictly increasing");
  }
  for (const auto& m : w.O_T_TCP) {
    if (!allFinite(m.data(), 16)) throw f.value("O_T_TCP", "not finite");
  }
  if (!allFinite(w.gripper_width_m.data(), n)) throw f.value("gripper_width_m", "not finite");
  for (size_t i = 0; i < n; ++i) {
    if (auto p = rigidProblem(w.O_T_TCP[i], limits.rotation_tol)) {
      throw f.value("O_T_TCP", "not a rigid transform at [" + std::to_string(i) + "]: " + *p);
    }
  }
  for (double width : w.gripper_width_m) {
    if (width < 0.0) throw f.value("gripper_width_m", "negative width: " + numText(width));
  }
  return w;
}

msg::WaypointIssue readIssue(const object& o) {
  const std::vector<Entry> entries = entriesOf(o);
  const Fields f(entries, "WaypointIssue", {"index", "reason"}, false);
  msg::WaypointIssue issue;
  issue.index = f.field("index", readUint64);
  issue.reason = f.field("reason", readStr);
  return issue;
}

msg::SetWaypointsReply decodeReply(const std::vector<Entry>& entries) {
  const Fields f(entries, msg::SetWaypointsReply::kSchema, {"chunk_id", "accepted", "shortened", "skipped", "rejected"},
                 true);
  msg::SetWaypointsReply r;
  r.chunk_id = f.field("chunk_id", readUint64);
  r.accepted = f.field("accepted", readUint64);
  r.shortened = f.field("shortened", readUint64);
  r.skipped = f.field("skipped", readUint64);
  r.rejected = f.field("rejected", [](const object& o) {
    if (o.type != mt::ARRAY) throw fail("type", std::string("expected a list of WaypointIssue, got ") + pyType(o));
    std::vector<msg::WaypointIssue> items;
    for (uint32_t i = 0; i < o.via.array.size; ++i) {
      const object& item = o.via.array.ptr[i];
      const std::string at = "[" + std::to_string(i) + "]";
      if (item.type != mt::MAP) {
        throw fail("type", std::string("expected a map (WaypointIssue), got ") + pyType(item), at);
      }
      try {
        items.push_back(readIssue(item));
      } catch (const ContractError& e) {
        throw e.within("?", at);
      }
    }
    return items;
  });
  if (r.shortened > r.accepted) {
    throw f.value("shortened", std::to_string(r.shortened) + " > accepted " + std::to_string(r.accepted));
  }
  std::set<uint64_t> seen;
  for (size_t i = 0; i < r.rejected.size(); ++i) {
    if (!seen.insert(r.rejected[i].index).second) {
      throw f.value("rejected[" + std::to_string(i) + "].index",
                    "index " + std::to_string(r.rejected[i].index) + " listed twice");
    }
  }
  return r;
}

msg::Ack decodeAck(const std::vector<Entry>& entries) {
  const Fields f(entries, msg::Ack::kSchema, {"ok", "reason"}, true);
  msg::Ack a;
  a.ok = f.field("ok", readBool);
  a.reason = f.field("reason", readStr);
  return a;
}

msg::ErrorReply decodeError(const std::vector<Entry>& entries) {
  const Fields f(entries, msg::ErrorReply::kSchema, {"request", "error", "field"}, true);
  msg::ErrorReply e;
  e.request = f.field("request", readStr);
  e.error = f.field("error", readStr);
  e.field = f.field("field", readStr);
  return e;
}

template <class M>
M decodeEmpty(const std::vector<Entry>& entries) {
  const Fields f(entries, M::kSchema, {}, true);
  (void)f;
  return M{};
}

// SchemaName.parse: "^([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)/v([1-9][0-9]*)$".
bool parseSchema(const std::string& s, std::string& name, int& version) {
  const auto slash = s.rfind("/v");
  if (slash == std::string::npos || slash == 0) return false;
  name = s.substr(0, slash);
  const std::string ver = s.substr(slash + 2);
  if (ver.empty() || ver[0] < '1' || ver[0] > '9' || ver.size() > 9) return false;
  for (char c : ver) {
    if (c < '0' || c > '9') return false;
  }
  bool start = true;
  for (char c : name) {
    if (start) {
      if (c < 'a' || c > 'z') return false;
      start = false;
    } else if (c == '.') {
      start = true;
    } else if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) {
      return false;
    }
  }
  if (start) return false;
  version = std::stoi(ver);
  return true;
}

// ---- encoding (pack_obj(msg.to_dict())) ----

class Packer {
 public:
  void map(uint32_t n) { pk_.pack_map(n); }
  void list(uint32_t n) { pk_.pack_array(n); }
  void str(const std::string& s) {
    pk_.pack_str(static_cast<uint32_t>(s.size()));
    pk_.pack_str_body(s.data(), static_cast<uint32_t>(s.size()));
  }
  void i64(int64_t v) { pk_.pack_int64(v); }
  void u64(uint64_t v) { pk_.pack_uint64(v); }
  // Always msgpack float64 (0xcb + 8 bytes big-endian). msgpack-cxx's pack_double writes an integral double
  // (0.0, 1.0) as an integer, which the contract refuses for float64 fields.
  void f64(double v) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    char out[9];
    out[0] = static_cast<char>(0xcb);
    for (int i = 0; i < 8; ++i) out[1 + i] = static_cast<char>((bits >> (8 * (7 - i))) & 0xff);
    buf_.write(out, sizeof(out));
  }
  void boolean(bool v) { v ? pk_.pack_true() : pk_.pack_false(); }
  void ndarray(const std::string& dtype, const std::vector<uint64_t>& shape, const void* data, size_t bytes) {
    map(4);
    str("__nd__");
    boolean(true);
    str("dtype");
    str(dtype);
    str("shape");
    list(static_cast<uint32_t>(shape.size()));
    for (uint64_t d : shape) u64(d);
    str("data");
    pk_.pack_bin(static_cast<uint32_t>(bytes));
    pk_.pack_bin_body(static_cast<const char*>(data), static_cast<uint32_t>(bytes));
  }
  std::string bytes() const { return std::string(buf_.data(), buf_.size()); }

 private:
  msgpack::sbuffer buf_;
  msgpack::packer<msgpack::sbuffer> pk_{buf_};
};

void pack(Packer& p, const msg::ArmState& s) {
  p.map(12);
  p.str("schema");
  p.str(msg::ArmState::kSchema);
  p.str("t_ns");
  p.i64(s.t_ns);
  p.str("seq");
  p.u64(s.seq);
  p.str("O_T_TCP");
  p.ndarray("float64", {4, 4}, s.O_T_TCP.data(), 16 * sizeof(double));
  p.str("O_T_TCP_target");
  p.ndarray("float64", {4, 4}, s.O_T_TCP_target.data(), 16 * sizeof(double));
  p.str("q");
  p.ndarray("float64", {7}, s.q.data(), 7 * sizeof(double));
  p.str("dq");
  p.ndarray("float64", {7}, s.dq.data(), 7 * sizeof(double));
  p.str("gripper_width_m");
  p.f64(s.gripper_width_m);
  p.str("mode");
  p.str(s.mode);
  p.str("error");
  p.str(s.error);
  p.str("gate");
  p.map(4);
  p.str("accepted");
  p.u64(s.gate.accepted);
  p.str("shortened");
  p.u64(s.gate.shortened);
  p.str("pulled_back");
  p.u64(s.gate.pulled_back);
  p.str("rejected");
  p.u64(s.gate.rejected);
  p.str("loop_late");
  p.u64(s.loop_late);
}

void pack(Packer& p, const msg::SetWaypoints& w) {
  p.map(5);
  p.str("schema");
  p.str(msg::SetWaypoints::kSchema);
  p.str("chunk_id");
  p.u64(w.chunk_id);
  p.str("t_ns");
  p.ndarray("int64", {w.t_ns.size()}, w.t_ns.data(), w.t_ns.size() * sizeof(int64_t));
  p.str("O_T_TCP");
  p.ndarray("float64", {w.O_T_TCP.size(), 4, 4}, w.O_T_TCP.data(), w.O_T_TCP.size() * 16 * sizeof(double));
  p.str("gripper_width_m");
  p.ndarray("float64", {w.gripper_width_m.size()}, w.gripper_width_m.data(),
            w.gripper_width_m.size() * sizeof(double));
}

void pack(Packer& p, const msg::SetWaypointsReply& r) {
  p.map(6);
  p.str("schema");
  p.str(msg::SetWaypointsReply::kSchema);
  p.str("chunk_id");
  p.u64(r.chunk_id);
  p.str("accepted");
  p.u64(r.accepted);
  p.str("shortened");
  p.u64(r.shortened);
  p.str("skipped");
  p.u64(r.skipped);
  p.str("rejected");
  p.list(static_cast<uint32_t>(r.rejected.size()));
  for (const auto& issue : r.rejected) {
    p.map(2);
    p.str("index");
    p.u64(issue.index);
    p.str("reason");
    p.str(issue.reason);
  }
}

void pack(Packer& p, const msg::Ack& a) {
  p.map(3);
  p.str("schema");
  p.str(msg::Ack::kSchema);
  p.str("ok");
  p.boolean(a.ok);
  p.str("reason");
  p.str(a.reason);
}

void pack(Packer& p, const msg::ErrorReply& e) {
  p.map(4);
  p.str("schema");
  p.str(msg::ErrorReply::kSchema);
  p.str("request");
  p.str(e.request);
  p.str("error");
  p.str(e.error);
  p.str("field");
  p.str(e.field);
}

template <class M>
void pack(Packer& p, const M&) {   // the field-less requests
  p.map(1);
  p.str("schema");
  p.str(M::kSchema);
}

}  // namespace

Codec::Codec(const ContractLimits& limits) : limits_(limits) {}

msg::Any Codec::decode(std::string_view bytes) const {
  msgpack::object_handle oh;
  std::size_t offset = 0;
  try {
    oh = msgpack::unpack(bytes.data(), bytes.size(), offset);
  } catch (const std::exception& e) {
    throw fail("not_msgpack", std::string("not msgpack (") + e.what() + ")");
  }
  if (offset != bytes.size()) throw fail("not_msgpack", "not msgpack (ExtraData: unpack(b) received extra data.)");
  const object& root = oh.get();
  checkWire(root);
  if (root.type != mt::MAP) throw fail("not_a_map", std::string("expected a map, got ") + pyType(root));
  const std::vector<Entry> entries = entriesOf(root);
  const object* schema = findStr(entries, "schema");
  if (!schema) throw fail("no_schema", "no 'schema' key");
  if (schema->type != mt::STR) {
    throw ContractError("?", "", "unknown_schema",
                        std::string("schema must be a string like 'c2.state/v1', got ") + pyType(*schema));
  }
  const std::string label = asString(*schema);
  std::string name;
  int version = 0;
  if (!parseSchema(label, name, version)) {
    throw ContractError(label, "", "unknown_schema",
                        "malformed schema " + repr(label) + ": expected '<name>/v<N>', e.g. 'c2.state/v1'");
  }
  const auto known = knownSchemas().find(name);
  if (known == knownSchemas().end()) {
    throw ContractError(label, "", "unknown_schema", "unknown message name '" + name + "'");
  }
  if (version != known->second) {
    throw ContractError(label, "", "schema_version",
                        "version v" + std::to_string(version) + " not supported (this side knows v" +
                            std::to_string(known->second) + ")");
  }
  if (name == "c2.state") return decodeArmState(entries, limits_.rotation_tol);
  if (name == "c3.set_waypoints") return decodeSetWaypoints(entries, limits_);
  if (name == "c3.set_waypoints.reply") return decodeReply(entries);
  if (name == "c3.heartbeat") return decodeEmpty<msg::Heartbeat>(entries);
  if (name == "c3.start") return decodeEmpty<msg::Start>(entries);
  if (name == "c3.stop") return decodeEmpty<msg::Stop>(entries);
  if (name == "c3.home") return decodeEmpty<msg::Home>(entries);
  if (name == "c3.get_state") return decodeEmpty<msg::GetState>(entries);
  if (name == "ack") return decodeAck(entries);
  if (name == "error") return decodeError(entries);
  throw ContractError(label, "", "unexpected_message",
                      "franka_control reads C2/C3 messages only, got " + label);
}

std::string Codec::encode(const msg::Any& m) const {
  Packer p;
  std::visit([&p](const auto& v) { pack(p, v); }, m);
  std::string out = p.bytes();
  try {
    (void)decode(out);
  } catch (const ContractError& e) {
    throw std::logic_error(std::string("franka_control bug: the message it was about to send breaks its contract: ") +
                           e.what());
  }
  return out;
}

}  // namespace franka_control
