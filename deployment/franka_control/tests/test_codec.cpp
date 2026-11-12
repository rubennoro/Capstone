// The C++ codec against contracts/examples/ (contracts STRUCTURE.md section 8): good C3 requests decode with their
// values, broken ones are refused with the same field and problem, good C2 / reply / ack / error examples re-encode
// byte for byte.
#include <gtest/gtest.h>

#include <regex>

#include <msgpack.hpp>

#include "franka_control/codec.h"
#include "franka_control/pose_math.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

struct Example {
  std::string file, schema, field, problem;
  bool valid = false;
  YAML::Node values;
};

std::vector<Example> loadIndex() {
  const YAML::Node index = YAML::LoadFile((test::examplesDir() / "index.yaml").string());
  std::vector<Example> out;
  for (const auto& e : index["examples"]) {
    Example ex;
    ex.file = e["file"].as<std::string>();
    ex.schema = e["schema"].as<std::string>();
    ex.valid = e["valid"].as<bool>();
    if (!ex.valid) {
      ex.field = e["field"].as<std::string>();
      ex.problem = e["problem"].as<std::string>();
    }
    if (e["values"]) ex.values = e["values"];
    out.push_back(ex);
  }
  return out;
}

Codec codec() { return Codec(ContractLimits::load(test::limitsFile())); }

std::string bytesOf(const Example& e) { return test::readFile(test::examplesDir() / e.file); }

// A value path from index.yaml ("seq", "gate.pulled_back", "O_T_TCP[7][0][3]", "rejected[0].reason") as text.
// Numbers are compared as doubles, strings and bools as text.
struct Value {
  bool is_number = false;
  double number = 0.0;
  std::string text;
};

Value valueAt(const msg::Any& m, const std::string& path) {
  std::smatch match;
  auto num = [](double d) { return Value{true, d, ""}; };
  auto txt = [](const std::string& s) { return Value{false, 0.0, s}; };
  auto idx = [&](const std::string& p, const std::string& name, std::vector<int>& ix) {
    if (p.rfind(name, 0) != 0) return false;
    const std::string rest = p.substr(name.size());
    if (!rest.empty() && rest[0] != '[') return false;
    const std::regex re(R"(\[(\d+)\])");
    for (auto it = std::sregex_iterator(rest.begin(), rest.end(), re); it != std::sregex_iterator(); ++it) {
      ix.push_back(std::stoi((*it)[1]));
    }
    return true;
  };
  std::vector<int> ix;
  if (const auto* s = std::get_if<msg::ArmState>(&m)) {
    if (path == "seq") return num(static_cast<double>(s->seq));
    if (path == "t_ns") return num(static_cast<double>(s->t_ns));
    if (path == "mode") return txt(s->mode);
    if (path == "error") return txt(s->error);
    if (path == "gripper_width_m") return num(s->gripper_width_m);
    if (path == "loop_late") return num(static_cast<double>(s->loop_late));
    if (path == "gate.accepted") return num(static_cast<double>(s->gate.accepted));
    if (path == "gate.shortened") return num(static_cast<double>(s->gate.shortened));
    if (path == "gate.pulled_back") return num(static_cast<double>(s->gate.pulled_back));
    if (path == "gate.rejected") return num(static_cast<double>(s->gate.rejected));
    if (idx(path, "O_T_TCP_target", ix)) return num(s->O_T_TCP_target[static_cast<size_t>(4 * ix[0] + ix[1])]);
    ix.clear();
    if (idx(path, "O_T_TCP", ix)) return num(s->O_T_TCP[static_cast<size_t>(4 * ix[0] + ix[1])]);
    ix.clear();
    if (idx(path, "dq", ix)) return num(s->dq[static_cast<size_t>(ix[0])]);
    ix.clear();
    if (idx(path, "q", ix)) return num(s->q[static_cast<size_t>(ix[0])]);
  }
  if (const auto* w = std::get_if<msg::SetWaypoints>(&m)) {
    if (path == "chunk_id") return num(static_cast<double>(w->chunk_id));
    if (idx(path, "t_ns", ix)) return num(static_cast<double>(w->t_ns[static_cast<size_t>(ix[0])]));
    ix.clear();
    if (idx(path, "O_T_TCP", ix)) {
      return num(w->O_T_TCP[static_cast<size_t>(ix[0])][static_cast<size_t>(4 * ix[1] + ix[2])]);
    }
    ix.clear();
    if (idx(path, "gripper_width_m", ix)) return num(w->gripper_width_m[static_cast<size_t>(ix[0])]);
  }
  if (const auto* r = std::get_if<msg::SetWaypointsReply>(&m)) {
    if (path == "chunk_id") return num(static_cast<double>(r->chunk_id));
    if (path == "accepted") return num(static_cast<double>(r->accepted));
    if (path == "shortened") return num(static_cast<double>(r->shortened));
    if (path == "skipped") return num(static_cast<double>(r->skipped));
    if (std::regex_match(path, match, std::regex(R"(rejected\[(\d+)\]\.(index|reason))"))) {
      const auto& issue = r->rejected[static_cast<size_t>(std::stoi(match[1]))];
      return match[2] == "index" ? num(static_cast<double>(issue.index)) : txt(issue.reason);
    }
  }
  if (const auto* a = std::get_if<msg::Ack>(&m)) {
    if (path == "ok") return txt(a->ok ? "true" : "false");
    if (path == "reason") return txt(a->reason);
  }
  if (const auto* e = std::get_if<msg::ErrorReply>(&m)) {
    if (path == "request") return txt(e->request);
    if (path == "error") return txt(e->error);
    if (path == "field") return txt(e->field);
  }
  ADD_FAILURE() << "no value path " << path << " in " << msg::schemaOf(m);
  return txt("");
}

bool isC2OrC3(const Example& e) {
  return e.file.rfind("c2/", 0) == 0 || e.file.rfind("c3/", 0) == 0;
}

TEST(Codec, TheExampleFilesExist) {
  const auto examples = loadIndex();
  EXPECT_GE(examples.size(), 60u);
  for (const auto& e : examples) EXPECT_TRUE(std::filesystem::exists(test::examplesDir() / e.file)) << e.file;
}

TEST(Codec, EveryGoodExampleDecodesWithItsValues) {
  const Codec c = codec();
  int checked = 0;
  for (const auto& e : loadIndex()) {
    if (!e.valid || !isC2OrC3(e)) continue;
    SCOPED_TRACE(e.file);
    msg::Any m;
    ASSERT_NO_THROW(m = c.decode(bytesOf(e)));
    EXPECT_EQ(msg::schemaOf(m), e.schema);
    if (e.values) {
      for (auto it = e.values.begin(); it != e.values.end(); ++it) {
        const std::string path = it->first.as<std::string>();
        const Value got = valueAt(m, path);
        if (got.is_number) {
          EXPECT_DOUBLE_EQ(got.number, it->second.as<double>()) << path;
        } else {
          EXPECT_EQ(got.text, it->second.as<std::string>()) << path;
        }
      }
    }
    ++checked;
  }
  EXPECT_GT(checked, 10);
}

TEST(Codec, EveryBrokenExampleIsRefusedWithTheSameFieldAndProblem) {
  const Codec c = codec();
  int checked = 0;
  for (const auto& e : loadIndex()) {
    if (e.valid || !isC2OrC3(e)) continue;
    SCOPED_TRACE(e.file);
    try {
      c.decode(bytesOf(e));
      ADD_FAILURE() << "decoded, but the example is broken";
    } catch (const ContractError& err) {
      EXPECT_EQ(err.field(), e.field) << err.what();
      EXPECT_EQ(err.problem(), e.problem) << err.what();
      EXPECT_EQ(err.schema(), e.schema) << err.what();
    }
    ++checked;
  }
  EXPECT_GT(checked, 30);
}

TEST(Codec, GoodRepliesAndStatesReEncodeByteForByte) {
  const Codec c = codec();
  int checked = 0;
  for (const auto& e : loadIndex()) {
    if (!e.valid || !isC2OrC3(e)) continue;
    SCOPED_TRACE(e.file);
    const std::string bytes = bytesOf(e);
    const std::string again = c.encode(c.decode(bytes));
    EXPECT_EQ(again, bytes);
    ++checked;
  }
  EXPECT_GT(checked, 10);
}

TEST(Codec, LimitsComeFromTheContractsFile) {
  const ContractLimits l = ContractLimits::load(test::limitsFile());
  EXPECT_EQ(l.max_waypoints, 64u);
  EXPECT_DOUBLE_EQ(l.rotation_tol, 1e-6);
  EXPECT_DOUBLE_EQ(l.gripper_width_max_m, 0.08);
}

TEST(Codec, RowMajorWireAndEigenAgree) {
  const Codec c = codec();
  const Pose pose = test::makePose({0.45, 0.1, 0.3}, test::rotvec({M_PI, 0.1, 0.2}));
  msg::SetWaypoints w;
  w.chunk_id = 5;
  w.t_ns = {100};
  w.O_T_TCP = {rowMajorFromPose(pose)};
  w.gripper_width_m = {0.04};
  const auto back = std::get<msg::SetWaypoints>(c.decode(c.encode(w)));
  EXPECT_TRUE(poseFromRowMajor(back.O_T_TCP[0]).matrix().isApprox(pose.matrix(), 1e-12));
  EXPECT_DOUBLE_EQ(back.O_T_TCP[0][3], 0.45);   // translation in the last column, row-major
  // the column-major (libfranka/Eigen) layout sent without transposing is refused by the rigid check
  std::array<double, 16> col{};
  Eigen::Map<Eigen::Matrix4d>(col.data()) = pose.matrix();
  w.O_T_TCP = {col};
  try {
    c.encode(w);
    ADD_FAILURE() << "a transposed pose was encoded";
  } catch (const std::logic_error&) {
  }
}

TEST(Codec, TheEncoderWritesTheContractFieldOrder) {
  const Codec c = codec();
  msg::SetWaypointsReply r;
  r.chunk_id = 9;
  r.accepted = 3;
  r.shortened = 1;
  r.skipped = 2;
  r.rejected = {};
  const std::string bytes = c.encode(r);
  const msgpack::object_handle oh = msgpack::unpack(bytes.data(), bytes.size());
  const msgpack::object& o = oh.get();
  ASSERT_EQ(o.type, msgpack::type::MAP);
  std::vector<std::string> keys;
  for (uint32_t i = 0; i < o.via.map.size; ++i) keys.push_back(o.via.map.ptr[i].key.as<std::string>());
  EXPECT_EQ(keys, (std::vector<std::string>{"schema", "chunk_id", "accepted", "shortened", "skipped", "rejected"}));
  EXPECT_EQ(o.via.map.ptr[3].val.type, msgpack::type::POSITIVE_INTEGER);   // shortened is a count
}

TEST(Codec, MessagesOfOtherContractsAreUnexpected) {
  const Codec c = codec();
  msgpack::sbuffer buf;
  msgpack::packer<msgpack::sbuffer> pk(buf);
  pk.pack_map(1);
  pk.pack(std::string("schema"));
  pk.pack(std::string("c4.reset/v1"));
  try {
    c.decode(std::string(buf.data(), buf.size()));
    ADD_FAILURE();
  } catch (const ContractError& e) {
    EXPECT_EQ(e.problem(), "unexpected_message");
  }
}

}  // namespace
}  // namespace franka_control
