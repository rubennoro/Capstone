// The C2/C3 messages, wire-shaped (contracts STRUCTURE.md 4.1, 4.4, 4.5): matrices as row-major std::array<16>,
// fields in the contract's table order. The Supervisor converts to Eigen.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace franka_control {
namespace msg {

struct GateCounters {
  uint64_t accepted = 0;
  uint64_t shortened = 0;
  uint64_t pulled_back = 0;
  uint64_t rejected = 0;
};

struct ArmState {                                   // c2.state/v1
  static constexpr const char* kSchema = "c2.state/v1";
  int64_t t_ns = 0;
  uint64_t seq = 0;
  std::array<double, 16> O_T_TCP{};
  std::array<double, 16> O_T_TCP_target{};
  std::array<double, 7> q{};
  std::array<double, 7> dq{};
  double gripper_width_m = 0.0;
  std::string mode;
  std::string error;
  GateCounters gate;
  uint64_t loop_late = 0;
};

struct SetWaypoints {                               // c3.set_waypoints/v1
  static constexpr const char* kSchema = "c3.set_waypoints/v1";
  uint64_t chunk_id = 0;
  std::vector<int64_t> t_ns;
  std::vector<std::array<double, 16>> O_T_TCP;     // each row-major
  std::vector<double> gripper_width_m;
};

struct WaypointIssue {
  uint64_t index = 0;
  std::string reason;
};

struct SetWaypointsReply {                          // c3.set_waypoints.reply/v1
  static constexpr const char* kSchema = "c3.set_waypoints.reply/v1";
  uint64_t chunk_id = 0;
  uint64_t accepted = 0;
  uint64_t shortened = 0;
  uint64_t skipped = 0;
  std::vector<WaypointIssue> rejected;
};

struct Heartbeat { static constexpr const char* kSchema = "c3.heartbeat/v1"; };
struct Start { static constexpr const char* kSchema = "c3.start/v1"; };
struct Stop { static constexpr const char* kSchema = "c3.stop/v1"; };
struct Home { static constexpr const char* kSchema = "c3.home/v1"; };
struct GetState { static constexpr const char* kSchema = "c3.get_state/v1"; };

struct Ack {                                        // ack/v1
  static constexpr const char* kSchema = "ack/v1";
  bool ok = false;
  std::string reason;
};

struct ErrorReply {                                 // error/v1
  static constexpr const char* kSchema = "error/v1";
  std::string request;
  std::string error;
  std::string field;
};

using Any = std::variant<ArmState, SetWaypoints, SetWaypointsReply, Heartbeat, Start, Stop, Home, GetState, Ack,
                         ErrorReply>;

// The schema of whichever message `m` holds.
const char* schemaOf(const Any& m);
// True for the requests franka_control answers (C3).
bool isC3Request(const Any& m);

}  // namespace msg
}  // namespace franka_control
