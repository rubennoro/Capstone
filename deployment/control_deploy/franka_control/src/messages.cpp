// from rnoro/deployment:deployment/franka_control/src/messages.cpp @ 91b178b
// changes: Hello is a C3 request, HelloReply is not (control_deploy/STRUCTURE.md 3.2).
#include "franka_control/messages.h"

namespace franka_control {
namespace msg {

const char* schemaOf(const Any& m) {
  return std::visit([](const auto& v) { return std::decay_t<decltype(v)>::kSchema; }, m);
}

bool isC3Request(const Any& m) {
  return std::holds_alternative<SetWaypoints>(m) || std::holds_alternative<Heartbeat>(m) ||
         std::holds_alternative<Start>(m) || std::holds_alternative<Stop>(m) || std::holds_alternative<Home>(m) ||
         std::holds_alternative<GetState>(m) || std::holds_alternative<Hello>(m);
}

}  // namespace msg
}  // namespace franka_control
