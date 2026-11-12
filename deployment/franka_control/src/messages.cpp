#include "franka_control/messages.h"

namespace franka_control {
namespace msg {

const char* schemaOf(const Any& m) {
  return std::visit([](const auto& v) { return std::decay_t<decltype(v)>::kSchema; }, m);
}

bool isC3Request(const Any& m) {
  return std::holds_alternative<SetWaypoints>(m) || std::holds_alternative<Heartbeat>(m) ||
         std::holds_alternative<Start>(m) || std::holds_alternative<Stop>(m) || std::holds_alternative<Home>(m) ||
         std::holds_alternative<GetState>(m);
}

}  // namespace msg
}  // namespace franka_control
