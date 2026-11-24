// from rnoro/deployment:deployment/franka_control/include/franka_control/codec.h @ 91b178b
// changes: + static knownSchemas(); the codec knows exactly SHARED_SCHEMAS, hello included (control_deploy/STRUCTURE.md
// 3.2).
// The C2/C3 wire format: msgpack maps, arrays as {"__nd__", "dtype", "shape", "data"} (deployment/STRUCTURE.md 3.3).
// Decoding runs the contract's checks in the Python validator's exact order; encoding writes the canonical form.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "franka_control/contract_error.h"
#include "franka_control/contract_limits.h"
#include "franka_control/messages.h"

namespace franka_control {

class Codec {                                      // pure
 public:
  explicit Codec(const ContractLimits& limits);    // max_waypoints, rotation_tol
  msg::Any decode(std::string_view bytes) const;   // throws ContractError (schema, field, problem, detail)
  std::string encode(const msg::Any& m) const;     // canonical form; validates first (a bug throws, never sends)
  // Every schema this codec knows as "name/vN", sorted as whole strings by byte order (= SHARED_SCHEMAS, sent in
  // hello.schemas).
  static std::vector<std::string> knownSchemas();

 private:
  ContractLimits limits_;
};

}  // namespace franka_control
