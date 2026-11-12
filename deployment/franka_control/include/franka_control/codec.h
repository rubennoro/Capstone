// The C2/C3 wire format: msgpack maps, arrays as {"__nd__", "dtype", "shape", "data"} (contracts STRUCTURE.md 5).
// Decoding runs the contract's checks in the Python validator's exact order; encoding writes the canonical form.
#pragma once

#include <string>
#include <string_view>

#include "franka_control/contract_error.h"
#include "franka_control/contract_limits.h"
#include "franka_control/messages.h"

namespace franka_control {

class Codec {                                      // pure
 public:
  explicit Codec(const ContractLimits& limits);    // max_waypoints, rotation_tol
  msg::Any decode(std::string_view bytes) const;   // throws ContractError (schema, field, problem, detail)
  std::string encode(const msg::Any& m) const;     // canonical form; validates first (a bug throws, never sends)

 private:
  ContractLimits limits_;
};

}  // namespace franka_control
