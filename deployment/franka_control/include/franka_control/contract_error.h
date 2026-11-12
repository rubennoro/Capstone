// A message broke its contract (contracts STRUCTURE.md 6, the same fields as the Python ContractError).
#pragma once

#include <stdexcept>
#include <string>

namespace franka_control {

class ContractError : public std::runtime_error {
 public:
  // problem: one of not_msgpack, not_a_map, no_schema, unknown_schema, schema_version, missing, unknown_field,
  // type, dtype, shape, value, unexpected_message.
  ContractError(std::string schema, std::string field, std::string problem, std::string detail);

  const std::string& schema() const { return schema_; }
  const std::string& field() const { return field_; }
  const std::string& problem() const { return problem_; }
  const std::string& detail() const { return detail_; }

  // The same problem reported for another schema and/or under a field prefix (as Python's within()).
  ContractError within(const std::string& schema, const std::string& prefix = "") const;

  static std::string joinField(const std::string& prefix, const std::string& field);

 private:
  static std::string text(const std::string& schema, const std::string& field, const std::string& detail);
  std::string schema_, field_, problem_, detail_;
};

}  // namespace franka_control
