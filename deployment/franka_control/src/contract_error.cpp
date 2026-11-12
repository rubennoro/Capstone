#include "franka_control/contract_error.h"

namespace franka_control {

ContractError::ContractError(std::string schema, std::string field, std::string problem, std::string detail)
    : std::runtime_error(text(schema, field, detail)),
      schema_(std::move(schema)),
      field_(std::move(field)),
      problem_(std::move(problem)),
      detail_(std::move(detail)) {}

std::string ContractError::text(const std::string& schema, const std::string& field, const std::string& detail) {
  if (!field.empty()) return schema + ": field '" + field + "': " + detail;
  return schema + ": " + detail;
}

std::string ContractError::joinField(const std::string& prefix, const std::string& field) {
  if (prefix.empty()) return field;
  if (field.empty()) return prefix;
  if (field[0] == '[') return prefix + field;
  return prefix + "." + field;
}

ContractError ContractError::within(const std::string& schema, const std::string& prefix) const {
  return ContractError(schema, joinField(prefix, field_), problem_, detail_);
}

}  // namespace franka_control
