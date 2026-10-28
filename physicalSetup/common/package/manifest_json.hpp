// SPDX-License-Identifier: MIT
// Private JSON helpers shared by manifest.cpp and session.cpp.
#pragma once

#include <initializer_list>
#include <string>

#include <nlohmann/json.hpp>

#include "rig/package/package.hpp"

namespace rig {
namespace pkgjson {
nlohmann::json time_to_json(TimeNs t);  ///< kInvalidTime -> null
Status time_from_json(const nlohmann::json& j, const char* key, TimeNs& out, const std::string& ctx);
Status str_from_json(const nlohmann::json& j, const char* key, std::string& out, const std::string& ctx);
/// kDataLoss for a non-object or a key outside `allowed`.
Status check_keys(const nlohmann::json& j, std::initializer_list<const char*> allowed, const std::string& ctx);
}  // namespace pkgjson

std::string manifest_to_json_text(const Manifest& m);
Result<Manifest> manifest_from_json_text(const std::string& text, const std::string& ctx);

}  // namespace rig
