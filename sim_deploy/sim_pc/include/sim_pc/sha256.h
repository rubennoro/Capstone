// sim_deploy/sim_pc (E4): SHA-256 as lowercase hex (the model_sha256 check).
#pragma once

#include <filesystem>
#include <string>

namespace sim_pc {

std::string sha256Hex(const std::string& data);
std::string sha256FileHex(const std::filesystem::path& file);   // throws std::runtime_error if unreadable

}  // namespace sim_pc
