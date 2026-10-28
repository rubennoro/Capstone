// SPDX-License-Identifier: MIT
// SHA-256 per FIPS 180-4 (sections 4.1.2, 4.2.2, 5.3.3, 6.2). Verified against the FIPS 180-2 appendix B vectors in
// tests/package/test_sha256.cpp.
#include "rig/package/sha256.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <vector>

namespace rig {
namespace {

constexpr std::array<uint32_t, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr uint32_t rotr(uint32_t x, int n) noexcept { return (x >> n) | (x << (32 - n)); }

}  // namespace

Sha256::Sha256() noexcept { reset(); }

void Sha256::reset() noexcept {
  h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  buf_len_ = 0;
  total_bytes_ = 0;
}

void Sha256::block(const uint8_t* p) noexcept {
  uint32_t w[64];
  for (int t = 0; t < 16; ++t) {
    w[t] = (uint32_t{p[4 * t]} << 24) | (uint32_t{p[4 * t + 1]} << 16) | (uint32_t{p[4 * t + 2]} << 8) |
           uint32_t{p[4 * t + 3]};
  }
  for (int t = 16; t < 64; ++t) {
    const uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
    const uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
    w[t] = w[t - 16] + s0 + w[t - 7] + s1;
  }
  uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
  for (int t = 0; t < 64; ++t) {
    const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = h + S1 + ch + kK[static_cast<std::size_t>(t)] + w[t];
    const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = S0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h_[0] += a;
  h_[1] += b;
  h_[2] += c;
  h_[3] += d;
  h_[4] += e;
  h_[5] += f;
  h_[6] += g;
  h_[7] += h;
}

void Sha256::update(const void* data, std::size_t n) noexcept {
  const auto* p = static_cast<const uint8_t*>(data);
  total_bytes_ += n;
  if (buf_len_ > 0) {
    const std::size_t take = std::min(n, buf_.size() - buf_len_);
    std::memcpy(buf_.data() + buf_len_, p, take);
    buf_len_ += take;
    p += take;
    n -= take;
    if (buf_len_ < buf_.size()) return;
    block(buf_.data());
    buf_len_ = 0;
  }
  while (n >= 64) {
    block(p);
    p += 64;
    n -= 64;
  }
  if (n > 0) {
    std::memcpy(buf_.data(), p, n);
    buf_len_ = n;
  }
}

std::array<uint8_t, 32> Sha256::finish() noexcept {
  const uint64_t bits = total_bytes_ * 8;
  const uint8_t pad80 = 0x80;
  update(&pad80, 1);
  const uint8_t zero = 0;
  while (buf_len_ != 56) update(&zero, 1);
  uint8_t len[8];
  for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
  update(len, 8);
  std::array<uint8_t, 32> out{};
  for (std::size_t i = 0; i < 8; ++i) {
    out[4 * i] = static_cast<uint8_t>(h_[i] >> 24);
    out[4 * i + 1] = static_cast<uint8_t>(h_[i] >> 16);
    out[4 * i + 2] = static_cast<uint8_t>(h_[i] >> 8);
    out[4 * i + 3] = static_cast<uint8_t>(h_[i]);
  }
  return out;
}

std::string to_hex(const uint8_t* data, std::size_t n) {
  static const char* kDigits = "0123456789abcdef";
  std::string s(2 * n, '0');
  for (std::size_t i = 0; i < n; ++i) {
    s[2 * i] = kDigits[data[i] >> 4];
    s[2 * i + 1] = kDigits[data[i] & 0xF];
  }
  return s;
}

std::string sha256_hex(std::string_view bytes) {
  Sha256 h;
  h.update(bytes);
  const auto d = h.finish();
  return to_hex(d.data(), d.size());
}

Result<std::string> sha256_file_hex(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    const int e = errno;
    if (e == ENOENT) return not_found(path.string() + ": no such file");
    return io_error(path.string() + ": cannot open: " + std::strerror(e));
  }
  Sha256 h;
  std::vector<char> buf(1 << 16);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize got = in.gcount();
    if (got > 0) h.update(buf.data(), static_cast<std::size_t>(got));
  }
  if (in.bad()) return io_error(path.string() + ": read error");
  const auto d = h.finish();
  return to_hex(d.data(), d.size());
}

}  // namespace rig
