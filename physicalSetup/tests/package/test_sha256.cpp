// SPDX-License-Identifier: MIT
// T3: SHA-256 against the FIPS 180-2 appendix B / NIST CAVS vectors.
#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "rig/package/sha256.hpp"

namespace rig {
namespace {

TEST(Sha256, FipsVectors) {
  EXPECT_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  Sha256 h;
  const std::string chunk(1000, 'a');
  for (int i = 0; i < 1000; ++i) h.update(chunk);  // one million 'a'
  const auto d = h.finish();
  EXPECT_EQ(to_hex(d.data(), d.size()), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, IncrementalEqualsOneShotAtEveryBoundary) {
  std::string msg;
  for (int i = 0; i < 300; ++i) msg.push_back(static_cast<char>(i * 7));
  for (std::size_t len : {0u, 1u, 55u, 56u, 63u, 64u, 65u, 119u, 120u, 128u, 300u}) {
    const std::string m = msg.substr(0, len);
    for (std::size_t split = 0; split <= len; split += 13) {
      Sha256 h;
      h.update(m.substr(0, split));
      h.update(m.substr(split));
      const auto d = h.finish();
      EXPECT_EQ(to_hex(d.data(), d.size()), sha256_hex(m)) << len << "/" << split;
    }
  }
}

TEST(Sha256, File) {
  const std::string p = std::string(RIG_TEST_TMP_ROOT) + "/sha256_file.txt";
  { std::ofstream(p) << "abc"; }
  auto r = sha256_file_hex(p);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(*r, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(sha256_file_hex(p + ".missing").status().code(), StatusCode::kNotFound);
}

}  // namespace
}  // namespace rig
