// SPDX-License-Identifier: MIT
// Register constants versus the vendored ST header (third_party/st_lsm6dsox/lsm6dsox_reg.h), the datasheet values
// (DS12814 Rev 4, typed in again here independently), and the exact configuration byte sequence.
#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>

#include "rig/drivers/imu/lsm6dsox_driver.hpp"
#include "rig/drivers/imu/lsm6dsox_regs.hpp"

namespace rig {
namespace {

using namespace lsm6dsox;  // NOLINT

std::map<std::string, long> st_defines() {
  std::ifstream in(std::string(RIG_SOURCE_DIR) + "/third_party/st_lsm6dsox/lsm6dsox_reg.h");
  EXPECT_TRUE(in.good());
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string s = ss.str();
  std::map<std::string, long> m;
  const std::regex def(R"(#define\s+LSM6DSOX_([A-Z0-9_]+)\s+0x([0-9A-Fa-f]+)U?)");
  for (auto it = std::sregex_iterator(s.begin(), s.end(), def); it != std::sregex_iterator(); ++it)
    m[(*it)[1]] = std::stol((*it)[2], nullptr, 16);
  const std::regex en(R"(\bLSM6DSOX_([A-Za-z0-9_]+)\s*=\s*(0x[0-9A-Fa-f]+|\d+))");
  for (auto it = std::sregex_iterator(s.begin(), s.end(), en); it != std::sregex_iterator(); ++it)
    m["enum:" + (*it)[1].str()] = std::stol((*it)[2], nullptr, 0);
  return m;
}

TEST(Lsm6dsoxRegs, AddressesMatchStHeader) {
  const auto m = st_defines();
  const std::map<std::string, uint8_t> ours = {
      {"FUNC_CFG_ACCESS", kRegFuncCfgAccess}, {"FIFO_CTRL1", kRegFifoCtrl1},   {"FIFO_CTRL2", kRegFifoCtrl2},
      {"FIFO_CTRL3", kRegFifoCtrl3},           {"FIFO_CTRL4", kRegFifoCtrl4},   {"WHO_AM_I", kRegWhoAmI},
      {"CTRL1_XL", kRegCtrl1Xl},               {"CTRL2_G", kRegCtrl2G},         {"CTRL3_C", kRegCtrl3C},
      {"CTRL4_C", kRegCtrl4C},                 {"CTRL5_C", kRegCtrl5C},         {"CTRL6_C", kRegCtrl6C},
      {"CTRL7_G", kRegCtrl7G},                 {"CTRL8_XL", kRegCtrl8Xl},       {"CTRL9_XL", kRegCtrl9Xl},
      {"CTRL10_C", kRegCtrl10C},               {"STATUS_REG", kRegStatusReg},   {"OUT_TEMP_L", kRegOutTempL},
      {"FIFO_STATUS1", kRegFifoStatus1},       {"FIFO_STATUS2", kRegFifoStatus2}, {"TIMESTAMP0", kRegTimestamp0},
      {"TIMESTAMP2", kRegTimestamp2},          {"INTERNAL_FREQ_FINE", kRegInternalFreqFine},
      {"FIFO_DATA_OUT_TAG", kRegFifoDataOutTag}, {"FIFO_DATA_OUT_X_L", kRegFifoDataOutXL},
      {"FIFO_DATA_OUT_Z_H", kRegFifoDataOutZH}, {"ID", kWhoAmIValue}};
  for (const auto& [name, v] : ours) {
    ASSERT_TRUE(m.count(name)) << name;
    EXPECT_EQ(m.at(name), v) << name;
  }
  EXPECT_EQ(m.at("I2C_ADD_L") >> 1, kI2cAddrSa0Low);  // 8-bit read address 0xD5 -> 7-bit 0x6A
  EXPECT_EQ(m.at("I2C_ADD_H") >> 1, kI2cAddrSa0High);
}

TEST(Lsm6dsoxRegs, CodesMatchStEnums) {
  const auto m = st_defines();
  EXPECT_EQ(m.at("enum:2g"), 0);
  EXPECT_EQ(m.at("enum:16g"), 1);
  EXPECT_EQ(m.at("enum:4g"), 2);
  EXPECT_EQ(m.at("enum:8g"), 3);
  for (const AccelFs& f : kAccelFsTable) EXPECT_EQ(m.at("enum:" + std::to_string(f.g) + "g"), f.code);
  for (const GyroFs& f : kGyroFsTable) EXPECT_EQ(m.at("enum:" + std::to_string(f.dps) + "dps"), f.code3);
  EXPECT_EQ(m.at("enum:XL_ODR_833Hz"), 7);
  EXPECT_EQ(m.at("enum:GY_ODR_833Hz"), 7);
  EXPECT_EQ(m.at("enum:XL_BATCHED_AT_833Hz"), 7);
  EXPECT_EQ(m.at("enum:GY_BATCHED_AT_833Hz"), 7);
  EXPECT_EQ(m.at("enum:XL_ODR_6667Hz"), 10);
  EXPECT_EQ(m.at("enum:XL_ODR_12Hz5"), 1);
  EXPECT_EQ(m.at("enum:STREAM_MODE"), kFifoModeContinuous);
  EXPECT_EQ(m.at("enum:BYPASS_MODE"), kFifoModeBypass);
  EXPECT_EQ(m.at("enum:DEC_1"), dec_ts_code(1));
  EXPECT_EQ(m.at("enum:DEC_8"), dec_ts_code(8));
  EXPECT_EQ(m.at("enum:DEC_32"), dec_ts_code(32));
  EXPECT_EQ(m.at("enum:TEMP_BATCHED_AT_12Hz5"), odr_t_code(12.5));
  EXPECT_EQ(m.at("enum:TEMP_BATCHED_AT_52Hz"), odr_t_code(52));
  EXPECT_EQ(m.at("enum:TEMP_BATCHED_AT_1Hz6"), odr_t_code(1.6));
  EXPECT_EQ(m.at("enum:GYRO_NC_TAG"), kTagGyroNC);  // the next enumerators are implicit: XL 2, TEMP 3, TS 4
  EXPECT_EQ(kTagAccelNC, 2);
  EXPECT_EQ(kTagTemperature, 3);
  EXPECT_EQ(kTagTimestamp, 4);
  EXPECT_EQ(kTagCfgChange, 5);
  EXPECT_EQ(m.at("enum:SENSORHUB_NACK_TAG"), 0x19);
}

TEST(Lsm6dsoxRegs, DatasheetSensitivities) {
  // DS12814 Table 2 (LA_So, G_So).
  const std::map<int, double> acc = {{2, 0.061}, {4, 0.122}, {8, 0.244}, {16, 0.488}};
  const std::map<int, double> gyr = {{125, 4.375}, {250, 8.75}, {500, 17.5}, {1000, 35.0}, {2000, 70.0}};
  for (const auto& [g, s] : acc) EXPECT_DOUBLE_EQ(accel_fs(g)->mg_per_lsb, s) << g;
  for (const auto& [d, s] : gyr) EXPECT_DOUBLE_EQ(gyro_fs(d)->mdps_per_lsb, s) << d;
  EXPECT_FALSE(accel_fs(3).has_value());
  EXPECT_FALSE(gyro_fs(245).has_value());
}

TEST(Lsm6dsoxRegs, OdrLookupAcceptsBothSpellings) {
  EXPECT_EQ(odr_entry(833)->code, 0x7);
  EXPECT_EQ(odr_entry(416)->code, 0x6);
  EXPECT_EQ(odr_entry(417)->code, 0x6);
  EXPECT_EQ(odr_entry(1666)->code, 0x8);
  EXPECT_EQ(odr_entry(1667)->code, 0x8);
  EXPECT_EQ(odr_entry(3332)->code, 0x9);
  EXPECT_EQ(odr_entry(12.5)->code, 0x1);
  EXPECT_FALSE(odr_entry(800).has_value());
  EXPECT_FALSE(odr_entry(1000).has_value());
}

TEST(Lsm6dsoxRegs, TagLayoutAndHelpers) {
  // TAG_SENSOR[7:3], TAG_CNT[2:1], TAG_PARITY[0] (DS Table 194).
  EXPECT_EQ(tag_sensor(0x0B), kTagGyroNC);  // 00001 01 1
  EXPECT_EQ(tag_cnt(0x0B), 1);
  EXPECT_EQ(tag_parity(0x0B), 1);
  EXPECT_EQ(tag_sensor(0x26), kTagTimestamp);  // 00100 11 0
  EXPECT_EQ(tag_cnt(0x26), 3);
  EXPECT_EQ(fifo_level(0xFF, 0x01), 511);
  EXPECT_EQ(fifo_level(0x00, 0xFE), 512);  // flags do not leak into the level
  EXPECT_EQ(fifo_level(0x34, 0x00), 0x34);
}

TEST(Lsm6dsoxRegs, TrimmedTick) {
  EXPECT_DOUBLE_EQ(trimmed_tick_ns(0), 25000.0);
  EXPECT_NEAR(trimmed_tick_ns(10), 25000.0 / 1.015, 1e-9);
  EXPECT_NEAR(trimmed_tick_ns(-128), 25000.0 / (1.0 - 0.192), 1e-9);
  // Linux linearization 25000 - 37.5 * val agrees to first order.
  EXPECT_NEAR(trimmed_tick_ns(4), 25000.0 - 37.5 * 4, 1.0);
}

TEST(Lsm6dsoxConfig, DefaultSequenceExactBytes) {
  ImuConfig c;  // PLAN T1 defaults: 833 Hz, +-8 g, +-2000 dps, WTM 256, timestamp every slot, temp 12.5 Hz
  auto seq = lsm6dsox_config_sequence(c);
  ASSERT_TRUE(seq.ok()) << seq.status().to_string();
  const std::vector<std::pair<uint8_t, uint8_t>> want = {
      {0x12, 0x44},  // CTRL3_C: BDU | IF_INC
      {0x18, 0xE2},  // CTRL9_XL: DEN defaults | I3C_disable
      {0x07, 0x00},  // FIFO_CTRL1: WTM[7:0] = 256 & 0xff
      {0x08, 0x01},  // FIFO_CTRL2: WTM8 = 1
      {0x09, 0x77},  // FIFO_CTRL3: BDR_GY = BDR_XL = 833 Hz
      {0x10, 0x7C},  // CTRL1_XL: 833 Hz, FS 11 = +-8 g
      {0x11, 0x7C},  // CTRL2_G: 833 Hz, FS 110 = +-2000 dps
      {0x19, 0x20},  // CTRL10_C: TIMESTAMP_EN
      {0x0A, 0x66},  // FIFO_CTRL4: DEC_TS 01, ODR_T 10 (12.5 Hz), continuous 110
  };
  ASSERT_EQ(seq->size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    EXPECT_EQ((*seq)[i].reg, want[i].first) << i;
    EXPECT_EQ((*seq)[i].value, want[i].second) << (*seq)[i].name;
  }
}

TEST(Lsm6dsoxConfig, EveryFullScaleEncoding) {
  const std::map<int, uint8_t> acc = {{2, 0x70}, {4, 0x78}, {8, 0x7C}, {16, 0x74}};
  const std::map<int, uint8_t> gyr = {{125, 0x72}, {250, 0x70}, {500, 0x74}, {1000, 0x78}, {2000, 0x7C}};
  for (const auto& [g, v] : acc) {
    ImuConfig c;
    c.accel_fs_g = g;
    auto seq = lsm6dsox_config_sequence(c);
    ASSERT_TRUE(seq.ok());
    for (const auto& w : *seq) {
      if (w.reg == kRegCtrl1Xl) {
        EXPECT_EQ(w.value, v) << g;
      }
    }
  }
  for (const auto& [d, v] : gyr) {
    ImuConfig c;
    c.gyro_fs_dps = d;
    auto seq = lsm6dsox_config_sequence(c);
    ASSERT_TRUE(seq.ok());
    for (const auto& w : *seq) {
      if (w.reg == kRegCtrl2G) {
        EXPECT_EQ(w.value, v) << d;
      }
    }
  }
}

TEST(Lsm6dsoxConfig, RejectsInvalidValues) {
  auto bad = [](auto mutate, const char* field) {
    ImuConfig c;
    mutate(c);
    auto r = lsm6dsox_config_sequence(c);
    ASSERT_FALSE(r.ok()) << field;
    EXPECT_EQ(r.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(r.status().message().find(field), std::string::npos) << r.status().message();
  };
  bad([](ImuConfig& c) { c.odr_hz = 1000; }, "odr_hz");
  bad([](ImuConfig& c) { c.accel_fs_g = 6; }, "accel_fs_g");
  bad([](ImuConfig& c) { c.gyro_fs_dps = 4000; }, "gyro_fs_dps");
  bad([](ImuConfig& c) { c.fifo_watermark_words = 512; }, "fifo_watermark_words");
  bad([](ImuConfig& c) { c.fifo_watermark_words = 0; }, "fifo_watermark_words");
  bad([](ImuConfig& c) { c.timestamp_decimation = 4; }, "timestamp_decimation");
  bad([](ImuConfig& c) { c.temperature_batch_hz = 10; }, "temperature_batch_hz");
  bad([](ImuConfig& c) { c.fifo_burst_words = 0; }, "fifo_burst_words");
}

}  // namespace
}  // namespace rig
