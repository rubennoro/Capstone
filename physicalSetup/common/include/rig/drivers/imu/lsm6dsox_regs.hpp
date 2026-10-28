// SPDX-License-Identifier: MIT
/// @file rig/drivers/imu/lsm6dsox_regs.hpp
/// @brief LSM6DSOX register map, bit fields, codes and scale factors used by the rig (T1).
///
/// Every constant cites its source:
/// - "DS" = ST datasheet DS12814 Rev 4 (June 2024), the LSM6DSOX datasheet. (The plan's "DS13345" is not the
///   LSM6DSOX document number; DS12814 is.) Section / table numbers refer to that revision.
/// - "ST" = ST's BSD-3 driver lsm6dsox-pid, vendored at physicalSetup/third_party/st_lsm6dsox/lsm6dsox_reg.h
///   (commit 2e86793125f7), cited as "ST reg.h:<line>"; scale factors from lsm6dsox_reg.c of the same commit.
/// - "Linux" = drivers/iio/imu/st_lsm6dsx (GPL; read for facts only, no code copied).
/// - "AN5192" = ST application note for the LSM6DSO (same timestamp engine), as quoted in Linux st_lsm6dsx.h.
///   AN5272 (the LSM6DSOX note) could not be downloaded (st.com blocked); see T1.WORKLOG.
///
/// Every value marked [H] is verified on hardware day (docs/bringup_snippets/T1_imu.md).
#pragma once

#include <cstdint>
#include <optional>

namespace rig::lsm6dsox {

// ---------------------------------------------------------------------------------------------------------------
// Bus
// ---------------------------------------------------------------------------------------------------------------

/// 7-bit I2C address 110101x: 0x6A with SDO/SA0 = GND, 0x6B with SDO/SA0 = VDDIO (DS §5.1.2, Table 10;
/// ST reg.h:189-190 gives the 8-bit read forms 0xD5/0xD7). Modulino Movement default = 0x6A [H: i2cdetect].
inline constexpr uint8_t kI2cAddrSa0Low = 0x6A;
inline constexpr uint8_t kI2cAddrSa0High = 0x6B;
/// Max SCL: 400 kHz fast mode, 1 MHz fast mode plus (DS §4.4.2, Table 6).
inline constexpr int kMaxSclHz = 1000000;

// ---------------------------------------------------------------------------------------------------------------
// Register addresses (DS §8 Table 21 "Registers address map"; ST reg.h line given per register)
// ---------------------------------------------------------------------------------------------------------------

inline constexpr uint8_t kRegFuncCfgAccess = 0x01;  ///< ST reg.h:200. Bank select; must stay 0x00 (user bank).
inline constexpr uint8_t kRegFifoCtrl1 = 0x07;      ///< ST reg.h:258; DS §9.6.
inline constexpr uint8_t kRegFifoCtrl2 = 0x08;      ///< ST reg.h:264; DS §9.7.
inline constexpr uint8_t kRegFifoCtrl3 = 0x09;      ///< ST reg.h:286; DS §9.8.
inline constexpr uint8_t kRegFifoCtrl4 = 0x0A;      ///< ST reg.h:298; DS §9.9.
inline constexpr uint8_t kRegWhoAmI = 0x0F;         ///< ST reg.h:386; DS §9.14.
inline constexpr uint8_t kRegCtrl1Xl = 0x10;        ///< ST reg.h:387; DS §9.15.
inline constexpr uint8_t kRegCtrl2G = 0x11;         ///< ST reg.h:403; DS §9.16.
inline constexpr uint8_t kRegCtrl3C = 0x12;         ///< ST reg.h:417; DS §9.17.
inline constexpr uint8_t kRegCtrl4C = 0x13;         ///< ST reg.h:441; DS §9.18.
inline constexpr uint8_t kRegCtrl5C = 0x14;         ///< ST reg.h:465; DS §9.19.
inline constexpr uint8_t kRegCtrl6C = 0x15;         ///< ST reg.h:483; DS §9.20.
inline constexpr uint8_t kRegCtrl7G = 0x16;         ///< ST reg.h:499; DS §9.21.
inline constexpr uint8_t kRegCtrl8Xl = 0x17;        ///< ST reg.h:521; DS §9.22.
inline constexpr uint8_t kRegCtrl9Xl = 0x18;        ///< ST reg.h:541; DS §9.23.
inline constexpr uint8_t kRegCtrl10C = 0x19;        ///< ST reg.h:563; DS §9.24.
inline constexpr uint8_t kRegStatusReg = 0x1E;      ///< ST reg.h:673; DS §9.29.
inline constexpr uint8_t kRegOutTempL = 0x20;       ///< ST reg.h:689; DS §9.30.
inline constexpr uint8_t kRegFifoStatus1 = 0x3A;    ///< ST reg.h:817; DS §9.42.
inline constexpr uint8_t kRegFifoStatus2 = 0x3B;    ///< ST reg.h:823; DS §9.43.
inline constexpr uint8_t kRegTimestamp0 = 0x40;     ///< ST reg.h:845; DS §9.44. TIMESTAMP0..3 = 0x40..0x43, LE.
inline constexpr uint8_t kRegTimestamp2 = 0x42;     ///< ST reg.h:847. Writing 0xAA resets the counter (Linux
                                                    ///< st_lsm6dsx_buffer.c ST_LSM6DSX_REG_TS_RESET_ADDR). Unused.
inline constexpr uint8_t kRegInternalFreqFine = 0x63;  ///< ST reg.h:1075; DS §9.65 (read-only).
inline constexpr uint8_t kRegFifoDataOutTag = 0x78;    ///< ST reg.h:1158; DS §9.73.
inline constexpr uint8_t kRegFifoDataOutXL = 0x79;     ///< ST reg.h:1172; DS §9.74. Data = 0x79..0x7E.
inline constexpr uint8_t kRegFifoDataOutZH = 0x7E;     ///< ST reg.h:1177; DS §9.76.

// ---------------------------------------------------------------------------------------------------------------
// Fixed values and reset defaults (DS §8 Table 21 "Default" column)
// ---------------------------------------------------------------------------------------------------------------

inline constexpr uint8_t kWhoAmIValue = 0x6C;        ///< DS §9.14 "fixed at 6Ch"; ST reg.h:193 LSM6DSOX_ID.
inline constexpr uint8_t kCtrl3CDefault = 0x04;      ///< DS Table 21: CTRL3_C default 00000100 (IF_INC = 1).
inline constexpr uint8_t kCtrl9XlDefault = 0xE0;     ///< DS Table 21: CTRL9_XL default 11100000 (DEN_X/Y/Z).
inline constexpr uint8_t kPinCtrlDefault = 0x3F;     ///< DS Table 21: PIN_CTRL default 00111111.

// ---------------------------------------------------------------------------------------------------------------
// CTRL3_C bits (DS Table 56/57; ST reg.h:417-438)
// ---------------------------------------------------------------------------------------------------------------

inline constexpr uint8_t kCtrl3Boot = 1u << 7;     ///< Reboot memory content (reload trims); self-clearing.
inline constexpr uint8_t kCtrl3Bdu = 1u << 6;      ///< Block data update.
inline constexpr uint8_t kCtrl3IfInc = 1u << 2;    ///< Register auto-increment on multi-byte access (default 1).
inline constexpr uint8_t kCtrl3SwReset = 1u << 0;  ///< Software reset; self-clearing.

// CTRL9_XL (DS Table 73/74): I3C_disable = bit 1. "Recommended to set this bit to 1 during the initial device
// configuration phase, when the I3C interface is not used" (DS Table 74 note 1).
inline constexpr uint8_t kCtrl9I3cDisable = 1u << 1;
// CTRL10_C (DS Table 75/76; ST reg.h:563-575; Linux timer_en mask BIT(5)): TIMESTAMP_EN = bit 5.
inline constexpr uint8_t kCtrl10TimestampEn = 1u << 5;

// ---------------------------------------------------------------------------------------------------------------
// ODR / BDR codes (4-bit; same code table for CTRL1_XL.ODR_XL[7:4], CTRL2_G.ODR_G[7:4], FIFO_CTRL3.BDR_XL[3:0]
// and BDR_GY[7:4]). DS Tables 51, 55, 37; ST reg.h:2929-2939 (XL ODR), 2962-2972 (G ODR), 3687-3718 (BDR).
// Values are high-performance mode (XL_HM_MODE = 0, G_HM_MODE = 0, the reset defaults; DS Tables 65, 69).
// ---------------------------------------------------------------------------------------------------------------

struct OdrEntry {
  double hz;              ///< Nominal rate as written in the datasheet [Hz].
  uint8_t code;           ///< 4-bit code.
  int sim_period_ticks;   ///< SIMULATOR assumption: ODR period in 25 us timestamp ticks (40 kHz / ODR, both from
                          ///< the same oscillator per DS Table 174). Not used by the driver except as the
                          ///< initial guess of the slot period; the driver learns the real one from the FIFO.
};
inline constexpr OdrEntry kOdrTable[] = {
    {12.5, 0x1, 3200}, {26.0, 0x2, 1536}, {52.0, 0x3, 768},   {104.0, 0x4, 384},  {208.0, 0x5, 192},
    {416.0, 0x6, 96},  {833.0, 0x7, 48},  {1666.0, 0x8, 24},  {3332.0, 0x9, 12},  {6667.0, 0xA, 6},
};

/// ODR entry for a configured rate. Accepts the DS spellings (416, 1666, 3332) and ST's (417, 1667, 3333):
/// matches when within 0.5 % of a table value.
inline std::optional<OdrEntry> odr_entry(double hz) {
  for (const OdrEntry& e : kOdrTable) {
    const double rel = (hz - e.hz) / e.hz;
    if (rel > -0.005 && rel < 0.005) return e;
  }
  return std::nullopt;
}

/// Reverse lookup (code -> entry), for decoding readbacks.
inline std::optional<OdrEntry> odr_entry_from_code(uint8_t code) {
  for (const OdrEntry& e : kOdrTable)
    if (e.code == code) return e;
  return std::nullopt;
}

// ---------------------------------------------------------------------------------------------------------------
// Full scale and sensitivity
// ---------------------------------------------------------------------------------------------------------------

/// Accel FS[1:0]_XL in CTRL1_XL[3:2] with XL_FS_MODE = 0 (default): 00 = 2 g, 01 = 16 g, 10 = 4 g, 11 = 8 g
/// (DS Table 52; ST reg.h:2917-2920). Sensitivity [mg/LSB]: 0.061 / 0.122 / 0.244 / 0.488 (DS Table 2 LA_So;
/// ST lsm6dsox_reg.c lsm6dsox_from_fs{2,4,8,16}_to_mg).
struct AccelFs {
  int g;
  uint8_t code;        ///< FS[1:0]_XL value (2 bits).
  double mg_per_lsb;
};
inline constexpr AccelFs kAccelFsTable[] = {{2, 0x0, 0.061}, {4, 0x2, 0.122}, {8, 0x3, 0.244}, {16, 0x1, 0.488}};

/// Gyro FS in CTRL2_G[3:1] = {FS1_G, FS0_G, FS_125}: 250 = 000, 125 = 001, 500 = 010, 1000 = 100, 2000 = 110
/// (DS Tables 53/54; ST reg.h:2949-2953 packs fs_125 + fs_g the same way). Sensitivity [mdps/LSB]:
/// 4.375 / 8.75 / 17.5 / 35 / 70 (DS Table 2 G_So; ST lsm6dsox_reg.c lsm6dsox_from_fs*_to_mdps).
struct GyroFs {
  int dps;
  uint8_t code3;       ///< Value of CTRL2_G bits [3:1].
  double mdps_per_lsb;
};
inline constexpr GyroFs kGyroFsTable[] = {
    {125, 0x1, 4.375}, {250, 0x0, 8.75}, {500, 0x2, 17.5}, {1000, 0x4, 35.0}, {2000, 0x6, 70.0}};

inline std::optional<AccelFs> accel_fs(int g) {
  for (const AccelFs& f : kAccelFsTable)
    if (f.g == g) return f;
  return std::nullopt;
}
inline std::optional<GyroFs> gyro_fs(int dps) {
  for (const GyroFs& f : kGyroFsTable)
    if (f.dps == dps) return f;
  return std::nullopt;
}

/// Temperature: 256 LSB/degC, 0 LSB at 25 degC (DS §4.3 Table 4; ST lsm6dsox_reg.c lsm6dsox_from_lsb_to_celsius).
inline constexpr double kTempLsbPerC = 256.0;
inline constexpr double kTempOffsetC = 25.0;

// ---------------------------------------------------------------------------------------------------------------
// FIFO (DS §6.6, §9.6-9.9, §9.42-9.43, §9.73-9.76)
// ---------------------------------------------------------------------------------------------------------------

/// One FIFO word = 1 tag byte + 6 data bytes (DS §6.6.7).
inline constexpr int kFifoWordBytes = 7;
/// FIFO depth without compression: "3 KB" of 6-byte words = 512 words (DS §6.6; DIFF_FIFO is 10 bits,
/// WTM 9 bits). [H: the max FIFO level seen after an induced overrun.]
inline constexpr int kFifoCapacityWords = 512;
/// WTM[8:0] (FIFO_CTRL1[7:0] + FIFO_CTRL2[0]), 1 LSB = one 7-byte word (DS Tables 33, 35).
inline constexpr int kFifoWatermarkMax = 511;
inline constexpr uint8_t kFifoCtrl2Wtm8 = 1u << 0;

/// FIFO_CTRL4 (DS Table 39; ST reg.h:298-311): DEC_TS_BATCH[7:6], ODR_T_BATCH[5:4], FIFO_MODE[2:0].
inline constexpr uint8_t kFifoModeBypass = 0x0;      ///< 000 (also flushes the FIFO; DS §6.6.1).
inline constexpr uint8_t kFifoModeContinuous = 0x6;  ///< 110 "if the FIFO is full, the new sample overwrites the
                                                     ///< older one" (DS Table 39; ST reg.h:3731 STREAM_MODE = 6).
/// DEC_TS_BATCH codes: 00 not batched, 01 every max(BDR), 10 every 8th, 11 every 32nd (DS Table 39;
/// ST reg.h:3752-3757).
inline constexpr uint8_t dec_ts_code(int decimation) {
  return decimation == 1 ? 0x1 : decimation == 8 ? 0x2 : decimation == 32 ? 0x3 : 0x0;
}
/// ODR_T_BATCH codes: 00 off, 01 1.6 Hz, 10 12.5 Hz, 11 52 Hz (DS Table 39; ST reg.h:3740-3745).
inline constexpr uint8_t odr_t_code(double hz) {
  return hz <= 0.0 ? 0x0 : hz < 5.0 ? 0x1 : hz < 30.0 ? 0x2 : 0x3;
}
inline constexpr uint8_t fifo_ctrl4(int ts_decimation, double temp_hz, uint8_t mode) {
  return static_cast<uint8_t>((dec_ts_code(ts_decimation) << 6) | (odr_t_code(temp_hz) << 4) | (mode & 0x7));
}

/// FIFO_STATUS2 bits (DS Tables 120/121; ST reg.h:823-842).
inline constexpr uint8_t kFifoStatus2WtmIa = 1u << 7;      ///< Level >= WTM.
inline constexpr uint8_t kFifoStatus2OvrIa = 1u << 6;      ///< FIFO completely filled (overrun in continuous mode).
inline constexpr uint8_t kFifoStatus2FullIa = 1u << 5;     ///< FIFO full at the next ODR.
inline constexpr uint8_t kFifoStatus2CounterBdrIa = 1u << 4;
inline constexpr uint8_t kFifoStatus2OvrLatched = 1u << 3; ///< Latched overrun; reset when FIFO_STATUS2 is read.
inline constexpr uint8_t kFifoStatus2DiffMask = 0x03;      ///< DIFF_FIFO[9:8].

/// Unread words from FIFO_STATUS1/2 (DS §9.42/9.43; ST lsm6dsox_fifo_data_level_get: mask 0x03FF).
inline constexpr int fifo_level(uint8_t status1, uint8_t status2) {
  return ((status2 & kFifoStatus2DiffMask) << 8) | status1;
}

/// FIFO_DATA_OUT_TAG layout: TAG_SENSOR[7:3], TAG_CNT[2:1], TAG_PARITY[0] (DS Tables 194/195; ST reg.h:1158-1170).
inline constexpr uint8_t tag_sensor(uint8_t tag_byte) { return static_cast<uint8_t>(tag_byte >> 3); }
inline constexpr uint8_t tag_cnt(uint8_t tag_byte) { return static_cast<uint8_t>((tag_byte >> 1) & 0x3); }
inline constexpr uint8_t tag_parity(uint8_t tag_byte) { return static_cast<uint8_t>(tag_byte & 0x1); }

/// TAG_SENSOR values (DS Table 196; ST reg.h:3771-3792).
inline constexpr uint8_t kTagGyroNC = 0x01;
inline constexpr uint8_t kTagAccelNC = 0x02;
inline constexpr uint8_t kTagTemperature = 0x03;
inline constexpr uint8_t kTagTimestamp = 0x04;
inline constexpr uint8_t kTagCfgChange = 0x05;

// ---------------------------------------------------------------------------------------------------------------
// Timestamp (DS §9.44, Table 174; AN5192 via Linux st_lsm6dsx.h:203-211)
// ---------------------------------------------------------------------------------------------------------------

/// Nominal timestamp resolution 25 us/LSB (DS §9.44 Table 123; ST lsm6dsox_reg.c lsm6dsox_from_lsb_to_nsec).
inline constexpr double kNominalTickNs = 25000.0;
/// INTERNAL_FREQ_FINE: int8, step 0.15 % of the effective ODR and timestamp rate versus typical (DS Table 174).
inline constexpr double kFreqFineStep = 0.0015;

/// Trimmed tick period [ns]: t = 1 / (40000 Hz * (1 + 0.0015 * FREQ_FINE)) (AN5192 formula as quoted in Linux
/// st_lsm6dsx.h:203-206; Linux applies its linearization 25000 - 37.5 * val ns with the same constants to the
/// LSM6DSOX entry, st_lsm6dsx_core.c ts_sensitivity = 25000, ts_trim_coeff = 37500). We use the exact form.
/// [H: AN5272 §6.4 should state the same; the clock fit's slope reveals any error at once.]
inline double trimmed_tick_ns(int8_t freq_fine) {
  return kNominalTickNs / (1.0 + kFreqFineStep * static_cast<double>(freq_fine));
}

// ---------------------------------------------------------------------------------------------------------------
// Reset timing (Linux st_lsm6dsx_core.c st_lsm6dsx_reset_device: flush FIFO, SW_RESET, msleep(50), BOOT,
// msleep(50); DS Table 3 turn-on time 35 ms).
// ---------------------------------------------------------------------------------------------------------------

inline constexpr int kResetWaitMs = 50;
inline constexpr int kBootWaitMs = 50;
inline constexpr int kResetPollTimeoutMs = 200;

}  // namespace rig::lsm6dsox
