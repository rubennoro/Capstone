// SPDX-License-Identifier: MIT
// ec-imu-probe: hardware-day check of the LSM6DSOX on the Jetson (PLAN T1; acceptance H1 in PLAN section 5).
//
//   ec-imu-probe [--config rig.yaml] [--bus 7] [--addr 0x6a] [--duration 10] [--record DIR] [--raw N] [--sim]
//
// Prints: the I2C bus clock (device tree), WHO_AM_I, the configuration written and read back, a live line per
// second (rate, FIFO level, overruns, I2C errors, clock fit), then a summary with PASS/FAIL against H1:
// rate = ODR x (1 + 0.0015 x INTERNAL_FREQ_FINE) +- 0.5 %, 0 FIFO overruns, |g| = 9.81 +- 0.2 m/s^2 at rest, clock residual p99 < 100 us.
// --raw N: configure, then dump the first N raw FIFO words (tag, TAG_CNT, parity, values) and exit; this checks
//          the FIFO slot layout assumptions (timestamp first in a slot, one TAG_CNT per slot) on real silicon.
// --record DIR (created): DIR/data.csv (EuRoC imu0), DIR/sample_meta.csv (t_ns, chip_ticks, temp_c, flags), DIR/chip.csv
//          (clock brackets), DIR/clock_fit.json.
// Exit status: 0 if every check passed, 1 if any failed, 2 on usage/setup errors.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "CLI11.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/drivers/imu/lsm6dsox_driver.hpp"
#include "rig/drivers/imu/lsm6dsox_fifo.hpp"
#include "rig/drivers/imu/lsm6dsox_regs.hpp"
#include "rig/drivers/imu/lsm6dsox_sim.hpp"
#include "rig/package/layout.hpp"
#include "rig/time/clock.hpp"

namespace {

using namespace rig;             // NOLINT
using namespace rig::lsm6dsox;   // NOLINT

constexpr double kPi = 3.14159265358979323846;

bool g_all_pass = true;

void check(bool ok, const std::string& what, const std::string& detail) {
  std::printf("  [%s] %-34s %s\n", ok ? "PASS" : "FAIL", what.c_str(), detail.c_str());
  g_all_pass = g_all_pass && ok;
}

std::string fmt(const char* f, double a) {
  char b[128];
  std::snprintf(b, sizeof b, f, a);
  return b;
}

std::string hex2(unsigned v) {
  char b[8];
  std::snprintf(b, sizeof b, "0x%02x", v & 0xFFu);
  return b;
}

// /sys/bus/i2c/devices/i2c-N/of_node/clock-frequency: one big-endian u32 [Hz] (device-tree property).
std::string bus_clock(int bus) {
  std::ifstream f("/sys/bus/i2c/devices/i2c-" + std::to_string(bus) + "/of_node/clock-frequency", std::ios::binary);
  unsigned char b[4];
  if (!f.read(reinterpret_cast<char*>(b), 4)) return "unknown (no device-tree clock-frequency)";
  const unsigned long hz = (static_cast<unsigned long>(b[0]) << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
  return std::to_string(hz) + " Hz" + (hz < 400000 ? "  <-- below 400 kHz: 833 Hz with timestamps will not fit" : "");
}

const char* tag_name(uint8_t s) {
  switch (s) {
    case kTagGyroNC: return "GYRO";
    case kTagAccelNC: return "ACCEL";
    case kTagTemperature: return "TEMP";
    case kTagTimestamp: return "TS";
    case kTagCfgChange: return "CFG";
    default: return "?";
  }
}

int raw_dump(I2cBus& bus, const ImuConfig& c, int n_words) {
  ImuClock& clock = *real_imu_clock();
  std::vector<Lsm6dsoxReadback> rb;
  Status st = lsm6dsox_reset_and_configure(bus, c, clock, &rb, nullptr);
  if (!st.ok()) {
    std::printf("configure failed: %s\n", st.to_string().c_str());
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(60));  // ~50 slots at 833 Hz
  const uint8_t a = static_cast<uint8_t>(c.i2c_addr);
  uint8_t s[2];
  if (!bus.read_regs(a, kRegFifoStatus1, s, 2).ok()) return 1;
  const int level = fifo_level(s[0], s[1]);
  std::printf("FIFO_STATUS1/2 = 0x%02x 0x%02x: level %d words%s%s\n", s[0], s[1], level,
              (s[1] & kFifoStatus2OvrIa) ? " OVR" : "", (s[1] & kFifoStatus2WtmIa) ? " WTM" : "");
  const int n = std::min(level, n_words);
  std::vector<uint8_t> buf(static_cast<std::size_t>(n) * kFifoWordBytes);
  // One word per transaction here (no reliance on the 0x7E -> 0x78 wrap), then one burst to compare.
  for (int i = 0; i < n; ++i)
    if (!bus.read_regs(a, kRegFifoDataOutTag, buf.data() + i * kFifoWordBytes, kFifoWordBytes).ok()) return 1;
  int ts_first = 0, slots = 0, cnt_mismatch = 0, odd = 0;
  for (int i = 0; i < n; ++i) {
    const uint8_t* w = buf.data() + i * kFifoWordBytes;
    const uint8_t t = tag_sensor(w[0]);
    int bits = 0;
    for (int k = 0; k < 8; ++k) bits += (w[0] >> k) & 1;
    odd += bits & 1;
    std::printf("%4d  tag 0x%02x %-5s cnt %d par %d  data", i, w[0], tag_name(t), tag_cnt(w[0]), tag_parity(w[0]));
    for (int k = 1; k < 7; ++k) std::printf(" %02x", w[k]);
    if (t == kTagTimestamp) {
      const uint32_t v = w[1] | (w[2] << 8) | (w[3] << 16) | (static_cast<uint32_t>(w[4]) << 24);
      std::printf("  ticks %u", v);
      ++slots;
      if (i + 1 < n && tag_sensor(w[7]) != kTagTimestamp) {
        ts_first += 1;
        if (tag_cnt(w[7]) != tag_cnt(w[0])) ++cnt_mismatch;
      }
    } else if (t == kTagGyroNC || t == kTagAccelNC) {
      std::printf("  x %6d y %6d z %6d", static_cast<int16_t>(w[1] | (w[2] << 8)),
                  static_cast<int16_t>(w[3] | (w[4] << 8)), static_cast<int16_t>(w[5] | (w[6] << 8)));
    }
    std::printf("\n");
  }
  std::printf("timestamp words %d; followed by data with the same TAG_CNT: %d; different TAG_CNT: %d; "
              "odd-parity tag bytes %d/%d\n",
              slots, ts_first - cnt_mismatch, cnt_mismatch, odd, n);
  // Burst read of 8 words to confirm the 0x7E -> 0x78 address wrap: every 7th byte must be a valid tag.
  std::vector<uint8_t> burst(8 * kFifoWordBytes);
  if (bus.read_regs(a, kRegFifoDataOutTag, burst.data(), burst.size()).ok()) {
    int valid = 0;
    for (int i = 0; i < 8; ++i) {
      const uint8_t t = tag_sensor(burst[static_cast<std::size_t>(i * kFifoWordBytes)]);
      valid += (t == kTagGyroNC || t == kTagAccelNC || t == kTagTimestamp || t == kTagTemperature);
    }
    std::printf("burst read of 8 words from 0x78: %d/8 valid tags (8/8 = address wraps 0x7e -> 0x78)\n", valid);
  }
  (void)lsm6dsox_power_down(bus, a);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"ec-imu-probe: LSM6DSOX bring-up check (WHO_AM_I, config readback, rate, FIFO, clock fit)"};
  std::string config_path, record_dir, pairing = "tag";
  int bus = -1, addr = -1, raw = 0, ts_dec = 0;
  double duration = 10.0, odr = 0;
  int accel_fs = 0, gyro_fs = 0, poll_ms = 0;
  bool sim = false, no_rest = false;
  app.add_option("--config", config_path, "rig.yaml (uses imu.driver)");
  app.add_option("--bus", bus, "I2C bus number (default: config, else 7)");
  app.add_option("--addr", addr, "7-bit address, e.g. 0x6a")->transform([](std::string s) {
    return std::to_string(std::stoi(s, nullptr, 0));
  });
  app.add_option("--duration", duration, "seconds of streaming")->check(CLI::Range(1.0, 86400.0));
  app.add_option("--record", record_dir, "write data.csv, sample_meta.csv, chip.csv, clock_fit.json into DIR (created)");
  app.add_option("--raw", raw, "dump N raw FIFO words and exit")->check(CLI::Range(1, 512));
  app.add_option("--odr", odr, "override ODR [Hz]");
  app.add_option("--accel-fs", accel_fs, "override accel full scale [g]");
  app.add_option("--gyro-fs", gyro_fs, "override gyro full scale [dps]");
  app.add_option("--poll-ms", poll_ms, "override the FIFO poll period [ms] (e.g. 400 forces overruns)")
      ->check(CLI::Range(1, 1000));
  app.add_option("--ts-decimation", ts_dec, "FIFO timestamp decimation 1|8|32");
  app.add_option("--pairing", pairing, "slot pairing: tag (TAG_CNT) | ts (timestamp first)")
      ->check(CLI::IsMember({"tag", "ts"}));
  app.add_flag("--sim", sim, "use the register simulator (laptop dry run)");
  app.add_flag("--no-rest-check", no_rest, "skip the |g| check (rig is moving)");
  CLI11_PARSE(app, argc, argv);

  ImuConfig c;
  if (!config_path.empty()) {
    auto rc = load_rig_config(config_path);
    if (!rc.ok()) {
      std::fprintf(stderr, "%s\n", rc.status().to_string().c_str());
      return 2;
    }
    c = rc->imu.driver;
  }
  if (bus >= 0) c.i2c_bus = bus;
  if (addr >= 0) c.i2c_addr = addr;
  if (odr > 0) c.odr_hz = odr;
  if (accel_fs > 0) c.accel_fs_g = accel_fs;
  if (gyro_fs > 0) c.gyro_fs_dps = gyro_fs;
  if (ts_dec > 0) c.timestamp_decimation = ts_dec;
  if (poll_ms > 0) c.poll_period_ms = poll_ms;
  c.pair_by_tag_counter = pairing == "tag";
  c.queue_capacity = std::max<std::size_t>(c.queue_capacity, 65536);
  if (auto seq = lsm6dsox_config_sequence(c); !seq.ok()) {
    std::fprintf(stderr, "%s\n", seq.status().to_string().c_str());
    return 2;
  }

  std::unique_ptr<I2cBus> i2c;
  if (sim) {
    i2c = std::make_unique<Lsm6dsoxSim>(Lsm6dsoxSimOptions::from_config(c.sim, c.i2c_addr), real_imu_clock());
  } else {
    auto b = make_linux_i2c_bus(c.i2c_bus);
    if (!b.ok()) {
      std::fprintf(stderr, "%s\n", b.status().to_string().c_str());
      return 2;
    }
    i2c = std::move(b).value();
  }
  std::printf("== bus %s, address 0x%02x\n", i2c->name().c_str(), c.i2c_addr);
  if (!sim) std::printf("  I2C clock (device tree): %s\n", bus_clock(c.i2c_bus).c_str());

  std::printf("== probe\n");
  uint8_t who = 0;
  Status ws = i2c->read_regs(static_cast<uint8_t>(c.i2c_addr), kRegWhoAmI, &who, 1);
  check(ws.ok() && who == c.expected_who_am_i, "WHO_AM_I (0x0f)",
        ws.ok() ? "read " + hex2(who) + ", expected " + hex2(static_cast<unsigned>(c.expected_who_am_i))
                : ws.to_string());
  if (!ws.ok() || who != c.expected_who_am_i) return 1;

  if (raw > 0) return raw_dump(*i2c, c, raw);

  auto made = make_lsm6dsox(c, std::move(i2c), real_imu_clock());
  if (!made.ok()) {
    std::fprintf(stderr, "%s\n", made.status().to_string().c_str());
    return 2;
  }
  std::unique_ptr<ImuBackend> imu = std::move(made).value();
  std::printf("== reset + configure\n");
  const Status st = imu->start();
  auto seq = lsm6dsox_config_sequence(c);
  for (const Lsm6dsoxReadback& r : lsm6dsox_last_readback(*imu)) {
    std::string meaning;
    for (const auto& w : *seq)
      if (w.reg == r.reg) meaning = w.meaning;
    char d[96];
    std::snprintf(d, sizeof d, "wrote 0x%02x read 0x%02x  ", r.expected, r.actual);
    check(r.ok(), std::string(r.name) + " (" + hex2(r.reg) + ")", d + meaning);
  }
  check(st.ok(), "start()", st.ok() ? "reset, boot, readback OK" : st.to_string());
  if (!st.ok()) return 1;
  const ImuStats s0 = imu->stats();
  std::printf("  INTERNAL_FREQ_FINE = %d -> tick %.3f ns (nominal 25000)\n", s0.freq_fine, s0.nominal_tick_ns);

  std::ofstream data, chip, schip;
  if (!record_dir.empty()) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(record_dir, ec);
    const fs::path d(record_dir);
    data.open(d / layout::kImuData);
    schip.open(d / layout::kSampleMeta);
    chip.open(d / layout::kChip);
    if (ec || !data || !schip || !chip) {
      std::fprintf(stderr, "cannot write into %s%s\n", record_dir.c_str(), ec ? (": " + ec.message()).c_str() : "");
      return 2;
    }
    // Same files and header lines as a package raw/<imu id>/ (rig/package/layout.hpp, PIPELINE 6.1).
    data << layout::kImuDataHeader << '\n';
    schip << layout::kSampleMetaHeader << '\n';
    chip << layout::kChipHeader << '\n';
  }

  std::printf("== streaming %.0f s\n", duration);
  std::vector<ImuSample> all, batch;
  std::vector<ClockBracket> br;
  const TimeNs t_end = now_raw_ns() + static_cast<TimeNs>(duration * 1e9);
  TimeNs next_line = now_raw_ns() + 1000000000;
  while (now_raw_ns() < t_end && imu->status().ok()) {
    batch.clear();
    imu->read(batch, std::chrono::milliseconds(100));
    br.clear();
    imu->read_brackets(br);
    if (data) {
      for (const ImuSample& x : batch) {
        char line[256];
        std::snprintf(line, sizeof line, "%lld,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", static_cast<long long>(x.t_ns),
                      x.gyro_rps.x(), x.gyro_rps.y(), x.gyro_rps.z(), x.acc_mps2.x(), x.acc_mps2.y(), x.acc_mps2.z());
        data << line;
        schip << x.t_ns << ',' << x.chip_ticks << ',' << x.temp_c << ',' << x.flags << '\n';
      }
      for (const ClockBracket& b : br) chip << b.t_host_before_ns << ',' << b.chip_ticks << ',' << b.t_host_after_ns << '\n';
    }
    all.insert(all.end(), batch.begin(), batch.end());
    if (now_raw_ns() >= next_line) {
      next_line += 1000000000;
      const ImuStats s = imu->stats();
      std::printf("  rate %8.2f Hz  fifo max %3d  ovr %llu  i2c err %llu (retries %llu)  gaps %llu  "
                  "slope %+8.1f ppm  resid p99 %6.1f us  width p50 %6.1f us\n",
                  s.rate_hz_estimate, s.fifo_level_max, static_cast<unsigned long long>(s.fifo_overruns),
                  static_cast<unsigned long long>(s.i2c_errors), static_cast<unsigned long long>(s.i2c_retries),
                  static_cast<unsigned long long>(s.gaps),
                  (s.clock.fit.slope_ns_per_tick / s.nominal_tick_ns - 1.0) * 1e6, s.clock.residual_ns.p99 / 1e3,
                  s.clock.bracket_width_ns.p50 / 1e3);
      std::fflush(stdout);
    }
  }
  const ImuStats s = imu->stats();
  const Lsm6dsoxDiagnostics dg = lsm6dsox_diagnostics(*imu);
  const Status fin = imu->status();
  (void)imu->stop();
  if (!record_dir.empty()) {
    std::ofstream(std::filesystem::path(record_dir) / layout::kClockFit) << clock_fit_json(c, s) << "\n";
    std::printf("  recorded %zu samples into %s\n", all.size(), record_dir.c_str());
  }

  std::printf("== summary (%zu samples)\n", all.size());
  check(fin.ok(), "driver status", fin.to_string());
  double rate = kNaN;
  if (all.size() > 1) rate = static_cast<double>(all.size() - 1) * 1e9 / static_cast<double>(all.back().t_ns - all.front().t_ns);
  // The chip's real ODR is trimmed like its timestamp: ODR x (1 + 0.0015 x FREQ_FINE) (DS12814 Table 174; R1 F4).
  const double expected_rate = c.odr_hz * (1.0 + kFreqFineStep * s.freq_fine);
  check(std::abs(rate / expected_rate - 1.0) < 0.005, "sample rate (trimmed ODR +- 0.5 %)",
        fmt("%.3f Hz", rate) + fmt(" (expected %.3f", expected_rate) + fmt(" = ODR %.1f", c.odr_hz) +
            " x (1 + 0.0015 x " + std::to_string(s.freq_fine) + "))");
  check(s.fifo_overruns == 0, "FIFO overruns", std::to_string(s.fifo_overruns));
  check(s.queue_overruns == 0, "host queue overruns", std::to_string(s.queue_overruns));
  check(s.gaps == 0, "sample gaps", std::to_string(s.gaps) + fmt(", max interval %.3f ms", s.max_gap_ns / 1e6));
  check(s.unknown_tags == 0 && s.timestamp_regressions == 0, "unknown tags / ts regressions",
        std::to_string(s.unknown_tags) + " / " + std::to_string(s.timestamp_regressions));
  check(s.partial_slots == 0, "partial slots (gyro or accel missing)", std::to_string(s.partial_slots));
  check(c.timestamp_decimation != 1 || s.slots_without_timestamp == 0, "slots without timestamp word",
        std::to_string(s.slots_without_timestamp) + " (pairing " + pairing + ")");
  check(s.i2c_errors == 0, "I2C errors after retries", std::to_string(s.i2c_errors) + ", retries " + std::to_string(s.i2c_retries));
  check(s.clock.fit.valid && !s.clock.fit.slope_clamped, "clock fit valid, slope not clamped",
        fmt("slope %+.1f ppm vs trimmed tick", (s.clock.fit.slope_ns_per_tick / s.nominal_tick_ns - 1.0) * 1e6));
  check(s.clock.residual_ns.p99 < 100000.0, "clock residual p99 < 100 us",
        fmt("p50 %.1f us", s.clock.residual_ns.p50 / 1e3) + fmt(" p99 %.1f us", s.clock.residual_ns.p99 / 1e3) +
            fmt(", bracket width p50 %.1f us", s.clock.bracket_width_ns.p50 / 1e3) +
            ", torn reads " + std::to_string(s.torn_timestamp_reads));
  std::printf("  words: gyro %llu accel %llu ts %llu temp %llu cfg %llu; odd-parity tags %llu; learned period %.4f "
              "ticks; polls %llu (late %llu)\n",
              static_cast<unsigned long long>(dg.gyro_words), static_cast<unsigned long long>(dg.accel_words),
              static_cast<unsigned long long>(dg.timestamp_words), static_cast<unsigned long long>(dg.temperature_words),
              static_cast<unsigned long long>(dg.cfg_change_words), static_cast<unsigned long long>(dg.odd_parity_words),
              dg.learned_period_ticks, static_cast<unsigned long long>(dg.polls),
              static_cast<unsigned long long>(dg.poll_overruns));
  if (!all.empty()) {
    Eigen::Vector3d ma = Eigen::Vector3d::Zero(), mg = Eigen::Vector3d::Zero();
    Eigen::Vector3d va = Eigen::Vector3d::Zero(), vg = Eigen::Vector3d::Zero();
    for (const ImuSample& x : all) {
      ma += x.acc_mps2;
      mg += x.gyro_rps;
    }
    ma /= static_cast<double>(all.size());
    mg /= static_cast<double>(all.size());
    for (const ImuSample& x : all) {
      va += (x.acc_mps2 - ma).cwiseAbs2();
      vg += (x.gyro_rps - mg).cwiseAbs2();
    }
    va = (va / static_cast<double>(all.size())).cwiseSqrt();
    vg = (vg / static_cast<double>(all.size())).cwiseSqrt();
    std::printf("  accel mean [%+.4f %+.4f %+.4f] std [%.4f %.4f %.4f] m/s^2\n", ma.x(), ma.y(), ma.z(), va.x(),
                va.y(), va.z());
    std::printf("  gyro  mean [%+.5f %+.5f %+.5f] std [%.5f %.5f %.5f] rad/s (mean %+.3f %+.3f %+.3f dps)\n", mg.x(),
                mg.y(), mg.z(), vg.x(), vg.y(), vg.z(), mg.x() * 180 / kPi, mg.y() * 180 / kPi, mg.z() * 180 / kPi);
    std::printf("  temperature %.2f degC\n", static_cast<double>(all.back().temp_c));
    if (!no_rest) {
      const double gn = mean_acc_norm(all);
      check(std::abs(gn - 9.81) < 0.2, "|g| at rest 9.81 +- 0.2", fmt("%.4f m/s^2", gn));
    }
  }
  std::printf("== %s\n", g_all_pass ? "ALL PASS" : "SOME CHECKS FAILED");
  return g_all_pass ? 0 : 1;
}
