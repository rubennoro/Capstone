// SPDX-License-Identifier: MIT
// LSM6DSOX FIFO driver (PLAN T1). Sequence, threading and sources: rig/drivers/imu/lsm6dsox_driver.hpp and
// rig/drivers/imu/lsm6dsox_regs.hpp.
#include "rig/drivers/imu/lsm6dsox_driver.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include <nlohmann/json.hpp>

#include "rig/drivers/imu/lsm6dsox_fifo.hpp"
#include "rig/drivers/imu/lsm6dsox_regs.hpp"
#include "rig/time/clock.hpp"

namespace rig {

using namespace lsm6dsox;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int64_t kMs = 1000000;
constexpr int64_t kErrorWindowNs = 10000 * kMs;  // i2c_error_budget window (ImuConfig: "per 10 s")
constexpr int64_t kWarmupNs = 2000 * kMs;        // bracket every poll during the first 2 s
constexpr std::size_t kMaxBrackets = 100000;     // read_brackets() backlog bound
constexpr int kBackwardsResetCount = 3;          // consecutive "ticks went backwards" brackets -> chip restarted

std::string hex2(unsigned v) {
  char b[8];
  std::snprintf(b, sizeof b, "0x%02x", v & 0xFFu);
  return b;
}

std::string hz_str(double hz) {
  char b[32];
  std::snprintf(b, sizeof b, "%g", hz);
  return b;
}

uint32_t le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

const char* reg_name(uint8_t r) {
  switch (r) {
    case kRegFifoCtrl1: return "FIFO_CTRL1";
    case kRegFifoCtrl2: return "FIFO_CTRL2";
    case kRegFifoCtrl3: return "FIFO_CTRL3";
    case kRegFifoCtrl4: return "FIFO_CTRL4";
    case kRegWhoAmI: return "WHO_AM_I";
    case kRegCtrl1Xl: return "CTRL1_XL";
    case kRegCtrl2G: return "CTRL2_G";
    case kRegCtrl3C: return "CTRL3_C";
    case kRegCtrl9Xl: return "CTRL9_XL";
    case kRegCtrl10C: return "CTRL10_C";
    default: return "REG";
  }
}

std::string where(const I2cBus& bus, uint8_t addr7) { return bus.name() + " addr " + hex2(addr7); }

/// Runs `fn` up to 1 + retries times. `on_retry` is called for every retry.
Status with_retries(int retries, const std::function<Status()>& fn, const std::function<void()>& on_retry = {}) {
  Status st;
  for (int i = 0; i <= retries; ++i) {
    st = fn();
    if (st.ok()) return st;
    if (i < retries && on_retry) on_retry();
  }
  return st;
}

Status read_reg_retry(I2cBus& bus, uint8_t addr, uint8_t reg, uint8_t& v, int retries) {
  return with_retries(retries, [&] { return bus.read_regs(addr, reg, &v, 1); });
}
Status write_reg_retry(I2cBus& bus, uint8_t addr, uint8_t reg, uint8_t v, int retries) {
  return with_retries(retries, [&] { return bus.write_reg(addr, reg, v); });
}

/// Polls `reg` until (value & mask) == 0, tolerating NACKs (the chip may not answer while booting).
Status wait_bit_clear(I2cBus& bus, uint8_t addr, uint8_t reg, uint8_t mask, ImuClock& clock, const char* what) {
  const TimeNs deadline = clock.now_ns() + kResetPollTimeoutMs * kMs;
  Status last;
  for (;;) {
    uint8_t v = 0xFF;
    last = bus.read_regs(addr, reg, &v, 1);
    if (last.ok() && (v & mask) == 0) return Status::Ok();
    if (clock.now_ns() > deadline)
      return device_error(where(bus, addr) + ": " + what + " did not complete within " +
                          std::to_string(kResetPollTimeoutMs) + " ms (" + reg_name(reg) + " = " + hex2(v) +
                          (last.ok() ? "" : ", last error: " + last.to_string()) + ")");
    clock.sleep_until(clock.now_ns() + 5 * kMs, nullptr);
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------------------------------

Result<std::vector<Lsm6dsoxRegWrite>> lsm6dsox_config_sequence(const ImuConfig& c) {
  const auto odr = odr_entry(c.odr_hz);
  if (!odr) return invalid_argument("imu.driver.odr_hz: " + std::to_string(c.odr_hz) + " is not an LSM6DSOX ODR");
  const auto afs = accel_fs(c.accel_fs_g);
  if (!afs) return invalid_argument("imu.driver.accel_fs_g: " + std::to_string(c.accel_fs_g) + " not in 2|4|8|16");
  const auto gfs = gyro_fs(c.gyro_fs_dps);
  if (!gfs)
    return invalid_argument("imu.driver.gyro_fs_dps: " + std::to_string(c.gyro_fs_dps) +
                            " not in 125|250|500|1000|2000");
  if (c.fifo_watermark_words < 1 || c.fifo_watermark_words > kFifoWatermarkMax)
    return invalid_argument("imu.driver.fifo_watermark_words: " + std::to_string(c.fifo_watermark_words) +
                            " not in 1..511");
  if (c.timestamp_decimation != 1 && c.timestamp_decimation != 8 && c.timestamp_decimation != 32)
    return invalid_argument("imu.driver.timestamp_decimation: " + std::to_string(c.timestamp_decimation) +
                            " not in 1|8|32");
  const double th = c.temperature_batch_hz;
  if (!(th == 0.0 || std::abs(th - 1.6) < 1e-9 || std::abs(th - 12.5) < 1e-9 || std::abs(th - 52.0) < 1e-9))
    return invalid_argument("imu.driver.temperature_batch_hz: " + std::to_string(th) + " not in 0|1.6|12.5|52");
  if (c.i2c_addr < 0 || c.i2c_addr > 0x7F) return invalid_argument("imu.driver.i2c_addr: not a 7-bit address");
  if (c.poll_period_ms < 1 || c.bracket_period_ms < 1)
    return invalid_argument("imu.driver.poll_period_ms / bracket_period_ms must be >= 1");
  if (c.fifo_burst_words < 1 || c.fifo_burst_words > 256)
    return invalid_argument("imu.driver.fifo_burst_words: not in 1..256");
  if (c.i2c_retries < 0 || c.i2c_error_budget < 0) return invalid_argument("imu.driver.i2c_retries/budget < 0");
  if (c.queue_capacity < 16) return invalid_argument("imu.driver.queue_capacity < 16");
  if (!(c.gap_factor >= 1.0)) return invalid_argument("imu.driver.gap_factor < 1");

  const int w = c.fifo_watermark_words;
  const uint8_t f4 = fifo_ctrl4(c.timestamp_decimation, th, kFifoModeContinuous);
  std::vector<Lsm6dsoxRegWrite> s;
  s.push_back({kRegCtrl3C, static_cast<uint8_t>(kCtrl3Bdu | kCtrl3IfInc), "CTRL3_C",
               "BDU=1 (b6), IF_INC=1 (b2) [DS Tab 56/57]"});
  s.push_back({kRegCtrl9Xl, static_cast<uint8_t>(kCtrl9XlDefault | kCtrl9I3cDisable), "CTRL9_XL",
               "DEN_X/Y/Z defaults (b7:5), I3C_disable=1 (b1) [DS Tab 73/74 note 1]"});
  s.push_back({kRegFifoCtrl1, static_cast<uint8_t>(w & 0xFF), "FIFO_CTRL1",
               "WTM[7:0]; watermark " + std::to_string(w) + " words [DS Tab 32/33]"});
  s.push_back({kRegFifoCtrl2, static_cast<uint8_t>((w >> 8) & kFifoCtrl2Wtm8), "FIFO_CTRL2",
               "WTM8 (b0); no compression, no STOP_ON_WTM, no ODRCHG [DS Tab 34/35]"});
  s.push_back({kRegFifoCtrl3, static_cast<uint8_t>((odr->code << 4) | odr->code), "FIFO_CTRL3",
               "BDR_GY[7:4] = BDR_XL[3:0] = " + hex2(odr->code) + " (" + hz_str(odr->hz) + " Hz) [DS Tab 36/37]"});
  s.push_back({kRegCtrl1Xl, static_cast<uint8_t>((odr->code << 4) | (afs->code << 2)), "CTRL1_XL",
               "ODR_XL[7:4] = " + hex2(odr->code) + ", FS_XL[3:2] = " + std::to_string(afs->code) + " (+-" +
                   std::to_string(c.accel_fs_g) + " g), LPF2 off [DS Tab 49-52]"});
  s.push_back({kRegCtrl2G, static_cast<uint8_t>((odr->code << 4) | (gfs->code3 << 1)), "CTRL2_G",
               "ODR_G[7:4] = " + hex2(odr->code) + ", FS[3:1] = " + std::to_string(gfs->code3) + " (+-" +
                   std::to_string(c.gyro_fs_dps) + " dps) [DS Tab 53-55]"});
  s.push_back({kRegCtrl10C, kCtrl10TimestampEn, "CTRL10_C", "TIMESTAMP_EN=1 (b5) [DS Tab 75/76]"});
  s.push_back({kRegFifoCtrl4, f4, "FIFO_CTRL4",
               "DEC_TS_BATCH[7:6] = " + std::to_string(dec_ts_code(c.timestamp_decimation)) +
                   ", ODR_T_BATCH[5:4] = " + std::to_string(odr_t_code(th)) +
                   ", FIFO_MODE[2:0] = 110 continuous [DS Tab 38/39]"});
  return s;
}

Status lsm6dsox_power_down(I2cBus& bus, uint8_t addr7) {
  Status a = bus.write_reg(addr7, kRegCtrl1Xl, 0x00);
  Status b = bus.write_reg(addr7, kRegCtrl2G, 0x00);
  Status f = bus.write_reg(addr7, kRegFifoCtrl4, 0x00);
  if (!a.ok()) return a;
  if (!b.ok()) return b;
  return f;
}

Status lsm6dsox_reset_and_configure(I2cBus& bus, const ImuConfig& c, ImuClock& clock,
                                    std::vector<Lsm6dsoxReadback>* readback, int8_t* freq_fine) {
  RIG_ASSIGN_OR_RETURN(const std::vector<Lsm6dsoxRegWrite> seq, lsm6dsox_config_sequence(c));
  const uint8_t a = static_cast<uint8_t>(c.i2c_addr);
  const int rt = c.i2c_retries;
  const std::string at = where(bus, a);

  // 1. Probe.
  uint8_t who = 0;
  if (Status st = read_reg_retry(bus, a, kRegWhoAmI, who, rt); !st.ok())
    return st.with_context(at + ": WHO_AM_I read failed (wiring, address, i2c bus?)");
  if (who != static_cast<uint8_t>(c.expected_who_am_i))
    return device_error(at + ": WHO_AM_I (0x0f) = " + hex2(who) + ", expected " + hex2(c.expected_who_am_i) +
                        " (LSM6DSOX, DS12814 §9.14)");

  // 2. FIFO to bypass before the reset (Linux st_lsm6dsx_reset_device flushes first).
  RIG_RETURN_IF_ERROR(write_reg_retry(bus, a, kRegFifoCtrl4, 0x00, rt).with_context(at + ": FIFO_CTRL4"));
  // 3. Software reset; SW_RESET self-clears (DS Tab 57). Wait as Linux does, then confirm.
  RIG_RETURN_IF_ERROR(write_reg_retry(bus, a, kRegCtrl3C, static_cast<uint8_t>(kCtrl3IfInc | kCtrl3SwReset), rt)
                          .with_context(at + ": CTRL3_C SW_RESET"));
  clock.sleep_until(clock.now_ns() + kResetWaitMs * kMs, nullptr);
  RIG_RETURN_IF_ERROR(wait_bit_clear(bus, a, kRegCtrl3C, kCtrl3SwReset, clock, "SW_RESET"));
  // 4. Reboot memory content (reload trims); BOOT self-clears.
  RIG_RETURN_IF_ERROR(write_reg_retry(bus, a, kRegCtrl3C, static_cast<uint8_t>(kCtrl3IfInc | kCtrl3Boot), rt)
                          .with_context(at + ": CTRL3_C BOOT"));
  clock.sleep_until(clock.now_ns() + kBootWaitMs * kMs, nullptr);
  RIG_RETURN_IF_ERROR(wait_bit_clear(bus, a, kRegCtrl3C, kCtrl3Boot, clock, "BOOT"));

  // 5. The reset really happened: defaults (DS Table 21).
  struct Def {
    uint8_t reg;
    uint8_t value;
  };
  for (const Def& d : {Def{kRegCtrl3C, kCtrl3CDefault}, Def{kRegCtrl1Xl, 0x00}, Def{kRegCtrl2G, 0x00},
                       Def{kRegCtrl9Xl, kCtrl9XlDefault}, Def{kRegCtrl10C, 0x00}, Def{kRegFifoCtrl4, 0x00},
                       Def{kRegWhoAmI, static_cast<uint8_t>(c.expected_who_am_i)}}) {
    uint8_t v = 0;
    RIG_RETURN_IF_ERROR(read_reg_retry(bus, a, d.reg, v, rt).with_context(at + ": reset readback"));
    if (v != d.value)
      return device_error(at + ": after reset " + reg_name(d.reg) + " (" + hex2(d.reg) + ") = " + hex2(v) +
                          ", expected default " + hex2(d.value));
  }

  // 6. Timestamp trim (DS Table 174).
  uint8_t ff = 0;
  RIG_RETURN_IF_ERROR(read_reg_retry(bus, a, kRegInternalFreqFine, ff, rt).with_context(at + ": INTERNAL_FREQ_FINE"));
  if (freq_fine != nullptr) *freq_fine = static_cast<int8_t>(ff);

  // 7. Configure, then 8. read back every register.
  for (const Lsm6dsoxRegWrite& wr : seq)
    RIG_RETURN_IF_ERROR(write_reg_retry(bus, a, wr.reg, wr.value, rt).with_context(at + ": write " + wr.name));
  std::vector<Lsm6dsoxReadback> rb;
  std::string bad;
  for (const Lsm6dsoxRegWrite& wr : seq) {
    uint8_t v = 0;
    RIG_RETURN_IF_ERROR(read_reg_retry(bus, a, wr.reg, v, rt).with_context(at + ": readback " + wr.name));
    rb.push_back({wr.reg, wr.name, wr.value, v});
    if (v != wr.value)
      bad += std::string(bad.empty() ? "" : "; ") + wr.name + " (" + hex2(wr.reg) + "): wrote " + hex2(wr.value) +
             ", read " + hex2(v);
  }
  if (readback != nullptr) *readback = rb;
  if (!bad.empty()) {
    (void)lsm6dsox_power_down(bus, a);
    return device_error(at + ": configuration readback mismatch: " + bad);
  }
  // 9. Clear FIFO_OVR_LATCHED (reset when FIFO_STATUS2 is read, DS Table 121).
  uint8_t st[2];
  RIG_RETURN_IF_ERROR(
      with_retries(rt, [&] { return bus.read_regs(a, kRegFifoStatus1, st, 2); }).with_context(at + ": FIFO_STATUS"));
  return Status::Ok();
}

// ---------------------------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------------------------

class Lsm6dsoxDriver final : public ImuBackend {
 public:
  Lsm6dsoxDriver(ImuConfig c, std::unique_ptr<I2cBus> bus, std::shared_ptr<ImuClock> clock)
      : cfg_(std::move(c)), bus_(std::move(bus)), clock_(std::move(clock)), mapper_(cfg_.clock),
        dec_(FifoDecoder::Options{}) {
    odr_ = *odr_entry(cfg_.odr_hz);
    acc_scale_ = accel_fs(cfg_.accel_fs_g)->mg_per_lsb * 1e-3 * kStandardGravity;
    gyro_scale_ = gyro_fs(cfg_.gyro_fs_dps)->mdps_per_lsb * 1e-3 * kPi / 180.0;
    addr_ = static_cast<uint8_t>(cfg_.i2c_addr);
  }
  ~Lsm6dsoxDriver() override { (void)stop(); }

  const ImuConfig& config() const override { return cfg_; }

  Status start() override {
    if (thread_.joinable()) return failed_precondition(where(*bus_, addr_) + ": start(): already running");
    {
      std::lock_guard<std::mutex> lk(mu_);
      q_.clear();
      brackets_.clear();
      status_ = Status::Ok();
      pub_ = ImuStats{};
      delivered_ = 0;
      diag_ = Lsm6dsoxDiagnostics{};
    }
    ts_ = ImuStats{};
    ts_.fifo_watermark_words = cfg_.fifo_watermark_words;
    err_times_.clear();
    rate_win_.clear();
    last_t_ = kInvalidTime;
    have_ref_ = false;
    backwards_ = 0;
    gap_next_ = false;
    last_gyro_ = Eigen::Vector3d::Zero();
    last_acc_ = Eigen::Vector3d::Zero();
    polls_ = poll_overruns_ = 0;

    std::vector<Lsm6dsoxReadback> rb;
    int8_t ff = 0;
    Status st = lsm6dsox_reset_and_configure(*bus_, cfg_, *clock_, &rb, &ff);
    {
      std::lock_guard<std::mutex> lk(mu_);
      readback_ = rb;
      if (!st.ok()) status_ = st;
    }
    if (!st.ok()) return st;

    tick_ns_ = trimmed_tick_ns(ff);
    ts_.freq_fine = ff;
    ts_.nominal_tick_ns = tick_ns_;
    ChipClockConfig cc = cfg_.clock;
    cc.nominal_tick_ns = tick_ns_;
    mapper_ = ChipClockMapper(cc);
    FifoDecoder::Options o;
    o.pair_by_tag_counter = cfg_.pair_by_tag_counter;
    o.timestamp_decimation = cfg_.timestamp_decimation;
    o.nominal_period_ticks = static_cast<double>(odr_.sim_period_ticks);
    dec_ = FifoDecoder(o);
    t_start_ = clock_->now_ns();
    bracket();
    bracket();
    publish();

    stop_.store(false);
    running_.store(true);
    thread_ = std::thread([this] { run(); });
    return Status::Ok();
  }

  Status stop() override {
    if (!thread_.joinable()) return Status::Ok();
    stop_.store(true);
    clock_->wake();
    thread_.join();
    running_.store(false);
    cv_.notify_all();
    return lsm6dsox_power_down(*bus_, addr_).with_context(where(*bus_, addr_) + ": power down");
  }

  std::size_t read(std::vector<ImuSample>& out, std::chrono::nanoseconds timeout) override {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, timeout, [&] { return !q_.empty() || !running_.load(); });
    const std::size_t n = q_.size();
    out.insert(out.end(), q_.begin(), q_.end());
    q_.clear();
    delivered_ += n;
    return n;
  }

  std::size_t read_brackets(std::vector<ClockBracket>& out) override {
    std::lock_guard<std::mutex> lk(mu_);
    const std::size_t n = brackets_.size();
    out.insert(out.end(), brackets_.begin(), brackets_.end());
    brackets_.clear();
    return n;
  }

  ImuStats stats() const override {
    std::lock_guard<std::mutex> lk(mu_);
    ImuStats s = pub_;
    s.samples = delivered_;
    return s;
  }

  Status status() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return status_;
  }

  std::vector<Lsm6dsoxReadback> readback() const {
    std::lock_guard<std::mutex> lk(mu_);
    return readback_;
  }
  Lsm6dsoxDiagnostics diagnostics() const {
    std::lock_guard<std::mutex> lk(mu_);
    return diag_;
  }

 private:
  // ---- poll thread ----
  void run() {
    const int64_t poll_ns = static_cast<int64_t>(cfg_.poll_period_ms) * kMs;
    const int64_t bracket_ns = static_cast<int64_t>(cfg_.bracket_period_ms) * kMs;
    TimeNs next = clock_->now_ns();
    TimeNs next_bracket = next + poll_ns;
    while (!stop_.load()) {
      next += poll_ns;
      const TimeNs now0 = clock_->now_ns();
      if (now0 > next + poll_ns) {  // fell more than a period behind: re-anchor, do not burst
        ++poll_overruns_;
        next = now0;
      }
      clock_->sleep_until(next, &stop_);
      if (stop_.load()) break;
      ++polls_;
      const TimeNs now = clock_->now_ns();
      if (now >= next_bracket) {
        bracket();
        next_bracket = now + ((now - t_start_ < kWarmupNs) ? poll_ns : bracket_ns);
        if (fatal_) break;
      }
      poll_fifo();
      publish();
      if (fatal_) break;
    }
    publish();
    running_.store(false);
    cv_.notify_all();
  }

  // Counts a failed transaction; turns fatal beyond the budget.
  void count_error(const Status& st) {
    ++ts_.i2c_errors;
    const TimeNs now = clock_->now_ns();
    err_times_.push_back(now);
    while (!err_times_.empty() && err_times_.front() < now - kErrorWindowNs) err_times_.pop_front();
    if (static_cast<int64_t>(err_times_.size()) > cfg_.i2c_error_budget && !fatal_) {
      fatal_ = true;
      std::lock_guard<std::mutex> lk(mu_);
      status_ = device_error(where(*bus_, addr_) + ": I2C error budget exceeded (" +
                             std::to_string(err_times_.size()) + " failed transactions in 10 s > " +
                             std::to_string(cfg_.i2c_error_budget) + "); last: " + st.to_string());
    }
  }

  Status xfer(const std::function<Status()>& fn) {
    Status st = with_retries(cfg_.i2c_retries, fn, [&] { ++ts_.i2c_retries; });
    if (!st.ok()) count_error(st);
    return st;
  }

  void bracket() {
    uint8_t a[4], b[4];
    TimeNs t0 = 0, t1 = 0, t2 = 0;
    Status st = xfer([&] {
      t0 = clock_->now_ns();
      Status s = bus_->read_regs(addr_, kRegTimestamp0, a, 4);
      t1 = clock_->now_ns();
      return s;
    });
    if (!st.ok()) return;
    st = xfer([&] { return bus_->read_regs(addr_, kRegTimestamp0, b, 4); });
    t2 = clock_->now_ns();
    if (!st.ok()) return;
    // Consistency of two back-to-back reads: B - A must be within the elapsed time. A torn read (bytes latched on
    // both sides of a carry) is too high by >= 256 ticks, so B < A or B - A far too large (ascending reads).
    const uint32_t ra = le32(a), rb = le32(b);
    const uint32_t d = rb - ra;
    const double allowed = static_cast<double>(t2 - t0) / tick_ns_ * 1.05 + 2.0;
    if (static_cast<double>(d) > allowed) {
      ++ts_.torn_timestamp_reads;
      return;
    }
    const int64_t u = have_ref_ ? FifoDecoder::unwrap(ra, ref_ticks_) : static_cast<int64_t>(ra);
    Status ms = mapper_.add_bracket(t0, u, t1);
    if (!ms.ok() && have_ref_ && u < ref_ticks_) {
      if (++backwards_ >= kBackwardsResetCount) {  // the counter restarted (chip reset): start over
        mapper_.reset();
        backwards_ = 0;
        std::vector<RawSlot> tmp;
        dec_.mark_discontinuity(tmp);
        process(tmp);
        (void)mapper_.add_bracket(t0, u, t1);
      }
    } else {
      backwards_ = 0;
    }
    ref_ticks_ = u;
    have_ref_ = true;
    dec_.set_unwrap_reference(u);
    std::lock_guard<std::mutex> lk(mu_);
    if (brackets_.size() >= kMaxBrackets) brackets_.pop_front();
    brackets_.push_back(ClockBracket{t0, u, t1});
  }

  void poll_fifo() {
    uint8_t s[2] = {0, 0};
    if (!xfer([&] { return bus_->read_regs(addr_, kRegFifoStatus1, s, 2); }).ok()) return;
    const int level = fifo_level(s[0], s[1]);
    ts_.fifo_level_max = std::max(ts_.fifo_level_max, level);
    slots_.clear();
    if ((s[1] & (kFifoStatus2OvrIa | kFifoStatus2OvrLatched)) != 0) {
      ++ts_.fifo_overruns;
      dec_.mark_discontinuity(slots_);
    }
    if ((s[1] & kFifoStatus2FullIa) != 0) ++ts_.fifo_full_flags;
    int remaining = std::min(level, kFifoCapacityWords);
    buf_.resize(static_cast<std::size_t>(cfg_.fifo_burst_words) * kFifoWordBytes);
    while (remaining > 0 && !stop_.load()) {
      const int k = std::min(remaining, cfg_.fifo_burst_words);
      const std::size_t nbytes = static_cast<std::size_t>(k) * kFifoWordBytes;
      if (!xfer([&] { return bus_->read_regs(addr_, kRegFifoDataOutTag, buf_.data(), nbytes); }).ok()) {
        dec_.mark_discontinuity(slots_);
        break;
      }
      ts_.fifo_words += static_cast<uint64_t>(k);
      dec_.push(buf_.data(), static_cast<std::size_t>(k), slots_);
      remaining -= k;
    }
    process(slots_);
  }

  void process(const std::vector<RawSlot>& slots) {
    if (slots.empty()) return;
    const double period_ns = 1e9 / cfg_.odr_hz;
    std::vector<ImuSample> out;
    out.reserve(slots.size());
    const double temp = dec_.last_temperature_c();
    for (const RawSlot& r : slots) {
      if (r.ticks < 0) continue;
      TimeNs t = mapper_.map(r.ticks);
      if (t == kInvalidTime) continue;
      ImuSample smp;
      smp.chip_ticks = r.ticks;
      if (!mapper_.ready()) smp.flags |= kImuFlagTimeExtrapolated;
      if (last_t_ != kInvalidTime && t <= last_t_) t = last_t_ + 1;
      if (last_t_ != kInvalidTime) {
        const int64_t dt = t - last_t_;
        ts_.max_gap_ns = std::max(ts_.max_gap_ns, dt);
        if (static_cast<double>(dt) > cfg_.gap_factor * period_ns) {
          ++ts_.gaps;
          smp.flags |= kImuFlagAfterGap;
        }
      }
      if (r.after_gap) smp.flags |= kImuFlagAfterGap;
      smp.t_ns = t;
      bool gsat = false, asat = false;
      if (r.has_gyro) {
        for (int i = 0; i < 3; ++i) {
          last_gyro_[i] = r.gyro[static_cast<std::size_t>(i)] * gyro_scale_;
          gsat = gsat || r.gyro[static_cast<std::size_t>(i)] == 32767 || r.gyro[static_cast<std::size_t>(i)] == -32768;
        }
      } else {
        smp.flags |= kImuFlagGyroStale;
      }
      if (r.has_accel) {
        for (int i = 0; i < 3; ++i) {
          last_acc_[i] = r.accel[static_cast<std::size_t>(i)] * acc_scale_;
          asat = asat || r.accel[static_cast<std::size_t>(i)] == 32767 ||
                 r.accel[static_cast<std::size_t>(i)] == -32768;
        }
      } else {
        smp.flags |= kImuFlagAccStale;
      }
      if (gsat) {
        smp.flags |= kImuFlagGyroSaturated;
        ++ts_.gyro_saturations;
      }
      if (asat) {
        smp.flags |= kImuFlagAccSaturated;
        ++ts_.acc_saturations;
      }
      smp.gyro_rps = last_gyro_;
      smp.acc_mps2 = last_acc_;
      smp.temp_c = static_cast<float>(temp);
      last_t_ = t;
      ts_.last_t_ns = t;
      rate_win_.push_back(t);
      while (!rate_win_.empty() && rate_win_.front() < t - 1000 * kMs) rate_win_.pop_front();
      out.push_back(smp);
    }
    if (rate_win_.size() >= 2)
      ts_.rate_hz_estimate =
          static_cast<double>(rate_win_.size() - 1) * 1e9 / static_cast<double>(rate_win_.back() - rate_win_.front());
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (ImuSample& smp : out) {
        if (q_.size() >= cfg_.queue_capacity) {
          ++ts_.queue_overruns;
          gap_next_ = true;
          continue;
        }
        if (gap_next_) {
          smp.flags |= kImuFlagAfterGap;
          gap_next_ = false;
        }
        q_.push_back(smp);
      }
    }
    cv_.notify_all();
  }

  void publish() {
    const FifoDecoderStats& d = dec_.stats();
    ts_.unknown_tags = d.unknown_tags;
    ts_.timestamp_regressions = d.timestamp_regressions;
    ts_.partial_slots = d.partial_slots;
    ts_.slots_without_timestamp = d.slots_without_ts;
    ts_.temperature_words = d.temperature_words;
    ts_.clock = mapper_.stats();
    std::lock_guard<std::mutex> lk(mu_);
    const uint64_t delivered = delivered_;
    pub_ = ts_;
    pub_.samples = delivered;
    diag_.gyro_words = d.gyro_words;
    diag_.accel_words = d.accel_words;
    diag_.timestamp_words = d.timestamp_words;
    diag_.temperature_words = d.temperature_words;
    diag_.cfg_change_words = d.cfg_change_words;
    diag_.odd_parity_words = d.odd_parity_words;
    diag_.lost_slots = d.lost_slots;
    diag_.untimed_slots = d.untimed_slots;
    diag_.learned_period_ticks = dec_.period_ticks();
    diag_.polls = polls_;
    diag_.poll_overruns = poll_overruns_;
  }

  const ImuConfig cfg_;
  std::unique_ptr<I2cBus> bus_;
  std::shared_ptr<ImuClock> clock_;
  OdrEntry odr_{};
  double acc_scale_ = 0, gyro_scale_ = 0;
  uint8_t addr_ = 0;

  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> running_{false};

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<ImuSample> q_;
  std::deque<ClockBracket> brackets_;
  ImuStats pub_;
  uint64_t delivered_ = 0;
  Status status_;
  std::vector<Lsm6dsoxReadback> readback_;
  Lsm6dsoxDiagnostics diag_;

  // Owned by the poll thread (and by start() before the thread exists).
  ChipClockMapper mapper_;
  FifoDecoder dec_;
  ImuStats ts_;
  double tick_ns_ = kNominalTickNs;
  TimeNs t_start_ = 0;
  TimeNs last_t_ = kInvalidTime;
  bool have_ref_ = false;
  int64_t ref_ticks_ = 0;
  int backwards_ = 0;
  bool gap_next_ = false;
  bool fatal_ = false;
  std::deque<TimeNs> err_times_;
  std::deque<TimeNs> rate_win_;
  std::vector<RawSlot> slots_;
  std::vector<uint8_t> buf_;
  Eigen::Vector3d last_gyro_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_acc_ = Eigen::Vector3d::Zero();
  uint64_t polls_ = 0, poll_overruns_ = 0;
};

Result<std::unique_ptr<ImuBackend>> make_lsm6dsox(const ImuConfig& c, std::unique_ptr<I2cBus> bus,
                                                   std::shared_ptr<ImuClock> clock) {
  RIG_ASSIGN_OR_RETURN(auto seq, lsm6dsox_config_sequence(c));
  (void)seq;
  if (!bus) return invalid_argument("make_lsm6dsox(" + c.id + "): no bus");
  if (!clock) clock = real_imu_clock();
  return std::unique_ptr<ImuBackend>(std::make_unique<Lsm6dsoxDriver>(c, std::move(bus), std::move(clock)));
}

std::vector<Lsm6dsoxReadback> lsm6dsox_last_readback(const ImuBackend& backend) {
  const auto* d = dynamic_cast<const Lsm6dsoxDriver*>(&backend);
  return d != nullptr ? d->readback() : std::vector<Lsm6dsoxReadback>{};
}

Lsm6dsoxDiagnostics lsm6dsox_diagnostics(const ImuBackend& backend) {
  const auto* d = dynamic_cast<const Lsm6dsoxDriver*>(&backend);
  return d != nullptr ? d->diagnostics() : Lsm6dsoxDiagnostics{};
}

namespace {
nlohmann::json num(double v) { return std::isfinite(v) ? nlohmann::json(v) : nlohmann::json(nullptr); }
nlohmann::json summary(const SummaryStats& s) {
  return {{"count", s.count}, {"min", num(s.min)}, {"max", num(s.max)}, {"mean", num(s.mean)},
          {"p50", num(s.p50)}, {"p99", num(s.p99)}};
}
}  // namespace

std::string clock_fit_json(const ImuConfig& c, const ImuStats& s) {
  const ClockFit& f = s.clock.fit;
  nlohmann::json j;
  j["schema"] = 1;
  j["imu_id"] = c.id;
  j["method"] = c.clock.method == ClockFitMethod::kLowerEnvelope ? "lower_envelope" : "robust_linear";
  j["window_brackets"] = c.clock.window_brackets;
  j["freq_fine"] = s.freq_fine;
  j["trimmed_tick_ns"] = num(s.nominal_tick_ns);
  j["fit"] = {{"valid", f.valid},
              {"slope_ns_per_tick", num(f.slope_ns_per_tick)},
              {"slope_ppm_vs_trimmed",
               num(std::isfinite(s.nominal_tick_ns) ? (f.slope_ns_per_tick / s.nominal_tick_ns - 1.0) * 1e6 : kNaN)},
              {"offset_ns", num(f.offset_ns)},
              {"ref_ticks", f.ref_ticks},
              {"slope_clamped", f.slope_clamped}};
  j["residual_ns"] = summary(s.clock.residual_ns);
  j["bracket_width_ns"] = summary(s.clock.bracket_width_ns);
  j["brackets_added"] = s.clock.brackets_added;
  j["brackets_rejected"] = s.clock.brackets_rejected;
  j["slope_clamp_events"] = s.clock.slope_clamp_events;
  j["torn_timestamp_reads"] = s.torn_timestamp_reads;
  return j.dump(2);
}

double mean_acc_norm(const std::vector<ImuSample>& v) {
  if (v.empty()) return kNaN;
  double s = 0;
  for (const ImuSample& x : v) s += x.acc_mps2.norm();
  return s / static_cast<double>(v.size());
}

}  // namespace rig
