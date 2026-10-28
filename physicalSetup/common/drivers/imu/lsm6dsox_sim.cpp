// SPDX-License-Identifier: MIT
// Lsm6dsoxSim: register-level simulator. Model and assumptions: rig/drivers/imu/lsm6dsox_sim.hpp.
#include "rig/drivers/imu/lsm6dsox_sim.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>

#include "rig/drivers/imu/lsm6dsox_regs.hpp"
#include "rig/drivers/imu_backend.hpp"

namespace rig {

using namespace lsm6dsox;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr uint8_t kSimUnknownTag = 0x1F;

uint8_t with_parity(uint8_t b) {
  // SIMULATOR choice: bit 0 makes the byte's popcount even. The real definition is not documented in DS12814
  // ("parity check of TAG content"); the driver does not depend on it.
  int n = 0;
  for (int i = 1; i < 8; ++i) n += (b >> i) & 1;
  return static_cast<uint8_t>((b & 0xFE) | (n & 1));
}

bool is_read_only(uint8_t a) {
  // DS Table 21 "Type" column: R registers.
  return a == kRegWhoAmI || (a >= 0x1A && a <= 0x1E) || (a >= 0x20 && a <= 0x2D) || (a >= 0x35 && a <= 0x3B) ||
         (a >= 0x40 && a <= 0x43) || (a >= 0x49 && a <= 0x55) || a == kRegInternalFreqFine ||
         (a >= kRegFifoDataOutTag && a <= kRegFifoDataOutZH);
}

int16_t sat16(double v) {
  const double r = std::nearbyint(v);
  if (r > 32767.0) return 32767;
  if (r < -32768.0) return -32768;
  return static_cast<int16_t>(r);
}

}  // namespace

Lsm6dsoxSimOptions Lsm6dsoxSimOptions::from_config(const ImuSimConfig& c, int i2c_addr) {
  Lsm6dsoxSimOptions o;
  o.addr7 = static_cast<uint8_t>(i2c_addr);
  o.drift_ppm = c.chip_clock_drift_ppm;
  o.counter_start = static_cast<uint32_t>(static_cast<uint64_t>(c.chip_clock_offset_ticks));
  o.accel_noise_std_mps2 = c.accel_noise_std_mps2;
  o.gyro_noise_std_rps = c.gyro_noise_std_rps;
  o.motion = c.motion;
  o.nack_rate = c.i2c_error_rate;
  o.seed = c.seed;
  o.record_truth = false;  // backend "sim" runs unbounded and nobody drains the truth log (R1 F2); tests opt in
  return o;
}

struct Lsm6dsoxSim::Impl {
  using Word = std::array<uint8_t, kFifoWordBytes>;

  Lsm6dsoxSimOptions opt;
  std::shared_ptr<ImuClock> clock;
  mutable std::mutex mu;
  std::mt19937_64 rng;

  std::array<uint8_t, 256> regs{};
  uint8_t ptr = 0;
  std::deque<Word> fifo;
  Word latch{};
  bool ovr_sticky = false;
  bool ovr_latched = false;
  uint64_t overwritten = 0;
  uint64_t n_transactions = 0;
  uint64_t n_nacks = 0;
  int fail_next = 0;
  bool present = true;
  bool inject_unknown = false;

  double t0 = 0.0;             // host time of oscillator tick 0
  double tick = 25000.0;       // true tick [host ns]
  bool counter_en = false;
  int64_t en_osc = 0;          // oscillator tick at which the counter was (re)started
  uint32_t counter_start = 0;
  double boot_until = -1.0;
  double reset_until = -1.0;

  // Slot grid.
  bool grid = false;
  int64_t origin_osc = 0;
  int64_t period = 48;
  int64_t next_slot = 1;
  int g_every = 0, xl_every = 0;  // 0 = not batched
  int ts_every = 0;
  int temp_every = 0;
  int words_per_slot = 0;

  std::vector<Lsm6dsoxSimTruth> truth;

  Impl(Lsm6dsoxSimOptions o, std::shared_ptr<ImuClock> c) : opt(std::move(o)), clock(std::move(c)), rng(opt.seed) {
    t0 = static_cast<double>(clock->now_ns());
    tick = trimmed_tick_ns(opt.freq_fine) / (1.0 + opt.drift_ppm * 1e-6);
    counter_start = opt.counter_start;
    power_on_defaults();
  }

  void power_on_defaults() {
    regs.fill(0);
    regs[0x02] = kPinCtrlDefault;
    regs[kRegWhoAmI] = opt.who_am_i;
    regs[kRegCtrl3C] = kCtrl3CDefault;
    regs[kRegCtrl9Xl] = kCtrl9XlDefault;
    regs[kRegInternalFreqFine] = static_cast<uint8_t>(opt.freq_fine);
    fifo.clear();
    ovr_sticky = ovr_latched = false;
    counter_en = false;
    grid = false;
  }

  double osc_at(double t) const { return (t - t0) / tick; }
  double host_of_osc(double osc) const { return t0 + osc * tick; }
  int64_t counter_unwrapped_at(double t) const {
    if (!counter_en) return 0;
    return static_cast<int64_t>(counter_start) + static_cast<int64_t>(std::floor(osc_at(t))) - en_osc;
  }

  bool booting(double t) const { return t < boot_until; }

  void push_word(const Word& w) {
    if (static_cast<int>(fifo.size()) >= opt.fifo_capacity_words) {
      fifo.pop_front();
      ++overwritten;
      ovr_sticky = true;
      ovr_latched = true;
    }
    fifo.push_back(w);
  }

  Word make_word(uint8_t sensor, uint8_t cnt, const uint8_t* data6) const {
    Word w{};
    w[0] = with_parity(static_cast<uint8_t>((sensor << 3) | ((cnt & 3) << 1)));
    std::copy(data6, data6 + 6, w.begin() + 1);
    return w;
  }

  static void put16(uint8_t* p, int16_t v) {
    const uint16_t u = static_cast<uint16_t>(v);
    p[0] = static_cast<uint8_t>(u & 0xFF);
    p[1] = static_cast<uint8_t>(u >> 8);
  }

  double accel_mg_per_lsb() const {
    const uint8_t code = (regs[kRegCtrl1Xl] >> 2) & 0x3;
    for (const AccelFs& f : kAccelFsTable)
      if (f.code == code) return f.mg_per_lsb;
    return 0.061;
  }
  double gyro_mdps_per_lsb() const {
    const uint8_t code3 = (regs[kRegCtrl2G] >> 1) & 0x7;
    for (const GyroFs& f : kGyroFsTable)
      if (f.code3 == code3) return f.mdps_per_lsb;
    return 8.75;
  }

  void generate_slot(int64_t n) {
    const int64_t slot_osc = origin_osc + n * period;
    const double th = host_of_osc(static_cast<double>(slot_osc));
    const bool g = g_every > 0 && n % g_every == 0;
    const bool xl = xl_every > 0 && n % xl_every == 0;
    const bool ts = ts_every > 0 && n % ts_every == 0;
    const uint8_t cnt = static_cast<uint8_t>(n & 3);
    const double ts_s = (th - t0) * 1e-9;

    Eigen::Vector3d acc = opt.gravity_body, gyr = opt.gyro_bias_rps;
    if (opt.motion) {
      for (int i = 0; i < 3; ++i) {
        acc[i] += opt.accel_amp_mps2[i] * std::sin(2.0 * kPi * opt.motion_freq_hz[i] * ts_s + i);
        gyr[i] += opt.gyro_amp_rps[i] * std::sin(2.0 * kPi * opt.motion_freq_hz[i] * ts_s + 0.5 * i);
      }
    }
    std::normal_distribution<double> nd(0.0, 1.0);
    if (opt.accel_noise_std_mps2 > 0)
      for (int i = 0; i < 3; ++i) acc[i] += opt.accel_noise_std_mps2 * nd(rng);
    if (opt.gyro_noise_std_rps > 0)
      for (int i = 0; i < 3; ++i) gyr[i] += opt.gyro_noise_std_rps * nd(rng);

    uint8_t d[6];
    std::vector<Word> words;
    if (ts) {
      const int64_t c = counter_en ? static_cast<int64_t>(counter_start) + slot_osc - en_osc : 0;
      const uint32_t raw = static_cast<uint32_t>(static_cast<uint64_t>(c));
      d[0] = static_cast<uint8_t>(raw);
      d[1] = static_cast<uint8_t>(raw >> 8);
      d[2] = static_cast<uint8_t>(raw >> 16);
      d[3] = static_cast<uint8_t>(raw >> 24);
      d[4] = d[5] = 0;
      if (!opt.timestamp_after_data) words.push_back(make_word(kTagTimestamp, cnt, d));
    }
    if (g) {
      const double k = 180.0 / kPi * 1000.0 / gyro_mdps_per_lsb();
      for (int i = 0; i < 3; ++i) put16(d + 2 * i, sat16(gyr[i] * k));
      words.push_back(make_word(kTagGyroNC, cnt, d));
    }
    if (xl) {
      const double k = 1000.0 / kStandardGravity / accel_mg_per_lsb();
      for (int i = 0; i < 3; ++i) put16(d + 2 * i, sat16(acc[i] * k));
      words.push_back(make_word(kTagAccelNC, cnt, d));
    }
    if (ts && opt.timestamp_after_data) {
      const int64_t c = counter_en ? static_cast<int64_t>(counter_start) + slot_osc - en_osc : 0;
      const uint32_t raw = static_cast<uint32_t>(static_cast<uint64_t>(c));
      uint8_t t4[6] = {static_cast<uint8_t>(raw), static_cast<uint8_t>(raw >> 8), static_cast<uint8_t>(raw >> 16),
                       static_cast<uint8_t>(raw >> 24), 0, 0};
      words.push_back(make_word(kTagTimestamp, cnt, t4));
    }
    if (temp_every > 0 && n % temp_every == 0) {
      uint8_t t6[6] = {0, 0, 0, 0, 0, 0};
      put16(t6, sat16((opt.temperature_c - kTempOffsetC) * kTempLsbPerC));
      words.push_back(make_word(kTagTemperature, cnt, t6));
    }
    if (inject_unknown) {
      uint8_t z[6] = {0, 0, 0, 0, 0, 0};
      words.push_back(make_word(kSimUnknownTag, cnt, z));
      inject_unknown = false;
    }
    for (const Word& w : words) push_word(w);
    if (opt.record_truth && (g || xl) && counter_en) {
      Lsm6dsoxSimTruth tr;
      tr.ticks = static_cast<int64_t>(counter_start) + slot_osc - en_osc;
      tr.t_host_ns = static_cast<TimeNs>(std::llround(th));
      tr.gyro_rps = gyr;
      tr.acc_mps2 = acc;
      truth.push_back(tr);
    }
  }

  void catch_up(double t) {
    if (!grid) return;
    const double due = (osc_at(t) - static_cast<double>(origin_osc)) / static_cast<double>(period);
    if (due < static_cast<double>(next_slot)) return;
    int64_t last = static_cast<int64_t>(std::floor(due));
    // Far behind (nobody read for a long time): everything older than the FIFO depth is overwritten anyway.
    const int64_t keep = 2 * static_cast<int64_t>(opt.fifo_capacity_words) + 8;
    if (last - next_slot > keep) {
      overwritten += static_cast<uint64_t>((last - keep - next_slot) * std::max(1, words_per_slot));
      next_slot = last - keep;
      ovr_sticky = ovr_latched = true;
    }
    for (; next_slot <= last; ++next_slot) generate_slot(next_slot);
  }

  void reconfigure(double t) {
    catch_up(t);
    const uint8_t mode = regs[kRegFifoCtrl4] & 0x7;
    if (mode == kFifoModeBypass) {
      fifo.clear();
      ovr_sticky = ovr_latched = false;
    }
    const uint8_t bdr_xl = regs[kRegFifoCtrl3] & 0xF, bdr_g = regs[kRegFifoCtrl3] >> 4;
    const uint8_t odr_xl = regs[kRegCtrl1Xl] >> 4, odr_g = regs[kRegCtrl2G] >> 4;
    const auto per = [](uint8_t code) -> int64_t {
      const auto e = odr_entry_from_code(code);
      return e ? e->sim_period_ticks : 0;
    };
    const int64_t p_xl = (odr_xl != 0) ? per(bdr_xl) : 0;
    const int64_t p_g = (odr_g != 0) ? per(bdr_g) : 0;
    grid = mode == kFifoModeContinuous && (p_xl > 0 || p_g > 0);
    if (!grid) return;
    period = (p_xl > 0 && p_g > 0) ? std::min(p_xl, p_g) : std::max(p_xl, p_g);
    xl_every = p_xl > 0 ? static_cast<int>(p_xl / period) : 0;
    g_every = p_g > 0 ? static_cast<int>(p_g / period) : 0;
    const uint8_t dec = regs[kRegFifoCtrl4] >> 6;
    ts_every = dec == 1 ? 1 : dec == 2 ? 8 : dec == 3 ? 32 : 0;
    const uint8_t tcode = (regs[kRegFifoCtrl4] >> 4) & 0x3;
    const double temp_hz = tcode == 1 ? 1.6 : tcode == 2 ? 12.5 : tcode == 3 ? 52.0 : 0.0;
    const double slot_hz = 1e9 / (static_cast<double>(period) * tick);
    temp_every = temp_hz > 0 ? std::max(1, static_cast<int>(std::lround(slot_hz / temp_hz))) : 0;
    words_per_slot = (g_every > 0) + (xl_every > 0) + (ts_every > 0);
    origin_osc = static_cast<int64_t>(std::ceil(osc_at(t)));
    next_slot = 1;
  }

  uint8_t read_byte(uint8_t a, double t, int byte_index, double byte_ns) {
    switch (a) {
      case kRegCtrl3C: {
        uint8_t v = regs[a];
        if (t < reset_until) v |= kCtrl3SwReset;
        if (booting(t)) v |= kCtrl3Boot;
        return v;
      }
      case kRegFifoStatus1:
        return static_cast<uint8_t>(fifo.size() & 0xFF);
      case kRegFifoStatus2: {
        const int level = static_cast<int>(fifo.size());
        const int wtm = regs[kRegFifoCtrl1] | ((regs[kRegFifoCtrl2] & kFifoCtrl2Wtm8) << 8);
        uint8_t v = static_cast<uint8_t>((level >> 8) & kFifoStatus2DiffMask);
        if (wtm > 0 && level >= wtm) v |= kFifoStatus2WtmIa;
        if (ovr_sticky) v |= kFifoStatus2OvrIa;
        if (level + std::max(1, words_per_slot) > opt.fifo_capacity_words) v |= kFifoStatus2FullIa;
        if (ovr_latched) v |= kFifoStatus2OvrLatched;
        ovr_latched = false;  // "reset when this register is read" (DS Table 121)
        return v;
      }
      case 0x40:
      case 0x41:
      case 0x42:
      case 0x43: {
        const double tb = opt.tear_timestamp_reads ? t + byte_index * byte_ns : t;
        const uint32_t c = static_cast<uint32_t>(static_cast<uint64_t>(counter_unwrapped_at(tb)));
        return static_cast<uint8_t>(c >> (8 * (a - 0x40)));
      }
      case kRegFifoDataOutTag:
        if (!fifo.empty()) {
          latch = fifo.front();
          fifo.pop_front();
          ovr_sticky = false;
        } else {
          latch.fill(0);
        }
        return latch[0];
      default:
        if (a > kRegFifoDataOutTag && a <= kRegFifoDataOutZH) return latch[a - kRegFifoDataOutTag];
        return regs[a];
    }
  }

  void write_byte(uint8_t a, uint8_t v, double t) {
    if (a == kRegTimestamp2 && v == 0xAA) {  // counter reset (Linux st_lsm6dsx TS_RESET)
      en_osc = static_cast<int64_t>(std::floor(osc_at(t)));
      counter_start = 0;
      return;
    }
    if (is_read_only(a) || static_cast<int>(a) == opt.stuck_register) return;
    if (a == kRegCtrl3C) {
      if (v & kCtrl3SwReset) {
        catch_up(t);
        power_on_defaults();
        reset_until = t + static_cast<double>(opt.reset_busy_ns);
        return;
      }
      if (v & kCtrl3Boot) boot_until = t + static_cast<double>(opt.boot_busy_ns);
      regs[a] = static_cast<uint8_t>(v & ~(kCtrl3Boot | kCtrl3SwReset));
      return;
    }
    if (a == kRegCtrl10C) {
      catch_up(t);
      const bool en = (v & kCtrl10TimestampEn) != 0;
      if (en && !counter_en) {
        en_osc = static_cast<int64_t>(std::floor(osc_at(t)));
        counter_start = opt.counter_start;
      }
      counter_en = en;
      regs[a] = v;
      return;
    }
    regs[a] = v;
    if (a == kRegFifoCtrl3 || a == kRegFifoCtrl4 || a == kRegCtrl1Xl || a == kRegCtrl2G) reconfigure(t);
  }

  void step_ptr() {
    if ((regs[kRegCtrl3C] & kCtrl3IfInc) == 0) return;
    ptr = (ptr == kRegFifoDataOutZH) ? kRegFifoDataOutTag : static_cast<uint8_t>(ptr + 1);
  }

  // Common transaction frame. Returns false (and accounts the time) on a NACK.
  bool begin(uint8_t addr7, double& t_start) {
    ++n_transactions;
    const double now = static_cast<double>(clock->now_ns());
    double pre = 0.0;
    if (clock->is_virtual()) {
      std::exponential_distribution<double> ex(1.0 / std::max(1.0, opt.latency_exp_mean_ns));
      std::uniform_real_distribution<double> u01(0.0, 1.0);
      pre = static_cast<double>(opt.pre_latency_ns) + ex(rng);
      if (u01(rng) < opt.spike_probability) pre += u01(rng) * static_cast<double>(opt.spike_max_ns);
    }
    t_start = now + pre;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    bool nack = !present || addr7 != opt.addr7 || booting(t_start) || fail_next > 0 ||
                (opt.nack_rate > 0 && u01(rng) < opt.nack_rate);
    if (fail_next > 0) --fail_next;
    if (nack) {
      ++n_nacks;
      finish(now, t_start + 9.0 * 1e9 / opt.bus_hz);
      return false;
    }
    return true;
  }

  void finish(double now, double t_end) {
    if (!clock->is_virtual()) return;
    std::exponential_distribution<double> ex(1.0 / std::max(1.0, opt.latency_exp_mean_ns));
    const double post = static_cast<double>(opt.post_latency_ns) + ex(rng);
    clock->advance(static_cast<int64_t>(std::llround(t_end + post - now)));
  }
};

Lsm6dsoxSim::Lsm6dsoxSim(Lsm6dsoxSimOptions opt, std::shared_ptr<ImuClock> clock)
    : impl_(std::make_unique<Impl>(std::move(opt), clock ? std::move(clock) : real_imu_clock())) {}
Lsm6dsoxSim::~Lsm6dsoxSim() = default;

Status Lsm6dsoxSim::write_read(uint8_t addr7, const uint8_t* w, std::size_t wn, uint8_t* r, std::size_t rn) {
  if (addr7 > 0x7F || (wn > 0 && w == nullptr) || (rn > 0 && r == nullptr))
    return invalid_argument("sim:lsm6dsox: bad arguments");
  Impl& m = *impl_;
  std::lock_guard<std::mutex> lk(m.mu);
  const double now = static_cast<double>(m.clock->now_ns());
  double t;
  if (!m.begin(addr7, t)) return device_error("sim:lsm6dsox addr 0x" + std::to_string(addr7) + ": NACK");
  const double virt = m.clock->is_virtual() ? 1.0 : 0.0;
  const double byte_ns = virt * 9.0 * 1e9 / m.opt.bus_hz;
  double tb = t + byte_ns;  // address byte
  for (std::size_t i = 0; i < wn; ++i) {
    // A write before a read sets the register pointer (sub-address); further bytes would be register writes.
    if (i == 0) {
      m.ptr = w[0];
    } else {
      m.catch_up(tb);
      m.write_byte(m.ptr, w[i], tb);
      m.step_ptr();
    }
    tb += byte_ns;
  }
  if (rn > 0) {
    tb += byte_ns;  // repeated start + address
    m.catch_up(tb);
    for (std::size_t i = 0; i < rn; ++i) {
      r[i] = m.read_byte(m.ptr, tb, static_cast<int>(i), byte_ns);
      m.step_ptr();
    }
    tb += static_cast<double>(rn) * byte_ns;
  }
  m.finish(now, tb);
  return Status::Ok();
}

Status Lsm6dsoxSim::write(uint8_t addr7, const uint8_t* w, std::size_t wn) { return write_read(addr7, w, wn, nullptr, 0); }

void Lsm6dsoxSim::fail_next(int n) {
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->fail_next = n;
}
void Lsm6dsoxSim::set_nack_rate(double p) {
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->opt.nack_rate = p;
}
void Lsm6dsoxSim::set_present(bool present) {
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->present = present;
}
void Lsm6dsoxSim::inject_unknown_tag() {
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->inject_unknown = true;
}
uint8_t Lsm6dsoxSim::peek(uint8_t reg) const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->regs[reg];
}
std::vector<Lsm6dsoxSimTruth> Lsm6dsoxSim::drain_truth() {
  std::lock_guard<std::mutex> lk(impl_->mu);
  std::vector<Lsm6dsoxSimTruth> out;
  out.swap(impl_->truth);
  return out;
}
double Lsm6dsoxSim::host_time_of_ticks(double ticks) const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  const Impl& m = *impl_;
  return m.host_of_osc(static_cast<double>(m.en_osc) + ticks - static_cast<double>(m.counter_start));
}
uint64_t Lsm6dsoxSim::overwritten_words() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->overwritten;
}
uint64_t Lsm6dsoxSim::transactions() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->n_transactions;
}
uint64_t Lsm6dsoxSim::nacks() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->n_nacks;
}
int Lsm6dsoxSim::fifo_level() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return static_cast<int>(impl_->fifo.size());
}
double Lsm6dsoxSim::true_tick_ns() const { return impl_->tick; }

}  // namespace rig
