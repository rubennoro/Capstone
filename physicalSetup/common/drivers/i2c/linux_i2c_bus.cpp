// SPDX-License-Identifier: MIT
// LinuxI2cBus: /dev/i2c-N via the i2c-dev I2C_RDWR ioctl (linux/i2c-dev.h, Documentation/i2c/dev-interface.rst).
// write_read() sends one I2C_RDWR with two messages (write, then I2C_M_RD) so the kernel issues a repeated start
// between them (no STOP), as the LSM6DSOX read protocol requires (DS12814 §5.1.2 Tables 13/14).
// Errors: EREMOTEIO / ENXIO (NACK), EAGAIN (arbitration lost) -> kDeviceError; ETIMEDOUT -> kTimeout;
// anything else -> kIoError. EINTR is retried. No retries here: that is the caller's policy (I2cBus contract).
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

#include "rig/drivers/i2c_bus.hpp"

namespace rig {

namespace {

// Kernel limit per i2c_msg in i2c-dev (I2C_RDWR rejects len > 8192: drivers/i2c/i2c-dev.c).
constexpr std::size_t kMaxMsgLen = 8192;

class LinuxI2cBus final : public I2cBus {
 public:
  LinuxI2cBus(int fd, std::string path) : fd_(fd), path_(std::move(path)) {}
  ~LinuxI2cBus() override {
    if (fd_ >= 0) ::close(fd_);
  }
  LinuxI2cBus(const LinuxI2cBus&) = delete;
  LinuxI2cBus& operator=(const LinuxI2cBus&) = delete;

  Status write_read(uint8_t addr7, const uint8_t* w, std::size_t wn, uint8_t* r, std::size_t rn) override {
    if (addr7 > 0x7F) return invalid_argument(path_ + ": address " + std::to_string(addr7) + " > 0x7f");
    if (wn > kMaxMsgLen || rn > kMaxMsgLen) return invalid_argument(path_ + ": transfer longer than 8192 bytes");
    if ((wn > 0 && w == nullptr) || (rn > 0 && r == nullptr)) return invalid_argument(path_ + ": null buffer");
    if (wn == 0 && rn == 0) return invalid_argument(path_ + ": empty transaction");
    i2c_msg msgs[2];
    unsigned n = 0;
    if (wn > 0) {
      msgs[n].addr = addr7;
      msgs[n].flags = 0;
      msgs[n].len = static_cast<__u16>(wn);
      msgs[n].buf = const_cast<__u8*>(w);  // the kernel does not write a write-message buffer
      ++n;
    }
    if (rn > 0) {
      msgs[n].addr = addr7;
      msgs[n].flags = I2C_M_RD;
      msgs[n].len = static_cast<__u16>(rn);
      msgs[n].buf = r;
      ++n;
    }
    i2c_rdwr_ioctl_data data{msgs, n};
    int rc;
    do {
      rc = ::ioctl(fd_, I2C_RDWR, &data);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) return errno_status(errno, addr7);
    if (static_cast<unsigned>(rc) != n)
      return io_error(where(addr7) + ": I2C_RDWR completed " + std::to_string(rc) + " of " + std::to_string(n) +
                      " messages");
    return Status::Ok();
  }

  Status write(uint8_t addr7, const uint8_t* w, std::size_t wn) override {
    if (wn == 0) return invalid_argument(path_ + ": empty write");
    return write_read(addr7, w, wn, nullptr, 0);
  }

  std::string name() const override { return path_; }

 private:
  std::string where(uint8_t addr7) const {
    char b[8];
    std::snprintf(b, sizeof b, "0x%02x", addr7);
    return path_ + " addr " + b;
  }
  Status errno_status(int e, uint8_t addr7) const {
    const std::string m = where(addr7) + ": " + std::strerror(e);
    switch (e) {
      case EREMOTEIO:
      case ENXIO:
      case EAGAIN:
        return device_error(m + " (NACK or arbitration lost)");
      case ETIMEDOUT:
        return timeout_error(m);
      default:
        return io_error(m);
    }
  }

  int fd_;
  std::string path_;
};

}  // namespace

Result<std::unique_ptr<I2cBus>> make_linux_i2c_bus(int bus_number) {
  if (bus_number < 0) return invalid_argument("i2c bus number " + std::to_string(bus_number) + " < 0");
  const std::string path = "/dev/i2c-" + std::to_string(bus_number);
  const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    const int e = errno;
    if (e == ENOENT)
      return not_found(path + ": no such device (i2c-dev module loaded? `sudo modprobe i2c-dev`; bus number?)");
    return io_error(path + ": " + std::strerror(e) + (e == EACCES ? " (add the user to group i2c)" : ""));
  }
  return std::unique_ptr<I2cBus>(std::make_unique<LinuxI2cBus>(fd, path));
}

}  // namespace rig
