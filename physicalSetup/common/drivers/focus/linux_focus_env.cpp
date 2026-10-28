// SPDX-License-Identifier: MIT
// Linux FocusEnvironment: kernel-binding check, /dev/i2c-N, V4L2 focus_absolute (rig/drivers/camera/imx519_focus.hpp).
//
// Kernel binding (`UU`): i2c-tools i2cdetect.c prints "UU" when ioctl(fd, I2C_SLAVE, addr) fails with EBUSY;
// drivers/i2c/i2c-dev.c returns EBUSY from I2C_SLAVE when a client at that address has a driver bound
// (i2cdev_check_addr -> i2cdev_check: `dev->driver ? -EBUSY : 0`). The same fact is visible in sysfs as
// /sys/bus/i2c/devices/<bus>-<addr 4 hex>/driver. Note: I2C_RDWR (used by LinuxI2cBus) does NOT perform this check,
// so a userspace write to a bound address would succeed and fight the kernel driver; hence the explicit check.
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "rig/drivers/camera/imx519_focus.hpp"

namespace rig {
namespace {

namespace fs = std::filesystem;

std::string read_first_line(const fs::path& p) {
  std::ifstream f(p);
  std::string s;
  if (f) std::getline(f, s);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  return s;
}

class FdGuard {
 public:
  explicit FdGuard(int fd) : fd_(fd) {}
  ~FdGuard() {
    if (fd_ >= 0) ::close(fd_);
  }
  FdGuard(const FdGuard&) = delete;
  FdGuard& operator=(const FdGuard&) = delete;
  int get() const { return fd_; }

 private:
  int fd_;
};

int xioctl(int fd, unsigned long req, void* arg) {
  int r;
  do {
    r = ::ioctl(fd, req, arg);
  } while (r < 0 && errno == EINTR);
  return r;
}

class LinuxFocusEnvironment final : public FocusEnvironment {
 public:
  explicit LinuxFocusEnvironment(std::string sysroot) : sysroot_(std::move(sysroot)) {}

  bool i2c_address_kernel_bound(int bus, int addr7, std::string* detail) override {
    const std::string drv = i2c_sysfs_driver(sysroot_, bus, addr7);
    if (!drv.empty()) {
      if (detail) *detail = "sysfs driver " + drv;
      return true;
    }
    if (!sysroot_.empty()) return false;  // fake tree: no ioctl probe
    const std::string dev = "/dev/i2c-" + std::to_string(bus);
    FdGuard fd(::open(dev.c_str(), O_RDWR | O_CLOEXEC));
    if (fd.get() < 0) return false;  // open errors are reported by open_i2c_bus()
    if (::ioctl(fd.get(), I2C_SLAVE, static_cast<unsigned long>(addr7)) < 0 && errno == EBUSY) {
      if (detail) *detail = dev + " ioctl(I2C_SLAVE) EBUSY";
      return true;
    }
    return false;
  }

  Result<std::unique_ptr<I2cBus>> open_i2c_bus(int bus) override { return make_linux_i2c_bus(bus); }

  std::vector<std::string> v4l2_focus_candidates(const FocusConfig& c) override {
    if (!c.v4l2_subdev.empty()) return {c.v4l2_subdev};
    return find_v4l2_focus_candidates(sysroot_, c.i2c_bus, c.i2c_addr);
  }

  Result<V4l2ControlRange> v4l2_query_focus(const std::string& node) override {
    FdGuard fd(::open(node.c_str(), O_RDWR | O_CLOEXEC));
    if (fd.get() < 0) return not_found(node + ": open: " + std::strerror(errno));
    v4l2_queryctrl q{};
    q.id = V4L2_CID_FOCUS_ABSOLUTE;
    if (xioctl(fd.get(), VIDIOC_QUERYCTRL, &q) < 0) return not_found(node + ": no focus_absolute control");
    if (q.flags & V4L2_CTRL_FLAG_DISABLED) return not_found(node + ": focus_absolute disabled");
    V4l2ControlRange r;
    r.minimum = q.minimum;
    r.maximum = q.maximum;
    r.step = q.step > 0 ? q.step : 1;
    r.default_value = q.default_value;
    return r;
  }

  Status v4l2_set_focus(const std::string& node, int value) override {
    FdGuard fd(::open(node.c_str(), O_RDWR | O_CLOEXEC));
    if (fd.get() < 0) return io_error(node + ": open: " + std::strerror(errno));
    v4l2_control c{};
    c.id = V4L2_CID_FOCUS_ABSOLUTE;
    c.value = value;
    if (xioctl(fd.get(), VIDIOC_S_CTRL, &c) < 0) return device_error(node + ": VIDIOC_S_CTRL: " + std::strerror(errno));
    return Status::Ok();
  }

  Result<int> v4l2_get_focus(const std::string& node) override {
    FdGuard fd(::open(node.c_str(), O_RDWR | O_CLOEXEC));
    if (fd.get() < 0) return io_error(node + ": open: " + std::strerror(errno));
    v4l2_control c{};
    c.id = V4L2_CID_FOCUS_ABSOLUTE;
    if (xioctl(fd.get(), VIDIOC_G_CTRL, &c) < 0) return device_error(node + ": VIDIOC_G_CTRL: " + std::strerror(errno));
    return static_cast<int>(c.value);
  }

 private:
  std::string sysroot_;
};

}  // namespace

std::string i2c_sysfs_driver(const std::string& sysroot, int bus, int addr7) {
  char name[32];
  std::snprintf(name, sizeof(name), "%d-%04x", bus, addr7);
  const fs::path link = fs::path(sysroot + "/sys/bus/i2c/devices") / name / "driver";
  std::error_code ec;
  if (!fs::exists(link, ec)) return "";
  const fs::path target = fs::is_symlink(link, ec) ? fs::read_symlink(link, ec) : link;
  const std::string drv = target.filename().string();
  return drv.empty() ? std::string("bound") : drv;
}

std::vector<std::string> find_v4l2_focus_candidates(const std::string& sysroot, int bus, int addr7) {
  const fs::path cls = fs::path(sysroot + "/sys/class/video4linux");
  std::error_code ec;
  if (!fs::is_directory(cls, ec)) return {};
  char lens_client[32], sensor_client[32];
  std::snprintf(lens_client, sizeof(lens_client), "%d-%04x", bus, addr7);
  std::snprintf(sensor_client, sizeof(sensor_client), "%d-001a", bus);  // IMX519 at 0x1a (forum 8481 dmesg)
  std::vector<std::string> lens, sensor, video;
  std::vector<fs::path> entries;
  for (const auto& e : fs::directory_iterator(cls, ec)) entries.push_back(e.path());
  std::sort(entries.begin(), entries.end());
  for (const fs::path& p : entries) {
    const std::string node = p.filename().string();
    const std::string name = read_first_line(p / "name");
    const std::string dev = sysroot + "/dev/" + node;
    const bool on_bus = bus < 0 || name.find(" " + std::to_string(bus) + "-") != std::string::npos;
    if (name.find(lens_client) != std::string::npos ||
        (on_bus && (name.find("ak7375") != std::string::npos || name.find("vcm") != std::string::npos))) {
      lens.push_back(dev);
    } else if (on_bus && name.find("imx519") != std::string::npos) {
      (node.rfind("v4l-subdev", 0) == 0 ? sensor : video).push_back(dev);
    } else if (name.find(sensor_client) != std::string::npos) {
      (node.rfind("v4l-subdev", 0) == 0 ? sensor : video).push_back(dev);
    }
  }
  std::vector<std::string> out = lens;
  out.insert(out.end(), sensor.begin(), sensor.end());
  out.insert(out.end(), video.begin(), video.end());
  return out;
}

std::unique_ptr<FocusEnvironment> make_linux_focus_environment(const std::string& sysroot) {
  return std::make_unique<LinuxFocusEnvironment>(sysroot);
}

}  // namespace rig
