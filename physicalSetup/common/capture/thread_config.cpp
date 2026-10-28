// SPDX-License-Identifier: MIT
// apply_thread_config (rig/capture/thread_config.hpp): pthread_setaffinity_np(3), pthread_setschedparam(3) with
// SCHED_FIFO (sched(7): priorities 1..99), setpriority(2) on the thread id for SCHED_OTHER nice values.
#include "rig/capture/thread_config.hpp"

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace rig {

Status apply_thread_config(const ThreadConfig& c, const std::string& thread_name) {
  std::string problems;
  auto note = [&](const std::string& s) { problems += (problems.empty() ? "" : "; ") + s; };
  if (!thread_name.empty()) {
    const std::string n = thread_name.substr(0, 15);
    const int e = pthread_setname_np(pthread_self(), n.c_str());
    if (e != 0) note("setname: " + std::string(std::strerror(e)));
  }
  if (!c.cpus.empty()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    std::string ids;
    for (int cpu : c.cpus) {
      ids += (ids.empty() ? "" : ",") + std::to_string(cpu);
      if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &set);
    }
    const int e = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (e != 0) note("affinity {" + ids + "}: " + std::strerror(e));
  }
  if (c.priority > 0) {
    sched_param p{};
    p.sched_priority = c.priority;
    const int e = pthread_setschedparam(pthread_self(), SCHED_FIFO, &p);
    if (e != 0)
      note("SCHED_FIFO " + std::to_string(c.priority) + ": " + std::strerror(e) +
           (e == EPERM ? " (needs CAP_SYS_NICE or an rtprio limit, see BRINGUP)" : ""));
  } else if (c.nice != 0) {
    const pid_t tid = static_cast<pid_t>(::syscall(SYS_gettid));
    if (setpriority(PRIO_PROCESS, static_cast<id_t>(tid), c.nice) != 0)
      note("nice " + std::to_string(c.nice) + ": " + std::strerror(errno));
  }
  if (!problems.empty()) return unavailable("thread '" + thread_name + "': " + problems);
  return Status::Ok();
}

}  // namespace rig
