// New (control_deploy/STRUCTURE.md 9.1): KeyboardReader over a pipe, TerminalMode on a pty, KeyQueue. Fix wave: upper
// case keys, IXON off, s never dropped. + p (the start pose).
#include <gtest/gtest.h>

#include <pty.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "franka_control/keyboard_reader.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

struct Pipe {
  Pipe() { EXPECT_EQ(pipe(fds), 0); }
  ~Pipe() {
    closeWrite();
    if (fds[0] >= 0) close(fds[0]);
  }
  void write(const std::string& s) { ASSERT_EQ(::write(fds[1], s.data(), s.size()), static_cast<ssize_t>(s.size())); }
  void closeWrite() {
    if (fds[1] >= 0) close(fds[1]);
    fds[1] = -1;
  }
  int fds[2] = {-1, -1};
};

std::vector<Key> drain(KeyQueue& q) {
  std::vector<Key> out;
  while (auto k = q.pop()) out.push_back(*k);
  return out;
}

TEST(KeyboardReader, TheKeyTable) {
  EXPECT_EQ(KeyboardReader::map('h'), Key::kHomeArm);
  EXPECT_EQ(KeyboardReader::map('p'), Key::kStartPose);
  EXPECT_EQ(KeyboardReader::map('P'), Key::kStartPose);
  EXPECT_EQ(KeyboardReader::map('g'), Key::kHomeHand);
  EXPECT_EQ(KeyboardReader::map('s'), Key::kStop);
  EXPECT_EQ(KeyboardReader::map('q'), Key::kQuit);
  EXPECT_EQ(KeyboardReader::map('H'), Key::kHomeArm);   // caps lock: upper case acts like lower case
  EXPECT_EQ(KeyboardReader::map('G'), Key::kHomeHand);
  EXPECT_EQ(KeyboardReader::map('S'), Key::kStop);
  EXPECT_EQ(KeyboardReader::map('Q'), Key::kQuit);
  for (char c : {'x', 'X', ' ', '\x1b', '\n'}) EXPECT_FALSE(KeyboardReader::map(c)) << c;
}

TEST(KeyboardReader, PIsQueuedAndNamed) {
  Pipe p;
  KeyQueue q;
  SharedState shared;
  KeyboardReader reader(p.fds[0], q, [] {}, shared);
  p.write("pPh");
  p.closeWrite();
  testing::internal::CaptureStderr();
  reader.run();
  testing::internal::GetCapturedStderr();
  EXPECT_EQ(drain(q), (std::vector<Key>{Key::kStartPose, Key::kStartPose, Key::kHomeArm}));
  EXPECT_STREQ(keyChar(Key::kStartPose), "p");
}

TEST(KeyboardReader, KeysAreQueuedInOrderQuitCallsBackEofEnds) {
  Pipe p;
  KeyQueue q;
  SharedState shared;
  int quits = 0;
  KeyboardReader reader(p.fds[0], q, [&] { ++quits; }, shared);
  p.write("hgsxHGSQ\nq");
  p.closeWrite();   // EOF after the keys: run() ends by itself
  const std::string log = [&] {
    testing::internal::CaptureStderr();
    reader.run();
    return testing::internal::GetCapturedStderr();
  }();
  EXPECT_EQ(drain(q), (std::vector<Key>{Key::kHomeArm, Key::kHomeHand, Key::kStop, Key::kHomeArm, Key::kHomeHand,
                                       Key::kStop}));   // q and Q are never queued
  EXPECT_EQ(quits, 2);                                  // Q and q; EOF is not q
  EXPECT_NE(log.find("keys: h home arm, p start pose, g home hand, s stop, q quit"), std::string::npos) << log;
  EXPECT_NE(log.find("keyboard: stdin closed: keys disabled"), std::string::npos) << log;
}

TEST(KeyboardReader, AnEmptyEofNeverQuits) {
  Pipe p;
  KeyQueue q;
  SharedState shared;
  int quits = 0;
  KeyboardReader reader(p.fds[0], q, [&] { ++quits; }, shared);
  p.closeWrite();
  reader.run();
  EXPECT_EQ(quits, 0);
  EXPECT_TRUE(drain(q).empty());
}

TEST(KeyboardReaderThreads, ShutdownEndsTheReaderWithinATenthOfASecond) {
  Pipe p;
  KeyQueue q;
  SharedState shared;
  std::atomic<bool> done{false};
  KeyboardReader reader(p.fds[0], q, [] {}, shared);
  std::thread t([&] {
    reader.run();
    done.store(true);
  });
  p.write("h");
  ASSERT_TRUE(test::waitUntil([&] { return !drain(q).empty(); }, 2.0));
  const int64_t t0 = monotonicNowNs();
  shared.requestShutdown();
  ASSERT_TRUE(test::waitUntil([&] { return done.load(); }, 1.0));
  EXPECT_LT(nsToSeconds(monotonicNowNs() - t0), 0.1);
  t.join();
}

TEST(KeyQueue, The17thWaitingKeyIsDropped) {
  KeyQueue q;
  for (int i = 0; i < 16; ++i) q.push(Key::kStop);
  testing::internal::CaptureStderr();
  q.push(Key::kHomeArm);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("key h dropped"), std::string::npos) << log;
  const auto keys = drain(q);
  ASSERT_EQ(keys.size(), 16u);
  for (Key k : keys) EXPECT_EQ(k, Key::kStop);
}

TEST(KeyQueue, AStopIsNeverDropped) {
  KeyQueue q;
  for (int i = 0; i < 16; ++i) q.push(Key::kHomeHand);
  testing::internal::CaptureStderr();
  q.push(Key::kStop);
  q.push(Key::kHomeArm);   // still dropped
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_EQ(log.find("key s dropped"), std::string::npos) << log;
  EXPECT_NE(log.find("key h dropped"), std::string::npos) << log;
  const auto keys = drain(q);
  ASSERT_EQ(keys.size(), 17u);
  EXPECT_EQ(keys.back(), Key::kStop);   // after the keys typed before it
}

TEST(TerminalMode, OnAPtyItSetsCbreakAndRestores) {
  int master = -1, slave = -1;
  ASSERT_EQ(openpty(&master, &slave, nullptr, nullptr, nullptr), 0);
  termios before{};
  ASSERT_EQ(tcgetattr(slave, &before), 0);
  ASSERT_TRUE(before.c_lflag & ICANON);
  before.c_iflag |= IXON;   // a terminal with flow control on, as a normal login terminal has
  ASSERT_EQ(tcsetattr(slave, TCSANOW, &before), 0);
  {
    const TerminalMode mode(slave);
    EXPECT_TRUE(mode.active());
    termios now{};
    ASSERT_EQ(tcgetattr(slave, &now), 0);
    EXPECT_FALSE(now.c_lflag & ICANON);
    EXPECT_FALSE(now.c_lflag & ECHO);
    EXPECT_TRUE(now.c_lflag & ISIG);   // Ctrl-C stays SIGINT
    EXPECT_EQ(now.c_cc[VSUSP], static_cast<cc_t>(_POSIX_VDISABLE));
    EXPECT_EQ(now.c_cc[VQUIT], static_cast<cc_t>(_POSIX_VDISABLE));
    EXPECT_EQ(now.c_cc[VINTR], before.c_cc[VINTR]);
    EXPECT_FALSE(now.c_iflag & IXON);   // Ctrl-S cannot pause the terminal
  }
  termios after{};
  ASSERT_EQ(tcgetattr(slave, &after), 0);
  EXPECT_EQ(after.c_lflag, before.c_lflag);
  EXPECT_EQ(after.c_iflag, before.c_iflag);
  EXPECT_EQ(after.c_cc[VSUSP], before.c_cc[VSUSP]);
  EXPECT_EQ(after.c_cc[VQUIT], before.c_cc[VQUIT]);
  close(slave);
  close(master);
}

TEST(TerminalMode, OnAPipeItChangesNothing) {
  Pipe p;
  const TerminalMode mode(p.fds[0]);
  EXPECT_FALSE(mode.active());
}

}  // namespace
}  // namespace franka_control
