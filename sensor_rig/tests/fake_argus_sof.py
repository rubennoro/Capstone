#!/usr/bin/env python3
"""Fake ``argus_sof`` helper: same CLI, line protocol and shared-memory layout as native/argus_sof/argus_sof.cpp.

Extra behaviour switches come from the environment (the backend passes the real CLI unchanged):
  FAKE_SOF_FRAMES=N        frames to emit, then idle (default 30)
  FAKE_SOF_FPS=F           pacing (default 100, fast for tests)
  FAKE_SOF_RAW0=NS         sensor timestamp of frame 0 (default 1_000_000_000_000)
  FAKE_SOF_GAP_AT=K        skip seq K (a frame lost before userspace)
  FAKE_SOF_CRASH_AT=K      exit(3) with a message on stderr after frame K
  FAKE_SOF_NO_READY=1      never print READY (startup timeout)
  FAKE_SOF_ERROR=msg       print "E msg" instead of READY and exit 2
  FAKE_SOF_SHM_DIR=dir     where the ring file goes (default /dev/shm)
"""
import os
import struct
import sys
import time

args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
w, h, slots, name = int(args["--width"]), int(args["--height"]), int(args["--slots"]), args["--shm-name"]
env = os.environ
if env.get("FAKE_SOF_ERROR"):
    print(f"E {env['FAKE_SOF_ERROR']}", flush=True)
    sys.exit(2)
if env.get("FAKE_SOF_NO_READY"):
    time.sleep(60)
    sys.exit(0)
payload = w * h * 3 // 2
slot_bytes = 16 + payload
path = os.path.join(env.get("FAKE_SOF_SHM_DIR", "/dev/shm"), name)
with open(path, "wb") as f:
    f.write(b"ARGSOF01" + struct.pack("<4I", slots, slot_bytes, w, h) + b"\0" * 8)
    f.write(b"\0" * (slot_bytes * slots))
fd = os.open(path, os.O_RDWR)
import mmap  # noqa: E402

mm = mmap.mmap(fd, 32 + slot_bytes * slots)
print(f"READY {w} {h} NV12 {slots} {slot_bytes} {name}", flush=True)
n = int(env.get("FAKE_SOF_FRAMES", "30"))
period = 1.0 / float(env.get("FAKE_SOF_FPS", "100"))
raw0 = int(env.get("FAKE_SOF_RAW0", "1000000000000"))
gap = int(env.get("FAKE_SOF_GAP_AT", "-1"))
crash = int(env.get("FAKE_SOF_CRASH_AT", "-1"))
for seq in range(n):
    time.sleep(period)
    if seq == gap:
        continue
    slot = seq % slots
    off = 32 + slot * slot_bytes
    y = bytes([(seq * 7) % 256]) * (w * h)
    uv = bytes([128]) * (w * h // 2)
    mm[off:off + 8] = struct.pack("<Q", seq)
    mm[off + 16: off + 16 + payload] = y + uv
    mm[off + 8: off + 16] = struct.pack("<Q", seq)
    print(f"F {seq} {raw0 + seq * 33_333_333} 3000000 4.0000 {slot}", flush=True)
    if seq == crash:
        print("NvBufSurfaceMap failed (simulated)", file=sys.stderr, flush=True)
        sys.exit(3)
time.sleep(60)
