"""A tiny fake process for the launcher tests: `fake_proc.py MODE [options] <common arguments>`.

Modes: serve (ready, then wait for SIGTERM), never_ready, exit (exit with --code, after an optional FATAL line),
ignore_term (ready, ignores SIGTERM), grandchild (starts a child, writes its pid, ready), echo (writes to stdout and
stderr, ready), pwd (writes its working folder to --marker, ready), ctrl_c (ready, sends SIGINT to its parent as a
Ctrl-C would, exits 0 after --delay).
"""
import argparse
import os
import signal
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("--delay", type=float, default=0.0)
parser.add_argument("--code", type=int, default=1)
parser.add_argument("--fatal", default=None)
parser.add_argument("--ready", action="store_true", help="exit mode: write the ready file first")
parser.add_argument("--marker", default=None, help="file written when SIGTERM arrives")
parser.add_argument("--pid-file", default=None)
parser.add_argument("--station")
parser.add_argument("--session")
parser.add_argument("--ready-file")
parser.add_argument("--mock", action="store_true")
args = parser.parse_args()


def on_term(signum, frame):
    if args.marker:
        open(args.marker, "w").close()
    sys.exit(0)


def ready():
    if args.ready_file:
        open(args.ready_file, "w").close()


def wait_forever():
    while True:
        time.sleep(0.05)


signal.signal(signal.SIGTERM, signal.SIG_IGN if args.mode == "ignore_term" else on_term)
time.sleep(args.delay if args.mode in ("serve", "exit") else 0.0)
if args.mode in ("serve", "ignore_term"):
    ready()
    wait_forever()
elif args.mode == "never_ready":
    wait_forever()
elif args.mode == "exit":
    if args.ready:
        ready()
    if args.fatal:
        print("some log line", file=sys.stderr)
        print(args.fatal, file=sys.stderr)
    sys.exit(args.code)
elif args.mode == "grandchild":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(args.pid_file, "w") as f:
        f.write(str(child.pid))
    ready()
    wait_forever()
elif args.mode == "pwd":
    with open(args.marker, "w") as f:
        f.write(os.getcwd())
    ready()
    wait_forever()
elif args.mode == "echo":
    print("hello on stdout", flush=True)
    print("hello on stderr", file=sys.stderr, flush=True)
    ready()
    wait_forever()
elif args.mode == "ctrl_c":
    ready()
    os.kill(os.getppid(), signal.SIGINT)
    time.sleep(args.delay)
    sys.exit(0)
else:
    sys.exit(f"unknown mode {args.mode}")
