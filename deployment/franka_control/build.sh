#!/usr/bin/env bash
# from rnoro/main-sim:controlSW/franka_deploy/scripts/build_libfranka_pandapy.sh @ 4be9982
# changes: libfranka part only (no panda-py); the version comes from config.yaml; then builds franka_control.
#
#   franka_control/build.sh [--dry-run] [--no-apt] [--jobs N] [--prefix DIR] [--any-version] [--test]
#
# 1. Reads libfranka_version from franka_control/config.yaml (the one place it is written). Refuses a version outside
#    0.9.x (Panda, robot system 4.x, per Franka's compatibility table) unless --any-version: check Desk -> Settings ->
#    System and the table first. A wrong pairing fails at connect ("incompatible library version"), never half-works.
# 2. apt build dependencies (unless --no-apt).
# 3. libfranka: shallow clone of that tag (recursive), Release, BUILD_TESTS=OFF BUILD_EXAMPLES=OFF, installed into
#    --prefix (default ~/franka/libfranka-<version>). Skipped when that version is already installed there.
# 4. Configures and builds franka_control/build/ (Release, no sanitizers) against it.
# 5. Prints build/franka_control --version. With --test, also builds build-test/ (ASan + UBSan) and runs CTest.
# --dry-run prints every command and changes nothing. Idempotent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0
DO_APT=1
JOBS=$(nproc 2>/dev/null || echo 4)
PREFIX=""
ANY_VERSION=0
TEST=0
SRC="${HOME}/franka/src"

usage() { sed -n '5,16p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --no-apt) DO_APT=0; shift ;;
    --jobs) JOBS=${2:?}; shift 2 ;;
    --prefix) PREFIX=${2:?}; shift 2 ;;
    --any-version) ANY_VERSION=1; shift ;;
    --test) TEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

LIBFRANKA=$(sed -n 's/^libfranka_version:[[:space:]]*"\{0,1\}\([0-9][0-9.]*\)"\{0,1\}.*/\1/p' "$HERE/config.yaml")
if [ -z "$LIBFRANKA" ]; then
  echo "config.yaml has no 'libfranka_version: \"X.Y.Z\"' line" >&2
  exit 2
fi
if [ "$ANY_VERSION" -eq 0 ] && ! [[ "$LIBFRANKA" =~ ^0\.9\.[0-9]+$ ]]; then
  echo "libfranka $LIBFRANKA is outside 0.9.x (the Panda, robot system 4.x): check Desk -> Settings -> System and" >&2
  echo "Franka's compatibility table, then rerun with --any-version if the table says so" >&2
  exit 2
fi
PREFIX=${PREFIX:-${HOME}/franka/libfranka-${LIBFRANKA}}

say() { printf '\n== %s\n' "$*"; }
run() { printf '+ %s\n' "$*"; if [ "$DRY" -eq 0 ]; then "$@"; fi; }
[ "$DRY" -eq 1 ] && echo "(dry run: nothing is changed)"
echo "libfranka=$LIBFRANKA prefix=$PREFIX arch=$(uname -m) jobs=$JOBS"

say "1. build dependencies"
APT=(build-essential cmake git pkg-config libpoco-dev libeigen3-dev libzmq3-dev libyaml-cpp-dev googletest)
if [ "$DO_APT" -eq 1 ]; then
  missing=()
  for p in "${APT[@]}"; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || missing+=("$p")
  done
  if [ ${#missing[@]} -gt 0 ]; then run sudo apt-get install -y "${missing[@]}"; else echo "  all installed"; fi
else
  echo "  (--no-apt) needs: ${APT[*]}"
fi

say "2. libfranka $LIBFRANKA -> $PREFIX"
CFG="$PREFIX/lib/cmake/Franka/FrankaConfigVersion.cmake"
if [ -f "$CFG" ] && grep -q "PACKAGE_VERSION \"$LIBFRANKA\"" "$CFG"; then
  echo "  already installed ($CFG)"
else
  run mkdir -p "$SRC"
  if [ -d "$SRC/libfranka-$LIBFRANKA/.git" ]; then
    echo "  source present: $SRC/libfranka-$LIBFRANKA"
  else
    run git clone --recursive --branch "$LIBFRANKA" --depth 1 https://github.com/frankaemika/libfranka.git \
        "$SRC/libfranka-$LIBFRANKA"
  fi
  run cmake -S "$SRC/libfranka-$LIBFRANKA" -B "$SRC/libfranka-$LIBFRANKA/build" -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$PREFIX"
  run cmake --build "$SRC/libfranka-$LIBFRANKA/build" -j "$JOBS"
  run cmake --install "$SRC/libfranka-$LIBFRANKA/build"
fi

say "3. franka_control -> $HERE/build"
run cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DFRANKA_CONTROL_WITH_LIBFRANKA=ON -DFRANKA_CONTROL_BUILD_TESTS=OFF
run cmake --build "$HERE/build" -j "$JOBS"

say "4. version"
run "$HERE/build/franka_control" --version

if [ "$TEST" -eq 1 ]; then
  say "5. tests (ASan + UBSan) in $HERE/build-test"
  run cmake -S "$HERE" -B "$HERE/build-test" -DFRANKA_CONTROL_SANITIZE=ON -DCMAKE_PREFIX_PATH="$PREFIX"
  run cmake --build "$HERE/build-test" -j "$JOBS"
  run ctest --test-dir "$HERE/build-test" --output-on-failure
fi
