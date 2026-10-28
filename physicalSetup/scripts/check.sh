#!/usr/bin/env bash
# CI-style gate for physicalSetup: configure (Ninja) + build + ctest. Non-zero exit on any failure.
#
#   physicalSetup/scripts/check.sh <build_dir> [--sanitize address|undefined|thread|address,undefined]
#                                  [--build-type <type>] [--jobs N] [-- <extra cmake args>]
#
# Build trees must live outside the repo (laptop: the session scratch dir; IMPLEMENTATION_PLAN D10).
set -euo pipefail

usage() { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

[[ $# -ge 1 ]] || usage
BUILD_DIR=$1; shift
SANITIZE=""
BUILD_TYPE=RelWithDebInfo
JOBS=$(nproc)
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sanitize) SANITIZE=${2:?--sanitize needs a value}; shift 2 ;;
    --build-type) BUILD_TYPE=${2:?}; shift 2 ;;
    --jobs) JOBS=${2:?}; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) usage ;;
    *) echo "check.sh: unknown argument '$1'" >&2; usage ;;
  esac
done

SRC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_DIR=$(cd "$SRC_DIR/.." && pwd)
BUILD_DIR=$(realpath -m "$BUILD_DIR")
case "$BUILD_DIR/" in
  "$REPO_DIR"/*) echo "check.sh: refusing to build inside the repo ($BUILD_DIR)" >&2; exit 2 ;;
esac
mkdir -p "$BUILD_DIR"

echo "== configure: $BUILD_DIR (type=$BUILD_TYPE sanitize='${SANITIZE}')"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DRIG_SANITIZE="$SANITIZE" \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo "== build"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "== test"
export ASAN_OPTIONS=${ASAN_OPTIONS:-abort_on_error=1:detect_leaks=1:strict_string_checks=1:detect_stack_use_after_return=1}
export UBSAN_OPTIONS=${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}
export TSAN_OPTIONS=${TSAN_OPTIONS:-halt_on_error=1:second_deadlock_stack=1}
CTEST=(ctest --test-dir "$BUILD_DIR" --output-on-failure -j "$JOBS" --no-tests=error)
if [[ ",$SANITIZE," == *",thread,"* ]] && command -v setarch >/dev/null; then
  # gcc 9/11 TSan aborts with "unexpected memory mapping" under the high mmap ASLR entropy of recent kernels
  # (6.x, vm.mmap_rnd_bits = 32). Running without ASLR avoids it; it does not change what TSan checks.
  CTEST=(setarch "$(uname -m)" -R "${CTEST[@]}")
fi
"${CTEST[@]}"

echo "== check.sh: PASS ($BUILD_DIR)"
