#!/usr/bin/env bash
# Build libfranka + panda-py from source (aarch64 Jetson or x86_64 RT PC), pinned by robot generation.
#
#   controlSW/franka_deploy/scripts/build_libfranka_pandapy.sh --robot panda|fr3 [--dry-run] [--libfranka X.Y.Z]
#       [--panda-py REF] [--prefix DIR] [--src DIR] [--python PY] [--jobs N] [--no-apt]
#
# Version rule (Franka's libfranka/robot-system compatibility table; check the robot's system version in Desk ->
# Settings -> System before building, and override with --libfranka if the table says otherwise):
#   --robot panda   Panda, robot system 4.x  -> libfranka 0.9.x   (default 0.9.2, the last 0.9 release)
#   --robot fr3     FR3,   robot system 5.x  -> libfranka >= 0.10 (default 0.13.3: the newest line that needs no
#                                              pinocchio; 0.14+ adds a pinocchio dependency. FR3 system >= 5.7 may
#                                              require a newer libfranka -- VERIFY in the table)
#   A wrong pairing fails at connect with "incompatible library version" -- it never half-works.
# panda-py (JeanElsner/panda-py) is built from source against that libfranka (its PyPI/GitHub wheels are x86_64
# builds for fixed libfranka versions). Default ref v0.8.1 -- VERIFY it lists the chosen libfranka as supported.
#
# Steps: apt build deps -> clone + build + install libfranka into --prefix (default ~/franka/<libfranka version>)
# -> pip install panda-py from source into --python's environment with CMAKE_PREFIX_PATH=--prefix -> import check
# (panda_py.__version__, libfranka version) -> writes <prefix>/franka_versions.json and prints the two lines to put
# in configs/bridge.yaml (backend.panda_py_version / backend.libfranka_version).
# Idempotent: an installed libfranka of the same version in --prefix and an importable panda_py are kept.
# --dry-run prints every command and changes nothing (any machine).
set -euo pipefail

ROBOT=""
DRY=0
LIBFRANKA=""
PANDA_PY_REF="v0.8.1"
PREFIX=""
SRC="${HOME}/franka/src"
PY=python3
JOBS=$(nproc 2>/dev/null || echo 4)
DO_APT=1

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --robot) ROBOT=${2:?}; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --libfranka) LIBFRANKA=${2:?}; shift 2 ;;
    --panda-py) PANDA_PY_REF=${2:?}; shift 2 ;;
    --prefix) PREFIX=${2:?}; shift 2 ;;
    --src) SRC=${2:?}; shift 2 ;;
    --python) PY=${2:?}; shift 2 ;;
    --jobs) JOBS=${2:?}; shift 2 ;;
    --no-apt) DO_APT=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$ROBOT" in
  panda) DEFAULT=0.9.2; RANGE_OK() { [[ "$1" =~ ^0\.9\.[0-9]+$ ]]; } ;;
  fr3)   DEFAULT=0.13.3; RANGE_OK() { [[ "$1" =~ ^0\.(1[0-9])\.[0-9]+$ ]] || [[ "$1" =~ ^[1-9][0-9]*\.[0-9]+\.[0-9]+$ ]]; } ;;
  *) echo "--robot panda|fr3 is required (Panda system 4.x -> libfranka 0.9.x; FR3 -> >= 0.10)" >&2; exit 2 ;;
esac
LIBFRANKA=${LIBFRANKA:-$DEFAULT}
if ! RANGE_OK "$LIBFRANKA"; then
  echo "libfranka $LIBFRANKA is outside the range for --robot $ROBOT (panda: 0.9.x, fr3: >= 0.10)" >&2
  exit 2
fi
PREFIX=${PREFIX:-${HOME}/franka/libfranka-${LIBFRANKA}}

say() { printf '\n== %s\n' "$*"; }
run() { printf '+ %s\n' "$*"; if [ "$DRY" -eq 0 ]; then "$@"; fi; }
[ "$DRY" -eq 1 ] && echo "(dry run: nothing is changed)"
echo "robot=$ROBOT libfranka=$LIBFRANKA panda-py=$PANDA_PY_REF prefix=$PREFIX arch=$(uname -m) python=$PY"

say "1. build dependencies"
APT=(build-essential cmake git libpoco-dev libeigen3-dev python3-dev python3-pip)
if [ "$DO_APT" -eq 1 ]; then
  missing=()
  for p in "${APT[@]}"; do dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || missing+=("$p"); done
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

say "3. panda-py $PANDA_PY_REF against $PREFIX"
if [ "$DRY" -eq 0 ] && "$PY" -c "import panda_py" 2>/dev/null; then
  echo "  panda_py already importable: $("$PY" -c 'import panda_py; print(panda_py.__version__)')"
else
  if [ -d "$SRC/panda-py-$PANDA_PY_REF/.git" ]; then
    echo "  source present: $SRC/panda-py-$PANDA_PY_REF"
  else
    run git clone --recursive --branch "$PANDA_PY_REF" --depth 1 https://github.com/JeanElsner/panda-py.git \
        "$SRC/panda-py-$PANDA_PY_REF"
  fi
  printf '+ CMAKE_PREFIX_PATH=%s LD_LIBRARY_PATH=%s/lib %s -m pip install --no-cache-dir %s\n' "$PREFIX" "$PREFIX" "$PY" \
      "$SRC/panda-py-$PANDA_PY_REF"
  if [ "$DRY" -eq 0 ]; then
    CMAKE_PREFIX_PATH="$PREFIX" LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}" \
      "$PY" -m pip install --no-cache-dir "$SRC/panda-py-$PANDA_PY_REF"
  fi
fi

say "4. check + record versions"
REC="$PREFIX/franka_versions.json"
if [ "$DRY" -eq 1 ]; then
  echo "+ LD_LIBRARY_PATH=$PREFIX/lib $PY -c 'import panda_py, libfranka-linked module' -> $REC"
  PPV="<panda_py.__version__>"
else
  PPV=$(LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}" "$PY" -c 'import panda_py; print(panda_py.__version__)')
  printf '{"robot": "%s", "libfranka_version": "%s", "panda_py_version": "%s", "panda_py_ref": "%s", "arch": "%s"}\n' \
    "$ROBOT" "$LIBFRANKA" "$PPV" "$PANDA_PY_REF" "$(uname -m)" > "$REC"
  echo "  wrote $REC"
fi
cat <<EOF

Put in controlSW/franka_deploy/configs/bridge.yaml (backend:) and keep LD_LIBRARY_PATH=$PREFIX/lib for the bridge:
  panda_py_version: "$PPV"
  libfranka_version: "$LIBFRANKA"
Then: python -m franka_deploy check-config --config <deploy.yaml> --profile jetson   (panda_py_versions_pinned -> PASS)
EOF
