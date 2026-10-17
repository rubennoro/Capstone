#!/usr/bin/env bash
# One-time (idempotent) setup of a Jetson Orin Nano (JetPack 6) for sensor_rig recording.
#
#   sensor_rig/scripts/jetson_setup.sh [--dry-run] [--venv DIR] [--python PY] [--no-apt] [--no-udev] [--pip-spidev] [--yes]
#
# What it does (each step is skipped when already done):
#   1. apt: python3-gi + GStreamer introspection, gstreamer1.0-tools, python3-numpy/-opencv/-scipy/-yaml/-spidev
#      (JetPack's own OpenCV build; opencv is NEVER pip-installed into the venv).
#   2. venv with --system-site-packages (default ~/sensor_rig_venv) so it sees gi and JetPack's cv2, pip >= 23 in it
#      (22.04's pip 22.0 can't do pyproject editable installs), then
#      `pip install --no-deps -e sensor_rig` (no pip dependency resolution: no PyPI opencv/scipy/av wheels).
#      spidev is pip-installed into the venv if apt has no python3-spidev, or with --pip-spidev.
#   3. Import check of every module the recorder needs, with the package that provides a missing one.
#   4. Access: video group, an 'spi' group + udev rule for /dev/spidev*, spidev module loaded at boot.
#   5. Prints the manual steps (jetson-io, power mode, clock) and the doctor command.
#
# --dry-run prints every command instead of running it. --no-apt / --no-udev skip steps 1 / 4 (e.g. no sudo).
# --python PY creates the venv from PY instead of /usr/bin/python3 (tests).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG_DIR=$(cd "$SCRIPT_DIR/.." && pwd)          # sensor_rig/ (contains pyproject.toml)
VENV="${HOME}/sensor_rig_venv"
PY=/usr/bin/python3
DRY=0
DO_APT=1
DO_UDEV=1
YES=0
PIP_SPIDEV=0

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --venv) VENV=${2:?--venv needs a directory}; shift 2 ;;
    --python) PY=${2:?--python needs an interpreter}; shift 2 ;;
    --no-apt) DO_APT=0; shift ;;
    --no-udev) DO_UDEV=0; shift ;;
    --pip-spidev) PIP_SPIDEV=1; shift ;;
    --yes|-y) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
run() {
  # run CMD ARGS...: echo, then execute unless --dry-run
  printf '+ %s\n' "$*"
  if [ "$DRY" -eq 0 ]; then "$@"; fi
}
write_root_file() {
  # write_root_file PATH CONTENT: write a root-owned file only if its content differs
  local path=$1 content=$2
  if [ -f "$path" ] && [ "$(cat "$path")" = "$content" ]; then
    echo "  $path already up to date"
    return 0
  fi
  printf '+ write %s: %s\n' "$path" "$content"
  if [ "$DRY" -eq 0 ]; then printf '%s\n' "$content" | sudo tee "$path" > /dev/null; fi
}

[ -f "$PKG_DIR/pyproject.toml" ] || { echo "cannot find sensor_rig/pyproject.toml next to $SCRIPT_DIR" >&2; exit 1; }
if [ -f /etc/nv_tegra_release ]; then
  echo "Jetson: $(head -n 1 /etc/nv_tegra_release)"
else
  warn "/etc/nv_tegra_release not found: this does not look like a Jetson (continuing; Argus checks will fail)"
fi
[ "$DRY" -eq 1 ] && echo "(dry run: nothing is changed)"

# ---- 1. apt --------------------------------------------------------------------------------------------------------
APT_PKGS=(python3-venv python3-pip python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
          gstreamer1.0-tools python3-numpy python3-opencv python3-scipy python3-yaml)
if [ "$DO_APT" -eq 1 ]; then
  say "1. apt packages"
  missing=()
  for p in "${APT_PKGS[@]}" python3-spidev; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || missing+=("$p")
  done
  if [ ${#missing[@]} -eq 0 ]; then
    echo "  all installed"
  else
    run sudo apt-get update
    apt_yes=()
    [ "$YES" -eq 1 ] && apt_yes=(-y)
    run sudo apt-get install "${apt_yes[@]}" "${APT_PKGS[@]}"
    if ! run sudo apt-get install "${apt_yes[@]}" python3-spidev; then
      warn "python3-spidev not available from apt: installing spidev into the venv with pip"
      PIP_SPIDEV=1
    fi
  fi
  if command -v gst-inspect-1.0 > /dev/null && ! gst-inspect-1.0 nvarguscamerasrc > /dev/null 2>&1; then
    warn "GStreamer element nvarguscamerasrc not found: sudo apt install nvidia-l4t-gstreamer (part of JetPack)"
  fi
else
  say "1. apt packages: skipped (--no-apt)"
fi

# ---- 2. venv -------------------------------------------------------------------------------------------------------
say "2. venv $VENV (--system-site-packages)"
if [ -x "$VENV/bin/python" ]; then
  echo "  exists"
  if ! grep -q "include-system-site-packages = true" "$VENV/pyvenv.cfg" 2>/dev/null; then
    warn "$VENV was created without --system-site-packages: it cannot see python3-gi / JetPack's cv2. Remove it and rerun."
  fi
else
  run "$PY" -m venv --system-site-packages "$VENV"
fi
# Ubuntu 22.04's pip 22.0 cannot do PEP 660 editable installs of a pyproject-only package: upgrade pip in the venv.
if ! "$VENV/bin/python" -c 'import sys, pip; sys.exit(int(pip.__version__.split(".")[0]) < 23)' 2> /dev/null; then
  run "$VENV/bin/python" -m pip install --upgrade "pip>=23"
fi
run "$VENV/bin/python" -m pip install --no-deps -e "$PKG_DIR"
if [ "$PIP_SPIDEV" -eq 1 ]; then
  run "$VENV/bin/python" -m pip install --no-deps spidev
fi
if [ "$DRY" -eq 0 ]; then
  cv2_file=$("$VENV/bin/python" -c 'import cv2; print(cv2.__file__)' 2> /dev/null || true)
  case "$cv2_file" in
    "$VENV"/*) warn "cv2 is pip-installed in the venv ($cv2_file) and shadows JetPack's OpenCV:" \
                    "$VENV/bin/python -m pip uninstall opencv-python opencv-python-headless" ;;
  esac
fi

# ---- 3. import check -----------------------------------------------------------------------------------------------
say "3. import check"
if [ "$DRY" -eq 1 ]; then
  echo "+ $VENV/bin/python -c 'import numpy, cv2, yaml, scipy, spidev, gi' (one module at a time)"
else
  bad=0
  for m in numpy:python3-numpy cv2:python3-opencv yaml:python3-yaml scipy:python3-scipy \
           spidev:"python3-spidev (sudo apt), or rerun with --pip-spidev (pip install spidev into the venv)" \
           gi:python3-gi sensor_rig:"pip install --no-deps -e $PKG_DIR"; do
    mod=${m%%:*}; fix=${m#*:}
    if "$VENV/bin/python" -c "import $mod" 2> /dev/null; then
      echo "  ok      $mod"
    else
      echo "  MISSING $mod  -> install: $fix" >&2
      bad=1
    fi
  done
  if [ "$bad" -ne 0 ]; then
    echo "import check failed: install the packages above (sudo apt install ...) and rerun this script" >&2
    exit 1
  fi
fi

# ---- 4. device access ----------------------------------------------------------------------------------------------
USER_NAME=${SUDO_USER:-${USER:-$(id -un)}}
if [ "$DO_UDEV" -eq 1 ]; then
  say "4. device access for $USER_NAME (video group, spi group + udev rule, spidev at boot)"
  run sudo groupadd -f spi
  for g in video spi; do
    if id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx "$g"; then
      echo "  $USER_NAME already in $g"
    else
      run sudo usermod -aG "$g" "$USER_NAME"
      echo "  (log out and back in for the group change to apply)"
    fi
  done
  write_root_file /etc/udev/rules.d/99-sensor-rig-spidev.rules 'SUBSYSTEM=="spidev", KERNEL=="spidev*", GROUP="spi", MODE="0660"'
  write_root_file /etc/modules-load.d/sensor-rig-spidev.conf 'spidev'
  run sudo modprobe spidev
  run sudo udevadm control --reload-rules
  run sudo udevadm trigger --subsystem-match=spidev
else
  say "4. device access: skipped (--no-udev)"
fi

# ---- 5. manual steps -----------------------------------------------------------------------------------------------
say "5. manual steps (once)"
cat <<EOF
  a. sudo /opt/nvidia/jetson-io/jetson-io.py
       -> "Configure Jetson 24pin CSI Connector": Camera IMX219 Dual
       -> "Configure Jetson 40pin Header": enable spi1 (pins 19 MOSI, 21 MISO, 23 SCK, 24 CS0)
     save and reboot; then /dev/video0, /dev/video1 and /dev/spidev0.0 must exist.
  b. power: sudo nvpmodel -q (choose the highest mode with sudo nvpmodel -m <id>), then sudo jetson_clocks
  c. clock (no RTC battery by default; episode names use the wall clock): sudo timedatectl set-ntp true
  d. record to the NVMe, e.g. sudo mkdir -p /mnt/nvme/raw && sudo chown $USER_NAME /mnt/nvme/raw

Then, in a new login shell:
  source $VENV/bin/activate
  sensor-rig doctor --rig <rig.yaml> --out /mnt/nvme/raw
EOF
