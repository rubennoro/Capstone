#!/usr/bin/env bash
# One-time (idempotent) setup of a Jetson Orin Nano (JetPack 6 / L4T R36) for franka_deploy (policy + 1 kHz bridge).
#
#   controlSW/franka_deploy/scripts/jetson_deploy_setup.sh [--dry-run] [--yes] [--franka-if IFACE]
#       [--jetson-ip 172.16.0.1/24] [--venv DIR] [--torch-wheel URL] [--rt-repo R36.4] [--rt-core 5]
#       [--with-sensor-setup] [--skip STEP[,STEP...]]
#
# Steps (each one checks first and does nothing when already done; names for --skip in brackets):
#   0. [l4t]      JetPack/L4T check: /etc/nv_tegra_release must be R36 (JetPack 6). Anything else: stop.
#   1. [rt]       PREEMPT_RT kernel matching the installed L4T: NVIDIA's OTA rt-kernel apt repo for this L4T release
#                 + nvidia-l4t-rt-kernel{,-headers,-oot-modules} + nvidia-l4t-display-rt-kernel. Reboot needed.
#   2. [cmdline]  kernel args isolcpus=5 nohz_full=5 rcu_nocbs=5 appended to the APPEND line of
#                 /boot/extlinux/extlinux.conf (backup kept as extlinux.conf.franka.bak). Reboot needed.
#   3. [irq]      NIC IRQ(s) of the Franka interface pinned to core 5 (systemd oneshot unit franka-irq-pin, runs every
#                 boot because IRQ numbers can change); irqbalance disabled if present.
#   4. [power]    nvpmodel: highest-power mode (the MAXN* entry of /etc/nvpmodel.conf, else mode 0) + jetson_clocks,
#                 persisted with the franka-jetson-clocks unit.
#   5. [limits]   'realtime' group with rtprio 99 / memlock unlimited (/etc/security/limits.d/99-franka-rt.conf);
#                 the current user is added (log out/in).
#   6. [net]      static IP on the direct Franka link (NetworkManager connection 'franka-fci', no gateway, no DNS):
#                 Jetson 172.16.0.1/24, robot (Control unit) 172.16.0.2 by default.
#   7. [torch]    deployment venv (default ~/franka_deploy_venv, --system-site-packages for JetPack's cv2/gi) with
#                 NVIDIA's Jetson torch wheel (NOT PyPI torch: no CUDA on aarch64) + requirements-inference.txt
#                 (imagecodecs==2023.9.18 pin) + robotmq/pyzmq/scipy for the bridge.
#   8. [sensor]   only with --with-sensor-setup: runs sensor_rig/scripts/jetson_setup.sh (owned by the training stage;
#                 not edited here) with --dry-run passed through.
# Then: reboot, and run `python -m franka_deploy.tools.jetson_verify` (pass/fail for every item above).
#
# --dry-run prints every command and every file it would write, changes nothing, and runs on any Linux machine.
# Versions/URLs that must be re-checked against NVIDIA's docs for the exact L4T release are marked VERIFY.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FD_DIR=$(cd "$SCRIPT_DIR/.." && pwd)                       # controlSW/franka_deploy
REPO_DIR=$(cd "$FD_DIR/../.." && pwd)
DRY=0
YES=0
FRANKA_IF=""
JETSON_IP="172.16.0.1/24"
VENV="${HOME}/franka_deploy_venv"
RT_CORE=5
RT_REPO=""                                                 # default: derived from /etc/nv_tegra_release (e.g. r36.4)
# VERIFY: NVIDIA's JetPack 6.1/6.2 (L4T R36.4) PyTorch wheel; for other releases see
# https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
TORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
WITH_SENSOR=0
SKIP=","

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --yes|-y) YES=1; shift ;;
    --franka-if) FRANKA_IF=${2:?}; shift 2 ;;
    --jetson-ip) JETSON_IP=${2:?}; shift 2 ;;
    --venv) VENV=${2:?}; shift 2 ;;
    --torch-wheel) TORCH_WHEEL=${2:?}; shift 2 ;;
    --rt-repo) RT_REPO=${2:?}; shift 2 ;;
    --rt-core) RT_CORE=${2:?}; shift 2 ;;
    --with-sensor-setup) WITH_SENSOR=1; shift ;;
    --skip) SKIP=",${2:?},"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
skip() { case "$SKIP" in *",$1,"*) echo "  (skipped: --skip $1)"; return 0 ;; *) return 1 ;; esac; }
run() { printf '+ %s\n' "$*"; if [ "$DRY" -eq 0 ]; then "$@"; fi; }
write_root_file() {
  # write_root_file PATH CONTENT: root-owned file, rewritten only if the content differs
  local path=$1 content=$2
  if [ -f "$path" ] && [ "$(cat "$path")" = "$content" ]; then echo "  $path already up to date"; return 0; fi
  printf '+ write %s:\n%s\n' "$path" "$content" | sed 's/^/  /'
  if [ "$DRY" -eq 0 ]; then printf '%s\n' "$content" | sudo tee "$path" > /dev/null; fi
}
confirm() {
  [ "$YES" -eq 1 ] || [ "$DRY" -eq 1 ] && return 0
  read -r -p "$1 [y/N] " ans; [ "$ans" = "y" ] || [ "$ans" = "Y" ]
}
REBOOT=0
[ "$DRY" -eq 1 ] && echo "(dry run: nothing is changed)"

# ---- 0. L4T ---------------------------------------------------------------------------------------------------------
say "0. JetPack / L4T"
L4T_MAJOR="" L4T_REV=""
if [ -f /etc/nv_tegra_release ]; then
  head -n 1 /etc/nv_tegra_release
  L4T_MAJOR=$(sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9.]*\).*/\1/p' /etc/nv_tegra_release)
  L4T_REV=$(sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9]*\)\.\([0-9]*\).*/\2/p' /etc/nv_tegra_release)
  if [ "$L4T_MAJOR" != "36" ]; then
    echo "L4T R$L4T_MAJOR is not JetPack 6 (R36): this setup (RT kernel repo, torch wheel) is for R36 only" >&2
    [ "$DRY" -eq 1 ] || exit 1
  fi
else
  warn "/etc/nv_tegra_release not found: not a Jetson. Continuing only because this is a dry run."
  [ "$DRY" -eq 1 ] || exit 1
  L4T_MAJOR=36; L4T_REV=4                                  # dry run: show the commands for R36.4
fi
[ -n "$RT_REPO" ] || RT_REPO="r${L4T_MAJOR}.${L4T_REV}"
echo "  L4T R${L4T_MAJOR}.${L4T_REV} -> rt-kernel repo ${RT_REPO}"

# ---- 1. PREEMPT_RT kernel -------------------------------------------------------------------------------------------
say "1. PREEMPT_RT kernel (NVIDIA OTA packages for ${RT_REPO})"
if ! skip rt; then
  if uname -v | grep -q PREEMPT_RT; then
    echo "  running kernel is already PREEMPT_RT: $(uname -r)"
  elif dpkg-query -W -f='${Status}' nvidia-l4t-rt-kernel 2>/dev/null | grep -q "install ok installed"; then
    echo "  nvidia-l4t-rt-kernel installed but not running: select the RT entry in extlinux.conf / reboot"; REBOOT=1
  else
    # VERIFY against "Real-Time Kernel Using OTA Update" in the Jetson Linux developer guide for this release
    SRC=/etc/apt/sources.list.d/nvidia-l4t-apt-source.list
    LINE="deb https://repo.download.nvidia.com/jetson/rt-kernel ${RT_REPO} main"
    if [ -f "$SRC" ] && grep -qF "$LINE" "$SRC"; then echo "  $SRC already lists the rt-kernel repo"
    else printf '+ append to %s: %s\n' "$SRC" "$LINE"; [ "$DRY" -eq 1 ] || echo "$LINE" | sudo tee -a "$SRC" > /dev/null; fi
    if confirm "Install the PREEMPT_RT kernel packages now?"; then
      run sudo apt-get update
      run sudo apt-get install -y nvidia-l4t-rt-kernel nvidia-l4t-rt-kernel-headers nvidia-l4t-rt-kernel-oot-modules \
          nvidia-l4t-display-rt-kernel
      REBOOT=1
    fi
  fi
fi

# ---- 2. kernel cmdline ----------------------------------------------------------------------------------------------
say "2. kernel args isolcpus=${RT_CORE} nohz_full=${RT_CORE} rcu_nocbs=${RT_CORE}"
if ! skip cmdline; then
  EXT=/boot/extlinux/extlinux.conf
  ARGS="isolcpus=${RT_CORE} nohz_full=${RT_CORE} rcu_nocbs=${RT_CORE}"
  if grep -q "isolcpus=${RT_CORE}" /proc/cmdline && grep -q "nohz_full=${RT_CORE}" /proc/cmdline \
     && grep -q "rcu_nocbs=${RT_CORE}" /proc/cmdline; then
    echo "  already active in /proc/cmdline"
  elif [ -f "$EXT" ] && grep -E '^\s*APPEND' "$EXT" | grep -q "isolcpus=${RT_CORE}"; then
    echo "  already in $EXT (reboot to activate)"; REBOOT=1
  else
    # every APPEND line (the default entry and the RT entry the rt-kernel package adds)
    run sudo cp -n "$EXT" "$EXT.franka.bak"
    run sudo sed -i -E "/^\s*APPEND/ s/\$/ ${ARGS}/" "$EXT"
    REBOOT=1
  fi
fi

# ---- 3. NIC IRQ -> RT core ------------------------------------------------------------------------------------------
say "3. Franka NIC IRQ affinity -> core ${RT_CORE}"
if ! skip irq; then
  if [ -z "$FRANKA_IF" ]; then
    for n in /sys/class/net/eth* /sys/class/net/enP* /sys/class/net/eno* /sys/class/net/enp*; do
      [ -e "$n" ] && { FRANKA_IF=${n##*/}; break; }
    done
    echo "  --franka-if not given; using ${FRANKA_IF:-<none found>} (the devkit's RJ45 port)"
  fi
  PIN=/usr/local/sbin/franka-irq-pin
  PIN_SH="#!/bin/sh
# pin every IRQ whose /proc/interrupts line names ${FRANKA_IF:-eth0} (or the devkit NIC drivers: Realtek r8168/r8169 on Orin Nano, Tegra eqos/mgbe) to core ${RT_CORE}
for irq in \$(grep -E '${FRANKA_IF:-eth0}|eqos|nvethernet|mgbe|r8168|r8169' /proc/interrupts | cut -d: -f1 | tr -d ' '); do
  echo ${RT_CORE} > /proc/irq/\$irq/smp_affinity_list 2>/dev/null && echo \"irq \$irq -> core ${RT_CORE}\"
done"
  UNIT="[Unit]
Description=Pin the Franka NIC IRQs to the RT core
After=network-pre.target
[Service]
Type=oneshot
ExecStart=${PIN}
[Install]
WantedBy=multi-user.target"
  write_root_file "$PIN" "$PIN_SH"
  run sudo chmod 755 "$PIN"
  write_root_file /etc/systemd/system/franka-irq-pin.service "$UNIT"
  run sudo systemctl daemon-reload
  run sudo systemctl enable --now franka-irq-pin.service
  if systemctl list-unit-files 2>/dev/null | grep -q '^irqbalance'; then
    run sudo systemctl disable --now irqbalance
  else
    echo "  irqbalance not installed"
  fi
fi

# ---- 4. power mode + clocks -----------------------------------------------------------------------------------------
say "4. nvpmodel max + jetson_clocks"
if ! skip power; then
  MODE=0
  if [ -f /etc/nvpmodel.conf ]; then
    M=$(grep -E '^< POWER_MODEL ID=[0-9]+ NAME=MAXN' /etc/nvpmodel.conf | sed -E 's/.*ID=([0-9]+).*/\1/' | tail -n 1)
    [ -n "$M" ] && MODE=$M
  fi
  CUR=$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: //p' || true)
  echo "  current: ${CUR:-unknown}; target mode ${MODE}"
  if command -v nvpmodel > /dev/null && nvpmodel -q 2>/dev/null | tail -n 1 | grep -qx "$MODE"; then
    echo "  already mode $MODE"
  else
    run sudo nvpmodel -m "$MODE"
  fi
  write_root_file /etc/systemd/system/franka-jetson-clocks.service "[Unit]
Description=jetson_clocks (max clocks) for franka_deploy
After=nvpmodel.service
[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
[Install]
WantedBy=multi-user.target"
  run sudo systemctl daemon-reload
  run sudo systemctl enable --now franka-jetson-clocks.service
fi

# ---- 5. rtprio / memlock --------------------------------------------------------------------------------------------
say "5. realtime limits"
if ! skip limits; then
  getent group realtime > /dev/null || run sudo groupadd realtime
  id -nG "$USER" | tr ' ' '\n' | grep -qx realtime || run sudo usermod -aG realtime "$USER"
  write_root_file /etc/security/limits.d/99-franka-rt.conf "@realtime - rtprio 99
@realtime - memlock unlimited"
  echo "  (new limits apply after logging out and in; check with: ulimit -r -l)"
fi

# ---- 6. static IP on the Franka link --------------------------------------------------------------------------------
say "6. static IP ${JETSON_IP} on ${FRANKA_IF:-<iface>} (connection franka-fci)"
if ! skip net; then
  if command -v nmcli > /dev/null && nmcli -t -f NAME con show 2>/dev/null | grep -qx franka-fci; then
    echo "  NetworkManager connection franka-fci exists:"; nmcli -g ipv4.addresses con show franka-fci || true
  else
    run sudo nmcli con add type ethernet ifname "${FRANKA_IF:-eth0}" con-name franka-fci ipv4.method manual \
        ipv4.addresses "$JETSON_IP" ipv4.never-default yes ipv6.method disabled connection.autoconnect yes
    run sudo nmcli con up franka-fci
  fi
  echo "  NOTE: the devkit has one RJ45 port; with it on the robot, use Wi-Fi or a USB NIC for the internet."
fi

# ---- 7. deployment venv + torch -------------------------------------------------------------------------------------
say "7. venv ${VENV} + NVIDIA torch wheel + inference pins"
if ! skip torch; then
  [ -x "$VENV/bin/python" ] || run /usr/bin/python3 -m venv --system-site-packages "$VENV"
  if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  torch with CUDA already importable: $("$VENV/bin/python" -c 'import torch; print(torch.__version__)')"
  else
    # torch 2.5 nv24.08 needs cuSPARSELt (VERIFY for the chosen wheel: NVIDIA's install guide lists it)
    run "$VENV/bin/python" -m pip install --no-cache-dir "$TORCH_WHEEL"
  fi
  run "$VENV/bin/python" -m pip install --no-cache-dir -r "$FD_DIR/requirements-inference.txt"
  run "$VENV/bin/python" -m pip install --no-cache-dir robotmq pyzmq scipy pyyaml loguru
  echo "  (behavior_prompting / real-env / robot-utils / franka_deploy: on PYTHONPATH from the repo checkout, as on the laptop)"
fi

# ---- 8. sensor part (training-stage script, referenced not copied) --------------------------------------------------
say "8. sensor_rig setup (sensor_rig/scripts/jetson_setup.sh)"
if [ "$WITH_SENSOR" -eq 1 ] && ! skip sensor; then
  SENSOR_ARGS=()
  [ "$DRY" -eq 1 ] && SENSOR_ARGS+=(--dry-run)
  [ "$YES" -eq 1 ] && SENSOR_ARGS+=(--yes)
  bash "$REPO_DIR/sensor_rig/scripts/jetson_setup.sh" "${SENSOR_ARGS[@]}"
else
  echo "  not run (pass --with-sensor-setup, or run sensor_rig/scripts/jetson_setup.sh yourself)"
fi

say "done"
[ "$REBOOT" -eq 1 ] && echo "REBOOT REQUIRED (RT kernel / kernel args)."
echo "Then: PYTHONPATH=$FD_DIR $VENV/bin/python -m franka_deploy.tools.jetson_verify"
