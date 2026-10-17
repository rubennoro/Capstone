#!/usr/bin/env bash
# Build the argus_sof helper on the Jetson (JetPack 6). Output: sensor_rig/native/argus_sof/build/argus_sof
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
echo "built: $(pwd)/build/argus_sof   (rig.yaml: capture.backends.<cam>: argus_sof)"
