#!/usr/bin/env bash
# READ-ONLY check of the Jetson. Installs and changes nothing. Run first and paste the output back.
echo "== Board / JetPack =="
cat /etc/nv_tegra_release 2>/dev/null || echo "no /etc/nv_tegra_release (not a Jetson, or not flashed with JetPack)"
dpkg -l nvidia-jetpack 2>/dev/null | tail -1 || true
echo "power mode:"; (sudo -n nvpmodel -q 2>/dev/null || echo "  (run: sudo nvpmodel -q)")
echo "== OS / Python =="; lsb_release -ds 2>/dev/null; python3 --version; uname -r
echo "== CUDA =="; (nvcc --version 2>/dev/null | tail -1) || echo "nvcc not on PATH (often /usr/local/cuda/bin; fine for inference)"
echo "== Memory / disk =="; free -h | sed -n 1,2p; df -h / | tail -1
echo "== Python packages (system python) =="
python3 - <<'PY'
for m in ("torch","torchvision","numpy","cv2","zarr","robotmq","transformers"):
    try:
        mod=__import__(m); print(f"{m}: {getattr(mod,'__version__','?')}", "| cuda:" + str(mod.cuda.is_available()) if m=="torch" else "")
    except Exception as e: print(f"{m}: not installed")
PY
echo "== Existing env from our script? =="; ls -d "$HOME/bp-venv" 2>/dev/null || echo "no ~/bp-venv"
echo "== Network (needed once to download CLIP weights from Hugging Face) =="
curl -sI --max-time 8 https://huggingface.co | head -1 || echo "cannot reach huggingface.co"
