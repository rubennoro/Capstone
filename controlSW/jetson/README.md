# Jetson Orin Nano: running the trained policy

Plan context: [../details.md](../details.md), [../parts.md](../parts.md). The Jetson only does **inference**; training stays on the cluster.

1. **Preflight (read-only):** `bash controlSW/jetson/preflight.sh`. Shows JetPack version, Python, memory, disk, what's already installed.
2. **JetPack:** we use 7.2.x (Ubuntu 24.04, Python 3.12). `TORCH_INDEX` in `setup_jetson.sh` defaults to the upstream cu132 wheels; a
   harmless "compute capability 8.7" warning from torch is expected. The training env was Python 3.10, so version pins may need adjusting.
3. **Install:** `bash controlSW/jetson/setup_jetson.sh`. Idempotent: it reuses `~/bp-venv` and an existing CUDA torch.
4. **If `robotmq` has no aarch64 wheel:** build it from source (ask Claude; needs the upstream repo URL).
5. **Get the trained files onto the Jetson** (not in git; both live on the cluster):
   - checkpoint `best_slim_epoch=0050-...ckpt` (2.5 GB), the prompt dataset `firstreal-single.zarr.zip` (845 MB).
   - Needs a Northeastern Explorer account with access to the cluster paths in `results/block_raise/run_001_2026-10-05/README.md`.
6. **Serve:**
   `python behavior_prompting/behavior_prompting/deployment/detached_inference.py -i <best.ckpt> --dataset-path <firstreal-single.zarr.zip> --device cuda`
   The first run downloads CLIP weights from Hugging Face (internet once) and writes a `.yaml` next to the checkpoint.
7. **Time one call**, then try bf16 autocast / fewer DDIM steps if over ~0.5-1 s.
