# controlSW parts: compute options for the gripper

What could run the whole pipeline (capture + policy inference + Franka control) on the iPhUMI gripper, how much
engineering each option takes, and what it costs. **Budget: $300.** Plan details: [details.md](details.md).

Prices are approximate list prices, so check before ordering. Inference times are back-of-envelope estimates for the
run_001 policy (~410 M parameters, ~120–150 GFLOPs per call), not measurements. The target is **≲ 0.5–1 s per call**,
about the duration of one executed action chunk.

## Summary

| Option | Cost | Est. time per call | Engineering | Size on gripper | Fits $300? |
|---|---|---|---|---|---|
| **A. Raspberry Pi 5** (what we have) | $0 (owned) | ~3–10 s | **High**: heavy model optimization, likely retraining | 85×56 mm | Yes |
| **B. Pi 5 + Hailo-8 AI HAT+** | ~$110 | Unknown, likely 1–3 s | **Very high**: compile and quantize the model for Hailo; the diffusion loop partly stays on the CPU | Pi + HAT layer | Yes |
| **C. Jetson Orin Nano Super dev kit** (recommended) | ~$250 (+ ~$30 SSD) | ~0.3–0.6 s (fp32), ~0.15–0.3 s (bf16) | **Low**: runs the model unmodified | 100×79×21 mm + fan | **Yes** |
| **D. Jetson Orin NX 16 GB + mini carrier** | ~$600–900 | ~0.2–0.4 s (fp32) | **Low** | ~80–90×55–60 mm | No |
| **E. Orin Nano/NX module + custom carrier PCB** | ~$300–700 module + PCB | Same as C / D | **Medium**: PCB design, no model work | ~70×50 mm, smallest | No (module alone uses most of it) |
| **F. Jetson AGX Orin** | ~$2,000 | ~0.05–0.15 s | Low | 110×110×72 mm, heavy | No |
| **G. Compute off the gripper** (base-mounted) | Depends on what's mounted | Depends | **Low–medium**: cable routing | Cameras only | Depends |
| H. Rockchip RK3588 / Qualcomm QCS6490 boards | ~$100–400 | Unknown | **Very high**: convert to RKNN / QNN; transformers poorly supported | Pi-sized | Some |

**Recommendation: C.** It's the only option within budget that runs the model as is, and it leaves CPU headroom for the
Franka control loop. Later upgrade path: move the same module onto a smaller carrier (D or E) without changing software.

## The options in more detail

### A. Raspberry Pi 5 (owned)

- **What works:** capture and the software stack. PyTorch has aarch64 wheels.
- **What doesn't:** inference is CPU-only, so ~3–10 s per call on 4 cores, and slower on the 2 cores left once capture and
  the Franka loop get theirs. This means stop-and-go motion.
- **Engineering:**
  - No retraining (~1–3 s per call at best): DDIM steps 16 → 4–8, int8 quantization, ONNX Runtime/ExecuTorch export.
  - To get under a second: retrain a smaller policy (smaller image encoder, lower resolution, smaller U-Net, distillation).
  - Franka: libfranka 1 kHz loop on an isolated core of a 4-core CPU that's busy with inference. This is the highest-risk
    control setup of all the options.
- **Cost:** $0.

### B. Pi 5 + Hailo-8 AI HAT+ (26 TOPS)

- **What it adds:** an int8 neural accelerator attached to the Pi.
- **Engineering:**
  - Export the model to ONNX, compile it with Hailo's toolchain, and calibrate the int8 quantization.
  - The ViT image encoders are partially supported. The 16-step diffusion loop with prompt cross-attention is a poor fit,
    so the U-Net likely stays on the CPU.
  - Expect weeks of conversion work and some accuracy loss. Still not "without optimization".
- **Cost:** ~$110 for the HAT.

### C. Jetson Orin Nano Super Developer Kit (recommended)

- **Specs:** 1024-core Ampere GPU, 6-core Cortex-A78AE CPU, 8 GB shared memory, gigabit Ethernet, CSI camera connectors,
  USB3.
- **What works:**
  - `detached_inference.py --device cuda` with the run_001 checkpoint, unmodified. NVIDIA ships CUDA PyTorch wheels for
    JetPack.
  - The whole pipeline on one board: an isolated core for libfranka (JetPack has an official PREEMPT_RT kernel option),
    a core for capture, and the rest for inference.
- **Engineering:**
  - Setup: flash JetPack, install CUDA PyTorch + `behavior_prompting`, build `robotmq` and libfranka (or franky) for
    ARM64.
  - Same Franka integration work as every option: `FrankaCartesian` controller, `umi_franka_task`, TCP calibration (see
    [details.md](details.md) §3).
  - Optional: a one-line bf16 autocast for ~2× faster inference.
- **Limits:**
  - 8 GB shared CPU/GPU memory. The model needs ~1.6 GB (fp32), so it fits, but it's the tightest part of the setup.
  - Bigger and heavier than the Pi (heatsink + fan).
  - 7–25 W at 19 V, so it needs a power line to the gripper.
- **Cost:**

  | Item | Approx. cost |
  |---|---|
  | Jetson Orin Nano Super Developer Kit (includes power supply) | ~$249 |
  | NVMe SSD, 256 GB, M.2 2280 | ~$25–35 |
  | **Total** | **~$275–285** |

### D. Jetson Orin NX 16 GB on a mini carrier

- **Over C:** ~1.5–2× faster, 16 GB memory, 8 CPU cores, and a smaller off-the-shelf carrier (vendors include Connect Tech,
  Auvidea and Seeed; check current models for the needed I/O).
- **Engineering:** same as C. Same software, same module form factor.
- **Cost:** ~$600–900 (module + carrier). Over budget.

### E. Orin Nano/NX module on a custom carrier PCB

- **What it is:** the smallest way to run the model unmodified. A board we design with only power, Ethernet and camera
  inputs, sized to the module (~70×50 mm) and built into the gripper housing with an integrated heatsink.
- **Engineering:** PCB design and bring-up following NVIDIA's carrier design guide. Weeks of hardware work, but no model
  work.
- **Cost:** a module (~$300+ for the Nano, ~$600+ for the NX 16 GB) plus PCB fabrication and assembly (~$50–200 for small
  runs). Over budget now; a later step once the pipeline works on C.

### F. Jetson AGX Orin (32/64 GB)

- **Over C:** ~4× faster or more, lots of memory, 12 CPU cores.
- **Downsides:** 110×110×72 mm dev kit with a heavy heatsink, 15–60 W. Too big for the gripper; reasonable if mounted
  off the gripper (G).
- **Cost:** ~$2,000. Over budget.

### G. Compute off the gripper

- **What it is:** only the cameras (and optionally the Pi for capture and gripper I/O) stay on the gripper. The compute
  board mounts on the Franka's base or a nearby link, connected by Ethernet / camera cables. It's still self-contained,
  with no workstation.
- **Engineering:** cable routing along the arm. RMQ already works over TCP, so the software barely changes.
- **Cost:** whatever board is mounted. The Orin Nano Super kit from C still fits the budget here.
- **Why consider it:** the gripper stays as light as today, and there's no size limit on the compute board.

### H. Rockchip RK3588 / Qualcomm QCS6490 boards

- **What they are:** Pi-sized boards with NPUs (Radxa / Orange Pi for RK3588, RB3 Gen 2 / SOMs for QCS6490).
- **Engineering:** convert the model to RKNN or QNN/SNPE. Transformer and diffusion support is weak, so this is the most
  conversion work of any option for an uncertain result.
- **Cost:** ~$100–400 depending on board and RAM.

## Engineering that's needed with every option

No matter which compute board is used:
- `FrankaCartesian` controller + `umi_franka_task` in `real-env`.
- A libfranka 1 kHz control loop on a PREEMPT_RT kernel with an isolated core, proven stable under load. This is the
  go/no-go test.
- TCP frame calibration to match the training camera frame, verified with an open-loop demo replay.
- The capture process for the in-house cameras, gripper width, and Franka EE pose.
- Power and Ethernet routing to the gripper.
