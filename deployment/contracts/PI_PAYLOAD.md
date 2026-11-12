# C0: the Pi payload contract

What the Raspberry Pi 5 sends to the PC during deployment, exactly. This file is written for whoever implements the Pi
sender, and is the reference for `contracts/` (the C0 message classes and validators) and `process_frames/` (the
receiver). PLAN.md §4 C0 points here. Version: `c0.packet/v1`, `c0.time_sync_request/v1`, `c0.time_sync_reply/v1`.

## 1. The idea

The Pi pairs its two cameras into **packets**, like the iPhone pipeline (iPhUMI) did. There is **one packet per main
(IMX519) frame**. Each packet carries:

- that main frame;
- every ultrawide (IMX219) frame captured since the previous packet;
- a reference to the ultrawide frame that is this packet's **pair**.

The PC then picks frames over packets with iPhUMI's rule.

Unlike iPhUMI, the packet keeps **real capture times**: the PC computes the packet time from the main frame's sensor
timestamp and exposure, not from "arrival time minus a guess".

Why a packet can carry more than one ultrawide frame: H.264 frames depend on the frames before them, so **every
encoded ultrawide frame must reach the PC exactly once and in capture order**, even one that is nobody's pair. With
both cameras free-running at 30 FPS, a packet normally carries exactly one ultrawide frame. Timing jitter occasionally
gives 0 or 2.

## 2. Transport

| | |
|---|---|
| Library | ZeroMQ (libzmq ≥ 4.3; e.g. pyzmq on the Pi) |
| Frames socket | The Pi **binds a PUB** socket at `endpoints.pi_frames` (station.yaml, e.g. `tcp://0.0.0.0:5600` on the Pi). `process_frames` connects a SUB socket subscribed to everything (`""`). |
| Time-sync socket | The Pi **binds a REP** socket at `endpoints.pi_time_sync` (e.g. port 5601). `process_frames` connects a REQ. |
| One packet = one ZeroMQ **multipart** message | ZeroMQ delivers a multipart message whole or not at all, so a packet is never half-received. |
| Pi `SNDHWM` | 30 messages (about 1 s). If the PC falls behind, ZeroMQ drops whole packets, which the PC sees as a `packet_seq` gap. |
| Encoding of every map | msgpack, as everywhere in `contracts/` (PLAN §4): a map with string keys, in the order listed below, `schema` first; integers as msgpack ints, booleans as msgpack bools, strings as UTF-8. No floats are needed. |

The Pi may use the `contracts` Python package to build and check these messages (`contracts.c0_frames`), or write
the msgpack itself. Either way, the bytes must match this document.

## 3. The packet: parts of one multipart message

| Part | Content |
|---|---|
| 0 | the **header**, a msgpack map (section 4) |
| 1 | the **main frame**: one encoded frame, raw bytes |
| 2 … 1+N | the **ultrawide frames**, N ≥ 0 of them, raw bytes each, in capture order, matching `ultrawide` in the header one to one |

The number of parts is exactly `2 + len(header.ultrawide)`. Any other count is a contract violation.

## 4. The header (part 0)

| Key | Type | Meaning |
|---|---|---|
| `schema` | str | `"c0.packet/v1"` |
| `packet_seq` | uint64 | Counts **main frames captured** since the Pi sender started, from 0. It goes up by exactly 1 per packet, so a gap means lost packets. If the Pi itself drops a main frame, it still counts it. Going backwards means the sender restarted. |
| `main` | map | the main frame of part 1 (the frame map below) |
| `ultrawide` | list of maps | the N ultrawide frames of parts 2 … 1+N, in capture order (each a frame map); may be empty |
| `pair_seq` | int64 | the `frame_seq` of this packet's paired ultrawide frame, or **−1** if there is no pair (section 5). The paired frame is in this packet's `ultrawide` list or was sent in an earlier packet. |

**Frame map** (the same keys for `main` and each `ultrawide` entry):

| Key | Type | Meaning |
|---|---|---|
| `sensor` | str | `"imx519"` for main, `"imx219"` for ultrawide. Checked against the rig file, which catches swapped cameras. |
| `frame_seq` | uint64 | Counts **captured frames of this camera** from 0, per camera. It goes up by exactly 1 per frame of that camera, so a gap means that camera lost frames, which matters for its H.264 decoder. |
| `t_pi_ns` | int64 | libcamera `SensorTimestamp`, verbatim: the end of the first row's exposure, on the Pi's `CLOCK_BOOTTIME`, in ns. The same meaning as `t_ns` in the Pi's recordings (`camX_frames.csv`). |
| `exposure_ns` | int64 | libcamera `ExposureTime` × 1000 (ns); 0 if unknown, in which case `process_frames` uses its `default_exposure_ns` |
| `width`, `height` | uint32 | encoded frame size. Must equal the rig file (`rig_pi_v2.yaml`: main 1600×1200, ultrawide 1280×960). |
| `encoding` | str | `"h264"` (default) or `"jpeg"` (backup, PLAN §9). The same for every frame of a camera during a run. |
| `keyframe` | bool | H.264: true for a frame that decodes on its own (an IDR frame). JPEG: always true. |

## 5. Rules for the Pi sender

1. **One packet per main frame**, sent as soon as that main frame is encoded. The packet's main frame is part 1.
2. **Every encoded ultrawide frame is sent exactly once**: in the first packet sent after it was encoded, in capture
   order. Never resend one, and never skip one.
3. **Pairing:** `pair_seq` is the ultrawide frame with the **latest capture time not later than the main frame's
   capture time + 5 ms** (comparing `t_pi_ns`). It must be at most **50 ms** older than the main frame. If no
   ultrawide frame qualifies, `pair_seq` = −1.
   - This is "the newest ultrawide image at the moment of the main frame", which is what the iPhone pipeline used.
   - The 5 ms allows for the two cameras' exposures overlapping.
   - At 30 FPS each, a pair is normally 0–33 ms apart.
4. **H.264 settings:**
   - the same encoder settings as recording (main 15 Mbit/s, ultrawide 10 Mbit/s, as in `meta.json`);
   - Constrained Baseline profile (no B-frames);
   - one access unit per frame, Annex-B framing;
   - SPS/PPS sent before every keyframe;
   - a keyframe every 30 frames (1 s), as the recordings have.
5. **Counters:** `packet_seq` and each camera's `frame_seq` start at 0 and never reset during a run. A restarted
   sender starts again from 0, and the PC treats that as a restart (the camera goes down and the episode ends,
   PLAN §5).
6. Both cameras run at **30 FPS**, as recorded. A frame the Pi drops before encoding still advances its `frame_seq`,
   so the PC sees the gap.

## 6. Clock sync (the REQ/REP socket)

| Message | Map |
|---|---|
| request (PC → Pi) | `{schema: "c0.time_sync_request/v1"}` |
| reply (Pi → PC) | `{schema: "c0.time_sync_reply/v1", t_pi_ns: int64}`: the Pi's `CLOCK_BOOTTIME` in ns, read as close as possible to sending the reply |

`process_frames` sends about one request per second (10 back to back at start-up). It estimates the Pi → PC clock
offset and drift from the round trips (process_frames/STRUCTURE.md).

## 7. What the PC does with it (summary; the full design is in process_frames/STRUCTURE.md)

1. **Receive and check** every packet: part count, header, sizes, sensors, encodings. A broken packet is a contract
   violation: `process_frames` exits FATAL.
2. **Decode**, keeping each camera's chain:
   - part 1 goes to the main decoder;
   - parts 2 … are fed to the ultrawide decoder in order;
   - decoded ultrawide frames are kept by `frame_seq` (the last ~1 s).
   - A gap in a camera's `frame_seq` makes that camera unusable until its next keyframe.
3. **Packet time** = the main frame's mid-exposure time, `t_pi_ns − exposure_ns/2 + time_offset_ns` (the training
   rule), mapped to PC time by the clock sync. The paired ultrawide frame's own capture time is kept for logging and
   for the pair-gap statistic.
4. **Buffer:** the last ~1 s of packets. Each holds the main image, the paired ultrawide image (or none) and the
   packet time.
5. **Picking frames** (iPhUMI's `get_downsampled_camera_images` rule, over packets):
   - anchor on the newest packet's time T;
   - main images from the packets nearest T − 50 ms and T;
   - ultrawide images from the packets nearest T − 100 ms and T.
   - If a chosen packet is more than 2 frame periods from its target, or has no usable image for that camera, there
     are no images for this request (a "slow" case: the step is skipped).
6. **C1 reply:** unchanged in shape. The ultrawide images' times are their **packets'** times, as in iPhUMI, where
   both images of a packet share one timestamp.

## 8. Example (one packet)

```
part 0: {schema: "c0.packet/v1", packet_seq: 1201,
         main: {sensor: "imx519", frame_seq: 1201, t_pi_ns: 932118204000, exposure_ns: 8000000,
                width: 1600, height: 1200, encoding: "h264", keyframe: false},
         ultrawide: [{sensor: "imx219", frame_seq: 1188, t_pi_ns: 932112937000, exposure_ns: 6000000,
                      width: 1280, height: 960, encoding: "h264", keyframe: false}],
         pair_seq: 1188}
part 1: <main H.264 access unit, ~60 kB>
part 2: <ultrawide H.264 access unit, ~40 kB>
```

Here the ultrawide frame 1188 was captured 5.3 ms before main frame 1201, and is its pair.
