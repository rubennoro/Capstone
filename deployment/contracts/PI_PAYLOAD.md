# C0: the Pi payload contract

What the Raspberry Pi 5 sends to the PC during deployment, and what the PC expects of it. The Pi sender is fixed; the
PC adapts to it. Format: **`pi_stream/v1`**, read by `process_frames` with `stream: pi_v1` (the default).
Code: `contracts/pi_stream.py` (parse and check) and `process_frames/pi_stream_source.py` (convert). PLAN.md §4 C0
points here.

Sections 1–8 further down are an **appendix**: the original C0 design (msgpack packets plus a time-sync socket). The
Pi doesn't send it. It's kept because `process_frames` still uses that packet form internally, and `stream: c0`
still reads it.

## 0. `pi_stream/v1`

### 0.1 Transport

| | |
|---|---|
| Library | ZeroMQ |
| Socket | The Pi **binds a PUB** socket on port **5555**. `process_frames` connects a SUB, subscribed to everything, at `endpoints.pi_frames` (`station.yaml`, e.g. `tcp://10.42.0.78:5555`). |
| Direction | Pi → PC only. Nothing is sent back, and there is no time-sync socket. |
| Message | One ZeroMQ **multipart** message per main frame, delivered whole or not at all. |
| Frames | Both cameras **640×480 (4:3) at 30 FPS**, JPEG. Rig file: `process_frames/rigs/rig_pi_v2_640.yaml`. |

### 0.2 Parts

| Part | Content |
|---|---|
| 0 | the header, a JSON object (0.3) |
| 1 | the main frame (IMX519), one JPEG |
| 2 | the ultrawide frame (IMX219), one JPEG |

Exactly 3 parts, neither JPEG empty.

### 0.3 Header

All keys are required, with no others. Integers are JSON integers, never booleans or floats.

| Key | Type | Meaning |
|---|---|---|
| `version` | int | `1` |
| `packet_no` | int ≥ 0 | Counts packets since the sender started. Equal to `main.frame_no` in practice. |
| `t_send_ns` | int | The Pi clock (the same clock as `t_sensor_ns`), read just before sending |
| `clock_offset_ns` | int | Sent by the Pi; not used by the PC |
| `main` | object | The main frame of part 1 (frame object below) |
| `ultrawide` | object | The ultrawide frame of part 2 (frame object below) |
| `skew_ns` | int | `main.t_mid_ns − ultrawide.t_mid_ns`; not used by the PC |
| `uw_reused` | bool | Part 2 repeats an earlier packet's ultrawide frame, with the same `frame_no` |

**Frame object** (`main` and `ultrawide`):

| Key | Type | Meaning |
|---|---|---|
| `frame_no` | int ≥ 0 | Counts this camera's frames since the sender started |
| `t_sensor_ns` | int | libcamera `SensorTimestamp`: the end of the first row's exposure, on the Pi clock, in ns |
| `exposure_us` | int ≥ 0 | libcamera `ExposureTime`, in µs |
| `t_mid_ns` | int | Mid-exposure: `t_sensor_ns − exposure_us·1000/2` |

### 0.4 What the PC relies on

1. **`packet_no` goes up by 1 per packet.** A gap means lost packets (`packets lost` in the stats). Going backwards
   means the sender restarted: `process_frames` marks both cameras down and ignores the rest, so restart it (or
   `deploy run`) after restarting the Pi sender.
2. **Each camera's `frame_no` and `t_sensor_ns` only go up.** The exception is a repeated ultrawide frame
   (`uw_reused`), which carries the earlier frame's numbers. A skipped ultrawide `frame_no` shows up as a `drop`
   count, which is harmless with JPEG.
3. **The JPEGs are exactly the rig file's size** (640×480). Any other size is `FATAL`, naming the camera.
4. **`t_send_ns` is on the same clock as `t_sensor_ns`.** The PC maps Pi times to its own clock with it (0.5).
5. **Pairing is the Pi's choice:** the ultrawide frame in each packet is that packet's pair. The PC checks only that
   it exists. The gap shows in the stats as `pair gap`. On the live Pi, `skew_ns` was about 16 ms.
6. **Packets keep coming.** No new frame for 0.35 s makes a camera stale, and 2 s makes it down. No packet for 10 s
   makes the clock stale.

### 0.5 What the PC does with it

1. **Check** each message: parts, keys, types, `version`. Any violation is a contract violation and `process_frames`
   exits `FATAL`.
2. **Convert** it to the internal C0 packet (appendix §3–4):
   - `packet_seq` ← `packet_no`.
   - Frame map: `frame_seq` ← `frame_no`, `t_pi_ns` ← `t_sensor_ns`, `exposure_ns` ← `exposure_us`·1000. Width and
     height come from the JPEG header. `encoding` jpeg, `keyframe` true.
   - `pair_seq` ← the ultrawide `frame_no`.
   - The ultrawide frame is passed on, and decoded, only when its `frame_no` is new. A repeat is not decoded again,
     but stays the pair.
3. **Clock, one-way:** each packet's `t_send_ns` against its PC arrival time. In each second, the least-delayed
   packets are fitted for offset and drift (`OneWayClockMap`). PC times come out late by the minimum Wi-Fi latency,
   a few ms.
4. **From here on it's the same as appendix §7:** packet time is the main frame's mid-exposure. Then the 1 s buffer,
   frame picking (main images at T − 50 ms and T, ultrawide at T − 100 ms and T), and the C1 224×224 images.

### 0.6 Example

A packet as the live Pi sent it on 2026-11-12:

```
part 0: {"version": 1, "packet_no": 800, "t_send_ns": 1269437832529, "clock_offset_ns": 1794539164607668815,
         "main":      {"frame_no": 800, "t_sensor_ns": 1269392013000, "exposure_us": 25640, "t_mid_ns": 1269379193000},
         "ultrawide": {"frame_no": 799, "t_sensor_ns": 1269412129000, "exposure_us": 33287, "t_mid_ns": 1269395485500},
         "skew_ns": 16292500, "uw_reused": false}
part 1: <main JPEG, 640x480, ~48 kB>
part 2: <ultrawide JPEG, 640x480, ~75 kB>
```

- **Pair:** ultrawide frame 799 is the pair of main frame 800; their mid-exposures are 16.3 ms apart.
- **Timing:** the packet left the Pi 45.8 ms after the main frame's `t_sensor_ns`.

---

# Appendix: the original C0 design (`stream: c0`, not sent by the Pi)

Kept as the definition of the internal packet form (`contracts.c0_frames`, used by `process_frames` for both streams),
and as the format `stream: c0` reads. Code and docs cite these section numbers, for example "§5 rule 3" for the pairing
rule. Versions: `c0.packet/v1`, `c0.time_sync_request/v1`, `c0.time_sync_reply/v1`. Sizes below are the original
full-resolution ones (`rig_pi_v2.yaml`).

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

## 5. Rules for a C0 sender

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
