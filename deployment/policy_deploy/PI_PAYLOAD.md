<!-- adapted from rnoro/deployment:deployment/contracts/PI_PAYLOAD.md @ 91b178b
     changes: §0 unchanged apart from the one skew_ns sign note (in §0.3 and §0.6), §0.4 item 5 (the laptop judges the
     pair) and the fake Pi note in §0.6; the appendix keeps only §3-4 (the
     internal packet form, c0.packet/v1, which policy_deploy.frames uses internally); the c0 msgpack sender, its
     transport and the time_sync parts (§1, §2, §5-§8) are dropped: the Pi contract is final (PLAN §8.2). Code names:
     process_frames is now policy_deploy.frames, contracts.pi_stream is policy_deploy.contracts.pi_stream,
     contracts.c0_frames is policy_deploy.contracts.c0_packet, "the PC" in §0 is the laptop. -->
# C0: the Pi payload contract

What the Raspberry Pi 5 sends to the PC during deployment, and what the PC expects of it. The Pi sender is fixed; the
PC adapts to it. Format: **`pi_stream/v1`**, read by `policy_deploy.frames` (the only format; no `stream` setting).
Code: `contracts/pi_stream.py` (parse and check) and `process_frames/pi_stream_source.py` (convert). PLAN.md §4 C0
points here.

The **appendix** below (§3–4 only) is the internal packet form (`c0.packet/v1`) the receiver works on; the Pi never
sends it.

## 0. `pi_stream/v1`

### 0.1 Transport

| | |
|---|---|
| Library | ZeroMQ |
| Socket | The Pi **binds a PUB** socket on port **5555**. `process_frames` connects a SUB, subscribed to everything, at `tcp://<pi.address>:<pi.port>` (`station.yaml pi:`, e.g. `tcp://192.168.50.30:5555`). |
| Direction | Pi → PC only. Nothing is sent back, and there is no time-sync socket. |
| Message | One ZeroMQ **multipart** message per main frame, delivered whole or not at all. |
| Frames | Both cameras **640×480 (4:3) at 30 FPS**, JPEG. Rig file: `deployment/contracts/rigs/rig_pi_v3_640.yaml` (generated; DIRECTIONS.md "Camera constants"). |

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
| `skew_ns` | int | `main.t_mid_ns − ultrawide.t_mid_ns`; not used by the PC. **Note:** the live Pi sends this with the opposite sign (`ultrawide.t_mid_ns − main.t_mid_ns`, see §0.6), so nothing may use `skew_ns`; the pair gap is always computed from the two `t_mid_ns` |
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
   `policy_deploy run`) after restarting the Pi sender.
2. **Each camera's `frame_no` and `t_sensor_ns` only go up.** The exception is a repeated ultrawide frame
   (`uw_reused`), which carries the earlier frame's numbers. A skipped ultrawide `frame_no` shows up as a `drop`
   count, which is harmless with JPEG.
3. **The JPEGs are exactly the rig file's size** (640×480). Any other size is `FATAL`, naming the camera.
4. **`t_send_ns` is on the same clock as `t_sensor_ns`.** The PC maps Pi times to its own clock with it (0.5).
5. **Pairing is the Pi's choice, judged by the PC:** the ultrawide frame in each packet is that packet's pair. The PC
   uses it only if its mid-exposure lies within [main − 60 ms, main + 25 ms] (`pair_max_age_ms`, `pair_max_lead_ms`)
   and the same ultrawide frame has been the pair of at most 2 consecutive packets (`pair_max_reuse`); otherwise that
   packet has no ultrawide image, and a camera whose pairs keep being refused goes stale, then down
   (policy_deploy/STRUCTURE.md §4.3 steps 9-10). The gap (`main.t_mid_ns − ultrawide.t_mid_ns`, from the two
   `t_mid_ns`) shows in the stats as `pair gap`. On the live Pi it was about −16 ms (the ultrawide newer).
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
4. **From here on it's the serving of policy_deploy/STRUCTURE.md §4.3:** packet time is the main frame's mid-exposure. Then the 1 s buffer,
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


**Note (sign of `skew_ns`):** in this packet `main.t_mid_ns − ultrawide.t_mid_ns` = −16 292 500 ns, but the Pi sent
`skew_ns` = +16 292 500: the live Pi's `skew_ns` has the opposite sign to §0.3's definition. Nothing in policy_deploy
reads `skew_ns`; `pi_collect` computes the pair gap from the two `t_mid_ns`.

**The fake Pi** (`--mock`, tests) copies these live numbers: `skew_ns` with the live sign (`ultrawide.t_mid_ns −
main.t_mid_ns`, positive when the ultrawide is newer) and each packet sent 45.8 ms after its main frame's
`t_sensor_ns`.

---

# Appendix: the internal packet form (`c0.packet/v1`, never sent by the Pi)

The receiver converts each `pi_stream/v1` message into this packet form (`policy_deploy.contracts.c0_packet`) and
works on it, as `rnoro/deployment` did. It never crosses a socket. Section numbers are those of the original design;
the sender rules, the transport and the time-sync socket (§1, §2, §5-§8) are not carried over.

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
| `frame_seq` | uint64 | Counts **captured frames of this camera** from 0, per camera. It goes up by exactly 1 per frame of that camera, so a gap means that camera lost frames, which is harmless with JPEG (`frame_decoder.py` knows only `jpeg`). |
| `t_pi_ns` | int64 | libcamera `SensorTimestamp`, verbatim: the end of the first row's exposure, on the Pi's `CLOCK_BOOTTIME`, in ns. The same meaning as `t_ns` in the Pi's recordings (`camX_frames.csv`). |
| `exposure_ns` | int64 | libcamera `ExposureTime` × 1000 (ns); 0 if unknown, in which case `process_frames` uses its `default_exposure_ns` |
| `width`, `height` | uint32 | encoded frame size. Must equal the rig file (`../contracts/rigs/rig_pi_v3_640.yaml`: both 640×480). |
| `encoding` | str | always `"jpeg"` here (`pi_stream_source.py`; `"h264"` was the original design's default). The same for every frame of a camera during a run. |
| `keyframe` | bool | H.264: true for a frame that decodes on its own (an IDR frame). JPEG: always true. |

