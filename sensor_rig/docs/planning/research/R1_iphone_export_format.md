# R1: iPhUMI iPhone export format (one episode, byte-level-ish spec)

Sources:
- Swift app: `/home/ruben/Capstone/iPhUMI/ios_app/iPhUMI/` (mostly `DataCollection/DemonstrationData.swift` (DD), `DataCollection/ViewController.swift` (VC), `VideoWriter/*.swift`, `Util/DateManager.swift`, `Util/PhoneIdentifier.swift`).
- Real exports: `/home/ruben/Capstone/data/raw/2026-10-01/`, 47 folders: 2 `grippercalibration`, 45 `demonstration`, all `left`, iPhone 16 Pro (`iPhone17,1`).
- Python consumer: `/home/ruben/Capstone/iPhUMI/python_package/iphumi/demonstration_processing/`.
- Tooling: no ffprobe, cv2, or av on the system. I probed with PyAV 19.0.1 installed in a scratch venv (`.../scratchpad/venv_tools/env`), plus a hand-written MP4 atom walker (`.../scratchpad/atoms.py`). Every number below was measured on the real files unless marked "from code".

---------------------------------------------------------------------------------------------------

## 0. TL;DR for a writer

Per episode, write one folder `<NAME>/` containing:

| type | files |
|---|---|
| demonstration (with depth) | `<NAME>.json`, `<NAME>_rgb.mp4`, `<NAME>_ultrawidergb.mp4`, `<NAME>_depth.raw`, `<NAME>_depthpreview.mp4` |
| demonstration (no LiDAR) | `<NAME>.json`, `<NAME>_rgb.mp4`, `<NAME>_ultrawidergb.mp4`, with `hasDepth:false` and `depthTimes:[]`, `depthFrameIsRepeat:[]` |
| grippercalibration | `<NAME>.json`, `<NAME>_ultrawidergb.mp4` |

`NAME = <UTC ISO time with ':'→'-'>_<5 digits 1-9>_<session>_<demonstration|grippercalibration>_<left|right|head>`

Hard invariants the Python pipeline relies on:
1. `len(poseTimes) == len(poseTransforms) == len(rgbTimes) == frameCount == number of frames in _rgb.mp4` (demonstration).
2. `len(ultrawideRGBTimes) == frameCount`. Non-empty entries are at the main-frame indices that carried an ultrawide image; **the count of non-empty entries == frame count of `_ultrawidergb.mp4`**.
3. `len(depthTimes) == len(depthFrameIsRepeat) == frameCount == filesize(_depth.raw) / (256*192*2)` when depth is present.
4. Ultrawide MP4 PTS (seconds) == (main-frame timestamp of that ultrawide frame) − (main-frame timestamp of the first ultrawide frame). The detect stage uses `frame.pts * time_base + firstNonEmptyIndex/60` as the tag time.
5. The ultrawide image must be a **pinhole (zero-distortion) image** matching `ultrawideCameraIntrinsics`, because ArUco PnP runs with `dist_coeffs = 0`.
6. `deviceIdentifier` + `gripperID` must be a key pair in `calibration/gripper_calibration.yaml`, or the group stage asserts.

---------------------------------------------------------------------------------------------------

## 1. Naming and directory layout

### 1.1 Recording name construction (from code)

VC.swift:1091-1102 (on the initiating device):
```swift
recordingStartTime = Date()
recordingName = DateManager.getISOFormatter().string(from: recordingStartTime!)   // "2026-10-01T21:13:57.489Z"
var number = String()
for _ in 1...5 { number += "\(Int.random(in: 1...9))" }   // digits 1..9 only, never 0
recordingName += "_\(number)"
```
VC.swift:1336-1351 (setRecordingState):
```swift
var recordingName = "\(recordingName)".replacingOccurrences(of: ":", with: "-")
recordingName += "_\(sessionName)"
recordingName += getDemonstrationType() == .GripperCalibration ? "_grippercalibration" : "_demonstration"
recordingName += "_\(phoneSide.recordingNameComponent)"      // "left" | "right" | "head"  (VC.swift:68-82)
```
`DateManager.getISOFormatter()` (Util/DateManager.swift:13-17) is an `ISO8601DateFormatter` with `[.withInternetDateTime, .withFractionalSeconds]`. Its default time zone is GMT, so the result is always UTC with a literal `Z` and exactly 3 fractional digits.

Fields, using `2026-10-01T21-13-57.489Z_75494_firstreal_demonstration_left` as the example:

| field | example | rule |
|---|---|---|
| timestamp | `2026-10-01T21-13-57.489Z` | `YYYY-MM-DDTHH-MM-SS.mmmZ` in UTC, `:` replaced by `-`. It is `recordingStartTime` (wall-clock `Date()` at the button press on the initiator). |
| random id | `75494` | 5 chars, each `1`-`9`. In bimanual mode the initiator sends it to peers, so both sides share timestamp+id. |
| session | `firstreal` | Sanitized at VC.swift:2446-2452: only alphanumerics, `-`, `_`, space are kept, then `_` and space become `-`. **It never contains `_`**, and the Python side splits on `_` and expects exactly 5 parts (group.py). Default `no-session` blocks recording. |
| type | `demonstration` / `grippercalibration` | Note that the JSON `type` field uses `gripper_calibration` (with underscore) while the name uses `grippercalibration`. |
| side | `left` / `right` / `head` | |

### 1.2 Folder layout

DD.swift:368-421 and ExportViewController.swift:72-75: export goes to `<picked folder>/iPhUMI_export/<YYYY-MM-DD>/<NAME>/`. `<YYYY-MM-DD>` is `recordingName` up to the `T` (UTC date). Files inside:

| DemonstrationSaveType | filename |
|---|---|
| JSON | `<NAME>.json` |
| RGB | `<NAME>_rgb.mp4` |
| UltrawideRGB | `<NAME>_ultrawidergb.mp4` |
| DepthMap | `<NAME>_depth.raw` |
| DepthPreviewMap | `<NAME>_depthpreview.mp4` |

`saveExternally` (DD.swift:621-638) copies whichever of these exist and nothing else. There is no metadata.json, no device-info file, no app version, and no hidden files. `data/raw/2026-10-01/` corresponds to the `<YYYY-MM-DD>` level.

Python's `group` stage (process_stages/group.py) globs `**/*.json` recursively under `iphone_dir`, takes the basename, splits it on `_` (it must have 5 parts), and copies `<dir>/<title>*` to `tmp_demonstrations/<YYYY-MM-DD>/<title-without-side>/<side>{.json,_rgb.mp4,...}`. So the folder nesting is not strictly required, but the file basenames are.

### 1.3 Observed sizes

| episode | file | bytes |
|---|---|---|
| `2026-10-01T21-10-44.057Z_15647_firstreal_grippercalibration_left` (1589 frames, 26.5 s) | `.json` | 11,976 |
| | `_ultrawidergb.mp4` (265 frames) | 6,805,037 |
| `2026-10-01T21-13-57.489Z_75494_firstreal_demonstration_left` (996 frames, 16.6 s) | `.json` | 245,404 |
| | `_rgb.mp4` | 39,073,371 (~18.8 Mbit/s) |
| | `_ultrawidergb.mp4` (166 frames) | 4,269,069 (~2.05 Mbit/s) |
| | `_depth.raw` | 97,910,784 = 996 × 256 × 192 × 2 |
| | `_depthpreview.mp4` | 696,490 (~0.33 Mbit/s) |

The JSON is about 245 bytes per frame for demonstrations.

---------------------------------------------------------------------------------------------------

## 2. The JSON file

### 2.1 Encoding

- Swift `JSONEncoder()` with default options (DD.swift:579-582): **compact** (no spaces, no newlines, no trailing newline), UTF-8, `/` not escaped.
- **Key order is unspecified and differs between files.** It is hash-order: the calibration and demo files above have completely different key orders. Consumers do not care.
- Floats are Swift `Float` (float32), printed with the shortest round-trip representation: `-0.64491296`, `0.99999994`, `5.3933415`. Integral values print **without** a `.0`: `0`, `1`. Small magnitudes use exponent form: `-6.782484e-05`. Python equivalent: `s = str(np.float32(x)); s = s[:-2] if s.endswith('.0') else s`. numpy's float32 str also switches to exponent below 1e-4. Exact exponent thresholds are an unverified uncertainty, and this detail is cosmetic.
- Bools are `true`/`false`. Ints are plain.

### 2.2 Schema: all 32 keys (DD.swift:115-192 CodingKeys; decoder at 194-237)

"Req" means the decoder in the app uses `decode` (not `try?`), or the Python pipeline reads the key.

| key | type | demo example | calib example | notes |
|---|---|---|---|---|
| `poseTimes` | [str] | ISO-ms strings, len=frameCount | `[]` | Only filled for `type==demonstration` (DD:345-348). **Python: align.py:104** |
| `poseTransforms` | [[[f32×4]×4]] | 4×4 row-major | `[]` | Camera-to-world (see §4). **align.py:106** |
| `rgbTimes` | [str] | identical to poseTimes | `[]` | **align.py:139, gripper_util, label** |
| `ultrawideRGBTimes` | [str] | len=frameCount, `""` where no UW image | same, len=frameCount | **detect.py:65, align.py:142** |
| `depthTimes` | [str] | identical to poseTimes if depth, else `[]` | `[]` | align.py uses `.get("depthTimes", rgbTimes)` |
| `depthFrameIsRepeat` | [bool] | len=frameCount, all false observed | `[]` | true = depth missing this frame, last depth repeated (DD:332-341) |
| `recordingStartTime` | str | `"2026-10-01T21:13:57.489Z"` | same form | Wall clock at button press (initiator's, shared to peers) |
| `side` | str | `"left"` | `"left"` | |
| `type` | str enum | `"demonstration"` | `"gripper_calibration"` | DD:96-99 |
| `gripperCalibrationRunName` | str | `"2026-10-01T21-10-44.057Z_15647_firstreal_grippercalibration_left"` | `""` | Full NAME of the calibration episode, **with side suffix**. Python strips the side (generic_util.get_demonstration_path). Must exist in the same export. |
| `sessionName` | str | `"firstreal"` | `"firstreal"` | |
| `gripperID` | str | `"default"` | `"default"` | Key into gripper_calibration.yaml |
| `hasRGB` | bool | true | false | `type==demonstration` |
| `hasUltrawideRGB` | bool | true | true | always true |
| `hasDepth` | bool | true | false | Demo: true unless the first depthMap is nil (no LiDAR) (DD:297-300) |
| `frameCount` | int | 996 | 1589 | Number of `logFrame` calls = main ARFrames while recording |
| `taskStartTimestamps` | [str] | `[]` | `[]` | ISO-ms, parallel to taskNames |
| `taskEndTimestamps` | [str] | `[]` | `[]` | |
| `taskNames` | [str] | `[]` | `[]` | (all real data: labelType None) |
| `narrationStartTimestamps` | [str] | `[]` | `[]` | Per spoken word segment (DD:452-464) |
| `narrationEndTimestamps` | [str] | `[]` | `[]` | |
| `narrationTexts` | [str] | `[]` | `[]` | |
| `labelType` | str enum | `"None"` | `"None"` | `"Narration"`/`"Predefined"`/`"None"` (DD:101-105) |
| `hasAudio` | bool | false | false | true only if contact mic connected (VC:855-858) |
| `isVoiceHost` | bool | false | false | Multipeer voice host |
| `sidesPresent` | [str] | `["left"]` | `["left"]` | All sides in this recording (bimanual: `["left","right"]`). **validate.py deletes demos missing any listed side** |
| `formatVersion` | str | `"v1"` | `"v1"` | |
| `deviceName` | str | `"iPhone 16 Pro"` | same | Cosmetic (`UIDevice.commonName`, PhoneIdentifier.swift:31-82) |
| `deviceIdentifier` | str | `"iPhone17,1"` | same | **Required**: group.py → `build_demonstration_iphone_calibration` looks up `calibration_dict[gripperID][deviceIdentifier]` (asserts) |
| `mainCameraIntrinsics` | [[f32×3]×3] | `[[1278.4507,0,954.6028],[0,1278.4507,723.5324],[0,0,1]]` | same | Row-major K, pixels, for 1920×1440 |
| `ultrawideCameraIntrinsics` | [[f32×3]×3] | `[[246.64919,0,320.16495],[0,246.64919,239.53679],[0,0,1]]` | same | Row-major K for 640×480. **Used by detect (ArUco PnP)** |
| `isErrorCorrection` | bool | false | false | |

The app's decoder requires every key except `depthFrameIsRepeat`, `gripperID`, `isVoiceHost`, `sidesPresent`, `isErrorCorrection` (those use `try?`). This only matters if files are re-imported into the app. To be safe, write all 32.

### 2.3 Example (demonstration, arrays truncated, key order arbitrary)

```json
{"rgbTimes":["2026-10-01T21:13:57.449Z","2026-10-01T21:13:57.466Z","2026-10-01T21:13:57.483Z", ... ,"2026-10-01T21:14:14.039Z"],
 "taskEndTimestamps":[],
 "poseTransforms":[[[-0.64491296,0.50917274,-0.56993884,0.9632324],[0.008718036,0.75059605,0.6607038,-0.066551395],[0.7642062,0.42112765,-0.48850816,5.3933415],[0,0,0,0.99999994]], ...],
 "deviceIdentifier":"iPhone17,1","narrationTexts":[],
 "depthTimes":["2026-10-01T21:13:57.449Z", ...],"side":"left","frameCount":996,
 "ultrawideRGBTimes":["","2026-10-01T21:13:57.466Z","","","","","","2026-10-01T21:13:57.566Z", ... ,"",""],
 "taskStartTimestamps":[],"ultrawideCameraIntrinsics":[[246.64919,0,320.16495],[0,246.64919,239.53679],[0,0,1]],
 "hasAudio":false,"mainCameraIntrinsics":[[1278.4507,0,954.6028],[0,1278.4507,723.5324],[0,0,1]],
 "hasRGB":true,"depthFrameIsRepeat":[false,false, ...],"recordingStartTime":"2026-10-01T21:13:57.489Z",
 "formatVersion":"v1","hasUltrawideRGB":true,"hasDepth":true,"narrationEndTimestamps":[],"type":"demonstration",
 "isVoiceHost":false,"sidesPresent":["left"],"gripperID":"default","deviceName":"iPhone 16 Pro",
 "poseTimes":["2026-10-01T21:13:57.449Z", ...],"labelType":"None","isErrorCorrection":false,"taskNames":[],
 "gripperCalibrationRunName":"2026-10-01T21-10-44.057Z_15647_firstreal_grippercalibration_left",
 "sessionName":"firstreal","narrationStartTimestamps":[]}
```

### 2.4 Example (gripper calibration)

```json
{"hasDepth":false,"hasRGB":false,"deviceName":"iPhone 16 Pro","depthTimes":[],"deviceIdentifier":"iPhone17,1",
 "poseTransforms":[],"narrationStartTimestamps":[],"ultrawideCameraIntrinsics":[[246.64919,0,320.16495],[0,246.64919,239.53679],[0,0,1]],
 "frameCount":1589,"type":"gripper_calibration","taskEndTimestamps":[],
 "mainCameraIntrinsics":[[1278.4507,0,954.6028],[0,1278.4507,723.5324],[0,0,1]],"gripperCalibrationRunName":"",
 "hasAudio":false,"narrationTexts":[],"sidesPresent":["left"],"rgbTimes":[],"depthFrameIsRepeat":[],
 "ultrawideRGBTimes":["","","","2026-10-01T21:10:44.066Z","","","","","","2026-10-01T21:10:44.166Z", ... ],
 "isErrorCorrection":false,"recordingStartTime":"2026-10-01T21:10:44.057Z","poseTimes":[],"narrationEndTimestamps":[],
 "gripperID":"default","sessionName":"firstreal","formatVersion":"v1","hasUltrawideRGB":true,"taskStartTimestamps":[],
 "side":"left","isVoiceHost":false,"labelType":"None","taskNames":[]}
```
In a calibration, poses and main video are not saved, but `frameCount` and `ultrawideRGBTimes` still count every 60 Hz main ARFrame.

### 2.5 Task / narration fields when used (from code)

- Predefined: `taskStartTimestamps[i]`, `taskEndTimestamps[i]` are ISO-ms wall-clock times (`Date()` at button/voice events, VC:1381,1408,1571-1578). `taskNames[i]` is the task string. Assert: end > start (DD:563).
- Narration: speech segments become `narrationStartTimestamps`/`narrationEndTimestamps`/`narrationTexts` (one per recognized word). The tasks are then derived from them (DD:437-570).
- label.py converts task timestamps to aligned-frame indices via searchsorted on `aligned_frame_times` (built from poseTimes). `scripts/label_whole_demo_as_task.py` (repo) fills one task spanning `rgbTimes[0]..rgbTimes[-1]` for labelType None data, **after grouping**.

---------------------------------------------------------------------------------------------------

## 3. Timestamps and frame matching

### 3.1 Clock

- The only per-frame time source is `ARFrame.timestamp` (seconds since boot, the mach host-time / `systemUptime` timebase). VC.swift:1811.
- It is converted to wall-clock UNIX time with an offset computed at recording start (VC.swift:684-689, called at 1332):
  `arKitTimeOffset = Date().timeIntervalSince1970 - ProcessInfo.processInfo.systemUptime`, then
  `poseTime = Date(timeIntervalSince1970: frame.timestamp + arKitTimeOffset)` (VC:2083-2090).
  So the JSON time strings are **capture time on a wall clock, sampled once per recording**. They are formatted with millisecond precision (ISO formatter; rounding vs truncation is not verified).
- `recordingStartTime` is `Date()` at the button press. Observed: in all 45 demos, `recordingStartTime − poseTimes[0]` = **+14 to +51 ms** (the first logged frame was captured slightly before the press time because of pipeline latency).
- In bimanual mode, alignment across phones relies on these wall clocks (align.py intersects pose time ranges). Each phone's own NTP/wall clock is the sync mechanism. Jetson + second device: use a common, disciplined clock (PTP/chrony) and convert sensor timestamps to UNIX time the same way.

### 3.2 One record per main frame

`DemonstrationData.logFrame` (DD.swift:271-350) is called once per ARFrame (`session(_:didUpdate:)`, VC:1850) while recording. Each call:
- computes `poseTime` string (same for all streams in that call), `frameTime = CMTimeMake(Int64(arkitTimestamp*10000), 10000)`;
- appends the main image to rgb mp4 at `frameTime` and `rgbTimes.append(poseTime)`;
- if ARKit attached a new ultrawide image (private ivar `_capturedUltraWideImage`, VC:87-100), appends it to the ultrawide mp4 at the **same main `frameTime`** and `ultrawideRGBTimes.append(poseTime)`, else `ultrawideRGBTimes.append("")`. The ultrawide's own timestamp (`_ultraWideImageTimestamp`) is read but **not used** (VC:1847-1848);
- depth: appends `depthMap ?? lastDepthPixelBuffer` to raw + preview at `frameTime`, `depthTimes.append(poseTime)`, `depthFrameIsRepeat.append(depthMap == nil)`;
- `poseTimes.append(poseTime)`, `poseTransforms.append(frame.camera.transform)` (demonstration only);
- `frameCount += 1`.

So `poseTimes == rgbTimes == depthTimes` element-wise (verified for all 45 demos), and `ultrawideRGBTimes[i]` is either `""` or equal to `poseTimes[i]` (verified).

### 3.3 Rates (observed)

| stream | rate | resolution | evidence |
|---|---|---|---|
| main RGB + pose + depth | 60 Hz (frame dt 16/17 ms). 0 gaps >17 ms in the sample demo. | 1920×1440 | poseTimes diffs; mp4 PTS steps 10/600 s |
| ultrawide | 10 Hz: every 6th main frame (index gaps: 6 ×159, 5 ×3, 7 ×3) | 640×480 | ultrawideRGBTimes, mp4 PTS steps ~60/600 |
| depth | 60 Hz, never repeated in this dataset | 256×192 | depthFrameIsRepeat all false |

First non-empty ultrawide index per episode: 0..5 (observed). The ultrawide writer starts on that frame (DD:287-295). detect.py turns this into `time_offset = i/60`.

### 3.4 Dropped frames

- There is no placeholder for a dropped main frame. It just doesn't exist: the frame is absent from every list and video, and the timestamps show a larger gap. The video PTS also reflect the real gap (VFR timing).
- Ultrawide "drop" = `""` in `ultrawideRGBTimes`.
- Depth drop = repeat of previous depth frame with `depthFrameIsRepeat=true`.
- `VideoWriter.appendVideo` silently drops a frame if `!isReadyForMoreMediaData` (VideoWriter.swift:76-79) **but still appends the time string**. That would break the count invariant; it never happened in the real data (all counts match). A writer must never do this.
- Abort rules (episode discarded and never exported): fps < 31 after the first 1 s (VC:1821); position speed > 7 m/s between frames (VC:306, 1829-1844). Recording can't start if fps ≤ 50 (VC:1240) or session is `no-session`, etc. (VC:1206-1245).
- Python's align.py resamples poses on a uniform 60 Hz grid (`np.arange(start, end, 1/60)`), interpolating pose and picking the most-recent-past frame index for each stream. So small irregularities are tolerated.

---------------------------------------------------------------------------------------------------

## 4. Pose

- Source: `frame.camera.transform` (VC:1810). ARKit **camera-to-world** transform (`W_T_I`) of the main (wide) camera.
- Camera frame (ARKit convention, sensor-native landscape orientation; images are 1920×1440, landscape, no rotation in mp4 tkhd matrix): **+x right along the image's long axis, +y up (image up), +z backward toward the user; looking along −z.** This differs from the OpenCV optical frame (x right, y down, z forward) by a 180° rotation about x: `T_cv = T_arkit @ diag(1,-1,-1,1)`.
- World frame: ARKit gravity-aligned (`worldAlignment` default `.gravity`): +y up (opposite gravity), x/z horizontal. Heading and origin are wherever the session started. Single-device mode only adds an identity anchor named "local world frame" (VC:867-870) and doesn't move the origin. Observed positions in the sample: x∈[0.96,1.22], y∈[−0.29,−0.07], z∈[5.36,5.63] m, i.e. the origin is far from the gripper (session start position). The docs say the origin doesn't matter (relative actions).
- Bimanual: the non-host calls `session.setWorldOrigin(relativeTransform: hostWorldAnchor.transform)` (VC:1676-1677), so both phones share the host's world frame. Peers must be within 1 m to record (VC:1230).
- Encoding: 4×4 float32, **row-major** list of rows (custom Codable at DD.swift:16-46 transposes simd's column-major storage). Translation is in `[r][3]` for r=0..2. Units are meters. Last row `[0,0,0,1]`; float noise like `0.99999994` occurs. No quaternions anywhere in the export.
- Sanity check on real data: det(R)=1±4e-7. The mean camera +z axis in world is [−0.46, 0.76, −0.41], so the optical axis (−z) points downward/forward, consistent with a gripper looking down.
- Downstream: `W_T_TCP = W_T_I @ I_T_TCP` with `tcp_pose_in_arkit_frame` from `calibration/gripper_calibration.yaml` (gripper_util.py:186-220). **For the Jetson, the VIO pose must be expressed as the pose of the main camera in the ARKit camera convention** (or the yaml's `tcp_pose_in_arkit_frame` must be re-derived for the new mount, under a new `deviceIdentifier` key).
- No tracking state, quality, or covariance field is exported. Tracking-quality checks happen only live (abort rules) and in Python (`align_validate` max pose delta, `filters.max_position_delta`).
- No per-frame intrinsics are stored.

---------------------------------------------------------------------------------------------------

## 5. Intrinsics and camera metadata

- `mainCameraIntrinsics` is `session.currentFrame.camera.intrinsics`, captured **once** on the first ARFrame after the view loads (VC:1767-1769, guarded by `== nil`). It is not per recording and not per frame, and it is reused for every episode recorded in that view session.
- `ultrawideCameraIntrinsics` is `frame._ultraWideCamera.intrinsics` (private ivar), also captured once (VC:1771-1773).
- Both are 3×3 row-major `[[fx,0,cx],[0,fy,cy],[0,0,1]]` in pixels of the recorded resolution. **No distortion coefficients anywhere.** detect_aruco_iphone.py uses `dist_coeffs = zeros(5)`.
- Observed variants in this export:
  - 39 files: main `fx=fy=1406.3428, cx=953.2503, cy=723.22003`; UW `fx=fy=259.01538, cx=320.2935, cy=239.67455`
  - 8 files: main `fx=fy=1278.4507, cx=954.6028, cy=723.5324`; UW `fx=fy=246.64919, cx=320.16495, cy=239.53679`

  This implies main HFOV ≈ 2·atan(960/1278..1406) ≈ 69-74°, and ultrawide HFOV ≈ 2·atan(320/247..259) ≈ 102-105° at 640×480. The ARKit ultrawide stream is therefore a ~100° **rectified pinhole** image. A 120° Jetson fisheye must be undistorted/cropped to a pinhole 640×480 before encoding, and its K written here.
- Other camera metadata: `deviceName`, `deviceIdentifier` only. No lens position, exposure, or serials. (The app stores the ultrawide lens position in UserDefaults only, VC:1776-1782, not exported.)
- Per-device extrinsics (TCP offset, AR-tag Z distance from ultrawide) live in Python `calibration/gripper_calibration.yaml` keyed by `gripperID` → `deviceIdentifier`. The current file has `default: {"iPhone16,1": ..., "iPhone17,1": ...}`, with fields `gripper_transform.tcp_pose_in_arkit_frame` (4×4) and `gripper_transform.artag_z_distance_from_ultrawide` (0.098079 m, used as ±0.08 m depth gate for tags). A Jetson device needs a new identifier entry (e.g. `"jetson-umi-v1"`) or must masquerade as `iPhone17,1`.

---------------------------------------------------------------------------------------------------

## 6. Video files (all three MP4s)

Common to `_rgb.mp4`, `_ultrawidergb.mp4`, `_depthpreview.mp4` (AVAssetWriter, VideoWriter.swift / DepthPreviewVideoWriter.swift):

- Container: **QuickTime MOV** despite the `.mp4` extension: `fileType: .mov`, ftyp major brand `qt  `, minor 0, compatible `qt  `. Atom order: `ftyp(20) wide(8) mdat moov`. moov is **at the end** (not faststart).
- One video track (track_id 1). **No audio track** when `hasAudio=false` (all real data). No timecode (`tmcd`) track, no metadata track, no `udta`. Track has `tapt` (aperture) atom and a single-entry edit list `elst [(segment_dur=track_dur, media_time=0, rate=1)]`.
- Movie/media timescale **600**. PTS are VFR, from ARKit timestamps (CMTime 1/10000 rescaled to 1/600 → 1.67 ms quantization). First PTS = 0. Main-video PTS match `poseTimes − poseTimes[0]` within 1.3 ms (verified).
- `creation_time` metadata (mvhd + stream) = wall clock truncated to whole seconds, e.g. `2026-10-01T21:13:57.000000Z`. Stream metadata: `handler_name: "Core Media Video"`, `encoder: "H.264"`, `vendor_id: [0][0][0][0]`, `language: und`. The `minf` has a second `hdlr` `alis` "Core Media Data Handler".
- Codec: H.264 High profile, hardware encoder, default AVFoundation settings (only codec/width/height specified). **B-frames present** (ctts, first dts negative: −20 / −120 ticks), pattern `I B B P B P B P ...`, **keyframe every 29 frames** (sometimes 27). Stream `sdtp` present.
- Pixel format `yuvj420p` (full range, `color_range=2`). colr `nclc` primaries=1 (BT.709), transfer=1 (BT.709), matrix=6 (BT.601/SMPTE170M) for rgb/ultrawide. Depth preview has no colr atom (all unspecified). SAR 0/1 (unset). tkhd matrix identity (no rotation).
- PyAV's `average_rate` (frames/duration) is e.g. `149400/2491` ≈ 59.98 for main, `24900/2491` ≈ 9.996 for ultrawide. `base_rate`/`guessed_rate` = 60 or 10. cv2 `CAP_PROP_FPS` is used only for visualization output fps (label.py:155, visualize.py:161).
- The track/movie duration equals last PTS + one frame duration of the **main** stream for all three files (e.g. 9964/600 s). On the ultrawide track, the last frame is therefore "held" until the main video's end (writer session end time). This is cosmetic.

| file | res | frames (sample) | profile/level (avcC) | bitrate | first PTS | PTS step (1/600) |
|---|---|---|---|---|---|---|
| `_rgb.mp4` | 1920×1440 | 996 = frameCount | High@5.0 (`64 00 32`) | ~18.8 Mb/s | 0 | 10 (9/11 jitter) |
| `_ultrawidergb.mp4` | 640×480 | 166 = #non-empty UW times | High@3.0 (`64 00 1e`) | ~2.05 Mb/s | 0 | 60 (50/59/61/70) |
| `_depthpreview.mp4` | 256×192 | 996 = depthTimes len | High@1.2 (`64 00 0c`) | ~0.33 Mb/s | 0 | 10 |

Source pixel formats (from code): main and ultrawide are the ARKit `capturedImage` / ultrawide `CVPixelBuffer`s (YCbCr 420f biplanar), fed through an adaptor declared as 32ARGB. Depth preview is `OneComponent32Float` = depth[m]/1.0, which the encoder turns into luma. Measured: preview gray ≈ `clip(depth_m, 0, 1) * 255` (corr 0.9999), with chroma neutral.

Audio (from code, not observed): if a contact mic is connected, an AAC 48 kHz stereo track is added to **both** `_rgb.mp4` and `_ultrawidergb.mp4` (VideoWriter.swift:46-58, DD:263-269), and `hasAudio=true`. The audio sample buffers come from ARKit (`providesAudioData`).

Requirements from the pipeline (what actually matters): readable by PyAV and cv2; frame count == JSON counts; decoded frame order; ultrawide PTS in seconds relative to first UW frame (detect uses `frame.pts * stream.time_base`); `stream.frames` header count is used as a tqdm total and in gen_replay_buffer_zarr.py:588 (`n_source_frames = in_stream.frames`), so **the container must report the frame count** (MOV/MP4 stsz does; MKV wouldn't). Codec/profile/level/B-frames/brand/timescale are cosmetic. libx264 via PyAV into an MP4/MOV with VFR PTS is functionally equivalent.

---------------------------------------------------------------------------------------------------

## 7. Depth: `_depth.raw`

- DepthVideoWriter.swift: headerless concatenation of frames. Each frame is ARKit `sceneDepth.depthMap` (DepthFloat32, meters) converted with `vImageConvert_PlanarFtoPlanar16F` to **IEEE float16, little-endian**, row-major, **192 rows × 256 cols**, no row padding, no footer. Frame bytes = 98,304.
- Reader: `np.fromfile(p, '<f2').reshape(-1, 192, 256)` (utils/depth_util.py:5-12 uses `(192, 256)`).
- Sample stats: min 0.0736 m, max 2.762 m, median 0.344 m, no NaN. Frame count = `len(depthTimes)`. Repeats (`depthFrameIsRepeat`) are full duplicate frames.
- The depth preview mp4 is a visualization only. Python reads `.raw`. Default dataset build has `include_depth: false` (config/build_umi_dataset_iphone.yaml:10). visualize.py checks `os.path.exists(depth_path)` and writes placeholders if absent. **The Jetson rig has no depth sensor: write `hasDepth:false`, `depthTimes:[]`, `depthFrameIsRepeat:[]` and omit both depth files, exactly like a non-LiDAR iPhone.** Caveat: align.py then computes `depth_frame_idx` over an empty list, which yields −1 (via `np.clip(...,0,-1)`). That is harmless only while depth isn't consumed. Alternatively write synthetic depth, which is not recommended.

---------------------------------------------------------------------------------------------------

## 8. Gripper calibration episodes

- `type: "gripper_calibration"`, name suffix `_grippercalibration_<side>`. `hasRGB=false`, `hasDepth=false`, `hasUltrawideRGB=true`. Only `.json` + `_ultrawidergb.mp4` are written.
- `poseTimes`, `poseTransforms`, `rgbTimes`, `depthTimes`, `depthFrameIsRepeat`, and task/narration lists are all `[]`. `gripperCalibrationRunName = ""`. `frameCount` and `ultrawideRGBTimes` still have one entry per 60 Hz main frame.
- Recorded by opening/closing the gripper in front of the ultrawide camera (ArUco finger tags). Python `detect` then `calibrate` produce `<side>_gripper_range.json` (min/max width, left↔right linear maps) from the tag tvecs.
- Demonstrations reference it via `gripperCalibrationRunName` (full name with side). The app requires the referenced calibration to be for the same side and the same gripperID (VC:1214-1223), and Python requires it to be present in the processed demonstrations dir (generic_util.get_gripper_calibration_run_dir asserts). create_session auto-includes referenced calibrations.

---------------------------------------------------------------------------------------------------

## 9. Multi-device / bimanual

- Each device writes its **own** folder. All share `<timestamp>_<id>_<session>_<type>` (from the initiator via `RecordingMessage`, VC:1115-1129 / 2244-2265) and differ in `_<side>` suffix. `recordingStartTime` is identical across devices.
- `sidesPresent` lists every participating side (e.g. `["left","right"]` or with `"head"`). Python's group_validate deletes the demo if any listed side's JSON is missing. `create_session` with no `sides=` mode requires all demos to have the same side set. The real data are single-gripper `["left"]`. In single mode, left and right are canonicalized to `right` in the replay buffer.
- `isVoiceHost`: which device ran speech recognition. `head` side: no gripper calibration needed (`gripperCalibrationRunName` may be empty, VC:1213).
- Shared world frame: see §4 (setWorldOrigin from the host's "world frame" anchor). No peer info (IDs, relative transforms) is written to the export.

---------------------------------------------------------------------------------------------------

## 10. Required vs cosmetic summary

**Required (pipeline breaks or misbehaves without it):** file basenames and 5-part `_` split; `type` in the name; JSON keys `poseTimes`, `poseTransforms`, `rgbTimes`, `ultrawideRGBTimes`, `depthTimes` (may be `[]`), `ultrawideCameraIntrinsics`, `mainCameraIntrinsics` (copied into calibration yaml), `deviceIdentifier`, `gripperID`, `gripperCalibrationRunName`, `sessionName`, `sidesPresent`, `taskNames`/`taskStartTimestamps`/`taskEndTimestamps` (label stage reads them; `[]` ok), `hasDepth` (share script), `type`; count invariants; ISO-8601 UTC times with `Z` (parsed by `datetime.fromisoformat` after `Z`→`+00:00`); pinhole undistorted ultrawide; VFR-correct ultrawide PTS; container frame count.

**Used lightly:** `isErrorCorrection`, `labelType` (repo script), `note` (optional, display only, never written by app).

**Cosmetic:** `deviceName`, `formatVersion`, `hasAudio`, `isVoiceHost`, `recordingStartTime` (not read by Python), `frameCount` (not read by Python, but keep consistent), narration lists, key order, float formatting, MOV vs MP4 brand, codec profile/level/GOP/B-frames, timescale 600, creation_time, handler names.

---------------------------------------------------------------------------------------------------

## 11. Writer recipe (Python, per episode)

1. `t_press = time.time()`, `name_ts = iso_ms(t_press).replace(':','-')`, `rid = ''.join(random.choice('123456789') for _ in range(5))`, `NAME = f"{name_ts}_{rid}_{session}_{kind}_{side}"`. Folder `out/iPhUMI_export/{name_ts[:10]}/NAME/`.
2. Drive everything off the **main camera frames at 60 Hz**. For each main frame i with capture time t_i (sensor clock converted to UNIX via an offset fixed at start):
   - `s = iso_ms(t_i)` (`YYYY-MM-DDTHH:MM:SS.mmmZ`, UTC).
   - Demo: write main image (1920×1440 or chosen res, with matching `mainCameraIntrinsics`) at PTS `t_i − t_0`; append s to `rgbTimes`, `poseTimes`; append `T_world_cam` (ARKit camera convention, gravity-aligned y-up world, meters, 4×4 row-major float32) to `poseTransforms`, interpolated from VIO to t_i.
   - Ultrawide: if an ultrawide frame arrived since the last main frame (target every 6th main frame), undistort it to 640×480 pinhole, write it at PTS `t_i − t_firstUW`, append s; else append `""`. Start the UW writer on the first frame that has one. Never write an ultrawide frame without its time string, or vice versa.
   - `frameCount += 1`.
3. Calibration: same loop, but don't write main video/poses. Only the ultrawide mp4 + `ultrawideRGBTimes` + `frameCount`.
4. Write JSON compactly with all 32 keys (§2.2). `formatVersion:"v1"`, `labelType:"None"` (or Predefined with tasks), `hasDepth:false`, depth arrays `[]`, `deviceIdentifier` matching a gripper_calibration.yaml entry.
5. Drop the whole episode (delete the folder) on abort conditions (fps < 31 after 1 s, > 7 m/s jump, VIO failure), as the app does.

---------------------------------------------------------------------------------------------------

## 12. Open uncertainties

1. ISO8601DateFormatter: unknown whether milliseconds are rounded or truncated (±1 ms; immaterial).
2. Swift Float shortest-repr exponent thresholds vs numpy (cosmetic).
3. Whether the ARKit ultrawide image is truly distortion-free or only approximately so. The pipeline assumes pinhole, which matters for how the Jetson ultrawide must be rectified.
4. ARKit's exact ultrawide-to-main frame association (the private ivar presumably holds the ultrawide image captured nearest to that main frame). The real UW capture time is discarded and the main frame time is used instead (up to ~8 ms error at 60 Hz).
5. Audio track layout untested (no `hasAudio:true` sample in this export).
6. No `head`-side or bimanual sample in this export. The bimanual details come from code only.
7. Behaviour of downstream stages with `hasDepth:false` / missing depth files was checked by reading code only (align gives `depth_frame_idx=-1`, visualize handles absence, build default `include_depth:false`), not run.
8. The `default`/`iPhone17,1` gripper_calibration.yaml entry is a copy of the iPhone 15 Pro mount. A new device ID entry will need a measured `tcp_pose_in_arkit_frame` for the Jetson mount (expressed relative to the main camera in the ARKit camera convention).
