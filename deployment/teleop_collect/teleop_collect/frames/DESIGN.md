# frames: design

The laptop process that receives the Pi's camera stream, watches both cameras, keeps the newest JPEGs for the page
and, while an episode records, writes every received JPEG unchanged into the episode folder (PLAN.md 3, 6;
STRUCTURE.md 5).

## What it does and does not do

- **Does:** receive C0 `pi_stream/v1` (SUB, receive only); check every packet against the contract and the rig file;
  map Pi time to laptop time (one-way fit from `t_send_ns`); track `ok` / `stale` / `down` per camera with the pair
  rule; serve R1 (teleop: start/stop recording, health) and U2 (ui: previews); write `main.mkv`, `ultrawide.mkv`,
  `main_frames.csv`, `ultrawide_frames.csv`, `packets.csv`, `clock_pi.csv`; play a recorded Pi episode under `--mock`.
- **Does not:** decode, resize or re-encode any image (no 224×224 prep, no C1); talk to the PC; decide anything about
  episodes (teleop owns the folder, `meta.yaml`, `COMPLETE`, aborting and deleting). A recording failure is reported
  (R1 `recording_error`, summary `error`), never fatal for the process.

## Threads and data flow

| Thread | Runs |
|---|---|
| receiver | `Receiver.run`: `PiStreamPacketSource` (C0 SUB, feeds `OneWayClockMap`) → checks → `CameraHealth` → one `RecordItem` per accepted packet → sinks `LatestFrames.update`, `FrameRecorder.enqueue` |
| writer | one per recording: `FrameRecorder._writer` takes `RecordItem`s from a bounded queue and owns the six files |
| main | `FramesApp._main_loop`: `FramesServers.serve_once` (R1 then U2), the ready file, the stats line |
| fake-pi-frames | `--mock` only: `FakePiSender` on `station.mock_pi_endpoint()` |

```
Pi ──C0──► PiStreamPacketSource ──Packet──► Receiver ──RecordItem──► LatestFrames ◄── U2 get_preview (ui)
                 │ t_send, arrival             │ health                └─► FrameRecorder.enqueue ─queue─► writer ─► files
                 ▼                             ▼                                       ▲
           OneWayClockMap                CameraHealth ◄── R1 get_health / start / stop (teleop)
```

`RecordItem` (recorder.py) carries the raw bytes (`EncodedFrame`s), the Pi times, the mapped laptop times, the raw
one-way sample (`t_send_pi_ns`, `arrival_ns`) and the pair verdict. `ultrawide` is set only when the packet brings a
new ultrawide frame (pi_stream carries at most one); a reused one is `None`, so it is never written twice.

## Decisions

- **Bytes in = bytes on disk** (PLAN 6.1, 13.6): `MkvWriter` muxes each JPEG as one `av.Packet` of an `mjpeg`
  stream (`yuvj420p`, rig size, time_base 1/1000, `pts = i × 1000 / playback_fps`). Frame `i` of a video is row `i`
  of its CSV (`video_index`); true times are only in the CSV.
- **Flushing:** CSVs every 0.5 s; the Matroska muxer gets `cluster_time_limit=500` and `flush_packets=1`, so a crash
  loses at most about 0.5 s (an episode without `COMPLETE` is deleted by teleop anyway).
- **enqueue never blocks** (PLAN 6.4, 7): a full queue (`record.queue_max_packets`) or a write error becomes the
  recording's first `error`; from then on nothing is accepted or written; `stop()` still drains, closes, joins and
  returns the `RecordingSummary` with the error. teleop aborts the episode on it. Any exception in a write is such
  an error (the writer never dies of it), and `stop()` never waits forever: a dead writer's files are closed by
  `stop()`, a stuck one (no room for the stop mark within 1 s, or not finished within 2 s) is left as a daemon thread
  with the error set, so the main thread keeps answering R1/U2 and sees SIGTERM.
- **R1 start refusals**, first that applies (STRUCTURE 4.2): `already recording`, `<camera> camera <status>`,
  `Pi clock not synced`, `<dir>: not a folder`, `<dir>: already has recording files`. The last looks only at frames'
  own six files: teleop has already opened its recorders in the folder when it calls start.
- **Health unchanged from pd:** main counts every accepted packet; the ultrawide counts a frame once, when it is first
  an accepted pair, at its own mid-exposure (the frozen-ultrawide fix: a pair always refused makes it stale, then
  down). A Pi restart latches both cameras `down` for the rest of the run, and nothing more is recorded.
- **Unsynced packets** (no fresh clock) are checked and gap-counted but not accepted: no `RecordItem`, no health.
- **Mid-exposure times** use the rig file's time model (`camera_time_ns`, the default exposure when 0), as in pd:.
  `exposure_us` and `t_sensor_pi_ns` are stored raw next to them.
- **`pair_gap_ms`** is stored for refused pairs too (when the pair was found), so the CSV explains each verdict.
- **Two REP servers in the main thread**, R1 served first with half of `server.poll_s` each (R1 answers within
  ~10 ms with the default 0.02 s, well inside teleop's 50 ms `timeouts.health_s`).
- **Shutdown** (SIGINT/SIGTERM): the receiver is joined, an open recording is stopped (its files closed), then the
  fake Pi and the sockets; well inside the launcher's 5 s.

## Copied and changed (fd = rnoro/final-deployment @ 2ffe641)

| File | From fd:policy_deploy/policy_deploy/frames/ | Changes |
|---|---|---|
| camera_health, packet_pairer, fake_pi_sender, pi_recording, recording_packet_source | same name | imports only |
| pair_rule, _dataproc/{\_\_init\_\_,config,paths} | same name | none |
| receiver | receiver.py | no decoding / PacketBuffer; JPEG-header size check; RecordItem to sinks; order check against the last accepted packet; counters `accepted` (was `buffered`), no `unusable` |
| ultrawide_store | ultrawide_store.py | holds `EncodedFrame`s |
| pi_stream_source, packet_source | same | keep the raw one-way sample (`last_t_send_ns`, `last_arrival_ns`) |
| settings | settings.py | no select spacing/tolerance, no probe; + `record.*`; R1/U2 endpoints |
| app, \_\_main\_\_, errors, rig_file, fake_pi, pi_collect | same | R1/U2 instead of C1; ready via `ProcessArgs.write_ready`; `RecordingRefused`; module paths |
| mkv_writer, recorder, preview, servers | — | new |

Every file's header lists its changes exactly; `tests/frames/test_copies.py` checks the copies byte for byte against
`git show 2ffe641:...`.

## Tests (`tests/frames/`, free ports only)

pd:'s receiver, frozen-ultrawide, frame-time, pi_v1, health, pairer, fake-Pi, Pi-recording, pi_collect, rig-file,
settings and main tests, adapted (no images: bytes, times and verdicts on `RecordItem`s). New: `test_mkv_writer`
(byte round trip, playback times, on disk while open), `test_recorder` (rows per camera, reused ultrawide once,
`video_index` = row, column order, summary, refusals incl. teleop's files present, full queue, disk error, drain,
flush), `test_servers` (R1 refusals in order, stop/health, U2 only newer frames), `test_pipeline` (the process
against the fake Pi: a recorded episode's files against the bytes and times sent), `test_main` (subprocess, SIGTERM
while recording < 5 s), `test_real_episode` (`~/Downloads/episode_000045`, skipped if absent), `test_copies`.

## How to extend

- **A new recorded column:** add it to the `RecordItem` (receiver builds it), to `PACKET_COLUMNS` / `FRAME_COLUMNS`
  and the row in `FrameRecorder._write`, then PLAN 6.2 and `test_recorder`.
- **A new recorded file:** open it in `FrameRecorder._open`, add its name to `RECORDING_FILES` (refusal check), write
  it in the writer thread only, close it in `_close_files`.
- **A new R1/U2 request:** the message in `teleop_collect/contracts` (E1), a handler in `FramesServers`.
- **A third camera** would need C0, the rig file and `CAMERAS` together: not a frames-only change.
