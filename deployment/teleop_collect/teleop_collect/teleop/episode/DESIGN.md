# teleop/episode: design

Files only, no sockets. Used by teleop's episode flow (`episode_flow.py`, E3). All new code; nothing copied from fd:.

## What it does, and does not

- Does: the task and episode folders under `data_root` (PLAN 6.1), teleop's five per-episode files (PLAN 6.2), and
  `meta.yaml` + `COMPLETE` at a save (PLAN 6.3).
- Does not: the frames files (`*.mkv`, `packets.csv`, `*_frames.csv`, `clock_pi.csv`: E2, in the same folder), the
  session log (E3), or decide when an episode is aborted (E3).

## Classes and data flow

| File | Class | Job |
|---|---|---|
| `store.py` | `TaskStore` | tasks under `data_root`: `tasks`, `create`, `open`, `clean_incomplete`, `delete_task(name)` |
| | `probe_writable` | `data_root` made if missing and written to (start-up step 9, the page's storage folder change) |
| | `Task` | `episodes()` (complete ones, read from meta.yaml), `new_episode()`, `delete(name)` |
| | `EpisodeDir` | `finish(meta)`: meta.yaml then COMPLETE, fsynced; `discard()` |
| | `EpisodeInfo` | name, outcome, duration_s, frames_main (E3 converts it to the U1 record) |
| `recorders.py` | `EpisodeRecording` | opens and owns the five files below; `counts()`, `error`, `close()` |
| `meta.py` | `build_meta` | the meta.yaml dict, plain YAML types |

```
Start: Task.new_episode() ─► EpisodeDir ─► EpisodeRecording(ep, waypoints=K)
  pc-listener thread ─► states.write(ArmState, arrival_ns, t_ns)       robot_states.csv
  main loop ─► commands.write(...)  keys.write(...)  pc_clock.hello/fit(...)  events.write(...)
Stop:  close() ─► success/fail: EpisodeDir.finish(build_meta(..., counts=rec.counts())) │ discard: EpisodeDir.discard()
Start-up: TaskStore.clean_incomplete()
```

Files and columns (PLAN 6.2): `robot_states.csv` (`t_pc_ns, t_ns, arrival_ns, seq, mode, O_T_TCP_00..33,
O_T_TCP_target_00..33, q_0..6, dq_0..6, gripper_width_m, gate_*, loop_late, error`); `commands.csv` (`t_send_ns,
t_reply_ns, request, chunk_id, n`, K slots `wp<k>_t_pc_ns, wp<k>_00..33, wp<k>_width_m`, the twist (6), `slow`,
`accepted, shortened, skipped, rejected, rejected_reasons, ack_ok, ack_reason`); `keys.csv` (`t_ns, held, slow,
focus`); `clock_pc.csv` (`kind` = hello | fit, `t_send_ns, t_recv_ns, t_pc_ns, t_ns, offset_ns, drift_ppm, samples,
used, rtt_min_ms`); `events.jsonl` (`{"t_ns", "kind", "detail"}`). Floats are written as `repr` (exact round trip),
booleans 1/0, missing values empty.

## Decisions

- **Numbers never reused (PLAN 13.5):** `next_episode` in task.yaml is advanced (temp file + rename, fsynced) before
  the folder is made, and never below an existing folder. Names `episode_%03d`, more digits beyond 999.
- **Crash safety (PLAN 6.1):** COMPLETE is written last; `clean_incomplete` deletes every `episode_*` folder without
  it, only inside valid task folders. Every row is flushed when written (that is more often than the "once a second"
  required).
- **Deletion is narrow (PLAN 13.4):** `_remove_episode_dir` deletes only a real (not symlinked) folder named
  `episode_*` directly inside its own task folder, after resolving symlinks. Anything else is a `TaskError`.
  `TaskStore.delete_task` (the page's delete task) deletes a whole task the same narrow way: a valid task name, a real
  (not symlinked) folder with a valid task.yaml directly inside `data_root`; it returns how many episodes it held.
- **The storage folder can change at runtime:** teleop's `EpisodeFlow.set_root` makes a new `TaskStore` for the new
  folder; frames records into whatever episode folder teleop passes, so nothing else changes.
- **The listener never gets an exception:** `StateRecorder.write` is locked (main may close at the same time), ignores
  writes after close and keeps a write error in `EpisodeRecording.error` for the main loop (RECORDING_FAILED).
  The main-thread recorders raise `OSError` as usual.
- **`EpisodeRecording(episode, waypoints=2)`:** commands.csv needs K for its columns, so the constructor takes K
  (STRUCTURE 8.2 lists only `episode`). A set with more waypoints than slots is a `ValueError`, never silently cut.
- **`Task.episodes()` is cheap** (E3 may call it from the 20 ms loop): each episode's `EpisodeInfo` is cached in the
  `Task` by folder name, keyed on the stat (mtime, size, inode) of COMPLETE and meta.yaml, so each meta.yaml is
  parsed once; a repeat call only lists the folder and stats two files per episode (about 4 ms for 200 episodes).
  A changed meta.yaml is re-read and a deleted folder dropped from the cache. YAML is read with libyaml's
  `CSafeLoader` (falling back to `SafeLoader`), for task.yaml too.
- **meta.yaml (PLAN 6.3, 13.12):** task name and one-line description only; the full hello (with `F_T_TCP` as a 4×4
  list), the frames summary, the rig (`rig_file` path plus the parsed file as `rig`; `rig: null` and a one-line `rig_note`
  if it cannot be read, never failing the save), counts (frames' own plus `EpisodeRecording.counts()`), the teleop settings with the lead
  and the worst step, both clock fits, and `warnings` (2026-11-20): the data problems seen while recording, one
  `{code, what, t_s}` each (exactly those keys, else ValueError), from EpisodeFlow; `meta_version` stays 1.
  `Task.episodes` reads `outcome`, `duration_s`, `counts.frames_main`.

## Tests

`tests/episode/`: names, task.yaml, numbering with gaps and beyond 999, finish order, delete rules, discard refusing
symlinks and paths outside its task, clean_incomplete, delete_task (refusing bad names, non-tasks, symlinks),
probe_writable; the episodes() cache (a repeat call for 200 episodes
under 20 ms, new / deleted / changed episodes seen); every recorder's header and exact values, padding, replies,
counts, flushing, close idempotent, files never overwritten, a listener thread writing while closing, a state write
error kept; build_meta's content, validation and round trip through `finish` / `episodes`.

## How to extend

- **A new recorded file:** a class in `recorders.py` with `write`, `counts`, `close` (use `_CsvFile` for CSV), opened
  in `EpisodeRecording.__init__` (closing the others on failure) and in `close()`; add its name to `FILES`.
- **A new column:** append it to the column tuple and the row in the same place; old readers go by header name.
- **A new meta field:** add it in `build_meta` as plain types; bump `META_VERSION` if a field changes meaning.
