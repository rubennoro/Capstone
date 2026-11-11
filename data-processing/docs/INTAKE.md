# Intake (`dataproc/intake.py`): decisions

Intake files converted recordings from `incoming/` into a task folder. **It is no longer a CLI command:**
`dataproc calibrate` and `dataproc run` call it through `dataproc/pi_intake.py` (`run_intake(root, task, new=...,
calibration=..., staging=...)`). DESIGN.md's "Intake" section holds the rules; this file adds the decisions made
when implementing them (tests in `tests/test_intake.py` and `tests/run/test_run_intake_staging.py`). The CLI-style
names below (`--task NAME`, `--new`, `--calibration`) are the `run_intake` arguments `task`, `new` and
`calibration`.

How the two commands use it:
- **`calibrate`**: `new=True, calibration=True`. The folder is the first free one of `TASK`, `TASK_2`, `TASK_3`, …
  (`paths.resolve_intake_target`; any existing entry counts as taken), and the recording goes to
  `calibration/<pi_name>/raw/`.
- **`run`**: the **staging target**, `staging=True`. The folder must already exist with its `task.yaml` (it is never
  created; `new` and `calibration` are refused), and the recording goes to `tasks/<folder>/.staging/<pi_name>/raw/`
  instead of `episode_N`. `run` later moves it on with `move_slot` (to `episode_N`, numbered from `episode_1`, when
  it passed every stage; to `failed/<pi_name>/` otherwise), which appends the new slot's ledger row.
  `episodes.csv` stays append-only: the newest row of a recording is where it is now. `reconcile` knows all four
  slot kinds (`episode_N`, `calibration/<x>`, `.staging/<x>`, `failed/<x>`).
- Without `staging`, intake still numbers `episode_N` from `episode_0` (max existing + 1); only `run`'s promotion
  starts at `episode_1`.

## One run, step by step

Each step names the method of `Intake` (in `dataproc/intake.py`) that does it, in the order `Intake.run` calls
them. See "Code map" below.

1. **Root missing** (`run`): print "nothing to intake", exit 0. Nothing is created.
2. **Lock** (`lock`): `flock(LOCK_EX | LOCK_NB)` on `<root>/.intake.lock`. If another process holds it, exit 1 with
   "another intake is running". The lock file is the only thing a run creates when there is nothing to do.
3. **Root pre-flight, then reconcile** (`preflight_root`, `reconcile`). First, `incoming/` (if it exists), `tasks/` and `quarantine/` (or, for
   the last two, the nearest existing parent they would be created in) must be writable, else exit 1 with
   "permission denied". The check uses `os.access(..., effective_ids=True)` (effective uid, as Linux supports), and
   the message names the uid that was checked. A `PermissionError` during reconcile is also reported cleanly
   (exit 1, no traceback). Then **reconcile** every task folder under `tasks/`, not only the target. Any `episode_N/raw` or `calibration/<x>/raw`
   without an `episodes.csv` row gets a row rebuilt from its `meta.json` and `MANIFEST.sha256`, with
   `intake_unix_ns` = now. Each rebuilt row prints a line starting `reconciled:`.
   - The Pi dir name is lost once the episode is renamed to `raw`. So right before every move, intake atomically
     writes `tasks/<folder>/.intake_pending.json` (`{"slot", "pi_name", "kind"}`). Reconcile takes `pi_name` from it
     when its slot matches. Otherwise it uses `<x>` for calibration, and leaves it empty (with a warning) for
     `episode_N`.
   - The file is overwritten on each move and never deleted.
   - An empty `episode_N/` (a crash between `mkdir` and the rename) is reported. Its number stays taken.
   - All folders are reconciled, so dedupe sees a complete ledger even after a crash in a `--new` run (whose folder
     is never the target again).
4. **Incoming missing** (`run`): print a message, exit 0.
5. **Scan `incoming/`** (`scan_incoming`, sorted by name). Entries are handled as follows:
   - name starts with `.`: ignored, never listed;
   - symlink (to anything, even dangling): always a candidate, so it is quarantined (`episode_is_symlink`). The
     link itself is moved, and its target is never touched or written to;
   - not a directory: left, shown as "skipped: not a directory";
   - directory with any dot-entry other than `.verified`: left, shown as "skipped: transfer in progress";
   - directory without `.verified`: left, shown as "skipped: not verified yet";
   - directory this process cannot write to (`os.access(W_OK | X_OK)`; moving a directory to another parent
     rewrites its `..`, so it needs write permission on itself): left, shown as "skipped: permission denied (owner
     uid …, mode …)". Nothing is created for it;
   - otherwise: a candidate.
6. **No candidates** (`run`): print "nothing waiting", exit 0. No task folder and no quarantine folder are created.
7. **Target folder** (`open_target`, with `check_same_filesystem` and `preflight_target`): `paths.resolve_intake_target(tasks, NAME, --new)`. Exit 1 for:
   - the target is not a directory;
   - its `task.yaml` is invalid;
   - it has episodes but no `task.yaml`;
   - `incoming/` and `tasks/` (or `quarantine/`) are on different filesystems (`st_dev` pre-check);
   - **target pre-flight**: the existing target folder and, with `--calibration`, its `calibration/` (or the
     target folder if `calibration/` does not exist yet) must be writable. Together with step 3, this runs before
     anything is created, so a read-only setup never leaves a task folder, `task.yaml` or an empty `episode_N/`
     behind.

   An existing folder with no `task.yaml` and no episodes (a crash between `mkdir` and writing `task.yaml`) is
   accepted, and `task.yaml` is written on its first ingest.
8. **Order** (`in_order`): candidates whose `meta.json` cannot be parsed come first, by name. Then the rest by (`start_unix_ns`,
   `session_id`, `episode_number`, name).
9. **Per candidate, all checks run, and every failing reason is recorded** (`handle`: `validate`, `transfer_restarted`, `check_duplicate`, `check_rig`). Right after validation, the
   dot-entry check runs again for every candidate. If a dot-entry other than `.verified` has appeared (a re-send
   started during validation), the episode is left in `incoming/` as "skipped: transfer in progress", whatever the
   validation found. It is neither quarantined nor ingested.
   - `RawEpisode.validate(decode_videos=True, check_manifest=True)`. Each distinct error code is a reason, in report
     order. Warnings are not reasons.
   - `duplicate`: the Pi dir name or the (`session_id`, `episode_number`) pair is in the `episodes.csv` of any task
     folder, or was accepted earlier in this run. A candidate with invalid meta is checked by name only.
   - `rig_mismatch`: `meta.rig_id` ≠ the target's `task.yaml` `rig`. In a folder created in this run, the reference
     is the rig of the first ingested episode, i.e. the oldest one that passes every check. The check is skipped
     while there is no reference yet, or when meta is invalid.

   Reasons are ordered: validator codes, then `duplicate`, then `rig_mismatch`.
10. **Quarantine** (`quarantine`, via `Quarantine.put`; any reason):
    - create `quarantine/<name>/` (`<name>__2`, `__3`, … if any entry with that name exists);
    - `os.rename` the episode into it as **`raw/`**. If the rename fails, the just-created directory is removed
      with `rmdir` (which only succeeds while it is empty), so no reason-only folder is left behind;
    - write `reason.json` (atomically). If intake crashes between the rename and this write, the next run's
      reconcile writes a `reason.json` with `reasons: ["interrupted"]` and `report: null`.

    Using a wrapper directory, rather than putting `reason.json` inside the episode, keeps the raw episode
    byte-identical (no `extra_file` if it is re-validated). It also means a symlinked episode's target is never
    written to. To retry, move `quarantine/<name>/raw` back to `incoming/<name>`.

    `reason.json` contains:

    ```json
    {"pi_name": "...", "reasons": ["hash_mismatch", "duplicate"],
     "details": {"duplicate": {"pi_name": {"value": "...", "taken_by": "block_raise/episode_3"},
                               "session_episode": {"value": ["s01", "4"], "taken_by": "..."}},
                 "rig_mismatch": {"episode_rig": "rig_v2", "task_rig": "rig_v1"}},
     "report": {"path": ..., "ok": false, "problems": [...]},
     "quarantined_at": "2026-11-04T12:00:00+00:00", "quarantined_unix_ns": 0,
     "dataproc_version": "0.1.0", "target_task": "block_raise", "mode": "demonstration", "episode": "raw/"}
    ```

11. **Ingest** (`ingest`; no reasons):
    - Create the task folder and `task.yaml` lazily, on the first ingest, so a run that only quarantines creates no
      folder. `task.yaml` gets: `label` = `--task NAME` as given (so `--new` folders `block_raise_2` are labelled
      `block_raise`), `rig` = `meta.rig_id`, `created` = ISO UTC, `notes` "". It is written atomically (dot temp
      file, fsync, rename).
    - Write `.intake_pending.json`, `mkdir episode_N`, then `os.rename(incoming/<name>, episode_N/raw)`. `EXDEV` is
      an error (exit 1); nothing is ever copied. If the rename fails, the empty `episode_N/` (or
      `calibration/<name>/`) is removed again with `rmdir`, so the number is not burned.
    - **Rollback of a failed first ingest.** If the folder still holds no episode afterwards and this run wrote its
      `task.yaml`, intake also removes what this run created: `task.yaml`, a `.intake_pending.json` it created, an
      empty `calibration/` it created, the task folder itself (only if this run made it and it is now empty,
      including a `--new` `<task>_k` folder), and `tasks/` (same condition). So a failed first move leaves no task
      folder behind, and the next run picks the same folder name. Nothing that existed before the run is touched,
      and no episode data is ever removed: if the rename did happen, for example when interrupted right after it,
      `episode_N/raw` is kept and the next run reconciles its row.
    - Append one row to `episodes.csv`: `O_APPEND`, one `write` of the full line (with the header first if the file
      is new or empty), then `fsync`. Values are written with Python's `csv` module, so a `session_id` containing a
      comma is quoted. `manifest_sha256` is computed before the move.
12. **`--calibration`** (`handle` / `ingest`, flag `calibration_taken`):
    - It walks the same order and quarantines failures as usual. The first candidate that passes every check
      (validate, duplicate, rig) moves to `calibration/<name>/raw/` (`__2`, … on a name collision), with ledger row
      `kind=calibration`, `slot=calibration/<name>`.
    - Every later candidate is left in `incoming/` without being validated, shown as "waiting".
    - It creates the task folder and `task.yaml` if they are missing.
13. **Exit code and summary** (`stop_on_error`, `format_summary`, `cli_intake`):
    - Exit 0 when the run completes, even if some episodes were quarantined.
    - The summary prints one line per `incoming/` entry, `<pi name>  ->  episode_N | calibration/<x> |
      quarantined: r1, r2 | skipped: … | waiting: …`, then the totals line `K ingested into tasks/<folder>
      [(created)], Q quarantined, S left in incoming`.
    - Exit 1 for: lock held, target folder problems (including permissions), EXDEV, a malformed `episodes.csv`
      (header or field count, in any task folder), and any unexpected exception (traceback on stderr).
    - On an error while handling a candidate, that candidate gets an `error` line (`<name>  ->  error: <message>
      (left in incoming)` or `(already moved out of incoming; the next intake reconciles its ledger row)`). The
      summary of what was done is printed, with its totals line starting `before the error:` and ending
      `, N failed`. A partial run never prints the "nothing waiting" line. Then `stopped at <name>; nothing after it
      was processed` goes to stderr. Everything already moved stays moved and has its ledger row (or gets it from
      the next run's reconcile).

## Code map

`run_intake(...)` is `Intake(root, task, new, calibration, log).run()`:

```python
with self.lock():                        # step 2
    self.preflight_root()                # step 3
    self.reconcile()                     # step 3: reconcile_task_folder per folder + Quarantine.repair_missing_reasons
    candidates = self.scan_incoming()    # steps 4-6
    self.open_target()                   # step 7: TaskTarget, check_same_filesystem, preflight_target, DuplicateIndex
    with self.stop_on_error():           # step 13: error outcome + partial result on the exception
        for candidate in self.in_order(candidates):   # step 8
            self.handle(candidate)       # steps 9-12
```

`handle(candidate)`: `calibration_taken` -> "waiting"; `validate` -> `transfer_restarted` -> `check_duplicate` ->
`check_rig` -> `quarantine` (any reason) or `ingest`.

| Piece | Role |
| --- | --- |
| `Intake` | one run; `run` plus one method per step (above) |
| `Ledger` | one `episodes.csv`: `rows`, `slots`, `make_row`, `append` (fsync'd `O_APPEND`) |
| `Quarantine` | `put` (`quarantine/<name>[__k]/raw` + `reason.json`), `repair_missing_reasons` |
| `TaskTarget` | the target folder: `ensure` (folder + `task.yaml`), `next_slot`, `write_pending`, `note_made`, `rollback_if_unused` |
| `DuplicateIndex` | Pi dir names and (session, episode) pairs in every ledger plus this run: `add`, `check` |
| `Candidate` | one `incoming/` entry to validate (name, path, `RawEpisode`, meta) |
| `dataproc/fsutil.py` | generic plumbing: `write_atomic`, `fsync_dir`, `rename_no_copy` (EXDEV -> `CrossDeviceError`), `rmdir_if_empty`, `sha256_file`, `free_name`, `nearest_existing`, `not_writable`, `FileLock` |

## What other commands must tolerate in a task folder

- **Empty `episode_N/` directories.** A crash (power loss, kill -9) between `mkdir episode_N` and the rename
  leaves one. Its number stays taken (it is never refilled), and intake only prints a note about it. `process` and
  `status` must skip any `episode_N/` that has no `raw/`, without treating it as an error.
- **Dot-files.** `.intake_pending.json`, and possibly a leftover `.task.yaml.tmp-<pid>` or `.<name>.tmp-<pid>` from
  an interrupted atomic write, are normal. `process` and `status` must ignore every dot-prefixed entry in a task
  folder.

## Residual risk: a re-send during the move

The dot-entry re-check runs after validation, right before the move. The Pi could still start, or be in the middle
of, an rsync re-send into `incoming/<name>` between that check and the `os.rename`. A running rsync receiver works
relative to the destination directory it entered. When intake renames that directory, rsync follows it, so it
**can modify `episode_N/raw`** (or `quarantine/<name>/raw`). It writes its dot temp file there and renames it over
a data file, which can leave a stray dot-file or changed bytes in an ingested episode. `process` re-validates the
manifest (`check` stage), so changed bytes are caught there as `hash_mismatch`. A re-send started after the move
creates a fresh `incoming/<name>/`, which the next intake quarantines as `duplicate`.

This needs a Pi-side contract violation: the contract forbids modifying an episode after `COMPLETE` and re-sending
one that has `.synced`. The window between the re-check and the rename is very short. The risk is accepted and not
guarded further.


## Python API

```python
from dataproc.intake import run_intake, format_summary, read_ledger, cli_intake
res = run_intake(root, "block_raise", new=False, calibration=False, log=print)   # IntakeResult
res.outcomes          # [EpisodeOutcome(pi_name, status, slot, reasons, quarantine_dir, note)]
res.folder, res.task_dir, res.created_folder, res.reconciled, res.nothing_waiting
read_ledger(task_dir / "episodes.csv")   # [dict] with LEDGER_COLUMNS keys
```

`status` is one of `ingested`, `quarantined`, `skipped`, `waiting`. `IntakeError` (and its subclass
`LockHeldError`) are the exit-1 conditions. `cli_intake(root, name, new, calibration)` returns the exit code.
