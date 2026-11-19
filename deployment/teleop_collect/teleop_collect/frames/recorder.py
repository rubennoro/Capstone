"""FrameRecorder and RecordItem: an episode's camera files, written on the "writer" thread (STRUCTURE.md 5.1, 5.2).

New. While an episode records, the receiver hands every accepted packet to enqueue() as a RecordItem; the writer
thread takes them from a bounded queue and writes, into the episode folder (PLAN 6.1, 6.2):

    main.mkv, ultrawide.mkv     every main JPEG, every *new* ultrawide JPEG, exactly as received (MkvWriter)
    main_frames.csv             one row per main frame         (row i = frame i of main.mkv)
    ultrawide_frames.csv        one row per new ultrawide frame (row i = frame i of ultrawide.mkv)
    packets.csv                 one row per recorded packet, with its raw one-way clock sample
    clock_pi.csv                the Pi -> laptop fit, one row every clock_log_every_s (and one at the end)

enqueue() never blocks. A full queue or a write error is the recording's first `error`: from then on nothing more is
accepted or written (the episode is aborted by teleop, PLAN 7), and stop() still drains, closes and joins. The CSV files
are flushed at least every 0.5 s; the .mkv files flush their clusters about every 0.5 s (mkv_writer.py). stop()
never waits forever (STOP_PUT_TIMEOUT_S, STOP_JOIN_TIMEOUT_S): a dead or stuck writer is the recording's error, so the
main thread keeps answering R1/U2 and sees SIGTERM.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Mapping

import av

from teleop_collect.contracts import CAMERAS, CameraSummary, ClockMap, EncodedFrame, RecordingSummary

from teleop_collect.frames.errors import RecordingRefused
from teleop_collect.frames.mkv_writer import MkvWriter

log = logging.getLogger("frames.recorder")

#: The files a recording writes; a folder holding any of them is refused (R1 start_recording).
VIDEO_FILES = {cam: f"{cam}.mkv" for cam in CAMERAS}
FRAMES_FILES = {cam: f"{cam}_frames.csv" for cam in CAMERAS}
PACKETS_FILE = "packets.csv"
CLOCK_FILE = "clock_pi.csv"
RECORDING_FILES = (*VIDEO_FILES.values(), *FRAMES_FILES.values(), PACKETS_FILE, CLOCK_FILE)

#: Column order: PLAN.md 6.2 (packets.csv, <camera>_frames.csv) and STRUCTURE.md 5.2 (clock_pi.csv).
PACKET_COLUMNS = ("packet_no", "t_send_pi_ns", "arrival_ns", "main_frame_no", "ultrawide_frame_no", "uw_reused",
                  "pair_used", "pair_gap_ms")
FRAME_COLUMNS = ("video_index", "frame_no", "t_sensor_pi_ns", "exposure_us", "t_mid_pi_ns", "t_mid_ns", "packet_no",
                 "arrival_ns", "bytes")
CLOCK_COLUMNS = ("t_ns", "offset_ns", "drift_ppm", "samples", "used")

#: The CSV files are flushed at least this often (STRUCTURE.md 5.2: at least once per second).
FLUSH_EVERY_NS = 500_000_000
#: stop(): how long to wait for room for the stop mark, then for the writer to finish (R1 stop_recording allows 2 s,
#: the launcher's SIGTERM 5 s). A writer that misses either is the recording's error; stop() returns anyway.
STOP_PUT_TIMEOUT_S = 1.0
STOP_JOIN_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class RecordItem:
    """One accepted packet, as the receiver hands it to the preview and the recorder (STRUCTURE.md 5.1).

    Times: `_pi_ns` Pi clock, `_ns` laptop CLOCK_MONOTONIC. The t_mid values are mid-exposures by the rig file's time
    model; the mapped ones use the Pi clock map at the packet's arrival.
    """

    packet_no: int
    t_send_pi_ns: int                     # the Pi's send stamp (one-way clock sample, with arrival_ns)
    arrival_ns: int
    main: EncodedFrame
    main_t_mid_pi_ns: int
    main_t_mid_ns: int
    ultrawide: EncodedFrame | None        # only when its frame_no is new (a reused one is None)
    ultrawide_t_mid_pi_ns: int | None
    ultrawide_t_mid_ns: int | None
    ultrawide_frame_no: int               # the packet's pair (the frame it names, new or reused); -1: none
    uw_reused: bool                       # no new ultrawide frame: the pair is an earlier one
    pair_used: bool                       # the pair rule accepted the pair
    pair_gap_ms: float | None             # main mid - pair mid (Pi clock), when the pair was found


def _b(v: bool) -> str:
    return "1" if v else "0"


def _opt(v: int | None) -> str:
    return "" if v is None else str(int(v))


class _CameraCounts:
    def __init__(self) -> None:
        self.frames = 0
        self.bytes = 0
        self.frame_no_gaps = 0
        self.last_frame_no: int | None = None
        self.first_t_mid_ns: int | None = None
        self.last_t_mid_ns: int | None = None

    def add(self, frame_no: int, size: int, t_mid_ns: int) -> None:
        if self.last_frame_no is not None and frame_no > self.last_frame_no + 1:
            self.frame_no_gaps += frame_no - self.last_frame_no - 1
        self.last_frame_no = frame_no
        self.frames += 1
        self.bytes += size
        if self.first_t_mid_ns is None:
            self.first_t_mid_ns = t_mid_ns
        self.last_t_mid_ns = t_mid_ns

    def summary(self) -> CameraSummary:
        return CameraSummary(frames=self.frames, frame_no_gaps=self.frame_no_gaps, bytes=self.bytes,
                             first_t_mid_ns=self.first_t_mid_ns, last_t_mid_ns=self.last_t_mid_ns)


_STOP = object()


class _Session:
    """One recording: its files (owned by the writer thread once it runs), queue, counters and first error."""

    def __init__(self, folder: Path, queue_max: int) -> None:
        self.folder = folder
        self.queue: queue.Queue = queue.Queue(maxsize=queue_max)
        self.queue_max = queue_max
        self.lock = threading.Lock()
        self.accepting = True
        self.error = ""
        self.videos: dict[str, MkvWriter] = {}
        self.csv: dict[str, IO[str]] = {}               # file name -> open text file
        self.counts = {cam: _CameraCounts() for cam in CAMERAS}
        self.packets = 0
        self.packet_no_gaps = 0
        self.last_packet_no: int | None = None
        self.uw_reused = 0
        self.pair_refused = 0
        self.thread: threading.Thread | None = None

    def fail(self, error: str) -> None:
        """The first failure is kept; nothing more is accepted."""
        with self.lock:
            self.accepting = False
            first = not self.error
            if first:
                self.error = error
        if first:
            log.error("recording %s failed: %s", self.folder, error)


class FrameRecorder:
    """Writes the camera files of one recording at a time (start .. stop). Thread-safe: start/stop from the main
    thread, enqueue from the receiver thread."""

    def __init__(self, queue_max: int, clock_log_every_s: float, sizes: Mapping[str, tuple[int, int]], fps: int,
                 clock_map: ClockMap, clock: Callable[[], int] = time.monotonic_ns) -> None:
        if queue_max < 1:
            raise ValueError(f"queue_max must be >= 1, got {queue_max}")
        self._queue_max = int(queue_max)
        self._clock_every_ns = max(1, int(round(clock_log_every_s * 1e9)))
        self._sizes = {cam: (int(sizes[cam][0]), int(sizes[cam][1])) for cam in CAMERAS}
        self._fps = int(fps)
        self._clock_map = clock_map
        self._clock = clock
        self._lock = threading.Lock()
        self._session: _Session | None = None

    # ------------------------------------------------------------------------------------------------ state

    @property
    def recording(self) -> bool:
        return self._session is not None

    @property
    def error(self) -> str:
        """"" or the first failure of the current recording."""
        s = self._session
        return s.error if s is not None else ""

    def frames_written(self, camera: str) -> int:
        """Frames of that camera written in the current recording (0 when not recording)."""
        s = self._session
        return s.counts[camera].frames if s is not None else 0

    def queue_size(self) -> int:
        s = self._session
        return s.queue.qsize() if s is not None else 0

    @property
    def folder(self) -> Path | None:
        s = self._session
        return s.folder if s is not None else None

    # ------------------------------------------------------------------------------------------------ start

    def start(self, episode_dir: Path) -> None:
        """Open every file in episode_dir and start the writer. RecordingRefused (the R1 refusal text) if a
        recording is running, the folder is missing or already has recording files, or a file cannot be created."""
        folder = Path(episode_dir)
        with self._lock:
            if self._session is not None:
                raise RecordingRefused("already recording")
            if not folder.is_dir():
                raise RecordingRefused(f"{folder}: not a folder")
            if any((folder / name).exists() for name in RECORDING_FILES):
                raise RecordingRefused(f"{folder}: already has recording files")
            s = _Session(folder, self._queue_max)
            try:
                self._open(s)
            except (OSError, av.error.FFmpegError, ValueError) as e:
                self._close_files(s)
                for name in RECORDING_FILES:                    # leave the folder as it was, so a retry works
                    try:
                        (folder / name).unlink(missing_ok=True)
                    except OSError:
                        pass
                raise RecordingRefused(f"{folder}: cannot create the recording files ({e})") from e
            s.thread = threading.Thread(target=self._writer, args=(s,), name="writer", daemon=True)
            self._session = s
            s.thread.start()
        log.info("recording into %s", folder)

    def _open(self, s: _Session) -> None:
        for cam in CAMERAS:
            w, h = self._sizes[cam]
            s.videos[cam] = MkvWriter(s.folder / VIDEO_FILES[cam], w, h, self._fps)
        for name, columns in ((FRAMES_FILES["main"], FRAME_COLUMNS), (FRAMES_FILES["ultrawide"], FRAME_COLUMNS),
                              (PACKETS_FILE, PACKET_COLUMNS), (CLOCK_FILE, CLOCK_COLUMNS)):
            fh = (s.folder / name).open("x", encoding="utf-8", newline="")
            s.csv[name] = fh
            fh.write(",".join(columns) + "\n")
            fh.flush()

    # ------------------------------------------------------------------------------------------------ enqueue

    def enqueue(self, item: RecordItem) -> None:
        """Queue one packet for the writer; never blocks. Ignored when not recording or after a failure. A full
        queue is the recording's failure."""
        s = self._session
        if s is None:
            return
        with s.lock:
            if not s.accepting:
                return
            try:
                s.queue.put_nowait(item)
                return
            except queue.Full:
                pass
        s.fail(f"recording could not keep up: the write queue is full ({s.queue_max} packets): "
               f"check the disk under {s.folder}")

    # ------------------------------------------------------------------------------------------------ stop

    def stop(self) -> RecordingSummary:
        """Stop accepting, let the writer drain the queue, close every file, join the writer; the summary.
        RuntimeError if not recording."""
        with self._lock:
            s = self._session
            if s is None:
                raise RuntimeError("not recording")
            with s.lock:
                s.accepting = False
            self._finish_writer(s)
            self._session = None
        st = self._clock_map.stats(self._clock())
        summary = RecordingSummary(
            main=s.counts["main"].summary(), ultrawide=s.counts["ultrawide"].summary(), packets=s.packets,
            packet_no_gaps=s.packet_no_gaps, uw_reused=s.uw_reused, pair_refused=s.pair_refused, error=s.error,
            pi_clock_offset_ns=None if st.offset_ns is None else int(st.offset_ns),
            pi_clock_drift_ppm=None if st.drift_ppm is None else float(st.drift_ppm))
        log.info("recording %s stopped: %d packets, main %d, ultrawide %d frames%s", s.folder, s.packets,
                 s.counts["main"].frames, s.counts["ultrawide"].frames, f", error: {s.error}" if s.error else "")
        return summary

    def _finish_writer(self, s: _Session) -> None:
        """Hand the writer its stop mark and join it, never waiting forever: a dead writer's files are closed here;
        a stuck one (full queue, hung disk) is left behind as a daemon thread, with the recording's error set."""
        t = s.thread
        if t is None or not t.is_alive():
            s.fail("internal error: the writer thread had stopped before the end of the recording")
            self._close_files(s)
            return
        try:
            s.queue.put(_STOP, timeout=STOP_PUT_TIMEOUT_S)
        except queue.Full:
            s.fail(f"the writer is stuck: no room in its queue for {STOP_PUT_TIMEOUT_S:g} s: check the disk under "
                   f"{s.folder}")
            return
        t.join(STOP_JOIN_TIMEOUT_S)
        if t.is_alive():
            s.fail(f"the writer did not finish within {STOP_JOIN_TIMEOUT_S:g} s: check the disk under {s.folder}")

    # ------------------------------------------------------------------------------------------------ writer

    def _writer(self, s: _Session) -> None:
        """Thread "writer": owns the files until the end of the recording."""
        next_flush = self._clock() + FLUSH_EVERY_NS
        next_clock = self._clock()
        try:
            while True:
                now = self._clock()
                if now >= next_clock:
                    next_clock = now + self._clock_every_ns
                    self._guard(s, CLOCK_FILE, lambda: self._clock_row(s, now))
                if now >= next_flush:
                    next_flush = now + FLUSH_EVERY_NS
                    self._flush(s)
                wait_ns = max(0, min(next_clock, next_flush) - self._clock())
                try:
                    item = s.queue.get(timeout=max(wait_ns / 1e9, 0.001))
                except queue.Empty:
                    continue
                if item is _STOP:
                    break
                if not s.error:
                    self._write(s, item)
            if not s.error:
                self._guard(s, CLOCK_FILE, lambda: self._clock_row(s, self._clock()))
        except Exception as e:                              # never let the thread die with the files open
            log.exception("writer failed")
            s.fail(f"internal error in the writer: {type(e).__name__}: {e}")
        finally:
            self._close_files(s)

    def _guard(self, s: _Session, name: str, fn: Callable[[], None]) -> bool:
        """Run one write; any failure becomes the recording's error naming the file (and stops accepting). False
        if it failed."""
        if s.error:
            return False
        try:
            fn()
            return True
        except (OSError, av.error.FFmpegError) as e:
            s.fail(f"disk error: {s.folder / name}: {e}")
            return False
        except Exception as e:                              # never kill the writer: the error stops the recording
            log.exception("writing %s failed", s.folder / name)
            s.fail(f"internal error writing {s.folder / name}: {type(e).__name__}: {e}")
            return False

    def _write(self, s: _Session, it: RecordItem) -> None:
        cams = [("main", it.main, it.main_t_mid_pi_ns, it.main_t_mid_ns)]
        if it.ultrawide is not None:
            cams.append(("ultrawide", it.ultrawide, it.ultrawide_t_mid_pi_ns, it.ultrawide_t_mid_ns))
        for cam, frame, mid_pi, mid in cams:
            box: dict[str, int] = {}
            if not self._guard(s, VIDEO_FILES[cam], lambda: box.update(i=s.videos[cam].write(frame.data))):
                return
            info = frame.info
            row = (box["i"], info.frame_seq, info.t_pi_ns, info.exposure_ns // 1000, mid_pi, mid, it.packet_no,
                   it.arrival_ns, len(frame.data))
            if not self._guard(s, FRAMES_FILES[cam],
                               lambda: s.csv[FRAMES_FILES[cam]].write(",".join(map(str, row)) + "\n")):
                return
            s.counts[cam].add(int(info.frame_seq), len(frame.data), int(mid))
        gap = "" if it.pair_gap_ms is None else f"{it.pair_gap_ms:.6f}"
        row = (str(it.packet_no), str(it.t_send_pi_ns), str(it.arrival_ns), str(it.main.info.frame_seq),
               str(it.ultrawide_frame_no), _b(it.uw_reused), _b(it.pair_used), gap)
        if not self._guard(s, PACKETS_FILE, lambda: s.csv[PACKETS_FILE].write(",".join(row) + "\n")):
            return
        if s.last_packet_no is not None and it.packet_no > s.last_packet_no + 1:
            s.packet_no_gaps += it.packet_no - s.last_packet_no - 1
        s.last_packet_no = it.packet_no
        s.packets += 1
        s.uw_reused += int(it.uw_reused)
        s.pair_refused += int(not it.pair_used)

    def _clock_row(self, s: _Session, now: int) -> None:
        st = self._clock_map.stats(now)
        drift = "" if st.drift_ppm is None else f"{st.drift_ppm:.6f}"
        s.csv[CLOCK_FILE].write(f"{now},{_opt(st.offset_ns)},{drift},{st.samples},{st.used}\n")

    def _flush(self, s: _Session) -> None:
        for name, fh in s.csv.items():
            if not self._guard(s, name, fh.flush):
                return

    def _close_files(self, s: _Session) -> None:
        """Close every file (idempotent); a failure is the recording's error."""
        for cam, video in list(s.videos.items()):
            try:
                video.close()
            except Exception as e:
                s.fail(f"disk error: {video.path}: {e}")
        for name, fh in list(s.csv.items()):
            try:
                fh.close()
            except Exception as e:
                s.fail(f"disk error: {s.folder / name}: {e}")
