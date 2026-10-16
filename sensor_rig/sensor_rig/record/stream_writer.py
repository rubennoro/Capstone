"""Per-episode camera stream writer (P1 §3.4): gating, K encode workers, in-order index writer.

    capture pump --submit(frame)--> [gate start/stop, sensor-gap check] --put_nowait--> encode_q (bounded)
    EncodeWorker xK : JPEG encode -> write <t_ns>.jpg -> index_q
    IndexWriter     : min-heap on ep_seq -> rows appended strictly in ep_seq order (data.csv + frames_meta.csv)

Guarantees: a row is only written after its file exists; a failed encode/write removes the file and logs a drop;
``data.csv`` is strictly increasing (a violating frame is dropped as ``write_error``).
"""

from __future__ import annotations

import heapq
import math
import os
import queue
import threading
import time
from typing import Callable

import numpy as np

from sensor_rig.drivers.argus_logic import GapDetector

_STOP = object()


class CameraStreamWriter:
    def __init__(self, cam_name: str, writer, encoder, fps: float, start_ns: int | None = None, n_workers: int = 2,
                 queue_frames: int | None = None, queue_seconds: float = 0.75, flush_period_s: float = 0.5,
                 log: Callable[[str], None] | None = None):
        self.cam = cam_name
        self.writer = writer
        self.index = writer.camera_writer(cam_name)
        self.encoder = encoder
        self.fps = float(fps)
        # None = not started yet (every frame is 'before'); set atomically with begin() so that no frame with
        # t_ns >= start_ns can slip past (a frame submitted before begin() has t_ns <= t_arrival < start_ns).
        self.start_ns: int | None = None if start_ns is None else int(start_ns)
        self.stop_ns: int | None = None
        self.done = False                 # saw a frame with t_ns > stop_ns
        self.queue_frames = int(queue_frames) if queue_frames else max(2, math.ceil(queue_seconds * fps))
        self.flush_period_s = flush_period_s
        self._log = log or (lambda s: None)
        self._lock = threading.Lock()
        self._drop_lock = threading.Lock()
        self._encode_q: queue.Queue = queue.Queue(maxsize=self.queue_frames)
        self._index_q: queue.Queue = queue.Queue()
        self._gaps = GapDetector(fps)
        self._next_ep_seq = 0
        self._last_t_ns: int | None = None
        self.drops = {"sensor_gap": 0, "queue_full": 0, "encode_error": 0, "write_error": 0, "nonmonotonic_ts": 0}
        self.submitted = 0
        self.written = 0
        self.encode_ms: list[float] = []
        self.latency_ns: list[int] = []
        self.frame_times: list[int] = []     # t_ns of frames accepted into the episode (for live fps)
        self.queue_full_times: list[int] = []
        self._workers = [threading.Thread(target=self._encode_loop, name=f"enc-{cam_name}-{i}", daemon=True)
                         for i in range(max(1, int(n_workers)))]
        self._indexer = threading.Thread(target=self._index_loop, name=f"idx-{cam_name}", daemon=True)
        for t in self._workers:
            t.start()
        self._indexer.start()
        self._finished = False

    # ---- called by the capture pump thread ----
    def submit(self, f) -> str:
        """Offer a frame. Returns 'before', 'after', 'queued' or 'dropped'."""
        with self._lock:
            if self._finished:
                return "after"
            if self.start_ns is None or f.t_ns < self.start_ns:
                return "before"
            if self.stop_ns is not None and f.t_ns > self.stop_ns:
                self.done = True
                return "after"
            ep_seq = self._next_ep_seq
            self._next_ep_seq += 1
            if self._last_t_ns is not None and f.t_ns <= self._last_t_ns:
                # Duplicate / backwards stamp (backend bug): never encode it. Files are named <t_ns>.jpg, so a
                # duplicate would overwrite the earlier frame's JPEG.
                self._drop(ep_seq, f.t_ns, "nonmonotonic_ts", 1)
                self._gaps.last_seq = int(f.seq)   # the frame arrived: not a sensor gap
                self._index_q.put((ep_seq, None))
                return "dropped"
            self._last_t_ns = int(f.t_ns)
            missing = self._gaps.update(f.seq, f.t_ns)
            if missing:
                self._drop(ep_seq, f.t_ns, "sensor_gap", missing)
            self.latency_ns.append(int(f.t_arrival_ns) - int(f.t_ns))
            try:
                self._encode_q.put_nowait((ep_seq, f))
            except queue.Full:
                self.queue_full_times.append(int(f.t_ns))
                self._drop(ep_seq, f.t_ns, "queue_full", 1)
                self._index_q.put((ep_seq, None))
                return "dropped"
            self.submitted += 1
            self.frame_times.append(int(f.t_ns))
            return "queued"

    def _drop(self, ep_seq: int, t_ns: int, reason: str, count: int) -> None:
        with self._drop_lock:
            self.drops[reason] += count
        try:
            self.writer.add_drop(self.cam, ep_seq, t_ns, reason, count)
        except ValueError:
            # raw_episode.DROP_REASONS (WP-A) may not list this reason yet: log it as write_error in drops.csv;
            # meta.json sensors.<cam>.drops still has the precise count.
            self.writer.add_drop(self.cam, ep_seq, t_ns, "write_error", count)

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def begin_locked(self, start_ns: int) -> None:
        """Set the start time; the caller holds ``self.lock`` (so several streams can start atomically)."""
        self.start_ns = int(start_ns)

    def request_stop(self, stop_ns: int) -> None:
        with self._lock:
            self.stop_ns = int(stop_ns)

    # ---- workers ----
    def _encode_loop(self) -> None:
        while True:
            item = self._encode_q.get()
            if item is _STOP:
                return
            ep_seq, f = item
            path = self.writer.frame_path(self.cam, f.t_ns)
            t0 = time.perf_counter()
            try:
                data = self.encoder.encode(f.image)
            except Exception as e:
                self._log(f"[{self.cam}] encode error at t={f.t_ns}: {e!r}")
                self._drop(ep_seq, f.t_ns, "encode_error", 1)
                self._index_q.put((ep_seq, None))
                continue
            enc_ms = (time.perf_counter() - t0) * 1e3
            try:
                with open(path, "wb") as fh:
                    fh.write(data)
            except Exception as e:
                self._log(f"[{self.cam}] write error at t={f.t_ns}: {e!r}")
                try:
                    os.remove(path)
                except OSError:
                    pass
                self._drop(ep_seq, f.t_ns, "write_error", 1)
                self._index_q.put((ep_seq, None))
                continue
            self.encode_ms.append(enc_ms)
            meta = {"t_arrival_ns": int(f.t_arrival_ns), "sensor_seq": int(f.seq),
                    "exposure_ns": f.meta.get("exposure_ns"), "gain": f.meta.get("gain"),
                    "encode_ms": round(enc_ms, 3), "bytes": len(data)}
            self._index_q.put((ep_seq, (int(f.t_ns), path.name, meta, path)))

    def _index_loop(self) -> None:
        heap: list = []
        expected = 0
        last_flush = time.monotonic()
        stop = False
        while True:
            try:
                item = self._index_q.get(timeout=0.1)
                if item is _STOP:
                    stop = True
                else:
                    heapq.heappush(heap, (item[0], id(item), item[1]))
            except queue.Empty:
                pass
            while heap and heap[0][0] == expected:
                ep_seq, _, payload = heapq.heappop(heap)
                expected += 1
                if payload is None:
                    continue
                t_ns, fname, meta, path = payload
                try:
                    self.index.add(ep_seq, t_ns, fname, meta)
                    self.written += 1
                except ValueError as e:   # non-increasing stamp from the backend
                    self._log(f"[{self.cam}] index error: {e}")
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    self._drop(ep_seq, t_ns, "write_error", 1)
            if time.monotonic() - last_flush > self.flush_period_s:
                self.index.flush()
                last_flush = time.monotonic()
            if stop and not heap:
                self.index.flush()
                return
            if stop and heap and heap[0][0] != expected:
                # cannot happen (every ep_seq is posted exactly once); avoid hanging forever
                self._log(f"[{self.cam}] index writer: missing ep_seq {expected}, flushing remaining")
                expected = heap[0][0]

    def finish(self) -> dict:
        """Drain workers and the index writer; returns stream statistics."""
        with self._lock:
            self._finished = True
        for _ in self._workers:
            self._encode_q.put(_STOP)
        for t in self._workers:
            t.join()
        self._index_q.put(_STOP)
        self._indexer.join()
        return self.stats()

    def stats(self) -> dict:
        enc = np.asarray(self.encode_ms) if self.encode_ms else None
        lat = np.asarray(self.latency_ns) if self.latency_ns else None
        ft = self.frame_times
        fps = (len(ft) - 1) / ((ft[-1] - ft[0]) / 1e9) if len(ft) > 1 and ft[-1] > ft[0] else None
        return {
            "fps_measured": fps,
            "drops": dict(self.drops),
            "latency_ms_median": None if lat is None else float(np.median(lat)) / 1e6,
            "encode_ms_p50": None if enc is None else float(np.percentile(enc, 50)),
            "encode_ms_p99": None if enc is None else float(np.percentile(enc, 99)),
        }
