"""
EdgeVid LowBand — Multi-Camera Pipeline + FastAPI Server
Connects every component. Runs the full pipeline.
"Every second your camera records, ours decides."
"""

import os
import cv2
import time
import json
import asyncio
import base64
import threading
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from snn_gate import SNNSpikeGate
from detector import PersonDetector
from scorer import FrameScorer
from anomaly_detector import AnomalyDetector
from compressor import DualCompressor
from pre_buffer import PreEventBuffer
from database import ForensicDatabase

# ============================================================
# SHARED COMPONENTS (thread-safe or use locks)
# ============================================================
detector = PersonDetector(confidence=0.3)
detector_lock = threading.Lock()
database = ForensicDatabase()
compressor = DualCompressor(storage_dir="storage")

# Ensure directories
os.makedirs("storage/events", exist_ok=True)
os.makedirs("storage/idle", exist_ok=True)
os.makedirs("storage/clips", exist_ok=True)
os.makedirs("storage/prebuffer", exist_ok=True)

# WebSocket clients
ws_clients = set()


# ============================================================
# THREADED CAMERA READER — Prevents cap.read() from blocking
# ============================================================
class CameraReader:
    """Read camera frames in a dedicated thread so cap.read() never blocks the pipeline."""
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"⚠️ DSHOW failed for source {source}, trying default backend...")
            self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
            if not ret:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            if self._frame is not None:
                return self._ret, self._frame.copy()
            return False, None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self._running = False
        time.sleep(0.1)
        self.cap.release()


# ============================================================
# CAMERA DETECTION — Probe for available cameras
# ============================================================
def _is_bad_camera(cap):
    """Returns True if camera should be excluded:
    - IR/Windows Hello cameras (near-grayscale output)
    - Broken/virtual cameras outputting random noise (high inter-frame diff)
    """
    try:
        # Flush the buffer
        for _ in range(5):
            cap.read()
        ret1, frame1 = cap.read()
        if not ret1 or frame1 is None:
            return True

        # --- IR check: all channels nearly identical (grayscale output) ---
        b, g, r = cv2.split(frame1)
        diff_rg = float(np.mean(np.abs(r.astype(np.int16) - g.astype(np.int16))))
        diff_rb = float(np.mean(np.abs(r.astype(np.int16) - b.astype(np.int16))))
        if diff_rg < 8 and diff_rb < 8:
            print(f"  → IR/grayscale camera detected (diff_rg={diff_rg:.1f}, diff_rb={diff_rb:.1f})")
            return True

        # --- Noise check: compare two consecutive frames 80ms apart ---
        # A broken/virtual camera outputting random static will have very high
        # inter-frame difference (pure noise changes completely every frame).
        # Real cameras, even with motion, stay below ~20 mean-abs-diff.
        import time as _time
        _time.sleep(0.08)
        ret2, frame2 = cap.read()
        if not ret2 or frame2 is None:
            return True
        inter_frame_diff = float(np.mean(np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))))
        if inter_frame_diff > 25:
            print(f"  → Noisy/broken camera detected (inter-frame diff={inter_frame_diff:.1f})")
            return True

        return False
    except Exception:
        return False


def detect_available_cameras(max_check=5):
    """Probe camera indices 0-4 to find all unique connected cameras.
    Filters out Windows virtual duplicates AND IR/Windows Hello cameras."""
    # Step 1: Find all indices that can produce a frame
    candidates = []
    for i in range(max_check):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # Skip IR or noisy/broken cameras
                    if _is_bad_camera(cap):
                        print(f"⚠️ Camera index {i} is IR/noisy/broken — skipped")
                        cap.release()
                        continue
                    candidates.append({
                        'index': i,
                        'cam_id': f'cam_{i}',
                        'name': 'Built-in Camera' if i == 0 else f'External Camera {i}',
                        'resolution': f'{w}x{h}',
                    })
            cap.release()
        except Exception:
            pass

    if len(candidates) <= 1:
        return candidates

    # Step 2: Open all candidates simultaneously to detect duplicates.
    # A Windows virtual duplicate will fail when the real camera is already open.
    caps = {}
    unique = []
    for cam in candidates:
        idx = cam['index']
        try:
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if c.isOpened():
                ret, _ = c.read()
                if ret:
                    caps[idx] = c
                    unique.append(cam)
                else:
                    c.release()
                    print(f"⚠️ Camera index {idx} can't read when others are open — duplicate, skipped")
            else:
                print(f"⚠️ Camera index {idx} can't open simultaneously — duplicate, skipped")
        except Exception:
            pass

    # Release all
    for c in caps.values():
        c.release()

    return unique


# ============================================================
# PER-CAMERA INSTANCE — Each camera has its own AI pipeline
# ============================================================
class CameraInstance:
    """Encapsulates all pipeline state and AI components for one camera."""

    def __init__(self, cam_id, source):
        self.cam_id = cam_id
        self.source = source
        # Per-camera AI components (SNN/scoring/anomaly are stateful per-camera)
        self.spike_gate = SNNSpikeGate(threshold=0.15)
        self.scorer = FrameScorer()
        self.anomaly_detector = AnomalyDetector(loiter_threshold_sec=20, fps=15)
        self.prebuffer = PreEventBuffer(buffer_seconds=30, fps=15)

        # Per-camera pipeline state
        self.state = {
            'running': False,
            'frame_count': 0,
            'current_score': 0,
            'current_category': 'IDLE',
            'current_frame': None,
            'current_detections': [],
            'last_detections': [],       # Persist for crosshair overlay
            'last_det_frame': 0,         # Frame# of last YOLO detection
            'fps': 0,
            'last_alert': None,
            'snn_spike': False,
            'snn_membrane': 0.0,
            'snn_diff': 0.0,
            'session_name': None,
            'session_start': None,
        }

        # Per-camera event clip writer
        self.clip_state = {
            'writer': None,
            'path': None,
            'start_frame': 0,
            'cooldown': 0,
            'frame_shape': None,
            'category': 'IDLE',
            'start_time': None,
        }

        self.thread = None
        self.reader = None

    def start(self, session_name=None):
        """Start this camera's pipeline thread."""
        if self.state['running']:
            return False
        self.state['session_name'] = session_name or f"Demo_{datetime.now().strftime('%H%M%S')}"
        self.state['session_start'] = datetime.now().isoformat()
        self.state['running'] = True
        self.state['current_category'] = 'IDLE'
        self.state['current_score'] = 0
        self.thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop this camera's pipeline."""
        if not self.state['running']:
            return False
        self.state['running'] = False
        self.state['current_frame'] = None
        self.state['current_score'] = 0
        self.state['current_category'] = 'IDLE'
        self.state['current_detections'] = []
        self.state['last_detections'] = []
        self.state['fps'] = 0
        self.state['snn_spike'] = False
        self.state['snn_membrane'] = 0.0
        self.state['snn_diff'] = 0.0
        self.spike_gate.prev_frame = None
        self.spike_gate.membrane_potential = 0.0
        self.spike_gate.lif_neuron.reset()
        return True

    def clear(self):
        """Clear per-camera AI state for fresh demo."""
        self.spike_gate.frame_count = 0
        self.spike_gate.spike_count = 0
        self.spike_gate.spike_history.clear()
        self.spike_gate.diff_history.clear()
        self.spike_gate.prev_frame = None
        self.spike_gate.membrane_potential = 0.0
        self.spike_gate.lif_neuron.reset()
        self.scorer.score_history.clear()
        self.anomaly_detector.tracked_objects.clear()
        self.anomaly_detector.alerts.clear()
        self.anomaly_detector.frame_counter = 0
        self.state['frame_count'] = 0
        self.state['current_score'] = 0
        self.state['current_category'] = 'IDLE'
        self.state['last_alert'] = None
        self.state['snn_spike'] = False
        self.state['snn_membrane'] = 0.0
        self.state['last_detections'] = []
        self.state['last_det_frame'] = 0

    def _start_event_clip(self, frame, frame_number, category='EVENT'):
        h, w = frame.shape[:2]
        self.clip_state['frame_shape'] = (w, h)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"clip_{self.cam_id}_{category}_{frame_number}_{ts}.mp4"
        filepath = os.path.join("storage", "clips", filename)
        # Try H.264 (avc1) first — browser-compatible; fall back to mp4v
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        write_fps = 15.0 if category == 'EVENT' else 12.0 if category == 'NORMAL' else 8.0
        test_writer = cv2.VideoWriter(filepath, fourcc, write_fps, (w, h))
        if not test_writer.isOpened():
            test_writer.release()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            test_writer = cv2.VideoWriter(filepath, fourcc, write_fps, (w, h))
        self.clip_state['writer'] = test_writer
        self.clip_state['path'] = filepath
        self.clip_state['start_frame'] = frame_number
        self.clip_state['cooldown'] = 0
        self.clip_state['category'] = category
        self.clip_state['start_time'] = datetime.now().isoformat()
        return filepath

    def _stop_event_clip(self, frame_number):
        if self.clip_state['writer'] is not None:
            self.clip_state['writer'].release()
            duration = (frame_number - self.clip_state['start_frame']) / 15.0
            path = self.clip_state['path']
            cat = self.clip_state.get('category', 'EVENT')
            start_ts = self.clip_state.get('start_time', datetime.now().isoformat())
            meta = {
                'filename': os.path.basename(path),
                'camera': self.cam_id,
                'category': cat,
                'quality': 'HD' if cat == 'EVENT' else 'MEDIUM' if cat == 'NORMAL' else 'LOW',
                'start_time': start_ts,
                'end_time': datetime.now().isoformat(),
                'duration_sec': round(duration, 1),
                'start_frame': self.clip_state['start_frame'],
                'end_frame': frame_number,
                'fps': 15 if cat == 'EVENT' else 12 if cat == 'NORMAL' else 8,
            }
            meta_path = path.replace('.mp4', '.json')
            try:
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)
            except Exception:
                pass
            self.clip_state['writer'] = None
            self.clip_state['path'] = None
            self.clip_state['start_frame'] = 0
            self.clip_state['cooldown'] = 0
            self.clip_state['category'] = 'IDLE'
            self.clip_state['start_time'] = None
            return path, duration
        return None, 0

    def _draw_crosshair_overlay(self, frame, dets):
        """Draw tactical bounding boxes with crosshairs, corner brackets, and labels."""
        display_frame = frame.copy()

        for det in dets:
            x1, y1, x2, y2 = det['box']
            cls = det.get('class_name', '?')
            conf = det.get('confidence', 0)
            is_person = det.get('is_person', False)

            # Colors: bright green for person, amber for other objects
            if is_person:
                color = (0, 255, 100)
                accent = (0, 220, 80)
            else:
                color = (255, 200, 0)
                accent = (220, 180, 0)

            w_box = x2 - x1
            h_box = y2 - y1

            # Main bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Inner offset box for double-box tactical look
            offset = 3
            if w_box > 50 and h_box > 50:
                cv2.rectangle(display_frame, (x1 + offset, y1 + offset),
                              (x2 - offset, y2 - offset), accent, 1)

            # Center crosshair
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cross_size = max(18, min(w_box, h_box) // 5)

            # Crosshair lines
            cv2.line(display_frame, (cx - cross_size, cy), (cx - 4, cy), color, 1)
            cv2.line(display_frame, (cx + 4, cy), (cx + cross_size, cy), color, 1)
            cv2.line(display_frame, (cx, cy - cross_size), (cx, cy - 4), color, 1)
            cv2.line(display_frame, (cx, cy + 4), (cx, cy + cross_size), color, 1)

            # Crosshair center dot
            cv2.circle(display_frame, (cx, cy), 3, color, -1)

            # Crosshair ring
            cv2.circle(display_frame, (cx, cy), cross_size // 2, color, 1)

            # Corner brackets (tactical look)
            blen = max(12, min(30, w_box // 3, h_box // 3))
            thickness = 2
            for (cx1, cy1), (dx, dy) in [
                ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))
            ]:
                cv2.line(display_frame, (cx1, cy1), (cx1 + blen * dx, cy1), color, thickness)
                cv2.line(display_frame, (cx1, cy1), (cx1, cy1 + blen * dy), color, thickness)

            # Label background + text
            label = f"{cls} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Camera ID watermark
        cv2.putText(display_frame, self.cam_id.upper(), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2, cv2.LINE_AA)

        return display_frame

    def _run_pipeline(self):
        """Main pipeline loop — runs in a separate thread."""
        print(f"📹 [{self.cam_id}] Opening camera {self.source} with threaded reader...")
        self.reader = CameraReader(self.source)

        if not self.reader.isOpened():
            print(f"❌ [{self.cam_id}] Failed to open camera source: {self.source}")
            self.state['running'] = False
            return

        time.sleep(0.3)
        self.state['running'] = True
        frame_number = 0
        fps_counter = 0
        fps_timer = time.time()
        no_frame_count = 0
        last_yolo_detections = []

        print(f"🚀 [{self.cam_id}] Pipeline Started! (source={self.source})")

        while self.state['running']:
          try:
            ret, frame = self.reader.read()
            if not ret or frame is None:
                no_frame_count += 1
                if no_frame_count > 300:
                    print(f"❌ [{self.cam_id}] Camera feed lost. Stopping.")
                    break
                time.sleep(0.01)
                continue
            no_frame_count = 0

            frame_number += 1
            fps_counter += 1
            start_time = time.time()

            # Resize for speed
            h_orig, w_orig = frame.shape[:2]
            if w_orig > 640:
                scale = 640 / w_orig
                frame = cv2.resize(frame, (640, int(h_orig * scale)))

            # Always encode raw frame first so feed never freezes
            _, raw_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            self.state['current_frame'] = base64.b64encode(raw_buf).decode('utf-8')
            self.state['frame_count'] = frame_number

            # Pre-buffer (every 3rd frame)
            if frame_number % 3 == 0:
                self.prebuffer.add_frame(frame, frame_number)

            # ---- Step 1: SNN Spike Gate ----
            spike, diff_score, membrane = self.spike_gate.process_frame(frame)
            self.state['snn_spike'] = bool(spike)
            self.state['snn_membrane'] = float(round(float(membrane), 4))
            self.state['snn_diff'] = float(round(float(diff_score), 4))

            if not spike:
                # Periodic YOLO every 8 frames — always detect persons
                if frame_number % 8 == 0:
                    with detector_lock:
                        periodic_dets = detector.detect(frame)
                    person_dets = [d for d in periodic_dets if d['is_person']]
                    if person_dets:
                        spike = True
                        self.state['snn_spike'] = True
                        last_yolo_detections = periodic_dets
                    else:
                        last_yolo_detections = []

            if not spike:
                # Truly idle
                self.state['current_score'] = 0.0
                self.state['current_category'] = 'IDLE'
                self.state['current_detections'] = []

                if frame_number % 5 == 0:
                    compressor.compress_idle(frame, frame_number)

                if self.clip_state['writer'] is not None:
                    self.clip_state['cooldown'] += 1
                    self.clip_state['writer'].write(frame)
                    if self.clip_state['cooldown'] > 45:
                        path, dur = self._stop_event_clip(frame_number)
                        if path:
                            print(f"🎬 [{self.cam_id}] Clip saved: {path} ({dur:.1f}s)")
            else:
                # SPIKE — full processing
                with detector_lock:
                    detections = last_yolo_detections if last_yolo_detections else detector.detect(frame)
                last_yolo_detections = []

                alerts = self.anomaly_detector.update(frame, detections, frame_number)
                anomaly_flag = self.anomaly_detector.has_active_anomaly()
                score, category = self.scorer.calculate_score(
                    detections, diff_score, frame.shape,
                    anomaly_flag=anomaly_flag
                )

                if category == "EVENT":
                    if frame_number % 3 == 0:
                        frame_path, _, _ = compressor.compress_event(frame, detections, frame_number)
                    else:
                        frame_path = None
                    if frame_number % 10 == 0:
                        database.log_event(
                            frame_number=frame_number, score=score,
                            category=category, detections=detections,
                            event_type="PERSON_DETECTED", severity="HIGH",
                            frame_path=frame_path, compression_type="zstd+roi",
                            camera_id=self.cam_id
                        )
                    if self.clip_state['writer'] is None:
                        self._start_event_clip(frame, frame_number, category='EVENT')
                    elif self.clip_state['category'] != 'EVENT':
                        self.clip_state['category'] = 'EVENT'
                    self.clip_state['cooldown'] = 0
                    self.clip_state['writer'].write(frame)

                elif category == "NORMAL":
                    if frame_number % 5 == 0:
                        compressor.compress_normal(frame, frame_number)
                    if self.clip_state['writer'] is None and len([d for d in detections if d.get('is_person')]) > 0:
                        self._start_event_clip(frame, frame_number, category='NORMAL')
                    if self.clip_state['writer'] is not None:
                        self.clip_state['cooldown'] += 1
                        self.clip_state['writer'].write(frame)
                        if self.clip_state['cooldown'] > 45:
                            path, dur = self._stop_event_clip(frame_number)
                            if path:
                                print(f"🎬 [{self.cam_id}] Clip saved: {path} ({dur:.1f}s)")
                else:
                    if frame_number % 5 == 0:
                        compressor.compress_idle(frame, frame_number)

                for alert in alerts:
                    prebuf_info = self.prebuffer.save_pre_event(alert['type'])
                    event_id = database.log_event(
                        frame_number=frame_number, score=95.0,
                        category="EVENT", detections=detections,
                        event_type=alert['type'], severity="CRITICAL",
                        anomaly_flag=True, duration=alert.get('duration_sec', 0),
                        prebuffer_path=prebuf_info['filepath'] if prebuf_info else None,
                        compression_type="zstd+roi",
                        camera_id=self.cam_id
                    )
                    database.log_alert(event_id, alert['type'], alert['message'])
                    self.state['last_alert'] = alert

                self.state['current_score'] = float(score)
                self.state['current_category'] = str(category)
                self.state['current_detections'] = detections
                # Persist detections for crosshair overlay
                self.state['last_detections'] = detections
                self.state['last_det_frame'] = frame_number

            # ---- Draw crosshair overlay ----
            # Use current detections, or persist last detections for up to 15 frames (~1s)
            dets = self.state.get('current_detections', [])
            if not dets and (frame_number - self.state.get('last_det_frame', 0)) < 15:
                dets = self.state.get('last_detections', [])

            if dets:
                display_frame = self._draw_crosshair_overlay(frame, dets)
                _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
                self.state['current_frame'] = base64.b64encode(buffer).decode('utf-8')
            else:
                # Camera ID watermark even without detections
                display_frame = frame.copy()
                cv2.putText(display_frame, self.cam_id.upper(), (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2, cv2.LINE_AA)
                _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                self.state['current_frame'] = base64.b64encode(buffer).decode('utf-8')

            # FPS calculation
            if time.time() - fps_timer >= 1.0:
                self.state['fps'] = fps_counter
                if fps_counter > 0:
                    print(f"📊 [{self.cam_id}] FPS={fps_counter} | F#{frame_number} | Cat={self.state['current_category']} | Score={self.state['current_score']}")
                fps_counter = 0
                fps_timer = time.time()

            if frame_number % 1296000 == 0:
                self.spike_gate.auto_recalibrate()

            if frame_number % 1000 == 0:
                database.log_system_stats(
                    total_frames=frame_number,
                    processed_frames=self.spike_gate.spike_count,
                    compute_savings=self.spike_gate.get_compute_savings(),
                    storage_savings=compressor.get_savings_percent(),
                    spike_rate=self.spike_gate.get_spike_rate(),
                    avg_score=self.scorer.get_avg_score()
                )

            # Adaptive frame rate control
            cat = self.state['current_category']
            target_fps = 15 if cat == 'EVENT' else 12 if cat == 'NORMAL' else 8
            elapsed = time.time() - start_time
            sleep_time = max(0, (1 / target_fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

          except Exception as e:
            print(f"⚠️ [{self.cam_id}] Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
            continue

        # Cleanup
        if self.clip_state['writer'] is not None:
            self._stop_event_clip(frame_number)
        self.reader.release()
        self.state['running'] = False
        self.state['current_frame'] = None
        self.state['fps'] = 0
        self.state['snn_spike'] = False
        print(f"⏹️ [{self.cam_id}] Pipeline stopped.")

    def get_ws_data(self):
        """Build WebSocket-safe data dict for this camera."""
        cat = str(self.state.get('current_category', 'IDLE'))
        target_fps = 15 if cat == 'EVENT' else 12 if cat == 'NORMAL' else 8

        safe_dets = []
        for d in self.state.get('current_detections', []):
            try:
                box = d.get('box', (0, 0, 0, 0))
                safe_dets.append({
                    'box': [int(b) for b in box],
                    'class': str(d.get('class_name', '?')),
                    'conf': float(d.get('confidence', 0)),
                    'is_person': bool(d.get('is_person', False))
                })
            except Exception:
                pass

        # Include persisted detections if current is empty
        if not safe_dets:
            for d in self.state.get('last_detections', []):
                try:
                    box = d.get('box', (0, 0, 0, 0))
                    safe_dets.append({
                        'box': [int(b) for b in box],
                        'class': str(d.get('class_name', '?')),
                        'conf': float(d.get('confidence', 0)),
                        'is_person': bool(d.get('is_person', False))
                    })
                except Exception:
                    pass

        return {
            "cam_id": self.cam_id,
            "source": self.source,
            "frame": self.state.get('current_frame'),
            "score": _safe_number(self.state.get('current_score', 0)),
            "category": cat,
            "fps": int(self.state.get('fps', 0)),
            "target_fps": int(target_fps),
            "frame_count": int(self.state.get('frame_count', 0)),
            "spike_rate": float(round(_safe_number(self.spike_gate.get_spike_rate()), 1)),
            "active_tracks": int(self.anomaly_detector.get_active_tracks()),
            "alerts": int(len(self.anomaly_detector.alerts)),
            "snn_spike": bool(self.state.get('snn_spike', False)),
            "snn_membrane": _safe_number(self.state.get('snn_membrane', 0)),
            "snn_diff": _safe_number(self.state.get('snn_diff', 0)),
            "snn_threshold": float(self.spike_gate.threshold),
            "last_alert": self.state.get('last_alert'),
            "session_name": self.state.get('session_name'),
            "timestamp": datetime.now().isoformat(),
            "detections": safe_dets
        }


# ============================================================
# ACTIVE CAMERAS REGISTRY
# ============================================================
cameras = {}          # cam_id -> CameraInstance
cameras_lock = threading.Lock()


def _safe_number(v):
    """Convert numpy/other numeric types to native Python for JSON serialization."""
    if v is None:
        return 0
    try:
        if isinstance(v, (int, float, bool)):
            return v
        return float(v)
    except (TypeError, ValueError):
        return 0


# ============================================================
# FASTAPI SERVER
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Stop all cameras on shutdown
    with cameras_lock:
        for cam in cameras.values():
            cam.stop()

app = FastAPI(
    title="EdgeVid LowBand API",
    version="2.0.0",
    description="The Camera That Thinks — Multi-Camera Neuromorphic Edge-AI DVR",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React build as static files
_BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "edgevid-dashboard", "build")
_BUILD_DIR = os.path.normpath(_BUILD_DIR)
if os.path.isdir(_BUILD_DIR):
    app.mount("/static", StaticFiles(directory=os.path.join(_BUILD_DIR, "static")), name="static")
    app.mount("/build", StaticFiles(directory=_BUILD_DIR), name="build_root")


# ---- REST Endpoints ----

@app.get("/")
def root():
    active = [cid for cid, c in cameras.items() if c.state['running']]
    return {
        "name": "EdgeVid LowBand",
        "version": "2.0.0",
        "active_cameras": active,
        "tagline": "The Camera That Thinks — Multi-Camera"
    }


@app.get("/api/cameras/detect")
def api_detect_cameras():
    """Probe for all available cameras connected to this device."""
    available = detect_available_cameras()
    for cam in available:
        cam_id = f"cam_{cam['index']}"
        cam['cam_id'] = cam_id
        cam['active'] = cam_id in cameras and cameras[cam_id].state['running']
    return {"cameras": available, "count": len(available)}


@app.get("/api/cameras")
def api_list_cameras():
    """List all active camera instances."""
    result = []
    for cam_id, cam in cameras.items():
        result.append({
            "cam_id": cam_id,
            "source": cam.source,
            "running": cam.state['running'],
            "fps": cam.state['fps'],
            "category": cam.state['current_category'],
            "score": cam.state['current_score'],
            "frame_count": cam.state['frame_count'],
        })
    return {"cameras": result, "count": len(result)}


@app.get("/api/stats")
def get_stats():
    total_frames = sum(c.state['frame_count'] for c in cameras.values())
    active_count = sum(1 for c in cameras.values() if c.state['running'])
    first_cam = next((c for c in cameras.values() if c.state['running']), None)
    if first_cam:
        return {
            "frame_count": total_frames,
            "current_score": first_cam.state['current_score'],
            "current_category": first_cam.state['current_category'],
            "fps": first_cam.state['fps'],
            "spike_rate": first_cam.spike_gate.get_spike_rate(),
            "compute_savings": first_cam.spike_gate.get_compute_savings(),
            "storage_savings": compressor.get_savings_percent(),
            "avg_score": first_cam.scorer.get_avg_score(),
            "active_tracks": first_cam.anomaly_detector.get_active_tracks(),
            "total_alerts": len(first_cam.anomaly_detector.alerts),
            "active_cameras": active_count,
            "score_distribution": first_cam.scorer.get_score_distribution(),
            "buffer_status": first_cam.prebuffer.get_buffer_status()
        }
    return {
        "frame_count": 0, "current_score": 0, "current_category": "IDLE",
        "fps": 0, "active_cameras": 0,
    }


@app.get("/api/events")
def get_events(category: str = None, limit: int = 50):
    events = database.get_recent_events(limit=limit, category=category)
    return {"events": events, "count": len(events)}


@app.get("/api/alerts")
def get_alerts(unacknowledged_only: bool = False):
    alerts = database.get_recent_alerts(unacknowledged_only=unacknowledged_only)
    return {"alerts": alerts, "count": len(alerts)}


@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int):
    database.acknowledge_alert(alert_id)
    return {"status": "acknowledged", "alert_id": alert_id}


# ---- Camera Control Endpoints ----

@app.get("/api/camera/status")
def camera_status():
    result = {}
    any_running = False
    for cam_id, cam in cameras.items():
        result[cam_id] = {
            "running": cam.state['running'],
            "fps": cam.state['fps'],
            "frame_count": cam.state['frame_count'],
        }
        if cam.state['running']:
            any_running = True
    return {
        "camera_on": any_running,
        "cameras": result,
        "count": len(cameras),
    }


@app.post("/api/camera/start")
def camera_start(source: int = 0, session_name: str = None):
    """Start a camera pipeline. Creates a CameraInstance if not exists."""
    cam_id = f"cam_{source}"
    with cameras_lock:
        if cam_id in cameras and cameras[cam_id].state['running']:
            return {"status": "already_running", "cam_id": cam_id}
        cam = CameraInstance(cam_id, source)
        cameras[cam_id] = cam
    cam.start(session_name)
    return {"status": "started", "cam_id": cam_id, "source": source,
            "session": cam.state['session_name']}


@app.post("/api/camera/stop")
def camera_stop(cam_id: str = None, source: int = None):
    """Stop a specific camera or the first running camera."""
    if cam_id is None and source is not None:
        cam_id = f"cam_{source}"
    if cam_id is None:
        cam_id = next((cid for cid, c in cameras.items() if c.state['running']), None)
    if cam_id and cam_id in cameras:
        cameras[cam_id].stop()
        return {"status": "stopped", "cam_id": cam_id}
    return {"status": "not_found"}


@app.post("/api/camera/stop_all")
def camera_stop_all():
    """Stop all running cameras."""
    stopped = []
    for cam_id, cam in cameras.items():
        if cam.state['running']:
            cam.stop()
            stopped.append(cam_id)
    return {"status": "stopped_all", "cameras": stopped}


@app.get("/api/clips")
def list_clips(response: Response):
    response.headers["Cache-Control"] = "no-store"
    clips_dir = "storage/clips"
    if not os.path.exists(clips_dir):
        return {"clips": []}
    clips = []
    for f in sorted(os.listdir(clips_dir), reverse=True):
        if f.endswith('.mp4'):
            fpath = os.path.join(clips_dir, f)
            size_kb = os.path.getsize(fpath) / 1024
            meta_path = fpath.replace('.mp4', '.json')
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as mf:
                        meta = json.load(mf)
                except Exception:
                    pass
            parts = f.replace('.mp4', '').split('_')
            timestamp_str = ''
            cat_from_name = 'EVENT'
            cam_from_name = ''
            try:
                if f.startswith('clip_cam_'):
                    # New: clip_cam_0_EVENT_123_20260305_120000.mp4
                    cam_from_name = f"cam_{parts[2]}"
                    cat_from_name = parts[3]
                    timestamp_str = f"{parts[5]}_{parts[6]}"
                elif f.startswith('clip_'):
                    cat_from_name = parts[1]
                    timestamp_str = f"{parts[3]}_{parts[4]}"
                else:
                    timestamp_str = f"{parts[3]}_{parts[4]}"
            except (IndexError, ValueError):
                pass
            readable_time = meta.get('start_time', '')
            if not readable_time and timestamp_str:
                try:
                    dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    readable_time = dt.isoformat()
                except ValueError:
                    readable_time = ''
            category = meta.get('category', cat_from_name)
            quality = meta.get('quality', 'HD' if category == 'EVENT' else 'MEDIUM' if category == 'NORMAL' else 'LOW')
            clips.append({
                "filename": f,
                "size_kb": round(size_kb, 1),
                "category": category,
                "quality": quality,
                "camera": meta.get('camera', cam_from_name or 'cam_0'),
                "fps": meta.get('fps', 15),
                "duration_sec": meta.get('duration_sec', round(size_kb / 50, 1)),
                "start_time": readable_time,
                "end_time": meta.get('end_time', ''),
            })
    return {"clips": clips, "count": len(clips)}


@app.get("/api/clips/{filename}")
def download_clip(filename: str):
    safe = os.path.basename(filename)
    filepath = os.path.join("storage", "clips", safe)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/mp4", filename=safe)
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/api/prebuffer")
def list_prebuffer(response: Response):
    response.headers["Cache-Control"] = "no-store"
    pb_dir = "storage/prebuffer"
    if not os.path.exists(pb_dir):
        return {"prebuffer": []}
    items = []
    for f in sorted(os.listdir(pb_dir), reverse=True):
        if f.endswith('.avi'):
            fpath = os.path.join(pb_dir, f)
            size_kb = os.path.getsize(fpath) / 1024
            # prebuffer_1_LOITERING_20260305_142300.avi
            # prebuffer_9_SCENE_ANOMALY_20260305_112707.avi
            stem = f.replace('.avi', '')
            event_type = 'UNKNOWN'
            readable_time = ''
            try:
                import re as _re
                m = _re.search(r'_(\d{8})_(\d{6})$', stem)
                if m:
                    ts_str = f"{m.group(1)}_{m.group(2)}"
                    dt = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
                    readable_time = dt.isoformat()
                    # Event type is between "prebuffer_N_" and "_YYYYMMDD"
                    first_us = stem.index('_')
                    second_us = stem.index('_', first_us + 1)
                    event_type = stem[second_us + 1:m.start()]
            except (IndexError, ValueError):
                pass
            items.append({
                "filename": f,
                "size_kb": round(size_kb, 1),
                "event_type": event_type,
                "start_time": readable_time,
                "duration_sec": round(size_kb / 40, 1),
            })
    return {"prebuffer": items, "count": len(items)}


@app.get("/api/prebuffer/{filename}")
def download_prebuffer(filename: str):
    safe = os.path.basename(filename)
    filepath = os.path.join("storage", "prebuffer", safe)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/x-msvideo", filename=safe)
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/api/session/clear")
def clear_session():
    cursor = database.conn.cursor()
    cursor.execute("DELETE FROM events")
    cursor.execute("DELETE FROM alerts")
    cursor.execute("DELETE FROM system_stats")
    database.conn.commit()
    for cam in cameras.values():
        cam.clear()
    compressor.stats = {k: 0 for k in compressor.stats}
    compressor.idle_batch.clear()
    return {"status": "cleared", "timestamp": datetime.now().isoformat()}


@app.get("/api/savings")
def get_savings():
    first_cam = next(iter(cameras.values()), None)
    spike_count = first_cam.spike_gate.spike_count if first_cam else 0
    frame_count = first_cam.spike_gate.frame_count if first_cam else 0
    storage_pct = compressor.get_savings_percent()
    return {
        "storage_savings_percent": storage_pct,
        "monthly_savings_inr": compressor.get_savings_rupees(40000),
        "yearly_savings_inr": compressor.get_savings_rupees(40000) * 12,
        "compression_stats": compressor.stats,
        "frames_processed": spike_count,
        "frames_skipped": frame_count - spike_count
    }


@app.get("/api/summary")
def get_summary():
    first_cam = next(iter(cameras.values()), None)
    return {
        "event_summary": database.get_event_summary(),
        "total_events": len(database.get_recent_events(limit=99999)),
        "compute_savings": first_cam.spike_gate.get_compute_savings() if first_cam else 0,
        "storage_savings": compressor.get_savings_percent()
    }


@app.get("/api/export")
def export_csv():
    filepath = database.export_to_csv()
    if filepath and os.path.exists(filepath):
        return FileResponse(
            filepath,
            media_type="text/csv",
            filename=f"edgevid_forensic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    return JSONResponse({"error": "no data to export"}, status_code=404)


# ---- Serve React App (catch-all) ----
@app.get("/app", include_in_schema=False)
@app.get("/app/{path:path}", include_in_schema=False)
def serve_react(path: str = ""):
    index = os.path.join(_BUILD_DIR, "index.html") if os.path.isdir(_BUILD_DIR) else None
    if index and os.path.exists(index):
        return FileResponse(index)
    return JSONResponse({"error": "React build not found. Run: npm run build"}, status_code=404)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    f = os.path.join(_BUILD_DIR, "favicon.ico")
    return FileResponse(f) if os.path.exists(f) else JSONResponse({}, status_code=404)

@app.get("/manifest.json", include_in_schema=False)
def manifest():
    f = os.path.join(_BUILD_DIR, "manifest.json")
    return FileResponse(f, media_type="application/json") if os.path.exists(f) else JSONResponse({}, status_code=404)

@app.get("/logo192.png", include_in_schema=False)
def logo192():
    f = os.path.join(_BUILD_DIR, "logo192.png")
    return FileResponse(f, media_type="image/png") if os.path.exists(f) else JSONResponse({}, status_code=404)

@app.get("/logo512.png", include_in_schema=False)
def logo512():
    f = os.path.join(_BUILD_DIR, "logo512.png")
    return FileResponse(f, media_type="image/png") if os.path.exists(f) else JSONResponse({}, status_code=404)


# ---- MJPEG Stream (uses first active camera) ----

def generate_mjpeg():
    while True:
        cam = next((c for c in cameras.values() if c.state['running']), None)
        if cam and cam.state['current_frame']:
            frame_bytes = base64.b64decode(cam.state['current_frame'])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')
        time.sleep(0.066)


@app.get("/stream")
def video_stream():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---- WebSocket: Real-time Multi-Camera Feed ----

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    print("✅ WebSocket client connected")
    try:
        while True:
            cam_data = {}
            for cam_id, cam in cameras.items():
                if cam.state['running'] or cam.state.get('current_frame'):
                    cam_data[cam_id] = cam.get_ws_data()

            # Build payload with multi-cam support
            first_cam_data = next(iter(cam_data.values()), None)

            payload = {
                "cameras": cam_data,
                "active_count": sum(1 for c in cameras.values() if c.state['running']),
                "timestamp": datetime.now().isoformat(),
            }

            # Backward compat: merge lightweight fields from first camera (exclude frame to avoid doubling)
            if first_cam_data:
                for k, v in first_cam_data.items():
                    if k != 'frame':
                        payload[k] = v

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.066)
    except WebSocketDisconnect:
        print("❌ WebSocket client disconnected")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ws_clients.discard(websocket)


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🧠 EdgeVid LowBand — The Camera That Thinks")
    print("   Multi-Camera Neuromorphic Edge-AI DVR v2.0")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("Dashboard: http://localhost:8000/app")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
