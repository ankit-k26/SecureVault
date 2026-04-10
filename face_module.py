"""
AI SecureVault - Face Recognition Module
=========================================
Handles:
  • Encoding known faces and persisting them to disk
  • Real-time recognition from webcam frames
  • Liveness detection via Eye-Aspect-Ratio (EAR) blink analysis
  • Intruder capture (saves frame + logs to DB)

Dependencies: face_recognition, opencv-python, scipy, numpy
"""

from __future__ import annotations

import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

try:
    import face_recognition
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False

from config import (
    ENCODINGS_FILE,
    FACE_TOLERANCE,
    FACE_MODEL,
    MIN_FACE_FRAMES,
    LIVENESS_BLINK_THRESH,
    LIVENESS_REQUIRED,
    INTRUDER_DIR,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)
from database import get_logger, log_intruder

logger = get_logger("face_module")


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceRecord:
    username: str
    encoding: np.ndarray           # 128-dim face embedding
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RecognitionResult:
    matched:    bool   = False
    username:   str    = "unknown"
    confidence: float  = 0.0       # 0.0 = no match, 1.0 = perfect
    liveness_ok: bool  = False
    face_found: bool   = False


# ─────────────────────────────────────────────────────────────────────────────
# EAR-based liveness helper
# ─────────────────────────────────────────────────────────────────────────────

def _eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio (EAR).
    eye_landmarks: array of 6 (x,y) points ordered as per dlib 68-point model.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Returns a float; <LIVENESS_BLINK_THRESH indicates a blink.
    """
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Encoding store
# ─────────────────────────────────────────────────────────────────────────────

class EncodingStore:
    """Persists a list[FaceRecord] to a pickle file."""

    def __init__(self, path: str = ENCODINGS_FILE):
        self.path = path
        self._records: list[FaceRecord] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self._records = pickle.load(f)
                logger.info("Loaded %d face records from %s", len(self._records), self.path)
            except Exception as exc:
                logger.warning("Could not load encodings (%s). Starting fresh.", exc)
                self._records = []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self._records, f)
        logger.info("Saved %d face records.", len(self._records))

    def add(self, record: FaceRecord) -> None:
        # Replace any existing encoding for the same username
        self._records = [r for r in self._records if r.username != record.username]
        self._records.append(record)
        self.save()

    def remove(self, username: str) -> bool:
        before = len(self._records)
        self._records = [r for r in self._records if r.username != username]
        if len(self._records) < before:
            self.save()
            return True
        return False

    def list_users(self) -> list[str]:
        return [r.username for r in self._records]

    @property
    def records(self) -> list[FaceRecord]:
        return list(self._records)


# ─────────────────────────────────────────────────────────────────────────────
# Face Manager — public API used by auth & GUI
# ─────────────────────────────────────────────────────────────────────────────

class FaceManager:
    """
    Central class for face-recognition operations.

    Usage:
        mgr = FaceManager()
        mgr.register_face_from_images("alice", ["alice1.jpg", "alice2.jpg"])
        result = mgr.recognize_frame(bgr_frame)
    """

    def __init__(self):
        if not _FR_AVAILABLE:
            raise RuntimeError(
                "face_recognition library not installed. "
                "Run: pip install face_recognition"
            )
        self._store = EncodingStore()
        # Liveness tracking per session
        self._blink_counter: int = 0
        self._ear_history: list[float] = []
        self._blink_detected: bool = False

    # ── Registration ──────────────────────────────────────────────────────────

    def register_face_from_images(self, username: str, image_paths: list[str]) -> bool:
        """
        Compute the mean encoding from one or more training images and store it.
        Returns True on success.
        """
        encodings: list[np.ndarray] = []
        for path in image_paths:
            img = face_recognition.load_image_file(path)
            found = face_recognition.face_encodings(img, model=FACE_MODEL)
            if not found:
                logger.warning("No face detected in %s — skipping.", path)
                continue
            encodings.append(found[0])   # use the first detected face

        if not encodings:
            logger.error("No valid faces found for user '%s'.", username)
            return False

        mean_enc = np.mean(encodings, axis=0)
        self._store.add(FaceRecord(username=username, encoding=mean_enc))
        logger.info("Registered face for '%s' from %d image(s).", username, len(encodings))
        return True

    def register_face_from_frame(self, username: str, bgr_frame: np.ndarray) -> bool:
        """
        Register a face directly from a single OpenCV BGR frame (used in the GUI
        enrolment wizard).
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, model=FACE_MODEL)
        if not encodings:
            logger.warning("register_face_from_frame: no face in frame.")
            return False
        self._store.add(FaceRecord(username=username, encoding=encodings[0]))
        logger.info("Registered face for '%s' from live frame.", username)
        return True

    def remove_face(self, username: str) -> bool:
        return self._store.remove(username)

    def list_registered_users(self) -> list[str]:
        return self._store.list_users()

    # ── Recognition ───────────────────────────────────────────────────────────

    def recognize_frame(self, bgr_frame: np.ndarray) -> RecognitionResult:
        """
        Analyse one BGR frame.

        1. Detect face locations.
        2. Compute encodings for each detected face.
        3. Compare against known records.
        4. Optionally check liveness (EAR blink).
        5. Return a RecognitionResult.
        """
        result = RecognitionResult()

        if not self._store.records:
            logger.debug("No stored face records — recognition skipped.")
            return result

        # Resize for speed, run face detection on small frame, scale back
        scale = 0.5
        small = cv2.resize(bgr_frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small, model=FACE_MODEL)
        if not locations:
            return result   # face_found stays False

        result.face_found = True

        # Scale locations back to original frame size
        locations_full = [
            (int(t / scale), int(r / scale), int(b / scale), int(l / scale))
            for (t, r, b, l) in locations
        ]

        encodings = face_recognition.face_encodings(
            cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
            locations_full,
            model=FACE_MODEL,
        )

        known_encs  = [rec.encoding for rec in self._store.records]
        known_names = [rec.username  for rec in self._store.records]

        best_match_name = "unknown"
        best_confidence = 0.0

        for encoding in encodings:
            distances = face_recognition.face_distance(known_encs, encoding)
            min_idx   = int(np.argmin(distances))
            min_dist  = float(distances[min_idx])
            confidence = max(0.0, 1.0 - min_dist)   # convert distance → confidence

            if min_dist <= FACE_TOLERANCE and confidence > best_confidence:
                best_confidence = confidence
                best_match_name = known_names[min_idx]

        result.confidence = best_confidence
        result.matched    = best_match_name != "unknown"
        result.username   = best_match_name

        # Liveness check using facial landmarks + EAR
        if LIVENESS_REQUIRED:
            result.liveness_ok = self._check_liveness(bgr_frame, locations_full)
        else:
            result.liveness_ok = True

        return result

    # ── Liveness detection ────────────────────────────────────────────────────

    def _check_liveness(
        self, bgr_frame: np.ndarray, locations: list[tuple]
    ) -> bool:
        """
        Detect a blink to verify the subject is live (not a photo).
        Returns True once a blink has been observed in the current session.
        If already confirmed, remains True.
        """
        if self._blink_detected:
            return True

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        landmarks_list = face_recognition.face_landmarks(rgb, face_locations=locations)

        for landmarks in landmarks_list:
            left_eye  = np.array(landmarks.get("left_eye",  []))
            right_eye = np.array(landmarks.get("right_eye", []))

            if left_eye.size == 0 or right_eye.size == 0:
                continue

            ear = (_eye_aspect_ratio(left_eye) + _eye_aspect_ratio(right_eye)) / 2.0
            self._ear_history.append(ear)

            # Keep a rolling window of the last 20 EAR readings
            if len(self._ear_history) > 20:
                self._ear_history.pop(0)

            # A blink = EAR dips below threshold then rises back
            if len(self._ear_history) >= 3:
                recent = self._ear_history[-3:]
                if recent[1] < LIVENESS_BLINK_THRESH and recent[2] > LIVENESS_BLINK_THRESH:
                    self._blink_counter += 1
                    logger.debug("Blink detected (count=%d)", self._blink_counter)
                    if self._blink_counter >= 1:
                        self._blink_detected = True
                        return True

        return False

    def reset_liveness(self) -> None:
        """Call this at the start of each authentication session."""
        self._blink_counter = 0
        self._ear_history   = []
        self._blink_detected = False

    # ── Intruder capture ──────────────────────────────────────────────────────

    def capture_intruder(
        self,
        bgr_frame: np.ndarray,
        confidence: float = 0.0,
        notes: str = "",
    ) -> str:
        """
        Save an intruder frame to INTRUDER_DIR and log it in the database.
        Returns the saved image path.
        """
        os.makedirs(INTRUDER_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid  = uuid.uuid4().hex[:6]
        filename = f"intruder_{ts}_{uid}.jpg"
        path = os.path.join(INTRUDER_DIR, filename)

        cv2.imwrite(path, bgr_frame)
        log_intruder(path, confidence=confidence, notes=notes)
        logger.warning("Intruder captured → %s (conf=%.2f)", path, confidence)
        return path

    # ── Frame annotation (for GUI preview) ───────────────────────────────────

    @staticmethod
    def annotate_frame(
        bgr_frame: np.ndarray,
        result: RecognitionResult,
        locations: list[tuple] | None = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the frame for display in the GUI.
        If locations are not provided, a quick re-detect is performed.
        """
        frame = bgr_frame.copy()

        if locations is None:
            scale = 0.5
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_s = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            raw   = face_recognition.face_locations(rgb_s, model=FACE_MODEL)
            locations = [
                (int(t/scale), int(r/scale), int(b/scale), int(l/scale))
                for (t, r, b, l) in raw
            ]

        color  = (0, 255, 0) if result.matched else (0, 0, 255)
        label  = f"{result.username} ({result.confidence:.0%})" if result.matched else "Unknown"

        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, label,
                (left + 4, bottom - 8),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55, (255, 255, 255), 1,
            )

        # Liveness indicator
        lv_text  = "LIVE ✓" if result.liveness_ok else "Blink to verify"
        lv_color = (0, 255, 128) if result.liveness_ok else (0, 165, 255)
        cv2.putText(
            frame, lv_text,
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lv_color, 2,
        )

        return frame


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI test (python face_module.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from database import init_db

    init_db()
    mgr = FaceManager()

    if len(sys.argv) == 3 and sys.argv[1] == "register":
        username   = sys.argv[2]
        cap        = cv2.VideoCapture(0)
        print(f"[INFO] Look at camera. Press SPACE to capture for '{username}', Q to quit.")
        captured   = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Register Face — Press SPACE", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                tmp = f"/tmp/_reg_{uuid.uuid4().hex}.jpg"
                cv2.imwrite(tmp, frame)
                captured.append(tmp)
                print(f"  Captured #{len(captured)}")
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured:
            ok = mgr.register_face_from_images(username, captured)
            print("Registration", "SUCCESS" if ok else "FAILED")
            # Clean up temp files
            for f in captured:
                os.unlink(f)
        sys.exit(0)

    # Default: live recognition demo
    mgr.reset_liveness()
    cap    = cv2.VideoCapture(0)
    streak = 0
    print("[INFO] Running live recognition. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = mgr.recognize_frame(frame)

        if result.matched:
            streak += 1
            if streak >= MIN_FACE_FRAMES:
                print(f"[AUTH] Recognised: {result.username}  conf={result.confidence:.1%}  live={result.liveness_ok}")
                streak = 0
        else:
            streak = 0
            if result.face_found:
                mgr.capture_intruder(frame, confidence=result.confidence, notes="CLI test")

        annotated = FaceManager.annotate_frame(frame, result)
        cv2.imshow("AI SecureVault — Face Recognition", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
