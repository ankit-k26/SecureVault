"""
AI SecureVault - Camera Capability Test
========================================
Run:  python test_camera.py

Tests the connected camera(s) and reports:
  - Supported FPS (requested vs granted)
  - Supported resolutions
  - Actual measured FPS from live frames
  - Backend / driver info
"""

import cv2
import time


# ── Config ────────────────────────────────────────────────────────────────────

CAMERA_INDEX      = 0          # change if you have multiple cameras
MEASURE_DURATION  = 3          # seconds to measure actual live FPS

# Common resolutions to probe
RESOLUTIONS_TO_TEST = [
    (160,  120),
    (320,  240),
    (640,  480),
    (800,  600),
    (1280, 720),
    (1920, 1080),
]

# FPS values to probe
FPS_TO_TEST = [15, 24, 30, 60, 90, 120, 240]


# ── Helpers ───────────────────────────────────────────────────────────────────

def separator(char="─", width=60):
    print(char * width)


def open_camera(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌  Could not open camera at index {index}.")
        return None
    return cap


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_backend(cap: cv2.VideoCapture):
    separator()
    print("📷  CAMERA BACKEND")
    separator()
    backend = cap.getBackendName()
    print(f"  Backend : {backend}")
    print(f"  Index   : {CAMERA_INDEX}")


def test_default_properties(cap: cv2.VideoCapture):
    separator()
    print("⚙️   DEFAULT PROPERTIES (before any changes)")
    separator()
    w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fmt = cap.get(cv2.CAP_PROP_FORMAT)
    buf = cap.get(cv2.CAP_PROP_BUFFERSIZE)
    print(f"  Resolution : {int(w)} x {int(h)}")
    print(f"  FPS        : {fps}")
    print(f"  Format     : {fmt}")
    print(f"  Buffer size: {buf}")


def test_max_fps(cap: cv2.VideoCapture):
    separator()
    print("🚀  MAX FPS DETECTION (absurd-value trick)")
    separator()
    cap.set(cv2.CAP_PROP_FPS, 10000)
    granted = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Requested : 10000 fps")
    print(f"  Granted   : {granted} fps")
    if granted <= 0:
        print("  ⚠️  Camera returned 0 — driver may not support FPS negotiation.")
    else:
        print(f"  ✅  Hardware maximum appears to be {int(granted)} fps")
    return granted


def test_fps_support(cap: cv2.VideoCapture):
    separator()
    print("📊  FPS PROBE  (request each value, read back granted)")
    separator()
    print(f"  {'Requested':>12}  {'Granted':>10}  {'Match?':>8}")
    separator("-")
    supported = []
    for fps in FPS_TO_TEST:
        cap.set(cv2.CAP_PROP_FPS, fps)
        granted = cap.get(cv2.CAP_PROP_FPS)
        match = "✅" if abs(granted - fps) < 1 else "⚠️ "
        print(f"  {fps:>12}  {granted:>10.1f}  {match:>8}")
        if abs(granted - fps) < 1:
            supported.append(fps)
    print()
    if supported:
        print(f"  Likely supported FPS values : {supported}")
    else:
        print("  ⚠️  Driver may not support FPS negotiation via OpenCV.")


def test_resolution_support(cap: cv2.VideoCapture):
    separator()
    print("🖼️   RESOLUTION PROBE")
    separator()
    print(f"  {'Requested':>14}  {'Granted':>14}  {'Match?':>8}")
    separator("-")
    supported = []
    for w, h in RESOLUTIONS_TO_TEST:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        match = "✅" if (gw == w and gh == h) else "⚠️ "
        print(f"  {f'{w}x{h}':>14}  {f'{gw}x{gh}':>14}  {match:>8}")
        if gw == w and gh == h:
            supported.append((w, h))
    print()
    if supported:
        print(f"  Confirmed supported resolutions : {[f'{w}x{h}' for w,h in supported]}")


def test_measured_fps(cap: cv2.VideoCapture):
    separator()
    print(f"⏱️   MEASURED LIVE FPS  ({MEASURE_DURATION}s capture at default resolution)")
    separator()

    # Reset to defaults first
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10000)   # ask for max

    # Warm-up: discard first few frames
    for _ in range(5):
        cap.read()

    frame_count = 0
    start       = time.perf_counter()
    deadline    = start + MEASURE_DURATION

    while time.perf_counter() < deadline:
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_count += 1

    elapsed     = time.perf_counter() - start
    measured    = frame_count / elapsed if elapsed > 0 else 0

    print(f"  Frames captured : {frame_count}")
    print(f"  Elapsed         : {elapsed:.2f}s")
    print(f"  Measured FPS    : {measured:.1f}")

    if measured >= 55:
        tier = "🟢  Excellent (60+ fps capable)"
    elif measured >= 28:
        tier = "🟡  Good (30 fps)"
    elif measured >= 18:
        tier = "🟠  Moderate (24 fps)"
    else:
        tier = "🔴  Low (<20 fps) — consider reducing resolution"

    print(f"  Assessment      : {tier}")
    return measured


def test_recommended_config(measured_fps: float, max_fps_granted: float):
    separator("═")
    print("✅  RECOMMENDED config.py VALUES")
    separator("═")

    fps = int(max_fps_granted) if max_fps_granted > 0 else int(measured_fps)
    fps = max(fps, 1)

    # Pick best resolution tier based on fps
    if fps >= 60:
        w, h = 640, 480
    elif fps >= 30:
        w, h = 640, 480
    else:
        w, h = 320, 240

    skip = 1 if fps >= 30 else 2

    print(f"  CAMERA_WIDTH  = {w}")
    print(f"  CAMERA_HEIGHT = {h}")
    print(f"  CAMERA_FPS    = {fps}   # or remove and use auto-detect")
    print(f"  FRAME_SKIP    = {skip}")
    separator("═")
    print()
    print("  Auto-detect snippet for CameraWorker.run():")
    print()
    print("    cap.set(cv2.CAP_PROP_FPS, 10000)")
    print("    actual_fps = cap.get(cv2.CAP_PROP_FPS)")
    print("    if actual_fps <= 0:")
    print("        actual_fps = 30")
    print("    self._actual_fps = int(actual_fps)")
    print()
    print("    # in the loop:")
    print("    self.msleep(1000 // self._actual_fps)")
    separator("═")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    separator("═")
    print("  AI SecureVault — Camera Capability Test")
    separator("═")
    print()

    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        return

    try:
        test_backend(cap)
        test_default_properties(cap)
        max_fps = test_max_fps(cap)
        test_fps_support(cap)
        test_resolution_support(cap)
        measured = test_measured_fps(cap)
        test_recommended_config(measured, max_fps)
    finally:
        cap.release()
        print("Camera released. Done.")


if __name__ == "__main__":
    main()