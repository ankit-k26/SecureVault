"""
AI SecureVault - Configuration Constants
All tuneable parameters for the application live here.
"""

import os

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
FACE_DATA_DIR   = os.path.join(DATA_DIR, "face_data")
ENC_FILES_DIR   = os.path.join(DATA_DIR, "encrypted_files")
LOGS_DIR        = os.path.join(DATA_DIR, "logs")
INTRUDER_DIR    = os.path.join(DATA_DIR, "intruder_logs")
DB_PATH         = os.path.join(DATA_DIR, "database.db")

# ── Face recognition ──────────────────────────────────────────────────────────
FACE_TOLERANCE          = 0.50   # Lower = stricter match (0.0–1.0)
FACE_MODEL              = "hog"  # "hog" (CPU-fast) or "cnn" (GPU-accurate)
ENCODINGS_FILE          = os.path.join(FACE_DATA_DIR, "encodings.pkl")
MIN_FACE_FRAMES         = 3      # Consecutive matching frames to confirm identity
LIVENESS_BLINK_THRESH   = 0.25   # Eye-aspect-ratio threshold for blink detection
LIVENESS_REQUIRED       = True   # Require liveness check before granting access

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0
CAMERA_WIDTH      = 640
CAMERA_HEIGHT     = 480
CAMERA_FPS        = 30
FRAME_SKIP        = 2   # Process every Nth frame (reduces CPU load)

# ── Authentication ────────────────────────────────────────────────────────────
MAX_FACE_ATTEMPTS     = 3    # Failed attempts before fallback to password
MAX_PASSWORD_ATTEMPTS = 3    # Failed password attempts before lockout
LOCKOUT_DURATION_SEC  = 300  # 5 minutes
BCRYPT_ROUNDS         = 12   # Work factor for password hashing

# ── Encryption ────────────────────────────────────────────────────────────────
KDF_ITERATIONS      = 480_000  # PBKDF2-HMAC-SHA256 iterations
KDF_SALT_SIZE       = 32       # Salt bytes
FERNET_KEY_FILE     = os.path.join(DATA_DIR, ".vault_key")  # Derived, NOT stored plain
SESSION_KEY_TTL_SEC = 3600     # Key stays in memory max 1 hour

# ── GUI ───────────────────────────────────────────────────────────────────────
APP_TITLE       = "AI SecureVault"
WINDOW_WIDTH    = 960
WINDOW_HEIGHT   = 680
THEME_PRIMARY   = "#1a1a2e"
THEME_ACCENT    = "#e94560"
THEME_SURFACE   = "#16213e"
THEME_TEXT      = "#eaeaea"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(LOGS_DIR, "securevault.log")

# Ensure all directories exist when this module is imported
for _dir in (DATA_DIR, FACE_DATA_DIR, ENC_FILES_DIR, LOGS_DIR, INTRUDER_DIR):
    os.makedirs(_dir, exist_ok=True)
