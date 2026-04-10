# 🛡️ AI SecureVault

A desktop file-security application combining **face recognition**, **password authentication**, and **AES encryption** built with Python + PyQt5.

---

## Features

| Category | Details |
|---|---|
| **Face Auth** | OpenCV + face_recognition, liveness detection (EAR blink), multi-user |
| **Password Auth** | bcrypt hashing, configurable lockout after failed attempts |
| **Encryption** | AES-128 via Fernet, PBKDF2-HMAC-SHA256 key derivation, key NEVER stored |
| **GUI** | PyQt5, dark theme, live camera feed (~30 fps), file browser |
| **Logging** | SQLite events, intruder photo capture with timestamps |
| **Packaging** | PyInstaller `.exe` / binary support |

---

## Project Structure

```
ai_securevault/
├── main.py             # Entry point
├── config.py           # All tuneable constants
├── auth.py             # Authentication controller (MFA orchestration)
├── face_module.py      # Face recognition + liveness + intruder capture
├── encryption.py       # Fernet/AES vault manager
├── database.py         # SQLite schema + logging helpers
├── gui.py              # PyQt5 UI (all screens)
├── requirements.txt
├── securevault.spec    # PyInstaller spec
└── data/
    ├── encrypted_files/    # Protected vault folder
    ├── face_data/          # Stored face encodings (.pkl)
    ├── intruder_logs/      # Captured intruder images
    ├── logs/               # Text log file
    └── database.db         # SQLite database
```

---

## Quick Start

### 1. Install dependencies

**Windows (Python 3.12) — do these steps in order:**

```bash
# Step 1 — pin setuptools before anything else
pip install --force-reinstall setuptools==69.5.1

# Step 2 — install the pre-built dlib wheel (place the .whl in the project folder first)
pip install dlib-19.24.99-cp312-cp312-win_amd64.whl

# Step 3 — install the rest
pip install -r requirements.txt
```

**Linux:**

```bash
sudo apt install cmake libboost-all-dev libopenblas-dev
pip install -r requirements.txt
```

**macOS:**

```bash
brew install cmake boost
pip install -r requirements.txt
```

> **Virtual environment (recommended for all platforms):**
> ```bash
> python -m venv venv
> source venv/bin/activate      # Windows: venv\Scripts\activate
> ```

### 2. Run

```bash
python main.py
```

On **first launch** you will be prompted to create an admin account. After logging in, go to **Settings → Enrol Face** to register your face.

---

## Face Enrolment Tips

Good enrolment quality is the single biggest factor in recognition accuracy.

- Capture **5–10 frames** during enrolment (the window accepts up to 10)
- Vary your pose slightly across captures — straight on, slight left/right tilt
- Ensure **consistent, even lighting** — avoid bright backlighting or harsh shadows
- Remove glasses if you usually authenticate without them (or enrol both ways)
- After enrolment, delete `data/face_data/encodings.pkl` and re-enrol if recognition remains unreliable — stale or corrupt encodings from an earlier session can interfere

---

## Authentication Flow

```
App Launch
    │
    ▼
┌─────────────────────────┐
│  FACE RECOGNITION       │  ← Live webcam, liveness (blink required)
│  • Matched + alive?     │
└────────┬────────────────┘
         │ YES → Authenticated ✅
         │ NO  (3 fails)
         ▼
┌─────────────────────────┐
│  PASSWORD FALLBACK      │  ← bcrypt-verified
│  • Correct password?    │
└────────┬────────────────┘
         │ YES → Authenticated ✅
         │ NO  (3 fails)
         ▼
    LOCKOUT 5 min 🔒
```

When face recognition fails, the camera is automatically stopped and the password form is shown. Click **"Try Face Again"** to restart the camera and retry face auth at any time.

---

## Encryption Model

- Vault password → PBKDF2-HMAC-SHA256 (480,000 iterations) → 256-bit key
- Key converted to Fernet (AES-128-CBC + HMAC-SHA256)
- **Salt only** is stored on disk (`.vault_salt`); the key is **never persisted**
- Session key lives in memory with a 1-hour TTL; zeroed on logout/exit
- File format: `[ASVF magic 4 bytes] + [Fernet ciphertext]`
- On app exit → all files are automatically re-encrypted

---

## Building an .exe (Windows)

```bash
pip install pyinstaller
pyinstaller securevault.spec
```

The output will be in `dist/AI_SecureVault/`.
Distribute the entire `AI_SecureVault/` folder (not just the `.exe`).

To create a **fake-folder launcher**, rename `AI_SecureVault.exe` to something innocuous like `My Documents.exe` and place it alongside an empty `My Documents` folder.

---

## Configuration

All parameters are in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `FACE_TOLERANCE` | 0.50 | Match strictness — lower is stricter (0.0–1.0) |
| `LIVENESS_REQUIRED` | True | Require a blink before granting access |
| `MIN_FACE_FRAMES` | 3 | Consecutive matching frames needed to confirm identity |
| `MAX_FACE_ATTEMPTS` | 3 | Failed face attempts before falling back to password |
| `MAX_PASSWORD_ATTEMPTS` | 3 | Failed password attempts before lockout |
| `LOCKOUT_DURATION_SEC` | 300 | Lockout duration in seconds (default 5 min) |
| `KDF_ITERATIONS` | 480,000 | PBKDF2 iterations for key derivation |
| `SESSION_KEY_TTL_SEC` | 3600 | In-memory key TTL (1 hour) |
| `FRAME_SKIP` | 4 | Run recognition every Nth frame (~7–8 checks/sec at 30 fps) |
| `FACE_MODEL` | `"hog"` | Detection model — `"hog"` (CPU) or `"cnn"` (GPU, more accurate) |
| `BCRYPT_ROUNDS` | 12 | bcrypt work factor — increase for stronger password hashing |

---

## Performance Notes

The camera feed targets **~30 fps** display. Face recognition (the expensive part) runs on a separate counter controlled by `FRAME_SKIP` so it never blocks the display loop:

- **FRAME_SKIP 4** (default) → recognition runs ~7–8×/sec, display stays fluid
- **FRAME_SKIP 2** → recognition runs ~15×/sec, may cause slight lag on slower CPUs
- **FRAME_SKIP 6** → recognition runs ~5×/sec, very smooth but slightly slower to trigger

If the feed is still laggy, switch `FACE_MODEL` from `"hog"` to `"cnn"` only if you have a CUDA-capable GPU — on CPU, CNN mode is significantly slower.

---

## Security Notes

- The vault key is **derived** from your password; it is never written to disk
- Intruder photos are stored locally — not transmitted anywhere
- bcrypt work factor is 12 by default; increase `BCRYPT_ROUNDS` for stronger hashing
- For production use, consider adding full-disk encryption as an additional layer

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Face not recognised after enrolment | Delete `data/face_data/encodings.pkl` and re-enrol with 5–10 well-lit captures |
| Password Unlock button unresponsive | Fixed in current version — ensure you are running the latest `gui.py` |
| Camera feed laggy | Increase `FRAME_SKIP` in `config.py` (try 4 or 6) |
| `dlib` fails to install on Windows | Use the pre-built wheel: `pip install dlib-19.24.99-cp312-cp312-win_amd64.whl` |
| `face_recognition` import error | Ensure `setuptools==69.5.1` was installed **before** dlib/face_recognition |
| No face detected during enrolment | Improve lighting; avoid strong backlight; face the camera straight on |

---

## Limitations / Roadmap

- [ ] Cloud backup of encrypted vault
- [ ] Multi-device face sync
- [ ] Yubikey / TOTP second factor
- [ ] Automatic screen-lock on idle
- [ ] macOS / Linux tray icon