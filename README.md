# 🛡️ AI SecureVault

A desktop file-security application combining **face recognition**, **password authentication**, and **AES encryption** built with Python + PyQt5.

---

## Features

| Category | Details |
|---|---|
| **Face Auth** | OpenCV + face_recognition, liveness detection (EAR blink), multi-user |
| **Password Auth** | bcrypt hashing, configurable lockout after failed attempts |
| **Encryption** | AES-128 via Fernet, PBKDF2-HMAC-SHA256 key derivation, key NEVER stored |
| **GUI** | PyQt5, dark theme, live camera feed, file browser |
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

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

> **Windows note:** `dlib` can be tricky to compile. Use a pre-built wheel:
> ```
> pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.0-cp310-cp310-win_amd64.whl
> ```
> Adjust for your Python version.

> **Linux note:**
> ```
> sudo apt install cmake libboost-all-dev libopenblas-dev
> ```

### 2. Run

```bash
python main.py
```

On **first launch** you will be prompted to create an admin account. After logging in, go to **Settings → Enrol Face** to register your face.

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
| `FACE_TOLERANCE` | 0.50 | Strictness (lower = stricter) |
| `LIVENESS_REQUIRED` | True | Require blink for liveness |
| `MAX_FACE_ATTEMPTS` | 3 | Fails before password fallback |
| `MAX_PASSWORD_ATTEMPTS` | 3 | Fails before lockout |
| `LOCKOUT_DURATION_SEC` | 300 | Lockout duration (seconds) |
| `KDF_ITERATIONS` | 480,000 | PBKDF2 iterations |
| `SESSION_KEY_TTL_SEC` | 3600 | In-memory key TTL |
| `FRAME_SKIP` | 2 | Process every Nth camera frame |

---

## Security Notes

- The vault key is **derived** from your password; it's never written to disk
- Intruder photos are stored locally — not transmitted anywhere
- bcrypt work factor is 12 by default; increase `BCRYPT_ROUNDS` for more security
- For production use, consider adding full-disk encryption as an additional layer

---

## Limitations / Roadmap

- [ ] Cloud backup of encrypted vault
- [ ] Multi-device face sync
- [ ] Yubikey / TOTP second factor
- [ ] Automatic screen-lock on idle
- [ ] macOS / Linux tray icon
