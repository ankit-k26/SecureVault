"""
AI SecureVault - Database & Logging Module
Handles SQLite schema, event logging, and structured log queries.
"""

import sqlite3
import logging
import os
from datetime import datetime
from contextlib import contextmanager

from config import DB_PATH, LOG_FILE, LOGS_DIR


# ── Python logger (file + console) ───────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger attached to file + console."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.DEBUG)
        log.addHandler(_file_handler)
        log.addHandler(_console_handler)
    return log


# ── SQLite helpers ────────────────────────────────────────────────────────────
@contextmanager
def _get_conn():
    """Yield a thread-safe SQLite connection with WAL mode."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema initialisation ─────────────────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they do not already exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    NOT NULL UNIQUE,
                password_hash TEXT  NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                is_active   INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS auth_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT,
                event_type  TEXT    NOT NULL,   -- 'face_success','face_fail','pwd_success',
                                                --  'pwd_fail','lockout','logout'
                method      TEXT,               -- 'face' | 'password'
                ip_address  TEXT,
                timestamp   TEXT    NOT NULL DEFAULT (datetime('now')),
                details     TEXT
            );

            CREATE TABLE IF NOT EXISTS intruder_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path  TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL DEFAULT (datetime('now')),
                confidence  REAL,
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS encryption_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                action      TEXT    NOT NULL,   -- 'encrypt' | 'decrypt'
                file_count  INTEGER NOT NULL DEFAULT 0,
                username    TEXT,
                timestamp   TEXT    NOT NULL DEFAULT (datetime('now')),
                status      TEXT    NOT NULL DEFAULT 'success'
            );
        """)
    get_logger("db").info("Database initialised at %s", DB_PATH)


# ── Event logging helpers ─────────────────────────────────────────────────────
def log_auth_event(
    event_type: str,
    username: str = None,
    method: str = None,
    details: str = None,
) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO auth_events (username, event_type, method, details) VALUES (?,?,?,?)",
            (username, event_type, method, details),
        )


def log_intruder(image_path: str, confidence: float = None, notes: str = None) -> int:
    """Insert an intruder capture record and return its row id."""
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO intruder_logs (image_path, confidence, notes) VALUES (?,?,?)",
            (image_path, confidence, notes),
        )
        return cur.lastrowid


def log_encryption_event(action: str, file_count: int, username: str = None, status: str = "success") -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO encryption_events (action, file_count, username, status) VALUES (?,?,?,?)",
            (action, file_count, username, status),
        )


# ── Query helpers ─────────────────────────────────────────────────────────────
def get_auth_events(limit: int = 100) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM auth_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_intruder_logs(limit: int = 50) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM intruder_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_user(username: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1", (username,)
        ).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT id,username,created_at,is_active FROM users").fetchall()
    return [dict(r) for r in rows]


def add_user_record(username: str, password_hash: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?,?)",
            (username, password_hash),
        )


def update_password_hash(username: str, new_hash: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE username=?", (new_hash, username)
        )


def deactivate_user(username: str) -> None:
    with _get_conn() as conn:
        conn.execute("UPDATE users SET is_active=0 WHERE username=?", (username,))

def delete_intruder_log(log_id: int) -> bool:
    """Delete a single intruder record by ID. Returns True if a row was deleted."""
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM intruder_logs WHERE id=?", (log_id,))
        return cur.rowcount > 0


def clear_all_intruder_logs() -> int:
    """
    Delete ALL intruder records from the DB and remove their image files from disk.
    Returns the number of records deleted.
    """
    import os
    rows = get_intruder_logs(limit=100_000)   # fetch all first to get file paths
    with _get_conn() as conn:
        conn.execute("DELETE FROM intruder_logs")
    
    deleted_files = 0
    for row in rows:
        path = row.get("image_path", "")
        if path and os.path.exists(path):
            try:
                os.remove(path)
                deleted_files += 1
            except OSError as e:
                get_logger("db").warning("Could not remove intruder image %s: %s", path, e)

    get_logger("db").info(
        "Cleared %d intruder DB records, removed %d image files.",
        len(rows), deleted_files,
    )
    return len(rows)
