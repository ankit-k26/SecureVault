"""
AI SecureVault - Authentication Module
========================================
Orchestrates Multi-Factor Authentication:
  1. Face recognition (primary)
  2. Password / PIN (fallback)
  3. Lockout after repeated failures
  4. Session token management

This module is the bridge between face_module, encryption, and database.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import bcrypt
import numpy as np

from config import (
    MAX_FACE_ATTEMPTS,
    MAX_PASSWORD_ATTEMPTS,
    LOCKOUT_DURATION_SEC,
    BCRYPT_ROUNDS,
    MIN_FACE_FRAMES,
    FACE_TOLERANCE,
    LIVENESS_REQUIRED,
)
from database import (
    get_logger,
    log_auth_event,
    get_user,
    add_user_record,
    update_password_hash,
    deactivate_user,
    init_db,
)
from encryption import VaultManager, derive_fernet_key
from face_module import FaceManager, RecognitionResult

logger = get_logger("auth")


# ─────────────────────────────────────────────────────────────────────────────
# Auth state enums
# ─────────────────────────────────────────────────────────────────────────────

class AuthState(Enum):
    IDLE            = auto()
    SCANNING_FACE   = auto()
    AWAIT_PASSWORD  = auto()
    LOCKED_OUT      = auto()
    AUTHENTICATED   = auto()
    FAILED          = auto()


class AuthMethod(Enum):
    FACE     = "face"
    PASSWORD = "password"


@dataclass
class AuthResult:
    success:    bool        = False
    username:   str         = ""
    method:     AuthMethod  = AuthMethod.FACE
    message:    str         = ""
    session_token: str      = ""


# ─────────────────────────────────────────────────────────────────────────────
# Lockout tracker (in-memory; resets on app restart)
# ─────────────────────────────────────────────────────────────────────────────

class _LockoutTracker:
    def __init__(self):
        self._face_fails: dict[str, int]   = {}  # ip/session → count
        self._pwd_fails:  dict[str, int]   = {}
        self._locked_until: dict[str, float] = {}

    def _key(self) -> str:
        return "local"   # single-user desktop → one key is enough

    def is_locked(self) -> bool:
        key   = self._key()
        until = self._locked_until.get(key, 0)
        if time.time() < until:
            return True
        if key in self._locked_until:
            del self._locked_until[key]
        return False

    def remaining_lockout(self) -> float:
        key   = self._key()
        until = self._locked_until.get(key, 0)
        return max(0.0, until - time.time())

    def record_face_fail(self) -> int:
        key = self._key()
        self._face_fails[key] = self._face_fails.get(key, 0) + 1
        return self._face_fails[key]

    def record_pwd_fail(self) -> int:
        key = self._key()
        self._pwd_fails[key] = self._pwd_fails.get(key, 0) + 1
        total = self._pwd_fails[key]
        if total >= MAX_PASSWORD_ATTEMPTS:
            self._locked_until[key] = time.time() + LOCKOUT_DURATION_SEC
            logger.warning("Lockout triggered for %d seconds.", LOCKOUT_DURATION_SEC)
        return total

    def face_fails(self) -> int:
        return self._face_fails.get(self._key(), 0)

    def pwd_fails(self) -> int:
        return self._pwd_fails.get(self._key(), 0)

    def reset(self) -> None:
        key = self._key()
        self._face_fails.pop(key, None)
        self._pwd_fails.pop(key, None)
        self._locked_until.pop(key, None)


_lockout = _LockoutTracker()


# ─────────────────────────────────────────────────────────────────────────────
# Password helpers
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Return a bcrypt hash string (includes salt)."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(BCRYPT_ROUNDS)).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Constant-time bcrypt comparison."""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# User management
# ─────────────────────────────────────────────────────────────────────────────

class UserManager:
    """
    Manages vault users (password + face).
    Wraps database + face store operations.
    """

    def __init__(self, face_mgr: FaceManager):
        self._face_mgr = face_mgr

    def create_user(self, username: str, password: str) -> bool:
        """Create a new user record with a hashed password."""
        if get_user(username):
            logger.warning("User '%s' already exists.", username)
            return False
        pw_hash = hash_password(password)
        add_user_record(username, pw_hash)
        logger.info("User '%s' created.", username)
        return True

    def change_password(self, username: str, new_password: str) -> bool:
        if not get_user(username):
            return False
        update_password_hash(username, hash_password(new_password))
        logger.info("Password updated for '%s'.", username)
        return True

    def remove_user(self, username: str) -> bool:
        deactivate_user(username)
        self._face_mgr.remove_face(username)
        logger.info("User '%s' deactivated.", username)
        return True

    def register_face(self, username: str, image_paths: list[str]) -> bool:
        return self._face_mgr.register_face_from_images(username, image_paths)

    def register_face_from_frame(self, username: str, frame: np.ndarray) -> bool:
        return self._face_mgr.register_face_from_frame(username, frame)

    def has_face(self, username: str) -> bool:
        return username in self._face_mgr.list_registered_users()


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    username:  str
    token:     str = field(default_factory=lambda: secrets.token_hex(32))
    method:    AuthMethod = AuthMethod.FACE
    born:      float = field(default_factory=time.monotonic)

    def is_expired(self, ttl: float = 3600.0) -> bool:
        return time.monotonic() - self.born > ttl

    def refresh(self) -> None:
        self.born = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# AuthController — main public interface
# ─────────────────────────────────────────────────────────────────────────────

class AuthController:
    """
    Central authentication controller.

    Lifecycle:
        ctrl = AuthController()
        ctrl.start_face_auth()

        # Called per-frame from the GUI camera loop:
        result, state = ctrl.process_face_frame(bgr_frame)

        # If face fails, switch to password:
        result = ctrl.authenticate_password("alice", "s3cr3t")

        # After success, unlock vault:
        ctrl.post_auth_unlock(password)

        # On exit:
        ctrl.logout()
    """

    def __init__(self):
        init_db()
        self._face_mgr   = FaceManager()
        self._vault      = VaultManager()
        self._user_mgr   = UserManager(self._face_mgr)
        self._session:   Optional[Session] = None
        self._state:     AuthState = AuthState.IDLE

        # Consecutive positive-recognition frame counter
        self._positive_streak: int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> AuthState:
        return self._state

    @property
    def is_authenticated(self) -> bool:
        return self._session is not None and not self._session.is_expired()

    @property
    def current_user(self) -> Optional[str]:
        return self._session.username if self._session else None

    @property
    def vault(self) -> VaultManager:
        return self._vault

    @property
    def face_manager(self) -> FaceManager:
        return self._face_mgr

    @property
    def user_manager(self) -> UserManager:
        return self._user_mgr

    @property
    def lockout_remaining(self) -> float:
        return _lockout.remaining_lockout()

    # ── Face auth flow ────────────────────────────────────────────────────────

    def start_face_auth(self) -> None:
        """Reset liveness + streak counters and enter SCANNING_FACE state."""
        self._face_mgr.reset_liveness()
        self._positive_streak = 0
        self._state = AuthState.SCANNING_FACE
        logger.info("Face authentication started.")

    def process_face_frame(
        self, bgr_frame: np.ndarray
    ) -> tuple[RecognitionResult, AuthState]:
        """
        Feed one camera frame into the recognition pipeline.

        Returns (RecognitionResult, current_AuthState).

        State transitions:
          SCANNING_FACE
            → AUTHENTICATED  if MIN_FACE_FRAMES consecutive matches + liveness OK
            → AWAIT_PASSWORD if MAX_FACE_ATTEMPTS failed face detections
        """
        if self._state not in (AuthState.SCANNING_FACE,):
            return RecognitionResult(), self._state

        if _lockout.is_locked():
            self._state = AuthState.LOCKED_OUT
            return RecognitionResult(), self._state

        result = self._face_mgr.recognize_frame(bgr_frame)

        if result.matched and result.liveness_ok:
            self._positive_streak += 1
            if self._positive_streak >= MIN_FACE_FRAMES:
                return self._face_success(result)
        else:
            if self._positive_streak > 0:
                self._positive_streak = 0

            if result.face_found and not result.matched:
                # Unknown face → capture intruder
                fails = _lockout.record_face_fail()
                self._face_mgr.capture_intruder(
                    bgr_frame,
                    confidence=result.confidence,
                    notes=f"fail #{fails}",
                )
                log_auth_event("face_fail", method="face",
                               details=f"conf={result.confidence:.2f}")

                if fails >= MAX_FACE_ATTEMPTS:
                    logger.warning("Max face attempts reached — switching to password.")
                    self._state = AuthState.AWAIT_PASSWORD

        return result, self._state

    def _face_success(self, result: RecognitionResult) -> tuple[RecognitionResult, AuthState]:
        """Handle a confirmed face match."""
        username = result.username
        _lockout.reset()
        self._session = Session(username=username, method=AuthMethod.FACE)
        self._state   = AuthState.AUTHENTICATED
        log_auth_event("face_success", username=username, method="face",
                       details=f"conf={result.confidence:.2f}")
        logger.info("Face authentication SUCCESS for '%s'.", username)
        return result, self._state

    # ── Password auth flow ────────────────────────────────────────────────────

    def authenticate_password(self, username: str, password: str) -> AuthResult:
        """
        Validate username + password against stored bcrypt hash.
        Enforces lockout on MAX_PASSWORD_ATTEMPTS failures.
        """
        if _lockout.is_locked():
            self._state = AuthState.LOCKED_OUT
            return AuthResult(
                success=False,
                message=f"Account locked. Try again in {int(_lockout.remaining_lockout())}s",
            )

        user = get_user(username)
        if not user:
            # Deliberately vague — don't leak whether username exists
            _lockout.record_pwd_fail()
            log_auth_event("pwd_fail", username=username, method="password",
                           details="unknown_user")
            return AuthResult(success=False, message="Invalid credentials.")

        if not verify_password(password, user["password_hash"]):
            fails = _lockout.record_pwd_fail()
            log_auth_event("pwd_fail", username=username, method="password")
            remaining = MAX_PASSWORD_ATTEMPTS - fails
            msg = (
                f"Wrong password. {remaining} attempt(s) left."
                if remaining > 0
                else "Account locked due to too many failures."
            )
            logger.warning("Password fail for '%s' (attempt %d).", username, fails)
            return AuthResult(success=False, message=msg)

        # Success
        _lockout.reset()
        self._session = Session(username=username, method=AuthMethod.PASSWORD)
        self._state   = AuthState.AUTHENTICATED
        log_auth_event("pwd_success", username=username, method="password")
        logger.info("Password authentication SUCCESS for '%s'.", username)

        return AuthResult(
            success=True,
            username=username,
            method=AuthMethod.PASSWORD,
            message="Authentication successful.",
            session_token=self._session.token,
        )

    # ── Post-auth vault operations ────────────────────────────────────────────

    def post_auth_unlock(self, password: str) -> bool:
        """
        Called after successful authentication to unlock the vault.
        The password is used to derive the Fernet key.
        """
        if not self.is_authenticated:
            logger.error("post_auth_unlock called without active session.")
            return False
        return self._vault.unlock(password, username=self.current_user)

    def lock_and_logout(self) -> None:
        """Re-encrypt vault and clear session."""
        username = self.current_user
        if self._vault.is_unlocked:
            # We need the key still in session memory to lock
            self._vault.lock(username=username)
        self._session = None
        self._state   = AuthState.IDLE
        log_auth_event("logout", username=username)
        logger.info("User '%s' logged out — vault locked.", username)

    def emergency_lock(self, password: str) -> int:
        """
        Lock vault using password when session key has expired or is unavailable.
        """
        count = self._vault.lock_with_password(password, username=self.current_user)
        self._session = None
        self._state   = AuthState.IDLE
        return count

    # ── Admin / setup helpers ─────────────────────────────────────────────────

    def setup_first_user(self, username: str, password: str) -> bool:
        """
        Convenience method for first-run setup — creates DB user.
        Face registration must be done separately through UserManager.
        """
        return self._user_mgr.create_user(username, password)

    def has_any_user(self) -> bool:
        from database import list_users
        return len(list_users()) > 0

    def has_any_face(self) -> bool:
        return len(self._face_mgr.list_registered_users()) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor (convenient for GUI)
# ─────────────────────────────────────────────────────────────────────────────

_controller: Optional[AuthController] = None


def get_controller() -> AuthController:
    global _controller
    if _controller is None:
        _controller = AuthController()
    return _controller


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, getpass

    ctrl = get_controller()

    if not ctrl.has_any_user():
        print("=== First-run setup ===")
        username = input("Username: ")
        password = getpass.getpass("Password: ")
        ctrl.setup_first_user(username, password)
        print(f"User '{username}' created.")
        sys.exit(0)

    print("Users:", [u["username"] for u in __import__("database").list_users()])
    username = input("Username: ")
    password = getpass.getpass("Password: ")
    result   = ctrl.authenticate_password(username, password)
    print("Result:", result)

    if result.success:
        ctrl.post_auth_unlock(password)
        print("Vault:", "UNLOCKED" if ctrl.vault.is_unlocked else "LOCKED")
        input("Press Enter to lock and logout...")
        ctrl.lock_and_logout()
        print("Logged out.")
