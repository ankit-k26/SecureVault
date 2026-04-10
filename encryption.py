"""
AI SecureVault - Encryption Module
====================================
Provides AES-128-CBC (via Fernet) encryption with PBKDF2-HMAC-SHA256 key derivation.

Key security model
------------------
• The vault key is DERIVED from the user's password using PBKDF2 + a per-vault salt.
• The raw key is NEVER written to disk — only the salt is persisted.
• In-memory key has a configurable TTL; it is zeroed after use.
• Each file is encrypted with Fernet (AES-128-CBC + HMAC-SHA256 + IV).

File format on disk
-------------------
  [4-byte magic "ASVF"] + [Fernet ciphertext bytes]

  The magic lets us quickly skip already-encrypted files on re-encrypt passes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import struct
import time
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from config import (
    KDF_ITERATIONS,
    KDF_SALT_SIZE,
    ENC_FILES_DIR,
    SESSION_KEY_TTL_SEC,
    DATA_DIR,
)
from database import get_logger, log_encryption_event

logger = get_logger("encryption")

MAGIC        = b"ASVF"        # 4-byte file header magic
SALT_FILE    = os.path.join(DATA_DIR, ".vault_salt")  # persisted salt (NOT the key)
EXTENSION    = ".enc"         # appended to encrypted files


# ─────────────────────────────────────────────────────────────────────────────
# Key derivation
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_create_salt() -> bytes:
    """Return the vault salt, creating it on first run."""
    if os.path.exists(SALT_FILE):
        with open(SALT_FILE, "rb") as f:
            return f.read()
    salt = os.urandom(KDF_SALT_SIZE)
    os.makedirs(os.path.dirname(SALT_FILE), exist_ok=True)
    with open(SALT_FILE, "wb") as f:
        f.write(salt)
    logger.info("New vault salt created at %s", SALT_FILE)
    return salt


def derive_fernet_key(password: str) -> bytes:
    """
    Derive a 32-byte URL-safe base64 Fernet key from the user's password.
    The PBKDF2 salt is loaded from disk (or created on first call).
    """
    salt = _load_or_create_salt()
    kdf  = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=KDF_ITERATIONS,
        backend=default_backend(),
    )
    raw_key = kdf.derive(password.encode("utf-8"))
    return base64.urlsafe_b64encode(raw_key)   # Fernet requires URL-safe base64


# ─────────────────────────────────────────────────────────────────────────────
# Session key holder (in-memory, TTL-expiring)
# ─────────────────────────────────────────────────────────────────────────────

class _SessionKey:
    """Holds a derived Fernet key in memory for SESSION_KEY_TTL_SEC seconds."""

    def __init__(self):
        self._key:  Optional[bytes]  = None
        self._born: float            = 0.0

    def set(self, key: bytes) -> None:
        self._key  = key
        self._born = time.monotonic()

    def get(self) -> Optional[bytes]:
        if self._key is None:
            return None
        if time.monotonic() - self._born > SESSION_KEY_TTL_SEC:
            self.clear()
            return None
        return self._key

    def clear(self) -> None:
        if self._key:
            # Overwrite before GC
            self._key = b"\x00" * len(self._key)
        self._key  = None
        self._born = 0.0

    @property
    def is_active(self) -> bool:
        return self.get() is not None


_session = _SessionKey()


# ─────────────────────────────────────────────────────────────────────────────
# Low-level file encrypt / decrypt
# ─────────────────────────────────────────────────────────────────────────────

def _is_encrypted(path: str) -> bool:
    """Return True if the file starts with our magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == MAGIC
    except OSError:
        return False


def encrypt_file(src: str, fernet: Fernet, delete_original: bool = True) -> str:
    """
    Encrypt a single file in-place (or to .enc sibling).

    Writes: MAGIC + fernet.encrypt(plaintext)
    Returns the path of the encrypted file.
    """
    if _is_encrypted(src):
        logger.debug("Already encrypted, skipping: %s", src)
        return src

    with open(src, "rb") as f:
        plaintext = f.read()

    ciphertext = fernet.encrypt(plaintext)
    dst = src + EXTENSION if not src.endswith(EXTENSION) else src

    with open(dst, "wb") as f:
        f.write(MAGIC)
        f.write(ciphertext)

    if delete_original and dst != src:
        os.remove(src)
        logger.debug("Encrypted %s → %s", src, dst)

    return dst


def decrypt_file(src: str, fernet: Fernet, delete_encrypted: bool = True) -> str:
    """
    Decrypt a single .enc file back to its original.

    Returns the path of the plaintext file.
    Raises InvalidToken if the key is wrong or file is corrupted.
    """
    if not _is_encrypted(src):
        logger.debug("Not an encrypted file, skipping: %s", src)
        return src

    with open(src, "rb") as f:
        f.read(4)                  # consume magic
        ciphertext = f.read()

    try:
        plaintext = fernet.decrypt(ciphertext)
    except InvalidToken as exc:
        logger.error("Decryption failed for %s: %s", src, exc)
        raise

    # Strip the .enc extension to recover original filename
    dst = src[:-len(EXTENSION)] if src.endswith(EXTENSION) else src + ".dec"

    with open(dst, "wb") as f:
        f.write(plaintext)

    if delete_encrypted and dst != src:
        os.remove(src)
        logger.debug("Decrypted %s → %s", src, dst)

    return dst


# ─────────────────────────────────────────────────────────────────────────────
# Vault Manager — public API
# ─────────────────────────────────────────────────────────────────────────────

class VaultManager:
    """
    High-level interface for encrypting/decrypting an entire protected folder.

    Typical flow:
        vault = VaultManager()
        vault.unlock(password, username)      # derives key, decrypts folder
        ...user works with files...
        vault.lock(username)                  # re-encrypts folder, clears key
    """

    def __init__(self, protected_dir: str = ENC_FILES_DIR):
        self.protected_dir = protected_dir
        os.makedirs(self.protected_dir, exist_ok=True)

    # ── Session management ────────────────────────────────────────────────────

    def unlock(self, password: str, username: str = None) -> bool:
        """
        Derive key, store in session, decrypt all files in protected_dir.
        Returns True on success.
        """
        logger.info("Vault unlock requested by '%s'.", username)
        try:
            key    = derive_fernet_key(password)
            fernet = Fernet(key)

            count = self._decrypt_folder(fernet)
            _session.set(key)

            log_encryption_event("decrypt", count, username)
            logger.info("Vault unlocked — %d file(s) decrypted.", count)
            return True

        except InvalidToken:
            logger.error("Vault unlock FAILED — wrong password.")
            log_encryption_event("decrypt", 0, username, status="failed_bad_key")
            return False
        except Exception as exc:
            logger.error("Vault unlock error: %s", exc)
            log_encryption_event("decrypt", 0, username, status=f"error: {exc}")
            return False

    def lock(self, username: str = None) -> int:
        """
        Encrypt all files in protected_dir and clear the session key.
        Returns the number of files encrypted.
        """
        logger.info("Vault lock requested by '%s'.", username)
        key = _session.get()
        if key is None:
            logger.warning("Lock called but no active session key — re-deriving not possible.")
            return 0

        fernet = Fernet(key)
        count  = self._encrypt_folder(fernet)
        _session.clear()

        log_encryption_event("encrypt", count, username)
        logger.info("Vault locked — %d file(s) encrypted.", count)
        return count

    def lock_with_password(self, password: str, username: str = None) -> int:
        """
        Lock without a live session — derive key from password first.
        Useful for emergency re-encryption.
        """
        key    = derive_fernet_key(password)
        fernet = Fernet(key)
        count  = self._encrypt_folder(fernet)
        log_encryption_event("encrypt", count, username)
        logger.info("Vault locked (pwd path) — %d file(s) encrypted.", count)
        return count

    @property
    def is_unlocked(self) -> bool:
        return _session.is_active

    def refresh_session(self) -> None:
        """Renew the TTL timestamp of the current session key."""
        key = _session.get()
        if key:
            _session.set(key)   # resets the born timestamp

    # ── Folder traversal ──────────────────────────────────────────────────────

    def _collect_files(self) -> list[str]:
        """Recursively collect all non-hidden files in protected_dir."""
        result = []
        for root, dirs, files in os.walk(self.protected_dir):
            # Skip hidden dirs (e.g. .git)
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if fname.startswith("."):
                    continue
                result.append(os.path.join(root, fname))
        return result

    def _encrypt_folder(self, fernet: Fernet) -> int:
        files   = self._collect_files()
        count   = 0
        errors  = 0
        for path in files:
            if _is_encrypted(path):
                continue   # already encrypted
            try:
                encrypt_file(path, fernet)
                count += 1
            except Exception as exc:
                logger.error("Could not encrypt %s: %s", path, exc)
                errors += 1
        if errors:
            logger.warning("%d file(s) failed to encrypt.", errors)
        return count

    def _decrypt_folder(self, fernet: Fernet) -> int:
        files   = self._collect_files()
        count   = 0
        errors  = 0
        for path in files:
            if not _is_encrypted(path):
                continue   # plaintext or unknown
            try:
                decrypt_file(path, fernet)
                count += 1
            except InvalidToken:
                raise   # bubble up — means wrong password
            except Exception as exc:
                logger.error("Could not decrypt %s: %s", path, exc)
                errors += 1
        if errors:
            logger.warning("%d file(s) failed to decrypt.", errors)
        return count

    # ── File listing (for in-app browser) ────────────────────────────────────

    def list_files(self) -> list[dict]:
        """
        Return metadata for all files in the protected directory.
        Shows decrypted filenames when vault is unlocked.
        """
        result = []
        for path in self._collect_files():
            stat = os.stat(path)
            name = os.path.basename(path)
            display = name.removesuffix(EXTENSION) if not self.is_unlocked else name
            result.append({
                "name":     display,
                "path":     path,
                "size":     stat.st_size,
                "modified": stat.st_mtime,
                "encrypted": _is_encrypted(path),
            })
        return result

    def add_file(self, src_path: str, username: str = None) -> str:
        """
        Copy an external file into the vault, encrypting it immediately
        if a session is active.
        """
        import shutil
        dst = os.path.join(self.protected_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst)

        key = _session.get()
        if key:
            fernet = Fernet(key)
            dst    = encrypt_file(dst, fernet)
            log_encryption_event("encrypt", 1, username)
            logger.info("Added and encrypted file: %s", dst)
        else:
            logger.warning("File added unencrypted (vault locked): %s", dst)

        return dst

    def delete_file(self, path: str) -> bool:
        """Securely delete a file from the vault (3-pass overwrite)."""
        if not os.path.exists(path):
            return False
        size = os.path.getsize(path)
        with open(path, "r+b") as f:
            for _ in range(3):
                f.seek(0)
                f.write(os.urandom(size))
        os.remove(path)
        logger.info("Securely deleted: %s", path)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, getpass
    from database import init_db

    init_db()
    vault = VaultManager()
    cmd   = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "unlock":
        pw = getpass.getpass("Password: ")
        ok = vault.unlock(pw, username="cli_test")
        print("Unlocked ✓" if ok else "Failed ✗ — wrong password")

    elif cmd == "lock":
        pw  = getpass.getpass("Password (to re-encrypt): ")
        cnt = vault.lock_with_password(pw, username="cli_test")
        print(f"Locked — {cnt} file(s) encrypted.")

    elif cmd == "status":
        files = vault.list_files()
        print(f"Vault: {'UNLOCKED' if vault.is_unlocked else 'LOCKED'}")
        print(f"Files ({len(files)}):")
        for f in files:
            enc = "[ENC]" if f["encrypted"] else "     "
            print(f"  {enc} {f['name']}  ({f['size']} bytes)")
    else:
        print(f"Unknown command: {cmd}. Use: unlock | lock | status")
