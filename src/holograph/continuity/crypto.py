"""Encryption-at-rest for the memory store.

The persistent memory holds personal, medical, and identity data — it must not
sit on disk as plaintext. SQLCipher (transparent whole-DB encryption) is the
ideal, but it requires a native lib that isn't always present. This module
provides a portable Fernet-based file-at-rest scheme with honestly-stated
security properties:

    * At rest, only `<db>.enc` exists — authenticated ciphertext (Fernet =
      AES-128-CBC + HMAC-SHA256). The plaintext DB never persists at rest.
    * While the store is OPEN, a plaintext working copy exists under a
      0700 work directory so SQLite can operate on it. It is re-encrypted and
      securely overwritten on close.
    * Residual risk (disclosed, not hidden): if the process is killed without
      close(), a plaintext working copy can linger in the 0700 work dir until
      the next clean open, which removes stale work files. For SQLCipher-grade
      "never plaintext on disk," install pysqlcipher3 and use the SQLCipher
      backend instead.

Key handling: a Fernet key is read from $HOLOGRAPH_KEY if set, else from a
0600 key file (generated on first use). Losing the key means losing the memory
— that is the point of encryption, not a bug.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet


def get_or_create_key(key_path: Path) -> bytes:
    """Return a Fernet key. $HOLOGRAPH_KEY overrides; else read/create a 0600 file."""
    env = os.environ.get("HOLOGRAPH_KEY")
    if env:
        return env.encode() if isinstance(env, str) else env
    key_path = Path(key_path).expanduser()
    if key_path.is_file():
        return key_path.read_bytes().strip()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    # Write with restrictive perms from the start.
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    os.chmod(key_path, 0o600)
    return key


def encrypt_file(plain_path: Path, enc_path: Path, key: bytes) -> None:
    f = Fernet(key)
    data = Path(plain_path).read_bytes()
    token = f.encrypt(data)
    enc_path = Path(enc_path)
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(enc_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token)
    finally:
        os.close(fd)
    os.chmod(enc_path, 0o600)


def decrypt_file(enc_path: Path, plain_path: Path, key: bytes) -> None:
    f = Fernet(key)
    token = Path(enc_path).read_bytes()
    data = f.decrypt(token)
    plain_path = Path(plain_path)
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(plain_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.chmod(plain_path, 0o600)


def secure_delete(path: Path) -> None:
    """Best-effort overwrite-then-unlink of a plaintext working file."""
    path = Path(path)
    if not path.is_file():
        return
    try:
        size = path.stat().st_size
        with open(path, "r+b", buffering=0) as fh:
            fh.write(os.urandom(size))
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass
    try:
        path.unlink()
    except OSError:
        pass


def work_dir(base: Path) -> Path:
    """Return a 0700 working directory for plaintext-while-open files."""
    d = Path(base).expanduser() / ".work"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


__all__ = ["get_or_create_key", "encrypt_file", "decrypt_file", "secure_delete", "work_dir"]
