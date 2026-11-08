"""
config/secrets_manager.py
=========================
Secure Persistent Secret & Key Management for CyberGraph.

Manages:
1. HMAC Signing Secret for device authorization tokens.
2. Admin Access Token for operator authentication.
3. Session Signing Secret for secure HttpOnly cookie verification.

Follows the hierarchy:
- Primary: Environment variables (CYBERGRAPH_HMAC_SECRET, CYBERGRAPH_ADMIN_TOKEN, CYBERGRAPH_SECRET_KEY)
- Fallback: Atomically managed local secret store (.secrets.json) with restricted file permissions.
Never logs or leaks secret bytes.
"""

import os
import json
import secrets
import stat
import logging

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
SECRET_FILE = os.path.join(HERE, ".secrets.json")


def _read_persistent_secrets() -> dict:
    if not os.path.exists(SECRET_FILE):
        return {}
    try:
        with open(SECRET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read persistent secrets file: %s", exc)
        return {}


def _save_persistent_secrets(secrets_dict: dict) -> None:
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = stat.S_IRUSR | stat.S_IWUSR  # 0o600: read/write by owner only
        fd = os.open(SECRET_FILE, flags, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(secrets_dict, f, indent=2)
    except Exception as exc:
        logger.warning("Could not persist secrets to disk: %s", exc)


def get_hmac_secret() -> str:
    """Return persistent HMAC secret key for signed device tokens."""
    env_secret = os.environ.get("CYBERGRAPH_HMAC_SECRET")
    if env_secret:
        return env_secret.strip()

    data = _read_persistent_secrets()
    if data.get("hmac_secret"):
        return data["hmac_secret"]

    new_secret = secrets.token_hex(32)
    data["hmac_secret"] = new_secret
    _save_persistent_secrets(data)
    return new_secret


def get_admin_token() -> str:
    """Return persistent administrative access token."""
    env_token = os.environ.get("CYBERGRAPH_ADMIN_TOKEN")
    if env_token:
        return env_token.strip()

    data = _read_persistent_secrets()
    if data.get("admin_token"):
        return data["admin_token"]

    new_token = secrets.token_hex(24)
    data["admin_token"] = new_token
    _save_persistent_secrets(data)
    return new_token


def get_session_secret() -> str:
    """Return persistent Flask session signing secret."""
    env_sec = os.environ.get("CYBERGRAPH_SECRET_KEY")
    if env_sec:
        return env_sec.strip()

    data = _read_persistent_secrets()
    if data.get("session_secret"):
        return data["session_secret"]

    new_sec = secrets.token_hex(32)
    data["session_secret"] = new_sec
    _save_persistent_secrets(data)
    return new_sec

