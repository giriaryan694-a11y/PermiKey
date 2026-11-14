"""Short-lived signed sessions and same-origin CSRF checks for PermiKey."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
from http.cookies import CookieError, SimpleCookie
from urllib.parse import urlsplit

SESSION_COOKIE_NAME = "PermiKeySession"
SESSION_TTL_SECONDS = 20 * 60
_LOCAL_SECRET = secrets.token_bytes(48)


def get_session_secret() -> bytes | None:
    """Return the configured shared secret, or a process-local secret for local use.

    Vercel must use an environment secret because separate function instances need
    to verify sessions minted by one another. Never use a hard-coded fallback.
    """
    configured = os.environ.get("PERMIKEY_SESSION_SECRET", "")
    if configured:
        secret = configured.encode("utf-8")
        return secret if len(secret) >= 32 else None
    if os.environ.get("VERCEL"):
        return None
    return _LOCAL_SECRET


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64url(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_session(secret: bytes, now: int | None = None) -> tuple[str, str]:
    """Return (signed_cookie_value, csrf_token)."""
    issued = int(time.time() if now is None else now)
    payload = {
        "v": 1,
        "iat": issued,
        "exp": issued + SESSION_TTL_SECONDS,
        "sid": secrets.token_urlsafe(24),
        "csrf": secrets.token_urlsafe(32),
    }
    encoded = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _b64url(hmac.new(secret, encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}", payload["csrf"]


def _cookie_value(cookie_header: str | None) -> str | None:
    if not cookie_header:
        return None
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_header)
    except CookieError:
        return None
    morsel = cookie.get(SESSION_COOKIE_NAME)
    return morsel.value if morsel else None


def verify_session(
    cookie_header: str | None,
    csrf_header: str | None,
    secret: bytes,
    now: int | None = None,
) -> tuple[bool, str | None]:
    """Verify signature, expiry and synchronizer CSRF token; return (ok, error_code)."""
    token = _cookie_value(cookie_header)
    if not token or "." not in token:
        return False, "session_expired"
    encoded, supplied_signature = token.rsplit(".", 1)
    expected_signature = _b64url(
        hmac.new(secret, encoded.encode("ascii", "ignore"), hashlib.sha256).digest()
    )
    if not hmac.compare_digest(supplied_signature, expected_signature):
        return False, "session_expired"
    try:
        payload = json.loads(_unb64url(encoded).decode("utf-8"))
        current = int(time.time() if now is None else now)
        issued = int(payload["iat"])
        expires = int(payload["exp"])
        csrf = str(payload["csrf"])
        if payload.get("v") != 1 or expires <= current or issued > current + 30:
            return False, "session_expired"
        if expires - issued != SESSION_TTL_SECONDS:
            return False, "session_expired"
    except (ValueError, TypeError, KeyError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error):
        return False, "session_expired"
    if not csrf_header or not hmac.compare_digest(csrf, csrf_header):
        return False, "csrf_invalid"
    return True, None


def is_same_origin(origin: str | None, host: str | None, scheme: str) -> bool:
    """Require an explicit Origin matching the request host and scheme."""
    if not origin or not host or origin == "null":
        return False
    try:
        parsed = urlsplit(origin)
        if parsed.scheme.lower() != scheme.lower():
            return False
        if not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
            return False
        if parsed.username or parsed.password:
            return False
        return hmac.compare_digest(parsed.netloc.lower(), host.lower())
    except (ValueError, TypeError):
        return False


def request_scheme(forwarded_proto: str | None, is_vercel: bool = False) -> str:
    proto = (forwarded_proto or "").split(",", 1)[0].strip().lower()
    if proto in {"http", "https"}:
        return proto
    return "https" if is_vercel else "http"


def validate_request_security(
    *,
    cookie_header: str | None,
    csrf_header: str | None,
    origin: str | None,
    host: str | None,
    scheme: str,
    fetch_site: str | None,
    secret: bytes,
) -> tuple[bool, str | None]:
    """Apply browser-origin, Fetch Metadata, session and CSRF validation."""
    if fetch_site and fetch_site.lower() != "same-origin":
        return False, "origin_rejected"
    if not is_same_origin(origin, host, scheme):
        return False, "origin_rejected"
    return verify_session(cookie_header, csrf_header, secret)


def make_session_cookie(value: str, secure: bool) -> str:
    cookie = (
        f"{SESSION_COOKIE_NAME}={value}; Path=/; Max-Age={SESSION_TTL_SECONDS}; "
        "HttpOnly; SameSite=Strict"
    )
    return cookie + ("; Secure" if secure else "")
