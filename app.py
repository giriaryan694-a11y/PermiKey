"""Run PermiKey locally with: python app.py"""
from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from wsgiref.simple_server import make_server

from checker import validate_credentials
from security import (
    SESSION_TTL_SECONDS,
    get_session_secret,
    issue_session,
    make_session_cookie,
    request_scheme,
    validate_request_security,
    is_same_origin,
)

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
MAX_BODY = 12000


def _json_response(start_response, status: int, payload: dict, extra_headers=None):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store, max-age=0"),
        ("Pragma", "no-cache"),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
        ("X-Frame-Options", "DENY"),
        ("Vary", "Cookie, Origin"),
    ]
    headers.extend(extra_headers or [])
    phrase = {
        200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found",
        405: "Method Not Allowed", 413: "Payload Too Large", 415: "Unsupported Media Type",
        503: "Service Unavailable",
    }.get(status, "Error")
    start_response(f"{status} {phrase}", headers)
    return [body]


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")
    host = environ.get("HTTP_HOST") or environ.get("SERVER_NAME", "")
    is_vercel = bool(os.environ.get("VERCEL"))
    scheme = request_scheme(environ.get("HTTP_X_FORWARDED_PROTO") or environ.get("wsgi.url_scheme"), is_vercel)
    fetch_site = environ.get("HTTP_SEC_FETCH_SITE")
    origin = environ.get("HTTP_ORIGIN")

    if path == "/api/session":
        if method != "GET":
            return _json_response(start_response, 405, {"error": "Use GET to start a temporary session."})
        if fetch_site and fetch_site.lower() not in {"same-origin", "none"}:
            return _json_response(start_response, 403, {"error": "Session requests must come from this site.", "code": "origin_rejected"})
        if origin and not is_same_origin(origin, host, scheme):
            return _json_response(start_response, 403, {"error": "Session requests must come from this site.", "code": "origin_rejected"})
        secret = get_session_secret()
        if secret is None:
            return _json_response(start_response, 503, {"error": "Server protection is not configured. Set PERMIKEY_SESSION_SECRET to a random value of at least 32 characters."})
        cookie_value, csrf_token = issue_session(secret)
        cookie = make_session_cookie(cookie_value, scheme == "https")
        return _json_response(start_response, 200, {
            "session": "active", "csrf_token": csrf_token, "expires_in": SESSION_TTL_SECONDS,
        }, [("Set-Cookie", cookie)])

    if path == "/api/validate":
        if method != "POST":
            return _json_response(start_response, 405, {"error": "Use POST for credential validation."})
        secret = get_session_secret()
        if secret is None:
            return _json_response(start_response, 503, {"error": "Server protection is not configured. Set PERMIKEY_SESSION_SECRET to a random value of at least 32 characters."})
        secure_ok, security_error = validate_request_security(
            cookie_header=environ.get("HTTP_COOKIE"),
            csrf_header=environ.get("HTTP_X_PERMIKEY_CSRF"),
            origin=origin,
            host=host,
            scheme=scheme,
            fetch_site=fetch_site,
            secret=secret,
        )
        if not secure_ok:
            messages = {
                "session_expired": "Your temporary session expired. Refresh the page and try again.",
                "csrf_invalid": "The request security token is missing or invalid. Refresh the page and try again.",
                "origin_rejected": "This request did not originate from PermiKey.",
            }
            return _json_response(start_response, 403, {
                "error": messages.get(security_error, "Request security check failed."),
                "code": security_error,
            })
        if "application/json" not in environ.get("CONTENT_TYPE", "").lower():
            return _json_response(start_response, 415, {"error": "Content-Type must be application/json."})
        try:
            length = int(environ.get("CONTENT_LENGTH") or "0")
        except ValueError:
            length = 0
        if length < 1 or length > MAX_BODY:
            return _json_response(start_response, 413, {"error": "Request body must be between 1 and 12000 bytes."})
        try:
            data = json.loads(environ["wsgi.input"].read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _json_response(start_response, 400, {"error": "Request body must contain valid UTF-8 JSON."})
        if not isinstance(data, dict):
            return _json_response(start_response, 400, {"error": "Request body must be a JSON object."})
        if str(data.get("website", "")).strip():
            return _json_response(start_response, 400, {"error": "Request rejected."})
        status, payload = validate_credentials(data)
        return _json_response(start_response, status, payload)

    if path == "/" or path == "/index.html":
        file_path = INDEX
    else:
        body = b"Not found"
        start_response("404 Not Found", [
            ("Content-Type", "text/plain; charset=utf-8"),
            ("X-Content-Type-Options", "nosniff"),
            ("Content-Length", str(len(body))),
        ])
        return [body]

    content = file_path.read_bytes()
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    headers = [
        ("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else "")),
        ("Content-Length", str(len(content))),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
        ("Cache-Control", "no-store, max-age=0"),
        ("X-Frame-Options", "DENY"),
    ]
    start_response("200 OK", headers)
    return [content]


if __name__ == "__main__":
    print("PermiKey local server: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop. Request bodies and credentials are not logged.")
    with make_server("127.0.0.1", 8000, application) as server:
        server.serve_forever()
