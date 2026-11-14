"""Vercel Python Functions: GET /api/session and POST /api/validate."""
from http.server import BaseHTTPRequestHandler
import json
import os

from checker import validate_credentials
from security import (
    SESSION_TTL_SECONDS,
    get_session_secret,
    issue_session,
    make_session_cookie,
    request_scheme,
    validate_request_security,
)


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict, extra_headers: list[tuple[str, str]] | None = None) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Vary", "Cookie, Origin")
        for name, value in extra_headers or []:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _secret(self) -> bytes | None:
        return get_session_secret()

    def _scheme(self) -> str:
        return request_scheme(self.headers.get("X-Forwarded-Proto"), bool(os.environ.get("VERCEL")))

    def do_OPTIONS(self):
        self._send(405, {"error": "Method not allowed."})

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route != "/api/session":
            self._send(405, {"error": "Use GET /api/session to start a temporary session."})
            return

        fetch_site = self.headers.get("Sec-Fetch-Site")
        origin = self.headers.get("Origin")
        if fetch_site and fetch_site.lower() not in {"same-origin", "none"}:
            self._send(403, {"error": "Session requests must come from this site.", "code": "origin_rejected"})
            return
        if origin and not self._same_origin(origin):
            self._send(403, {"error": "Session requests must come from this site.", "code": "origin_rejected"})
            return

        secret = self._secret()
        if secret is None:
            self._send(503, {"error": "Server protection is not configured. Set PERMIKEY_SESSION_SECRET to a random value of at least 32 characters."})
            return
        cookie_value, csrf_token = issue_session(secret)
        secure = self._scheme() == "https"
        self._send(200, {
            "session": "active",
            "csrf_token": csrf_token,
            "expires_in": SESSION_TTL_SECONDS,
        }, [("Set-Cookie", make_session_cookie(cookie_value, secure))])

    def _same_origin(self, origin: str) -> bool:
        from security import is_same_origin
        return is_same_origin(origin, self.headers.get("Host"), self._scheme())

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/validate":
            self._send(404, {"error": "Not found."})
            return

        secret = self._secret()
        if secret is None:
            self._send(503, {"error": "Server protection is not configured. Set PERMIKEY_SESSION_SECRET to a random value of at least 32 characters."})
            return

        secure_ok, security_error = validate_request_security(
            cookie_header=self.headers.get("Cookie"),
            csrf_header=self.headers.get("X-PermiKey-CSRF"),
            origin=self.headers.get("Origin"),
            host=self.headers.get("Host"),
            scheme=self._scheme(),
            fetch_site=self.headers.get("Sec-Fetch-Site"),
            secret=secret,
        )
        if not secure_ok:
            messages = {
                "session_expired": "Your temporary session expired. Refresh the page and try again.",
                "csrf_invalid": "The request security token is missing or invalid. Refresh the page and try again.",
                "origin_rejected": "This request did not originate from PermiKey.",
            }
            self._send(403, {"error": messages.get(security_error, "Request security check failed."), "code": security_error})
            return

        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            self._send(415, {"error": "Content-Type must be application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "Invalid Content-Length."})
            return
        if length < 1 or length > 12000:
            self._send(413, {"error": "Request body must be between 1 and 12000 bytes."})
            return
        try:
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"error": "Request body must contain valid UTF-8 JSON."})
            return
        if not isinstance(data, dict):
            self._send(400, {"error": "Request body must be a JSON object."})
            return
        if str(data.get("website", "")).strip():
            self._send(400, {"error": "Request rejected."})
            return
        status, response = validate_credentials(data)
        self._send(status, response)
