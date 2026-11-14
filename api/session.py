"""Vercel Python Function: GET /api/session."""
from http.server import BaseHTTPRequestHandler
import json
import os

from security import SESSION_TTL_SECONDS, get_session_secret, issue_session, make_session_cookie, request_scheme, is_same_origin


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

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/api/session":
            self._send(404, {"error": "Not found."})
            return
        fetch_site = self.headers.get("Sec-Fetch-Site")
        if fetch_site and fetch_site.lower() not in {"same-origin", "none"}:
            self._send(403, {"error": "Session requests must come from this site.", "code": "origin_rejected"})
            return
        scheme = request_scheme(self.headers.get("X-Forwarded-Proto"), bool(os.environ.get("VERCEL")))
        origin = self.headers.get("Origin")
        if origin and not is_same_origin(origin, self.headers.get("Host"), scheme):
            self._send(403, {"error": "Session requests must come from this site.", "code": "origin_rejected"})
            return
        secret = get_session_secret()
        if secret is None:
            self._send(503, {"error": "Server protection is not configured. Set PERMIKEY_SESSION_SECRET to a random value of at least 32 characters."})
            return
        cookie_value, csrf_token = issue_session(secret)
        self._send(200, {
            "session": "active",
            "csrf_token": csrf_token,
            "expires_in": SESSION_TTL_SECONDS,
        }, [("Set-Cookie", make_session_cookie(cookie_value, scheme == "https"))])

    def do_POST(self):
        self._send(405, {"error": "Method not allowed."})

    def do_OPTIONS(self):
        self._send(405, {"error": "Method not allowed."})
