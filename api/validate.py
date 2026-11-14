"""Vercel Python Function: POST /api/validate"""
from http.server import BaseHTTPRequestHandler
import json

from checker import validate_credentials


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(405, {"error": "Method not allowed."})

    def do_GET(self):
        self._send(405, {"error": "Use POST for credential validation."})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/validate":
            self._send(404, {"error": "Not found."})
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
        status, response = validate_credentials(data)
        self._send(status, response)
