"""Run PermiKey locally with: python app.py"""
from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from wsgiref.simple_server import make_server

from checker import validate_credentials

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
MAX_BODY = 12000


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")

    if path == "/api/validate":
        if method != "POST":
            body = json.dumps({"error": "Use POST for credential validation."}).encode()
            start_response("405 Method Not Allowed", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Cache-Control", "no-store"),
                ("Content-Length", str(len(body))),
            ])
            return [body]
        if "application/json" not in environ.get("CONTENT_TYPE", "").lower():
            body = json.dumps({"error": "Content-Type must be application/json."}).encode()
            start_response("415 Unsupported Media Type", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Cache-Control", "no-store"),
                ("Content-Length", str(len(body))),
            ])
            return [body]
        try:
            length = int(environ.get("CONTENT_LENGTH") or "0")
        except ValueError:
            length = 0
        if length < 1 or length > MAX_BODY:
            body = json.dumps({"error": "Request body must be between 1 and 12000 bytes."}).encode()
            start_response("413 Payload Too Large", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Cache-Control", "no-store"),
                ("Content-Length", str(len(body))),
            ])
            return [body]
        try:
            data = json.loads(environ["wsgi.input"].read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = json.dumps({"error": "Request body must contain valid UTF-8 JSON."}).encode()
            start_response("400 Bad Request", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Cache-Control", "no-store"),
                ("Content-Length", str(len(body))),
            ])
            return [body]
        status, payload = validate_credentials(data)
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        phrase = "OK" if status == 200 else "Bad Request"
        start_response(f"{status} {phrase}", [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Cache-Control", "no-store, max-age=0"),
            ("Pragma", "no-cache"),
            ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"),
            ("Content-Length", str(len(body))),
        ])
        return [body]

    if path == "/" or path == "/index.html":
        file_path = INDEX
    else:
        # Serve only the app shell. Never expose Python source, tests, or project files.
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
