import io
import json
import unittest

from app import application
from security import (
    SESSION_TTL_SECONDS,
    get_session_secret,
    issue_session,
    make_session_cookie,
    validate_request_security,
    verify_session,
)


class SessionSecurityTests(unittest.TestCase):
    def setUp(self):
        self.secret = b"unit-test-secret-material-at-least-32-bytes-long"

    def test_signed_session_and_csrf_token_validate(self):
        cookie_value, csrf = issue_session(self.secret, now=1_000)
        cookie = make_session_cookie(cookie_value, secure=True)
        ok, error = verify_session(cookie, csrf, self.secret, now=1_001)
        self.assertTrue(ok)
        self.assertIsNone(error)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertIn("Secure", cookie)
        self.assertIn(f"Max-Age={SESSION_TTL_SECONDS}", cookie)

    def test_missing_or_wrong_csrf_token_is_rejected(self):
        cookie_value, csrf = issue_session(self.secret, now=1_000)
        cookie = make_session_cookie(cookie_value, secure=False)
        self.assertEqual(verify_session(cookie, None, self.secret, now=1_001), (False, "csrf_invalid"))
        self.assertEqual(verify_session(cookie, csrf + "x", self.secret, now=1_001), (False, "csrf_invalid"))

    def test_expired_session_is_rejected(self):
        cookie_value, csrf = issue_session(self.secret, now=1_000)
        cookie = make_session_cookie(cookie_value, secure=False)
        self.assertEqual(
            verify_session(cookie, csrf, self.secret, now=1_000 + SESSION_TTL_SECONDS),
            (False, "session_expired"),
        )

    def test_tampered_signature_is_rejected(self):
        cookie_value, csrf = issue_session(self.secret, now=1_000)
        changed = cookie_value[:-1] + ("A" if cookie_value[-1] != "A" else "B")
        cookie = make_session_cookie(changed, secure=False)
        self.assertEqual(verify_session(cookie, csrf, self.secret, now=1_001), (False, "session_expired"))

    def test_cross_origin_and_cross_site_requests_are_rejected(self):
        cookie_value, csrf = issue_session(self.secret, now=1_000)
        cookie = make_session_cookie(cookie_value, secure=False)
        common = dict(
            cookie_header=cookie,
            csrf_header=csrf,
            host="permi-key.vercel.app",
            scheme="https",
            secret=self.secret,
        )
        self.assertEqual(
            validate_request_security(origin="https://attacker.example", fetch_site="cross-site", **common),
            (False, "origin_rejected"),
        )
        self.assertEqual(
            validate_request_security(origin="https://permi-key.vercel.app", fetch_site="same-site", **common),
            (False, "origin_rejected"),
        )

    def test_missing_origin_is_rejected(self):
        cookie_value, csrf = issue_session(self.secret, now=1_000)
        cookie = make_session_cookie(cookie_value, secure=False)
        self.assertEqual(
            validate_request_security(
                cookie_header=cookie,
                csrf_header=csrf,
                origin=None,
                host="localhost:8000",
                scheme="http",
                fetch_site="same-origin",
                secret=self.secret,
            ),
            (False, "origin_rejected"),
        )


class LocalApplicationSecurityTests(unittest.TestCase):
    def _call_app(self, path, method="GET", body=b"", extra=None):
        environ = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "QUERY_STRING": "",
            "CONTENT_LENGTH": str(len(body)),
            "CONTENT_TYPE": "application/json",
            "wsgi.input": io.BytesIO(body),
            "wsgi.url_scheme": "http",
            "HTTP_HOST": "127.0.0.1:8000",
            "SERVER_NAME": "127.0.0.1",
            "SERVER_PORT": "8000",
        }
        environ.update(extra or {})
        captured = {}

        def start_response(status, headers):
            captured["status"] = status
            captured["headers"] = dict(headers)

        response = b"".join(application(environ, start_response))
        return captured, json.loads(response.decode("utf-8"))

    def test_session_endpoint_issues_http_only_cookie_and_csrf_token(self):
        captured, payload = self._call_app("/api/session")
        self.assertTrue(captured["status"].startswith("200"))
        self.assertTrue(payload["csrf_token"])
        cookie = captured["headers"]["Set-Cookie"]
        self.assertIn("PermiKeySession=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertNotIn("Secure", cookie)  # Local HTTP only.

    def test_validation_without_session_is_blocked(self):
        captured, payload = self._call_app(
            "/api/validate",
            method="POST",
            body=b'{"provider":"not-a-provider"}',
            extra={"HTTP_ORIGIN": "http://127.0.0.1:8000", "HTTP_SEC_FETCH_SITE": "same-origin"},
        )
        self.assertTrue(captured["status"].startswith("403"))
        self.assertEqual(payload["code"], "session_expired")

    def test_validation_with_session_reaches_provider_validation(self):
        secret = get_session_secret()
        cookie_value, csrf = issue_session(secret)
        cookie = make_session_cookie(cookie_value, secure=False)
        captured, payload = self._call_app(
            "/api/validate",
            method="POST",
            body=b'{"provider":"not-a-provider"}',
            extra={
                "HTTP_COOKIE": cookie,
                "HTTP_X_PERMIKEY_CSRF": csrf,
                "HTTP_ORIGIN": "http://127.0.0.1:8000",
                "HTTP_SEC_FETCH_SITE": "same-origin",
            },
        )
        self.assertTrue(captured["status"].startswith("400"))
        self.assertIn("Unsupported", payload["error"])

    def test_honeypot_submission_is_rejected(self):
        secret = get_session_secret()
        cookie_value, csrf = issue_session(secret)
        cookie = make_session_cookie(cookie_value, secure=False)
        captured, payload = self._call_app(
            "/api/validate",
            method="POST",
            body=b'{"provider":"not-a-provider","website":"bot-filled-this"}',
            extra={
                "HTTP_COOKIE": cookie,
                "HTTP_X_PERMIKEY_CSRF": csrf,
                "HTTP_ORIGIN": "http://127.0.0.1:8000",
                "HTTP_SEC_FETCH_SITE": "same-origin",
            },
        )
        self.assertTrue(captured["status"].startswith("400"))
        self.assertEqual(payload["error"], "Request rejected.")


if __name__ == "__main__":
    unittest.main()
