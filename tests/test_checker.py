import json
import unittest
from unittest.mock import patch

import checker


class CheckerInputTests(unittest.TestCase):
    def test_rejects_unknown_provider(self):
        status, body = checker.validate_credentials({"provider": "unknown", "token": "x"})
        self.assertEqual(status, 400)
        self.assertIn("Unsupported", body["error"])

    def test_requires_token(self):
        status, body = checker.validate_credentials({"provider": "openai", "token": "  "})
        self.assertEqual(status, 400)
        self.assertIn("token", body["error"])

    def test_rejects_overlong_key(self):
        status, body = checker.validate_credentials({"provider": "openai", "token": "x" * 8193})
        self.assertEqual(status, 400)
        self.assertIn("too long", body["error"])

    def test_successful_models_response(self):
        payload = {"object": "list", "data": [{"id": "example-model"}]}
        with patch.object(checker, "_json_request", return_value=(200, {}, payload)):
            status, result = checker.validate_credentials({"provider": "openai", "token": "dummy-test-value"})
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["metadata"]["model_count"], 1)
        self.assertNotIn("dummy-test-value", json.dumps(result))

    def test_403_is_not_labelled_invalid(self):
        with patch.object(checker, "_json_request", return_value=(403, {}, {"message": "forbidden"})):
            result = checker.validate_cloudflare("dummy-test-value")
        self.assertEqual(result["status"], "restricted")

    def test_github_fine_grained_scope_not_guessed(self):
        with patch.object(checker, "_json_request", return_value=(200, {}, {"login": "demo", "type": "User"})):
            result = checker.validate_github("dummy-test-value")
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["checks"][1]["status"], "unknown")

    def test_provider_response_is_redacted(self):
        with patch.object(checker, "_json_request", return_value=(403, {}, {"message": "credential dummy-test-value denied"})):
            status, result = checker.validate_credentials({"provider": "cloudflare", "token": "dummy-test-value"})
        self.assertEqual(status, 200)
        self.assertNotIn("dummy-test-value", json.dumps(result))
        self.assertIn("[redacted]", json.dumps(result))

    def test_vercel_adapter_is_registered(self):
        self.assertIn("vercel", checker.PROVIDERS)


class CloudProviderTests(unittest.TestCase):
    def test_cloudflare_ai_gateway_checks_account_scoped_list(self):
        payload = {"success": True, "result": [{"id": "gateway-one", "name": "prod"}]}
        with patch.object(checker, "_json_request", return_value=(200, {}, payload)) as request:
            status, result = checker.validate_credentials({
                "provider": "cloudflare_ai_gateway",
                "token": "dummy-cloudflare-token",
                "account_id": "a" * 32,
            })
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["metadata"]["gateway_count"], 1)
        args, kwargs = request.call_args
        self.assertIn("/ai-gateway/gateways", args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer dummy-cloudflare-token")
        self.assertNotIn("dummy-cloudflare-token", json.dumps(result))

    def test_cloudflare_workers_checks_script_listing(self):
        payload = {"success": True, "result": [{"id": "edge-api"}]}
        with patch.object(checker, "_json_request", return_value=(200, {}, payload)) as request:
            result = checker.validate_cloudflare_workers("token", "b" * 32)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["metadata"]["worker_script_count"], 1)
        self.assertIn("/workers/scripts", request.call_args.args[0])

    def test_gcp_project_uses_resource_manager_permission_test(self):
        payload = {"permissions": ["resourcemanager.projects.get", "storage.buckets.list"]}
        with patch.object(checker, "_json_request", return_value=(200, {}, payload)) as request:
            status, result = checker.validate_credentials({
                "provider": "gcp", "token": "dummy-google-token", "project_id": "demo-project-123"
            })
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["metadata"]["granted_count"], 2)
        args, kwargs = request.call_args
        self.assertIn("cloudresourcemanager.googleapis.com/v1/projects/demo-project-123:testIamPermissions", args[0])
        self.assertEqual(kwargs["method"], "POST")
        self.assertIn("resourcemanager.projects.get", kwargs["body"].decode())
        self.assertNotIn("dummy-google-token", json.dumps(result))

    def test_gcp_storage_uses_get_test_iam_permissions(self):
        payload = {"permissions": ["storage.buckets.get", "storage.objects.list"]}
        with patch.object(checker, "_json_request", return_value=(200, {}, payload)) as request:
            result = checker.validate_gcp_storage("token", "demo-bucket-123")
        self.assertEqual(result["status"], "valid")
        args, kwargs = request.call_args
        self.assertIn("storage.googleapis.com/storage/v1/b/demo-bucket-123/iam/testPermissions", args[0])
        self.assertEqual(kwargs["method"], "GET")
        self.assertIn("permissions=storage.buckets.get", args[0])
        self.assertIsNone(kwargs["body"])

    def test_aws_s3_uses_zero_key_list_permission_probe(self):
        identity = checker._result("AWS STS", "valid", "ok", [], {"account": "123456789012", "arn": "arn:aws:iam::123456789012:user/demo"})
        with patch.object(checker, "validate_aws", return_value=identity), patch.object(
            checker, "_request", return_value=(200, {"x-amz-bucket-region": "us-east-1"}, b"<ListBucketResult><KeyCount>0</KeyCount></ListBucketResult>")
        ) as request:
            status, result = checker.validate_credentials({
                "provider": "aws_s3",
                "access_key": "AKIAEXAMPLE",
                "secret_key": "dummy-secret-value",
                "bucket_name": "demo-bucket-123",
                "region": "us-east-1",
            })
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["checks"][1]["status"], "confirmed")
        args, kwargs = request.call_args
        self.assertIn("max-keys=0", args[0])
        self.assertIn("list-type=2", args[0])
        self.assertEqual(kwargs["method"], "GET")
        self.assertNotIn("dummy-secret-value", json.dumps(result))

    def test_new_providers_require_resource_identifiers(self):
        for data, required in [
            ({"provider": "cloudflare_workers", "token": "x"}, "account_id"),
            ({"provider": "gcp", "token": "x"}, "project_id"),
            ({"provider": "gcp_storage", "token": "x"}, "bucket_name"),
            ({"provider": "aws_s3", "access_key": "x", "secret_key": "y"}, "bucket_name"),
        ]:
            status, response = checker.validate_credentials(data)
            self.assertEqual(status, 400)
            self.assertIn(required, response["error"])

    def test_new_provider_ids_are_registered(self):
        for provider in ("cloudflare_ai_gateway", "cloudflare_workers", "gcp", "gcp_storage", "aws_s3"):
            self.assertIn(provider, checker.PROVIDERS)


if __name__ == "__main__":
    unittest.main()
