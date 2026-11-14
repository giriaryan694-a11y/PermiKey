"""Stateless, provider-specific credential checks for PermiKey.

Only fixed provider URLs are contacted. Submitted credentials are never logged,
returned, written to disk, or retained between requests.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

TIMEOUT_SECONDS = 9
MAX_RESPONSE_BYTES = 1_500_000
USER_AGENT = "PermiKey/1.0 (+https://github.com/giriaryan694-a11y)"


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Make a bounded HTTPS request to a hard-coded provider endpoint."""
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS provider endpoints are allowed.")
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise ValueError("Provider response exceeded the size limit.")
            return response.status, {k.lower(): v for k, v in response.headers.items()}, data
    except urllib.error.HTTPError as exc:
        data = exc.read(MAX_RESPONSE_BYTES)
        return exc.code, {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}, data


def _json_request(url: str, **kwargs: Any) -> tuple[int, dict[str, str], dict[str, Any] | list[Any] | None]:
    status, headers, raw = _request(url, **kwargs)
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    return status, headers, payload


def _clean_error(payload: Any) -> str:
    """Return a short provider error without ever echoing request credentials."""
    if isinstance(payload, dict):
        error = payload.get("error") or payload.get("message") or payload.get("errors")
        if isinstance(error, dict):
            error = error.get("message") or error.get("type") or "Provider rejected the request."
        if isinstance(error, list):
            error = "; ".join(str(x) for x in error[:2])
        if isinstance(error, str):
            return re.sub(r"\s+", " ", error)[:220]
    return "The provider rejected the request or returned an unexpected response."


def _result(
    provider: str,
    status: str,
    summary: str,
    checks: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "status": status,
        "summary": summary,
        "checks": checks,
        "metadata": metadata or {},
    }


def _classify_http(provider: str, status: int, payload: Any, headers: dict[str, str]) -> dict[str, Any]:
    message = _clean_error(payload)
    if status == 401:
        return _result(provider, "invalid", "The provider did not accept this credential.", [
            {"name": "Authentication", "status": "denied", "detail": "HTTP 401: unauthorized."}
        ])
    if status == 403:
        return _result(provider, "restricted", "The provider denied this endpoint request. The response alone may not distinguish invalid credentials from missing access or an organization policy.", [
            {"name": "Authentication / endpoint access", "status": "denied", "detail": "HTTP 403. Check the provider message and the selected endpoint's permission requirements."}
        ], {"provider_message": message})
    if status == 498:
        return _result(provider, "invalid", "The provider reported an invalid token.", [
            {"name": "Credential authentication", "status": "denied", "detail": "HTTP 498: invalid token."}
        ])
    if status == 404:
        return _result(provider, "inconclusive", "The provider endpoint was not found for this API or credential type.", [
            {"name": "Endpoint check", "status": "unknown", "detail": "HTTP 404. Authentication could not be established by this endpoint."}
        ])
    if status == 429:
        return _result(provider, "rate_limited", "The provider rate-limited the check. No conclusion about the credential can be made.", [
            {"name": "Rate limit", "status": "unknown", "detail": "HTTP 429. Try again after the provider's retry window."}
        ])
    if status >= 500:
        return _result(provider, "provider_error", "The provider is temporarily unavailable.", [
            {"name": "Provider availability", "status": "unknown", "detail": f"HTTP {status}."}
        ])
    return _result(provider, "error", "The provider returned an unexpected response.", [
        {"name": "Provider response", "status": "unknown", "detail": f"HTTP {status}: {message}"}
    ])


def _bearer_models(provider: str, url: str, token: str, *, extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    if extra_headers:
        headers.update(extra_headers)
    status, response_headers, payload = _json_request(url, headers=headers)
    if status != 200:
        return _classify_http(provider, status, payload, response_headers)
    if not isinstance(payload, dict):
        return _result(provider, "inconclusive", "The provider response was not valid JSON.", [
            {"name": "Authentication", "status": "unknown", "detail": "The response could not be interpreted."}
        ])
    data = payload.get("data")
    models = [item.get("id") for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)] if isinstance(data, list) else []
    checks = [
        {"name": "Credential authentication", "status": "confirmed", "detail": "The provider accepted the credential."},
        {"name": "Model catalogue access", "status": "confirmed", "detail": f"The authenticated endpoint returned {len(models)} model entries."},
        {"name": "Model inference permission", "status": "unknown", "detail": "Not run by default. Enable the optional minimal inference test to check one generation request."},
    ]
    return _result(provider, "valid", "Credential accepted. Model catalogue access confirmed; inference permissions were not tested.", checks, {
        "model_count": len(models),
        "sample_models": models[:6],
    })


def validate_openai(token: str) -> dict[str, Any]:
    return _bearer_models("OpenAI", "https://api.openai.com/v1/models", token)


def validate_nvidia(token: str) -> dict[str, Any]:
    return _bearer_models("NVIDIA NIM", "https://integrate.api.nvidia.com/v1/models", token)


def validate_groq(token: str) -> dict[str, Any]:
    return _bearer_models("Groq", "https://api.groq.com/openai/v1/models", token)


def validate_mistral(token: str) -> dict[str, Any]:
    return _bearer_models("Mistral", "https://api.mistral.ai/v1/models", token)


def validate_cohere(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://api.cohere.com/v1/models",
        headers={"Authorization": f"Bearer {token}"},
    )
    if status != 200:
        return _classify_http("Cohere", status, payload, headers)
    models = payload.get("models", []) if isinstance(payload, dict) else []
    model_names = [m.get("name") for m in models if isinstance(m, dict) and isinstance(m.get("name"), str)] if isinstance(models, list) else []
    return _result("Cohere", "valid", "Credential accepted. Accessible model catalogue returned.", [
        {"name": "Credential authentication", "status": "confirmed", "detail": "The provider accepted the credential."},
        {"name": "Model catalogue access", "status": "confirmed", "detail": f"The endpoint returned {len(model_names)} model entries."},
        {"name": "Model inference permission", "status": "unknown", "detail": "Not run by default. Enable the optional minimal inference test to check one generation request."},
    ], {"model_count": len(model_names), "sample_models": model_names[:6]})


def validate_anthropic(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": token, "anthropic-version": "2023-06-01"},
    )
    if status != 200:
        return _classify_http("Anthropic", status, payload, headers)
    data = payload.get("data", []) if isinstance(payload, dict) else []
    models = [m.get("id") for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)] if isinstance(data, list) else []
    return _result("Anthropic", "valid", "Credential accepted. Model catalogue access confirmed; message-generation permissions were not tested.", [
        {"name": "Credential authentication", "status": "confirmed", "detail": "The provider accepted the credential."},
        {"name": "Model catalogue access", "status": "confirmed", "detail": f"The endpoint returned {len(models)} model entries."},
        {"name": "Message generation permission", "status": "unknown", "detail": "Not run by default. Enable the optional minimal inference test to check one message request."},
    ], {"model_count": len(models), "sample_models": models[:6]})


def validate_gemini(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=50",
        headers={"x-goog-api-key": token},
    )
    if status != 200:
        return _classify_http("Google Gemini", status, payload, headers)
    models = payload.get("models", []) if isinstance(payload, dict) else []
    names: list[str] = []
    methods: set[str] = set()
    for model in models if isinstance(models, list) else []:
        if isinstance(model, dict):
            name = model.get("name")
            if isinstance(name, str):
                names.append(name.removeprefix("models/"))
            supported = model.get("supportedGenerationMethods", [])
            if isinstance(supported, list):
                methods.update(str(x) for x in supported if isinstance(x, str))
    return _result("Google Gemini", "valid", "API key accepted. The model catalogue and provider-advertised methods were returned.", [
        {"name": "API key authentication", "status": "confirmed", "detail": "Google accepted the x-goog-api-key header."},
        {"name": "Model catalogue access", "status": "confirmed", "detail": f"{len(names)} model entries returned in the first page."},
        {"name": "Generation method availability", "status": "confirmed" if "generateContent" in methods else "unknown", "detail": "The catalogue advertises generateContent for at least one returned model." if "generateContent" in methods else "No generateContent method was visible in this page."},
        {"name": "Project-level API restrictions", "status": "unknown", "detail": "Not exposed by this API response; key restrictions may also be configured in Google Cloud."},
    ], {"model_count": len(names), "sample_models": names[:6], "advertised_methods": sorted(methods)[:10]})


def validate_openrouter(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {token}"},
    )
    if status != 200:
        return _classify_http("OpenRouter", status, payload, headers)
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        data = {}
    metadata: dict[str, Any] = {}
    scalar_fields = (
        "label", "limit", "limit_remaining", "limit_reset", "usage", "usage_daily",
        "usage_weekly", "usage_monthly", "is_free_tier", "include_byok_in_limit",
    )
    for key in scalar_fields:
        value = data.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if key in data:
                metadata[key] = value
    budget_fields = ("scope", "limit_usd", "spend_usd", "remaining_usd", "reset_interval", "resets_at", "include_byok")
    budgets = data.get("budgets")
    if isinstance(budgets, list):
        metadata["budgets"] = [
            {k: budget.get(k) for k in budget_fields if k in budget}
            for budget in budgets[:12] if isinstance(budget, dict)
        ]
    effective = data.get("effective_budget")
    if isinstance(effective, dict):
        metadata["effective_budget"] = {k: effective.get(k) for k in budget_fields if k in effective}
    free_usage = data.get("free_model_daily_requests")
    if isinstance(free_usage, dict):
        metadata["free_model_daily_requests"] = {
            k: free_usage.get(k) for k in ("used", "limit", "remaining") if k in free_usage
        }
    return _result("OpenRouter", "valid", "Credential accepted. Key metadata and credit-budget fields are accessible.", [
        {"name": "Credential authentication", "status": "confirmed", "detail": "The provider accepted the credential."},
        {"name": "Key metadata access", "status": "confirmed", "detail": "The key information endpoint returned metadata."},
        {"name": "Model request permission", "status": "unknown", "detail": "Not run by default. Enable the optional minimal inference test to check one model request."},
    ], metadata)


def validate_huggingface(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
    )
    if status != 200:
        return _classify_http("Hugging Face", status, payload, headers)
    name = payload.get("name") if isinstance(payload, dict) else None
    return _result("Hugging Face", "valid", "Token accepted by the Hub identity endpoint.", [
        {"name": "Token authentication", "status": "confirmed", "detail": "The Hub accepted the token."},
        {"name": "Fine-grained resource permissions", "status": "unknown", "detail": "This identity response is not a complete permission audit. Repository and organisation access can vary by resource."},
    ], {"account_name": name if isinstance(name, str) else None})


def validate_github(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"},
    )
    if status != 200:
        return _classify_http("GitHub", status, payload, headers)
    classic_scopes = headers.get("x-oauth-scopes")
    checks = [
        {"name": "Credential authentication", "status": "confirmed", "detail": "GitHub accepted the token at GET /user."},
    ]
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        if isinstance(payload.get("login"), str):
            metadata["login"] = payload["login"]
        if isinstance(payload.get("type"), str):
            metadata["account_type"] = payload["type"]
    if classic_scopes is not None:
        scopes = [s.strip() for s in classic_scopes.split(",") if s.strip()]
        metadata["classic_token_scopes"] = scopes
        checks.append({"name": "Classic-token scopes", "status": "confirmed", "detail": ", ".join(scopes) if scopes else "No classic OAuth scopes reported."})
    else:
        checks.append({"name": "Fine-grained token permissions", "status": "unknown", "detail": "GitHub did not return classic OAuth scopes. Fine-grained grants are not reliably exposed by this identity endpoint."})
    checks.append({"name": "Repository read/write access", "status": "unknown", "detail": "Not probed. Access varies by repository and selected permission level."})
    return _result("GitHub", "valid", "Token accepted. Scope visibility depends on token type.", checks, metadata)


def validate_codeberg(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://codeberg.org/api/v1/user",
        headers={"Authorization": f"token {token}"},
    )
    if status != 200:
        return _classify_http("Codeberg", status, payload, headers)
    metadata = {}
    if isinstance(payload, dict):
        for key in ("login", "username"):
            if isinstance(payload.get(key), str):
                metadata["account"] = payload[key]
                break
    return _result("Codeberg", "valid", "Token accepted by the Codeberg / Forgejo user endpoint.", [
        {"name": "Credential authentication", "status": "confirmed", "detail": "The Codeberg API returned the authenticated user."},
        {"name": "Token scopes", "status": "unknown", "detail": "The user endpoint does not return the token's complete granted scopes."},
        {"name": "Repository access", "status": "unknown", "detail": "Not probed. Access can differ for each repository."},
    ], metadata)


def validate_gitlab(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://gitlab.com/api/v4/personal_access_tokens/self",
        headers={"PRIVATE-TOKEN": token},
    )
    if status != 200:
        return _classify_http("GitLab", status, payload, headers)
    if not isinstance(payload, dict):
        payload = {}
    scopes = payload.get("scopes") if isinstance(payload.get("scopes"), list) else []
    granular = payload.get("granular_scopes") if isinstance(payload.get("granular_scopes"), list) else []
    metadata = {
        "token_name": payload.get("name") if isinstance(payload.get("name"), str) else None,
        "active": payload.get("active") if isinstance(payload.get("active"), bool) else None,
        "expires_at": payload.get("expires_at") if isinstance(payload.get("expires_at"), str) else None,
        "scopes": [str(s) for s in scopes[:30]],
        "granular_scopes": granular[:20],
    }
    detail = ", ".join(str(s) for s in scopes) if scopes else "No standard scopes were reported."
    if granular:
        detail += f"; {len(granular)} granular scope entries returned."
    active = payload.get("active") is True
    return _result("GitLab", "valid" if active else "restricted", "GitLab returned the personal access token's self-information.", [
        {"name": "Personal access token", "status": "confirmed", "detail": "The token authenticated to the PAT self endpoint."},
        {"name": "Token state", "status": "confirmed" if active else "denied", "detail": "Active." if active else "GitLab reports this token is not active."},
        {"name": "Granted scopes", "status": "confirmed" if scopes or granular else "unknown", "detail": detail},
    ], metadata)


def validate_npm(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://registry.npmjs.org/-/whoami",
        headers={"Authorization": f"Bearer {token}"},
    )
    if status != 200:
        return _classify_http("npm Registry", status, payload, headers)
    username = payload.get("username") if isinstance(payload, dict) else None
    return _result("npm Registry", "valid", "Token accepted by the npm registry identity endpoint.", [
        {"name": "Registry authentication", "status": "confirmed", "detail": "The registry returned the authenticated npm identity."},
        {"name": "Package publish permission", "status": "unknown", "detail": "Not tested. Authentication alone does not prove publish access to a package."},
    ], {"username": username if isinstance(username, str) else None})



def validate_vercel(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://api.vercel.com/v2/user",
        headers={"Authorization": f"Bearer {token}"},
    )
    if status != 200:
        return _classify_http("Vercel", status, payload, headers)
    user = payload.get("user", payload) if isinstance(payload, dict) else {}
    if not isinstance(user, dict):
        user = {}
    metadata = {}
    for key in ("username", "name", "id"):
        if isinstance(user.get(key), str):
            metadata["account_id" if key == "id" else key] = user[key]
    return _result("Vercel", "valid", "Access token accepted by the Vercel user endpoint.", [
        {"name": "Token authentication", "status": "confirmed", "detail": "Vercel returned the authenticated user profile."},
        {"name": "Team and project permissions", "status": "unknown", "detail": "The user endpoint does not enumerate every team, project, or resource permission."},
    ], metadata)


def validate_cloudflare(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://api.cloudflare.com/client/v4/user/tokens/verify",
        headers={"Authorization": f"Bearer {token}"},
    )
    if status != 200:
        return _classify_http("Cloudflare", status, payload, headers)
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    if not isinstance(result, dict) or payload.get("success") is not True:
        return _result("Cloudflare", "inconclusive", "Cloudflare did not confirm token verification.", [
            {"name": "Token verification", "status": "unknown", "detail": _clean_error(payload)}
        ])
    state = result.get("status") if isinstance(result.get("status"), str) else "unknown"
    active = state == "active"
    return _result("Cloudflare", "valid" if active else "restricted", f"Cloudflare reports this token as {state}.", [
        {"name": "Token verification", "status": "confirmed", "detail": f"Cloudflare token status: {state}."},
        {"name": "Granted permission groups", "status": "unknown", "detail": "The verify endpoint does not return the token's permission policies."},
    ], {"token_id": result.get("id"), "token_status": state, "expires_on": result.get("expires_on"), "not_before": result.get("not_before")})



CLOUDFLARE_ACCOUNT_ID_RE = re.compile(r"^[a-f0-9]{32}$")
GCP_PROJECT_ID_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
GCS_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")


def _validate_cloudflare_account_resource(token: str, account_id: str, resource: str) -> dict[str, Any]:
    """Read-only, account-scoped Cloudflare resource check."""
    if not CLOUDFLARE_ACCOUNT_ID_RE.fullmatch(account_id):
        return _result("Cloudflare", "error", "Invalid Cloudflare Account ID format.", [
            {"name": "Input validation", "status": "denied", "detail": "Use the 32-character hexadecimal Account ID from the Cloudflare dashboard."}
        ])

    paths = {
        "ai_gateway": (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai-gateway/gateways?per_page=100",
            "Cloudflare AI Gateway",
            "AI Gateway Read",
            "gateway",
        ),
        "workers": (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/workers/scripts",
            "Cloudflare Workers",
            "Workers Scripts Read",
            "Worker script",
        ),
    }
    if resource not in paths:
        return _result("Cloudflare", "error", "Unsupported Cloudflare resource check.", [
            {"name": "Resource selection", "status": "denied", "detail": "This resource adapter is not registered."}
        ])

    url, label, required_permission, item_label = paths[resource]
    status, headers, payload = _json_request(url, headers={"Authorization": f"Bearer {token}"})
    if status != 200:
        result = _classify_http(label, status, payload, headers)
        if status == 403:
            result["summary"] = f"Cloudflare denied access to the {resource.replace('_', ' ')} listing endpoint. The token may be valid but lack the required account permission or account access."
            result["checks"] = [
                {"name": "Account resource read access", "status": "denied", "detail": f"The endpoint requires {required_permission} or a broader accepted permission. HTTP 403 does not prove the token itself is invalid."}
            ]
        return result

    if not isinstance(payload, dict) or payload.get("success") is not True:
        return _result(label, "inconclusive", "Cloudflare did not return a successful resource-list response.", [
            {"name": "Resource read access", "status": "unknown", "detail": _clean_error(payload)}
        ])
    result_items = payload.get("result")
    if not isinstance(result_items, list):
        result_items = []
    names: list[str] = []
    for item in result_items[:8]:
        if isinstance(item, dict):
            value = item.get("name") or item.get("id")
            if isinstance(value, str):
                names.append(value)
    count_name = "gateway_count" if resource == "ai_gateway" else "worker_script_count"
    list_permission_detail = f"Cloudflare accepted the token for the account-scoped list endpoint ({required_permission})."
    checks = [
        {"name": "Credential and account access", "status": "confirmed", "detail": "The Cloudflare API accepted the token for the supplied account."},
        {"name": f"{item_label} list/read permission", "status": "confirmed", "detail": list_permission_detail},
        {"name": "Create, edit, and delete permissions", "status": "unknown", "detail": "A read-only list endpoint cannot establish write or delete permissions."},
    ]
    return _result(label, "valid", f"Cloudflare accepted the token. {len(result_items)} {item_label.lower()} entries were returned on the first page.", checks, {
        "account_id": account_id,
        count_name: len(result_items),
        "sample_names": names,
    })


def validate_cloudflare_ai_gateway(token: str, account_id: str) -> dict[str, Any]:
    return _validate_cloudflare_account_resource(token, account_id, "ai_gateway")


def validate_cloudflare_workers(token: str, account_id: str) -> dict[str, Any]:
    return _validate_cloudflare_account_resource(token, account_id, "workers")


def _test_google_permissions(
    token: str,
    url: str,
    permissions: list[str],
    *,
    provider: str,
    label: str,
    resource_name: str,
    method: str = "POST",
) -> dict[str, Any]:
    body = json.dumps({"permissions": permissions}, separators=(",", ":")).encode("utf-8") if method == "POST" else None
    status, headers, payload = _json_request(
        url,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        body=body,
    )
    if status != 200:
        result = _classify_http(provider, status, payload, headers)
        if status == 403:
            result["summary"] = f"Google accepted the request path but denied this permission inspection. The token may lack the required OAuth scope or access to {resource_name}."
        return result
    granted = payload.get("permissions", []) if isinstance(payload, dict) else []
    granted_set = set(item for item in granted if isinstance(item, str)) if isinstance(granted, list) else set()
    checks = [
        {"name": permission, "status": "confirmed" if permission in granted_set else "denied", "detail": "Google IAM reports this permission for the selected resource." if permission in granted_set else "Google did not include this permission in the granted-permissions response."}
        for permission in permissions
    ]
    return _result(provider, "valid", f"Google authenticated the request and returned an IAM permission test for {resource_name}.", checks, {
        "resource": resource_name,
        "granted_permissions": sorted(granted_set),
        "granted_count": len(granted_set),
        "tested_count": len(permissions),
    })


def validate_gcp(token: str, project_id: str) -> dict[str, Any]:
    """Check a Google OAuth access token against project-level IAM permissions."""
    if not GCP_PROJECT_ID_RE.fullmatch(project_id):
        return _result("Google Cloud", "error", "Invalid Google Cloud project ID format.", [
            {"name": "Input validation", "status": "denied", "detail": "Project IDs are 6–30 characters, start with a lowercase letter, and contain lowercase letters, digits, or hyphens."}
        ])
    permissions = [
        "resourcemanager.projects.get",
        "resourcemanager.projects.getIamPolicy",
        "storage.buckets.list",
        "iam.serviceAccounts.list",
        "serviceusage.services.list",
    ]
    project = urllib.parse.quote(project_id, safe="")
    return _test_google_permissions(
        token,
        f"https://cloudresourcemanager.googleapis.com/v1/projects/{project}:testIamPermissions",
        permissions,
        provider="Google Cloud",
        label="Project IAM permissions",
        resource_name=f"project {project_id}",
    )


def validate_gcp_storage(token: str, bucket_name: str) -> dict[str, Any]:
    """Use Cloud Storage's testIamPermissions method; no object content is read."""
    if not GCS_BUCKET_RE.fullmatch(bucket_name) or ".." in bucket_name or len(bucket_name) > 63:
        return _result("Google Cloud Storage", "error", "Invalid Cloud Storage bucket name format.", [
            {"name": "Input validation", "status": "denied", "detail": "Enter a 3–63 character bucket name using lowercase letters, digits, dots, underscores, or hyphens."}
        ])
    permissions = [
        "storage.buckets.get",
        "storage.buckets.getIamPolicy",
        "storage.objects.list",
        "storage.objects.get",
        "storage.objects.create",
        "storage.objects.delete",
    ]
    query = urllib.parse.urlencode([("permissions", item) for item in permissions])
    bucket = urllib.parse.quote(bucket_name, safe="")
    return _test_google_permissions(
        token,
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/iam/testPermissions?{query}",
        permissions,
        provider="Google Cloud Storage",
        label="Bucket IAM permissions",
        resource_name=f"bucket {bucket_name}",
        method="GET",
    )

def validate_datadog(token: str, app_key: str = "", site: str = "US1") -> dict[str, Any]:
    hosts = {
        "US1": "api.datadoghq.com", "US3": "api.us3.datadoghq.com", "US5": "api.us5.datadoghq.com",
        "EU1": "api.datadoghq.eu", "AP1": "api.ap1.datadoghq.com", "AP2": "api.ap2.datadoghq.com",
        "US2": "api.us2.datadoghq.com",
        "US Gov": "api.ddog-gov.com", "UK1": "api.uk1.datadoghq.com",
    }
    host = hosts.get(site, hosts["US1"])
    if app_key:
        status, headers, payload = _json_request(
            f"https://{host}/api/v2/validate_keys",
            headers={"DD-API-KEY": token, "DD-APPLICATION-KEY": app_key},
        )
        if status != 200:
            if status == 403:
                return _result("Datadog", "invalid", "Datadog rejected the API key or application key.", [
                    {"name": "Credential validation", "status": "denied", "detail": "Datadog's validation API documents HTTP 403 for invalid key credentials."}
                ])
            return _classify_http("Datadog", status, payload, headers)
        return _result("Datadog", "valid", "Datadog accepted both the API key and application key.", [
            {"name": "API key", "status": "confirmed", "detail": "Datadog accepted the API key."},
            {"name": "Application key", "status": "confirmed", "detail": "Datadog accepted the application key."},
            {"name": "Specific product permissions", "status": "unknown", "detail": "The key-validation endpoint does not enumerate resource-level permissions."},
        ])
    status, headers, payload = _json_request(
        f"https://{host}/api/v1/validate",
        headers={"DD-API-KEY": token},
    )
    if status != 200:
        if status == 403:
            return _result("Datadog", "invalid", "Datadog rejected the API key.", [
                {"name": "API key validation", "status": "denied", "detail": "Datadog documents HTTP 403 when the API key is invalid."}
            ])
        return _classify_http("Datadog", status, payload, headers)
    valid = isinstance(payload, dict) and payload.get("valid") is True
    if not valid:
        return _result("Datadog", "inconclusive", "Datadog did not return valid=true.", [
            {"name": "API key", "status": "unknown", "detail": _clean_error(payload)}
        ])
    return _result("Datadog", "valid", "Datadog accepted the API key.", [
        {"name": "API key authentication", "status": "confirmed", "detail": "The Datadog validation endpoint returned valid=true."},
        {"name": "Read/reporting permissions", "status": "unknown", "detail": "Some Datadog read APIs also require an application key; this check does not enumerate resource permissions."},
    ])


def validate_slack(token: str) -> dict[str, Any]:
    status, headers, payload = _json_request(
        "https://slack.com/api/auth.test",
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/x-www-form-urlencoded"},
        body=b"",
    )
    if status != 200 or not isinstance(payload, dict) or payload.get("ok") is not True:
        if status == 200:
            return _result("Slack", "invalid", "Slack's auth.test method rejected this credential.", [
                {"name": "Authentication", "status": "denied", "detail": _clean_error(payload)}
            ])
        return _classify_http("Slack", status, payload, headers)
    metadata = {}
    for key, out_key in (("team", "workspace"), ("user", "user"), ("team_id", "workspace_id"), ("user_id", "user_id")):
        if isinstance(payload.get(key), str):
            metadata[out_key] = payload[key]
    return _result("Slack", "valid", "Slack accepted the token and returned its authenticated identity.", [
        {"name": "Token authentication", "status": "confirmed", "detail": "Slack auth.test succeeded."},
        {"name": "Granted scopes", "status": "unknown", "detail": "auth.test confirms identity but does not enumerate all granted scopes."},
    ], metadata)


def _aws_signing_key(secret: str, date: str, region: str, service: str = "sts") -> bytes:
    def sign(key: bytes, message: str) -> bytes:
        return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
    k_date = sign(("AWS4" + secret).encode("utf-8"), date)
    k_region = sign(k_date, region)
    k_service = sign(k_region, service)
    return sign(k_service, "aws4_request")


def validate_aws(access_key: str, secret_key: str, region: str = "us-east-1", session_token: str = "") -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9-]{1,30}", region):
        return _result("AWS STS", "error", "Invalid AWS region format.", [
            {"name": "Input validation", "status": "denied", "detail": "Use a standard AWS region such as us-east-1."}
        ])
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    host = f"sts.{region}.amazonaws.com"
    body = urllib.parse.urlencode({"Action": "GetCallerIdentity", "Version": "2011-06-15"}).encode("ascii")
    payload_hash = hashlib.sha256(body).hexdigest()
    canonical_headers = f"content-type:application/x-www-form-urlencoded; charset=utf-8\nhost:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        "Host": host,
        "X-Amz-Content-Sha256": payload_hash,
        "X-Amz-Date": amz_date,
    }
    if session_token:
        canonical_headers += f"x-amz-security-token:{session_token}\n"
        signed_headers += ";x-amz-security-token"
        headers["X-Amz-Security-Token"] = session_token
    canonical_request = "POST\n/\n\n" + canonical_headers + "\n" + signed_headers + "\n" + payload_hash
    scope = f"{date_stamp}/{region}/sts/aws4_request"
    string_to_sign = "AWS4-HMAC-SHA256\n" + amz_date + "\n" + scope + "\n" + hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    signing_key = _aws_signing_key(secret_key, date_stamp, region)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    try:
        status, response_headers, raw = _request(f"https://{host}/", method="POST", headers=headers, body=body)
    except Exception as exc:
        return _result("AWS STS", "error", "The AWS identity check could not be completed.", [
            {"name": "STS request", "status": "unknown", "detail": str(exc)[:180]}
        ])
    if status != 200:
        try:
            root = ET.fromstring(raw)
            message = root.findtext(".//Message") or root.findtext(".//message") or "AWS rejected the signed request."
        except ET.ParseError:
            message = "AWS rejected the signed request."
        if status in (400, 403):
            return _result("AWS STS", "invalid", "AWS did not accept these credentials for GetCallerIdentity.", [
                {"name": "Credential authentication", "status": "denied", "detail": f"HTTP {status}: {message[:180]}"}
            ])
        return _classify_http("AWS STS", status, {"message": message}, response_headers)
    try:
        root = ET.fromstring(raw)
        account = root.findtext(".//Account")
        arn = root.findtext(".//Arn")
        user_id = root.findtext(".//UserId")
    except ET.ParseError:
        return _result("AWS STS", "inconclusive", "AWS returned an unreadable identity response.", [
            {"name": "STS identity", "status": "unknown", "detail": "Could not parse the XML response."}
        ])
    return _result("AWS STS", "valid", "AWS accepted the signed credentials and identified the caller.", [
        {"name": "Credential authentication", "status": "confirmed", "detail": "STS GetCallerIdentity succeeded."},
        {"name": "IAM permissions", "status": "unknown", "detail": "GetCallerIdentity does not require IAM permissions, so this result does not reveal allowed AWS actions."},
    ], {"account": account, "arn": arn, "user_id": user_id, "region": region})



def validate_aws_s3(
    access_key: str,
    secret_key: str,
    bucket_name: str,
    region: str = "us-east-1",
    session_token: str = "",
) -> dict[str, Any]:
    """Confirm AWS identity, then test only s3:ListBucket with max-keys=0."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket_name):
        return _result("Amazon S3", "error", "Invalid general-purpose S3 bucket name format.", [
            {"name": "Input validation", "status": "denied", "detail": "Use a 3–63 character lowercase S3 bucket name. Directory buckets and dotted TLS endpoint names are not covered by this adapter."}
        ])
    identity = validate_aws(access_key, secret_key, region, session_token)
    if identity.get("status") != "valid":
        identity["provider"] = "Amazon S3"
        identity["summary"] = "AWS caller identity could not be confirmed, so the S3 permission probe was not attempted."
        return identity

    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    host = f"{bucket_name}.s3.{region}.amazonaws.com"
    canonical_query = "list-type=2&max-keys=0"
    payload_hash = hashlib.sha256(b"").hexdigest()
    canonical_headers = f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    headers = {
        "Host": host,
        "X-Amz-Content-Sha256": payload_hash,
        "X-Amz-Date": amz_date,
    }
    if session_token:
        canonical_headers += f"x-amz-security-token:{session_token}\n"
        signed_headers += ";x-amz-security-token"
        headers["X-Amz-Security-Token"] = session_token
    canonical_request = "GET\n/\n" + canonical_query + "\n" + canonical_headers + "\n" + signed_headers + "\n" + payload_hash
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "AWS4-HMAC-SHA256\n" + amz_date + "\n" + scope + "\n" + hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    signing_key = _aws_signing_key(secret_key, date_stamp, region, "s3")
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    try:
        status, response_headers, raw = _request(
            f"https://{host}/?{canonical_query}",
            method="GET",
            headers=headers,
        )
    except Exception:
        return _result("Amazon S3", "timeout", "The S3 bucket permission check could not be completed.", [
            {"name": "S3 ListBucket", "status": "unknown", "detail": "The signed request failed or timed out. No conclusion about bucket access can be made."}
        ])

    if status == 200:
        try:
            root = ET.fromstring(raw)
            key_count_text = root.findtext(".//KeyCount")
            returned = int(key_count_text) if key_count_text and key_count_text.isdigit() else 0
        except (ET.ParseError, ValueError):
            returned = 0
        return _result("Amazon S3", "valid", "AWS identity was confirmed and the bucket accepted the zero-key ListObjectsV2 permission probe.", [
            {"name": "AWS caller identity", "status": "confirmed", "detail": f"STS identified the caller in AWS account {identity.get('metadata', {}).get('account') or 'unknown'}."},
            {"name": "s3:ListBucket", "status": "confirmed", "detail": "ListObjectsV2 accepted max-keys=0. No object names or object contents were requested."},
            {"name": "Object read permission (s3:GetObject)", "status": "unknown", "detail": "Not tested. Bucket listing does not prove access to object contents."},
            {"name": "Object write/delete permissions", "status": "unknown", "detail": "Not tested. PermiKey does not upload or delete objects."},
        ], {
            "account": identity.get("metadata", {}).get("account"),
            "arn": identity.get("metadata", {}).get("arn"),
            "bucket": bucket_name,
            "region": response_headers.get("x-amz-bucket-region", region),
            "keys_returned": returned,
        })

    region_hint = response_headers.get("x-amz-bucket-region")
    if status in (301, 307) or (region_hint and region_hint != region):
        return _result("Amazon S3", "inconclusive", "AWS credentials are valid, but the bucket is served from a different region.", [
            {"name": "AWS caller identity", "status": "confirmed", "detail": "STS GetCallerIdentity succeeded."},
            {"name": "S3 bucket region", "status": "unknown", "detail": f"The provider indicates region {region_hint or 'redirect'}. Select that region and run the check again."},
        ], {"bucket": bucket_name, "expected_region": region, "reported_region": region_hint})
    if status == 403:
        return _result("Amazon S3", "restricted", "AWS credentials are valid, but S3 did not allow the bucket listing check.", [
            {"name": "AWS caller identity", "status": "confirmed", "detail": "STS GetCallerIdentity succeeded."},
            {"name": "s3:ListBucket", "status": "denied", "detail": "S3 returned HTTP 403 for the zero-key ListObjectsV2 request. Bucket policy, IAM policy, or expected-owner constraints may deny it."},
            {"name": "Object access permissions", "status": "unknown", "detail": "Not tested. A denied bucket listing does not establish object read/write access."},
        ], {"bucket": bucket_name, "region": region})
    if status == 404:
        return _result("Amazon S3", "inconclusive", "S3 could not confirm this bucket for the supplied region and caller.", [
            {"name": "AWS caller identity", "status": "confirmed", "detail": "STS GetCallerIdentity succeeded."},
            {"name": "Bucket existence/access", "status": "unknown", "detail": "HTTP 404 can indicate a missing bucket or a bucket that is not visible to this caller."},
        ], {"bucket": bucket_name, "region": region})
    try:
        root = ET.fromstring(raw)
        code = root.findtext(".//Code") or "S3Error"
        message = root.findtext(".//Message") or "S3 rejected the request."
    except ET.ParseError:
        code, message = f"HTTP {status}", "S3 rejected the request."
    return _result("Amazon S3", "error", "AWS identity was confirmed, but the S3 endpoint returned an unexpected response.", [
        {"name": "AWS caller identity", "status": "confirmed", "detail": "STS GetCallerIdentity succeeded."},
        {"name": "S3 ListBucket", "status": "unknown", "detail": f"{code}: {message[:180]}"},
    ], {"bucket": bucket_name, "region": region, "http_status": status})


# Inference tests are opt-in because even a one-token response may consume a
# provider's paid quota. These probes use one fixed, harmless prompt and request
# the smallest output budget supported by each API family.
_INFERENCE_PROVIDERS = {
    "openai", "nvidia", "groq", "openrouter", "gemini",
    "anthropic", "mistral", "cohere",
}


def _choose_inference_model(provider_id: str, model_ids: list[Any]) -> str | None:
    candidates = [
        model for model in model_ids
        if isinstance(model, str) and model and not any(
            word in model.lower() for word in (
                "embed", "rerank", "moderation", "whisper", "transcribe",
                "text-to-speech", "tts", "dall-e", "image-generation",
            )
        )
    ]
    if provider_id == "openai":
        preferences = ("gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o")
        for preferred in preferences:
            if preferred in candidates:
                return preferred
    if provider_id == "gemini":
        candidates = [model for model in candidates if "gemini" in model.lower()]
    return candidates[0] if candidates else None


def _post_inference_probe(
    provider_id: str, token: str, model: str
) -> tuple[int, dict[str, str], dict[str, Any] | list[Any] | None]:
    """Send one opt-in, one-token generation request to a fixed provider URL."""
    common_headers = {"Content-Type": "application/json"}
    prompt = "Reply with OK."
    if provider_id in {"openai", "nvidia", "groq", "openrouter", "mistral"}:
        urls = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "mistral": "https://api.mistral.ai/v1/chat/completions",
        }
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if provider_id == "openai":
            body["max_completion_tokens"] = 1
        else:
            body["max_tokens"] = 1
        headers = {**common_headers, "Authorization": f"Bearer {token}"}
        return _json_request(urls[provider_id], method="POST", headers=headers, body=json.dumps(body).encode("utf-8"))

    if provider_id == "anthropic":
        body = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": prompt}]}
        headers = {**common_headers, "x-api-key": token, "anthropic-version": "2023-06-01"}
        return _json_request("https://api.anthropic.com/v1/messages", method="POST", headers=headers, body=json.dumps(body).encode("utf-8"))

    if provider_id == "gemini":
        model_path = model if model.startswith("models/") else f"models/{model}"
        safe_model_path = urllib.parse.quote(model_path, safe="/.-_:")
        url = f"https://generativelanguage.googleapis.com/v1beta/{safe_model_path}:generateContent"
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 1, "temperature": 0}}
        headers = {**common_headers, "x-goog-api-key": token}
        return _json_request(url, method="POST", headers=headers, body=json.dumps(body).encode("utf-8"))

    if provider_id == "cohere":
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1}
        headers = {**common_headers, "Authorization": f"Bearer {token}"}
        return _json_request("https://api.cohere.com/v2/chat", method="POST", headers=headers, body=json.dumps(body).encode("utf-8"))

    raise ValueError("Inference probing is not available for this provider.")


def _set_inference_check(result: dict[str, Any], check: dict[str, str]) -> None:
    checks = result.setdefault("checks", [])
    for index, current in enumerate(checks):
        check_name = str(current.get("name", "")).lower()
        if "inference" in check_name or "model request permission" in check_name or "generation permission" in check_name:
            checks[index] = check
            return
    checks.append(check)


def _with_optional_inference(
    result: dict[str, Any], provider_id: str, token: str, requested: bool
) -> dict[str, Any]:
    if not requested or result.get("status") != "valid":
        return result

    model_ids = result.get("metadata", {}).get("sample_models", [])
    if provider_id == "openrouter" and not model_ids:
        catalogue_status, _catalogue_headers, catalogue = _json_request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {token}"},
        )
        catalogue_data = catalogue.get("data", []) if isinstance(catalogue, dict) else []
        if catalogue_status == 200 and isinstance(catalogue_data, list):
            model_ids = [item.get("id") for item in catalogue_data if isinstance(item, dict) and isinstance(item.get("id"), str)]
    model = _choose_inference_model(provider_id, model_ids if isinstance(model_ids, list) else [])
    if not model:
        inference_check = {
            "name": "Model inference permission",
            "status": "unknown",
            "detail": "The catalogue returned no suitable text-generation model to test. No generation request was sent.",
        }
        _set_inference_check(result, inference_check)
        return result

    try:
        status, _headers, payload = _post_inference_probe(provider_id, token, model)
    except (TimeoutError, OSError, ValueError) as exc:
        _set_inference_check(result, {
            "name": "Model inference permission",
            "status": "unknown",
            "detail": f"The optional generation probe could not be completed ({type(exc).__name__}).",
        })
        return result

    if 200 <= status < 300 and isinstance(payload, dict):
        probe_status = "confirmed"
        detail = f"The provider accepted a minimal generation request for model {model}. The provider may still apply separate limits or policies to other models."
        summary = "Credential accepted. The optional minimal inference request also succeeded."
    elif status in (401, 403):
        probe_status = "denied"
        detail = f"The provider rejected the generation request for model {model} (HTTP {status}). Catalogue access alone does not grant inference access."
        summary = "Credential accepted for catalogue access, but the optional inference request was denied for the selected model."
    else:
        probe_status = "unknown"
        provider_error = _clean_error(payload)
        detail = f"The generation probe for model {model} returned HTTP {status}. {provider_error[:160]}"
        summary = "Credential accepted for catalogue access; the optional inference probe was inconclusive."

    _set_inference_check(result, {
        "name": "Model inference permission",
        "status": probe_status,
        "detail": detail,
    })
    result["summary"] = summary
    result.setdefault("metadata", {})["inference_probe"] = {
        "model": model,
        "max_output_tokens": 1,
        "may_consume_provider_quota": True,
        "http_status": status,
    }
    return result


PROVIDERS = {
    "openai": lambda d: _with_optional_inference(validate_openai(d["token"]), "openai", d["token"], d.get("test_inference", False)),
    "nvidia": lambda d: _with_optional_inference(validate_nvidia(d["token"]), "nvidia", d["token"], d.get("test_inference", False)),
    "groq": lambda d: _with_optional_inference(validate_groq(d["token"]), "groq", d["token"], d.get("test_inference", False)),
    "openrouter": lambda d: _with_optional_inference(validate_openrouter(d["token"]), "openrouter", d["token"], d.get("test_inference", False)),
    "gemini": lambda d: _with_optional_inference(validate_gemini(d["token"]), "gemini", d["token"], d.get("test_inference", False)),
    "anthropic": lambda d: _with_optional_inference(validate_anthropic(d["token"]), "anthropic", d["token"], d.get("test_inference", False)),
    "mistral": lambda d: _with_optional_inference(validate_mistral(d["token"]), "mistral", d["token"], d.get("test_inference", False)),
    "cohere": lambda d: _with_optional_inference(validate_cohere(d["token"]), "cohere", d["token"], d.get("test_inference", False)),
    "huggingface": lambda d: validate_huggingface(d["token"]),
    "github": lambda d: validate_github(d["token"]),
    "codeberg": lambda d: validate_codeberg(d["token"]),
    "gitlab": lambda d: validate_gitlab(d["token"]),
    "npm": lambda d: validate_npm(d["token"]),
    "cloudflare": lambda d: validate_cloudflare(d["token"]),
    "cloudflare_ai_gateway": lambda d: validate_cloudflare_ai_gateway(d["token"], d["account_id"]),
    "cloudflare_workers": lambda d: validate_cloudflare_workers(d["token"], d["account_id"]),
    "gcp": lambda d: validate_gcp(d["token"], d["project_id"]),
    "gcp_storage": lambda d: validate_gcp_storage(d["token"], d["bucket_name"]),
    "vercel": lambda d: validate_vercel(d["token"]),
    "datadog": lambda d: validate_datadog(d["token"], d.get("app_key", ""), d.get("site", "US1")),
    "slack": lambda d: validate_slack(d["token"]),
    "aws": lambda d: validate_aws(d["access_key"], d["secret_key"], d.get("region", "us-east-1"), d.get("session_token", "")),
    "aws_s3": lambda d: validate_aws_s3(d["access_key"], d["secret_key"], d["bucket_name"], d.get("region", "us-east-1"), d.get("session_token", "")),
}


def validate_credentials(data: Any) -> tuple[int, dict[str, Any]]:
    """Validate one user-supplied credential object without retaining it."""
    if not isinstance(data, dict):
        return 400, {"error": "Request body must be a JSON object."}
    provider = data.get("provider")
    if not isinstance(provider, str) or provider not in PROVIDERS:
        return 400, {"error": "Unsupported provider."}
    if "test_inference" in data and not isinstance(data["test_inference"], bool):
        return 400, {"error": "test_inference must be true or false."}
    if data.get("test_inference") and provider not in _INFERENCE_PROVIDERS:
        return 400, {"error": "Optional inference testing is not supported for this provider."}

    required_fields = {
        "aws": ["access_key", "secret_key"],
        "aws_s3": ["access_key", "secret_key", "bucket_name"],
        "cloudflare_ai_gateway": ["token", "account_id"],
        "cloudflare_workers": ["token", "account_id"],
        "gcp": ["token", "project_id"],
        "gcp_storage": ["token", "bucket_name"],
    }
    required = required_fields.get(provider, ["token"])
    for field in required:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            return 400, {"error": f"Missing required field: {field}."}
        if len(value) > 8192 or "\x00" in value:
            return 400, {"error": "Credential input is too long or contains invalid characters."}

    for field in ("app_key", "session_token", "account_id", "project_id", "bucket_name"):
        value = data.get(field, "")
        if value is not None and (not isinstance(value, str) or len(value) > 8192):
            return 400, {"error": f"Invalid optional field: {field}."}
    if provider == "datadog":
        site = data.get("site", "US1")
        if not isinstance(site, str) or site not in {"US1", "US3", "US5", "EU1", "AP1", "AP2", "US2", "UK1", "US Gov"}:
            return 400, {"error": "Unsupported Datadog site."}
    if provider in {"aws", "aws_s3"}:
        region = data.get("region", "us-east-1")
        if not isinstance(region, str) or region not in {
            "us-east-1", "us-east-2", "us-west-1", "us-west-2", "ap-south-1",
            "ap-southeast-1", "ap-southeast-2", "eu-west-1", "eu-central-1",
        }:
            return 400, {"error": "Unsupported AWS STS region."}

    if provider in {"cloudflare_ai_gateway", "cloudflare_workers"}:
        account_id = data.get("account_id", "")
        if not isinstance(account_id, str) or not CLOUDFLARE_ACCOUNT_ID_RE.fullmatch(account_id.strip().lower()):
            return 400, {"error": "Cloudflare Account ID must be 32 hexadecimal characters."}
    if provider == "gcp":
        project_id = data.get("project_id", "")
        if not isinstance(project_id, str) or not GCP_PROJECT_ID_RE.fullmatch(project_id.strip()):
            return 400, {"error": "Invalid Google Cloud project ID."}
    if provider == "gcp_storage":
        bucket_name = data.get("bucket_name", "")
        if not isinstance(bucket_name, str) or not GCS_BUCKET_RE.fullmatch(bucket_name.strip()) or ".." in bucket_name:
            return 400, {"error": "Invalid Google Cloud Storage bucket name."}
    if provider == "aws_s3":
        bucket_name = data.get("bucket_name", "")
        if not isinstance(bucket_name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket_name.strip()):
            return 400, {"error": "Invalid general-purpose S3 bucket name."}

    # Work with a short-lived normalized copy; no storage or logging is performed.
    clean = dict(data)
    for key, value in clean.items():
        if isinstance(value, str):
            clean[key] = value.strip().lower() if key == "account_id" else value.strip()
    try:
        result = PROVIDERS[provider](clean)
        secrets = [clean.get(name, "") for name in ("token", "access_key", "secret_key", "app_key", "session_token")]
        secrets = [value for value in secrets if isinstance(value, str) and value]

        def redact(value: Any) -> Any:
            if isinstance(value, str):
                for secret in secrets:
                    value = value.replace(secret, "[redacted]")
                return value
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            return value

        return 200, redact(result)
    except (TimeoutError, urllib.error.URLError):
        return 200, _result(provider, "timeout", "The provider did not respond before the timeout.", [
            {"name": "Network request", "status": "unknown", "detail": "Timed out. No conclusion about credential validity can be made."}
        ])
    except Exception:
        # Do not return exception strings, request headers, or input values to clients.
        return 200, _result(provider, "error", "The check failed unexpectedly. No credential data was retained.", [
            {"name": "Checker", "status": "unknown", "detail": "Unexpected internal error. Try again or report the provider and status."}
        ])
