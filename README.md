# PermiKey

**Check API credentials. Understand the access they actually have.**

PermiKey is a stateless credential-validation and permission-inspection tool. It checks credentials against official provider APIs and reports what those APIs confirm, deny, or do not expose.

## Try PermiKey

**Hosted app:** <https://permi-key.vercel.app/>

No account is required. Choose a service from the start page or use the sidebar search to find one.

## Supported services

### AI and model providers

OpenAI, NVIDIA NIM, Groq, OpenRouter, Google Gemini, Anthropic, Mistral AI, Cohere, and Hugging Face.

### Code and package hosting

GitHub, Codeberg, GitLab, and npm Registry.

### Cloud and infrastructure

Amazon Web Services (STS identity), Amazon S3, Cloudflare, Cloudflare AI Gateway, Cloudflare Workers, Google Cloud project IAM, Google Cloud Storage, and Vercel.

### Observability

Datadog.

### Collaboration

Slack.

## How to use it

1. Open <https://permi-key.vercel.app/>.
2. Choose a provider from the category cards or the sidebar.
3. Enter the credential and any provider-specific account, project, bucket, or region details requested.
4. Select **Validate credential**.
5. Review the result. Credential authentication, resource access, and permissions are reported as separate checks where possible.

The form is cleared after the request completes. PermiKey does not provide a credential vault or saved result history.

## Optional AI inference test

For supported AI providers, the credential form includes an optional **Run an optional minimal inference test** checkbox. It is off by default.

When enabled, PermiKey first checks the provider's model catalogue, then sends one short fixed prompt to one suitable text-generation model and requests at most one output token. This can still consume provider quota or incur charges. The test confirms only that the specific model accepted that minimal request at that moment; it does not prove access to every model, future availability, or every feature.

The optional test is supported for OpenAI, NVIDIA NIM, Groq, OpenRouter, Gemini, Anthropic, Mistral, and Cohere. It is not run for Hugging Face or any other non-AI integration. Leave it unchecked to perform credential and catalogue checks without sending a generation request.

## Understanding results

- **Confirmed:** the provider returned evidence that the specific check succeeded.
- **Denied:** the provider explicitly rejected that specific access check.
- **Unknown:** the provider did not expose enough information, the check was not run, or the response was inconclusive.

A valid credential does not automatically have every permission. Some providers expose scopes or IAM permissions; others only expose identity or resource-list endpoints. PermiKey does not guess missing permissions or treat a visible model as proof of generation access.

## Privacy and safe use

- PermiKey has no application database, user accounts, credential vault, or saved validation history.
- Credentials are submitted to the PermiKey backend over HTTPS and are then sent to the selected provider's official API. Do not enter credentials you do not own or are not authorised to test.
- Credential fields are cleared in the browser when the request finishes. Application code does not intentionally log or persist submitted credentials.
- The hosted service runs on Vercel. Vercel, network infrastructure, and the provider may process routine request, security, or rate-limit metadata under their own policies. “No database” does not mean that third-party infrastructure has no logs.
- Checks are designed to be read-only. The optional AI inference test is the exception in that it creates a minimal generation request and may consume provider usage. It does not change provider settings or resources.
- Never paste production credentials into a service you do not trust. For sensitive use, review the source and consider running your own copy locally.

## Run locally

PermiKey uses the Python standard library and does not require a database or package installation.

```bash
python app.py
```

Then open <http://127.0.0.1:8000> in your browser. Stop the local server with `Ctrl+C`.

## Companion project

**SecretBurrow:** <https://giriaryan694-a11y.github.io/SecretBurrow/>

SecretBurrow analyzes repositories and firmware locally in the browser to help uncover exposed secrets, credentials, hashes, encodings, cryptographic material, and suspicious high-entropy data.

## Credits

Made By Aryan Giri | giriaryan694-a11y

## License

PermiKey is distributed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for the complete license text.
