# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in ManufacturingIQ, please report it privately by emailing the project maintainer. **Do not** create a public GitHub issue.

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

You can expect an acknowledgment within 48 hours, and we will work to address the issue promptly.

## Security Best Practices

When deploying ManufacturingIQ:

1. **API Keys**: Use strong, unique API keys. Rotate them regularly. Never commit them to version control.
2. **Google OAuth**: Configure proper redirect URIs. Use email allowlists to restrict dashboard access.
3. **Environment Variables**: All secrets go in `.env` (backend) or `.streamlit/secrets.toml` (dashboard). Both are gitignored.
4. **HTTPS**: Always use HTTPS in production. The API keys and OAuth tokens are transmitted in headers/cookies.
5. **Dependencies**: Keep dependencies updated. Run `pip-audit` or `safety check` regularly.
6. **Logging**: In production, set `ENV=production` to avoid leaking sensitive data in human-readable logs.