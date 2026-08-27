# Security

## Do not publish secrets

Never commit `.env.local`, API keys, provider tokens, cookies, database files, private sensor data, packet captures or unredacted location information.

Before each commit, enable the supplied hook and run the scanner:

```powershell
git config core.hooksPath .githooks
uv run python scripts/check_git_safety.py --staged
```

If a real credential is ever committed, revoke or rotate it first. Deleting the current file does not remove the value from Git history.

## phyphox boundary

The phyphox remote interface is HTTP without a password. Use it only on a trusted local network or personal hotspot, keep the session short, and disable remote access immediately after capture. Do not expose the phone address through port forwarding or a public reverse proxy.

## Reporting a vulnerability

Do not place credentials, private datasets or exploit details in a public Issue. If GitHub Private Vulnerability Reporting is enabled for this repository, use that channel. Otherwise, open a minimal Issue asking the maintainer for a private contact channel without including sensitive details.

## Support status

This repository is a competition preview, not a hardened multi-tenant cloud service. The default configuration is intended for local use on `127.0.0.1`.
