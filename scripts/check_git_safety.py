from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath

MAX_TRACKED_FILE_BYTES = 10 * 1024 * 1024

_DATABASE_SUFFIXES = (
    ".sqlite",
    ".sqlite-shm",
    ".sqlite-wal",
    ".sqlite-journal",
    ".sqlite3",
    ".sqlite3-shm",
    ".sqlite3-wal",
    ".sqlite3-journal",
    ".db",
    ".db-shm",
    ".db-wal",
    ".db-journal",
)
_SENSITIVE_SUFFIXES = (
    ".log",
    ".trace",
    ".har",
    ".pcap",
    ".pcapng",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
    ".kdbx",
)
_SENSITIVE_FILENAMES = {
    "credentials.json",
    "service-account.json",
    "service_account.json",
    "secret.json",
    "secrets.json",
}
_SENSITIVE_DIRECTORIES = {".secrets", "private_data", "private-data"}
_SECRET_PATTERNS = (
    (
        "private key",
        re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "OpenAI-compatible secret-shaped token",
        re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    ),
    ("GitHub token", re.compile(rb"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}")),
    ("GitHub fine-grained token", re.compile(rb"github_pat_[A-Za-z0-9_]{20,}")),
    ("AWS access key", re.compile(rb"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])")),
    (
        "Bearer authorization value",
        re.compile(
            rb"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/=-]{16,}"
        ),
    ),
    (
        "non-empty API key or token assignment",
        re.compile(
            rb"(?m)^[ \t]*(?:LLM_API_KEY|PPIO_API_KEY|OPENAI_API_KEY|API_KEY|"
            rb"SECRET_KEY|ACCESS_TOKEN|AUTH_TOKEN)[ \t]*=[ \t]*[\"']?"
            rb"[A-Za-z0-9._~+/=-]{16,}"
        ),
    ),
)


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str


def path_violation(path: str) -> str | None:
    normalized = PurePosixPath(path.replace("\\", "/"))
    name = normalized.name.casefold()
    parts = {part.casefold() for part in normalized.parts[:-1]}

    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "local environment file"
    if name in _SENSITIVE_FILENAMES:
        return "credential/secret filename"
    if any(name.endswith(suffix) for suffix in _DATABASE_SUFFIXES):
        return "local database or journal"
    if any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES):
        return "log, trace, credential, or network-capture file"
    if parts & _SENSITIVE_DIRECTORIES:
        return "private-data directory"
    return None


def content_violations(content: bytes) -> list[str]:
    reasons: list[str] = []
    if content.startswith(b"SQLite format 3\x00"):
        reasons.append("SQLite database content")
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(content):
            reasons.append(label)
    return reasons


def _git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _selected_paths(mode: str) -> list[str]:
    if mode == "staged":
        raw = _git_bytes(
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
        )
    else:
        raw = _git_bytes("ls-files", "-z")
    return [item.decode("utf-8") for item in raw.split(b"\x00") if item]


def _index_content(path: str) -> bytes:
    return _git_bytes("show", f":{path}")


def scan_index(mode: str) -> list[Finding]:
    findings: list[Finding] = []
    for path in _selected_paths(mode):
        path_reason = path_violation(path)
        if path_reason:
            findings.append(Finding(path, path_reason))
            continue

        content = _index_content(path)
        if len(content) > MAX_TRACKED_FILE_BYTES:
            findings.append(
                Finding(
                    path,
                    f"file is larger than {MAX_TRACKED_FILE_BYTES // (1024 * 1024)} MiB",
                )
            )
            continue
        findings.extend(Finding(path, reason) for reason in content_violations(content))
    return findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Block sensitive or unsafe files before they enter PocketLab history."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="Scan staged additions/changes.")
    mode.add_argument("--tracked", action="store_true", help="Scan every tracked index file.")
    return parser


def cli() -> int:
    args = build_parser().parse_args()
    mode = "staged" if args.staged else "tracked"
    try:
        findings = scan_index(mode)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        print(f"Git safety scan could not inspect the index: {detail}")
        return 2

    if findings:
        print("Git safety gate blocked the operation:")
        for finding in findings:
            print(f"- {finding.path}: {finding.reason}")
        print("Keep the file local, remove it from the index, or sanitize it before committing.")
        return 1

    print(f"Git safety gate passed ({mode}, {len(_selected_paths(mode))} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
