from __future__ import annotations

import hashlib
from pathlib import Path


def normalized_text_bytes(path: Path) -> bytes:
    """Read text evidence with repository-independent newline semantics."""

    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def file_sha256(path: Path) -> str:
    """Hash the exact bytes of a binary or otherwise opaque file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text_sha256(path: Path) -> str:
    """Hash text content while treating LF and CRLF checkouts identically."""

    return hashlib.sha256(normalized_text_bytes(path)).hexdigest()


def source_file_sha256(path: Path, media_type: str) -> str:
    """Hash textual evidence canonically and binary evidence byte-for-byte."""

    normalized_media_type = media_type.partition(";")[0].strip().lower()
    if (
        normalized_media_type.startswith("text/")
        or normalized_media_type == "application/json"
        or normalized_media_type.endswith("+json")
    ):
        return normalized_text_sha256(path)
    return file_sha256(path)
