"""Secret/credential loading from `.env.local` (gitignored).

Credentials are NEVER hardcoded. They live in `.env.local` at the project root and are read into the
process environment on demand. Used for MOSDAC SFTP, HuggingFace, and OpenAI.
"""
from __future__ import annotations

import os
from pathlib import Path

# project root = two levels up from this file (src/data/env.py -> ps12/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_LOCAL = PROJECT_ROOT / ".env.local"

_loaded = False


def load_env(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from `.env.local` into os.environ. Idempotent.

    Returns the dict of keys found in the file (values omitted from the return for safety).
    """
    global _loaded
    path = Path(path) if path else ENV_LOCAL
    found: dict[str, str] = {}
    if not path.exists():
        _loaded = True
        return found
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value[:1] in ('"', "'"):                 # quoted value: take inside the quotes
            q = value[0]
            end = value.find(q, 1)
            value = value[1:end] if end != -1 else value[1:]
        elif "#" in value:                          # unquoted: drop inline comment
            value = value.split("#", 1)[0].strip()
        found[key] = "***"  # do not echo real values
        if override or key not in os.environ:
            os.environ[key] = value
    _loaded = True
    return found


def get_secret(name: str, required: bool = False, default: str | None = None) -> str | None:
    """Return a secret from the environment (loading `.env.local` first)."""
    if not _loaded:
        load_env()
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"Required secret '{name}' is not set. Add it to {ENV_LOCAL} "
            f"(e.g. `{name}=...`) — this file is gitignored and never committed."
        )
    return value


def mosdac_credentials() -> tuple[str, str]:
    """Return (username, password) for MOSDAC SFTP, or raise a clear error."""
    user = get_secret("MOSDAC_USERNAME", required=True)
    pwd = get_secret("MOSDAC_PASSWORD", required=True)
    return user, pwd  # type: ignore[return-value]
