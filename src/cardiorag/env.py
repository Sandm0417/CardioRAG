"""Optional local-environment loader. Never required for import or tests."""

from __future__ import annotations

import os
from pathlib import Path


def load_env(project_root: Path | None = None) -> Path | None:
    """Load `.env` if present. Missing file is not an error.

    Existing process environment variables are not overwritten.
    """
    root = project_root or Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
    return env_path
