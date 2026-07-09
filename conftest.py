"""Shared pytest configuration.

Loads environment variables from ``.env`` (if present) so tests and the CLI can
find API keys, and ensures the project root is importable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project root importable as ``src``.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env if python-dotenv is available.
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv optional at test time
    pass

# Ensure reports dir exists for tests that generate output.
(_ROOT / "reports").mkdir(exist_ok=True)
