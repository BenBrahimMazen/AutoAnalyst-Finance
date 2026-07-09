"""PDF report generator (markdown -> HTML -> WeasyPrint).

Falls back to writing a ``.md`` file if WeasyPrint (or its pango/cairo system
dependencies) is unavailable, so the pipeline always produces a deliverable.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import markdown as md

logger = logging.getLogger(__name__)


def _prepare_gtk_for_weasyprint() -> None:
    """Make the GTK3 runtime DLLs discoverable so WeasyPrint can import.

    WeasyPrint loads libgobject/pango/cairo via ctypes at import time. On
    Windows these ship in the GTK3-Runtime ``bin`` folder; if that folder is
    not on the process PATH (common when the process started before the runtime
    was installed, or via an IDE/agent shell), WeasyPrint fails with
    "cannot load library 'libgobject-2.0-0'". We locate the runtime and register
    it with the OS DLL loader (``os.add_dll_directory``) plus prepend it to
    PATH. A manual override is honored via the ``WEASYPRINT_GTK_BIN`` env var.
    Safe no-op on non-Windows or when the runtime is absent.
    """
    if sys.platform != "win32":
        return
    candidates: list[Path] = []
    override = os.getenv("WEASYPRINT_GTK_BIN")
    if override:
        candidates.append(Path(override))
    for key in ("ProgramFiles", "ProgramFiles(x86)"):
        pf = Path(os.environ[key]) if os.environ.get(key) else None
        if not pf:
            continue
        candidates.append(pf / "GTK3-Runtime Win64" / "bin")
        candidates.append(pf / "GTK3-Runtime" / "bin")
    for cand in candidates:
        try:
            if cand.is_dir() and (cand / "libgobject-2.0-0.dll").is_file():
                try:
                    os.add_dll_directory(str(cand))  # type: ignore[attr-defined]
                except (AttributeError, OSError):  # pragma: no cover - <3.8 or weird
                    pass
                os.environ["PATH"] = str(cand) + os.pathsep + os.environ.get("PATH", "")
                logger.info("Registered GTK3 runtime DLL directory: %s", cand)
                return
        except Exception as exc:  # noqa: BLE001
            logger.debug("GTK candidate %s unusable: %s", cand, exc)


# Register GTK DLLs before any `from weasyprint import ...` happens downstream.
_prepare_gtk_for_weasyprint()

# Styled HTML shell: dark navy header, clean tables, serif body.
_HTML_TEMPLATE: str = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 2cm; }}
  body {{ font-family: Georgia, 'Times New Roman', serif; color: #1f2937; line-height: 1.5; font-size: 11pt; }}
  h1, h2, h3 {{ font-family: 'Helvetica Neue', Arial, sans-serif; color: #0f2748; }}
  h1 {{ background: #0f2748; color: #ffffff; padding: 16px 20px; border-radius: 6px; font-size: 20pt; margin-top: 0; }}
  h2 {{ border-bottom: 2px solid #0f2748; padding-bottom: 4px; margin-top: 24px; font-size: 15pt; }}
  h3 {{ color: #1d4ed8; font-size: 12pt; margin-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10pt; }}
  th {{ background: #0f2748; color: #fff; text-align: left; padding: 8px; }}
  td {{ border: 1px solid #d1d5db; padding: 6px 8px; }}
  tr:nth-child(even) td {{ background: #f3f4f6; }}
  code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }}
  blockquote {{ border-left: 4px solid #1d4ed8; margin: 10px 0; padding: 6px 14px; background: #eff6ff; }}
  a {{ color: #1d4ed8; text-decoration: none; word-break: break-all; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _reports_dir() -> Path:
    """Return (and create) the reports directory at the project root."""
    reports = Path(__file__).resolve().parents[2] / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return reports


def generate_pdf(markdown_text: str, ticker: str) -> str:
    """Convert ``markdown_text`` to a PDF report.

    Args:
        markdown_text: Full report in markdown.
        ticker: Ticker symbol, used in the output filename.

    Returns:
        Absolute path to the generated PDF (or ``.md`` fallback).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_ticker = "".join(c for c in (ticker or "report") if c.isalnum() or c in (".", "_", "-"))
    reports = _reports_dir()

    html_body = md.markdown(markdown_text, extensions=["tables", "fenced_code"])
    html = _HTML_TEMPLATE.format(body=html_body)

    pdf_path = reports / f"{safe_ticker}_{timestamp}.pdf"
    try:
        from weasyprint import HTML

        HTML(string=html).write_pdf(str(pdf_path))
        logger.info("PDF written to %s", pdf_path)
        return str(pdf_path)
    except Exception as exc:  # noqa: BLE001 - WeasyPrint often missing system deps
        logger.warning("WeasyPrint failed (%s) — saving markdown fallback.", exc)
        md_path = reports / f"{safe_ticker}_{timestamp}.md"
        md_path.write_text(markdown_text, encoding="utf-8")
        return str(md_path)
