"""CLI entry point for running a full AutoAnalyst Finance analysis.

Usage:
    python scripts/run_analysis.py --ticker AAPL
    python scripts/run_analysis.py --ticker AAPL --depth deep
    python scripts/run_analysis.py --ticker BNP.PA --open

Builds the LangGraph graph, seeds an empty state, runs the pipeline, prints a
summary, and optionally opens the generated PDF.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root (parent of /scripts) is importable as `src`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load environment variables (API keys) from .env before src modules read them.
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv optional
    pass

from src.agents.orchestrator import build_graph  # noqa: E402
from src.state.factory import make_empty_state  # noqa: E402

# ANSI colors for the console summary.
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"


def _enable_windows_ansi() -> None:
    """Enable ANSI escape sequences on legacy Windows consoles (best effort)."""
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:  # noqa: BLE001
            pass


def _risk_color(score: int) -> str:
    """Return the ANSI color code for a risk score band."""
    if score < 40:
        return _GREEN
    if score <= 75:
        return _YELLOW
    return _RED


def main() -> int:
    """Parse args, run the pipeline, print a summary. Returns an exit code."""
    _enable_windows_ansi()

    parser = argparse.ArgumentParser(description="Run an AutoAnalyst Finance analysis.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL or BNP.PA")
    parser.add_argument("--depth", default="quick", choices=["quick", "deep"], help="Analysis depth")
    parser.add_argument("--open", action="store_true", help="Open the generated PDF when done")
    args = parser.parse_args()

    graph = build_graph()
    initial_state = make_empty_state(args.ticker, analysis_depth=args.depth)
    config = {"configurable": {"thread_id": args.ticker}}

    print(f"Analyzing {args.ticker} (depth={args.depth}) ...\n")
    result = graph.invoke(initial_state, config=config)

    company = result.get("company_name") or args.ticker
    risk_score: int = result.get("risk_score", 0)
    sentiment = result.get("aggregate_sentiment", "neutral")
    red_flags = result.get("red_flags", []) or []
    pdf_path = result.get("pdf_path", "")
    errors = result.get("errors", []) or []
    completed = result.get("completed_steps", []) or []

    color = _risk_color(risk_score)
    print("=" * 60)
    print(f"  {company} ({args.ticker})")
    print("=" * 60)
    print(f"  Risk score : {color}{risk_score}/100{_RESET}")
    print(f"  Sentiment  : {sentiment}")
    print(f"  Red flags  : {len(red_flags)}")
    print(f"  PDF report : {pdf_path or '(none)'}")
    if completed:
        print(f"  Steps done : {', '.join(completed)}")
    if errors:
        print(f"\n  Warnings ({len(errors)}):")
        for err in errors[:10]:
            print(f"    - {err}")
    print("=" * 60)

    if args.open and pdf_path:
        try:
            if sys.platform.startswith("win"):
                os.startfile(pdf_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{pdf_path}"')
            else:
                os.system(f'xdg-open "{pdf_path}"')
        except Exception as exc:  # noqa: BLE001
            print(f"Could not open PDF automatically: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
