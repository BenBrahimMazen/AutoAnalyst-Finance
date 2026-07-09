"""Agent 2 — Data Collector.

Pulls all raw inputs the downstream agents need: price history, quarterly
financial statements, macro indicators, and sector peers. Fills the
corresponding ``AnalysisState`` keys and is robust to per-source failures.
"""

from __future__ import annotations

import logging
from typing import Any

from src.state.schema import AnalysisState
from src.tools.financial_data import FinancialDataFetcher

logger = logging.getLogger(__name__)


def run_data_collector(state: AnalysisState) -> AnalysisState:
    """Collect price, statements, macro data, and peers for ``state["ticker"]``.

    Args:
        state: Current pipeline state; must contain at least ``ticker``.

    Returns:
        Updated state with ``company_name``, ``price_history``,
        ``financial_statements``, ``macro_data`` and ``peer_tickers`` populated.
        Errors are appended to ``state["errors"]`` and ``"data_collector"`` is
        added to ``completed_steps`` on success.
    """
    ticker: str = state.get("ticker", "")
    fetcher = FinancialDataFetcher()

    errors: list[str] = list(state.get("errors", []))
    completed_steps: list[str] = list(state.get("completed_steps", []))

    # 2. Price history
    price_history: dict[str, Any] = {}
    try:
        price_history = fetcher.get_price_history(ticker)
        if not price_history:
            errors.append(f"data_collector: no price history for {ticker}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"data_collector: price history failed — {exc}")
        logger.warning("Price history failed for %s: %s", ticker, exc)

    # 3. Financial statements
    financial_statements: dict[str, Any] = {}
    try:
        financial_statements = fetcher.get_financial_statements(ticker)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"data_collector: statements failed — {exc}")
        logger.warning("Statements failed for %s: %s", ticker, exc)

    # 4. Company name from info, falling back to the ticker.
    info: dict[str, Any] = financial_statements.get("info", {}) or {}
    company_name: str = info.get("longName") or info.get("shortName") or ticker

    # 5. Macro data
    macro_data: dict[str, Any] = {}
    try:
        macro_data = fetcher.get_macro_data()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"data_collector: macro data failed — {exc}")
        logger.warning("Macro data failed for %s: %s", ticker, exc)

    # 6. Sector peers
    sector: str = info.get("sector", "") or ""
    peer_tickers: list[str] = []
    try:
        peer_tickers = fetcher.get_sector_peers(sector, ticker)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"data_collector: peer lookup failed — {exc}")
        logger.warning("Peer lookup failed for %s: %s", ticker, exc)

    completed_steps.append("data_collector")

    return {
        **state,
        "company_name": company_name,
        "price_history": price_history,
        "financial_statements": financial_statements,
        "macro_data": macro_data,
        "peer_tickers": peer_tickers,
        "errors": errors,
        "completed_steps": completed_steps,
    }


if __name__ == "__main__":
    _demo_state = AnalysisState(
        ticker="AAPL",
        company_name="",
        analysis_depth="quick",
        price_history={},
        financial_statements={},
        macro_data={},
        peer_tickers=[],
        valuation_ratios={},
        profitability_ratios={},
        liquidity_ratios={},
        dcf_estimate={},
        peer_comparison={},
        fundamental_interpretation="",
        news_articles=[],
        aggregate_sentiment="neutral",
        sentiment_positive_avg=0.0,
        sentiment_negative_avg=0.0,
        key_topics=[],
        red_flags=[],
        risk_score=0,
        risk_summary="",
        executive_summary="",
        full_report_markdown="",
        pdf_path="",
        messages=[],
        errors=[],
        completed_steps=[],
    )
    _result = run_data_collector(_demo_state)
    print("Company:", _result["company_name"])
    print("Current price:", _result["price_history"].get("current_price"))
    print("Macro:", _result["macro_data"])
    print("Peers:", _result["peer_tickers"])
    print("Completed:", _result["completed_steps"])
    if _result["errors"]:
        print("Errors:", _result["errors"])
