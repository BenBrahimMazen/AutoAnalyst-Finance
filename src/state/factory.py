"""Factory for constructing an empty, fully-keyed :class:`AnalysisState`.

LangGraph requires every TypedDict key to be present in the initial state, so we
centralize the "blank slate" construction here. Both the CLI and the tests use it.
"""

from __future__ import annotations

from src.state.schema import AnalysisState


def make_empty_state(ticker: str, analysis_depth: str = "quick") -> AnalysisState:
    """Return a fresh ``AnalysisState`` with all keys initialized.

    Args:
        ticker: Stock symbol to analyze.
        analysis_depth: ``"quick"`` or ``"deep"``.

    Returns:
        A dict satisfying the full :class:`AnalysisState` TypedDict, ready to
        pass to ``graph.invoke``.
    """
    return AnalysisState(
        ticker=ticker,
        company_name="",
        analysis_depth=analysis_depth,
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
