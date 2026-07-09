"""Shared LangGraph state definition.

``AnalysisState`` is the single TypedDict that every agent reads from and writes
to. It is the in-memory "shared memory" of the pipeline. Nodes return a *new*
state dict (``{**state, ...}``) rather than mutating in place, so LangGraph's
checkpointer can diff state across steps.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AnalysisState(TypedDict):
    """Full pipeline state shared across all agents.

    Fields are grouped by the agent that produces them.
    """

    # --- Input -------------------------------------------------------------
    ticker: str
    company_name: str
    analysis_depth: str  # "quick" or "deep"

    # --- Agent 2 outputs (data collector) ---------------------------------
    price_history: dict
    financial_statements: dict
    macro_data: dict
    peer_tickers: list[str]

    # --- Agent 3 outputs (fundamental analyst) ----------------------------
    valuation_ratios: dict
    profitability_ratios: dict
    liquidity_ratios: dict
    dcf_estimate: dict
    peer_comparison: dict
    fundamental_interpretation: str

    # --- Agent 4 outputs (sentiment analyst) ------------------------------
    news_articles: list[dict]
    aggregate_sentiment: str  # "bullish" / "neutral" / "bearish"
    sentiment_positive_avg: float
    sentiment_negative_avg: float
    key_topics: list[str]

    # --- Agent 5 outputs (risk detector) ----------------------------------
    red_flags: list[dict]
    risk_score: int  # 0-100
    risk_summary: str

    # --- Agent 6 outputs (report writer) ----------------------------------
    executive_summary: str
    full_report_markdown: str
    pdf_path: str

    # --- Metadata ----------------------------------------------------------
    messages: Annotated[list, add_messages]
    errors: list[str]
    completed_steps: list[str]
