"""Agent and end-to-end tests (Steps 4, 5, 8, 10).

Tests that hit the network (yfinance / Tavily / OpenAI) are marked
``@pytest.mark.network`` so they can be deselected in offline CI. The full
pipeline test additionally requires API keys, so it is excluded by default in CI
(``-k "not full_pipeline"``).
"""

from __future__ import annotations

import pytest

from src.agents.data_collector import run_data_collector
from src.agents.fundamental import run_fundamental_analyst
from src.agents.orchestrator import build_graph
from src.agents.risk_detector import run_risk_detector
from src.state.factory import make_empty_state


@pytest.mark.network
def test_data_collector_aapl() -> None:
    """End-to-end test: run data collector on AAPL."""
    state = make_empty_state("AAPL")
    result = run_data_collector(state)
    assert result["company_name"] == "Apple Inc."
    assert result["price_history"]["current_price"] > 0
    assert "data_collector" in result["completed_steps"]


@pytest.mark.network
def test_data_collector_invalid_ticker() -> None:
    """Invalid ticker should not crash — should add to errors."""
    state = make_empty_state("INVALIDXXX999")
    result = run_data_collector(state)
    assert len(result["errors"]) > 0


@pytest.mark.network
def test_fundamental_analyst_aapl() -> None:
    """Test fundamental analyst on pre-fetched AAPL data."""
    state = run_data_collector(make_empty_state("AAPL"))
    result = run_fundamental_analyst(state)
    assert result["valuation_ratios"]["pe_ratio"] is not None
    assert result["dcf_estimate"] is not None
    assert len(result["fundamental_interpretation"]) > 50
    assert "fundamental_analyst" in result["completed_steps"]


def test_risk_detector_no_crash() -> None:
    """Risk detector should run on any state without crashing."""
    state = make_empty_state("AAPL")
    state["valuation_ratios"] = {}
    state["profitability_ratios"] = {}
    state["liquidity_ratios"] = {}
    result = run_risk_detector(state)
    assert 0 <= result["risk_score"] <= 100


def test_risk_detector_scores_flags() -> None:
    """Triggered flags contribute the expected severity points."""
    state = make_empty_state("AAPL")
    state["valuation_ratios"] = {"pe_ratio": 60}            # low: 5
    state["profitability_ratios"] = {"net_margin": 2.0, "roe": -1.0, "revenue_growth_yoy": -10}
    state["liquidity_ratios"] = {
        "debt_to_equity": 250,   # high: 25
        "free_cash_flow": -1e9,  # high: 25
        "current_ratio": 0.5,    # high: 25
    }
    state["aggregate_sentiment"] = "bearish"
    result = run_risk_detector(state)
    assert result["risk_score"] == 100  # 25+25+25+15+15+15+5 = 125 -> capped at 100
    names = {f["name"] for f in result["red_flags"]}
    assert "High debt burden" in names


@pytest.mark.network
def test_full_pipeline_aapl() -> None:
    """Run the full LangGraph pipeline on AAPL."""
    graph = build_graph()
    initial_state = make_empty_state("AAPL")
    result = graph.invoke(
        initial_state, config={"configurable": {"thread_id": "test-aapl"}}
    )
    assert "report_writer" in result["completed_steps"]
    assert len(result["full_report_markdown"]) > 500
    assert result["pdf_path"].endswith(".pdf") or result["pdf_path"].endswith(".md")
