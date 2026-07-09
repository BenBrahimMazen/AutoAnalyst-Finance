"""Agent 5 — Risk Detector.

Flag detection is intentionally rule-based and deterministic (no LLM). Each rule
is a small predicate over the shared state. The LLM is used only to synthesize
the final risk paragraph so the score is reproducible and auditable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.prompts.risk import RISK_PROMPT
from src.state.schema import AnalysisState
from src.tools.llm import get_llm, llm_available

logger = logging.getLogger(__name__)

# Points contributed by each severity.
_SEVERITY_POINTS: dict[str, int] = {"high": 25, "medium": 15, "low": 5}


def _de(state: AnalysisState) -> float:
    """Debt-to-equity helper (0 when missing)."""
    try:
        return float((state.get("liquidity_ratios", {}) or {}).get("debt_to_equity") or 0)
    except (TypeError, ValueError):
        return 0.0


def _fcf(state: AnalysisState) -> float:
    """Free cash flow helper (0 when missing)."""
    try:
        return float((state.get("liquidity_ratios", {}) or {}).get("free_cash_flow") or 0)
    except (TypeError, ValueError):
        return 0.0


def _current_ratio(state: AnalysisState) -> float:
    """Current ratio helper (treat missing as 999 = no flag)."""
    val = (state.get("liquidity_ratios", {}) or {}).get("current_ratio")
    return float(val) if val is not None else 999.0


def _interest_coverage(state: AnalysisState) -> float:
    """Interest coverage helper (999 = no flag when missing)."""
    val = (state.get("liquidity_ratios", {}) or {}).get("interest_coverage")
    return float(val) if val is not None else 999.0


# Each rule: name, severity, human-readable message, and a predicate over state.
RED_FLAG_RULES: list[dict] = [
    {
        "name": "High debt burden",
        "severity": "high",
        "check": "debt_to_equity > 200",
        "message": "Debt-to-equity > 200% — high leverage risk in a rising rate environment",
        "fn": lambda s: _de(s) > 200,
    },
    {
        "name": "Negative free cash flow",
        "severity": "high",
        "check": "free_cash_flow < 0",
        "message": "Negative FCF — company burning cash, may need external financing",
        "fn": lambda s: _fcf(s) < 0,
    },
    {
        "name": "Current ratio below 1",
        "severity": "high",
        "check": "current_ratio < 1.0",
        "message": "Current ratio < 1 — short-term liabilities exceed short-term assets",
        "fn": lambda s: 0 < _current_ratio(s) < 1.0,
    },
    {
        "name": "Poor interest coverage",
        "severity": "high",
        "check": "interest_coverage < 2",
        "message": "Interest coverage < 2x — earnings may not cover debt service",
        "fn": lambda s: 0 < _interest_coverage(s) < 2.0,
    },
    {
        "name": "Declining revenue",
        "severity": "medium",
        "check": "revenue_growth_yoy < -5%",
        "message": "Revenue declining >5% YoY — demand or competitive pressure signal",
        "fn": lambda s: float((s.get("profitability_ratios", {}) or {}).get("revenue_growth_yoy") or 0) < -5,
    },
    {
        "name": "Compressed margins",
        "severity": "medium",
        "check": "net_margin < 3%",
        "message": "Net margin < 3% — limited pricing power or significant cost pressure",
        "fn": lambda s: 0 <= float((s.get("profitability_ratios", {}) or {}).get("net_margin") or 999) < 3,
    },
    {
        "name": "Negative news sentiment",
        "severity": "medium",
        "check": "aggregate_sentiment == bearish",
        "message": "Predominantly negative news sentiment — reputational or operational concerns",
        "fn": lambda s: s.get("aggregate_sentiment") == "bearish",
    },
    {
        "name": "Expensive valuation",
        "severity": "low",
        "check": "pe_ratio > 50",
        "message": "P/E > 50 — priced for perfection, vulnerable to any earnings miss",
        "fn": lambda s: float((s.get("valuation_ratios", {}) or {}).get("pe_ratio") or 0) > 50,
    },
    {
        "name": "Negative ROE",
        "severity": "medium",
        "check": "roe < 0",
        "message": "Negative ROE — company destroying shareholder value",
        "fn": lambda s: float((s.get("profitability_ratios", {}) or {}).get("roe") or 0) < 0,
    },
]


def _compute_risk_score(flags: list[dict]) -> int:
    """Sum severity points across triggered flags, capped at 100."""
    total = sum(_SEVERITY_POINTS.get(f.get("severity", "low"), 5) for f in flags)
    return min(total, 100)


def _synthesize_risk_summary(state: AnalysisState, flags: list[dict], risk_score: int) -> str:
    """Ask GPT-4o-mini for a 3-paragraph risk assessment, with a safe fallback."""
    macro = state.get("macro_data", {}) or {}
    flags_list = "\n".join(
        f"- [{f['severity'].upper()}] {f['name']}: {f['message']}" for f in flags
    ) or "- No red flags triggered."
    topics = ", ".join(state.get("key_topics", []) or []) or "N/A"

    try:
        prompt = RISK_PROMPT.format(
            company=state.get("company_name") or state.get("ticker", ""),
            ticker=state.get("ticker", ""),
            fed_rate=float(macro.get("fed_funds_rate") or 0.0),
            yield_10y=float(macro.get("us_10y_yield") or 0.0),
            cpi=float(macro.get("cpi_yoy") or 0.0),
            gdp=float(macro.get("gdp_growth") or 0.0),
            n_flags=len(flags),
            risk_score=risk_score,
            flags_list=flags_list,
            sentiment=state.get("aggregate_sentiment", "neutral"),
            topics=topics,
        )
    except (KeyError, ValueError) as exc:
        logger.warning("Risk prompt formatting failed: %s", exc)
        return flags_list

    if not llm_available():
        logger.warning("OPENAI_API_KEY missing — returning rule-based risk summary.")
        return flags_list
    try:
        return str(get_llm(temperature=0.3).invoke(prompt).content).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Risk synthesis LLM call failed: %s", exc)
        return flags_list


def run_risk_detector(state: AnalysisState) -> AnalysisState:
    """Evaluate every red-flag rule and synthesize a risk summary.

    Args:
        state: Pipeline state produced by the fundamental + sentiment agents.

    Returns:
        Updated state with ``red_flags``, ``risk_score`` and ``risk_summary``.
    """
    errors: list[str] = list(state.get("errors", []))
    completed_steps: list[str] = list(state.get("completed_steps", []))

    triggered_flags: list[dict] = []
    for rule in RED_FLAG_RULES:
        fn: Callable[[AnalysisState], bool] = rule["fn"]  # type: ignore[assignment]
        try:
            if bool(fn(state)):
                triggered_flags.append({
                    "name": rule["name"],
                    "severity": rule["severity"],
                    "check": rule["check"],
                    "message": rule["message"],
                })
        except Exception as exc:  # noqa: BLE001 - missing data must not crash detection
            logger.debug("Rule %s skipped: %s", rule.get("name"), exc)

    risk_score = _compute_risk_score(triggered_flags)
    risk_summary = _synthesize_risk_summary(state, triggered_flags, risk_score)

    completed_steps.append("risk_detector")

    return {
        **state,
        "red_flags": triggered_flags,
        "risk_score": risk_score,
        "risk_summary": risk_summary,
        "errors": errors,
        "completed_steps": completed_steps,
    }


if __name__ == "__main__":
    _empty = AnalysisState(
        ticker="AAPL", company_name="Apple Inc.", analysis_depth="quick",
        price_history={}, financial_statements={}, macro_data={}, peer_tickers=[],
        valuation_ratios={"pe_ratio": 55}, profitability_ratios={"net_margin": 2.0, "roe": -1.0},
        liquidity_ratios={"debt_to_equity": 250, "current_ratio": 0.8, "free_cash_flow": -1e9},
        dcf_estimate={}, peer_comparison={}, fundamental_interpretation="", news_articles=[],
        aggregate_sentiment="bearish", sentiment_positive_avg=0.0, sentiment_negative_avg=0.0,
        key_topics=[], red_flags=[], risk_score=0, risk_summary="",
        executive_summary="", full_report_markdown="", pdf_path="",
        messages=[], errors=[], completed_steps=[],
    )
    _out = run_risk_detector(_empty)
    print("Risk score:", _out["risk_score"])
    for _f in _out["red_flags"]:
        print(f"  [{_f['severity']}] {_f['name']}")
