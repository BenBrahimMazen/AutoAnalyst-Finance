"""Agent 6 — Report Writer.

Asks GPT-4o-mini for an executive summary and an investment conclusion, then
assembles the full markdown report and renders it to PDF via WeasyPrint.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.prompts.report import CONCLUSION_PROMPT, EXECUTIVE_SUMMARY_PROMPT
from src.state.schema import AnalysisState
from src.tools.llm import get_llm, llm_available
from src.tools.pdf_generator import generate_pdf

logger = logging.getLogger(__name__)


def _fmt(value: Any) -> str:
    """Format a value for display in a markdown table cell."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _kv_table(data: dict, headers: tuple[str, str] = ("Metric", "Value")) -> str:
    """Render a 2-column markdown table from a flat dict."""
    if not data:
        return "_(No data available.)_"
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    for key, value in data.items():
        lines.append(f"| {key} | {_fmt(value)} |")
    return "\n".join(lines)


def _peer_table(peer_comparison: dict) -> str:
    """Render the peer comparison as a markdown table."""
    peers = (peer_comparison or {}).get("peer_data", [])
    if not peers:
        return "_(No peer data available.)_"
    lines = ["| Peer | P/E | P/B | Net Margin |", "|---|---|---|---|"]
    for peer in peers:
        margin = peer.get("net_margin")
        lines.append(
            f"| {peer.get('ticker', '—')} | {_fmt(peer.get('pe_ratio'))} | "
            f"{_fmt(peer.get('pb_ratio'))} | "
            f"{_fmt(margin)+'%' if margin is not None else '—'} |"
        )
    return "\n".join(lines)


def _data_summary(state: AnalysisState) -> str:
    """Compact text summary of the key metrics for the LLM prompts."""
    val = state.get("valuation_ratios", {}) or {}
    prof = state.get("profitability_ratios", {}) or {}
    liq = state.get("liquidity_ratios", {}) or {}
    return (
        f"P/E {val.get('pe_ratio')}, P/B {val.get('pb_ratio')}, "
        f"net margin {prof.get('net_margin')}%, ROE {prof.get('roe')}%, "
        f"debt/equity {liq.get('debt_to_equity')}, "
        f"revenue growth {prof.get('revenue_growth_yoy')}%, "
        f"risk score {state.get('risk_score', 0)}/100, "
        f"sentiment {state.get('aggregate_sentiment')}."
    )


def _generate_executive_summary(state: AnalysisState) -> str:
    """Ask GPT-4o-mini for a 3-sentence executive summary, with fallback."""
    if not llm_available():
        logger.warning("OPENAI_API_KEY missing — using boilerplate executive summary.")
        return (
            f"{state.get('company_name', state.get('ticker', ''))} carries a "
            f"risk score of {state.get('risk_score', 0)}/100 with "
            f"{state.get('aggregate_sentiment', 'neutral')} news sentiment."
        )
    try:
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            company=state.get("company_name") or state.get("ticker", ""),
            ticker=state.get("ticker", ""),
            data_summary=_data_summary(state),
        )
        return str(get_llm(temperature=0.3).invoke(prompt).content).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Executive summary LLM call failed: %s", exc)
        return f"Executive summary unavailable (LLM error). {_data_summary(state)}"


def _generate_conclusion(state: AnalysisState) -> str:
    """Ask GPT-4o-mini for the investment conclusion, with fallback."""
    dcf = state.get("dcf_estimate", {}) or {}
    price = (state.get("price_history", {}) or {}).get("current_price")
    if not llm_available():
        return (
            f"We rate {state.get('ticker', '')} Neutral pending full LLM analysis. "
            f"Risk score {state.get('risk_score', 0)}/100."
        )
    try:
        prompt = CONCLUSION_PROMPT.format(
            company=state.get("company_name") or state.get("ticker", ""),
            ticker=state.get("ticker", ""),
            risk_score=state.get("risk_score", 0),
            sentiment=state.get("aggregate_sentiment", "neutral"),
            dcf_value=dcf.get("intrinsic_value_per_share"),
            current_price=price,
        )
        return str(get_llm(temperature=0.3).invoke(prompt).content).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Conclusion LLM call failed: %s", exc)
        return "Investment conclusion unavailable (LLM error)."


def _build_markdown(state: AnalysisState, executive_summary: str, conclusion: str) -> str:
    """Assemble the full report markdown from the populated state."""
    company = state.get("company_name") or state.get("ticker", "")
    ticker = state.get("ticker", "")
    today = datetime.now().strftime("%Y-%m-%d")

    valuation = state.get("valuation_ratios", {}) or {}
    profitability = state.get("profitability_ratios", {}) or {}
    dcf = state.get("dcf_estimate", {}) or {}
    peer_comparison = state.get("peer_comparison", {}) or {}
    articles = state.get("news_articles", []) or []
    red_flags = state.get("red_flags", []) or []
    fundamental_interp = state.get("fundamental_interpretation", "")

    current_price = (state.get("price_history", {}) or {}).get("current_price")
    intrinsic = dcf.get("intrinsic_value_per_share")

    sections: list[str] = [
        f"# {company} ({ticker}) — Investment Research",
        f"**Date:** {today} &nbsp;|&nbsp; **Risk score:** {state.get('risk_score', 0)}/100 "
        f"&nbsp;|&nbsp; **Sentiment:** {state.get('aggregate_sentiment', 'neutral')} "
        f"&nbsp;|&nbsp; **Current price:** {_fmt(current_price)}",
        "",
        "## Executive Summary",
        executive_summary,
        "",
        "## Financial Performance",
        "",
        "### Valuation ratios",
        _kv_table(valuation),
        "",
        "### Profitability & growth ratios",
        _kv_table(profitability),
        "",
    ]

    if fundamental_interp:
        sections += ["### Fundamental interpretation", fundamental_interp, ""]

    if intrinsic is not None:
        upside = None
        if current_price:
            try:
                upside = ((intrinsic - float(current_price)) / float(current_price)) * 100.0
            except (TypeError, ZeroDivisionError):
                upside = None
        upside_str = f" ({upside:+.1f}% vs current)" if upside is not None else ""
        sections += [
            "## DCF Valuation",
            f"Intrinsic value per share: **${intrinsic:,.2f}**{upside_str}.",
            "",
            _kv_table(dcf.get("assumptions", {}), headers=("Assumption", "Value")),
            "",
        ]
    elif dcf.get("note"):
        sections += ["## DCF Valuation", f"_{dcf['note']}_", ""]

    sections += [
        "## Peer Comparison",
        _peer_table(peer_comparison),
        f"\nSector averages — P/E: {_fmt(peer_comparison.get('sector_avg_pe'))}, "
        f"P/B: {_fmt(peer_comparison.get('sector_avg_pb'))}, "
        f"net margin: {_fmt(peer_comparison.get('sector_avg_margin'))}%.",
        "",
    ]

    sections.append("## News & Sentiment")
    if articles:
        for art in articles[:5]:
            sentiment_label = art.get("sentiment", "—")
            title = art.get("title", "Untitled")
            url = art.get("url", "")
            link = f"[link]({url})" if url else ""
            sections.append(f"- **({sentiment_label})** {title} {link}")
    else:
        sections.append("_(No news articles retrieved.)_")
    topics = state.get("key_topics", []) or []
    if topics:
        sections.append(f"\n**Key topics:** {', '.join(topics)}")
    sections.append("")

    sections.append("## Risk Assessment")
    if red_flags:
        lines = ["| Severity | Flag | Detail |", "|---|---|---|"]
        for flag in red_flags:
            lines.append(f"| {flag.get('severity', '').upper()} | {flag.get('name', '')} | {flag.get('message', '')} |")
        sections.append("\n".join(lines))
    else:
        sections.append("_(No red flags triggered.)_")
    sections.append("")
    sections.append(state.get("risk_summary", ""))
    sections.append("")

    sections.append("## Investment Conclusion")
    sections.append("> " + conclusion.replace("\n\n", "\n>\n> "))

    return "\n".join(sections)


def run_report_writer(state: AnalysisState) -> AnalysisState:
    """Produce the executive summary, full markdown report, and PDF.

    Args:
        state: Pipeline state after risk detection.

    Returns:
        Updated state with ``executive_summary``, ``full_report_markdown`` and
        ``pdf_path``.
    """
    errors: list[str] = list(state.get("errors", []))
    completed_steps: list[str] = list(state.get("completed_steps", []))

    executive_summary = _generate_executive_summary(state)
    conclusion = _generate_conclusion(state)
    full_report_markdown = _build_markdown(state, executive_summary, conclusion)

    pdf_path = ""
    try:
        pdf_path = generate_pdf(full_report_markdown, state.get("ticker", "report"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"report_writer: PDF generation failed — {exc}")
        logger.warning("PDF generation failed: %s", exc)

    completed_steps.append("report_writer")

    return {
        **state,
        "executive_summary": executive_summary,
        "full_report_markdown": full_report_markdown,
        "pdf_path": pdf_path,
        "errors": errors,
        "completed_steps": completed_steps,
    }


if __name__ == "__main__":
    _demo = AnalysisState(
        ticker="AAPL", company_name="Apple Inc.", analysis_depth="quick",
        price_history={"current_price": 225.0}, financial_statements={}, macro_data={}, peer_tickers=[],
        valuation_ratios={"pe_ratio": 32.0, "pb_ratio": 45.0, "market_cap": 3.4e12},
        profitability_ratios={"net_margin": 25.0, "roe": 150.0, "revenue_growth_yoy": 2.0},
        liquidity_ratios={"debt_to_equity": 145, "free_cash_flow": 1e10},
        dcf_estimate={"intrinsic_value_per_share": 210.0, "assumptions": {"wacc": 0.09}},
        peer_comparison={"peer_data": [{"ticker": "MSFT", "pe_ratio": 35, "pb_ratio": 12, "net_margin": 35.0}],
                         "sector_avg_pe": 33.0, "sector_avg_pb": 10.0, "sector_avg_margin": 30.0},
        fundamental_interpretation="Apple maintains strong margins and returns capital aggressively.",
        news_articles=[{"title": "Apple beats earnings", "url": "https://example.com", "sentiment": "positive"}],
        aggregate_sentiment="bullish", sentiment_positive_avg=0.7, sentiment_negative_avg=0.1,
        key_topics=["AI", "iPhone", "Services"],
        red_flags=[{"severity": "high", "name": "High debt burden", "message": "D/E > 200"}],
        risk_score=25, risk_summary="Moderate risk profile overall.",
        executive_summary="", full_report_markdown="", pdf_path="",
        messages=[], errors=[], completed_steps=[],
    )
    _out = run_report_writer(_demo)
    print("PDF path:", _out["pdf_path"])
    print("Report length:", len(_out["full_report_markdown"]), "chars")
