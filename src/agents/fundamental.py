"""Agent 3 — Fundamental Analyst.

Computes valuation, profitability, and liquidity ratios from yfinance data; runs
a two-stage DCF; compares the company to its sector peers; and asks GPT-4o-mini
for a plain-English interpretation. Every helper degrades to ``None`` on missing
data rather than raising.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from src.prompts.fundamental import FUNDAMENTAL_PROMPT
from src.state.schema import AnalysisState
from src.tools.llm import get_llm, llm_available

logger = logging.getLogger(__name__)

# Row-name aliases used by yfinance statements (these change across versions).
_INCOME_ALIASES: dict[str, list[str]] = {
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"],
    "ebitda": ["Normalized EBITDA", "EBITDA", "EBIT"],
}
_BALANCE_ALIASES: dict[str, list[str]] = {
    "total_assets": ["Total Assets"],
    "stockholders_equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity From Parent Interest"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _latest_row(df: pd.DataFrame, aliases: list[str]) -> float | None:
    """Return the most recent value for the first matching line item.

    Robust to the DataFrame's orientation: yfinance exposes line items as the
    index in some versions and as the columns in others (1.5.x puts dates on
    the index). We look the alias up on either axis, then pick the latest value
    by date label (ISO strings sort chronologically), falling back to the
    positional last entry.

    Args:
        df: Statement DataFrame (line items on one axis, dates on the other).
        aliases: Candidate line-item names to look up.

    Returns:
        The latest numeric value, or ``None`` if unavailable.
    """
    if df is None or getattr(df, "empty", True):
        return None
    for alias in aliases:
        series: pd.Series | None = None
        if alias in df.index:
            series = df.loc[alias]
        elif alias in df.columns:
            series = df[alias]
        if series is None:
            continue
        try:
            series = series.dropna()
            if series.empty:
                continue
            labels = list(series.index)
            # Date labels are ISO strings after serialization ("2026-03-31...");
            # pick the chronologically latest, else the positional last.
            if labels and all(isinstance(lab, str) and lab[:4].isdigit() for lab in labels):
                return float(series[max(labels)])
            return float(series.iloc[-1])
        except Exception:  # noqa: BLE001
            continue
    return None


def _ttm_value(df: pd.DataFrame, aliases: list[str]) -> float | None:
    """Return the trailing-twelve-months value for the first matching line item.

    A flow metric like net income must be measured over the same horizon as the
    stock it is compared against: using a single quarter's net income against
    point-in-time equity or total assets understates ROE/ROA by roughly the number
    of quarters summed. We therefore sum the most recent four quarters (true TTM),
    and annualize — sum × 4 / count — when fewer than four are available so the
    result stays comparable to the balance-sheet snapshot.

    Args:
        df: Statement DataFrame (line items on one axis, dates on the other).
        aliases: Candidate line-item names to look up.

    Returns:
        The TTM (or annualized) numeric value, or ``None`` if unavailable.
    """
    if df is None or getattr(df, "empty", True):
        return None
    for alias in aliases:
        series: pd.Series | None = None
        if alias in df.index:
            series = df.loc[alias]
        elif alias in df.columns:
            series = df[alias]
        if series is None:
            continue
        try:
            series = series.dropna()
            if series.empty:
                continue
            labels = list(series.index)
            # ISO date labels sort chronologically; reindex so the tail is
            # genuinely the latest quarters regardless of source ordering.
            if labels and all(isinstance(lab, str) and lab[:4].isdigit() for lab in labels):
                series = series.reindex(sorted(labels))
            recent = [float(v) for v in series.to_list()[-4:]]
            ttm = float(sum(recent))
            if len(recent) < 4:
                ttm = ttm * 4.0 / len(recent)
            return ttm
        except Exception:  # noqa: BLE001
            continue
    return None


def _statements_to_df(stmt: Any) -> pd.DataFrame:
    """Reconstruct a statement DataFrame from its ``to_dict()`` form.

    Args:
        stmt: Either a DataFrame, a nested dict, or ``None``.

    Returns:
        A DataFrame with line items as the index (possibly empty).
    """
    if stmt is None:
        return pd.DataFrame()
    if isinstance(stmt, pd.DataFrame):
        return stmt
    try:
        return pd.DataFrame.from_dict(stmt, orient="index")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Ratio computations
# ---------------------------------------------------------------------------
def compute_valuation_ratios(info: dict) -> dict:
    """Extract valuation multiples from the yfinance ``info`` dict.

    Args:
        info: ``stock.info`` dictionary.

    Returns:
        Dict with ``pe_ratio``, ``forward_pe``, ``pb_ratio``, ``ps_ratio``,
        ``ev_ebitda`` and ``market_cap``. Missing values are ``None``.
    """
    info = info or {}
    return {
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "pb_ratio": info.get("priceToBook"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "market_cap": info.get("marketCap"),
    }


def compute_profitability_ratios(
    info: dict, income: pd.DataFrame, balance: pd.DataFrame
) -> dict:
    """Compute profitability ratios.

    Args:
        info: ``stock.info`` dictionary.
        income: Quarterly income statement DataFrame.
        balance: Quarterly balance sheet DataFrame.

    Returns:
        Dict with ``net_margin``, ``ebitda_margin``, ``roe``, ``roa``,
        ``revenue_growth_yoy`` and ``earnings_growth_yoy`` (all percentages).
        Returns ``None`` for any ratio that cannot be computed.
    """
    info = info or {}
    revenue = _latest_row(income, _INCOME_ALIASES["revenue"])
    net_income = _latest_row(income, _INCOME_ALIASES["net_income"])
    # TTM net income so ROE/ROA compare a full-year flow to the balance-sheet
    # stock. Latest-quarter net_income is retained for net_margin, where both
    # numerator and denominator are same-period quarterly flows.
    net_income_ttm = _ttm_value(income, _INCOME_ALIASES["net_income"])
    ebitda = _latest_row(income, _INCOME_ALIASES["ebitda"])
    total_assets = _latest_row(balance, _BALANCE_ALIASES["total_assets"])
    equity = _latest_row(balance, _BALANCE_ALIASES["stockholders_equity"])

    def _pct(numer: float | None, denom: float | None) -> float | None:
        if numer is None or not denom:
            return None
        return round(float(numer) / float(denom) * 100.0, 2)

    revenue_growth = info.get("revenueGrowth")
    earnings_growth = info.get("earningsGrowth")

    return {
        "net_margin": _pct(net_income, revenue),
        "ebitda_margin": _pct(ebitda, revenue),
        "roe": _pct(net_income_ttm, equity),
        "roa": _pct(net_income_ttm, total_assets),
        "revenue_growth_yoy": round(float(revenue_growth) * 100.0, 2) if revenue_growth is not None else None,
        "earnings_growth_yoy": round(float(earnings_growth) * 100.0, 2) if earnings_growth is not None else None,
    }


def compute_liquidity_ratios(info: dict, balance: pd.DataFrame) -> dict:
    """Compute liquidity / leverage ratios.

    Args:
        info: ``stock.info`` dictionary.
        balance: Quarterly balance sheet DataFrame.

    Returns:
        Dict with ``current_ratio``, ``debt_to_equity``, ``interest_coverage``,
        ``free_cash_flow`` and ``fcf_yield``. Missing values are ``None``.
    """
    info = info or {}
    current_assets = _latest_row(balance, _BALANCE_ALIASES["current_assets"])
    current_liabilities = _latest_row(balance, _BALANCE_ALIASES["current_liabilities"])
    market_cap = info.get("marketCap")
    free_cash_flow = info.get("freeCashflow")
    ebitda = info.get("operatingCashflow")  # proxy for earnings before interest
    interest_expense = info.get("interestExpense")  # may be absent in info

    current_ratio: float | None = None
    if current_assets is not None and current_liabilities:
        current_ratio = round(float(current_assets) / float(current_liabilities), 2)

    debt_to_equity_raw = info.get("debtToEquity")
    debt_to_equity: float | None = float(debt_to_equity_raw) if debt_to_equity_raw is not None else None

    interest_coverage: float | None = None
    if ebitda and interest_expense:
        try:
            interest_coverage = round(float(ebitda) / abs(float(interest_expense)), 2)
        except (TypeError, ZeroDivisionError):
            interest_coverage = None

    fcf_yield: float | None = None
    if free_cash_flow and market_cap:
        try:
            fcf_yield = round(float(free_cash_flow) / float(market_cap) * 100.0, 2)
        except (TypeError, ZeroDivisionError):
            fcf_yield = None

    return {
        "current_ratio": current_ratio,
        "debt_to_equity": debt_to_equity,
        "interest_coverage": interest_coverage,
        "free_cash_flow": free_cash_flow,
        "fcf_yield": fcf_yield,
    }


def compute_dcf(
    fcf: float,
    growth_rate_5y: float,
    terminal_growth: float,
    wacc: float,
    shares_outstanding: int,
) -> dict:
    """Two-stage discounted cash flow valuation.

    Args:
        fcf: Latest annual free cash flow (absolute dollars).
        growth_rate_5y: Stage-1 annual growth rate (e.g. 0.08 for 8%).
        terminal_growth: Perpetual growth rate after stage 1.
        wacc: Weighted average cost of capital (discount rate).
        shares_outstanding: Diluted shares outstanding.

    Returns:
        Dict with ``intrinsic_value_per_share`` and supporting detail, or a note
        explaining why the DCF is not applicable.
    """
    if not fcf or fcf <= 0 or not shares_outstanding or shares_outstanding <= 0:
        return {"intrinsic_value_per_share": None, "note": "Negative FCF — DCF not applicable"}

    if wacc <= terminal_growth:
        # Avoid divide-by-zero in the terminal value.
        terminal_growth = max(terminal_growth, 0.0)
        wacc = terminal_growth + 0.01

    # Stage 1: explicit 5-year forecast.
    stage1_pv = 0.0
    projected = fcf
    for year in range(1, 6):
        projected *= (1.0 + growth_rate_5y)
        stage1_pv += projected / ((1.0 + wacc) ** year)

    # Stage 2: Gordon growth terminal value.
    fcf_year5 = fcf * ((1.0 + growth_rate_5y) ** 5)
    terminal_value = fcf_year5 * (1.0 + terminal_growth) / (wacc - terminal_growth)
    terminal_pv = terminal_value / ((1.0 + wacc) ** 5)

    total_equity_value = stage1_pv + terminal_pv
    intrinsic_value_per_share = total_equity_value / shares_outstanding

    return {
        "intrinsic_value_per_share": round(intrinsic_value_per_share, 2),
        "stage1_pv": round(stage1_pv, 2),
        "terminal_pv": round(terminal_pv, 2),
        "total_equity_value": round(total_equity_value, 2),
        "assumptions": {
            "growth_rate_5y": growth_rate_5y,
            "terminal_growth": terminal_growth,
            "wacc": wacc,
        },
    }


# Long-run sustainable growth anchor and the stage-1 ceiling. yfinance's
# ``earningsGrowth`` / ``revenueGrowth`` are single-quarter YoY figures — far too
# volatile to feed straight into a 5-year forward DCF (hypergrowth names can post
# 80%+ in a single quarter). We blend the two near-term signals with a GDP-like
# anchor and cap the result so one explosive quarter cannot inflate the valuation.
_GROWTH_LONG_RUN: float = 0.04
_GROWTH_CEILING: float = 0.25


def _estimate_growth_rate_5y(info: dict) -> float:
    """Estimate a defensible 5-year forward growth rate for the stage-1 DCF.

    Blends yfinance's quarterly ``revenueGrowth`` and ``earningsGrowth`` (near-term
    signals) with a long-run sustainable anchor, then caps and floors the result.

    Args:
        info: ``stock.info`` dictionary.

    Returns:
        A growth rate clamped to ``[0.0, _GROWTH_CEILING]``.
    """
    info = info or {}
    signals: list[float] = []
    for key in ("revenueGrowth", "earningsGrowth"):
        raw = info.get(key)
        if raw is None:
            continue
        try:
            signals.append(float(raw))
        except (TypeError, ValueError):
            continue
    near_term = float(np.mean(signals)) if signals else _GROWTH_LONG_RUN
    blended = 0.7 * near_term + 0.3 * _GROWTH_LONG_RUN
    return float(max(0.0, min(_GROWTH_CEILING, blended)))


def compare_to_peers(
    peer_tickers: list[str], company_valuation: dict, company_profitability: dict
) -> dict:
    """Compare the company's valuation/profitability against its peers.

    Fetches ``stock.info`` for each peer (with a 0.5s pause between calls to
    avoid yfinance rate limiting) and aggregates sector averages.

    Args:
        peer_tickers: Competitor tickers.
        company_valuation: Output of :func:`compute_valuation_ratios`.
        company_profitability: Output of :func:`compute_profitability_ratios`.

    Returns:
        Dict with ``peer_data`` (list per peer), and sector-average ``pe``,
        ``pb``, and ``margin``.
    """
    peer_data: list[dict] = []
    pe_values: list[float] = []
    pb_values: list[float] = []
    margin_values: list[float] = []

    for ticker in peer_tickers:
        try:
            time.sleep(0.5)  # avoid yfinance rate limiting
            info = yf.Ticker(ticker).info or {}
            pe = info.get("trailingPE")
            pb = info.get("priceToBook")
            margin_raw = info.get("profitMargins")
            margin = float(margin_raw) * 100.0 if margin_raw is not None else None

            if pe is not None:
                pe_values.append(float(pe))
            if pb is not None:
                pb_values.append(float(pb))
            if margin is not None:
                margin_values.append(margin)

            peer_data.append({
                "ticker": ticker,
                "pe_ratio": pe,
                "pb_ratio": pb,
                "net_margin": margin,
                "roe": info.get("returnOnEquity"),
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Peer fetch failed for %s: %s", ticker, exc)
            peer_data.append({"ticker": ticker, "pe_ratio": None, "pb_ratio": None, "net_margin": None, "roe": None})

    def _mean(values: list[float]) -> float | None:
        return round(float(np.mean(values)), 2) if values else None

    return {
        "peer_data": peer_data,
        "sector_avg_pe": _mean(pe_values),
        "sector_avg_pb": _mean(pb_values),
        "sector_avg_margin": _mean(margin_values),
        "company_pe": company_valuation.get("pe_ratio"),
        "company_net_margin": company_profitability.get("net_margin"),
    }


def _interpret(
    company: str,
    valuation: dict,
    profitability: dict,
    liquidity: dict,
    dcf: dict,
    peer_comparison: dict,
    macro: dict,
) -> str:
    """Ask GPT-4o-mini for a 3-paragraph interpretation, with a safe fallback."""
    prompt = FUNDAMENTAL_PROMPT.format(
        company=company,
        valuation=valuation,
        profitability=profitability,
        liquidity=liquidity,
        dcf=dcf,
        peer_comparison=peer_comparison,
        fed_rate=macro.get("fed_funds_rate", "N/A"),
        yield_10y=macro.get("us_10y_yield", "N/A"),
    )
    if not llm_available():
        logger.warning("OPENAI_API_KEY missing — skipping fundamental interpretation.")
        return (
            f"Fundamental interpretation unavailable (no LLM key configured). "
            f"P/E {valuation.get('pe_ratio')}, net margin {profitability.get('net_margin')}%, "
            f"ROE {profitability.get('roe')}%."
        )
    try:
        llm = get_llm(temperature=0.2)
        return str(llm.invoke(prompt).content).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fundamental interpretation LLM call failed: %s", exc)
        return f"Fundamental interpretation unavailable (LLM error). Raw ratios: {valuation}."


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
def run_fundamental_analyst(state: AnalysisState) -> AnalysisState:
    """Run the full fundamental analysis and populate the relevant state keys.

    Args:
        state: Pipeline state produced by the data collector.

    Returns:
        Updated state with ``valuation_ratios``, ``profitability_ratios``,
        ``liquidity_ratios``, ``dcf_estimate``, ``peer_comparison`` and
        ``fundamental_interpretation``.
    """
    errors: list[str] = list(state.get("errors", []))
    completed_steps: list[str] = list(state.get("completed_steps", []))

    statements: dict = state.get("financial_statements", {}) or {}
    info: dict = statements.get("info", {}) or {}
    income_df = _statements_to_df(statements.get("income_stmt"))
    balance_df = _statements_to_df(statements.get("balance_sheet"))

    try:
        valuation = compute_valuation_ratios(info)
    except Exception as exc:  # noqa: BLE001
        valuation = {}
        errors.append(f"fundamental_analyst: valuation failed — {exc}")

    try:
        profitability = compute_profitability_ratios(info, income_df, balance_df)
    except Exception as exc:  # noqa: BLE001
        profitability = {}
        errors.append(f"fundamental_analyst: profitability failed — {exc}")

    try:
        liquidity = compute_liquidity_ratios(info, balance_df)
    except Exception as exc:  # noqa: BLE001
        liquidity = {}
        errors.append(f"fundamental_analyst: liquidity failed — {exc}")

    # DCF inputs from yfinance info, with sensible defaults.
    try:
        fcf = info.get("freeCashflow") or 0.0
        growth_rate_5y = _estimate_growth_rate_5y(info)
        shares = info.get("sharesOutstanding") or 0
        dcf_estimate = compute_dcf(
            fcf=float(fcf),
            growth_rate_5y=growth_rate_5y,
            terminal_growth=0.025,
            wacc=0.09,
            shares_outstanding=int(shares),
        )
    except Exception as exc:  # noqa: BLE001
        dcf_estimate = {"intrinsic_value_per_share": None, "note": f"DCF failed — {exc}"}
        errors.append(f"fundamental_analyst: DCF failed — {exc}")

    try:
        peer_comparison = compare_to_peers(state.get("peer_tickers", []), valuation, profitability)
    except Exception as exc:  # noqa: BLE001
        peer_comparison = {"peer_data": [], "sector_avg_pe": None}
        errors.append(f"fundamental_analyst: peer comparison failed — {exc}")

    interpretation = _interpret(
        company=state.get("company_name") or state.get("ticker", ""),
        valuation=valuation,
        profitability=profitability,
        liquidity=liquidity,
        dcf=dcf_estimate,
        peer_comparison=peer_comparison,
        macro=state.get("macro_data", {}) or {},
    )

    completed_steps.append("fundamental_analyst")

    return {
        **state,
        "valuation_ratios": valuation,
        "profitability_ratios": profitability,
        "liquidity_ratios": liquidity,
        "dcf_estimate": dcf_estimate,
        "peer_comparison": peer_comparison,
        "fundamental_interpretation": interpretation,
        "errors": errors,
        "completed_steps": completed_steps,
    }


if __name__ == "__main__":
    from src.agents.data_collector import run_data_collector

    _state = AnalysisState(
        ticker="AAPL", company_name="", analysis_depth="quick",
        price_history={}, financial_statements={}, macro_data={}, peer_tickers=[],
        valuation_ratios={}, profitability_ratios={}, liquidity_ratios={}, dcf_estimate={},
        peer_comparison={}, fundamental_interpretation="", news_articles=[],
        aggregate_sentiment="neutral", sentiment_positive_avg=0.0, sentiment_negative_avg=0.0,
        key_topics=[], red_flags=[], risk_score=0, risk_summary="",
        executive_summary="", full_report_markdown="", pdf_path="",
        messages=[], errors=[], completed_steps=[],
    )
    _state = run_data_collector(_state)
    _out = run_fundamental_analyst(_state)
    print("Valuation:", _out["valuation_ratios"])
    print("Profitability:", _out["profitability_ratios"])
    print("DCF:", _out["dcf_estimate"])
    print("Peers:", _out["peer_comparison"].get("sector_avg_pe"))
    print("Interpretation:", _out["fundamental_interpretation"][:200], "...")
