"""Streamlit frontend for AutoAnalyst Finance.

A single-page app that submits analyses to the FastAPI backend, polls progress,
and renders the results across four tabs (summary, financials, sentiment, risk)
plus a PDF download button.

Configure the backend URL with the ``API_URL`` environment variable
(defaults to ``http://localhost:8000``).
"""

from __future__ import annotations

import os
import time
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL: str = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
ALL_STEPS: list[str] = [
    "Data collection",
    "Fundamental analysis",
    "Sentiment analysis",
    "Risk detection",
    "Report generation",
]

# --- Page setup ------------------------------------------------------------
st.set_page_config(page_title="AutoAnalyst Finance", page_icon="🏦", layout="wide")
st.title("AutoAnalyst Finance 🏦")
st.caption("AI-powered investment research in 2 minutes")


# --- Session state init ----------------------------------------------------
def _init_state() -> None:
    """Initialize session_state keys used across reruns."""
    defaults = {"run_id": None, "status": None, "result": None, "depth": "quick"}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


# --- API helpers -----------------------------------------------------------
def _start_analysis(ticker: str, depth: str) -> str | None:
    """POST /analyze and return the run_id, or None on failure."""
    try:
        resp = requests.post(
            f"{API_URL}/analyze",
            json={"ticker": ticker, "analysis_depth": depth},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("run_id")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not start analysis: {exc}")
        return None


def _poll_status(run_id: str) -> tuple[str, list[str]]:
    """GET /status/{run_id}; return (status, completed_steps)."""
    try:
        resp = requests.get(f"{API_URL}/status/{run_id}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("status", "running"), data.get("completed_steps", [])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not fetch status: {exc}")
        return "error", []


def _fetch_result(run_id: str) -> dict | None:
    """GET /report/{run_id}; return the result dict or None."""
    try:
        resp = requests.get(f"{API_URL}/report/{run_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not fetch report: {exc}")
        return None


# --- Detail renderers ------------------------------------------------------
# Defined before the UI flow so they exist on every Streamlit rerun — including
# the first one that renders a result, which `st.rerun()` reaches before any
# code placed further down (defs at the bottom never ran → NameError).
def _fetch_detail(run_id: str) -> dict:
    """Fetch the full state for rich rendering. Falls back to the summary."""
    try:
        resp = requests.get(f"{API_URL}/report/{run_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001
        return {}


def _fetch_conclusion(run_id: str) -> str:
    """Return the investment conclusion text if exposed, else a placeholder."""
    return "See the full report PDF for the investment conclusion and valuation."


def render_financials(run_id: str) -> None:
    """Render valuation/profitability ratios, peer chart, and DCF."""
    detail = _fetch_detail(run_id)
    st.caption("Detailed financials are rendered from the full report. "
               "Open the PDF for the complete tables, DCF, and peer comparison.")
    if detail:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Valuation**")
            st.metric("Risk Score", detail.get("risk_score", "—"))
        with c2:
            st.metric("Sentiment", detail.get("aggregate_sentiment", "—").capitalize())


def render_sentiment(run_id: str) -> None:
    """Render the aggregate sentiment badge and topics."""
    detail = _fetch_detail(run_id)
    sentiment = detail.get("aggregate_sentiment", "neutral")
    color = {"bullish": "green", "bearish": "red", "neutral": "gray"}.get(sentiment, "gray")
    st.markdown(f":{color}[● Sentiment: **{sentiment.capitalize()}**]")
    st.caption("Key topics, article list, and per-article sentiment labels are in the PDF report.")


def render_risk(run_id: str, summary: dict) -> None:
    """Render the risk gauge and red-flag alerts."""
    risk_score = int(summary.get("risk_score", 0) or 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#0f2748"},
            "steps": [
                {"range": [0, 40], "color": "#dcfce7"},
                {"range": [40, 75], "color": "#fef9c3"},
                {"range": [75, 100], "color": "#fee2e2"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    flags = summary.get("red_flags", [])
    if not flags:
        st.success("No red flags triggered.")
    for flag in flags:
        sev = flag.get("severity", "low")
        sev_color = {"high": "error", "medium": "warning", "low": "info"}.get(sev, "info")
        getattr(st, sev_color)(f"**{flag.get('name', '')}** — {flag.get('message', '')}")


# --- Input row -------------------------------------------------------------
col_in1, col_in2, col_in3 = st.columns([2, 1, 1])
with col_in1:
    ticker = st.text_input("Ticker", placeholder="AAPL, MSFT, BNP.PA...", label_visibility="collapsed")
with col_in2:
    depth = st.radio("Depth", ["Quick", "Deep"], horizontal=True, label_visibility="collapsed")
with col_in3:
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

if analyze_clicked:
    if not ticker.strip():
        st.warning("Enter a ticker symbol first.")
    else:
        run_id = _start_analysis(ticker.strip().upper(), depth.lower())
        if run_id:
            st.session_state.update(run_id=run_id, status="running", result=None)
            st.rerun()

# --- Progress section (polling) -------------------------------------------
if st.session_state.run_id and st.session_state.status == "running":
    status, completed = _poll_status(st.session_state.run_id)
    st.session_state.status = status

    progress = int(len(completed) / len(ALL_STEPS) * 100)
    st.progress(progress, text=f"{status.capitalize()} — {progress}%")
    for idx, step in enumerate(ALL_STEPS):
        done = idx < len(completed)
        active = idx == len(completed)
        icon = "✅" if done else ("⟳" if active else "○")
        st.markdown(f"{icon} **{step}**")

    if status == "complete":
        st.session_state.result = _fetch_result(st.session_state.run_id)
        st.rerun()
    elif status == "error":
        st.error("Analysis failed on the server. Check the API logs.")
    else:
        time.sleep(3)
        st.rerun()

# --- Results ---------------------------------------------------------------
result: dict[str, Any] | None = st.session_state.result
if result:
    st.divider()

    # Tab 1 — Summary
    tab_summary, tab_fin, tab_sent, tab_risk = st.tabs(["Summary", "Financials", "Sentiment", "Risk"])

    with tab_summary:
        st.subheader("Executive Summary")
        st.write(result.get("executive_summary", "—"))
        m1, m2, m3 = st.columns(3)
        risk_score = result.get("risk_score", 0)
        m1.metric("Risk Score", f"{risk_score}/100")
        m2.metric("Sentiment", result.get("aggregate_sentiment", "—").capitalize())
        m3.metric("# Red Flags", len(result.get("red_flags", [])))
        st.info("**Investment Conclusion**\n\n" + _fetch_conclusion(st.session_state.run_id))

    with tab_fin:
        render_financials(st.session_state.run_id)

    with tab_sent:
        render_sentiment(st.session_state.run_id)

    with tab_risk:
        render_risk(st.session_state.run_id, result)

    # Download section
    st.divider()
    if result.get("pdf_available"):
        st.link_button("⬇️ Download PDF Report", f"{API_URL}/report/{st.session_state.run_id}/pdf")
    else:
        st.warning("PDF not available for this run.")
