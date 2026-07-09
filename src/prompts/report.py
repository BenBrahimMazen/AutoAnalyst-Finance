"""Prompt templates for the report writer agent."""

from __future__ import annotations

EXECUTIVE_SUMMARY_PROMPT: str = """
Write a 3-sentence executive summary for {company} ({ticker}).
Sentence 1: Current financial health — mention one specific metric.
Sentence 2: The single most important risk.
Sentence 3: Overall assessment in plain English.
Tone: professional investment research. Be specific and direct.
Data available: {data_summary}
"""

CONCLUSION_PROMPT: str = """
Write an investment conclusion for {company} ({ticker}).
Start with one of: "We rate {ticker} Overweight", "We rate {ticker} Neutral",
or "We rate {ticker} Underweight".
Follow with: primary reasoning (1 sentence), key risk to the thesis (1 sentence),
valuation comment referencing the DCF or P/E vs peers (1 sentence).
Max 4 sentences total. Be direct and specific.
Risk score: {risk_score}/100. Sentiment: {sentiment}.
DCF intrinsic value: {dcf_value}. Current price: {current_price}.
"""
