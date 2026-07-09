"""Prompt templates for the fundamental analyst agent."""

from __future__ import annotations

FUNDAMENTAL_PROMPT: str = """
You are a senior equity analyst. Based on the following financial data for {company},
write a 3-paragraph fundamental analysis:

Paragraph 1: Valuation — is the stock cheap, fair, or expensive vs peers and history?
Paragraph 2: Profitability and growth — is the business healthy and growing?
Paragraph 3: Financial health — can the company sustain itself? Any concerns?

Be specific: use the actual numbers provided. Be direct about your assessment.
Max 200 words total.

Valuation ratios: {valuation}
Profitability ratios: {profitability}
Liquidity ratios: {liquidity}
DCF estimate: {dcf}
Peer comparison: {peer_comparison}
Macro context: Fed funds rate {fed_rate}%, US 10Y yield {yield_10y}%
"""
