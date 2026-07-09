"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Inbound analysis request."""

    ticker: str = Field(..., description="Stock symbol, e.g. AAPL or BNP.PA")
    analysis_depth: str = Field("quick", description='"quick" or "deep"')


class AnalysisStatusResponse(BaseModel):
    """Progress/status response for a running or finished analysis."""

    run_id: str
    status: str  # "running" | "complete" | "error"
    completed_steps: list[str]
    progress_pct: int  # 0-100 based on completed_steps count


class AnalysisResultResponse(BaseModel):
    """Final structured result for a completed analysis."""

    run_id: str
    ticker: str
    company_name: str
    risk_score: int
    aggregate_sentiment: str
    executive_summary: str
    red_flags: list[dict]
    pdf_available: bool
