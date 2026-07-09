"""FastAPI application exposing the analysis pipeline as a REST API.

Runs are tracked in an in-memory store keyed by ``run_id``. Analysis executes in
a background thread (the LangGraph pipeline is synchronous). No database.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

# Load environment variables (API keys) from .env before src modules read them.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:  # pragma: no cover - dotenv optional
    pass

from src.agents.orchestrator import build_graph
from src.api.schemas import (
    AnalysisRequest,
    AnalysisResultResponse,
    AnalysisStatusResponse,
)
from src.state.factory import make_empty_state
from src.tools.llm import llm_available

logger = logging.getLogger(__name__)

# Total pipeline steps used to compute progress.
_TOTAL_STEPS: int = 5

# In-memory run registry: run_id -> {"status", "state", "thread_id"}.
_runs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=4)
_graph = build_graph()

app = FastAPI(title="AutoAnalyst Finance API", version="0.1.0")


def _run_pipeline(run_id: str, ticker: str, depth: str) -> None:
    """Execute the full graph for a run, updating the in-memory store.

    Args:
        run_id: Unique run identifier.
        ticker: Stock symbol.
        depth: ``"quick"`` or ``"deep"``.
    """
    entry = _runs.get(run_id)
    if entry is None:
        return
    try:
        # Stream node-by-node so /status can report incremental progress. With
        # stream_mode="values" LangGraph yields the full accumulated state (with
        # reducers applied) after each node; assigning it to the run entry exposes
        # the latest completed_steps to status polling instead of jumping 0 -> 100.
        for chunk in _graph.stream(
            make_empty_state(ticker, analysis_depth=depth),
            config={"configurable": {"thread_id": run_id}},
            stream_mode="values",
        ):
            if isinstance(chunk, dict):
                entry["state"] = chunk
        entry["status"] = "complete"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis %s failed", run_id)
        entry["status"] = "error"
        entry["error"] = str(exc)


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "models_loaded": llm_available()}


@app.post("/analyze")
def analyze(request: AnalysisRequest) -> dict:
    """Start an analysis in the background.

    Args:
        request: :class:`AnalysisRequest` body.

    Returns:
        ``{"run_id": str, "status": "started"}``.
    """
    run_id = uuid.uuid4().hex
    _runs[run_id] = {"status": "running", "state": make_empty_state(request.ticker, request.analysis_depth)}
    _executor.submit(_run_pipeline, run_id, request.ticker, request.analysis_depth)
    return {"run_id": run_id, "status": "started"}


def _progress_pct(completed_steps: list[str]) -> int:
    """Compute completion percentage from the number of completed steps."""
    return min(100, int(round(len(completed_steps) / _TOTAL_STEPS * 100)))


@app.get("/status/{run_id}", response_model=AnalysisStatusResponse)
def status(run_id: str) -> AnalysisStatusResponse:
    """Return the current status of a run."""
    entry = _runs.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    state = entry.get("state") or {}
    return AnalysisStatusResponse(
        run_id=run_id,
        status=entry.get("status", "running"),
        completed_steps=list(state.get("completed_steps", []) or []),
        progress_pct=_progress_pct(state.get("completed_steps", []) or []),
    )


@app.get("/report/{run_id}", response_model=AnalysisResultResponse)
def report(run_id: str) -> AnalysisResultResponse:
    """Return the structured result of a completed run."""
    entry = _runs.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    if entry.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Report not yet complete")
    state = entry.get("state") or {}
    pdf_path = state.get("pdf_path", "")
    return AnalysisResultResponse(
        run_id=run_id,
        ticker=state.get("ticker", ""),
        company_name=state.get("company_name", ""),
        risk_score=int(state.get("risk_score", 0) or 0),
        aggregate_sentiment=state.get("aggregate_sentiment", "neutral"),
        executive_summary=state.get("executive_summary", ""),
        red_flags=list(state.get("red_flags", []) or []),
        pdf_available=bool(pdf_path and Path(pdf_path).exists()),
    )


@app.get("/report/{run_id}/pdf")
def report_pdf(run_id: str) -> FileResponse:
    """Stream the generated PDF for a run."""
    entry = _runs.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    if entry.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Report not yet complete")
    pdf_path = entry.get("state", {}).get("pdf_path", "")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail="PDF not available")
    return FileResponse(path=pdf_path, media_type="application/pdf", filename=Path(pdf_path).name)
