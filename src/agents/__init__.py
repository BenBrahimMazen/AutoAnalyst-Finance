"""LangGraph agent nodes for the AutoAnalyst Finance pipeline.

Each module in this package exposes a ``run_<agent>(state)`` function that
takes the shared :class:`~src.state.schema.AnalysisState`, performs its work,
and returns the updated state.
"""

__all__ = [
    "orchestrator",
    "data_collector",
    "fundamental",
    "sentiment",
    "risk_detector",
    "report_writer",
]
