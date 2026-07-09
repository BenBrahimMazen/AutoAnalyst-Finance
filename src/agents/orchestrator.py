"""LangGraph orchestrator.

Wires the six agents into a compiled graph:

    data_collector
        -> fundamental_analyst
        -> sentiment_analyst
        -> risk_detector
        -> [risk_score > 75 ? add_critical_warning :]
        -> report_writer
        -> END

The fundamental and sentiment analysts run sequentially (true fan-out parallelism
would require the ``Send`` API). ``MemorySaver`` is used as the checkpointer so
completed runs can be inspected via thread_id.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.data_collector import run_data_collector
from src.agents.fundamental import run_fundamental_analyst
from src.agents.report_writer import run_report_writer
from src.agents.risk_detector import run_risk_detector
from src.agents.sentiment import run_sentiment_analyst
from src.state.schema import AnalysisState

logger = logging.getLogger(__name__)

# Threshold above which a run is flagged as critically risky.
HIGH_RISK_THRESHOLD: int = 75


def route_after_risk(state: AnalysisState) -> str:
    """Conditional routing function used after the risk detector.

    Args:
        state: Current pipeline state.

    Returns:
        ``"high_risk"`` if ``risk_score`` exceeds the threshold, else ``"normal"``.
    """
    return "high_risk" if state.get("risk_score", 0) > HIGH_RISK_THRESHOLD else "normal"


def add_critical_warning(state: AnalysisState) -> AnalysisState:
    """Intermediate node reached only when ``risk_score`` exceeds the threshold.

    Flags the run as critically risky by appending a warning to ``messages``.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with a critical-risk warning message appended.
    """
    warning = (
        f"CRITICAL RISK ALERT: {state.get('company_name', state.get('ticker', ''))} "
        f"scored {state.get('risk_score', 0)}/100 — above the {HIGH_RISK_THRESHOLD} "
        f"threshold. Escalate to manual review."
    )
    logger.warning(warning)
    return {**state, "messages": [warning]}


def build_graph():
    """Build and compile the LangGraph execution graph.

    Returns:
        A compiled :class:`StateGraph` ready for ``.invoke(state, config=...)``.
    """
    graph: StateGraph = StateGraph(AnalysisState)

    graph.add_node("data_collector", run_data_collector)
    graph.add_node("fundamental_analyst", run_fundamental_analyst)
    graph.add_node("sentiment_analyst", run_sentiment_analyst)
    graph.add_node("risk_detector", run_risk_detector)
    graph.add_node("add_critical_warning", add_critical_warning)
    graph.add_node("report_writer", run_report_writer)

    graph.set_entry_point("data_collector")
    graph.add_edge("data_collector", "fundamental_analyst")
    graph.add_edge("fundamental_analyst", "sentiment_analyst")
    graph.add_edge("sentiment_analyst", "risk_detector")

    graph.add_conditional_edges(
        "risk_detector",
        route_after_risk,
        {"high_risk": "add_critical_warning", "normal": "report_writer"},
    )
    graph.add_edge("add_critical_warning", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile(checkpointer=MemorySaver())
