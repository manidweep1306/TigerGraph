"""
graph.py — Backward-compatibility re-export module.
The core LangGraph Agentic Pipeline orchestrator has been moved to:
backend.pipelines.agentic_orchestrator
"""

from backend.pipelines.agentic_orchestrator import (
    AgentState,
    build_investigation_graph,
    run_agentic_pipeline,
)

__all__ = [
    "AgentState",
    "build_investigation_graph",
    "run_agentic_pipeline",
]
