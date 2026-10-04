"""
Pipeline C: Agentic GraphRAG (wrapper around graph.py LangGraph orchestrator)
Per spec §2.3 — full agentic investigation pipeline.
"""

import logging
from backend.config.unified_config import config

logger = logging.getLogger(__name__)


def run(question: str, question_id: str) -> dict:
    """
    Agentic Pipeline C.
    Input: {question: str, question_id: str}
    Output: {answer, exit_type, sources, tokens_used, latency_ms,
             ledger, trace, stop_reason, strategy_changed,
             n_chunks_retrieved, n_sources_cited}
    """
    try:
        from backend.pipelines.agentic_orchestrator import run_agentic_pipeline
        thresholds = config.frozen_thresholds_data
        agent_config = config.agent_config_data
        result = run_agentic_pipeline(question, question_id, thresholds, agent_config)
        return result
    except Exception as e:
        logger.error(f"Agentic pipeline error: {e}")
        return {
            "answer": "NO_EVIDENCE_RETRIEVED",
            "exit_type": "ABSTAIN",
            "sources": [],
            "tokens_used": 0,
            "latency_ms": 0,
            "ledger": None,
            "trace": [],
            "stop_reason": "error",
            "strategy_changed": False,
            "n_chunks_retrieved": 0,
            "n_sources_cited": 0,
            "step_count": 0,
        }
