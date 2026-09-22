"""
Pipeline C: Agentic GraphRAG (wrapper around graph.py LangGraph orchestrator)
Per spec §2.3 — full agentic investigation pipeline.
"""

import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _load_config() -> tuple[dict, dict]:
    config_dir = Path("./config")
    with open(config_dir / "frozen_thresholds.json") as f:
        thresholds = json.load(f)
    with open(config_dir / "agent_config.json") as f:
        agent_config = json.load(f)
    return thresholds, agent_config


def run(question: str, question_id: str) -> dict:
    """
    Agentic Pipeline C.
    Input: {question: str, question_id: str}
    Output: {answer, exit_type, sources, tokens_used, latency_ms,
             ledger, trace, stop_reason, strategy_changed,
             n_chunks_retrieved, n_sources_cited}
    """
    try:
        from backend.graph import run_agentic_pipeline
        thresholds, agent_config = _load_config()
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
        }
