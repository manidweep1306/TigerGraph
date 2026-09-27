"""
AggregationAgent — combines ≥2 RESOLVED/SUPPORTED claims into a derived claim.
Per spec §3.1: invocable when requires_aggregation==true AND ≥2 matching-shape slots resolved.

Per Model Mapping:
Fast Intermediate Agent using Groq llama-3.1-8b-instant.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.config.unified_config import config
from backend.core.llm_client import get_groq_client

logger = logging.getLogger(__name__)
LOG_DIR = Path(config.paths.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)

AGGREGATION_PROMPT = """You are an aggregation reasoning engine. 
Given a list of evidence claims and an aggregation type (count/compare/rank), 
perform the aggregation and state the result clearly.

Output a JSON object:
{
  "derived_claim_text": "The clear, specific result of the aggregation",
  "reasoning": "Brief explanation of how you arrived at this",
  "aggregated_value": "the numeric or ranked result"
}"""


def run(input_data: dict, question_id: str, step_id: int) -> dict:
    """
    AggregationAgent execution using fast model (llama-3.1-8b-instant).

    Input per spec §3.2:
      {"source_claims": [claim_id, ...], "aggregation_type": "count|compare|rank"}

    Output per spec §3.2:
      {"derived_claim_text": str, "derived_from": [claim_id], "tokens_used": int}
    """
    t_start = time.time()
    source_claim_ids = input_data.get("source_claims", [])
    aggregation_type = input_data.get("aggregation_type", "count")
    source_claims_text = input_data.get("source_claims_text", [])

    if len(source_claims_text) < 2:
        _log_invocation("AggregationAgent", question_id, step_id,
                        str(source_claim_ids), 0, t_start, "failed: insufficient claims")
        return {"derived_claim_text": "", "derived_from": source_claim_ids, "tokens_used": 0}

    try:
        claims_text = "\n".join([f"- {c}" for c in source_claims_text])
        user_content = (
            f"Aggregation type: {aggregation_type}\n\n"
            f"Source claims:\n{claims_text}\n\n"
            f"Perform the {aggregation_type} aggregation and return JSON."
        )

        groq = get_groq_client()
        data, response = groq.generate_json(
            prompt=user_content,
            system_instruction=AGGREGATION_PROMPT,
            use_fast_model=True
        )
        tokens_used = response.tokens_used

        output = {
            "derived_claim_text": data.get("derived_claim_text", response.text[:500] if not data else ""),
            "derived_from": source_claim_ids,
            "tokens_used": tokens_used,
            "reasoning": data.get("reasoning", ""),
            "aggregated_value": data.get("aggregated_value", ""),
        }

    except Exception as e:
        logger.error(f"AggregationAgent error: {e}")
        output = {"derived_claim_text": "", "derived_from": source_claim_ids, "tokens_used": 0}

    _log_invocation(
        "AggregationAgent", question_id, step_id,
        f"type={aggregation_type}, claims={len(source_claim_ids)}",
        output["tokens_used"], t_start,
        f"derived: {output['derived_claim_text'][:100]}",
    )
    return output


def _log_invocation(agent_name: str, question_id: str, step_id: int, input_summary: str,
                     tokens_used: int, t_start: float, output_summary: str):
    record = {
        "agent_name": agent_name,
        "question_id": question_id,
        "step_id": step_id,
        "input_summary": str(input_summary)[:300],
        "tokens_used": tokens_used,
        "latency_ms": int((time.time() - t_start) * 1000),
        "output_summary": str(output_summary)[:300],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_path = LOG_DIR / "agent_invocations.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to log invocation: {e}")
