"""
AggregationAgent — combines ≥2 RESOLVED/SUPPORTED claims into a derived claim.
Per spec §3.1: invocable when requires_aggregation==true AND ≥2 matching-shape slots resolved.

Optimized with:
- Direct GSQL get_events_by_criteria query routing for count/aggregation questions
  (e.g., event counts with competitor thresholds) before falling back to LLM chunk scanning.
- Fast Intermediate Agent using Groq llama-3.1-8b-instant for textual synthesis.
"""

import json
import logging
import re
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
    AggregationAgent execution using GSQL parameterized query first,
    falling back to fast model (llama-3.1-8b-instant).

    Input per spec §3.2:
      {"source_claims": [claim_id, ...], "aggregation_type": "count|compare|rank", ...}

    Output per spec §3.2:
      {"derived_claim_text": str, "derived_from": [claim_id], "tokens_used": int}
    """
    t_start = time.time()
    source_claim_ids = input_data.get("source_claims", [])
    aggregation_type = input_data.get("aggregation_type", "count")
    source_claims_text = input_data.get("source_claims_text", [])
    question = input_data.get("question", "")
    slot_desc = input_data.get("slot_description", "")

    # Priority 1: Check if question or slot is a count/aggregation query with criteria
    search_context = f"{question} {slot_desc} " + " ".join(source_claims_text)
    if "how many" in search_context.lower() or "count" in aggregation_type.lower() or "more than" in search_context.lower():
        try:
            from backend.db.tigergraph_client import get_tg_client
            tg = get_tg_client()

            # Extract criteria: games, sport, min_competitors
            games_match = re.search(r'(\d{4}\s+(?:Summer|Winter))', search_context, re.I)
            sport_match = re.search(
                r'\b(biathlon|shooting|cycling|athletics|swimming|sailing|weightlifting|rowing|judo|canoeing|boxing|archery|fencing|gymnastics|tennis|badminton|equestrian|skiing|bobsleigh|curling|figure skating|ice hockey|luge|short track|skeleton|snowboard|speed skating)\b',
                search_context, re.I
            )
            comp_match = re.search(r'(?:more than|exceeding|>)\s*(\d+)\s*competitors?', search_context, re.I)
            if not comp_match:
                comp_match = re.search(r'competitors?\s*>?\s*(\d+)', search_context, re.I)

            if games_match and comp_match:
                games_str = games_match.group(1)
                sport_str = sport_match.group(1).capitalize() if sport_match else ""
                min_comp = int(comp_match.group(1))

                events = tg.get_events_by_criteria(games=games_str, sport=sport_str, min_competitors=min_comp)
                if events:
                    count_val = len(events)
                    event_titles = [e.get("title", "") for e in events[:5]]
                    derived_claim = (
                        f"There were {count_val} {sport_str} events at the {games_str} Olympics "
                        f"that had more than {min_comp} competitors."
                    )
                    doc_ids = [e.get("doc_id", "") for e in events if e.get("doc_id")]
                    output = {
                        "derived_claim_text": derived_claim,
                        "derived_from": doc_ids or source_claim_ids,
                        "tokens_used": 0,
                        "reasoning": f"GSQL get_events_by_criteria verified {count_val} events matching: {', '.join(event_titles)}",
                        "aggregated_value": str(count_val),
                    }
                    _log_invocation(
                        "AggregationAgent", question_id, step_id,
                        f"GSQL criteria: games={games_str}, sport={sport_str}, min_comp={min_comp}",
                        0, t_start, f"derived: {output['derived_claim_text']}"
                    )
                    return output
        except Exception as e:
            logger.warning(f"AggregationAgent GSQL criteria fallback: {e}")

    # Priority 2: Fall back to LLM unstructured chunk aggregation
    if len(source_claims_text) < 1:
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
        output.get("tokens_used", 0), t_start,
        f"derived: {output.get('derived_claim_text', '')[:100]}",
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
