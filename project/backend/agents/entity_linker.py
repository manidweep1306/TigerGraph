"""
EntityLinkerAgent — resolves query entity strings to graph node IDs.
Per spec §3.1: invocable when any EMPTY or SUPPORTED slot references an unlinked entity.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

ENTITY_LINK_PROMPT = """You are an entity extraction and normalization expert for Olympic sports data.

Given a question or text, extract the key named entities and normalize them:
- Athlete names (normalize to common English spellings)
- Olympic Games (e.g., "2018 Winter Olympics", "2008 Summer Olympics")
- Sports/Events (e.g., "men's 100m sprint", "women's figure skating")
- Countries/NOC codes (e.g., "United States" → "USA", "Great Britain" → "GBR")
- Venues

OUTPUT FORMAT (JSON only):
{
  "entities": [
    {
      "original_text": "...",
      "normalized_name": "...",
      "entity_type": "ATHLETE|EVENT|COUNTRY|VENUE|GAMES|SPORT",
      "search_variants": ["variant1", "variant2"]
    }
  ]
}"""


def run(input_data: dict, question_id: str, step_id: int) -> dict:
    """
    EntityLinkerAgent execution.

    Input per spec §3.2:
      {"query_string": str, "graph": TigerGraphInstance ref}

    Output per spec §3.2:
      {"entity_id": str|null, "match_confidence": float, "candidates": [str]}
    """
    t_start = time.time()
    query_string = input_data.get("query_string", "")
    tokens_used = 0

    try:
        # Step 1: Extract entities from query using LLM
        from backend.core.gemini_utils import generate_content_with_retry
        model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config=genai.GenerationConfig(temperature=0.0),
            system_instruction=ENTITY_LINK_PROMPT,
        )
        response = generate_content_with_retry(
            model,
            f"Extract entities from: {query_string}"
        )
        raw = response.text.strip()
        tokens_used += _estimate_tokens(query_string + raw)

        # Parse
        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            extracted = json.loads(raw.strip())
            entities = extracted.get("entities", [])
        except Exception:
            entities = []

        if not entities:
            output = {
                "entity_id": None,
                "match_confidence": 0.0,
                "candidates": [],
                "extracted_entities": [],
            }
            _log_invocation("EntityLinkerAgent", question_id, step_id,
                            query_string, tokens_used, t_start, "no_entities_extracted")
            return output

        # Step 2: Lookup each entity in TigerGraph
        from backend.db.tigergraph_client import get_tg_client
        tg = get_tg_client()

        best_entity_id = None
        best_confidence = 0.0
        all_candidates = []

        for entity_info in entities:
            name = entity_info.get("normalized_name", "")
            variants = entity_info.get("search_variants", [name])

            for variant in variants:
                matches = tg.entity_lookup_by_name(variant, top_k=3)
                for match in matches:
                    attrs = match.get("attributes", {})
                    entity_id = match.get("v_id", "")
                    all_candidates.append(entity_id)

                    # Confidence: exact match = 1.0, partial = 0.7
                    stored_name = attrs.get("name", "").lower()
                    if stored_name == variant.lower():
                        conf = 1.0
                    elif variant.lower() in stored_name or stored_name in variant.lower():
                        conf = 0.75
                    else:
                        conf = 0.5

                    if conf > best_confidence:
                        best_confidence = conf
                        best_entity_id = entity_id

        output = {
            "entity_id": best_entity_id,
            "match_confidence": best_confidence,
            "candidates": list(set(all_candidates))[:5],
            "extracted_entities": entities,
        }

    except Exception as e:
        logger.error(f"EntityLinkerAgent error: {e}")
        output = {"entity_id": None, "match_confidence": 0.0, "candidates": []}

    latency_ms = int((time.time() - t_start) * 1000)
    output["tokens_used"] = tokens_used
    _log_invocation("EntityLinkerAgent", question_id, step_id,
                    query_string[:200], tokens_used, t_start,
                    f"entity_id={output.get('entity_id')}, conf={output.get('match_confidence', 0):.2f}")
    return output


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return max(1, len(text) // 4)


def _log_invocation(agent_name: str, question_id: str, step_id: int,
                     input_summary: str, tokens_used: int, t_start: float,
                     output_summary: str) -> None:
    """Per spec §3.3 — mandatory invocation log."""
    import json
    record = {
        "agent_name": agent_name,
        "question_id": question_id,
        "step_id": step_id,
        "input_summary": input_summary[:300],
        "tokens_used": tokens_used,
        "latency_ms": int((time.time() - t_start) * 1000),
        "output_summary": output_summary[:300],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_path = LOG_DIR / "agent_invocations.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to log agent invocation: {e}")
