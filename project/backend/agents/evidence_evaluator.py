"""
EvidenceEvaluatorAgent — mandatory post-processor for ALL other agents.
Per spec §3.1: ALWAYS runs immediately after every other agent's execution.
Never independently selected by VoI.

Performs:
1. Entailment check (does the text actually support the claim?)
2. Confidence scoring
3. Contradiction detection against existing claims
4. Triple extraction (subject, predicate, object)
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
LOG_DIR = Path("./logs")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

EVALUATOR_PROMPT = """You are an evidence evaluator for a factual question-answering system about Olympic sports.

Given:
1. A target slot description (what information is needed)
2. Raw agent output (text passages or structured data)
3. The original question

You must determine:
1. Does the evidence PASS entailment? (does it actually provide the needed information?)
2. What is the confidence score (0.0 to 1.0)?
3. Are there any contradictions with existing claims?
4. Extract the specific claim as a (subject, predicate, object) triple.

Be strict: PASS only if the evidence directly and clearly addresses the slot.

OUTPUT FORMAT (JSON only):
{
  "claim_text": "A single, clear, specific factual statement derived from the evidence",
  "entailment_flag": "PASS" or "FAIL",
  "confidence": 0.0-1.0,
  "subject": "the entity or topic",
  "predicate": "the relationship or attribute",
  "object": "the value or answer",
  "contradiction_detected": false,
  "contradicted_claim_id": null,
  "reasoning": "Brief explanation of the entailment decision"
}

CONFIDENCE GUIDELINES:
- 0.95+: Direct, explicit, unambiguous statement matching the slot
- 0.80-0.94: Clear support with minor inference required
- 0.60-0.79: Implicit support, some interpretation needed  
- 0.40-0.59: Weak support, significant inference required
- Below 0.4: Return FAIL"""


def run(input_data: dict, question_id: str, step_id: int,
        target_slot_id: str, existing_claims: Optional[list] = None) -> dict:
    """
    EvidenceEvaluatorAgent execution.

    Input per spec §3.2:
      {"raw_agent_output": object, "target_slot_id": str}

    Output per spec §3.2:
      {
        "claim_text": str,
        "entailment_flag": "PASS|FAIL",
        "confidence": float 0-1,
        "contradiction_detected": bool,
        "contradicted_claim_id": str|null
      }
    Plus: subject, predicate, object, provenance_chain, derived_from, step_id, target_slot_id
    """
    t_start = time.time()
    raw_output = input_data.get("raw_agent_output", {})
    slot_description = input_data.get("slot_description", "")
    original_question = input_data.get("original_question", "")
    existing_claims = existing_claims or []

    # Extract text evidence from raw agent output
    evidence_text = _extract_evidence_text(raw_output)
    provenance = _extract_provenance(raw_output)

    if not evidence_text:
        result = _fail_result(target_slot_id, step_id, "no_evidence_extracted")
        _log_invocation("EvidenceEvaluatorAgent", question_id, step_id,
                        f"slot={target_slot_id}", 0, t_start, "FAIL: no evidence")
        return result

    try:
        # Check for contradictions against existing claims
        existing_summary = ""
        if existing_claims:
            existing_summary = "\n\nExisting claims for this slot:\n" + "\n".join([
                f"- [{c.get('claim_id', '')}] {c.get('claim_text', '')}"
                for c in existing_claims[:5]
            ])

        model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config=genai.GenerationConfig(temperature=0.0),
            system_instruction=EVALUATOR_PROMPT,
        )

        user_content = (
            f"Original question: {original_question}\n\n"
            f"Target slot (what we need): {slot_description}\n\n"
            f"Evidence text:\n{evidence_text[:3000]}\n"
            f"{existing_summary}\n\n"
            f"Evaluate this evidence."
        )

        from backend.core.gemini_utils import generate_content_with_retry
        response = generate_content_with_retry(model, user_content)
        raw = response.text.strip()
        tokens_used = max(1, len(user_content + raw) // 4)

        # Parse output
        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
        except Exception:
            logger.warning(f"EvidenceEvaluator JSON parse failed, using fallback")
            data = _parse_fallback(raw)

        # Build full output per spec §3.2
        output = {
            "claim_text": data.get("claim_text", evidence_text[:200]),
            "entailment_flag": data.get("entailment_flag", "FAIL"),
            "confidence": float(data.get("confidence", 0.0)),
            "contradiction_detected": bool(data.get("contradiction_detected", False)),
            "contradicted_claim_id": data.get("contradicted_claim_id"),
            "subject": data.get("subject", ""),
            "predicate": data.get("predicate", ""),
            "object": data.get("object", ""),
            "provenance_chain": provenance,
            "derived_from": [],  # EvidenceEvaluator produces first-hand claims
            "target_slot_id": target_slot_id,
            "step_id": step_id,
            "tokens_used": tokens_used,
            "reasoning": data.get("reasoning", ""),
            # Round 1 temporal fields — always null
            "valid_from": None,
            "valid_to": None,
            "source_authority": None,
        }

    except Exception as e:
        logger.error(f"EvidenceEvaluatorAgent error: {e}")
        output = _fail_result(target_slot_id, step_id, str(e))
        tokens_used = 0

    _log_invocation(
        "EvidenceEvaluatorAgent", question_id, step_id,
        f"slot={target_slot_id}, evidence_len={len(evidence_text)}",
        output.get("tokens_used", 0), t_start,
        f"flag={output['entailment_flag']}, conf={output['confidence']:.2f}, "
        f"claim={output['claim_text'][:80]}",
    )
    return output


def _extract_evidence_text(raw_output: dict) -> str:
    """Extract usable text from any agent's raw output."""
    if not raw_output:
        return ""

    # SimilaritySearchAgent: chunks list
    if "chunks" in raw_output:
        texts = [c.get("text", "") for c in raw_output.get("chunks", [])[:3]]
        return "\n\n---\n\n".join(texts)

    # DocumentRetrievalAgent: content field
    if "content" in raw_output:
        return raw_output["content"]

    # GraphTraversalAgent: entity context + games
    if "entity_context" in raw_output:
        parts = [raw_output.get("entity_context", "")]
        if raw_output.get("games_found"):
            parts.append("Olympic Games: " + ", ".join(raw_output["games_found"]))
        if raw_output.get("neighbor_entities"):
            parts.append("Related entities: " + ", ".join(raw_output["neighbor_entities"]))
        return "\n".join(parts)

    # AggregationAgent: derived claim
    if "derived_claim_text" in raw_output:
        return raw_output["derived_claim_text"]

    # EntityLinkerAgent: entity info
    if "entity_id" in raw_output and raw_output.get("entity_id"):
        return f"Entity linked: {raw_output['entity_id']} (confidence: {raw_output.get('match_confidence', 0):.2f})"

    # Fallback: stringify
    return json.dumps(raw_output)[:1000]


def _extract_provenance(raw_output: dict) -> list[str]:
    """Extract source passages / IDs for provenance chain."""
    provenance = []
    if "chunks" in raw_output:
        for c in raw_output.get("chunks", [])[:3]:
            if c.get("chunk_id"):
                provenance.append(c["chunk_id"])
    elif "doc_id" in raw_output:
        provenance.append(raw_output["doc_id"])
    elif "entity_id" in raw_output and raw_output.get("entity_id"):
        provenance.append(raw_output["entity_id"])
    return provenance


def _fail_result(target_slot_id: str, step_id: int, reason: str) -> dict:
    return {
        "claim_text": "",
        "entailment_flag": "FAIL",
        "confidence": 0.0,
        "contradiction_detected": False,
        "contradicted_claim_id": None,
        "subject": "",
        "predicate": "",
        "object": "",
        "provenance_chain": [],
        "derived_from": [],
        "target_slot_id": target_slot_id,
        "step_id": step_id,
        "tokens_used": 0,
        "reasoning": f"fail: {reason}",
        "valid_from": None,
        "valid_to": None,
        "source_authority": None,
    }


def _parse_fallback(raw: str) -> dict:
    """Best-effort extraction when JSON parsing fails."""
    entailment = "PASS" if "PASS" in raw else "FAIL"
    # Try to extract confidence
    import re
    conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
    confidence = float(conf_match.group(1)) if conf_match else (0.7 if entailment == "PASS" else 0.0)
    return {
        "claim_text": raw[:300],
        "entailment_flag": entailment,
        "confidence": confidence,
        "contradiction_detected": False,
        "contradicted_claim_id": None,
        "subject": "",
        "predicate": "",
        "object": raw[:100],
        "reasoning": "parsed from free text",
    }


def _log_invocation(agent_name, question_id, step_id, input_summary,
                     tokens_used, t_start, output_summary):
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
        logger.error(f"Failed to log EvidenceEvaluator invocation: {e}")
