"""
Stage 6: Ladder — ADAPTIVE mode router.
Per spec §6: routes a question to the cheapest rung that can answer it.
Escalates at most once per question (RAG → GraphRAG → Agentic).
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
LOG_DIR = Path("./logs")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

ROUTE_PROMPT = """You are a question complexity classifier for a GraphRAG system.

Classify the question into one of three tiers:
- RAG: Simple, single-hop questions. Clear entity, direct answer in one passage.
  Examples: "How many nations competed in X?", "Who won the gold medal in X?"
- GraphRAG: Multi-entity or temporal questions needing relationships across documents.
  Examples: "Who won at the Olympics BEFORE Y?", "Which event at the Y Olympics had the most X?"
- Agentic: Complex questions requiring multi-step investigation, aggregation across many docs,
  or ambiguous entities needing resolution.
  Examples: "How many biathlon events had more than 73 competitors?", questions needing counting across 10+ docs.

Question types mapped to tiers:
- lookup → RAG
- temporal → GraphRAG  
- multi_hop → GraphRAG or Agentic
- aggregation → Agentic
- superlative → Agentic or GraphRAG

Output ONLY one word: RAG, GraphRAG, or Agentic"""

CONFIDENCE_PROMPT = """Given a question and a generated answer, rate the confidence that the answer is correct and complete on a scale of 0.0 to 1.0.

Consider:
- Is the answer specific and concrete?
- Does it directly address the question?
- Does it avoid hedging phrases like "I'm not sure" or "might be"?
- Is it clearly grounded in evidence?

Output ONLY a float between 0.0 and 1.0, nothing else."""


def route(question: str, question_id: str, thresholds: dict) -> str:
    """
    Route a question to the appropriate pipeline rung.
    Returns: "RAG", "GraphRAG", or "Agentic"
    """
    try:
        model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config=genai.GenerationConfig(temperature=0.0),
            system_instruction=ROUTE_PROMPT,
        )
        response = model.generate_content(f"Question: {question}")
        rung = response.text.strip()

        # Normalize
        if "graphrag" in rung.lower():
            rung = "GraphRAG"
        elif "agentic" in rung.lower():
            rung = "Agentic"
        else:
            rung = "RAG"

        logger.info(f"Ladder routed q={question_id} → {rung}")
        return rung

    except Exception as e:
        logger.error(f"Ladder routing failed: {e}. Defaulting to Agentic.")
        return "Agentic"


def assess_confidence(answer: str, question: str) -> float:
    """
    Assess confidence of a pipeline answer for escalation decision.
    Returns float 0.0–1.0.
    """
    if not answer or answer in ("NO_EVIDENCE_RETRIEVED", "NO_ENTITY_MATCH"):
        return 0.0

    try:
        model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config=genai.GenerationConfig(temperature=0.0),
            system_instruction=CONFIDENCE_PROMPT,
        )
        response = model.generate_content(
            f"Question: {question}\n\nAnswer: {answer}"
        )
        conf = float(response.text.strip())
        return max(0.0, min(1.0, conf))

    except Exception:
        # Heuristic fallback
        if len(answer) < 10:
            return 0.2
        if any(h in answer.lower() for h in ["uncertain", "i don't", "not sure", "unclear"]):
            return 0.3
        return 0.6


def should_escalate(answer: str, question: str, current_rung: str,
                     thresholds: dict) -> bool:
    """
    Decide whether to escalate to the next rung.
    Per spec §1.3: at most ONE escalation per question.
    """
    θ_grounding = thresholds.get("θ_grounding", 0.5)
    confidence = assess_confidence(answer, question)
    return confidence < θ_grounding and current_rung != "Agentic"


def next_rung(current_rung: str) -> str:
    """Get the next rung in the escalation chain."""
    chain = {"RAG": "GraphRAG", "GraphRAG": "Agentic"}
    return chain.get(current_rung, "Agentic")


def log_escalation(question_id: str, from_rung: str, to_rung: str,
                    confidence_score: float, threshold: float) -> None:
    """Per spec §11.1."""
    record = {
        "question_id": question_id,
        "from_rung": from_rung,
        "to_rung": to_rung,
        "confidence_score": confidence_score,
        "threshold_used": threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_path = LOG_DIR / "escalation_log.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to log escalation: {e}")
