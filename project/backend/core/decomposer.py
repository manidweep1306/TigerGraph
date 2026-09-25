"""
Stage 2: Decomposer — converts a question into a slot set.
Per spec §3.4 and §4.6: a single LLM call, not an iterative agent.
Outputs slots with CENTRAL/PERIPHERAL labels + requires_aggregation flag.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from backend.core.ledger import Slot, SlotCriticality, SlotState

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

DECOMPOSE_SYSTEM_PROMPT = """You are an expert evidence-planning system for a GraphRAG investigation engine.

Your task: Given a question, decompose it into a list of INFORMATION SLOTS — the distinct pieces of information that must be found to answer the question completely.

For each slot, you must:
1. Write a clear, specific description of what information is needed
2. Label criticality: CENTRAL (question cannot be answered without it) or PERIPHERAL (adds detail but not required)
3. Optionally identify an assumed_entity (entity name the slot assumes exists)

Also determine if the question requires aggregation (counting, comparing, ranking across multiple pieces of data).

QUESTION TYPES AND DECOMPOSITION GUIDANCE:
- lookup: Usually 1-2 CENTRAL slots
- temporal: 1-3 CENTRAL slots (often need to find the "immediately before/after" Olympics first)  
- multi_hop: 2-4 CENTRAL slots with sequential dependencies
- aggregation: Multiple CENTRAL slots (one per item to count/compare) + requires_aggregation=true
- superlative: Multiple CENTRAL slots (need all candidates) + requires_aggregation=true

OUTPUT FORMAT (JSON only, no markdown):
{
  "slots": [
    {
      "slot_id": "s1",
      "description": "...",
      "criticality": "CENTRAL" | "PERIPHERAL",
      "assumed_entity": "entity name or null"
    }
  ],
  "requires_aggregation": true | false,
  "question_type_assessment": "lookup | temporal | multi_hop | aggregation | superlative"
}"""


@dataclass
class DecompositionResult:
    question_id: str
    question: str
    slots: list[Slot]
    requires_aggregation: bool
    question_type_assessment: str
    raw_llm_output: str


def decompose_question(question: str, question_id: str,
                       context: Optional[dict] = None) -> DecompositionResult:
    """
    Decompose a question into slots for the Ledger.
    Called once at start OR once during a decomposition revision (§5).

    Args:
        question: The question string
        question_id: For logging
        context: Optional ledger state for revision-mode decomposition (§5.5)

    Returns:
        DecompositionResult with slots ready to load into Ledger
    """
    model = genai.GenerativeModel(
        model_name=MODEL,
        generation_config=genai.GenerationConfig(temperature=0.0),
        system_instruction=DECOMPOSE_SYSTEM_PROMPT,
    )

    # Build user prompt
    user_content = f"Question: {question}"
    if context:
        user_content += f"\n\nExisting evidence context (revision mode):\n{json.dumps(context, indent=2)}"
    user_content += "\n\nDecompose this question into information slots."

    from backend.core.gemini_utils import generate_content_with_retry
    response = generate_content_with_retry(model, user_content)
    raw = response.text.strip()

    # Parse JSON output
    try:
        # Handle potential markdown code fence
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        logger.warning(f"Decomposer JSON parse error for q={question_id}: {e}. Fallback.")
        data = _fallback_decomposition(question)

    # Build Slot objects
    slots = []
    for slot_data in data.get("slots", []):
        slot = Slot(
            slot_id=f"{question_id}_{slot_data['slot_id']}",
            description=slot_data["description"],
            criticality=SlotCriticality(slot_data.get("criticality", "CENTRAL")),
            assumed_entity=slot_data.get("assumed_entity"),
            requires_aggregation=data.get("requires_aggregation", False),
        )
        slots.append(slot)

    if not slots:
        # Absolute fallback: one CENTRAL slot
        slots = [Slot(
            slot_id=f"{question_id}_s1",
            description=f"Find the answer to: {question}",
            criticality=SlotCriticality.CENTRAL,
        )]

    return DecompositionResult(
        question_id=question_id,
        question=question,
        slots=slots,
        requires_aggregation=data.get("requires_aggregation", False),
        question_type_assessment=data.get("question_type_assessment", "unknown"),
        raw_llm_output=raw,
    )


def _fallback_decomposition(question: str) -> dict:
    """Rule-based fallback when LLM output fails to parse."""
    q_lower = question.lower()

    if any(w in q_lower for w in ["how many", "count", "number of"]):
        return {
            "slots": [
                {"slot_id": "s1", "description": "Find all relevant events/items matching the criteria",
                 "criticality": "CENTRAL", "assumed_entity": None},
                {"slot_id": "s2", "description": "Count the matching items",
                 "criticality": "CENTRAL", "assumed_entity": None},
            ],
            "requires_aggregation": True,
            "question_type_assessment": "aggregation",
        }
    elif any(w in q_lower for w in ["immediately before", "immediately after", "previous", "next"]):
        return {
            "slots": [
                {"slot_id": "s1", "description": "Identify the reference Olympics year",
                 "criticality": "CENTRAL", "assumed_entity": None},
                {"slot_id": "s2", "description": "Find the adjacent Olympics",
                 "criticality": "CENTRAL", "assumed_entity": None},
                {"slot_id": "s3", "description": "Find the answer at the adjacent Olympics",
                 "criticality": "CENTRAL", "assumed_entity": None},
            ],
            "requires_aggregation": False,
            "question_type_assessment": "temporal",
        }
    elif any(w in q_lower for w in ["which", "highest", "most", "best", "largest"]):
        return {
            "slots": [
                {"slot_id": "s1", "description": "Find all candidates and their values",
                 "criticality": "CENTRAL", "assumed_entity": None},
                {"slot_id": "s2", "description": "Identify the superlative",
                 "criticality": "CENTRAL", "assumed_entity": None},
            ],
            "requires_aggregation": True,
            "question_type_assessment": "superlative",
        }
    else:
        return {
            "slots": [
                {"slot_id": "s1", "description": f"Find the answer to: {question}",
                 "criticality": "CENTRAL", "assumed_entity": None},
            ],
            "requires_aggregation": False,
            "question_type_assessment": "lookup",
        }
