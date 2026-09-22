"""
Stage 8: Synthesis + Claim Validation + Completeness Gate
Per spec §8 — deterministic exit rule, never LLM-driven.

INVARIANTS:
- Synthesis exit type is decided by PURE FUNCTION over slot states — never by LLM
- Completeness Gate can only leave output unchanged OR downgrade ONE level
- Completeness Gate NEVER calls Decomposer, agents, or Ledger
- Claim Validation can only remove/downgrade, never add new claims
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import google.generativeai as genai
from dotenv import load_dotenv

from backend.core.ledger import Ledger, SlotState

load_dotenv()
logger = logging.getLogger(__name__)
LOG_DIR = Path("./logs")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

ExitType = Literal["ANSWER", "PARTIAL", "ABSTAIN"]

# ─── Synthesis ────────────────────────────────────────────────────────────────

def synthesis_exit(ledger: Ledger) -> ExitType:
    """
    Pure function over slot states. Per spec §8.1.
    No LLM call permitted here — this is deterministic.
    """
    central_slots = ledger.central_slots()

    if any(s.state == SlotState.UNRESOLVABLE for s in central_slots):
        return "ABSTAIN"
    if all(s.state == SlotState.RESOLVED for s in central_slots):
        return "ANSWER"
    return "PARTIAL"


def compose_answer_text(exit_type: ExitType, ledger: Ledger,
                         question: str, question_id: str) -> str:
    """
    LLM composes text consistent with the already-decided exit_type.
    Per spec §8.1: LLM only composes text, never chooses exit type.
    """
    # Build evidence summary from claims
    resolved_claims = []
    partial_claims = []
    unresolvable_slots = []

    for slot in ledger.slots.values():
        slot_claims = ledger.get_claims_for_slot(slot.slot_id)
        passing = [c for c in slot_claims if c.entailment_flag == "PASS"]

        if slot.state == SlotState.RESOLVED and passing:
            best = max(passing, key=lambda c: c.confidence)
            resolved_claims.append({
                "description": slot.description,
                "claim": best.source_passage,
                "confidence": best.confidence,
                "sources": best.provenance_chain,
            })
        elif slot.state in (SlotState.SUPPORTED, SlotState.CONTESTED) and passing:
            best = max(passing, key=lambda c: c.confidence)
            partial_claims.append({
                "description": slot.description,
                "claim": best.source_passage,
                "confidence": best.confidence,
            })
        elif slot.state == SlotState.UNRESOLVABLE:
            unresolvable_slots.append(slot.description)

    evidence_text = ""
    if resolved_claims:
        evidence_text += "RESOLVED EVIDENCE:\n"
        for c in resolved_claims:
            evidence_text += f"  - {c['claim']} (sources: {', '.join(c['sources'][:2])})\n"
    if partial_claims:
        evidence_text += "PARTIAL EVIDENCE:\n"
        for c in partial_claims:
            evidence_text += f"  - {c['claim']}\n"
    if unresolvable_slots:
        evidence_text += "UNRESOLVABLE:\n"
        for d in unresolvable_slots:
            evidence_text += f"  - {d}\n"

    compose_prompt = _get_compose_prompt(exit_type)

    model = genai.GenerativeModel(
        model_name=MODEL,
        generation_config=genai.GenerationConfig(temperature=0.0),
        system_instruction=compose_prompt,
    )

    user_content = (
        f"Question: {question}\n\n"
        f"Evidence gathered:\n{evidence_text}\n\n"
        f"Compose the final answer."
    )

    response = model.generate_content(user_content)
    return response.text.strip()


def _get_compose_prompt(exit_type: ExitType) -> str:
    prompts = {
        "ANSWER": (
            "You are composing a final answer for a factual question about Olympic sports. "
            "You have RESOLVED all required evidence. Compose a clear, concise, direct answer. "
            "Cite your sources where relevant. Do not hedge unnecessarily."
        ),
        "PARTIAL": (
            "You are composing a partial answer for a factual question about Olympic sports. "
            "You have found some but not all required evidence. "
            "State what you know confidently, and clearly identify what you could not determine. "
            "Be honest about the gaps."
        ),
        "ABSTAIN": (
            "You are composing an abstention for a factual question about Olympic sports. "
            "You could not resolve the key facts needed. "
            "Explain what you searched for, what conflicting information was found, "
            "and why you cannot provide a reliable answer."
        ),
    }
    return prompts.get(exit_type, prompts["PARTIAL"])


# ─── Claim Validation ─────────────────────────────────────────────────────────

def claim_validation(drafted_answer: str, ledger: Ledger,
                      exit_type: ExitType, question_id: str,
                      step_id: int) -> tuple[str, ExitType]:
    """
    Per spec §8.2: re-run entailment check on claims in drafted answer.
    Can only remove claims and downgrade exit_type by one level.
    NEVER calls agents, Decomposer, or Ledger write.
    """
    # Build set of claims used in the answer (by searching for claim text fragments)
    broken_central = False

    for slot in ledger.central_slots():
        if slot.state == SlotState.RESOLVED:
            slot_claims = [ledger.claims.get(cid) for cid in slot.claim_ids]
            for claim in slot_claims:
                if claim and claim.entailment_flag == "PASS":
                    # Re-check: is the claim still in the answer?
                    if claim.source_passage and len(claim.source_passage) > 20:
                        # Simple heuristic: check if key terms appear
                        key_terms = claim.source_passage.split()[:5]
                        if not any(term.lower() in drafted_answer.lower()
                                   for term in key_terms if len(term) > 3):
                            # Claim seems stripped — flag it
                            broken_central = True
                            _log_claim_validation_failure(
                                question_id, claim.claim_id,
                                exit_type,
                                "answer_missing_claim_text",
                            )

    new_exit = exit_type
    if broken_central:
        if exit_type == "ANSWER":
            new_exit = "PARTIAL"
        elif exit_type == "PARTIAL":
            new_exit = "ABSTAIN"
        logger.info(f"Claim validation downgraded exit: {exit_type} → {new_exit}")

    return drafted_answer, new_exit


# ─── Completeness Gate ────────────────────────────────────────────────────────

def completeness_gate(drafted_answer: str, original_question: str,
                       exit_type: ExitType, question_id: str) -> tuple[str, ExitType]:
    """
    Per spec §8.3: single LLM boolean check.
    Can only downgrade by ONE level and append a note.
    HARD PROHIBITION: NEVER calls Decomposer, any agent, or Ledger.
    """
    model = genai.GenerativeModel(
        model_name=MODEL,
        generation_config=genai.GenerationConfig(temperature=0.0),
        system_instruction=(
            "You are a completeness checker. Given a question and a candidate answer, "
            "output ONLY the word PASS or FAIL (nothing else).\n"
            "PASS: the answer addresses the question adequately.\n"
            "FAIL: the answer is missing key parts, is off-topic, or is evasive."
        ),
    )

    response = model.generate_content(
        f"Question: {original_question}\n\nAnswer: {drafted_answer}\n\nPASS or FAIL?"
    )
    result = response.text.strip().upper()
    gate_pass = result.startswith("PASS")

    new_exit = exit_type
    new_answer = drafted_answer
    note_appended = False

    if not gate_pass:
        if exit_type == "ANSWER":
            new_exit = "PARTIAL"
            new_answer += "\n\n[Note: completeness gate flagged: may not fully address question]"
            note_appended = True
        elif exit_type == "PARTIAL":
            # stays PARTIAL, append note
            new_answer += "\n\n[Note: completeness gate flagged: may not fully address question]"
            note_appended = True
        # ABSTAIN is never further downgraded per spec §8.3

    _log_completeness_gate(question_id, gate_pass, exit_type, new_exit, note_appended)

    return new_answer, new_exit


# ─── Logging helpers ──────────────────────────────────────────────────────────

def _log_claim_validation_failure(question_id: str, claim_id: str,
                                    old_exit: str, reason: str) -> None:
    record = {
        "question_id": question_id,
        "claim_id": claim_id,
        "reason": reason,
        "forced_downgrade": True,
        "old_exit_type": old_exit,
        "new_exit_type": "PARTIAL" if old_exit == "ANSWER" else "ABSTAIN",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_log(LOG_DIR / "claim_validation_failures.jsonl", record)


def _log_completeness_gate(question_id: str, gate_pass: bool,
                             exit_before: str, exit_after: str,
                             note_appended: bool) -> None:
    record = {
        "question_id": question_id,
        "result": "PASS" if gate_pass else "FAIL",
        "exit_type_before": exit_before,
        "exit_type_after": exit_after,
        "note_appended": note_appended,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_log(LOG_DIR / "completeness_gate_log.jsonl", record)


def _append_log(path: Path, record: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to write synthesis log: {e}")
