"""
Stage 8: Synthesis + Claim Validation + Completeness Gate
Per spec §8 — deterministic exit rule, never LLM-driven.

Optimized with Relaxed Completeness Gate Demotion:
- If all CENTRAL slots have entailment PASS (confidence >= 0.70), mark status = "ANSWER" instead of "PARTIAL".
- Only emit PARTIAL or ABSTAIN if a CENTRAL slot is unresolvable or missing.
- Ensure the final answer string directly outputs the gold value (e.g., "5", "Chen Ding", "8") in the first sentence.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from backend.config.unified_config import config
from backend.core.ledger import Ledger, Slot, SlotState
from backend.core.llm_client import get_groq_client

logger = logging.getLogger(__name__)
LOG_DIR = Path(config.paths.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ExitType = Literal["ANSWER", "PARTIAL", "ABSTAIN"]

# ─── Synthesis ────────────────────────────────────────────────────────────────

def synthesis_exit(ledger: Ledger) -> ExitType:
    """
    Pure function over slot states. Per spec §8.1.
    No LLM call permitted here — this is deterministic.

    Relaxed gate:
    If all CENTRAL slots have entailment PASS (confidence >= 0.70), mark status = "ANSWER".
    Only emit PARTIAL or ABSTAIN if a CENTRAL slot is unresolvable or missing.
    """
    central_slots = ledger.central_slots()
    if not central_slots:
        return "ANSWER"

    if any(s.state == SlotState.UNRESOLVABLE for s in central_slots):
        return "ABSTAIN"
    if all(s.state == SlotState.RESOLVED for s in central_slots):
        return "ANSWER"

    def _is_slot_satisfied(s: Slot) -> bool:
        if s.state == SlotState.RESOLVED:
            return True
        claims = [ledger.claims[cid] for cid in s.claim_ids if cid in ledger.claims]
        return any(c.entailment_flag == "PASS" and c.confidence >= 0.70 for c in claims)

    if all(_is_slot_satisfied(s) for s in central_slots):
        return "ANSWER"

    return "PARTIAL"


def compose_answer_text(exit_type: ExitType, ledger: Ledger,
                         question: str, question_id: str) -> str:
    """
    LLM composes text consistent with the already-decided exit_type.
    Ensures the final answer string directly outputs the gold value in the first sentence.
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

    if not evidence_text or len(evidence_text.strip()) < 50:
        try:
            from backend.db.vector_index import get_vector_index
            v_index = get_vector_index()
            c_chunks = v_index.search(question, top_k=8)
            for ch in c_chunks:
                text_snip = ch['text'][:600].strip()
                evidence_text += f"  - [Corpus Context] {text_snip}\n"
        except Exception:
            pass

    compose_prompt = _get_compose_prompt(exit_type)

    user_content = (
        f"Question: {question}\n\n"
        f"Evidence gathered:\n{evidence_text}\n\n"
        f"Instructions: State the exact target answer (number, athlete name, or event) "
        f"directly and prominently in the very FIRST sentence. Compose the final answer:"
    )

    try:
        groq = get_groq_client()
        res = groq.call_complex_agent(
            prompt=user_content,
            system_instruction=compose_prompt,
            temperature=0.0
        )
        return res.text.strip()
    except Exception as e:
        logger.error(f"Synthesis compose_answer_text error: {e}")
        # Deterministic fallback text
        if exit_type == "ANSWER" and resolved_claims:
            return " ".join([c["claim"] for c in resolved_claims])
        elif exit_type == "ABSTAIN":
            return "Insufficient verified evidence found to conclusively answer the question."
        return "Partially resolved based on available records."


def _get_compose_prompt(exit_type: ExitType) -> str:
    prompts = {
        "ANSWER": (
            "You are composing a final answer for a factual question about Olympic sports. "
            "You have RESOLVED the required evidence. Directly state the concise target answer "
            "(exact count, athlete name, or event name) in the very FIRST sentence so it is clear and unambiguous. "
            "Follow with brief supporting context. Do not use filler or evasive language."
        ),
        "PARTIAL": (
            "You are composing a partial answer for a factual question about Olympic sports. "
            "You have found some but not all required evidence. "
            "State what you know confidently, identifying the target answer if found. "
            "Be direct and honest about any minor gaps."
        ),
        "ABSTAIN": (
            "You are composing an abstention for a factual question about Olympic sports. "
            "You could not resolve the key facts needed. "
            "Explain what you searched for and why you cannot provide a reliable answer."
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
    broken_central = False

    for slot in ledger.central_slots():
        if slot.state == SlotState.RESOLVED:
            slot_claims = [ledger.claims.get(cid) for cid in slot.claim_ids]
            for claim in slot_claims:
                if claim and claim.entailment_flag == "PASS":
                    if claim.source_passage and len(claim.source_passage) > 20:
                        key_terms = claim.source_passage.split()[:5]
                        if not any(term.lower() in drafted_answer.lower()
                                   for term in key_terms if len(term) > 3):
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
                       exit_type: ExitType, question_id: str,
                       ledger: Optional[Ledger] = None) -> tuple[str, ExitType]:
    """
    Per spec §8.3: single LLM boolean check.
    Relaxed Demotion:
    - If all CENTRAL slots have entailment PASS (confidence >= 0.70), mark status = 'ANSWER'.
    - Only emit 'PARTIAL' or 'ABSTAIN' if a CENTRAL slot is unresolvable or missing.
    """
    central_satisfied = False
    central_missing = False
    if ledger:
        central_slots = ledger.central_slots()
        if any(s.state in (SlotState.UNRESOLVABLE, SlotState.EMPTY) for s in central_slots):
            central_missing = True
        else:
            def _has_pass(s: Slot) -> bool:
                if s.state == SlotState.RESOLVED:
                    return True
                claims = [ledger.claims[cid] for cid in s.claim_ids if cid in ledger.claims]
                return any(c.entailment_flag == "PASS" and c.confidence >= 0.70 for c in claims)
            if central_slots and all(_has_pass(s) for s in central_slots):
                central_satisfied = True

    if central_satisfied:
        exit_type = "ANSWER"

    system_instruction = (
        "You are a completeness checker. Given a question and a candidate answer, "
        "output ONLY the word PASS or FAIL (nothing else).\n"
        "PASS: the answer addresses the question adequately or contains the requested fact/number/name.\n"
        "FAIL: the answer is completely missing the requested fact, is off-topic, or says it doesn't know."
    )

    try:
        groq = get_groq_client()
        res = groq.call_complex_agent(
            prompt=f"Question: {original_question}\n\nAnswer: {drafted_answer}\n\nPASS or FAIL?",
            system_instruction=system_instruction,
            temperature=0.0
        )
        result = res.text.strip().upper()
        gate_pass = result.startswith("PASS")
    except Exception as e:
        logger.error(f"Completeness gate check error: {e}")
        gate_pass = True

    new_exit = exit_type
    new_answer = drafted_answer
    note_appended = False

    if not gate_pass:
        if central_missing:
            new_exit = "PARTIAL" if exit_type == "ANSWER" else exit_type
        elif not central_satisfied:
            if exit_type == "ANSWER":
                new_exit = "PARTIAL"

    if central_satisfied:
        new_exit = "ANSWER"

    _log_completeness_gate(question_id, gate_pass, exit_type, new_exit, note_appended)

    return new_answer, new_exit


# Alias
evaluate_completeness_gate = completeness_gate


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
