"""
Stage 1: Ledger and Slot State Machine
Per spec §4 — the heart of the Agentic pipeline.

INVARIANTS (never violate):
- The Ledger never evaluates evidence. All entailment/confidence logic is in EvidenceEvaluatorAgent.
- A claim with entailment_flag == FAIL is never written to the Ledger.
- RESOLVED and UNRESOLVABLE are terminal states.
- RESOLVED → CONTESTED is NOT permitted.
- Confidence inheritance: min over all derived_from chain.
"""

import uuid
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Slot State Machine
# ─────────────────────────────────────────────────────────────────────────────

class SlotState(str, Enum):
    EMPTY = "EMPTY"
    SUPPORTED = "SUPPORTED"
    CONTESTED = "CONTESTED"
    RESOLVED = "RESOLVED"
    UNRESOLVABLE = "UNRESOLVABLE"

    def ordinal(self) -> int:
        """Per spec §4.1 — for VoI gain calculations."""
        return {
            SlotState.EMPTY: 0,
            SlotState.SUPPORTED: 1,
            SlotState.CONTESTED: 1,
            SlotState.RESOLVED: 2,
            SlotState.UNRESOLVABLE: 2,
        }[self]

    def is_terminal(self) -> bool:
        return self in (SlotState.RESOLVED, SlotState.UNRESOLVABLE)

    def is_open(self) -> bool:
        return self in (SlotState.EMPTY, SlotState.SUPPORTED, SlotState.CONTESTED)


class SlotCriticality(str, Enum):
    CENTRAL = "CENTRAL"
    PERIPHERAL = "PERIPHERAL"


@dataclass
class Slot:
    slot_id: str
    description: str
    criticality: SlotCriticality
    state: SlotState = SlotState.EMPTY
    assumed_entity: Optional[str] = None  # for premise-contradiction detection §5.2
    requires_aggregation: bool = False

    # Tracking
    claim_ids: list[str] = field(default_factory=list)
    action_attempts: dict[str, int] = field(default_factory=dict)  # agent_type → count

    def weight(self) -> int:
        """VoI weight per spec §6.1."""
        return 3 if self.criticality == SlotCriticality.CENTRAL else 1

    def confidence_bar(self, thresholds: dict) -> float:
        """Minimum confidence required for RESOLVED per §4.3."""
        if self.criticality == SlotCriticality.CENTRAL:
            return thresholds.get("bar_CENTRAL", 0.75)
        return thresholds.get("bar_PERIPHERAL", 0.5)

    def record_action_attempt(self, agent_type: str) -> None:
        self.action_attempts[agent_type] = self.action_attempts.get(agent_type, 0) + 1

    def has_attempted(self, agent_type: str) -> bool:
        return self.action_attempts.get(agent_type, 0) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Claim Record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    claim_id: str
    question_id: str
    step_id: int
    target_slot_id: str

    # Core triple
    subject: str
    predicate: str
    object: str

    # Evidence
    source_passage: str
    entailment_flag: str  # "PASS" | "FAIL"
    confidence: float

    # Provenance
    provenance_chain: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)  # claim_ids

    # Contradiction tracking
    contradiction_detected: bool = False
    contradicted_claim_id: Optional[str] = None

    # Temporal metadata (null in Round 1, populated in Round 2)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source_authority: Optional[str] = None

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "question_id": self.question_id,
            "step_id": self.step_id,
            "target_slot_id": self.target_slot_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source_passage": self.source_passage,
            "entailment_flag": self.entailment_flag,
            "confidence": self.confidence,
            "provenance_chain": self.provenance_chain,
            "derived_from": self.derived_from,
            "contradiction_detected": self.contradiction_detected,
            "contradicted_claim_id": self.contradicted_claim_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "source_authority": self.source_authority,
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Ledger
# ─────────────────────────────────────────────────────────────────────────────

class Ledger:
    """
    Passive state recorder. Never evaluates evidence.
    All evaluation logic lives exclusively in EvidenceEvaluatorAgent.

    Per spec §4.9:
    - Receives EvaluatedClaim objects from EvidenceEvaluatorAgent
    - Applies slot state transitions per §4.2
    - Performs confidence inheritance per §4.7
    - Logs every write to ledger_writes.jsonl
    """

    def __init__(self, question_id: str, thresholds: dict):
        self.question_id = question_id
        self.thresholds = thresholds
        self.slots: dict[str, Slot] = {}
        self.claims: dict[str, Claim] = {}
        self._log_file = LOG_DIR / "ledger_writes.jsonl"

    def add_slot(self, slot: Slot) -> None:
        self.slots[slot.slot_id] = slot

    def get_slot(self, slot_id: str) -> Optional[Slot]:
        return self.slots.get(slot_id)

    def open_slots(self) -> list[Slot]:
        """Slots in EMPTY, SUPPORTED, or CONTESTED state."""
        return [s for s in self.slots.values() if s.state.is_open()]

    def central_slots(self) -> list[Slot]:
        return [s for s in self.slots.values()
                if s.criticality == SlotCriticality.CENTRAL]

    def all_central_resolved(self) -> bool:
        return all(s.state == SlotState.RESOLVED for s in self.central_slots())

    def any_central_unresolvable(self) -> bool:
        return any(s.state == SlotState.UNRESOLVABLE for s in self.central_slots())

    def coverage_fraction(self) -> float:
        """Fraction of CENTRAL slots RESOLVED (for completeness metric)."""
        central = self.central_slots()
        if not central:
            return 1.0
        resolved = sum(1 for s in central if s.state == SlotState.RESOLVED)
        return resolved / len(central)

    def get_claims_for_slot(self, slot_id: str) -> list[Claim]:
        return [c for c in self.claims.values() if c.target_slot_id == slot_id]

    def write(self, evaluated_claim: dict) -> tuple[str, SlotState]:
        """
        The Ledger's single write function. Per spec §4.9.

        Input: EvaluatedClaim object from EvidenceEvaluatorAgent.
        Returns: (claim_id, resulting_slot_state)

        NEVER raises — failed writes are logged and return gracefully.
        """
        slot_id = evaluated_claim.get("target_slot_id", "")
        entailment = evaluated_claim.get("entailment_flag", "FAIL")

        slot = self.slots.get(slot_id)
        if slot is None:
            self._log_invalid_transition(slot_id, "UNKNOWN", "WRITE", "slot_not_found")
            return ("", SlotState.EMPTY)

        # Per spec §4.9 and invariant #3: FAIL claims are never written
        if entailment != "PASS":
            self._log_discarded_claim(evaluated_claim, slot)
            return ("", slot.state)

        # Terminal state guard (invariant #4)
        if slot.state.is_terminal():
            # Log post_resolution_conflict if RESOLVED, but don't change state
            if slot.state == SlotState.RESOLVED:
                self._log_post_resolution_conflict(evaluated_claim, slot)
            return ("", slot.state)

        # Build the Claim record
        claim_id = str(uuid.uuid4())
        claim = self._build_claim(claim_id, evaluated_claim, slot_id)

        # Confidence inheritance per §4.7
        claim.confidence = self._apply_confidence_inheritance(claim)

        # Check for contradiction
        existing_claims = self.get_claims_for_slot(slot_id)
        contradiction_claim = self._detect_contradiction(claim, existing_claims)
        if contradiction_claim:
            claim.contradiction_detected = True
            claim.contradicted_claim_id = contradiction_claim.claim_id

        # Compute new slot state per §4.2
        old_state = slot.state
        new_state = self._compute_new_state(slot, claim, existing_claims)

        # Validate transition
        if not self._is_valid_transition(old_state, new_state):
            self._log_invalid_transition(slot_id, old_state, new_state, "invalid_transition")
            return ("", old_state)

        # Commit: write claim and update slot
        self.claims[claim_id] = claim
        slot.state = new_state
        slot.claim_ids.append(claim_id)

        # Log to ledger_writes.jsonl per §11.3
        self._log_write(claim, slot)

        return (claim_id, new_state)

    def mark_unresolvable(self, slot_id: str) -> None:
        """
        Mark a CONTESTED slot as UNRESOLVABLE per §4.5.
        Called by coverage-check when no actions remain for the slot.
        """
        slot = self.slots.get(slot_id)
        if slot and slot.state == SlotState.CONTESTED:
            slot.state = SlotState.UNRESOLVABLE
            logger.info(f"Slot {slot_id} marked UNRESOLVABLE")

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _build_claim(self, claim_id: str, evaluated_claim: dict, slot_id: str) -> Claim:
        """Construct a Claim from EvidenceEvaluatorAgent output."""
        return Claim(
            claim_id=claim_id,
            question_id=self.question_id,
            step_id=evaluated_claim.get("step_id", 0),
            target_slot_id=slot_id,
            subject=evaluated_claim.get("subject", ""),
            predicate=evaluated_claim.get("predicate", ""),
            object=evaluated_claim.get("object", ""),
            source_passage=evaluated_claim.get("claim_text", ""),
            entailment_flag=evaluated_claim.get("entailment_flag", "PASS"),
            confidence=evaluated_claim.get("confidence", 0.5),
            provenance_chain=evaluated_claim.get("provenance_chain", []),
            derived_from=evaluated_claim.get("derived_from", []),
            valid_from=None,    # Round 1: always null
            valid_to=None,      # Round 1: always null
            source_authority=None,  # Round 1: always null
        )

    def _apply_confidence_inheritance(self, claim: Claim) -> float:
        """
        Per spec §4.7:
        confidence = min(own_evidence_confidence,
                         min(c.confidence for c in derived_from))
        """
        own_conf = claim.confidence
        if not claim.derived_from:
            return own_conf

        ancestor_confs = []
        for parent_id in claim.derived_from:
            parent = self.claims.get(parent_id)
            if parent:
                ancestor_confs.append(parent.confidence)

        if ancestor_confs:
            return min(own_conf, min(ancestor_confs))
        return own_conf

    def _detect_contradiction(self, new_claim: Claim,
                               existing: list[Claim]) -> Optional[Claim]:
        """
        Per §4.4: same (subject, predicate) but different object → CONTESTED.
        """
        for existing_claim in existing:
            if (existing_claim.subject == new_claim.subject
                    and existing_claim.predicate == new_claim.predicate
                    and existing_claim.object != new_claim.object):
                return existing_claim
        return None

    def _compute_new_state(self, slot: Slot, new_claim: Claim,
                            existing_claims: list[Claim]) -> SlotState:
        """
        Apply state transition rules per §4.2.
        """
        # If contradiction, go to CONTESTED
        if new_claim.contradiction_detected:
            return SlotState.CONTESTED

        # Check if RESOLVED criteria are met per §4.3
        all_claims = existing_claims + [new_claim]
        if self._meets_resolved_criteria(slot, all_claims):
            return SlotState.RESOLVED

        # Otherwise: EMPTY → SUPPORTED, SUPPORTED/CONTESTED → stays or upgrades
        if slot.state == SlotState.EMPTY:
            return SlotState.SUPPORTED
        return slot.state  # SUPPORTED stays SUPPORTED until RESOLVED criteria met

    def _meets_resolved_criteria(self, slot: Slot, claims: list[Claim]) -> bool:
        """
        Four criteria per §4.3 (ALL required):
        1. ≥1 claim with entailment_flag == PASS
        2. claim.confidence ≥ bar(slot.criticality)
        3. no CONTESTED sibling remains unresolved
        4. CENTRAL: ≥2 independent-source claims agree OR 1 high-conf direct claim
           PERIPHERAL: conditions 1–3 sufficient
        """
        bar = slot.confidence_bar(self.thresholds)
        passing = [c for c in claims if c.entailment_flag == "PASS"]

        # Criterion 1
        if not passing:
            return False

        # Criterion 2: at least one passing claim meets the confidence bar
        high_conf = [c for c in passing if c.confidence >= bar]
        if not high_conf:
            return False

        # Criterion 3: no unresolved contradiction (handled by state — if CONTESTED,
        # we still check if new evidence resolves it)
        # For PERIPHERAL, criteria 1-3 suffice
        if slot.criticality == SlotCriticality.PERIPHERAL:
            return True

        # Criterion 4 for CENTRAL:
        # Option A: 1 claim with confidence ≥ 0.9 AND provenance_chain depth ≤ 1
        direct_high_conf = [c for c in high_conf
                            if c.confidence >= 0.9 and len(c.provenance_chain) <= 1]
        if direct_high_conf:
            return True

        # Option B: ≥2 independent-source claims agree on same (subject, predicate, object)
        if self._has_independent_agreement(passing):
            return True

        return False

    def _has_independent_agreement(self, claims: list[Claim]) -> bool:
        """
        Per §4.3: two claims agree on same (subject, predicate, object) AND
        their derived_from chains don't share a common upstream claim.
        """
        for i, c1 in enumerate(claims):
            for c2 in claims[i+1:]:
                if (c1.subject == c2.subject
                        and c1.predicate == c2.predicate
                        and c1.object == c2.object
                        and not set(c1.provenance_chain) & set(c2.provenance_chain)):
                    return True
        return False

    def _is_valid_transition(self, from_state: SlotState, to_state: SlotState) -> bool:
        """Per §4.2 — only explicitly listed transitions are valid."""
        valid = {
            SlotState.EMPTY: {SlotState.EMPTY, SlotState.SUPPORTED, SlotState.CONTESTED, SlotState.RESOLVED},
            SlotState.SUPPORTED: {SlotState.SUPPORTED, SlotState.CONTESTED, SlotState.RESOLVED},
            SlotState.CONTESTED: {SlotState.CONTESTED, SlotState.RESOLVED, SlotState.UNRESOLVABLE},
            SlotState.RESOLVED: {SlotState.RESOLVED},      # idempotent only
            SlotState.UNRESOLVABLE: {SlotState.UNRESOLVABLE},  # terminal
        }
        return to_state in valid.get(from_state, set())

    # ─── Logging helpers ────────────────────────────────────────────────────

    def _log_write(self, claim: Claim, slot: Slot) -> None:
        """Log to ledger_writes.jsonl per §11.3."""
        record = claim.to_dict()
        record["resulting_slot_state"] = slot.state.value
        self._append_log(self._log_file, record)

    def _log_discarded_claim(self, evaluated_claim: dict, slot: Slot) -> None:
        """Log discarded (entailment FAIL) claims."""
        record = {
            "event": "discarded_claim",
            "question_id": self.question_id,
            "target_slot_id": slot.slot_id,
            "slot_state_unchanged": slot.state.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append_log(self._log_file, record)

    def _log_invalid_transition(self, slot_id: str, from_state, to_state,
                                 reason: str) -> None:
        """LedgerGuard log per §4.2 — invalid transition attempts."""
        record = {
            "agent_name": "LedgerGuard",
            "question_id": self.question_id,
            "step_id": -1,
            "event": "invalid_transition_attempt",
            "slot_id": slot_id,
            "attempted_from": str(from_state),
            "attempted_to": str(to_state),
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_summary": f"slot={slot_id} from={from_state} to={to_state}",
            "tokens_used": 0,
            "latency_ms": 0,
            "output_summary": f"rejected: {reason}",
        }
        # Write to agent_invocations.jsonl per §3.3
        self._append_log(LOG_DIR / "agent_invocations.jsonl", record)

    def _log_post_resolution_conflict(self, evaluated_claim: dict, slot: Slot) -> None:
        """Log conflict against a RESOLVED slot without changing its state."""
        record = {
            "event": "post_resolution_conflict",
            "question_id": self.question_id,
            "target_slot_id": slot.slot_id,
            "slot_state_unchanged": slot.state.value,
            "claim_text": evaluated_claim.get("claim_text", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append_log(self._log_file, record)

    @staticmethod
    def _append_log(path: Path, record: dict) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log {path}: {e}")

    def snapshot(self) -> dict:
        """Full state snapshot for decomposition revision logging."""
        return {
            "slots": [
                {
                    "slot_id": s.slot_id,
                    "description": s.description,
                    "criticality": s.criticality.value,
                    "state": s.state.value,
                }
                for s in self.slots.values()
            ],
            "claim_count": len(self.claims),
        }
