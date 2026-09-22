"""
Stage 5: VoI (Value-of-Information) Scorer
Per spec §6 — selects the highest-gain/cost-ratio action at each agent loop step.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.core.ledger import Ledger, Slot, SlotState, SlotCriticality

logger = logging.getLogger(__name__)
LOG_DIR = Path("./logs")


# Default prior gains per agent type (bootstrapped until calibration data accumulates)
# Represents expected ordinal gain per invocation
PRIOR_GAINS = {
    "EntityLinkerAgent": {"CENTRAL": 2.5, "PERIPHERAL": 0.8},
    "GraphTraversalAgent": {"CENTRAL": 2.0, "PERIPHERAL": 0.7},
    "SimilaritySearchAgent": {"CENTRAL": 1.5, "PERIPHERAL": 0.5},
    "DocumentRetrievalAgent": {"CENTRAL": 1.2, "PERIPHERAL": 0.4},
    "AggregationAgent": {"CENTRAL": 2.8, "PERIPHERAL": 1.0},
}

# Token cost estimates per agent invocation (includes mandatory EvidenceEvaluator call)
TOKEN_COSTS = {
    "EntityLinkerAgent": 350,     # LLM for extraction + evaluator
    "GraphTraversalAgent": 150,   # GSQL only + evaluator
    "SimilaritySearchAgent": 250, # embedding + evaluator
    "DocumentRetrievalAgent": 400,# doc fetch + evaluator
    "AggregationAgent": 500,      # aggregation LLM + evaluator
}


class Candidate:
    def __init__(self, agent_type: str, slot: Slot,
                 predicted_gain: float, predicted_cost: int):
        self.agent_type = agent_type
        self.slot = slot
        self.predicted_gain = predicted_gain
        self.predicted_cost = predicted_cost
        self.score = predicted_gain / max(predicted_cost, 1)

    def __repr__(self):
        return (f"Candidate({self.agent_type}, slot={self.slot.slot_id}, "
                f"gain={self.predicted_gain:.2f}, cost={self.predicted_cost}, "
                f"score={self.score:.4f})")


def generate_candidates(ledger: Ledger, step_history: list[dict],
                         thresholds: dict) -> list[Candidate]:
    """
    Generate all valid action candidates per spec §6.4.

    Returns empty list if no valid candidates (triggers no_candidate_actions stop).
    """
    candidates = []
    open_slots = ledger.open_slots()

    for slot in open_slots:
        # Check if any entity is linked for this slot (in any ledger claim)
        has_linked_entity = _slot_has_linked_entity(slot, ledger)
        has_doc_id = _slot_has_candidate_doc(slot, ledger)

        # EntityLinkerAgent: always valid for slots with unlinked entity references
        if not has_linked_entity:
            candidates.append(_make_candidate(
                "EntityLinkerAgent", slot, step_history, thresholds
            ))

        # GraphTraversalAgent: valid if entity linked
        if has_linked_entity:
            candidates.append(_make_candidate(
                "GraphTraversalAgent", slot, step_history, thresholds
            ))

        # SimilaritySearchAgent: always valid
        candidates.append(_make_candidate(
            "SimilaritySearchAgent", slot, step_history, thresholds
        ))

        # DocumentRetrievalAgent: valid if doc_id known
        if has_doc_id:
            candidates.append(_make_candidate(
                "DocumentRetrievalAgent", slot, step_history, thresholds
            ))

    # AggregationAgent: valid when requires_aggregation and ≥2 SUPPORTED/RESOLVED slots
    aggregation_slots = [s for s in ledger.slots.values()
                         if s.requires_aggregation
                         and s.state in (SlotState.SUPPORTED, SlotState.RESOLVED)]
    if len(aggregation_slots) >= 2:
        # Use the highest-criticality slot as the proxy slot for scoring
        proxy_slot = max(aggregation_slots, key=lambda s: s.weight())
        candidates.append(_make_candidate(
            "AggregationAgent", proxy_slot, step_history, thresholds
        ))

    return candidates


def select_action(candidates: list[Candidate], thresholds: dict) -> Optional[Candidate]:
    """
    Select the best candidate per spec §6.5.
    Returns None if best score < θ_voi (triggers threshold_not_cleared stop).
    """
    if not candidates:
        return None

    θ_voi = thresholds.get("θ_voi", 0.1)

    # Sort by score descending, apply tie-breaking per §6.5
    candidates = sorted(candidates, key=_sort_key, reverse=True)
    best = candidates[0]

    if best.score < θ_voi:
        logger.info(f"Best candidate score {best.score:.4f} < θ_voi {θ_voi}. threshold_not_cleared.")
        return None

    return best


def log_calibration(question_id: str, step_id: int, candidate: Candidate,
                     actual_gain: float, actual_cost: int, latency_ms: int) -> None:
    """
    Per spec §6.7 — log predicted vs actual gain/cost after each action.
    Per §6.8 — always include on_policy_only header.
    """
    log_path = LOG_DIR / "calibration_log.jsonl"

    # Write header on first call
    try:
        if log_path.stat().st_size == 0:
            _append_log(log_path, {
                "calibration_type": "on_policy_only",
                "note": "actual gain observed only for selected actions; no counterfactual claim",
            })
    except FileNotFoundError:
        _append_log(log_path, {
            "calibration_type": "on_policy_only",
            "note": "actual gain observed only for selected actions; no counterfactual claim",
        })

    record = {
        "question_id": question_id,
        "step_id": step_id,
        "agent_name": candidate.agent_type,
        "predicted_gain": candidate.predicted_gain,
        "actual_gain": actual_gain,
        "predicted_cost": candidate.predicted_cost,
        "actual_cost": actual_cost,
        "latency_ms": latency_ms,
        "monetary_cost_usd": None,  # not tracked
    }
    _append_log(log_path, record)


def compute_actual_gain(slot_states_before: dict[str, int],
                         slot_states_after: dict[str, int],
                         ledger: Ledger) -> float:
    """
    Compute actual gain from slot ordinal transitions per spec §6.1.
    gain = Σ weight(slot) × max(0, new_ordinal − old_ordinal)
    """
    total_gain = 0.0
    for slot_id, old_ordinal in slot_states_before.items():
        new_ordinal = slot_states_after.get(slot_id, old_ordinal)
        delta = max(0, new_ordinal - old_ordinal)
        slot = ledger.get_slot(slot_id)
        if slot:
            total_gain += slot.weight() * delta
    return total_gain


# ─── Private helpers ─────────────────────────────────────────────────────────

def _make_candidate(agent_type: str, slot: Slot,
                     step_history: list[dict], thresholds: dict) -> Candidate:
    """Create a Candidate with predicted gain/cost."""
    criticality_key = slot.criticality.value

    # Get historical average gain for this agent+criticality if available
    relevant_history = [
        h for h in step_history
        if h.get("agent_name") == agent_type
        and h.get("slot_criticality") == criticality_key
        and h.get("actual_gain") is not None
    ]

    if relevant_history:
        predicted_gain = sum(h["actual_gain"] for h in relevant_history) / len(relevant_history)
    else:
        predicted_gain = PRIOR_GAINS.get(agent_type, {}).get(criticality_key, 1.0)

    # Apply penalty if this agent type already attempted this slot
    if slot.has_attempted(agent_type):
        predicted_gain *= 0.5  # diminishing returns

    # Scale by slot weight
    predicted_gain *= slot.weight()

    predicted_cost = TOKEN_COSTS.get(agent_type, 300)

    return Candidate(agent_type, slot, predicted_gain, predicted_cost)


def _slot_has_linked_entity(slot: Slot, ledger: Ledger) -> bool:
    """Check if any claim for this slot has a linked entity (entity_id)."""
    for claim_id in slot.claim_ids:
        claim = ledger.claims.get(claim_id)
        if claim and claim.subject:  # subject = entity_id or name
            return True
    return False


def _slot_has_candidate_doc(slot: Slot, ledger: Ledger) -> bool:
    """Check if a candidate doc_id is known for this slot."""
    for claim_id in slot.claim_ids:
        claim = ledger.claims.get(claim_id)
        if claim and any(p.startswith("Q") for p in claim.provenance_chain):
            return True
    return False


def _sort_key(c: Candidate) -> tuple:
    """
    Tie-breaking per spec §6.5:
    1. score (primary)
    2. CENTRAL > PERIPHERAL
    3. not-yet-attempted on this slot > already attempted
    4. lowest predicted cost
    5. alphabetical agent_name
    """
    return (
        round(c.score, 6),                                     # 1. score
        1 if c.slot.criticality == SlotCriticality.CENTRAL else 0,  # 2. criticality
        0 if c.slot.has_attempted(c.agent_type) else 1,        # 3. not-yet-attempted
        -c.predicted_cost,                                     # 4. lower cost
        [-ord(ch) for ch in c.agent_type],                     # 5. alphabetical (negated for asc)
    )


def _append_log(path: Path, record: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to write calibration log: {e}")
