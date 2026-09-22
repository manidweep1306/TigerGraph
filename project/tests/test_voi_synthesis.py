"""
Stages 5 & 7 Tests: VoI Scorer and Synthesis
Per spec §13.5 and §13.7.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from backend.core.ledger import Ledger, Slot, SlotState, SlotCriticality
from backend.core.voi_scorer import (
    generate_candidates, select_action, compute_actual_gain, Candidate
)

THRESHOLDS = {
    "bar_CENTRAL": 0.75,
    "bar_PERIPHERAL": 0.5,
    "θ_voi": 0.1,
    "MAX_STEPS": 8,
    "MAX_TOKENS_PER_INVESTIGATION": 15000,
}


def make_ledger_with_slots(slots_spec: list[dict]) -> Ledger:
    ledger = Ledger("test-q", THRESHOLDS)
    for spec in slots_spec:
        slot = Slot(
            slot_id=spec["slot_id"],
            description=spec.get("description", "Test slot"),
            criticality=SlotCriticality(spec.get("criticality", "CENTRAL")),
        )
        if "state" in spec:
            slot.state = SlotState(spec["state"])
        ledger.add_slot(slot)
    return ledger


class TestVoIGainFormula:
    """§13.5 — Gain formula tests."""

    def test_single_central_slot_gain(self):
        """Action moves 1 CENTRAL slot EMPTY→SUPPORTED: gain = 3 × 1 = 3"""
        ledger = make_ledger_with_slots([{"slot_id": "s1", "criticality": "CENTRAL"}])
        before = {"s1": SlotState.EMPTY.ordinal()}
        after = {"s1": SlotState.SUPPORTED.ordinal()}
        gain = compute_actual_gain(before, after, ledger)
        assert gain == 3.0

    def test_multiple_peripheral_slots_gain(self):
        """Action moves 3 PERIPHERAL slots EMPTY→SUPPORTED: gain = 1+1+1 = 3"""
        ledger = make_ledger_with_slots([
            {"slot_id": "s1", "criticality": "PERIPHERAL"},
            {"slot_id": "s2", "criticality": "PERIPHERAL"},
            {"slot_id": "s3", "criticality": "PERIPHERAL"},
        ])
        before = {"s1": 0, "s2": 0, "s3": 0}
        after = {"s1": 1, "s2": 1, "s3": 1}
        gain = compute_actual_gain(before, after, ledger)
        assert gain == 3.0

    def test_noop_transition_zero_gain(self):
        """SUPPORTED→CONTESTED: ordinal stays 1→1, gain = 0."""
        ledger = make_ledger_with_slots([{"slot_id": "s1"}])
        before = {"s1": SlotState.SUPPORTED.ordinal()}  # 1
        after = {"s1": SlotState.CONTESTED.ordinal()}   # 1
        gain = compute_actual_gain(before, after, ledger)
        assert gain == 0.0


class TestVoICandidateGeneration:
    """§13.5 — Candidate generation."""

    def test_empty_candidates_on_no_open_slots(self):
        ledger = make_ledger_with_slots([
            {"slot_id": "s1", "state": "RESOLVED"},
        ])
        candidates = generate_candidates(ledger, [], THRESHOLDS)
        # No open slots → only AggregationAgent could be there, but requires_aggregation=False
        for c in candidates:
            assert c.agent_type == "AggregationAgent"  # shouldn't happen without req_agg

    def test_similarity_search_always_valid(self):
        ledger = make_ledger_with_slots([{"slot_id": "s1", "state": "EMPTY"}])
        candidates = generate_candidates(ledger, [], THRESHOLDS)
        agent_types = [c.agent_type for c in candidates]
        assert "SimilaritySearchAgent" in agent_types

    def test_entity_linker_when_no_entity_linked(self):
        ledger = make_ledger_with_slots([{"slot_id": "s1", "state": "EMPTY"}])
        candidates = generate_candidates(ledger, [], THRESHOLDS)
        agent_types = [c.agent_type for c in candidates]
        assert "EntityLinkerAgent" in agent_types


class TestVoITieBreaking:
    """§13.5 — CENTRAL-targeting candidate wins tie-breaking."""

    def test_central_preferred_over_peripheral_on_tie(self):
        ledger = make_ledger_with_slots([
            {"slot_id": "s_central", "criticality": "CENTRAL"},
            {"slot_id": "s_peripheral", "criticality": "PERIPHERAL"},
        ])
        s_central = ledger.slots["s_central"]
        s_peripheral = ledger.slots["s_peripheral"]

        # Use scores above θ_voi=0.1 so neither is filtered out
        c_central = Candidate("SimilaritySearchAgent", s_central, 3.0, 1000)
        c_peripheral = Candidate("SimilaritySearchAgent", s_peripheral, 3.0, 1000)
        # Set identical scores (both well above θ_voi)
        c_central.score = 0.5
        c_peripheral.score = 0.5

        selected = select_action([c_central, c_peripheral], THRESHOLDS)
        assert selected is not None
        assert selected.slot.criticality == SlotCriticality.CENTRAL

    def test_no_candidate_when_below_theta(self):
        ledger = make_ledger_with_slots([{"slot_id": "s1"}])
        # Create candidates with score below θ_voi
        s = ledger.slots["s1"]
        c = Candidate("SimilaritySearchAgent", s, 0.0001, 10000)
        c.score = 0.0000001  # below θ_voi=0.1

        selected = select_action([c], THRESHOLDS)
        assert selected is None


class TestSynthesisExitRule:
    """§13.7 — Synthesis exit rule is deterministic."""

    def test_all_central_resolved_gives_answer(self):
        from backend.core.synthesis import synthesis_exit
        ledger = make_ledger_with_slots([
            {"slot_id": "s1", "state": "RESOLVED"},
        ])
        assert synthesis_exit(ledger) == "ANSWER"

    def test_one_central_unresolvable_gives_abstain(self):
        from backend.core.synthesis import synthesis_exit
        ledger = make_ledger_with_slots([
            {"slot_id": "s1", "state": "RESOLVED"},
            {"slot_id": "s2", "state": "UNRESOLVABLE"},
        ])
        assert synthesis_exit(ledger) == "ABSTAIN"

    def test_mixed_supported_gives_partial(self):
        from backend.core.synthesis import synthesis_exit
        ledger = make_ledger_with_slots([
            {"slot_id": "s1", "state": "RESOLVED"},
            {"slot_id": "s2", "state": "SUPPORTED"},
        ])
        assert synthesis_exit(ledger) == "PARTIAL"

    def test_empty_slots_gives_partial(self):
        from backend.core.synthesis import synthesis_exit
        ledger = make_ledger_with_slots([{"slot_id": "s1", "state": "EMPTY"}])
        assert synthesis_exit(ledger) == "PARTIAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
