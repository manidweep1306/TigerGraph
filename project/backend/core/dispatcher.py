"""
Dispatcher — routes VoI-selected action_type to the correct named agent.
Per spec §3.4: a thin routing function, not itself an agent.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Agent registry — maps action type string to module
_AGENT_REGISTRY = {
    "EntityLinkerAgent": "backend.agents.entity_linker",
    "GraphTraversalAgent": "backend.agents.graph_traversal",
    "SimilaritySearchAgent": "backend.agents.similarity_search",
    "DocumentRetrievalAgent": "backend.agents.document_retrieval",
    "AggregationAgent": "backend.agents.aggregation",
}


def invoke(action_type: str, input_data: dict,
           question_id: str, step_id: int) -> dict:
    """
    Route action_type to the matching named agent and invoke it.
    Per spec §3: returns raw_result (before EvidenceEvaluatorAgent runs).

    EvidenceEvaluatorAgent is NOT invoked here — it must be called by the
    agent loop immediately after this function returns.
    """
    module_path = _AGENT_REGISTRY.get(action_type)
    if not module_path:
        raise ValueError(f"Unknown action type: {action_type}. "
                         f"Valid types: {list(_AGENT_REGISTRY.keys())}")

    import importlib
    module = importlib.import_module(module_path)

    if not hasattr(module, "run"):
        raise AttributeError(f"Agent module {module_path} has no 'run' function")

    logger.debug(f"Dispatching {action_type} for question={question_id}, step={step_id}")
    return module.run(input_data, question_id, step_id)


def build_agent_input(action_type: str, slot, ledger,
                       question: str, agent_config: dict) -> dict:
    """
    Build the input dict for a specific agent type.
    Extracts relevant context from the ledger and slot.
    """
    cfg = agent_config.get(action_type, {})

    if action_type == "EntityLinkerAgent":
        return {
            "query_string": slot.assumed_entity or slot.description or question,
            "graph": "tg_ref",  # actual client accessed inside agent
        }

    elif action_type == "GraphTraversalAgent":
        # Find the best linked entity from existing claims
        entity_id = _find_linked_entity(slot, ledger)
        return {
            "start_entity_id": entity_id or "",
            "max_hops": cfg.get("max_hops", 2),
            "graph": "tg_ref",
        }

    elif action_type == "SimilaritySearchAgent":
        # Use slot description or full question as query
        query = slot.description if slot.description else question
        return {
            "query_string": query,
            "top_k": cfg.get("top_k", 5),
            "corpus_index": "vector_ref",
        }

    elif action_type == "DocumentRetrievalAgent":
        doc_id = _find_candidate_doc(slot, ledger)
        return {
            "doc_id": doc_id or "",
        }

    elif action_type == "AggregationAgent":
        # Gather claim texts from SUPPORTED/RESOLVED slots
        from backend.core.ledger import SlotState
        source_claim_ids = []
        source_claims_text = []
        for s in ledger.slots.values():
            if s.state in (SlotState.SUPPORTED, SlotState.RESOLVED):
                for claim_id in s.claim_ids:
                    claim = ledger.claims.get(claim_id)
                    if claim and claim.entailment_flag == "PASS":
                        source_claim_ids.append(claim_id)
                        source_claims_text.append(claim.source_passage)
        return {
            "source_claims": source_claim_ids,
            "source_claims_text": source_claims_text,
            "aggregation_type": "count" if "how many" in question.lower() else "compare",
        }

    return {"query_string": question}


def _find_linked_entity(slot, ledger) -> Optional[str]:
    """Extract best entity_id from slot's existing claims."""
    for claim_id in slot.claim_ids:
        claim = ledger.claims.get(claim_id)
        if claim and claim.subject:
            return claim.subject
    # Also check all claims for entity mentions
    for claim in ledger.claims.values():
        if claim.subject and claim.subject.startswith("Q"):
            return claim.subject
    return None


def _find_candidate_doc(slot, ledger) -> Optional[str]:
    """Find a candidate doc_id from slot's claims' provenance."""
    for claim_id in slot.claim_ids:
        claim = ledger.claims.get(claim_id)
        if claim:
            for p in claim.provenance_chain:
                if p.startswith("Q"):
                    return p
    return None
