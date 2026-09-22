"""
graph.py — LangGraph Agentic Pipeline Orchestrator
Implements the full agent loop per spec §2.3 and §14 Stage 4/5/7.

Stage order (from spec §2.3):
Decomposer → [Agent Loop: VoI scorer → Dispatcher → Named Agent
              → EvidenceEvaluator → Ledger write → Coverage Check]
             → Synthesis → Claim Validation → Completeness Gate → Output
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict, Optional, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from backend.core.ledger import Ledger, Slot, SlotState
from backend.core.decomposer import decompose_question
from backend.core.voi_scorer import (
    generate_candidates, select_action, log_calibration,
    compute_actual_gain
)
from backend.core.dispatcher import invoke as dispatch_invoke, build_agent_input
from backend.core.synthesis import (
    synthesis_exit, compose_answer_text, claim_validation, completeness_gate
)
from backend.agents import evidence_evaluator

load_dotenv()
logger = logging.getLogger(__name__)
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Core
    question: str
    question_id: str

    # Investigation state
    ledger: Optional[Ledger]
    step_count: int
    tokens_used: int
    wall_time_start: float

    # Stopping
    stop_reason: Optional[str]
    decomposition_revisions_used: int
    consecutive_no_progress_steps: int  # for stall detection

    # Output
    exit_type: Optional[str]
    final_answer: str
    sources: list[str]

    # Tracing
    investigation_trace: list[dict]
    strategy_changed: bool
    agent_type_distribution: dict  # step→agent_type for strategy_changed calc

    # Config
    thresholds: dict
    agent_config: dict


# ─────────────────────────────────────────────────────────────────────────────
# Node: Decompose
# ─────────────────────────────────────────────────────────────────────────────

def node_decompose(state: AgentState) -> AgentState:
    """Initial decomposition or revision decomposition."""
    question = state["question"]
    question_id = state["question_id"]

    context = None
    if state.get("ledger") and state["decomposition_revisions_used"] > 0:
        # Revision mode — pass current ledger state as context per §5.5
        context = state["ledger"].snapshot()

    result = decompose_question(question, question_id, context)

    # Initialize or reset ledger with new slots
    ledger = Ledger(question_id, state["thresholds"])
    for slot in result.slots:
        ledger.add_slot(slot)

    # If this is a revision, port over existing claims that are still valid
    if state.get("ledger") and state["decomposition_revisions_used"] > 0:
        old_ledger = state["ledger"]
        for claim in old_ledger.claims.values():
            ledger.claims[claim.claim_id] = claim  # preserve history

        # Log decomposition revision per §11.5
        _log_decomposition_revision(
            question_id=question_id,
            trigger_type="revision",
            old_slots=state["ledger"].snapshot()["slots"],
            new_slots=[{"slot_id": s.slot_id, "description": s.description,
                        "criticality": s.criticality.value}
                       for s in result.slots],
            step_id=state["step_count"],
        )

    return {
        **state,
        "ledger": ledger,
        "investigation_trace": state.get("investigation_trace", []) + [{
            "step": "decompose",
            "slots": [{"id": s.slot_id, "desc": s.description,
                       "criticality": s.criticality.value}
                      for s in result.slots],
            "requires_aggregation": result.requires_aggregation,
        }],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Check Budget (runs at start of every loop iteration)
# ─────────────────────────────────────────────────────────────────────────────

def node_check_budget(state: AgentState) -> AgentState:
    """
    Per spec §7.2 — check all stop conditions in precedence order.
    Budget checks FIRST, before any other logic.
    """
    thresholds = state["thresholds"]
    elapsed = time.time() - state["wall_time_start"]

    stop_reason = None

    # Precedence per §7.2:
    if state["step_count"] >= thresholds.get("MAX_STEPS", 8):
        stop_reason = "budget_exhausted:steps"
    elif state["tokens_used"] >= thresholds.get("MAX_TOKENS_PER_INVESTIGATION", 15000):
        stop_reason = "budget_exhausted:tokens"
    elif elapsed >= thresholds.get("MAX_WALL_TIME_SECONDS", 45):
        stop_reason = "budget_exhausted:time"
    elif state["ledger"] and state["ledger"].all_central_resolved():
        stop_reason = "all_central_slots_resolved"

    if stop_reason:
        _log_stop_reason(state["question_id"], stop_reason, state["step_count"],
                         state["tokens_used"], elapsed)
        return {**state, "stop_reason": stop_reason}

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node: VoI Selection
# ─────────────────────────────────────────────────────────────────────────────

def node_voi_select(state: AgentState) -> AgentState:
    """Generate candidates and select best action per VoI score."""
    ledger = state["ledger"]
    if not ledger:
        return {**state, "stop_reason": "no_candidate_actions"}

    step_history = [t for t in state["investigation_trace"]
                    if t.get("step") == "action" and "actual_gain" in t]

    candidates = generate_candidates(ledger, step_history, state["thresholds"])

    if not candidates:
        elapsed = time.time() - state["wall_time_start"]
        _log_stop_reason(state["question_id"], "no_candidate_actions",
                         state["step_count"], state["tokens_used"], elapsed)
        return {**state, "stop_reason": "no_candidate_actions"}

    selected = select_action(candidates, state["thresholds"])

    if selected is None:
        elapsed = time.time() - state["wall_time_start"]
        _log_stop_reason(state["question_id"], "threshold_not_cleared",
                         state["step_count"], state["tokens_used"], elapsed)
        return {**state, "stop_reason": "threshold_not_cleared"}

    # Record slot ordinals BEFORE action for gain calculation
    slot_ordinals_before = {
        slot_id: slot.state.ordinal()
        for slot_id, slot in ledger.slots.items()
    }

    return {
        **state,
        "_selected_action": selected.agent_type,
        "_selected_slot_id": selected.slot.slot_id,
        "_predicted_gain": selected.predicted_gain,
        "_predicted_cost": selected.predicted_cost,
        "_slot_ordinals_before": slot_ordinals_before,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Execute Action
# ─────────────────────────────────────────────────────────────────────────────

def node_execute_action(state: AgentState) -> AgentState:
    """Dispatch to named agent, run EvidenceEvaluator, write to Ledger."""
    action_type = state.get("_selected_action")
    slot_id = state.get("_selected_slot_id")
    step_id = state["step_count"] + 1

    if not action_type or not slot_id:
        return state

    ledger = state["ledger"]
    slot = ledger.get_slot(slot_id)
    if not slot:
        return state

    t_action_start = time.time()

    # Build agent input
    input_data = build_agent_input(
        action_type, slot, ledger,
        state["question"], state["agent_config"]
    )

    # --- Step 1: Dispatch to named agent ---
    raw_result = dispatch_invoke(action_type, input_data, state["question_id"], step_id)
    slot.record_action_attempt(action_type)

    # --- Step 2: EvidenceEvaluatorAgent (MANDATORY after EVERY agent) ---
    existing_claims_for_slot = [
        c.to_dict() for c in ledger.get_claims_for_slot(slot_id)
    ]
    eval_input = {
        "raw_agent_output": raw_result,
        "target_slot_id": slot_id,
        "slot_description": slot.description,
        "original_question": state["question"],
    }
    evaluated_claim = evidence_evaluator.run(
        eval_input, state["question_id"], step_id, slot_id,
        existing_claims=existing_claims_for_slot
    )

    # --- Step 3: Ledger write ---
    old_state = slot.state
    claim_id, new_slot_state = ledger.write(evaluated_claim)

    # --- Compute actual gain for calibration ---
    slot_ordinals_after = {
        sid: s.state.ordinal() for sid, s in ledger.slots.items()
    }
    actual_gain = compute_actual_gain(
        state.get("_slot_ordinals_before", {}),
        slot_ordinals_after, ledger
    )
    actual_cost = raw_result.get("tokens_used", 0) + evaluated_claim.get("tokens_used", 0)

    # Log calibration per §6.7
    from backend.core.voi_scorer import Candidate
    proxy_candidate = Candidate(action_type, slot,
                                 state.get("_predicted_gain", 0),
                                 state.get("_predicted_cost", 0))
    log_calibration(
        state["question_id"], step_id, proxy_candidate,
        actual_gain, actual_cost, int((time.time() - t_action_start) * 1000)
    )

    # --- Stall detection: did we make progress? ---
    made_progress = actual_gain > 0
    consecutive_no_progress = state["consecutive_no_progress_steps"]
    if not made_progress:
        consecutive_no_progress += 1
    else:
        consecutive_no_progress = 0

    # Update agent type distribution (for strategy_changed calc)
    agent_dist = state.get("agent_type_distribution", {})
    agent_dist[str(step_id)] = action_type

    # Record trace entry
    trace_entry = {
        "step": "action",
        "step_id": step_id,
        "agent_type": action_type,
        "slot_id": slot_id,
        "slot_state_before": old_state.value,
        "slot_state_after": new_slot_state.value,
        "actual_gain": actual_gain,
        "actual_cost": actual_cost,
        "tokens_used": actual_cost,
        "claim_id": claim_id,
        "entailment": evaluated_claim.get("entailment_flag"),
        "confidence": evaluated_claim.get("confidence", 0),
    }

    return {
        **state,
        "step_count": step_id,
        "tokens_used": state["tokens_used"] + actual_cost,
        "consecutive_no_progress_steps": consecutive_no_progress,
        "agent_type_distribution": agent_dist,
        "investigation_trace": state["investigation_trace"] + [trace_entry],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Coverage Check (checks stall + decomp revision)
# ─────────────────────────────────────────────────────────────────────────────

def node_coverage_check(state: AgentState) -> AgentState:
    """
    Check for stall (K=3 consecutive no-progress steps) and trigger
    decomposition revision if budget allows.
    Per spec §5.
    """
    K = state["thresholds"].get("stall_trigger_k", 3)
    revision_limit = state["thresholds"].get("decomposition_revision_limit", 1)

    if (state["consecutive_no_progress_steps"] >= K
            and state["decomposition_revisions_used"] < revision_limit):
        logger.info(f"Stall detected at step {state['step_count']}. Triggering revision.")
        return {
            **state,
            "decomposition_revisions_used": state["decomposition_revisions_used"] + 1,
            "consecutive_no_progress_steps": 0,
            "_trigger_revision": True,
        }

    # Mark CONTESTED slots with no remaining actions as UNRESOLVABLE
    ledger = state["ledger"]
    if ledger:
        for slot in ledger.slots.values():
            if slot.state == SlotState.CONTESTED:
                # If budget is running low and no candidate can help, mark unresolvable
                remaining_budget = (
                    state["thresholds"].get("MAX_STEPS", 8) - state["step_count"]
                )
                if remaining_budget <= 1:
                    ledger.mark_unresolvable(slot.slot_id)

    return {**state, "_trigger_revision": False}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Synthesize
# ─────────────────────────────────────────────────────────────────────────────

def node_synthesize(state: AgentState) -> AgentState:
    """
    Run Synthesis → Claim Validation → Completeness Gate.
    Per spec §8 — exact sequence, no shortcuts.
    """
    ledger = state["ledger"]
    question = state["question"]
    question_id = state["question_id"]

    # Step 1: Synthesis exit rule (pure function, no LLM)
    exit_type = synthesis_exit(ledger)

    # Step 2: Compose answer text (LLM, constrained by exit_type)
    drafted_answer = compose_answer_text(exit_type, ledger, question, question_id)

    # Step 3: Claim Validation
    validated_answer, exit_type = claim_validation(
        drafted_answer, ledger, exit_type, question_id,
        step_id=state["step_count"]
    )

    # Step 4: Completeness Gate
    final_answer, exit_type = completeness_gate(
        validated_answer, question, exit_type, question_id
    )

    # Compute strategy_changed per spec §11.8
    agent_dist = state.get("agent_type_distribution", {})
    steps = sorted(agent_dist.keys(), key=lambda x: int(x))
    strategy_changed = (
        state["decomposition_revisions_used"] > 0
        or len(state.get("rung_path", ["Agentic"])) > 1
        or _agent_distribution_changed(agent_dist, steps)
    )

    # Collect all sources
    sources = []
    if ledger:
        for claim in ledger.claims.values():
            sources.extend(claim.provenance_chain)
    sources = list(dict.fromkeys(sources))[:20]  # deduplicate, cap at 20

    return {
        **state,
        "exit_type": exit_type,
        "final_answer": final_answer,
        "sources": sources,
        "strategy_changed": strategy_changed,
        "investigation_trace": state["investigation_trace"] + [{
            "step": "synthesize",
            "exit_type": exit_type,
            "answer_length": len(final_answer),
            "coverage_fraction": ledger.coverage_fraction() if ledger else 0,
        }],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph Routing Functions
# ─────────────────────────────────────────────────────────────────────────────

def route_after_budget(state: AgentState) -> str:
    """After budget check: continue loop or go to synthesis."""
    if state.get("stop_reason"):
        return "synthesize"
    return "voi_select"


def route_after_voi(state: AgentState) -> str:
    """After VoI selection: execute action or go to synthesis."""
    if state.get("stop_reason"):
        return "synthesize"
    return "execute_action"


def route_after_coverage(state: AgentState) -> str:
    """After coverage check: loop or trigger revision or synthesize."""
    if state.get("_trigger_revision"):
        return "decompose"  # triggers revision
    return "check_budget"  # next loop iteration


# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_agentic_graph() -> StateGraph:
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("decompose", node_decompose)
    graph.add_node("check_budget", node_check_budget)
    graph.add_node("voi_select", node_voi_select)
    graph.add_node("execute_action", node_execute_action)
    graph.add_node("coverage_check", node_coverage_check)
    graph.add_node("synthesize", node_synthesize)

    # Add edges
    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "check_budget")
    graph.add_conditional_edges("check_budget", route_after_budget, {
        "synthesize": "synthesize",
        "voi_select": "voi_select",
    })
    graph.add_conditional_edges("voi_select", route_after_voi, {
        "synthesize": "synthesize",
        "execute_action": "execute_action",
    })
    graph.add_edge("execute_action", "coverage_check")
    graph.add_conditional_edges("coverage_check", route_after_coverage, {
        "decompose": "decompose",
        "check_budget": "check_budget",
    })
    graph.add_edge("synthesize", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_agentic_pipeline(question: str, question_id: str,
                          thresholds: dict, agent_config: dict) -> dict:
    """
    Run the full Agentic pipeline for one question.
    Per spec §2.3 — returns complete output with ledger, trace, and tokens.
    """
    compiled_graph = build_agentic_graph()
    t_start = time.time()

    initial_state: AgentState = {
        "question": question,
        "question_id": question_id,
        "ledger": None,
        "step_count": 0,
        "tokens_used": 0,
        "wall_time_start": t_start,
        "stop_reason": None,
        "decomposition_revisions_used": 0,
        "consecutive_no_progress_steps": 0,
        "exit_type": None,
        "final_answer": "",
        "sources": [],
        "investigation_trace": [],
        "strategy_changed": False,
        "agent_type_distribution": {},
        "thresholds": thresholds,
        "agent_config": agent_config,
        "rung_path": ["Agentic"],
    }

    final_state = compiled_graph.invoke(initial_state)

    latency_ms = int((time.time() - t_start) * 1000)

    # Count chunks and citations
    n_chunks = sum(
        1 for t in final_state.get("investigation_trace", [])
        if t.get("step") == "action" and t.get("agent_type") == "SimilaritySearchAgent"
    )
    n_sources = len(final_state.get("sources", []))

    return {
        "answer": final_state["final_answer"],
        "exit_type": final_state.get("exit_type", "ABSTAIN"),
        "sources": final_state.get("sources", []),
        "tokens_used": final_state["tokens_used"],
        "latency_ms": latency_ms,
        "ledger": final_state.get("ledger"),
        "trace": final_state.get("investigation_trace", []),
        "stop_reason": final_state.get("stop_reason", "unknown"),
        "strategy_changed": final_state.get("strategy_changed", False),
        "n_chunks_retrieved": n_chunks,
        "n_sources_cited": n_sources,
        "step_count": final_state["step_count"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_stop_reason(question_id: str, stop_reason: str, step_count: int,
                      tokens_used: int, elapsed: float) -> None:
    """Per spec §11.2."""
    record = {
        "question_id": question_id,
        "stop_reason": stop_reason,
        "step_count": step_count,
        "tokens_used_total": tokens_used,
        "wall_time_seconds": round(elapsed, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_path = LOG_DIR / "stop_reason_log.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to log stop reason: {e}")


def _log_decomposition_revision(question_id: str, trigger_type: str,
                                  old_slots: list, new_slots: list,
                                  step_id: int,
                                  triggering_claim_id: Optional[str] = None) -> None:
    """Per spec §11.5."""
    record = {
        "question_id": question_id,
        "trigger_type": trigger_type,
        "old_slots": old_slots,
        "new_slots": new_slots,
        "triggering_claim_id": triggering_claim_id,
        "step_id_at_revision": step_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_path = LOG_DIR / "decomposition_revisions.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to log decomposition revision: {e}")


def _agent_distribution_changed(agent_dist: dict, steps: list) -> bool:
    """
    Per spec §11.8 strategy_changed: agent-type distribution in steps 4+
    differs from steps 1-3 by majority type.
    """
    if len(steps) < 4:
        return False
    early = [agent_dist[s] for s in steps[:3]]
    late = [agent_dist[s] for s in steps[3:]]
    early_majority = max(set(early), key=early.count) if early else None
    late_majority = max(set(late), key=late.count) if late else None
    return early_majority != late_majority
