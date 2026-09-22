# Adaptive Investigation Engine — Implementation Specification

**Status:** Architecture frozen. This document converts the frozen architecture into implementation-ready specification. It defines no new components, agents, pipelines, feedback loops, recovery mechanisms, or logs beyond what was already designed. Where the architecture leaves a detail open, this document closes it with an explicit, deterministic rule — it does not redesign.

**Audience:** any developer implementing this system without further access to the design discussion.

---

## Table of Contents

1. Execution Modes
2. Pipeline Contracts
3. Specialised-Agent Contracts
4. Ledger and Slot State Machine
5. Decomposition Revision
6. VoI (Value-of-Information) Implementation
7. Stopping Behavior
8. Synthesis and Final Gates
9. Benchmark and Evaluation Protocol
10. Configuration and Threshold Freezing
11. Logging and Artifact Schemas
12. Baseline Fairness
13. Testing
14. Project Implementation Order
15. Implementation Invariants

---

# 1. Execution Modes

Two modes exist. They are mutually exclusive per run and are set by a single top-level parameter, `run_mode`. No component below the entrypoint is aware of which mode invoked it except where explicitly stated.

## 1.1 Definitions

| Term | Definition |
|---|---|
| `BENCHMARK` mode | For a given question, all three pipelines (RAG, GraphRAG, Agentic) execute **independently and exclusively of each other**. No routing occurs. No pipeline's output, state, or Ledger is visible to another pipeline's execution of the same question. |
| `ADAPTIVE` mode | For a given question, exactly **one** pipeline executes, selected by the Ladder. If the confidence gate fires, exactly one escalation to the next rung occurs, sequentially, after the first rung's result is scored. No two rungs ever execute concurrently for the same question in this mode. |

## 1.2 Which questions use which mode

| Question set | Mode | Reason |
|---|---|---|
| 100 provided dev/eval questions | `BENCHMARK` | Problem statement requires "results" from all three pipelines on these questions in the metrics dashboard. |
| 50 hidden evaluation questions | `ADAPTIVE` | Problem statement requires "your system" output with "agentic trace" — this is the routed system, not a forced three-way run. |

No other question set exists. Do not run `BENCHMARK` mode on the 50 hidden questions. Do not run `ADAPTIVE` mode as the sole basis for the three-way comparison table.

## 1.3 Execution semantics per mode

### `BENCHMARK` mode

```
for question in dev_set:
    result_rag        = run_pipeline(RAG,       question, isolated_context)
    result_graphrag    = run_pipeline(GraphRAG,  question, isolated_context)
    result_agentic     = run_pipeline(Agentic,   question, isolated_context)
    # all three run to completion; order does not matter; no shared state
    write_benchmark_record(question, result_rag, result_graphrag, result_agentic)
```

- All three executions use a **fresh, isolated Ledger instance** per pipeline per question. A `BENCHMARK`-mode Agentic run's Ledger is discarded after that question and is never reused by RAG or GraphRAG on the same question, nor by the Agentic run on the next question.
- The three pipelines are **not** run concurrently for cost/rate-limit reasons in a hackathon environment, but "sequential execution" here does not mean "conditional execution" — all three run regardless of any other pipeline's result. This is the defining difference from `ADAPTIVE` mode.
- Cross-pipeline shared state permitted: **read-only access to the same corpus, same graph, same embedding index.** Nothing else.
- Cross-pipeline shared state forbidden: Ledger contents, slot states, VoI calibration history, agent invocation history, escalation decisions.

### `ADAPTIVE` mode

```
rung = Ladder.route(question)                     # one rung selected
result = run_pipeline(rung, question)
if rung in {RAG, GraphRAG} and confidence_gate(result) == LOW:
    escalated_rung = next_rung(rung)               # exactly one escalation
    result = run_pipeline(escalated_rung, question)
    log_escalation(question, from=rung, to=escalated_rung)
write_adaptive_record(question, result, rung_path)
```

- Only one pipeline executes per question, unless escalation fires, in which case exactly two execute **sequentially** (never both, then compare — the second only runs because the first was judged insufficient).
- If the Agentic rung is selected (directly or via escalation), it does not itself escalate further — escalation only occurs between RAG→GraphRAG or GraphRAG→Agentic. Agentic is terminal in the escalation chain.

## 1.4 Required outputs per mode

| Mode | Required output artifact |
|---|---|
| `BENCHMARK` | `benchmark_results.jsonl` — one record per dev question, containing all three pipelines' answers, tokens, latency, and accuracy scores (see §11.7) |
| `BENCHMARK` | Ladder-accuracy confusion matrix (see §9.6), derived from `benchmark_results.jsonl` by labeling each question with the cheapest rung that produced a correct answer, compared against what the Ladder would have selected |
| `ADAPTIVE` | `adaptive_results.jsonl` — one record per hidden question, containing the single system answer, full agentic trace (agents invoked, tokens, chunks, citations, strategy_changed, stop_reason), per §11 |

No other output is required by either mode. Do not generate a three-way comparison for the 50 hidden questions — ground truth for those questions is not available to you and is not the deliverable being asked for there.

---

# 2. Pipeline Contracts

Each pipeline is a function with a fixed input/output contract. A pipeline never calls another pipeline. A pipeline never inspects another pipeline's Ledger.

## 2.1 RAG Pipeline

| Property | Definition |
|---|---|
| Input | `{question: str, corpus_index: VectorIndex}` |
| Output | `{answer: str, sources: [chunk_id], tokens_used: int, latency_ms: int}` |
| Required fields on output | all four above; `sources` may be empty list but must exist |
| State changes | none persisted; entirely stateless per call |
| Failure behavior | if retrieval returns zero chunks, `answer` = literal string `"NO_EVIDENCE_RETRIEVED"`, `sources` = `[]`; this is not an exception, it is a valid output |
| Logging | one record per question written to `benchmark_results.jsonl` (in `BENCHMARK` mode) or `adaptive_results.jsonl` (in `ADAPTIVE` mode) |
| Starts | receiving a question string |
| Ends | returning the output object above; does not participate in slot logic, Ledger, or agents |

## 2.2 GraphRAG Pipeline

| Property | Definition |
|---|---|
| Input | `{question: str, graph: TigerGraphInstance}` |
| Output | `{answer: str, sources: [entity_id \| chunk_id], tokens_used: int, latency_ms: int}` |
| Required fields on output | same as RAG |
| State changes | none persisted; entirely stateless per call. Read-only graph traversal. |
| Failure behavior | if entity linking fails (no matching graph node for question entities), `answer` = `"NO_ENTITY_MATCH"`, `sources` = `[]` |
| Logging | same as RAG |
| Starts | receiving a question string |
| Ends | returning the output object; does not use the Ledger, slot machine, or specialised agents — GraphRAG here is single-shot, not agentic |

## 2.3 Agentic Pipeline

| Property | Definition |
|---|---|
| Input | `{question: str, graph: TigerGraphInstance, corpus_index: VectorIndex, config: FrozenThresholds}` |
| Output | `{answer: str, exit_type: ANSWER\|PARTIAL\|ABSTAIN, sources: [claim_id], tokens_used: int, latency_ms: int, ledger: LedgerInstance, trace: InvestigationTrace}` |
| Required fields | all of the above; `ledger` and `trace` are required for logging even though they are not shown to the end user |
| State changes | creates and owns exactly one fresh Ledger instance for the duration of this question; discarded after the question completes (in `BENCHMARK` mode) or persisted only for logging (in `ADAPTIVE` mode) |
| Failure behavior | never raises on "no evidence found" — this is represented as slot state `UNRESOLVABLE` and flows through to `ABSTAIN`, per §8 |
| Logging | writes to all five log files listed in §11 |
| Starts | receiving a question string |
| Ends | after Synthesis → Claim Validation → Completeness Gate have all run, per §8; this pipeline's boundary explicitly **includes** those three stages — they are not separate top-level components invoked by something else |

**Internal stage boundary within the Agentic Pipeline** (each defined fully in its own section below):

```
Decomposer  →  [Agent Loop: VoI scorer → Dispatcher → Named Agent → Evidence
                Evaluator → Ledger write → Coverage Check]  →  Synthesis  →
                Claim Validation  →  Completeness Gate  →  Output
```

No stage in this chain is skipped, reordered, or merged. Each is specified independently below.

---

# 3. Specialised-Agent Contracts

**Governing rule:** the VoI scorer selects an **action type**. The **Dispatcher** (a thin routing function, not itself an agent) maps the selected action type to exactly one named agent and invokes it. The named agent performs the action and returns raw results. The **Evidence Evaluator Agent** always runs next, on that agent's raw output, before anything reaches the Ledger. The **Ledger** then records — it does not evaluate.

```
VoI.select_action() → action_type
Dispatcher.invoke(action_type) → calls the one matching named agent
NamedAgent.execute(input) → raw_result
EvidenceEvaluatorAgent.evaluate(raw_result) → evaluated_claim
Ledger.write(evaluated_claim) → updates slot state, confidence, provenance
```

## 3.1 Agent Registry

| Agent | Responsibility | Invocable when |
|---|---|---|
| `EntityLinkerAgent` | Resolve a question or claim entity string to a graph node ID | any EMPTY or SUPPORTED slot whose description references an unlinked entity |
| `GraphTraversalAgent` | Traverse N hops from a linked entity to find related entities/edges | at least one entity already linked (via Ledger claims) |
| `SimilaritySearchAgent` | Vector search over the corpus index for a query string | any open slot, no precondition on prior linking |
| `DocumentRetrievalAgent` | Fetch full document/chunk content by ID | a candidate document ID is known (from similarity search or graph traversal result) |
| `AggregationAgent` | Combine ≥2 already-RESOLVED or SUPPORTED claims of matching shape into a derived claim (count / compare / rank) | `requires_aggregation == true` on the decomposition output AND ≥2 matching-shape slots are RESOLVED or SUPPORTED |
| `EvidenceEvaluatorAgent` | Run entailment check, confidence scoring, contradiction check on any other agent's raw output before it becomes a claim | always, immediately after every other agent's execution; never independently selected by VoI |

`EvidenceEvaluatorAgent` is not a VoI-selectable action. It is a mandatory post-processing step on every other agent's output. It has no "when invocable" condition beyond "always, after any other agent runs."

## 3.2 Input/Output Schemas

### EntityLinkerAgent
```json
Input:  {"query_string": "str", "graph": "TigerGraphInstance ref"}
Output: {"entity_id": "str | null", "match_confidence": "float 0-1", "candidates": ["str", ...]}
```

### GraphTraversalAgent
```json
Input:  {"start_entity_id": "str", "max_hops": "int", "graph": "TigerGraphInstance ref"}
Output: {"paths": [{"entities": ["str"], "edges": ["str"], "hop_count": "int"}], "tokens_used": "int"}
```

### SimilaritySearchAgent
```json
Input:  {"query_string": "str", "top_k": "int", "corpus_index": "VectorIndex ref"}
Output: {"chunks": [{"chunk_id": "str", "text": "str", "score": "float"}], "tokens_used": "int"}
```

### DocumentRetrievalAgent
```json
Input:  {"doc_id": "str"}
Output: {"content": "str", "doc_id": "str", "tokens_used": "int"}
```

### AggregationAgent
```json
Input:  {"source_claims": ["claim_id", ...], "aggregation_type": "count | compare | rank"}
Output: {"derived_claim_text": "str", "derived_from": ["claim_id", ...], "tokens_used": "int"}
```

### EvidenceEvaluatorAgent
```json
Input:  {"raw_agent_output": "object (any agent's output above)", "target_slot_id": "str"}
Output: {
  "claim_text": "str",
  "entailment_flag": "PASS | FAIL",
  "confidence": "float 0-1",
  "contradiction_detected": "bool",
  "contradicted_claim_id": "str | null"
}
```

## 3.3 Logging requirement (applies to all six agents)

Every invocation of every named agent (including `EvidenceEvaluatorAgent`) writes one record to `agent_invocations.jsonl`:

```json
{"agent_name": "str", "step_id": "int", "input_summary": "str", "tokens_used": "int", "latency_ms": "int", "output_summary": "str", "timestamp": "ISO8601"}
```

This is the field that satisfies the rubric's "specialised agents invoked" trace metric. No agent invocation is permitted to occur without this record being written — if an agent call fails to log, treat it as a failed invocation and do not use its output.

## 3.4 Explicit boundary statement

The following are **not** agents and must never be implemented as, wrapped as, or logged as agents, per the frozen architecture:

- **Ladder** — a routing function.
- **Ledger** — a passive recorder with one write function.
- **Synthesis** — a deterministic switch over slot states.
- **Completeness Gate** — a terminal boolean check.
- **Decomposer** — produces the slot set; it is a single call, not an iterative agent, and is not VoI-selectable.
- **VoI Scorer** — selects actions; it does not itself act on the world.
- **Dispatcher** — routes the VoI's selection to the correct agent; it performs no retrieval or evaluation itself.

---

# 4. Ledger and Slot State Machine

## 4.1 Slot states

| State | Meaning | Ordinal (for VoI gain, §6) |
|---|---|---|
| `EMPTY` | No claim yet targets this slot | 0 |
| `SUPPORTED` | ≥1 claim with `entailment_flag == PASS`, but has not met full RESOLVED criteria | 1 |
| `CONTESTED` | ≥2 claims targeting this slot disagree (see §4.4) | 1 |
| `RESOLVED` | Meets all four resolution criteria below (§4.3) | 2 |
| `UNRESOLVABLE` | Was `CONTESTED` and no basis exists to resolve it (terminal — see §4.5) | 2, gain=0 for all future transitions |

## 4.2 Allowed transitions

```
EMPTY ──(claim written, entailment PASS)──► SUPPORTED
EMPTY ──(claim written, entailment FAIL)──► EMPTY   (no-op; claim is discarded, not written)
SUPPORTED ──(contradicting claim written)──► CONTESTED
SUPPORTED ──(resolution criteria met)──► RESOLVED
CONTESTED ──(resolution criteria met after new evidence)──► RESOLVED
CONTESTED ──(no basis to resolve after all available actions exhausted)──► UNRESOLVABLE
RESOLVED ──► RESOLVED  (idempotent; new corroborating claims do not change state)
UNRESOLVABLE ──► UNRESOLVABLE  (terminal; no further transitions permitted)
```

**All other transitions are invalid.** In particular:
- `RESOLVED → CONTESTED` is **not permitted**. Once resolved, a slot does not regress. If a later claim contradicts a RESOLVED slot, that contradiction is logged as a `post_resolution_conflict` event on the claim record but does **not** change the slot's state. (This is a documented limitation, not a gap to fix — see architecture discussion on premature RESOLVED states.)
- `UNRESOLVABLE → SUPPORTED` or `UNRESOLVABLE → RESOLVED` is **not permitted**. `UNRESOLVABLE` is terminal for the remainder of that investigation.

**Invalid transition attempt behavior:** if code attempts a transition not listed above, the Ledger write function must reject the write, log an `invalid_transition_attempt` event (slot_id, attempted_from, attempted_to, timestamp) to `agent_invocations.jsonl` with `agent_name: "LedgerGuard"`, and leave the slot state unchanged. This must never raise an unhandled exception — it is a logged no-op.

## 4.3 RESOLVED criteria (all four required)

```
slot.state = RESOLVED  if and only if:
  1. ≥1 claim on this slot has entailment_flag == PASS
  2. claim.confidence ≥ bar(slot.criticality)
  3. no CONTESTED sibling claim on this slot remains unresolved
  4. IF slot.criticality == CENTRAL:
       (≥2 independent-source claims agree)
       OR
       (1 source claim with confidence ≥ 0.9 AND provenance_chain depth ≤ 1)
     IF slot.criticality == PERIPHERAL:
       condition 4 is not required; conditions 1-3 are sufficient
```

`bar(CENTRAL) = 0.75`, `bar(PERIPHERAL) = 0.5` — starting values, to be replaced with fit-split-derived values per §10 before freezing. These two constants live in `frozen_thresholds.json`, not hardcoded in application logic.

"Independent-source claims agree" means: two claims with the same `(subject, predicate)` and the same `object` (or objects within an agreed equivalence, e.g. same entity via different aliases), originating from `derived_from` chains that do not share a common upstream claim.

## 4.4 CONTESTED definition

A slot enters `CONTESTED` when two claims targeting the same slot have the same `(subject, predicate)` but **different** `object` values, and neither is yet confidently dominant. This is distinct from a premise contradiction (§5) — a CONTESTED slot disagrees on a *value*; a premise contradiction disagrees on an *assumption the slot's own description depends on*. CONTESTED never triggers decomposition revision by itself.

## 4.5 UNRESOLVABLE determination

A `CONTESTED` slot becomes `UNRESOLVABLE` only when **both** are true:
1. No candidate VoI action remains that could plausibly add a disambiguating claim (i.e., every action type applicable to this slot has already been attempted at least once for this slot in this investigation), **and**
2. The investigation is at a stop condition per §7 (budget exhausted, no action clears θ_voi, or all other CENTRAL slots resolved).

This state is set by the coverage-check step immediately before Synthesis is invoked — it is not set mid-loop while other actions remain viable.

## 4.6 CENTRAL vs PERIPHERAL

Assigned once, by the Decomposer, at slot-creation time. Not reassigned during the investigation (a slot's criticality does not change, even across a decomposition revision — the revision produces a new slot set with its own criticality assignments).

Rule for the Decomposer: a slot is `CENTRAL` if the question cannot be answered without it; `PERIPHERAL` if it adds detail, context, or robustness but the question is answerable without it. This determination is made by the Decomposer's LLM call, constrained to output exactly one of the two labels per slot — no third category, no omission.

## 4.7 Confidence inheritance (provenance propagation)

Enforced inside the Ledger's single write function — never computed elsewhere:

```
claim.confidence = min(
    claim.own_evidence_confidence,
    min(c.confidence for c in claim.derived_from)   # empty min = 1.0 if derived_from is []
)
claim.provenance_chain = flatten([c.provenance_chain for c in claim.derived_from]) + [claim.source_passage]
```

A claim with no `derived_from` (a direct, first-hand claim) has its own confidence unmodified. A claim aggregated or derived from others is capped at the weakest link in its ancestry, with no exceptions.

## 4.8 Contradiction representation

A contradiction is represented as a field on the *newer* claim record, never by deleting or mutating the earlier claim:

```json
{"contradiction_detected": true, "contradicted_claim_id": "<id of the earlier conflicting claim>"}
```

Both claims remain in the Ledger permanently. The slot's state (CONTESTED) is the aggregate signal; individual claim records are never deleted.

## 4.9 Ledger write function — full contract

| Property | Definition |
|---|---|
| Input | `EvaluatedClaim` object from `EvidenceEvaluatorAgent` output (§3.2) plus `target_slot_id` |
| Output | updated `slot_state`, written `claim_id` |
| Required fields on write | `subject, predicate, object, source_passage, confidence, provenance_chain, entailment_flag, derived_from, valid_from, valid_to, source_authority` — the last three default to `null` in Round 1 (§11.9) |
| State changes | exactly the slot-state transition per §4.2, and appends one immutable claim row |
| Failure behavior | if `entailment_flag == FAIL`, the claim is **not written** to the Ledger at all — no slot transition occurs, and this is logged as a `discarded_claim` event, not an error |
| Logging | every successful write appends to `ledger_writes.jsonl` (§11.3) |
| Starts | receiving an `EvaluatedClaim` | 
| Ends | returning the updated slot state; the Ledger never calls an agent, never calls VoI, never calls Synthesis |

---

# 5. Decomposition Revision

## 5.1 The three conditions, precisely

| Condition | Trigger definition | Action |
|---|---|---|
| **Normal continuation** | Neither of the two triggers below has fired | proceed to VoI action selection as normal |
| **Stall-triggered revision** | `K = 3` consecutive completed agent actions (post-Evidence-Evaluator) produce **zero** slot-state ordinal progress across all open slots | fire exactly one re-decomposition (if not already used) |
| **Premise-contradiction revision** | A written claim satisfies the whitelist in §5.2 against any existing slot's stored assumption | fire exactly one re-decomposition (if not already used) |

Both triggers draw from the **same single allowed revision**. Whichever fires first consumes it. If a revision has already occurred once in this investigation, neither trigger fires again — the investigation proceeds with the current (possibly still-imperfect) slot set to its normal stop conditions.

## 5.2 Premise-contradiction whitelist (exact, closed list — no other predicate qualifies)

A new claim counts as a premise contradiction against a slot if and only if it asserts one of:

1. `does_not_exist` for the entity the slot assumes exists (`slot.assumed_entity`)
2. the slot's assumed entity is of a different type than the slot description requires (e.g., slot assumes a Company, new claim asserts that ID is a Person)
3. the relationship type the slot's description presumes (e.g., `works_at`) never held between the entities in question

Disagreement about a *value* (e.g., two different founding dates for a confirmed-real, confirmed-correctly-typed company) is **not** a premise contradiction — it is CONTESTED (§4.4) and does not touch the Decomposer.

## 5.3 Revision bound (hard limit: exactly one)

```
investigation.decomposition_revisions_used: int, initialized to 0

on stall_trigger OR premise_contradiction_trigger:
    if investigation.decomposition_revisions_used >= 1:
        do_nothing()  # trigger condition is evaluated but revision is not performed
        continue_with_current_slots()
    else:
        investigation.decomposition_revisions_used += 1
        perform_revision()
```

## 5.4 What gets logged during a revision

One record to `decomposition_revisions.jsonl` (§11.5), containing: `question_id`, `trigger_type` (`stall` | `premise_contradiction`), `old_slots` (full snapshot), `new_slots` (full snapshot), `triggering_claim_id` (null if stall-triggered), `step_id_at_revision`.

## 5.5 Revision outcome handling

```
perform_revision():
    new_slots = Decomposer.decompose(question, context=current_ledger_state)
    # existing Ledger claims are NOT discarded — they remain and are re-evaluated
    # against the new slot set's coverage requirements on the next coverage check
    if new_slots produce measurable progress within the next 2 actions:
        continue investigation normally
    else:
        proceed directly to Synthesis (this will produce ABSTAIN or PARTIAL,
        per §8 — the investigation does not retry decomposition again)
```

## 5.6 Interaction with the Completeness Gate — explicit prohibition

**The Completeness Gate (§8.3) must never trigger a decomposition revision, regardless of outcome.** The one-revision budget is consumed only by the two triggers in §5.1, both of which occur strictly *before* Synthesis. Once Synthesis has run, the investigation's evidence-gathering phase is permanently closed. The Completeness Gate's only permitted actions are: pass the output unchanged, or downgrade `ANSWER → PARTIAL` / `PARTIAL → ABSTAIN`. It has no code path that re-enters the agent loop, calls the Decomposer, or calls any named agent.

---

# 6. VoI (Value-of-Information) Implementation

## 6.1 Gain formula

```
gain(action) = Σ_over_slots_touched_by_action [ weight(slot) × max(0, new_ordinal(slot) − old_ordinal(slot)) ]

weight(slot) = 3   if slot.criticality == CENTRAL
weight(slot) = 1   if slot.criticality == PERIPHERAL
```

Ordinals per §4.1: `EMPTY=0, SUPPORTED=1, CONTESTED=1, RESOLVED=2, UNRESOLVABLE=2`.

## 6.2 State-transition gain table (explicit, for implementation reference)

| From → To | Ordinal delta | Gain contribution (CENTRAL) | Gain contribution (PERIPHERAL) |
|---|---|---|---|
| EMPTY → SUPPORTED | +1 | 3 | 1 |
| EMPTY → CONTESTED | +1 | 3 | 1 |
| SUPPORTED → CONTESTED | 0 | 0 | 0 |
| SUPPORTED → RESOLVED | +1 | 3 | 1 |
| CONTESTED → RESOLVED | +1 | 3 | 1 |
| CONTESTED → UNRESOLVABLE | 0 | 0 | 0 |
| any → same state (no-op / discarded claim) | 0 | 0 | 0 |

An action touching multiple slots sums contributions across all of them.

## 6.3 Token cost

```
cost(action) = tokens_used by that agent's execution + tokens_used by the mandatory
               subsequent EvidenceEvaluatorAgent call on its output
```

This is the **primary** cost metric used in the `score = gain / cost` formula. Latency (ms) and monetary cost ($, if applicable) are recorded as **separate, secondary annotations per action** in the calibration log (§11.6) — they are never combined into `cost` and never affect action selection.

## 6.4 Candidate-action generation

At each step, the set of candidate actions is generated as:

```
candidates = []
for slot in open_slots (state in {EMPTY, SUPPORTED, CONTESTED}):
    if slot.criticality requires linking and slot has no linked entity:
        candidates.append((EntityLinkerAgent, slot))
    if slot has a linked entity:
        candidates.append((GraphTraversalAgent, slot))
    candidates.append((SimilaritySearchAgent, slot))   # always valid, no precondition
    if a candidate doc_id is known for this slot (from prior traversal/search):
        candidates.append((DocumentRetrievalAgent, slot))
if decomposition.requires_aggregation and ≥2 matching-shape slots are SUPPORTED or RESOLVED:
    candidates.append((AggregationAgent, matching_slot_group))
```

Each candidate is scored via `score = predicted_gain(candidate) / cost(candidate)`, where `predicted_gain` is estimated from the historical average gain that agent-type has produced on this slot's criticality level so far in this investigation (bootstrapped from the fit-split priors described in the earlier calibration protocol; frozen values live in `frozen_thresholds.json`).

## 6.5 Selection and tie-breaking

```
selected = argmax(score) over candidates

Tie-breaking (score equal to within 1e-6), applied in order until resolved:
  1. Prefer the action targeting the highest-criticality slot (CENTRAL over PERIPHERAL)
  2. Prefer the action type not yet attempted on this slot in this investigation
  3. Prefer the lowest predicted cost
  4. If still tied, prefer alphabetical order of agent_name (deterministic, reproducible)
```

## 6.6 No candidate available

```
if candidates == []:
    stop_reason = "no_candidate_actions"
    proceed directly to coverage check → Synthesis
```

This is a normal, loggable stop condition (§7), not an exception.

## 6.7 Predicted vs actual gain logging

After executing the selected action and observing the resulting slot-state transitions:

```json
{"step_id": int, "agent_name": "str", "predicted_gain": "float", "actual_gain": "float", "predicted_cost": "int", "actual_cost": "int"}
```

Written to `calibration_log.jsonl` (§11.6) on every single action, no exceptions.

## 6.8 On-policy labeling — mandatory

Every calibration log file and every dashboard view of calibration data must display the fixed annotation:

> `"calibration_type": "on_policy_only"` — actual gain is observed only for the action selected; no counterfactual claim about unselected actions is made.

This string is written once as a file-level header in `calibration_log.jsonl` and must not be omitted from any dashboard rendering of this data.

## 6.9 Regret analysis (optional, as specified)

If implemented: for a **sampled subset** (not exhaustive) of steps, replay the top-2 non-selected candidate actions offline and record their actual gain in `regret_log.jsonl` with `"sampled": true`. This file is optional. **Cut condition (fixed in advance, not re-litigated under time pressure):** if `calibration_log.jsonl` is not fully working and clean by the end of implementation day 5, `regret_log.jsonl` is not built, at all, full stop.

---

# 7. Stopping Behavior

## 7.1 Complete stop-reason enumeration

| `stop_reason` value | Fires when |
|---|---|
| `all_central_slots_resolved` | Coverage check finds every CENTRAL slot in state RESOLVED |
| `no_candidate_actions` | VoI candidate generation (§6.4) returns an empty set |
| `threshold_not_cleared` | Best candidate's score < `θ_voi` |
| `decomposition_retry_exhausted` | A revision was attempted (§5), the new slot set still shows no progress after 2 further actions, and revision budget is now spent |
| `budget_exhausted:steps` | Step count reaches `MAX_STEPS` |
| `budget_exhausted:tokens` | Cumulative tokens_used reaches `MAX_TOKENS_PER_INVESTIGATION` |
| `budget_exhausted:time` | Wall-clock time reaches `MAX_WALL_TIME` |

No other stop reason exists. Every investigation's terminal record must contain exactly one of these seven values.

## 7.2 Precedence when multiple conditions are true simultaneously

Evaluated in this fixed order; the first true condition wins and is the recorded `stop_reason`:

```
1. budget_exhausted:steps
2. budget_exhausted:tokens
3. budget_exhausted:time
4. all_central_slots_resolved
5. no_candidate_actions
6. decomposition_retry_exhausted
7. threshold_not_cleared
```

Rationale for this order (documented, not re-derived at implementation time): budget exhaustion is a hard external constraint and must be checked first regardless of internal state, so that runaway loops are always caught before any other logic executes on a given step.

## 7.3 Configurable budget values

```json
// frozen_thresholds.json (excerpt)
{
  "MAX_STEPS": 8,
  "MAX_TOKENS_PER_INVESTIGATION": 15000,
  "MAX_WALL_TIME_SECONDS": 45
}
```

These are starting values. Per §10, derive final values from the 95th percentile of an uncapped fit-split run before freezing.

## 7.4 Budget exhaustion is a normal, logged outcome

Budget checks must **never** raise an exception, timeout error, or unhandled interrupt. Each check is a normal conditional evaluated at the top of every loop iteration (per the precedence order above), and hitting a budget limit produces a completed investigation record with the corresponding `stop_reason`, flowing normally into Synthesis exactly like any other stop reason. Synthesis must be able to produce ANSWER, PARTIAL, or ABSTAIN off of a budget-exhausted investigation, using whatever slot states exist at the moment of exhaustion.

---

# 8. Synthesis and Final Gates

## 8.1 Synthesis — deterministic exit rule

```python
def synthesis_exit(slots: list[Slot]) -> Literal["ANSWER", "PARTIAL", "ABSTAIN"]:
    central_slots = [s for s in slots if s.criticality == "CENTRAL"]
    if any(s.state == "UNRESOLVABLE" for s in central_slots):
        return "ABSTAIN"
    if all(s.state == "RESOLVED" for s in central_slots):
        return "ANSWER"
    return "PARTIAL"
```

This function takes **only** slot states as input. No LLM call is permitted to override or second-guess this function's output. The LLM's role after this function returns is limited to *composing text* consistent with the exit type already decided — e.g., writing which gaps to name in a PARTIAL, or which reason to give in an ABSTAIN — never to choosing the exit type itself.

## 8.2 Claim Validation (independent of slot resolution)

Runs immediately after Synthesis drafts an answer, regardless of exit type (even ABSTAIN answers get their reasoning claims checked):

```
for claim used in drafted_answer_text:
    re-run entailment_flag check of claim.claim_text against claim.source_passage
    if entailment_flag == FAIL on re-check:
        strip claim from drafted_answer_text
        log to claim_validation_failures.jsonl
if drafted_answer_text now has zero supporting claims for a CENTRAL slot that
   Synthesis assumed was RESOLVED:
    do not silently re-run Synthesis — flag `claim_validation_conflict: true`
    and force downgrade one level (ANSWER→PARTIAL, PARTIAL→ABSTAIN)
```

This check is independent of and happens strictly after slot-state logic — it operates on the *drafted text*, not on the Ledger. It can only remove or downgrade, never add new claims or re-open investigation.

## 8.3 Completeness Gate (independent of slot coverage)

```
completeness_result = LLM_check(original_question, drafted_answer_after_validation)
# single boolean output: PASS or FAIL — no other output permitted

if completeness_result == FAIL:
    if current_exit_type == "ANSWER":
        downgrade to "PARTIAL"
        append note: "completeness gate flagged: may not fully address question"
    elif current_exit_type == "PARTIAL":
        keep as "PARTIAL"
        append note: "completeness gate flagged: may not fully address question"
    # ABSTAIN is never further downgraded; there is nothing below it
output → final answer, unchanged in wording except for the appended note above
```

**Hard prohibition, restated for implementation clarity:** the Completeness Gate function has no return path that calls the Decomposer, VoI scorer, any named agent, or the Ledger write function. Its only possible side effects are: (a) no change, or (b) one-level downgrade of `exit_type` plus an appended text note. It is a pure post-processing function over already-final text.

## 8.4 Sequence diagram — Synthesis through output

```
Coverage Check (stop condition met)
        │
        ▼
   Synthesis.exit_rule(slots) → ANSWER | PARTIAL | ABSTAIN
        │
        ▼
   Synthesis.compose_text(exit_type, ledger) → drafted_answer
        │
        ▼
   Claim Validation (may strip claims, may force one downgrade)
        │
        ▼
   Completeness Gate (may force one further downgrade, appends note)
        │
        ▼
   FINAL OUTPUT — no further stage follows
```

---

# 9. Benchmark and Evaluation Protocol

## 9.1 100 dev questions — exact execution

Run every one of the 100 questions through **all three pipelines independently**, in `BENCHMARK` mode (§1.3). Each pipeline uses a fresh, isolated state per question. Total pipeline executions for this phase: `100 × 3 = 300`.

## 9.2 50 hidden questions — exact execution

Run every one of the 50 questions through the full `ADAPTIVE` mode system (Ladder → selected rung → possible one escalation). Total pipeline executions for this phase: between 50 and 100 (depending on how many escalations fire), plus one Ladder invocation per question.

## 9.3 Matched-baseline requirement

Before generating any comparison chart from the 100-question benchmark, verify and record a config diff across exactly these fields:

```
model, temperature, embedding_model, chunk_size, corpus_version, eval_rubric_version
```

for the RAG, GraphRAG, and Agentic (fixed-action-order variant used as the "fixed-agent" baseline, per earlier design) configurations. If this diff is non-empty on any field other than the retrieval mechanism itself, the comparison is invalid and must not be published until resolved.

## 9.4 Evaluation metrics and formulas

| Metric | Formula |
|---|---|
| Accuracy (per question, per pipeline) | LLM-as-judge PASS/FAIL against ground truth (dev set) or held-out scoring (hidden set, scored externally) |
| BERTScore F1 | standard BERTScore F1 between generated answer and reference answer, where reference exists |
| Token efficiency | `tokens_used` per question, per pipeline, as logged |
| Completeness | fraction of CENTRAL slots RESOLVED at investigation end (Agentic pipeline only; RAG/GraphRAG do not have slots, so this metric is `N/A` for them, not zero) |
| Ladder routing accuracy | see §9.6 |

## 9.5 Data collected per question and per step

**Per question** (all pipelines, `BENCHMARK` mode): `question_id, pipeline, answer, exit_type (Agentic only, else null), tokens_used, latency_ms, accuracy_score, bertscore_f1, sources, n_chunks_retrieved, n_sources_cited`.

**Per step** (Agentic pipeline only, both modes): every record already specified in §3.3, §4.9, §6.7 — no additional per-step fields are collected beyond what those sections define.

## 9.6 Ladder routing accuracy — confusion matrix construction

Using `BENCHMARK`-mode results only:

```
for each of the 100 dev questions:
    ground_truth_rung = cheapest rung among {RAG, GraphRAG, Agentic} whose
                         accuracy_score == PASS for this question
                         (if RAG passes, ground_truth_rung = RAG;
                          else if GraphRAG passes, = GraphRAG;
                          else if Agentic passes, = Agentic;
                          else = "none_passed")
    predicted_rung = Ladder.route(question)   # run once, separately, not
                                                # part of the benchmark execution
    record (ground_truth_rung, predicted_rung)

build a 4x4 confusion matrix over {RAG, GraphRAG, Agentic, none_passed} x
{RAG, GraphRAG, Agentic}
```

This matrix is a required dashboard artifact for `BENCHMARK` mode (§1.4).

---

# 10. Configuration and Threshold Freezing

## 10.1 Configuration files

| File | Contains |
|---|---|
| `config/frozen_thresholds.json` | `θ_route, θ_grounding, θ_voi, bar_CENTRAL, bar_PERIPHERAL, MAX_STEPS, MAX_TOKENS_PER_INVESTIGATION, MAX_WALL_TIME_SECONDS` |
| `config/model_config.json` | `model, temperature, embedding_model, chunk_size, corpus_version, eval_rubric_version` — the fields checked in §9.3 |
| `config/agent_config.json` | per-agent parameters (e.g., `top_k` for SimilaritySearchAgent, `max_hops` default for GraphTraversalAgent) |

## 10.2 Fields tuned vs. fields fixed by design

| Tuned via cross-validation | Fixed by architecture (not tuned) |
|---|---|
| `θ_route, θ_grounding, θ_voi` | `weight(CENTRAL)=3, weight(PERIPHERAL)=1` (§6.1) |
| `bar_CENTRAL, bar_PERIPHERAL` | ordinal values (§4.1) |
| `MAX_STEPS, MAX_TOKENS_PER_INVESTIGATION, MAX_WALL_TIME_SECONDS` | `K=3` (stall trigger count, §5.1) |
| | decomposition revision limit = 1 (§5.3) |
| | premise-contradiction whitelist (§5.2) — closed list, not tunable |

## 10.3 5-fold cross-validation procedure

```
1. Split the 100 dev questions: 80 → dev-fit pool, 20 → held-out (untouched until step 5)
2. Partition the 80 dev-fit questions into 5 folds of 16 each
3. For each of θ_route, θ_grounding, θ_voi, bar_CENTRAL, bar_PERIPHERAL, MAX_STEPS,
   MAX_TOKENS_PER_INVESTIGATION, MAX_WALL_TIME_SECONDS:
     for each fold f in 1..5:
         train on the other 4 folds (64 questions) to derive a candidate value
         validate on fold f
     select the value with best average validation performance across the 5 folds
4. Write all selected values into config/frozen_thresholds.json
5. Run the 20 held-out dev questions ONCE with this file. Record results — this
   is your reported generalization-gap evidence (fit-split score vs. held-out score)
6. Run the 50 hidden questions ONCE with this same frozen file.
```

Budget values (`MAX_STEPS` etc.) are derived, as previously specified, from the 95th percentile of an **uncapped** run on the dev-fit pool, then subjected to the same cross-validation selection if any further tuning is applied.

## 10.4 Freeze process

```
1. On completing step 4 above, compute a SHA-256 hash of frozen_thresholds.json
2. Commit the file to version control with a commit message including the hash
   and a timestamp
3. Mark the file read-only at the OS level (chmod 444 or equivalent) for the
   remainder of the competition
```

## 10.5 Bug fixes after freezing

```
if a bug is discovered after freezing that changes system behavior:
    do NOT edit frozen_thresholds.json
    create frozen_thresholds_v2.json with a new timestamp and new commit
    re-run the held-out and/or hidden questions under v2
    report BOTH v1 and v2 results in the final submission, labeled by
    which config produced which
```

Silently re-running under a modified frozen file and reporting only the new numbers is prohibited by this specification.

## 10.6 Preventing accidental modification

- File permission lock (§10.4, step 3).
- CI/pre-commit hook (if available) that rejects any commit modifying `frozen_thresholds.json` after the freeze commit hash exists, unless the filename itself has changed (i.e., a new versioned file, per §10.5).
- Code review checklist item: "Does this change touch `frozen_thresholds.json`? If yes, has a new version file been created instead?"

---

# 11. Logging and Artifact Schemas

Exactly five architectural logs exist, plus the two mode-level result files from §1.4, plus one claim-validation failure log implied by §8.2, and the calibration/regret files from §6. **No additional logs are to be introduced.**

## 11.1 `escalation_log.jsonl`

```json
{"question_id": "str", "from_rung": "RAG|GraphRAG", "to_rung": "GraphRAG|Agentic", "confidence_score": "float", "threshold_used": "float", "timestamp": "ISO8601"}
```
Example:
```json
{"question_id": "q_0042", "from_rung": "RAG", "to_rung": "GraphRAG", "confidence_score": 0.31, "threshold_used": 0.5, "timestamp": "2026-09-20T04:12:03Z"}
```

## 11.2 `stop_reason_log.jsonl`

```json
{"question_id": "str", "stop_reason": "str (one of the 7 values in §7.1)", "step_count": "int", "tokens_used_total": "int", "wall_time_seconds": "float", "timestamp": "ISO8601"}
```

## 11.3 `ledger_writes.jsonl`

```json
{
  "claim_id": "str", "question_id": "str", "step_id": "int",
  "subject": "str", "predicate": "str", "object": "str",
  "source_passage": "str", "confidence": "float",
  "provenance_chain": ["str", "..."], "entailment_flag": "PASS|FAIL",
  "derived_from": ["claim_id", "..."],
  "valid_from": "ISO8601 | null", "valid_to": "ISO8601 | null",
  "source_authority": "str | null",
  "target_slot_id": "str", "resulting_slot_state": "str",
  "contradiction_detected": "bool", "contradicted_claim_id": "str | null"
}
```
`valid_from`, `valid_to`, `source_authority` are `null` for every Round 1 record — required fields, unpopulated values, per §11.9.

## 11.4 `agent_invocations.jsonl`

```json
{"agent_name": "str", "question_id": "str", "step_id": "int", "input_summary": "str", "tokens_used": "int", "latency_ms": "int", "output_summary": "str", "timestamp": "ISO8601"}
```
Includes `LedgerGuard` records for invalid-transition attempts (§4.2).

## 11.5 `decomposition_revisions.jsonl`

```json
{"question_id": "str", "trigger_type": "stall|premise_contradiction", "old_slots": [{"slot_id":"str","description":"str","criticality":"str"}], "new_slots": [{"slot_id":"str","description":"str","criticality":"str"}], "triggering_claim_id": "str | null", "step_id_at_revision": "int", "timestamp": "ISO8601"}
```

## 11.6 `calibration_log.jsonl`

File-level header record (first line):
```json
{"calibration_type": "on_policy_only", "note": "actual gain observed only for selected actions; no counterfactual claim"}
```
Per-action records:
```json
{"question_id": "str", "step_id": "int", "agent_name": "str", "predicted_gain": "float", "actual_gain": "float", "predicted_cost": "int", "actual_cost": "int", "latency_ms": "int", "monetary_cost_usd": "float | null"}
```

## 11.7 `benchmark_results.jsonl` (BENCHMARK mode only)

```json
{
  "question_id": "str",
  "rag": {"answer": "str", "sources": ["str"], "tokens_used": "int", "latency_ms": "int", "accuracy_score": "PASS|FAIL", "bertscore_f1": "float"},
  "graphrag": {"answer": "str", "sources": ["str"], "tokens_used": "int", "latency_ms": "int", "accuracy_score": "PASS|FAIL", "bertscore_f1": "float"},
  "agentic": {"answer": "str", "exit_type": "ANSWER|PARTIAL|ABSTAIN", "sources": ["str"], "tokens_used": "int", "latency_ms": "int", "accuracy_score": "PASS|FAIL", "bertscore_f1": "float", "n_chunks_retrieved": "int", "n_sources_cited": "int", "strategy_changed": "bool"}
}
```

## 11.8 `adaptive_results.jsonl` (ADAPTIVE mode only)

```json
{
  "question_id": "str",
  "rung_path": ["RAG"] or ["RAG","GraphRAG"] or ["Agentic"],
  "final_answer": "str",
  "exit_type": "ANSWER|PARTIAL|ABSTAIN|null (null if final rung was RAG or GraphRAG)",
  "tokens_used_total": "int",
  "latency_ms_total": "int",
  "n_chunks_retrieved": "int",
  "n_sources_cited": "int",
  "strategy_changed": "bool",
  "stop_reason": "str|null (null if final rung was RAG or GraphRAG, not Agentic)"
}
```

`strategy_changed` computation (§4 of prior patch set, restated exactly):
```
strategy_changed = (decomposition_revisions_used > 0)
                    OR (len(rung_path) > 1)   # i.e., an escalation occurred
                    OR (agent-type distribution in steps 4+ differs from
                        agent-type distribution in steps 1-3, by majority type)
```

## 11.9 `claim_validation_failures.jsonl`

```json
{"question_id": "str", "claim_id": "str", "reason": "entailment_recheck_failed", "forced_downgrade": "bool", "old_exit_type": "str", "new_exit_type": "str", "timestamp": "ISO8601"}
```

## 11.10 `completeness_gate_log.jsonl`

```json
{"question_id": "str", "result": "PASS|FAIL", "exit_type_before": "str", "exit_type_after": "str", "note_appended": "bool", "timestamp": "ISO8601"}
```

## 11.11 `regret_log.jsonl` (optional, per §6.9)

```json
{"question_id": "str", "step_id": "int", "selected_agent": "str", "alternative_agent": "str", "alternative_actual_gain": "float", "sampled": true}
```

---

# 12. Baseline Fairness

## 12.1 Fields that must match exactly across RAG, GraphRAG, and Agentic (fixed-order variant)

```
model                  (exact model identifier string)
temperature
embedding_model
chunk_size
corpus_version         (hash or version tag of the ingested corpus)
eval_rubric_version    (hash or version tag of the LLM-as-judge prompt + BERTScore config)
```

## 12.2 Permitted differences

Only the retrieval mechanism itself may differ: RAG's vector-only retrieval vs. GraphRAG's graph traversal vs. Agentic's multi-step agent loop. Reranking, if used, must be present (or absent) identically across all three, unless one pipeline's defining feature *is* a component the others structurally cannot have (e.g., Agentic's Ledger) — in which case the difference is documented, not hidden.

## 12.3 Validation check (must run before any comparison chart is generated)

```python
def validate_baseline_fairness(rag_cfg, graphrag_cfg, agentic_cfg):
    fields = ["model", "temperature", "embedding_model", "chunk_size",
              "corpus_version", "eval_rubric_version"]
    for f in fields:
        if not (rag_cfg[f] == graphrag_cfg[f] == agentic_cfg[f]):
            raise ConfigMismatchError(f"Field '{f}' differs across pipelines: "
                f"RAG={rag_cfg[f]}, GraphRAG={graphrag_cfg[f]}, Agentic={agentic_cfg[f]}")
    return True
```

This function must return `True` (or the process must halt) before `benchmark_results.jsonl` generation is considered valid for dashboard use. Include its pass/fail output as an appendix table in the final submission.

---

# 13. Testing

Each component is tested in isolation, using the dependency order from §14, before any integration test is attempted.

## 13.1 Ledger

| Test case | Input | Expected output |
|---|---|---|
| Basic write | Claim with `entailment_flag=PASS`, no `derived_from` | Slot transitions EMPTY→SUPPORTED; `confidence` = own confidence unmodified |
| Failed entailment | Claim with `entailment_flag=FAIL` | Claim is **not** written; slot state unchanged; `discarded_claim` logged |
| Contradiction | Two claims, same subject+predicate, different object | Second claim gets `contradiction_detected=true`; slot → CONTESTED |
| Confidence inheritance | Claim with `derived_from=[claim_a (conf 0.9), claim_b (conf 0.4)]`, own evidence conf 0.95 | Result confidence = `min(0.95, 0.9, 0.4) = 0.4` |
| RESOLVED — CENTRAL, single source | CENTRAL slot, one claim, confidence 0.95, provenance depth 1, entailment PASS | RESOLVED (meets condition 4's single-source exception) |
| RESOLVED — CENTRAL, single source, insufficient | CENTRAL slot, one claim, confidence 0.8 | **Not** RESOLVED — stays SUPPORTED (fails condition 4) |
| Invalid transition | Attempt `RESOLVED → CONTESTED` | Write rejected; `invalid_transition_attempt` logged; state unchanged |
| Terminal state | Attempt any write to an `UNRESOLVABLE` slot | Write rejected; state remains UNRESOLVABLE |

## 13.2 Decomposer

| Test case | Input | Expected output |
|---|---|---|
| Simple factual question | "Who founded Company X?" | 1 slot, CENTRAL, e.g. `{founder of Company X}` |
| Comparison question | "Compare revenue of X and Y" | ≥2 slots (CENTRAL), `requires_aggregation=true` |
| Slot-recall check | 10 hand-labeled questions | Decomposer's slot set overlaps hand-labels ≥ agreed threshold (report the raw number; no pass/fail bar defined by architecture, this is descriptive) |

## 13.3 RAG / GraphRAG (standalone)

| Test case | Input | Expected output |
|---|---|---|
| Normal question | question with known answer in corpus | Non-empty `answer`, non-empty `sources` |
| No evidence | question about entity not in corpus | `answer = "NO_EVIDENCE_RETRIEVED"` (RAG) or `"NO_ENTITY_MATCH"` (GraphRAG), `sources = []` — not an exception |

## 13.4 Agent loop (fixed action order, VoI disabled)

| Test case | Input | Expected output |
|---|---|---|
| Terminates normally | question with clean, findable answer | Reaches `all_central_slots_resolved`, valid stop_reason logged |
| Terminates on budget | question artificially given `MAX_STEPS=1` | `budget_exhausted:steps` logged, Synthesis still runs and produces PARTIAL or ABSTAIN |
| No candidates | slot set with no linkable entities and search yields nothing | `no_candidate_actions` logged |

## 13.5 VoI scorer

| Test case | Input | Expected output |
|---|---|---|
| Gain formula, single CENTRAL slot | action moves 1 CENTRAL slot EMPTY→SUPPORTED | `gain = 3 × 1 = 3` |
| Gain formula, multiple PERIPHERAL slots | action moves 3 PERIPHERAL slots EMPTY→SUPPORTED | `gain = 1×1 + 1×1 + 1×1 = 3` |
| No-op transition | action results in SUPPORTED→CONTESTED | `gain = 0` |
| Tie-break | two candidates with identical score | resolved per §6.5 order; test asserts CENTRAL-targeting candidate wins step 1 of tie-break |
| Empty candidates | no valid actions | returns `no_candidate_actions`, does not raise |

## 13.6 Ladder

| Test case | Input | Expected output |
|---|---|---|
| Known-easy question | single-hop, well-linked entity | Routes to RAG |
| Known-hard question | multi-hop, ambiguous entities | Routes to Agentic |
| Escalation | RAG confidence below θ_grounding | Escalates to GraphRAG exactly once, logged |

## 13.7 Synthesis / Claim Validation / Completeness Gate

| Test case | Input | Expected output |
|---|---|---|
| All CENTRAL RESOLVED | slot set fully resolved | `exit_type = ANSWER` |
| One CENTRAL UNRESOLVABLE | one CENTRAL slot UNRESOLVABLE, rest RESOLVED | `exit_type = ABSTAIN` |
| Mixed | some CENTRAL SUPPORTED-only | `exit_type = PARTIAL` |
| Claim validation strips a claim | drafted answer cites a claim that fails re-check | Claim stripped, logged, exit downgraded one level if it broke a CENTRAL RESOLVED assumption |
| Completeness gate fails on ANSWER | gate returns FAIL on a well-resolved but non-responsive answer | Downgrades to PARTIAL, note appended, **no re-investigation triggered** (explicitly assert no Decomposer/agent/Ledger calls occur after this point) |

---

# 14. Project Implementation Order

Each stage must pass its acceptance criteria before the next stage begins. This order is unchanged from the frozen architecture — no stage added, removed, or reordered.

| Stage | Deliverable | Acceptance criteria before proceeding |
|---|---|---|
| 1. Ledger | Slot state machine, write function, confidence inheritance | All §13.1 test cases pass against hand-written fake claims, no live retrieval |
| 2. Decomposer | Slot generation with CENTRAL/PERIPHERAL tagging | §13.2 test cases pass; slot-recall measured against 10 hand-labeled questions |
| 3. RAG / GraphRAG baselines | Standalone pipelines, matched config | §13.3 test cases pass; `validate_baseline_fairness` (§12.3) returns True across RAG/GraphRAG configs |
| 4. Agent loop, fixed order | Full loop without VoI, budgets enforced | §13.4 test cases pass; every stop_reason in §7.1 reachable except `threshold_not_cleared` (N/A without VoI) |
| 5. VoI scorer | Gain formula, candidate generation, tie-breaking, calibration log | §13.5 test cases pass; `calibration_log.jsonl` produces valid records with on-policy header |
| 6. Ladder | Routing + escalation | §13.6 test cases pass; escalation fires at most once per question |
| 7. Named agents + Dispatcher + Evidence Evaluator | All six agents per §3, logging per §3.3 | Every agent invocation produces a valid `agent_invocations.jsonl` record; Ledger write function contains zero evaluation logic (code review check) |
| 8. Synthesis + Claim Validation + Completeness Gate | Deterministic exit rule, independent validation stages | §13.7 test cases pass, including the explicit no-reinvestigation assertion |
| 9. Threshold cross-validation + freeze | `frozen_thresholds.json`, committed and hashed | §10.3–10.4 procedure completed; file is read-only; held-out gap reported |
| 10. Full BENCHMARK run | `benchmark_results.jsonl`, confusion matrix | All 100 questions × 3 pipelines complete; `validate_baseline_fairness` passes; confusion matrix (§9.6) generated |
| 11. Full ADAPTIVE run | `adaptive_results.jsonl` | All 50 hidden questions complete under frozen config; every record has a valid `stop_reason` or is a non-Agentic rung result |
| 12. Dashboard | Reads only from the log files in §11, no live computation | Every number on the dashboard traces to a committed JSONL file (manual audit checklist) |

---

# 15. Implementation Invariants

These rules must never be violated, in any implementation change, at any point:

1. **`BENCHMARK` mode always runs all three pipelines independently; `ADAPTIVE` mode always runs exactly one, with at most one escalation.** These modes are never mixed within a single question's execution.
2. **The Ledger never evaluates evidence.** All entailment, confidence, and contradiction logic lives exclusively in `EvidenceEvaluatorAgent`. The Ledger's write function only records and transitions state.
3. **A claim with `entailment_flag == FAIL` is never written to the Ledger.**
4. **`RESOLVED` and `UNRESOLVABLE` are terminal states.** No transition out of either is ever permitted.
5. **Exactly one decomposition revision is allowed per investigation, total**, regardless of how many times a trigger condition fires.
6. **The Completeness Gate can only leave output unchanged or downgrade it by one level.** It can never trigger a new investigation, call an agent, call the Decomposer, or write to the Ledger.
7. **Synthesis's exit type (`ANSWER`/`PARTIAL`/`ABSTAIN`) is decided by a pure function over slot states — never by an LLM's free judgment.**
8. **Budget exhaustion is always a normal, logged stop reason — never an exception, crash, or silent truncation.**
9. **`frozen_thresholds.json` is never edited after freezing.** A behavior-changing fix requires a new, separately versioned file, with both results reported.
10. **Calibration data is always labeled `on_policy_only`.** No dashboard or writeup may present it as a counterfactual or causal claim.
11. **Every named-agent invocation is logged with `agent_name` before its output is used anywhere downstream.** An unlogged invocation's output is treated as if it did not occur.
12. **Ladder, Ledger, Synthesis, Decomposer, VoI Scorer, Dispatcher, and Completeness Gate are never implemented, logged, or described as agents.** Only the six entities in the Agent Registry (§3.1) hold that designation.
13. **`valid_from`, `valid_to`, `source_authority` exist as required schema fields on every claim from Round 1 onward**, populated with `null` until Round 2 logic consumes them — they are never removed or omitted from the schema.
14. **No log file exists beyond the ones enumerated in §11.** No architectural component exists beyond the ones enumerated in this document and its predecessor design passes.
15. **Every number appearing on the dashboard must be traceable to a specific line in a specific committed JSONL artifact.** Nothing on the dashboard is computed live at display time.
