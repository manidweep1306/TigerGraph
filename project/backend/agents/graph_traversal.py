"""
GraphTraversalAgent — executes GSQL multi-hop traversal from a linked entity.
Per spec §3.1: invocable when at least one entity is already linked (via Ledger claims).
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
LOG_DIR = Path("./logs")


def run(input_data: dict, question_id: str, step_id: int) -> dict:
    """
    GraphTraversalAgent execution.

    Input per spec §3.2:
      {"start_entity_id": str, "max_hops": int, "graph": TigerGraphInstance ref}

    Output per spec §3.2:
      {"paths": [{"entities": [str], "edges": [str], "hop_count": int}], "tokens_used": int}
    """
    t_start = time.time()
    start_entity_id = input_data.get("start_entity_id", "")
    max_hops = input_data.get("max_hops", 2)

    if not start_entity_id:
        _log_invocation("GraphTraversalAgent", question_id, step_id,
                        "no_entity_id", 0, t_start, "failed: no start_entity_id")
        return {"paths": [], "tokens_used": 0, "entities_found": [],
                "doc_ids_found": [], "games_found": []}

    try:
        from backend.db.tigergraph_client import get_tg_client
        tg = get_tg_client()

        # Run multi-hop traversal
        result = tg.multi_hop_traversal(start_entity_id, max_hops=max_hops, max_results=20)
        entities_found = []
        edges_found = []

        if result and len(result) > 0:
            entities_found = [
                {"entity_id": e.get("v_id", ""), "name": e.get("attributes", {}).get("name", "")}
                for e in result[0].get("entities", [])
            ]
            edges_found = result[0].get("edges", [])

        # Also find adjacent Olympics for temporal reasoning
        games_hop = tg.run_query("events_at_games", {"year": 0, "season": "Summer"})

        # Get docs linked to start entity
        doc_results = tg.docs_by_entity(start_entity_id, top_k=5)
        doc_ids = [d.get("v_id", "") for d in doc_results]

        # Get medal info for the entity
        entity_data = tg.get_entity_by_id(start_entity_id)
        entity_context = ""
        if entity_data:
            attrs = entity_data.get("attributes", {}) if isinstance(entity_data, dict) else {}
            entity_context = f"Entity: {attrs.get('name', start_entity_id)}, Type: {attrs.get('entity_type', 'unknown')}"

        # Build structured paths
        paths = []
        for entity in entities_found:
            paths.append({
                "entities": [start_entity_id, entity["entity_id"]],
                "edges": [f"RELATED_ENTITY"],
                "hop_count": 1,
                "entity_names": [start_entity_id, entity["name"]],
            })

        # Community context
        community = tg.entity_community_context(start_entity_id)
        neighbors = []
        games_info = []
        if community and len(community) > 0:
            neighbors = [n.get("attributes", {}).get("name", "") for n in community[0].get("neighbors", [])]
            games_info = [
                f"{g.get('attributes', {}).get('year', '')} {g.get('attributes', {}).get('season', '')} Olympics"
                for g in community[0].get("games", [])
            ]

        output = {
            "paths": paths,
            "entities_found": entities_found,
            "doc_ids_found": doc_ids,
            "games_found": games_info,
            "entity_context": entity_context,
            "neighbor_entities": neighbors,
            "tokens_used": 0,  # GSQL queries don't consume LLM tokens
        }

    except Exception as e:
        logger.error(f"GraphTraversalAgent error: {e}")
        output = {"paths": [], "tokens_used": 0, "entities_found": [],
                  "doc_ids_found": [], "games_found": []}

    latency_ms = int((time.time() - t_start) * 1000)
    output["tokens_used"] = 0  # graph traversal uses no LLM tokens
    _log_invocation("GraphTraversalAgent", question_id, step_id,
                    f"entity={start_entity_id}, hops={max_hops}",
                    0, t_start,
                    f"found {len(output.get('entities_found', []))} entities, "
                    f"{len(output.get('doc_ids_found', []))} docs")
    return output


def _log_invocation(agent_name, question_id, step_id, input_summary,
                     tokens_used, t_start, output_summary):
    record = {
        "agent_name": agent_name,
        "question_id": question_id,
        "step_id": step_id,
        "input_summary": str(input_summary)[:300],
        "tokens_used": tokens_used,
        "latency_ms": int((time.time() - t_start) * 1000),
        "output_summary": str(output_summary)[:300],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_path = LOG_DIR / "agent_invocations.jsonl"
    try:
        import json
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to log invocation: {e}")
