"""
SimilaritySearchAgent — vector search over the corpus index with TigerGraph event criteria retrieval.
Per spec §3.1: always valid, no precondition on prior linking.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.config.unified_config import config, LOG_DIR

logger = logging.getLogger(__name__)
LOG_DIR = Path(config.paths.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run(input_data: dict, question_id: str, step_id: int) -> dict:
    """
    SimilaritySearchAgent execution combining semantic chunk search with
    TigerGraph event criteria resolution.

    Input per spec §3.2:
      {"query_string": str, "top_k": int, "corpus_index": VectorIndex ref}

    Output per spec §3.2:
      {"chunks": [{"chunk_id": str, "text": str, "score": float}], "tokens_used": int}
    """
    t_start = time.time()
    query_string = input_data.get("query_string", "")
    top_k = input_data.get("top_k", 5)

    if not query_string:
        _log_invocation("SimilaritySearchAgent", question_id, step_id,
                        "empty_query", 0, t_start, "failed: empty query string")
        return {"chunks": [], "tokens_used": 0}

    chunks = []
    tokens_used = max(1, len(query_string) // 4)

    # 1. Direct TigerGraph Event Criteria lookup (if sport/games/venue present)
    try:
        from backend.db.tigergraph_client import get_tg_client
        tg = get_tg_client()

        games_m = re.search(r'(\d{4}\s+(?:Summer|Winter))', query_string, re.I)
        sport_m = re.search(
            r'\b(cross-country skiing|cross country skiing|alpine skiing|freestyle skiing|short track speed skating|speed skating|figure skating|biathlon|shooting|cycling|athletics|swimming|sailing|weightlifting|rowing|judo|canoeing|boxing|archery|fencing|gymnastics|tennis|badminton|equestrian|skiing|walk|pole vault)\b',
            query_string, re.I
        )
        venue_m = re.search(
            r'(?:held at|at)\s+([A-Za-z\s]+(?:Gymnasium|Centre|Center|Stadium|Arena|Hall|Park|Velodrome))',
            query_string, re.I
        )

        g_str = games_m.group(1) if games_m else ""
        s_raw = sport_m.group(1) if sport_m else ""
        s_str = s_raw.title() if s_raw else ""
        if "cross-country" in s_str.lower() or "cross country" in s_str.lower():
            s_str = "Cross-country skiing"
        elif "alpine" in s_str.lower():
            s_str = "Alpine skiing"

        if venue_m:
            g_str = venue_m.group(1).strip()

        if g_str or s_str:
            events = tg.get_events_by_criteria(games=g_str, sport=s_str, min_competitors=0)
            if not events and s_str:
                events = tg.get_events_by_criteria(games=g_str, sport="", min_competitors=0)
            for ev in events[:20]:
                comp_str = f" Competitors: {ev.get('competitors')}." if ev.get('competitors') is not None else ""
                gold_str = f" Gold winner: {ev.get('gold')}." if ev.get('gold') else ""
                date_str = f" Date: {ev.get('date')}." if ev.get('date') else ""
                venue_str = f" Venue: {ev.get('venue')}." if ev.get('venue') else ""
                chunks.append({
                    "chunk_id": ev.get("doc_id", ""),
                    "text": f"Event: {ev.get('title')}.{comp_str}{gold_str}{date_str}{venue_str}",
                    "score": 0.95,
                    "doc_id": ev.get("doc_id", ""),
                })
    except Exception as e:
        logger.debug(f"Event criteria lookup in similarity_search: {e}")

    # 2. Vector search over corpus chunks
    try:
        from backend.db.vector_index import get_vector_index
        vector_index = get_vector_index()
        v_chunks = vector_index.search(query_string, top_k=top_k, score_threshold=0.0)
        chunks.extend(v_chunks)
    except Exception as e:
        logger.error(f"SimilaritySearchAgent vector search error: {e}")

    # Sort combined results by score before slicing
    chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    output = {
        "chunks": chunks[:top_k],
        "tokens_used": tokens_used,
    }

    best_score = f"{output['chunks'][0]['score']:.3f}" if output['chunks'] else "0.000"
    _log_invocation(
        "SimilaritySearchAgent", question_id, step_id,
        f"query={query_string[:100]}, top_k={top_k}",
        output["tokens_used"], t_start,
        f"found {len(output['chunks'])} chunks, best_score={best_score}",
    )
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
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to log invocation: {e}")
