"""
SimilaritySearchAgent — vector search over the corpus index.
Per spec §3.1: always valid, no precondition on prior linking.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import json

logger = logging.getLogger(__name__)
LOG_DIR = Path("./logs")


def run(input_data: dict, question_id: str, step_id: int) -> dict:
    """
    SimilaritySearchAgent execution.

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

    try:
        from backend.db.vector_index import get_vector_index
        vector_index = get_vector_index()

        chunks = vector_index.search(query_string, top_k=top_k, score_threshold=0.0)

        # Estimate embedding tokens (~query length)
        tokens_used = max(1, len(query_string) // 4)

        output = {
            "chunks": chunks,
            "tokens_used": tokens_used,
        }

    except Exception as e:
        logger.error(f"SimilaritySearchAgent error: {e}")
        output = {"chunks": [], "tokens_used": 0}

    _log_invocation(
        "SimilaritySearchAgent", question_id, step_id,
        f"query={query_string[:100]}, top_k={top_k}",
        output["tokens_used"], t_start,
        f"found {len(output['chunks'])} chunks, "
        f"best_score={output['chunks'][0]['score']:.3f if output['chunks'] else 0:.3f}",
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
