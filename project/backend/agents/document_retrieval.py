"""
DocumentRetrievalAgent — fetches full document/chunk content by ID.
Per spec §3.1: invocable when a candidate doc_id is known from prior search or traversal.
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
    DocumentRetrievalAgent execution.

    Input per spec §3.2:
      {"doc_id": str}

    Output per spec §3.2:
      {"content": str, "doc_id": str, "tokens_used": int}
    """
    t_start = time.time()
    doc_id = input_data.get("doc_id", "")

    if not doc_id:
        _log_invocation("DocumentRetrievalAgent", question_id, step_id,
                        "no_doc_id", 0, t_start, "failed: no doc_id provided")
        return {"content": "", "doc_id": "", "tokens_used": 0}

    try:
        from backend.db.tigergraph_client import get_tg_client
        tg = get_tg_client()

        # Fetch the document vertex
        doc = tg.get_vertex("Document", doc_id)
        if not doc:
            # Try fetching as Chunk
            doc = tg.get_vertex("Chunk", doc_id)

        content = ""
        if doc:
            if isinstance(doc, list):
                doc = doc[0]
            attrs = doc.get("attributes", {})
            content = attrs.get("text", "")
            title = attrs.get("title", "")
            url = attrs.get("url", "")
            if title:
                content = f"[Title: {title}]\n{content}"

        tokens_used = max(1, len(content) // 4)

        output = {
            "content": content,
            "doc_id": doc_id,
            "tokens_used": tokens_used,
        }

    except Exception as e:
        logger.error(f"DocumentRetrievalAgent error for doc_id={doc_id}: {e}")
        output = {"content": "", "doc_id": doc_id, "tokens_used": 0}

    _log_invocation(
        "DocumentRetrievalAgent", question_id, step_id,
        f"doc_id={doc_id}",
        output["tokens_used"], t_start,
        f"retrieved {len(output['content'])} chars",
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
