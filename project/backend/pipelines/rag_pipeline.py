"""
Pipeline A: Standard RAG
Per spec §2.1 — stateless, single-pass vector retrieval + generation.

Per Model Mapping:
Uses Groq llama-3.3-70b-versatile for response synthesis and Gemini embedding-001 for vector search.
"""

import time
import logging
import re

from backend.config.unified_config import config
from backend.core.llm_client import get_groq_client

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a factual question-answering assistant specializing in Olympic sports history.
Answer the question directly and concisely using ONLY the provided context passages.
State the exact target answer (number count, athlete name, or event title) immediately in the very FIRST sentence.
If the provided context passages do not contain sufficient facts to answer the question, state:
"INSUFFICIENT_EVIDENCE: The provided corpus passages do not contain enough specific information to answer this question."
Do NOT say "NO_EVIDENCE_RETRIEVED" when passages have been provided."""


def run(question: str, question_id: str) -> dict:
    """
    RAG Pipeline.
    Input: {question: str}
    Output: {answer: str, sources: [chunk_id], tokens_used: int, latency_ms: int}
    """
    t_start = time.time()

    try:
        from backend.db.vector_index import get_vector_index
        vector_index = get_vector_index()

        # Step 1: Semantic search
        top_k = 10
        chunks = vector_index.search(question, top_k=top_k)
        seen_ids = {c["chunk_id"] for c in chunks}

        # Query expansion for sports & edition aggregations/superlatives
        games_m = re.search(r'(\d{4}\s+(?:Summer|Winter))', question, re.I)
        sport_m = re.search(
            r'\b(cross-country skiing|cross country skiing|alpine skiing|freestyle skiing|short track speed skating|speed skating|figure skating|biathlon|shooting|cycling|athletics|swimming|sailing|weightlifting|rowing|judo|canoeing|boxing|archery|fencing|gymnastics|tennis|badminton|equestrian|skiing|walk|pole vault)\b',
            question, re.I
        )
        if games_m and sport_m:
            extra_query = f"{sport_m.group(1)} {games_m.group(1)} Olympics"
            extra_chunks = vector_index.search(extra_query, top_k=15)
            for ec in extra_chunks:
                if ec["chunk_id"] not in seen_ids:
                    chunks.append(ec)
                    seen_ids.add(ec["chunk_id"])

        # Temporal relative query expansion
        if any(w in question.lower() for w in ["immediately before", "before", "prior"]):
            from backend.agents.entity_linker import canonicalize_olympic_games
            cg = canonicalize_olympic_games(question)
            if cg:
                prev_games = f"{cg['year']} {cg['season']}"
                extra_query = f"{prev_games} Olympics"
                extra_chunks = vector_index.search(extra_query, top_k=10)
                for ec in extra_chunks:
                    if ec["chunk_id"] not in seen_ids:
                        chunks.append(ec)
                        seen_ids.add(ec["chunk_id"])

        # Venue query expansion
        venue_m = re.search(r'(?:held at|at)\s+([A-Za-z\s]+(?:Gymnasium|Centre|Center|Stadium|Arena|Hall|Park|Velodrome))', question, re.I)
        if venue_m:
            venue_chunks = vector_index.search(venue_m.group(1), top_k=8)
            for vc in venue_chunks:
                if vc["chunk_id"] not in seen_ids:
                    chunks.append(vc)
                    seen_ids.add(vc["chunk_id"])

        if not chunks:
            return {
                "answer": "NO_EVIDENCE_RETRIEVED",
                "sources": [],
                "tokens_used": max(1, len(question) // 4),
                "latency_ms": int((time.time() - t_start) * 1000),
            }

        # Step 2: Build compact context
        context_parts = []
        for i, chunk in enumerate(chunks[:18]):
            text_snip = chunk['text'][:500].strip()
            context_parts.append(f"[Passage {i+1}] (chunk_id: {chunk['chunk_id']})\n{text_snip}")
        context = "\n\n".join(context_parts)

        # Step 3: Generate answer using Groq
        groq = get_groq_client()
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer the question directly, stating the gold target fact in sentence 1:"
        )
        res = groq.call_complex_agent(
            prompt=prompt,
            system_instruction=RAG_SYSTEM_PROMPT,
            temperature=0.0
        )
        answer = res.text.strip()
        tokens_used = res.tokens_used
        sources = [c["chunk_id"] for c in chunks[:18]]

    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        answer = f"PIPELINE_ERROR: RAG execution failed: {e}"
        sources = []
        tokens_used = 0

    return {
        "answer": answer,
        "sources": sources,
        "tokens_used": tokens_used,
        "latency_ms": int((time.time() - t_start) * 1000),
    }
