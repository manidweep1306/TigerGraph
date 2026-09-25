"""
Pipeline A: Standard RAG
Per spec §2.1 — stateless, single-pass vector retrieval + generation.
"""

import os
import time
import logging
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
TOP_K = 5

RAG_SYSTEM_PROMPT = """You are a factual question-answering assistant specializing in Olympic sports history.
Answer the question using ONLY the provided context passages. 
If the context does not contain sufficient information, say "NO_EVIDENCE_RETRIEVED".
Be concise and specific. Cite which passages support your answer."""


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
        chunks = vector_index.search(question, top_k=TOP_K)

        if not chunks:
            return {
                "answer": "NO_EVIDENCE_RETRIEVED",
                "sources": [],
                "tokens_used": max(1, len(question) // 4),
                "latency_ms": int((time.time() - t_start) * 1000),
            }

        # Step 2: Build context
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Passage {i+1}] (chunk_id: {chunk['chunk_id']})\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        # Step 3: Generate answer
        from backend.core.gemini_utils import generate_content_with_retry
        model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config=genai.GenerationConfig(temperature=0.0),
            system_instruction=RAG_SYSTEM_PROMPT,
        )
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = generate_content_with_retry(model, prompt)
        answer = response.text.strip()

        tokens_used = max(1, len(prompt + answer) // 4)
        sources = [c["chunk_id"] for c in chunks]

    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        answer = "NO_EVIDENCE_RETRIEVED"
        sources = []
        tokens_used = 0

    return {
        "answer": answer,
        "sources": sources,
        "tokens_used": tokens_used,
        "latency_ms": int((time.time() - t_start) * 1000),
    }
