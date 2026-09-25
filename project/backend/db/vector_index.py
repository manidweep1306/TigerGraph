"""
TigerGraph Vector DB client — semantic search over corpus chunks.
Uses the TigerGraph Vector DB API for embedding storage and retrieval.
"""

import os
import json
import logging
import time
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 768


class VectorIndex:
    """
    TigerGraph Vector DB client for semantic search.

    TigerGraph Vector DB is accessed via the REST++ API.
    Embeddings are stored as FLOAT LIST attributes on Chunk vertices.
    """

    def __init__(self, tg_client=None):
        from backend.db.tigergraph_client import get_tg_client
        self._tg = tg_client or get_tg_client()
        self._graph_name = os.environ["TG_GRAPH_NAME"]

    # ─── Embedding Generation ──────────────────────────────────────────

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Generate embedding for a single text using Gemini embedding model."""
        from backend.core.gemini_utils import embed_content_with_retry
        result = embed_content_with_retry(
            model=f"models/{EMBEDDING_MODEL}",
            content=text,
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        )
        return result["embedding"]

    def embed_batch(self, texts: list[str],
                    task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """Batch embed multiple texts with rate-limit retry."""
        embeddings = []
        for i, text in enumerate(texts):
            try:
                emb = self.embed_text(text, task_type=task_type)
                embeddings.append(emb)
                if (i + 1) % 20 == 0:
                    logger.info(f"Embedded {i+1}/{len(texts)} texts")
                    time.sleep(0.5)  # gentle rate limiting
            except Exception as e:
                logger.warning(f"Embedding failed for text {i}: {e}. Using zeros.")
                embeddings.append([0.0] * EMBEDDING_DIM)
        return embeddings

    # ─── Semantic Search ───────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5,
               score_threshold: float = 0.0) -> list[dict]:
        """
        Semantic similarity search over corpus chunks.
        Returns list of {chunk_id, text, score, doc_id} dicts.
        """
        # Generate query embedding
        query_embedding = self.embed_text(query, task_type="RETRIEVAL_QUERY")

        # Use TigerGraph Vector Search REST API
        conn = self._tg._get_conn()
        try:
            # TigerGraph Vector DB search endpoint
            results = conn.runInstalledQuery(
                "vector_search_chunks",
                params={
                    "query_vector": query_embedding,
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                }
            )
            if results and len(results) > 0:
                chunks = results[0].get("chunks", [])
                return [
                    {
                        "chunk_id": c.get("v_id", ""),
                        "text": c.get("attributes", {}).get("text", ""),
                        "score": c.get("attributes", {}).get("score", 0.0),
                        "doc_id": c.get("attributes", {}).get("doc_id", ""),
                    }
                    for c in chunks
                    if c.get("attributes", {}).get("score", 0.0) >= score_threshold
                ]
        except Exception as e:
            logger.warning(f"Vector search via installed query failed: {e}. "
                           f"Falling back to cosine similarity.")
            return self._fallback_search(query_embedding, top_k, score_threshold)

        return []

    def _fallback_search(self, query_embedding: list[float], top_k: int,
                          score_threshold: float) -> list[dict]:
        """
        Fallback: fetch all chunk embeddings and compute cosine similarity locally.
        Used when the vector search query isn't installed yet.
        """
        import numpy as np

        conn = self._tg._get_conn()
        try:
            all_chunks = conn.getVertices("Chunk", limit=5000)
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

        q = np.array(query_embedding)
        q_norm = np.linalg.norm(q)

        scored = []
        for chunk in all_chunks:
            attrs = chunk.get("attributes", {})
            emb = attrs.get("embedding", [])
            if not emb or len(emb) != EMBEDDING_DIM:
                continue
            v = np.array(emb)
            score = float(np.dot(q, v) / (q_norm * np.linalg.norm(v) + 1e-9))
            if score >= score_threshold:
                scored.append({
                    "chunk_id": chunk.get("v_id", ""),
                    "text": attrs.get("text", ""),
                    "score": score,
                    "doc_id": attrs.get("doc_id", ""),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ─── Upsert Chunk with Embedding ──────────────────────────────────

    def upsert_chunk_embedding(self, chunk_id: str, embedding: list[float]) -> None:
        """Store an embedding on an existing Chunk vertex."""
        conn = self._tg._get_conn()
        conn.upsertVertex(
            vertexType="Chunk",
            vertexId=chunk_id,
            attributes={"embedding": embedding},
        )

    def upsert_chunks_with_embeddings(self,
                                       chunks: list[dict],
                                       embeddings: list[list[float]]) -> int:
        """Batch upsert chunks with pre-computed embeddings."""
        assert len(chunks) == len(embeddings)
        conn = self._tg._get_conn()

        vertex_data = []
        for chunk, emb in zip(chunks, embeddings):
            vertex_data.append((
                chunk["chunk_id"],
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "approx_tokens": chunk.get("approx_tokens", 0),
                    "embedding": emb,
                }
            ))

        result = conn.upsertVertices("Chunk", vertex_data)
        return result


# Module-level singleton
_vector_index: Optional[VectorIndex] = None


def get_vector_index() -> VectorIndex:
    global _vector_index
    if _vector_index is None:
        _vector_index = VectorIndex()
    return _vector_index
