"""
TigerGraph Vector DB client — semantic search over corpus chunks.
Uses the TigerGraph Vector DB API for embedding storage and retrieval.
Vector Embeddings: Gemini gemini-embedding-001 (768-dim Vectors).
"""

import logging
import time
import threading
from pathlib import Path
from typing import Optional
import numpy as np

from backend.config.unified_config import config
from backend.core.llm_client import get_embedding_client

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = config.gemini.embedding_model
EMBEDDING_DIM = config.gemini.embedding_dimension

# Module-level in-memory cache for Chunk embeddings and metadata
CHUNK_EMBEDDINGS_MATRIX: Optional[np.ndarray] = None  # shape (N, D), float32, normalized
CHUNK_METADATA: list[dict] = []  # list of {chunk_id, text, doc_id}
QUERY_EMBEDDINGS_CACHE: dict[str, list[float]] = {}  # query string -> embedding
_CACHE_LOCK = threading.Lock()
_DISK_CACHE_PATH = Path(config.paths.config_dir).parent / "data" / "chunk_embeddings_cache.npz"

# BM25 in-memory index
CHUNK_BM25_TOKENS: list[list[str]] = []
CHUNK_BM25_IDF: dict[str, float] = {}
CHUNK_BM25_AVGDL: float = 0.0


def _clean_bm25_tokens(text: str) -> list[str]:
    import unicodedata, re
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", text) if len(w) > 1]


def _build_bm25_index() -> None:
    global CHUNK_BM25_TOKENS, CHUNK_BM25_IDF, CHUNK_BM25_AVGDL
    from collections import Counter
    import math
    if not CHUNK_METADATA:
        return
    CHUNK_BM25_TOKENS = [_clean_bm25_tokens(m["text"]) for m in CHUNK_METADATA]
    N = len(CHUNK_BM25_TOKENS)
    CHUNK_BM25_AVGDL = sum(len(doc) for doc in CHUNK_BM25_TOKENS) / max(1, N)
    df = Counter()
    for doc in CHUNK_BM25_TOKENS:
        for term in set(doc):
            df[term] += 1
    CHUNK_BM25_IDF = {
        term: math.log(1.0 + (N - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }


def _ensure_chunk_cache(tg_conn=None) -> None:
    """Load all chunk embeddings into RAM once (disk cache or TigerGraph)."""
    global CHUNK_EMBEDDINGS_MATRIX, CHUNK_METADATA
    if CHUNK_EMBEDDINGS_MATRIX is not None and len(CHUNK_METADATA) > 0:
        if not CHUNK_BM25_TOKENS:
            _build_bm25_index()
        return

    with _CACHE_LOCK:
        if CHUNK_EMBEDDINGS_MATRIX is not None and len(CHUNK_METADATA) > 0:
            if not CHUNK_BM25_TOKENS:
                _build_bm25_index()
            return

        # 1. Try disk cache first for instant restart (<5ms)
        if _DISK_CACHE_PATH.exists():
            try:
                npz = np.load(str(_DISK_CACHE_PATH), allow_pickle=True)
                matrix = npz["matrix"].astype(np.float32)
                metadata = list(npz["metadata"])
                if len(matrix) > 0 and len(metadata) == len(matrix):
                    CHUNK_EMBEDDINGS_MATRIX = matrix
                    CHUNK_METADATA = metadata
                    logger.info(f"Loaded {len(metadata)} chunk embeddings from disk cache: {_DISK_CACHE_PATH}")
                    _build_bm25_index()
                    return
            except Exception as e:
                logger.warning(f"Could not load chunk cache from {_DISK_CACHE_PATH}: {e}")

        # 2. Fetch from TigerGraph if connection available
        if tg_conn is not None:
            try:
                logger.info("Pre-indexing: Fetching Chunk vertices from TigerGraph into RAM...")
                all_chunks = tg_conn.getVertices("Chunk", limit=25000)
                embs = []
                meta = []
                for chunk in all_chunks:
                    attrs = chunk.get("attributes", {})
                    emb = attrs.get("embedding", [])
                    if emb and len(emb) == EMBEDDING_DIM:
                        embs.append(emb)
                        meta.append({
                            "chunk_id": chunk.get("v_id", ""),
                            "text": attrs.get("text", ""),
                            "doc_id": attrs.get("doc_id", ""),
                        })
                if embs:
                    arr = np.array(embs, dtype=np.float32)
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms[norms == 0] = 1e-9
                    CHUNK_EMBEDDINGS_MATRIX = arr / norms
                    CHUNK_METADATA = meta
                    logger.info(f"Pre-indexed {len(meta)} chunk embeddings in memory.")
                    _build_bm25_index()

                    try:
                        _DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            str(_DISK_CACHE_PATH),
                            matrix=CHUNK_EMBEDDINGS_MATRIX,
                            metadata=np.array(CHUNK_METADATA, dtype=object),
                        )
                        logger.info(f"Saved chunk embeddings cache to {_DISK_CACHE_PATH}")
                    except Exception as e:
                        logger.warning(f"Could not save chunk cache to disk: {e}")
                    return
            except Exception as e:
                logger.warning(f"Failed to fetch chunks from TigerGraph for cache: {e}")


class VectorIndex:
    """
    TigerGraph Vector DB client with In-Memory Pre-Indexed Semantic Search.
    Embeddings are normalized and searched via vectorized matrix dot product (<2ms).
    """

    def __init__(self, tg_client=None):
        from backend.db.tigergraph_client import get_tg_client
        self._tg = tg_client or get_tg_client()
        self._graph_name = config.tigergraph.graph_name
        self._embed_client = get_embedding_client()

    # ─── Embedding Generation ──────────────────────────────────────────

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Generate embedding with memory caching."""
        cache_key = f"{task_type}:{text}"
        if cache_key in QUERY_EMBEDDINGS_CACHE:
            return QUERY_EMBEDDINGS_CACHE[cache_key]
        emb = self._embed_client.embed_content(text, task_type=task_type)
        QUERY_EMBEDDINGS_CACHE[cache_key] = emb
        return emb

    def embed_batch(self, texts: list[str],
                    task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """Batch embed multiple texts with caching."""
        uncached_indices = []
        uncached_texts = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            cache_key = f"{task_type}:{text}"
            if cache_key in QUERY_EMBEDDINGS_CACHE:
                results[i] = QUERY_EMBEDDINGS_CACHE[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_embs = self._embed_client.embed_batch(uncached_texts, task_type=task_type)
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embs):
                cache_key = f"{task_type}:{text}"
                QUERY_EMBEDDINGS_CACHE[cache_key] = emb
                results[idx] = emb

        return results

    def preload_cache(self) -> int:
        """Ensure in-memory vector cache is populated."""
        conn = self._tg._get_conn()
        _ensure_chunk_cache(conn)
        return len(CHUNK_METADATA) if CHUNK_METADATA else 0

    # ─── Semantic Search (Hybrid BM25 + Dense Vectors) ─────────────────

    def search(self, query: str, top_k: int = 5,
               score_threshold: float = 0.0, alpha: float = 0.6) -> list[dict]:
        """
        In-memory hybrid semantic search (BM25 + Dense Vectors < 2ms).
        alpha is weight of BM25 lexical matching (0.6 default for exact fact retrieval).
        """
        global CHUNK_EMBEDDINGS_MATRIX, CHUNK_METADATA, CHUNK_BM25_TOKENS, CHUNK_BM25_IDF, CHUNK_BM25_AVGDL

        # Ensure in-memory cache is active (only connect to TG if cache is empty)
        if CHUNK_EMBEDDINGS_MATRIX is None or len(CHUNK_METADATA) == 0:
            try:
                conn = self._tg._get_conn()
                _ensure_chunk_cache(conn)
            except Exception:
                conn = None
                _ensure_chunk_cache(None)
        else:
            conn = None

        if CHUNK_EMBEDDINGS_MATRIX is not None and len(CHUNK_METADATA) > 0:
            n = len(CHUNK_METADATA)
            if n == 0:
                return []

            # 1. BM25 Lexical Scoring
            q_tokens = _clean_bm25_tokens(query)
            bm25_scores = np.zeros(n, dtype=np.float32)
            k1 = 1.5
            b = 0.75
            for q_term in q_tokens:
                if q_term in CHUNK_BM25_IDF:
                    term_idf = CHUNK_BM25_IDF[q_term]
                    for i, doc in enumerate(CHUNK_BM25_TOKENS):
                        if q_term in doc:
                            tf = doc.count(q_term)
                            dl = len(doc)
                            numerator = tf * (k1 + 1.0)
                            denominator = tf + k1 * (1.0 - b + b * (dl / (CHUNK_BM25_AVGDL or 1.0)))
                            bm25_scores[i] += term_idf * (numerator / denominator)

            max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) > 0 else 0.0
            norm_bm25 = (bm25_scores / max_bm25) if max_bm25 > 0 else bm25_scores

            # 2. Dense Vector Scoring
            try:
                query_embedding = self.embed_text(query, task_type="RETRIEVAL_QUERY")
                q = np.array(query_embedding, dtype=np.float32)
                q_norm = float(np.linalg.norm(q))
                if q_norm > 0:
                    q = q / q_norm
                    vec_scores = CHUNK_EMBEDDINGS_MATRIX @ q
                    norm_vec = np.clip((vec_scores + 1.0) / 2.0, 0.0, 1.0)
                else:
                    norm_vec = np.zeros(n, dtype=np.float32)
                    alpha = 1.0
            except Exception as e:
                logger.warning(f"Dense embedding generation failed ({e}), relying on BM25.")
                norm_vec = np.zeros(n, dtype=np.float32)
                alpha = 1.0

            # 3. Hybrid Combination
            combined_scores = (1.0 - alpha) * norm_vec + alpha * norm_bm25

            k = min(top_k, n)
            if n <= top_k:
                top_indices = np.argsort(-combined_scores)
            else:
                partitioned = np.argpartition(-combined_scores, k)[:k]
                top_indices = partitioned[np.argsort(-combined_scores[partitioned])]

            results = []
            for idx in top_indices:
                score = float(combined_scores[idx])
                if score >= score_threshold:
                    meta = CHUNK_METADATA[idx]
                    results.append({
                        "chunk_id": meta["chunk_id"],
                        "text": meta["text"],
                        "score": score,
                        "doc_id": meta.get("doc_id", ""),
                    })
            return results

        # Fallback to TigerGraph installed query if cache was completely empty
        query_embedding = self.embed_text(query, task_type="RETRIEVAL_QUERY")
        if conn is not None:
            try:
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
                logger.warning(f"Fallback installed query search failed: {e}")

        return []

    # ─── Upsert Chunk with Embedding ──────────────────────────────────

    def upsert_chunk_embedding(self, chunk_id: str, embedding: list[float]) -> None:
        """Store an embedding on an existing Chunk vertex and update memory."""
        global CHUNK_EMBEDDINGS_MATRIX, CHUNK_METADATA
        conn = self._tg._get_conn()
        conn.upsertVertex(
            vertexType="Chunk",
            vertexId=chunk_id,
            attributes={"embedding": embedding},
        )
        # Invalidate cache so it reloads on next query
        with _CACHE_LOCK:
            CHUNK_EMBEDDINGS_MATRIX = None
            CHUNK_METADATA = []

    def upsert_chunks_with_embeddings(self,
                                       chunks: list[dict],
                                       embeddings: list[list[float]]) -> int:
        """Batch upsert chunks with pre-computed embeddings."""
        global CHUNK_EMBEDDINGS_MATRIX, CHUNK_METADATA
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
        # Invalidate cache to refresh
        with _CACHE_LOCK:
            CHUNK_EMBEDDINGS_MATRIX = None
            CHUNK_METADATA = []
        return result


# Module-level singleton
_vector_index: Optional[VectorIndex] = None


def get_vector_index() -> VectorIndex:
    global _vector_index
    if _vector_index is None:
        _vector_index = VectorIndex()
    return _vector_index
