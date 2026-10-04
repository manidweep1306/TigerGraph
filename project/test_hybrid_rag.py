import numpy as np
import re
import math
import unicodedata
from collections import Counter
from backend.db.vector_index import get_vector_index
from backend.db import vector_index
from backend.core.llm_client import get_groq_client
from backend.evaluation.evaluator import llm_judge

v = get_vector_index()
v.preload_cache()

def clean_tokens(text):
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", text) if len(w) > 1]

corpus_tokens = [clean_tokens(meta["text"]) for meta in vector_index.CHUNK_METADATA]
N = len(corpus_tokens)
avgdl = sum(len(doc) for doc in corpus_tokens) / max(1, N)
df = Counter()
for doc in corpus_tokens:
    for term in set(doc):
        df[term] += 1
idf = {term: math.log(1.0 + (N - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

def hybrid_search(query, top_k=5, k1=1.5, b=0.75, alpha=0.6):
    q_tokens = clean_tokens(query)
    bm25_scores = np.zeros(N, dtype=np.float32)
    for q_term in q_tokens:
        if q_term not in idf:
            continue
        term_idf = idf[q_term]
        for i, doc in enumerate(corpus_tokens):
            if q_term in doc:
                tf = doc.count(q_term)
                dl = len(doc)
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (dl / avgdl))
                bm25_scores[i] += term_idf * (numerator / denominator)
    
    # Normalize BM25 scores
    max_bm25 = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1.0
    norm_bm25 = bm25_scores / max_bm25

    # Dense vector scores
    q_emb = v.embed_text(query, task_type="RETRIEVAL_QUERY")
    q_vec = np.array(q_emb, dtype=np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)
    vec_scores = vector_index.CHUNK_EMBEDDINGS_MATRIX @ q_vec
    norm_vec = (vec_scores + 1.0) / 2.0  # map [-1, 1] to [0, 1]

    # Combine
    combined = (1 - alpha) * norm_vec + alpha * norm_bm25
    top_indices = np.argsort(-combined)[:top_k]
    return [vector_index.CHUNK_METADATA[idx] for idx in top_indices]

# Test Q2
q = "How many nations competed in Fencing at the 1988 Summer Olympics – Men's foil?"
gold = ["29"]
chunks = hybrid_search(q, top_k=5)
context = "\n\n".join([f"[Passage {i+1}] (chunk_id: {c['chunk_id']})\n{c['text']}" for i, c in enumerate(chunks)])
groq = get_groq_client()
res = groq.call_complex_agent(
    prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer concisely stating the number immediately:",
    system_instruction="You are a factual Olympic QA assistant. Answer concisely and accurately using the context.",
    temperature=0.0
)
print("Answer:", res.text)
print("Judge:", llm_judge(q, gold, res.text))
