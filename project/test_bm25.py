import numpy as np
import re
import math
import unicodedata
from collections import Counter
from backend.db.vector_index import get_vector_index, _ensure_chunk_cache
from backend.db import vector_index

v = get_vector_index()
v.preload_cache()

def clean_tokens(text):
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", text) if len(w) > 1]

# Build in-memory BM25 index over CHUNK_METADATA
corpus_tokens = [clean_tokens(meta["text"]) for meta in vector_index.CHUNK_METADATA]
N = len(corpus_tokens)
avgdl = sum(len(doc) for doc in corpus_tokens) / max(1, N)

df = Counter()
for doc in corpus_tokens:
    for term in set(doc):
        df[term] += 1

idf = {}
for term, freq in df.items():
    idf[term] = math.log(1.0 + (N - freq + 0.5) / (freq + 0.5))

def bm25_search(query, top_k=10, k1=1.5, b=0.75):
    q_tokens = clean_tokens(query)
    scores = np.zeros(N, dtype=np.float32)
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
                scores[i] += term_idf * (numerator / denominator)
    
    top_indices = np.argsort(-scores)[:top_k]
    return [(scores[idx], vector_index.CHUNK_METADATA[idx]) for idx in top_indices]

query = "How many nations competed in Fencing at the 1988 Summer Olympics – Men's foil?"
print(f"BM25 Search for: {query}")
results = bm25_search(query, top_k=5)
for s, meta in results:
    print(f"Score: {s:.2f} | Chunk: {meta['chunk_id']} | Text: {meta['text'][:140]}")
