import numpy as np
import re
import unicodedata
from backend.db.vector_index import get_vector_index, _ensure_chunk_cache
from backend.db.tigergraph_client import get_tg_client

v = get_vector_index()
v.preload_cache()

from backend.db import vector_index

query = "How many nations competed in Fencing at the 1988 Summer Olympics – Men's foil?"

def clean_tokens(text):
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", text) if len(w) > 1]

q_tokens = set(clean_tokens(query))
print("Query tokens:", q_tokens)

q_emb = v.embed_text(query, task_type="RETRIEVAL_QUERY")
q_vec = np.array(q_emb, dtype=np.float32)
q_vec = q_vec / np.linalg.norm(q_vec)
vec_scores = vector_index.CHUNK_EMBEDDINGS_MATRIX @ q_vec

hybrid_scores = []
for i, meta in enumerate(vector_index.CHUNK_METADATA):
    t_tokens = set(clean_tokens(meta["text"]))
    overlap = len(q_tokens & t_tokens)
    year_match = any(y in q_tokens and y in t_tokens for y in ["1988", "1992", "1994", "1996", "1998", "2000", "2002", "2004", "2006", "2008", "2010", "2012", "2014", "2016", "2018", "2020"])
    infobox_match = 0.3 if "[infobox" in meta["text"].lower() else 0.0
    lex_score = (overlap / max(1, len(q_tokens))) + (0.5 if year_match else 0.0) + infobox_match
    total_score = float(vec_scores[i]) + 0.8 * lex_score
    hybrid_scores.append((total_score, meta))

hybrid_scores.sort(key=lambda x: x[0], reverse=True)
for s, meta in hybrid_scores[:5]:
    print(f"Score: {s:.3f} | Chunk: {meta['chunk_id']} | Text: {meta['text'][:150]}")
