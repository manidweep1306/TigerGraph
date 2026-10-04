import sys
sys.path.insert(0, r"d:\TigerGraph\project")

from backend.db.vector_index import get_vector_index
vi = get_vector_index()

query = "Who won the gold medal in the men's 100m sprint in the 2012 Summer Olympics?"
chunks = vi.search(query, top_k=2)
print("Query:", query)
print("Found chunks:", len(chunks))
for i, c in enumerate(chunks):
    print(f"\n[Chunk {i+1}] ID: {c.get('chunk_id')} | Score: {c.get('score', 0):.4f}")
    print(c.get('text', '')[:200] + "...")
