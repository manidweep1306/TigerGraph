import os
from backend.core.llm_client import get_embedding_client

client = get_embedding_client()
print("Gemini API Key defined?", bool(client.api_key))

try:
    emb = client.embed_content("test")
    print("Embedding generated, length:", len(emb))
    print("Sample:", emb[:5])
except Exception as e:
    print(f"Failed to generate embedding: {e}")
