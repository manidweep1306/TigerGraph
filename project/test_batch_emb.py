from backend.core.llm_client import get_embedding_client
import google.generativeai as genai

client = get_embedding_client()
client._ensure_configured()
res = genai.embed_content(
    model="models/gemini-embedding-001",
    content=["test 1", "test 2"],
    task_type="RETRIEVAL_DOCUMENT",
    output_dimensionality=768
)
print("Type of res['embedding']:", type(res['embedding']))
print("Length:", len(res['embedding']))
print("Type of first element:", type(res['embedding'][0]))
if isinstance(res['embedding'][0], list):
    print("Length of first element:", len(res['embedding'][0]))
