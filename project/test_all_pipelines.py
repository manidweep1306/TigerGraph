import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.main import query_all_pipelines, QueryRequest
import asyncio

async def main():
    req = QueryRequest(question="What is the total revenue of the company?", pipeline="all")
    try:
        print("Running query through all pipelines...")
        res = await query_all_pipelines(req)
        print("--- RAG Result ---")
        print(res.get("results", {}).get("rag", {}).get("answer", "NO RAG ANSWER"))
        print(res.get("results", {}).get("rag", {}).keys())

        print("\n--- GraphRAG Result ---")
        print(res.get("results", {}).get("graphrag", {}).get("answer", "NO GRAPHRAG ANSWER"))
        print(res.get("results", {}).get("graphrag", {}).keys())

        print("\n--- Agentic Result ---")
        agentic_res = res.get("results", {}).get("agentic", {})
        print(agentic_res.get("answer", "NO AGENTIC ANSWER"))
        print(agentic_res.keys())
        print(f"Tokens Used: {agentic_res.get('tokens_used')}")
        print(f"Reasoning Steps: {agentic_res.get('step_count')}")
        print(f"Latency: {agentic_res.get('latency_ms')} ms")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
