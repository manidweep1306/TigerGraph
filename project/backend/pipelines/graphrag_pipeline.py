"""
Pipeline B: GraphRAG
Per spec §2.2 — combines TigerGraph GSQL traversal with vector search.
Single-shot, stateless (not agentic).
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

GRAPHRAG_SYSTEM_PROMPT = """You are a factual question-answering assistant specializing in Olympic sports history.
You have access to both structured graph data (entity relationships, medal records) and text passages.
Answer the question using the provided graph context AND text passages.
If the entity could not be found in the graph, use ONLY the text passages.
If neither provides sufficient information, say "NO_ENTITY_MATCH".
Be specific, cite sources, and use the graph structure to resolve entity relationships."""


def run(question: str, question_id: str) -> dict:
    """
    GraphRAG Pipeline.
    Input: {question: str}
    Output: {answer: str, sources: [entity_id|chunk_id], tokens_used: int, latency_ms: int}
    """
    t_start = time.time()
    sources = []
    tokens_used = 0

    try:
        from backend.db.tigergraph_client import get_tg_client
        from backend.db.vector_index import get_vector_index
        from backend.agents import entity_linker

        tg = get_tg_client()
        vector_index = get_vector_index()

        # Step 1: Entity linking
        link_result = entity_linker.run(
            {"query_string": question, "graph": "tg_ref"},
            question_id, step_id=0
        )
        tokens_used += link_result.get("tokens_used", 0)

        entity_id = link_result.get("entity_id")
        graph_context = ""

        # Step 2: Graph traversal (if entity found)
        if entity_id and link_result.get("match_confidence", 0) >= 0.5:
            sources.append(entity_id)

            # Multi-hop traversal
            traversal = tg.multi_hop_traversal(entity_id, max_hops=2, max_results=15)
            if traversal and traversal[0].get("entities"):
                entity_names = [
                    e.get("attributes", {}).get("name", "")
                    for e in traversal[0]["entities"][:10]
                ]
                graph_context = f"Graph entities related to query: {', '.join(entity_names)}\n\n"

            # Community context
            community = tg.entity_community_context(entity_id)
            if community and community[0].get("games"):
                games = [
                    f"{g.get('attributes', {}).get('year', '')} {g.get('attributes', {}).get('season', '')} Olympics"
                    for g in community[0]["games"][:5]
                ]
                graph_context += f"Olympic Games participated in: {', '.join(games)}\n\n"

            # Get docs for this entity
            entity_docs = tg.docs_by_entity(entity_id, top_k=3)
            if entity_docs:
                doc_ids = [d.get("v_id", "") for d in entity_docs]
                sources.extend(doc_ids)

            # Temporal: check adjacent Olympics if question mentions "before" or "after"
            q_lower = question.lower()
            if any(w in q_lower for w in ["before", "after", "previous", "next", "immediately"]):
                for year_str in ["2016", "2012", "2008", "2004", "2000", "2020", "2018", "2014"]:
                    if year_str in question:
                        year = int(year_str)
                        season = "Winter" if "winter" in q_lower else "Summer"
                        adj = tg.adjacent_olympics(year, season)
                        if adj:
                            years = [g.get("attributes", {}).get("year", 0) for g in adj]
                            years = sorted(years)
                            idx = years.index(year) if year in years else -1
                            if "before" in q_lower and idx > 0:
                                graph_context += f"Previous {season} Olympics: {years[idx-1]}\n"
                            elif "after" in q_lower and idx < len(years) - 1:
                                graph_context += f"Next {season} Olympics: {years[idx+1]}\n"
                        break

        else:
            graph_context = "Note: No specific entity was found in the graph for this query.\n\n"

        # Step 3: Vector search
        chunks = vector_index.search(question, top_k=5)
        tokens_used += max(1, len(question) // 4)

        # Step 4: Build combined context
        context_parts = []
        if graph_context:
            context_parts.append(f"[Graph Context]\n{graph_context}")

        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Text Passage {i+1}]\n{chunk['text']}")
            sources.append(chunk["chunk_id"])

        if not context_parts:
            return {
                "answer": "NO_ENTITY_MATCH",
                "sources": [],
                "tokens_used": tokens_used,
                "latency_ms": int((time.time() - t_start) * 1000),
            }

        context = "\n\n".join(context_parts)

        # Step 5: Generate answer
        model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config=genai.GenerationConfig(temperature=0.0),
            system_instruction=GRAPHRAG_SYSTEM_PROMPT,
        )
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = model.generate_content(prompt)
        answer = response.text.strip()
        tokens_used += max(1, len(prompt + answer) // 4)

    except Exception as e:
        logger.error(f"GraphRAG pipeline error: {e}")
        answer = "NO_ENTITY_MATCH"
        sources = []

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources))[:20],
        "tokens_used": tokens_used,
        "latency_ms": int((time.time() - t_start) * 1000),
    }
