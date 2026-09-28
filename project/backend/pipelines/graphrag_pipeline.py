"""
Pipeline B: GraphRAG
Per spec §2.2 — combines TigerGraph GSQL traversal with vector search.
Single-shot, stateless (not agentic).

Per Model Mapping:
Uses Groq llama-3.3-70b-versatile for response synthesis and Gemini embedding-001 for vector search.
"""

import time
import logging

from backend.config.unified_config import config
from backend.core.llm_client import get_groq_client

logger = logging.getLogger(__name__)

GRAPHRAG_SYSTEM_PROMPT = """You are a factual question-answering assistant specializing in Olympic sports history.
You have access to both structured graph data (entity relationships, medal records) and text passages.
Answer the question using the provided graph context AND text passages.
If the entity could not be found in the graph, use the text passages.
If the provided context does not contain sufficient facts to answer the question, state:
"INSUFFICIENT_EVIDENCE: The provided graph and text context do not contain enough specific information to answer this question."
Do NOT say "NO_ENTITY_MATCH" when graph or text context has been provided.
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

        # Step 1: Entity linking (uses fast model)
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

        # Step 5: Generate answer using Groq llama-3.3-70b-versatile
        groq = get_groq_client()
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        res = groq.call_complex_agent(
            prompt=prompt,
            system_instruction=GRAPHRAG_SYSTEM_PROMPT,
            temperature=0.0
        )
        answer = res.text.strip()
        tokens_used += res.tokens_used

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
