"""
TigerGraph client wrapper — handles all graph DB operations.
Wraps pyTigerGraph with connection pooling and query execution.
"""

import os
import json
import logging
from typing import Any, Optional
from functools import lru_cache

import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class TigerGraphClient:
    """
    Singleton-style client for TigerGraph Savanna.
    Lazy-connects on first use.
    """

    def __init__(self):
        self._conn: Optional[tg.TigerGraphConnection] = None

    def _get_conn(self) -> tg.TigerGraphConnection:
        if self._conn is None:
            host = os.environ["TG_HOST"]
            username = os.environ["TG_USERNAME"]
            password = os.environ["TG_PASSWORD"]
            graph_name = os.environ["TG_GRAPH_NAME"]
            secret = os.environ.get("TG_SECRET", "")

            self._conn = tg.TigerGraphConnection(
                host=host,
                username=username,
                password=password,
                graphname=graph_name,
                useCert=True,
            )
            if secret:
                token = self._conn.getToken(secret, setToken=True)
                logger.info(f"TigerGraph token acquired, expires: {token[2]}")
            logger.info(f"Connected to TigerGraph at {host}, graph: {graph_name}")
        return self._conn

    # ─── Schema Setup ─────────────────────────────────────────────────

    def install_schema(self, gsql_path: str) -> str:
        """Execute the schema.gsql DDL file via GSQL."""
        conn = self._get_conn()
        with open(gsql_path, "r") as f:
            gsql = f.read()
        result = conn.gsql(gsql)
        logger.info(f"Schema installed: {result[:200]}")
        return result

    def install_queries(self) -> None:
        """Install and publish all GSQL queries."""
        conn = self._get_conn()
        conn.gsql("USE GRAPH " + os.environ["TG_GRAPH_NAME"])
        result = conn.gsql("INSTALL QUERY ALL")
        logger.info(f"Queries installed: {result[:200]}")

    # ─── Vertex / Edge Upsert ─────────────────────────────────────────

    def upsert_vertices(self, vertex_type: str, vertices: list[dict]) -> int:
        """Batch upsert vertices. Returns count inserted/updated."""
        conn = self._get_conn()
        result = conn.upsertVertices(vertexType=vertex_type, vertices=vertices)
        return result

    def upsert_edges(self, src_type: str, edge_type: str, tgt_type: str,
                     edges: list[tuple]) -> int:
        """Batch upsert edges. edges = [(src_id, tgt_id, {attrs}), ...]"""
        conn = self._get_conn()
        result = conn.upsertEdges(
            sourceVertexType=src_type,
            edgeType=edge_type,
            targetVertexType=tgt_type,
            edges=edges,
        )
        return result

    # ─── GSQL Query Runners ───────────────────────────────────────────

    def run_query(self, query_name: str, params: dict = None) -> dict:
        """Run an installed GSQL query and return results."""
        conn = self._get_conn()
        params = params or {}
        try:
            results = conn.runInstalledQuery(query_name, params=params)
            return results
        except Exception as e:
            logger.error(f"Query {query_name} failed: {e}")
            raise

    def multi_hop_traversal(self, entity_id: str, max_hops: int = 2,
                            max_results: int = 20) -> dict:
        return self.run_query("multi_hop_traversal", {
            "start_entity": entity_id,
            "max_hops": max_hops,
            "max_results": max_results,
        })

    def entity_lookup_by_name(self, name_query: str, top_k: int = 5) -> list[dict]:
        result = self.run_query("entity_lookup_by_name", {
            "name_query": name_query,
            "top_k": top_k,
        })
        if result and "matched" in result[0]:
            return result[0]["matched"]
        return []

    def docs_by_entity(self, entity_id: str, top_k: int = 10) -> list[dict]:
        result = self.run_query("docs_by_entity", {
            "entity_v": entity_id,
            "top_k": top_k,
        })
        if result and "docs" in result[0]:
            return result[0]["docs"]
        return []

    def events_at_games(self, year: int, season: str = "Summer") -> dict:
        return self.run_query("events_at_games", {"year": year, "season": season})

    def gold_medalists(self, event_pattern: str, year: int = 0) -> list[dict]:
        result = self.run_query("gold_medalists", {
            "event_pattern": event_pattern,
            "year": year,
        })
        if result and "athletes" in result[0]:
            return result[0]["athletes"]
        return []

    def count_events_with_competitors(self, year: int, season: str = "Summer",
                                       min_competitors: int = 0,
                                       sport_filter: str = "") -> dict:
        return self.run_query("count_events_with_competitors", {
            "year": year,
            "season": season,
            "min_competitors": min_competitors,
            "sport_filter": sport_filter,
        })

    def adjacent_olympics(self, year: int, season: str = "Summer") -> list[dict]:
        result = self.run_query("adjacent_olympics", {"year": year, "season": season})
        if result and "all_games" in result[0]:
            return result[0]["all_games"]
        return []

    def entity_community_context(self, entity_id: str) -> dict:
        return self.run_query("entity_community_context", {"entity_v": entity_id})

    def get_doc_chunks(self, doc_id: str) -> list[dict]:
        result = self.run_query("get_doc_chunks", {"doc_v": doc_id})
        if result and "chunks" in result[0]:
            return result[0]["chunks"]
        return []

    def graph_stats(self) -> dict:
        result = self.run_query("graph_stats")
        if result:
            return result[0]
        return {}

    # ─── Direct vertex fetch ──────────────────────────────────────────

    def get_vertex(self, vertex_type: str, vertex_id: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            return conn.getVerticesById(vertex_type, [vertex_id])
        except Exception:
            return None

    def get_entity_by_id(self, entity_id: str) -> Optional[dict]:
        result = self.get_vertex("Entity", entity_id)
        if result:
            return result[0] if isinstance(result, list) else result
        return None


# Module-level singleton
_client: Optional[TigerGraphClient] = None


def get_tg_client() -> TigerGraphClient:
    global _client
    if _client is None:
        _client = TigerGraphClient()
    return _client
