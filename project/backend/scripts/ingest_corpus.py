"""
Corpus ingestion script — parses corpus.jsonl, chunks documents,
generates embeddings, extracts entities, and loads into TigerGraph.
"""

import json
import logging
import os
import re
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = int(os.environ.get("CHUNK_SIZE_TOKENS", 512))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP_TOKENS", 64))
EMBED_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", 20))
INGEST_BATCH = int(os.environ.get("INGEST_BATCH_SIZE", 50))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    approx_tokens: int


def tokenize_approx(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 chars."""
    return max(1, len(text) // 4)


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    """
    Split document text into overlapping chunks of ~CHUNK_SIZE_TOKENS tokens.
    Simple word-boundary splitting.
    """
    words = text.split()
    chunks = []
    chunk_idx = 0
    words_per_chunk = CHUNK_SIZE_TOKENS  # approx, 1 word ≈ 1 token for simple text
    step = words_per_chunk - CHUNK_OVERLAP

    i = 0
    while i < len(words):
        chunk_words = words[i:i + words_per_chunk]
        chunk_text = " ".join(chunk_words)
        chunk_tokens = tokenize_approx(chunk_text)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_chunk{chunk_idx}",
            doc_id=doc_id,
            chunk_index=chunk_idx,
            text=chunk_text,
            approx_tokens=chunk_tokens,
        ))
        chunk_idx += 1
        i += step
        if i >= len(words):
            break

    return chunks if chunks else [Chunk(
        chunk_id=f"{doc_id}_chunk0",
        doc_id=doc_id,
        chunk_index=0,
        text=text[:2000],
        approx_tokens=tokenize_approx(text[:2000]),
    )]


def extract_entities_from_infobox(text: str, doc_id: str) -> list[dict]:
    """
    Extract structured entities from Wikipedia Olympic infobox format.
    The corpus has structured [Infobox Olympic event] sections.
    """
    entities = []

    # Extract year and season from doc title pattern
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    season_match = re.search(r'(Summer|Winter)', text, re.I)

    year = int(year_match.group()) if year_match else 0
    season = season_match.group().capitalize() if season_match else "Summer"

    if year:
        games_id = f"{year}_{season}_Olympics"
        entities.append({
            "type": "OlympicGames",
            "id": games_id,
            "year": year,
            "season": season,
            "label": f"{year} {season} Olympics",
        })

    # Extract medal winners from infobox
    for medal in ["gold", "silver", "bronze"]:
        pattern = rf'{medal}\s*:\s*([^\n]+)'
        matches = re.findall(pattern, text, re.I)
        for match in matches:
            name = match.strip()
            if name and len(name) > 2 and len(name) < 100:
                # Normalize entity ID
                entity_id = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())[:50]
                entities.append({
                    "type": "Entity",
                    "id": f"athlete_{entity_id}",
                    "name": name,
                    "entity_type": "ATHLETE",
                    "normalized_name": name.lower(),
                    "medal_type": medal,
                    "doc_id": doc_id,
                    "year": year,
                    "season": season,
                })

    # Extract competitor count
    comp_match = re.search(r'competitors\s*:\s*(\d+)', text, re.I)
    if comp_match:
        entities.append({
            "type": "metadata",
            "key": "competitors",
            "value": int(comp_match.group(1)),
            "doc_id": doc_id,
        })

    # Extract nation count
    nation_match = re.search(r'nations\s*:\s*(\d+)', text, re.I)
    if nation_match:
        entities.append({
            "type": "metadata",
            "key": "nations",
            "value": int(nation_match.group(1)),
            "doc_id": doc_id,
        })

    return entities


def ingest_corpus(corpus_path: str, dry_run: bool = False, limit: int = None) -> None:
    """
    Main ingestion pipeline:
    1. Parse corpus.jsonl
    2. Chunk documents
    3. Extract entities from infoboxes
    4. Generate embeddings (Gemini text-embedding-004)
    5. Load into TigerGraph (graph vertices + edges + vector embeddings)
    """
    if not dry_run:
        from backend.db.tigergraph_client import get_tg_client
        from backend.db.vector_index import get_vector_index
        tg = get_tg_client()
        vi = get_vector_index()
    else:
        tg = None
        vi = None

    logger.info(f"Starting ingestion from {corpus_path}")
    if dry_run:
        logger.info("DRY RUN MODE — no data will be written to TigerGraph")

    # Load corpus
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
            if limit and len(documents) >= limit:
                break

    logger.info(f"Loaded {len(documents)} documents")

    # Track OlympicGames vertices to upsert once
    games_seen: dict[str, dict] = {}
    all_chunks: list[Chunk] = []
    all_entities: list[dict] = []
    doc_vertex_batch: list[tuple] = []

    for i, doc in enumerate(documents):
        doc_id = doc["doc_id"]
        text = doc.get("text", "")
        title = doc.get("title", "")

        # Upsert Document vertex
        doc_vertex_batch.append((doc_id, {
            "title": title,
            "url": doc.get("url", ""),
            "wikidata_qid": doc.get("wikidata_qid", ""),
            "wikipedia_pageid": doc.get("wikipedia_pageid", 0),
            "approx_tokens": doc.get("approx_tokens", tokenize_approx(text)),
            "text": text[:5000],  # store first 5k chars in graph
        }))

        # Chunk the document
        chunks = chunk_document(doc_id, text)
        all_chunks.extend(chunks)

        # Extract entities
        entities = extract_entities_from_infobox(text, doc_id)
        all_entities.extend(entities)

        # Track OlympicGames
        for e in entities:
            if e["type"] == "OlympicGames":
                games_seen[e["id"]] = e

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i+1}/{len(documents)} documents, "
                        f"{len(all_chunks)} chunks so far")

    logger.info(f"Total: {len(documents)} docs, {len(all_chunks)} chunks, "
                f"{len(games_seen)} OlympicGames, {len(all_entities)} entity records")

    if dry_run:
        logger.info("DRY RUN: Would ingest the above counts. Exiting.")
        return

    # ─── Phase 1: Load graph vertices ─────────────────────────────────
    logger.info("Phase 1: Loading Document vertices...")
    for batch_start in range(0, len(doc_vertex_batch), INGEST_BATCH):
        batch = doc_vertex_batch[batch_start:batch_start + INGEST_BATCH]
        tg.upsert_vertices("Document", batch)
        logger.info(f"  Documents: {batch_start + len(batch)}/{len(doc_vertex_batch)}")

    logger.info("Phase 2: Loading OlympicGames vertices...")
    games_batch = [
        (gid, {"year": g["year"], "season": g["season"],
               "label": g["label"], "host_city": ""})
        for gid, g in games_seen.items()
    ]
    if games_batch:
        tg.upsert_vertices("OlympicGames", games_batch)

    logger.info("Phase 3: Loading Entity vertices and edges...")
    entity_batch = []
    medal_edges = []
    doc_entity_edges = []

    for e in all_entities:
        if e["type"] == "Entity" and e.get("id"):
            entity_batch.append((e["id"], {
                "name": e["name"],
                "entity_type": e["entity_type"],
                "normalized_name": e.get("normalized_name", ""),
                "aliases": "",
            }))

            # Medal edge
            if e.get("medal_type") and e.get("year"):
                games_id = f"{e['year']}_{e['season']}_Olympics"
                medal_edges.append((e["id"], games_id, {
                    "medal_type": e["medal_type"],
                    "event_name": "",
                    "athlete_country": "",
                }))

            # Document → Entity edge
            doc_entity_edges.append((e["doc_id"], e["id"], {"mention_count": 1}))

    if entity_batch:
        for b in range(0, len(entity_batch), INGEST_BATCH):
            tg.upsert_vertices("Entity", entity_batch[b:b+INGEST_BATCH])

    if medal_edges:
        tg.upsert_edges("Entity", "WON_MEDAL", "OlympicGames", medal_edges)

    if doc_entity_edges:
        for b in range(0, len(doc_entity_edges), INGEST_BATCH * 2):
            tg.upsert_edges("Document", "MENTIONS_ENTITY", "Entity",
                            doc_entity_edges[b:b+INGEST_BATCH*2])

    # ─── Phase 2: Load chunks + generate embeddings ────────────────────
    logger.info(f"Phase 4: Loading {len(all_chunks)} Chunk vertices...")
    chunk_vertex_batch = [
        (c.chunk_id, {
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "approx_tokens": c.approx_tokens,
        })
        for c in all_chunks
    ]

    for b in range(0, len(chunk_vertex_batch), INGEST_BATCH):
        tg.upsert_vertices("Chunk", chunk_vertex_batch[b:b+INGEST_BATCH])

    # Load HAS_CHUNK edges
    chunk_edges = [(c.doc_id, c.chunk_id, {"chunk_index": c.chunk_index})
                   for c in all_chunks]
    for b in range(0, len(chunk_edges), INGEST_BATCH * 2):
        tg.upsert_edges("Document", "HAS_CHUNK", "Chunk", chunk_edges[b:b+INGEST_BATCH*2])

    logger.info("Phase 5: Generating embeddings and loading into Vector DB...")
    texts = [c.text for c in all_chunks]
    embeddings = vi.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

    for b in range(0, len(all_chunks), INGEST_BATCH):
        batch_chunks = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
             "chunk_index": c.chunk_index, "text": c.text,
             "approx_tokens": c.approx_tokens}
            for c in all_chunks[b:b+INGEST_BATCH]
        ]
        batch_embeddings = embeddings[b:b+INGEST_BATCH]
        vi.upsert_chunks_with_embeddings(batch_chunks, batch_embeddings)
        logger.info(f"  Embeddings: {b + len(batch_chunks)}/{len(all_chunks)}")

    # Print stats
    stats = tg.graph_stats()
    logger.info(f"\nIngestion complete! Graph stats: {stats}")


if __name__ == "__main__":
    default_path = os.environ.get("CORPUS_PATH", "data/corpus/corpus.jsonl")
    parser.add_argument("--corpus", default=default_path)
    parser.add_argument("--limit", type=int, default=None, help="Limit docs for testing")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't write")
    args = parser.parse_args()

    corpus_path = args.corpus
    candidates = [
        corpus_path,
        Path(corpus_path),
        Path(".") / corpus_path.replace("../", ""),
        Path("..") / corpus_path,
        Path(__file__).parent.parent.parent.parent / corpus_path.replace("../", ""),
        Path(__file__).parent.parent.parent / corpus_path.replace("../", ""),
        Path(__file__).parent.parent.parent.parent / "data" / "corpus" / Path(corpus_path).name,
        Path(__file__).parent.parent.parent / "data" / "corpus" / Path(corpus_path).name,
    ]
    for c in candidates:
        if c and Path(c).exists():
            corpus_path = str(Path(c).resolve())
            break

    ingest_corpus(corpus_path, dry_run=args.dry_run, limit=args.limit)
