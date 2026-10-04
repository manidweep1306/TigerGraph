"""
EntityLinkerAgent — resolves query entity strings to graph node IDs.
Per spec §3.1: invocable when any EMPTY or SUPPORTED slot references an unlinked entity.

Optimized with:
1. Olympic Game reference alias canonicalization (Summer/Winter, relative temporal "immediately before/after")
2. Venue lookup cleaning (lowercase, strip punctuation)
3. In-memory vertex name search and RAM vector search fallback (eliminating NO_ENTITY_MATCH)
"""

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.config.unified_config import config
from backend.core.llm_client import get_groq_client

logger = logging.getLogger(__name__)

LOG_DIR = Path(config.paths.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUMMER_OLYMPICS_YEARS = [
    1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936,
    1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984,
    1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024
]
WINTER_OLYMPICS_YEARS = [
    1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968,
    1972, 1976, 1980, 1984, 1988, 1992, 1994, 1998, 2002, 2006,
    2010, 2014, 2018, 2022
]

# RAM entity cache for ultra-fast lookup & vector fallback
_RAM_ENTITY_CACHE: Optional[list[dict]] = None
_CACHE_LOCK = threading.Lock()


def canonicalize_olympic_games(text: str) -> Optional[dict]:
    """
    Normalize Olympic game references:
    - 'YYYY Summer/Winter Olympics'
    - 'YYYY Summer' / 'YYYY Winter'
    - 'Summer/Winter Olympics held immediately before YYYY'
    - 'Summer/Winter Olympics held immediately after YYYY'
    into canonical IDs (e.g. '2012_Summer_Olympics', '2018_Winter_Olympics').
    """
    text_lower = text.lower()

    # Relative temporal pattern: e.g. "Summer Olympics held immediately before 2016"
    rel_match = re.search(
        r'(summer|winter)\s*olympics(?:\s*held)?\s*immediately\s*(before|after)\s*(\d{4})',
        text_lower
    )
    if rel_match:
        season = rel_match.group(1).capitalize()
        rel_type = rel_match.group(2)
        base_year = int(rel_match.group(3))
        years_list = SUMMER_OLYMPICS_YEARS if season == "Summer" else WINTER_OLYMPICS_YEARS

        if base_year in years_list:
            idx = years_list.index(base_year)
            target_year = None
            if rel_type == "before" and idx > 0:
                target_year = years_list[idx - 1]
            elif rel_type == "after" and idx < len(years_list) - 1:
                target_year = years_list[idx + 1]

            if target_year:
                cid = f"{target_year}_{season}_Olympics"
                return {
                    "canonical_id": cid,
                    "year": target_year,
                    "season": season,
                    "confidence": 1.0,
                    "source": "relative_temporal"
                }

    # Direct pattern: "2018 Winter Olympics", "2004 Summer", "1988 Summer"
    direct_match = re.search(r'\b(19\d{2}|20\d{2})\s*(summer|winter)(?:\s*olympics)?\b', text_lower)
    if direct_match:
        year = int(direct_match.group(1))
        season = direct_match.group(2).capitalize()
        cid = f"{year}_{season}_Olympics"
        return {
            "canonical_id": cid,
            "year": year,
            "season": season,
            "confidence": 1.0,
            "source": "direct_year_season"
        }

    return None


def clean_venue_name(name: str) -> str:
    """Lowercase and strip punctuation before querying TigerGraph vertices."""
    cleaned = re.sub(r'[^\w\s]', '', name).strip().lower()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def _ensure_entity_cache(tg) -> list[dict]:
    """Pre-index Entity vertices in RAM for instantaneous fallback matching."""
    global _RAM_ENTITY_CACHE
    if _RAM_ENTITY_CACHE is not None:
        return _RAM_ENTITY_CACHE

    with _CACHE_LOCK:
        if _RAM_ENTITY_CACHE is not None:
            return _RAM_ENTITY_CACHE
        try:
            conn = tg._get_conn()
            raw = conn.getVertices("Entity", limit=10000)
            _RAM_ENTITY_CACHE = [
                {
                    "entity_id": e.get("v_id", ""),
                    "name": e.get("attributes", {}).get("name", ""),
                    "normalized_name": (e.get("attributes", {}).get("normalized_name") or e.get("attributes", {}).get("name", "")).lower(),
                    "entity_type": e.get("attributes", {}).get("entity_type", "ATHLETE"),
                }
                for e in raw
            ]
            logger.info(f"EntityLinker cached {len(_RAM_ENTITY_CACHE)} entities in RAM.")
        except Exception as e:
            logger.warning(f"Could not pre-load Entity cache: {e}")
            _RAM_ENTITY_CACHE = []
        return _RAM_ENTITY_CACHE


def search_entities_in_memory(query_text: str, tg) -> list[tuple[str, float]]:
    """Search cached entity vertices in RAM with lexical and token matching."""
    cache = _ensure_entity_cache(tg)
    q_norm = clean_venue_name(query_text)
    if not q_norm:
        return []
    q_words = set(q_norm.split())
    matches = []

    for ent in cache:
        name_norm = clean_venue_name(ent["normalized_name"])
        if q_norm == name_norm:
            matches.append((ent["entity_id"], 1.0))
            continue
        if q_norm in name_norm or name_norm in q_norm:
            matches.append((ent["entity_id"], 0.85))
            continue
        ent_words = set(name_norm.split())
        overlap = len(q_words & ent_words)
        if overlap > 0:
            score = overlap / max(len(q_words), len(ent_words))
            if score >= 0.5:
                matches.append((ent["entity_id"], 0.70 * score))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:5]


ENTITY_LINK_PROMPT = """You are an entity extraction and normalization expert for Olympic sports data.

Given a question or text, extract the key named entities and normalize them:
- Athlete names (normalize to common English spellings)
- Olympic Games (e.g., "2018 Winter Olympics", "2008 Summer Olympics")
- Sports/Events (e.g., "men's 100m sprint", "women's figure skating", "biathlon")
- Countries/NOC codes (e.g., "United States" → "USA", "Great Britain" → "GBR")
- Venues (e.g., "Olympic Weightlifting Gymnasium")

OUTPUT FORMAT (JSON only):
{
  "entities": [
    {
      "original_text": "...",
      "normalized_name": "...",
      "entity_type": "ATHLETE|EVENT|COUNTRY|VENUE|GAMES|SPORT",
      "search_variants": ["variant1", "variant2"]
    }
  ]
}"""


def run(input_data: dict, question_id: str, step_id: int) -> dict:
    """
    EntityLinkerAgent execution with Olympic Canonicalization, Clean Venue Matching,
    and RAM Vector Search Fallback.
    """
    t_start = time.time()
    query_string = input_data.get("query_string", "")
    tokens_used = 0

    best_entity_id = None
    best_confidence = 0.0
    all_candidates = []
    extracted_entities = []

    try:
        from backend.db.tigergraph_client import get_tg_client
        tg = get_tg_client()

        # Step 0: Olympic Game Reference Canonicalization
        canon_games = canonicalize_olympic_games(query_string)
        if canon_games:
            best_entity_id = canon_games["canonical_id"]
            best_confidence = canon_games["confidence"]
            all_candidates.append(best_entity_id)
            extracted_entities.append({
                "original_text": query_string,
                "normalized_name": canon_games["canonical_id"],
                "entity_type": "GAMES",
                "search_variants": [canon_games["canonical_id"], f"{canon_games['year']} {canon_games['season']} Olympics"]
            })

        # Step 1: Extract entities using Fast LLM
        groq = get_groq_client()
        extracted, response = groq.generate_json(
            prompt=f"Extract entities from: {query_string}",
            system_instruction=ENTITY_LINK_PROMPT,
            use_fast_model=True
        )
        tokens_used += response.tokens_used
        llm_entities = extracted.get("entities", [])
        extracted_entities.extend(llm_entities)

        # Step 2: Lookup each entity in TigerGraph & RAM
        for entity_info in llm_entities:
            etype = entity_info.get("entity_type", "")
            orig_name = entity_info.get("normalized_name", "")
            variants = entity_info.get("search_variants", [orig_name])

            # For venue / games lookups, clean and strip punctuation
            is_venue_or_games = etype in ("VENUE", "GAMES") or "gymnasium" in orig_name.lower() or "stadium" in orig_name.lower()
            cleaned_variants = [clean_venue_name(v) for v in variants if v]
            search_pool = list(dict.fromkeys(variants + cleaned_variants))

            entity_matched = False
            for variant in search_pool:
                if not variant:
                    continue

                # 2a. Check if variant matches Olympic Games canonicalizer
                game_check = canonicalize_olympic_games(variant)
                if game_check:
                    all_candidates.append(game_check["canonical_id"])
                    if game_check["confidence"] > best_confidence:
                        best_confidence = game_check["confidence"]
                        best_entity_id = game_check["canonical_id"]
                    entity_matched = True

                # 2b. Exact lookup via TG installed query
                matches = tg.entity_lookup_by_name(variant, top_k=3)
                for match in matches:
                    attrs = match.get("attributes", {})
                    entity_id = match.get("v_id", "")
                    all_candidates.append(entity_id)

                    stored_name = attrs.get("name", "").lower()
                    if stored_name == variant.lower():
                        conf = 1.0
                    elif variant.lower() in stored_name or stored_name in variant.lower():
                        conf = 0.80
                    else:
                        conf = 0.60

                    if conf > best_confidence:
                        best_confidence = conf
                        best_entity_id = entity_id
                    entity_matched = True

                # 2c. If exact match fails, use RAM in-memory search over vertex names
                if not entity_matched:
                    ram_matches = search_entities_in_memory(variant, tg)
                    for eid, conf in ram_matches:
                        all_candidates.append(eid)
                        if conf > best_confidence:
                            best_confidence = conf
                            best_entity_id = eid
                            entity_matched = True

        # Step 3: RAM vector search fallback over chunks/entities if still no match
        if best_entity_id is None:
            try:
                from backend.db.vector_index import get_vector_index
                vi = get_vector_index()
                v_results = vi.search(query_string, top_k=2)
                for vr in v_results:
                    cid = vr.get("chunk_id", "")
                    doc_id = vr.get("doc_id", "")
                    if doc_id:
                        all_candidates.append(doc_id)
                        if best_entity_id is None:
                            best_entity_id = doc_id
                            best_confidence = 0.70
            except Exception as e:
                logger.debug(f"Vector search fallback in entity_linker: {e}")

        output = {
            "entity_id": best_entity_id,
            "match_confidence": best_confidence,
            "candidates": list(dict.fromkeys(all_candidates))[:5],
            "extracted_entities": extracted_entities,
        }

    except Exception as e:
        logger.error(f"EntityLinkerAgent error: {e}")
        output = {"entity_id": None, "match_confidence": 0.0, "candidates": []}

    output["tokens_used"] = tokens_used
    _log_invocation("EntityLinkerAgent", question_id, step_id,
                    query_string[:200], tokens_used, t_start,
                    f"entity_id={output.get('entity_id')}, conf={output.get('match_confidence', 0):.2f}")
    return output


def _log_invocation(agent_name: str, question_id: str, step_id: int,
                     input_summary: str, tokens_used: int, t_start: float,
                     output_summary: str) -> None:
    """Per spec §3.3 — mandatory invocation log."""
    record = {
        "agent_name": agent_name,
        "question_id": question_id,
        "step_id": step_id,
        "input_summary": input_summary[:300],
        "tokens_used": tokens_used,
        "latency_ms": int((time.time() - t_start) * 1000),
        "output_summary": output_summary[:300],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_path = LOG_DIR / "agent_invocations.jsonl"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to log agent invocation: {e}")
