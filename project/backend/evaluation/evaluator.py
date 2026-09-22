"""
Evaluation: LLM-as-judge + BERTScore
Per spec §9.4 — accuracy and BERTScore F1 per question per pipeline.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

JUDGE_PROMPT = """You are a strict factual accuracy judge for an Olympic sports question-answering system.

Given:
1. A question
2. The correct reference answer(s)
3. A candidate answer

Determine if the candidate answer is CORRECT.

Rules:
- PASS: The candidate answer contains the correct answer, even if phrased differently or with extra context.
- FAIL: The candidate answer is wrong, incomplete (missing key numbers/names), or says it doesn't know.
- For numeric answers: PASS only if the exact number matches.
- For name answers: PASS if the name is recognizably the same (spelling variants allowed).
- Partial credit: if the question asks for a count and the answer gives the right count, PASS.
- "NO_EVIDENCE_RETRIEVED" or "NO_ENTITY_MATCH" always = FAIL.

Output ONLY: PASS or FAIL"""


def llm_judge(question: str, reference_answers: list[str],
              candidate_answer: str) -> str:
    """LLM-as-judge accuracy assessment. Returns 'PASS' or 'FAIL'."""
    if not candidate_answer or candidate_answer in (
            "NO_EVIDENCE_RETRIEVED", "NO_ENTITY_MATCH", ""):
        return "FAIL"

    ref_str = " / ".join(reference_answers)

    try:
        model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config=genai.GenerationConfig(temperature=0.0),
            system_instruction=JUDGE_PROMPT,
        )
        response = model.generate_content(
            f"Question: {question}\n\nReference answer(s): {ref_str}\n\n"
            f"Candidate answer: {candidate_answer}\n\nPASS or FAIL?"
        )
        result = response.text.strip().upper()
        return "PASS" if result.startswith("PASS") else "FAIL"
    except Exception as e:
        logger.error(f"LLM judge error: {e}")
        # Fallback: simple string match
        for ref in reference_answers:
            if ref.lower() in candidate_answer.lower():
                return "PASS"
        return "FAIL"


def bertscore_f1(candidate: str, reference: str) -> float:
    """Compute BERTScore F1 between candidate and reference."""
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            [candidate], [reference],
            lang="en", verbose=False,
            model_type="distilbert-base-uncased",  # lightweight for hackathon speed
        )
        return float(F1[0])
    except ImportError:
        # bert_score not installed — use simple token overlap as fallback
        return _token_f1(candidate, reference)
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}. Using token F1 fallback.")
        return _token_f1(candidate, reference)


def _token_f1(candidate: str, reference: str) -> float:
    """Simple token-level F1 as BERTScore fallback."""
    c_tokens = set(candidate.lower().split())
    r_tokens = set(reference.lower().split())
    if not c_tokens or not r_tokens:
        return 0.0
    overlap = c_tokens & r_tokens
    precision = len(overlap) / len(c_tokens)
    recall = len(overlap) / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def validate_baseline_fairness(rag_cfg: dict, graphrag_cfg: dict,
                                 agentic_cfg: dict) -> bool:
    """
    Per spec §12.3 — must return True before benchmark results are published.
    Raises ConfigMismatchError if any field differs.
    """
    fields = ["model", "temperature", "embedding_model", "chunk_size",
              "corpus_version", "eval_rubric_version"]

    for f in fields:
        if not (rag_cfg.get(f) == graphrag_cfg.get(f) == agentic_cfg.get(f)):
            raise ValueError(
                f"Baseline fairness FAIL: Field '{f}' differs across pipelines: "
                f"RAG={rag_cfg.get(f)}, GraphRAG={graphrag_cfg.get(f)}, "
                f"Agentic={agentic_cfg.get(f)}"
            )

    logger.info("Baseline fairness check: PASSED")
    return True
