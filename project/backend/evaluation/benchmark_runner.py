"""
BENCHMARK mode runner — runs all 100 public questions × 3 pipelines.
Per spec §1.3 and §9.1: all three run independently, no shared state.
"""

import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)


def run_benchmark(questions_path: str, output_path: str = "./logs/benchmark_results.jsonl",
                  limit: int = None, model_config: dict = None) -> list[dict]:
    """
    Run BENCHMARK mode on the 100 public questions.
    Per spec §9.1: 100 × 3 = 300 pipeline executions, sequentially.

    Args:
        questions_path: Path to eval_public.jsonl
        output_path: Where to write benchmark_results.jsonl
        limit: Optional limit for testing (None = all 100)
        model_config: Must match across all 3 pipelines (§12.3)

    Returns:
        List of benchmark result records
    """
    from backend.pipelines import rag_pipeline, graphrag_pipeline, agentic_pipeline
    from backend.evaluation.evaluator import llm_judge, bertscore_f1, validate_baseline_fairness

    # Validate baseline fairness before running
    if model_config:
        validate_baseline_fairness(model_config, model_config, model_config)

    # Load questions
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if limit:
        questions = questions[:limit]

    logger.info(f"Running BENCHMARK on {len(questions)} questions × 3 pipelines")
    results = []

    for i, q_data in enumerate(questions):
        question_id = q_data["qid"]
        question = q_data["question"]
        gold_answers = q_data.get("answer", [])
        reference = gold_answers[0] if gold_answers else ""

        logger.info(f"[{i+1}/{len(questions)}] Processing {question_id}: {question[:60]}...")

        record = {"question_id": question_id, "question": question,
                  "qtype": q_data.get("qtype", "unknown")}

        # ── Pipeline A: RAG ─────────────────────────────────────────
        logger.debug(f"  Running RAG pipeline...")
        rag_result = rag_pipeline.run(question, question_id)
        rag_acc = llm_judge(question, gold_answers, rag_result["answer"])
        rag_bs = bertscore_f1(rag_result["answer"], reference) if reference else 0.0
        record["rag"] = {
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "tokens_used": rag_result["tokens_used"],
            "latency_ms": rag_result["latency_ms"],
            "accuracy_score": rag_acc,
            "bertscore_f1": round(rag_bs, 4),
        }

        # ── Pipeline B: GraphRAG ─────────────────────────────────────
        logger.debug(f"  Running GraphRAG pipeline...")
        grag_result = graphrag_pipeline.run(question, question_id)
        grag_acc = llm_judge(question, gold_answers, grag_result["answer"])
        grag_bs = bertscore_f1(grag_result["answer"], reference) if reference else 0.0
        record["graphrag"] = {
            "answer": grag_result["answer"],
            "sources": grag_result["sources"],
            "tokens_used": grag_result["tokens_used"],
            "latency_ms": grag_result["latency_ms"],
            "accuracy_score": grag_acc,
            "bertscore_f1": round(grag_bs, 4),
        }

        # ── Pipeline C: Agentic ──────────────────────────────────────
        logger.debug(f"  Running Agentic pipeline...")
        ag_result = agentic_pipeline.run(question, question_id)
        ag_acc = llm_judge(question, gold_answers, ag_result["answer"])
        ag_bs = bertscore_f1(ag_result["answer"], reference) if reference else 0.0
        record["agentic"] = {
            "answer": ag_result["answer"],
            "exit_type": ag_result.get("exit_type", "ABSTAIN"),
            "sources": ag_result.get("sources", []),
            "tokens_used": ag_result.get("tokens_used", 0),
            "latency_ms": ag_result.get("latency_ms", 0),
            "accuracy_score": ag_acc,
            "bertscore_f1": round(ag_bs, 4),
            "n_chunks_retrieved": ag_result.get("n_chunks_retrieved", 0),
            "n_sources_cited": ag_result.get("n_sources_cited", 0),
            "strategy_changed": ag_result.get("strategy_changed", False),
        }

        results.append(record)

        # Write incrementally (fail-safe)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"  RAG: {rag_acc} | GraphRAG: {grag_acc} | Agentic: {ag_acc}")

    logger.info(f"BENCHMARK complete. Results written to {output_path}")
    _print_summary(results)
    return results


def build_confusion_matrix(benchmark_results: list[dict],
                            thresholds: dict) -> dict:
    """
    Per spec §9.6: Ladder routing accuracy confusion matrix.
    ground_truth_rung = cheapest rung that produced PASS.
    """
    from backend.core.ladder import route

    matrix = {gt: {pred: 0 for pred in ["RAG", "GraphRAG", "Agentic"]}
              for gt in ["RAG", "GraphRAG", "Agentic", "none_passed"]}

    for record in benchmark_results:
        question = record.get("question", "")
        question_id = record.get("question_id", "")

        # Ground truth: cheapest PASS
        if record["rag"]["accuracy_score"] == "PASS":
            gt_rung = "RAG"
        elif record["graphrag"]["accuracy_score"] == "PASS":
            gt_rung = "GraphRAG"
        elif record["agentic"]["accuracy_score"] == "PASS":
            gt_rung = "Agentic"
        else:
            gt_rung = "none_passed"

        # Predicted rung
        pred_rung = route(question, question_id, thresholds)
        matrix[gt_rung][pred_rung] = matrix[gt_rung].get(pred_rung, 0) + 1

    return matrix


def _print_summary(results: list[dict]) -> None:
    rag_pass = sum(1 for r in results if r["rag"]["accuracy_score"] == "PASS")
    grag_pass = sum(1 for r in results if r["graphrag"]["accuracy_score"] == "PASS")
    ag_pass = sum(1 for r in results if r["agentic"]["accuracy_score"] == "PASS")
    n = len(results)

    avg_rag_tok = sum(r["rag"]["tokens_used"] for r in results) / max(n, 1)
    avg_grag_tok = sum(r["graphrag"]["tokens_used"] for r in results) / max(n, 1)
    avg_ag_tok = sum(r["agentic"]["tokens_used"] for r in results) / max(n, 1)

    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY ({n} questions)")
    print(f"{'='*60}")
    print(f"{'Pipeline':<15} {'Accuracy':>10} {'Avg Tokens':>12}")
    print(f"{'-'*40}")
    print(f"{'RAG':<15} {rag_pass/n*100:>9.1f}% {avg_rag_tok:>12.0f}")
    print(f"{'GraphRAG':<15} {grag_pass/n*100:>9.1f}% {avg_grag_tok:>12.0f}")
    print(f"{'Agentic':<15} {ag_pass/n*100:>9.1f}% {avg_ag_tok:>12.0f}")
    print(f"{'='*60}\n")
