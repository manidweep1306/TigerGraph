import asyncio
import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from backend.config.unified_config import config

load_dotenv()
logger = logging.getLogger(__name__)
LOG_DIR = Path(config.paths.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)


async def _process_question(
    q_data: dict,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    loop: asyncio.AbstractEventLoop,
    output_path: str,
    new_results: list,
    progress_counter: list,
    total_count: int,
):
    from backend.pipelines import rag_pipeline, graphrag_pipeline, agentic_pipeline
    from backend.evaluation.evaluator import llm_judge, bertscore_f1

    question_id = q_data["qid"]
    question = q_data["question"]
    gold_answers = q_data.get("answer", [])
    reference = gold_answers[0] if gold_answers else ""

    async with semaphore:
        record = {
            "question_id": question_id,
            "question": question,
            "qtype": q_data.get("qtype", "unknown")
        }

        # ── Pipeline A (RAG) & Pipeline B (GraphRAG) concurrently ────────
        rag_task = loop.run_in_executor(None, rag_pipeline.run, question, question_id)
        grag_task = loop.run_in_executor(None, graphrag_pipeline.run, question, question_id)
        rag_result, grag_result = await asyncio.gather(rag_task, grag_task)

        # ── Pipeline C (Agentic) ─────────────────────────────────────────
        ag_result = await loop.run_in_executor(None, agentic_pipeline.run, question, question_id)

        # ── LLM Judge & Metric evaluations concurrently ─────────────────
        judge_rag_task = loop.run_in_executor(None, llm_judge, question, gold_answers, rag_result["answer"])
        judge_grag_task = loop.run_in_executor(None, llm_judge, question, gold_answers, grag_result["answer"])
        judge_ag_task = loop.run_in_executor(None, llm_judge, question, gold_answers, ag_result["answer"])
        rag_acc, grag_acc, ag_acc = await asyncio.gather(judge_rag_task, judge_grag_task, judge_ag_task)

        rag_bs = bertscore_f1(rag_result["answer"], reference) if reference else 0.0
        grag_bs = bertscore_f1(grag_result["answer"], reference) if reference else 0.0
        ag_bs = bertscore_f1(ag_result["answer"], reference) if reference else 0.0

        record["rag"] = {
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "tokens_used": rag_result["tokens_used"],
            "latency_ms": rag_result["latency_ms"],
            "accuracy_score": rag_acc,
            "bertscore_f1": round(rag_bs, 4),
        }
        record["graphrag"] = {
            "answer": grag_result["answer"],
            "sources": grag_result["sources"],
            "tokens_used": grag_result["tokens_used"],
            "latency_ms": grag_result["latency_ms"],
            "accuracy_score": grag_acc,
            "bertscore_f1": round(grag_bs, 4),
        }
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

        # Thread-safe append to output file
        async with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            new_results.append(record)
            progress_counter[0] += 1
            idx = progress_counter[0]

        logger.info(
            f"[{idx}/{total_count}] Done {question_id} | "
            f"RAG: {rag_acc} | GraphRAG: {grag_acc} | Agentic: {ag_acc}"
        )


async def _async_run_benchmark(
    questions_to_process: list[dict],
    output_path: str,
    existing_records: list[dict],
    total_count: int,
) -> list[dict]:
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(5)
    write_lock = asyncio.Lock()
    new_results = []
    progress_counter = [len(existing_records)]

    tasks = [
        _process_question(
            q_data=q,
            semaphore=semaphore,
            write_lock=write_lock,
            loop=loop,
            output_path=output_path,
            new_results=new_results,
            progress_counter=progress_counter,
            total_count=total_count,
        )
        for q in questions_to_process
    ]

    if tasks:
        await asyncio.gather(*tasks)

    return existing_records + new_results


def run_benchmark(questions_path: str, output_path: Optional[str] = None,
                  limit: int = None, model_config: dict = None) -> list[dict]:
    """
    Run BENCHMARK mode on evaluation questions with async parallelism,
    multi-key rotator, RAM vector pre-indexing, and automatic resumption.
    """
    output_path = output_path or str(LOG_DIR / "benchmark_results.jsonl")
    from backend.evaluation.evaluator import validate_baseline_fairness
    from backend.db.vector_index import get_vector_index

    if model_config:
        validate_baseline_fairness(model_config, model_config, model_config)

    # 1. Load all questions
    all_questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_questions.append(json.loads(line))

    if limit:
        all_questions = all_questions[:limit]

    total_target = len(all_questions)

    # 2. Check for previously evaluated questions to auto-resume
    existing_records = []
    existing_qids = set()
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        existing_records.append(rec)
                        if "question_id" in rec:
                            existing_qids.add(rec["question_id"])
                    except Exception:
                        pass

    questions_to_process = [q for q in all_questions if q.get("qid") not in existing_qids]
    logger.info(
        f"Benchmark status: {len(existing_qids)}/{total_target} questions already completed. "
        f"Processing remaining {len(questions_to_process)} questions..."
    )

    if not questions_to_process:
        logger.info(f"All {total_target} benchmark questions are already evaluated in {output_path}")
        _print_summary(existing_records)
        return existing_records

    # 3. Pre-load chunk index and pre-compute query embeddings in batch
    try:
        v_index = get_vector_index()
        v_index.preload_cache()
        q_texts = [q["question"] for q in questions_to_process]
        logger.info(f"Pre-computing embeddings in batch for {len(q_texts)} questions...")
        v_index.embed_batch(q_texts, task_type="RETRIEVAL_QUERY")
        logger.info("Batch embedding precomputation complete.")
    except Exception as e:
        logger.warning(f"Batch embedding precomputation encountered notice: {e}")

    # 4. Execute async parallel benchmark runner
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                results = pool.submit(
                    asyncio.run,
                    _async_run_benchmark(questions_to_process, output_path, existing_records, total_target)
                ).result()
        else:
            results = loop.run_until_complete(
                _async_run_benchmark(questions_to_process, output_path, existing_records, total_target)
            )
    except Exception as e:
        logger.error(f"Benchmark run error: {e}")
        raise

    logger.info(f"BENCHMARK complete. Total evaluated records: {len(results)}. Results at: {output_path}")
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
    rag_pass = sum(1 for r in results if r.get("rag", {}).get("accuracy_score") == "PASS")
    grag_pass = sum(1 for r in results if r.get("graphrag", {}).get("accuracy_score") == "PASS")
    ag_pass = sum(1 for r in results if r.get("agentic", {}).get("accuracy_score") == "PASS")
    n = len(results)

    avg_rag_tok = sum(r.get("rag", {}).get("tokens_used", 0) for r in results) / max(n, 1)
    avg_grag_tok = sum(r.get("graphrag", {}).get("tokens_used", 0) for r in results) / max(n, 1)
    avg_ag_tok = sum(r.get("agentic", {}).get("tokens_used", 0) for r in results) / max(n, 1)

    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY ({n} questions)")
    print(f"{'='*60}")
    print(f"{'Pipeline':<15} {'Accuracy':>10} {'Avg Tokens':>12}")
    print(f"{'-'*40}")
    print(f"{'RAG':<15} {rag_pass/n*100:>9.1f}% {avg_rag_tok:>12.0f}")
    print(f"{'GraphRAG':<15} {grag_pass/n*100:>9.1f}% {avg_grag_tok:>12.0f}")
    print(f"{'Agentic':<15} {ag_pass/n*100:>9.1f}% {avg_ag_tok:>12.0f}")
    print(f"{'='*60}\n")
