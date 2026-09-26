"""
ADAPTIVE mode runner — runs 50 hidden questions through the Ladder.
Per spec §1.3 and §9.2: one rung per question, at most one escalation.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)


def run_adaptive(questions_path: str,
                 output_path: str = "./logs/adaptive_results.jsonl",
                 limit: int = None) -> list[dict]:
    """
    Run ADAPTIVE mode on the 50 hidden questions.
    Per spec §9.2: 50–100 pipeline executions depending on escalations.
    """
    import json as _json
    from backend.core import ladder as ladder_module
    from backend.pipelines import rag_pipeline, graphrag_pipeline, agentic_pipeline

    # Load thresholds
    cfg_path = Path("./config/frozen_thresholds.json")
    if not cfg_path.exists():
        cfg_path = Path(__file__).parent.parent.parent / "config" / "frozen_thresholds.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    # Load questions
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if limit:
        questions = questions[:limit]

    logger.info(f"Running ADAPTIVE on {len(questions)} questions")
    results = []

    for i, q_data in enumerate(questions):
        question_id = q_data["qid"]
        question = q_data["question"]

        logger.info(f"[{i+1}/{len(questions)}] {question_id}: {question[:60]}...")

        rung_path = []
        final_result = None
        tokens_total = 0
        latency_total = 0

        # Step 1: Route
        rung = ladder_module.route(question, question_id, thresholds)
        rung_path.append(rung)

        # Step 2: Run selected rung
        if rung == "RAG":
            result = rag_pipeline.run(question, question_id)
        elif rung == "GraphRAG":
            result = graphrag_pipeline.run(question, question_id)
        else:
            result = agentic_pipeline.run(question, question_id)

        tokens_total += result.get("tokens_used", 0)
        latency_total += result.get("latency_ms", 0)

        # Step 3: Check confidence for escalation (at most once)
        if rung != "Agentic":
            should_esc = ladder_module.should_escalate(
                result["answer"], question, rung, thresholds
            )
            if should_esc:
                next_r = ladder_module.next_rung(rung)
                conf = ladder_module.assess_confidence(result["answer"], question)
                ladder_module.log_escalation(
                    question_id, rung, next_r, conf,
                    thresholds.get("θ_grounding", 0.5)
                )

                rung_path.append(next_r)
                logger.info(f"  Escalating {rung} → {next_r}")

                if next_r == "GraphRAG":
                    result = graphrag_pipeline.run(question, question_id)
                else:
                    result = agentic_pipeline.run(question, question_id)

                tokens_total += result.get("tokens_used", 0)
                latency_total += result.get("latency_ms", 0)

        final_result = result

        # Build output record per spec §11.8
        record = {
            "question_id": question_id,
            "rung_path": rung_path,
            "final_answer": final_result.get("answer", ""),
            "exit_type": final_result.get("exit_type", None),
            "tokens_used_total": tokens_total,
            "latency_ms_total": latency_total,
            "n_chunks_retrieved": final_result.get("n_chunks_retrieved", 0),
            "n_sources_cited": len(final_result.get("sources", [])),
            "strategy_changed": (
                len(rung_path) > 1
                or final_result.get("strategy_changed", False)
            ),
            "stop_reason": final_result.get("stop_reason", None),
        }
        results.append(record)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"  Rung path: {' → '.join(rung_path)}, "
                    f"Tokens: {tokens_total}, "
                    f"Answer: {final_result.get('answer', '')[:50]}...")

    logger.info(f"ADAPTIVE complete. Results written to {output_path}")
    return results
