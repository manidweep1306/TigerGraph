"""
FastAPI Backend — main.py
Serves all three pipelines and the benchmark/adaptive runners via REST API.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Agentic GraphRAG API",
    description="TigerGraph Hackathon — Three-way benchmark: RAG vs GraphRAG vs Agentic GraphRAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    question_id: Optional[str] = None
    pipeline: Optional[str] = "all"  # "rag" | "graphrag" | "agentic" | "all"


class BenchmarkRequest(BaseModel):
    limit: Optional[int] = None  # None = all 100
    questions_path: Optional[str] = None


class AdaptiveRequest(BaseModel):
    limit: Optional[int] = None  # None = all 50
    questions_path: Optional[str] = None


# ─── Health & Config ──────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Returns baseline fairness validation status."""
    try:
        with open("./config/model_config.json") as f:
            model_cfg = json.load(f)
        from backend.evaluation.evaluator import validate_baseline_fairness
        fairness_ok = validate_baseline_fairness(model_cfg, model_cfg, model_cfg)
        return {
            "status": "healthy",
            "baseline_fairness": "PASS" if fairness_ok else "FAIL",
            "model": model_cfg.get("model"),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/config")
async def get_config():
    """Return current configuration."""
    configs = {}
    for name in ["frozen_thresholds", "model_config", "agent_config"]:
        path = Path(f"./config/{name}.json")
        if path.exists():
            with open(path) as f:
                configs[name] = json.load(f)
    return configs


# ─── Single Query Endpoints ───────────────────────────────────────────────────

@app.post("/query")
async def query_all_pipelines(req: QueryRequest):
    """
    Run a single question through all 3 pipelines (or selected pipeline).
    For live demo / Query Playground in dashboard.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    qid = req.question_id or f"live_{hash(question) % 100000}"
    pipeline = (req.pipeline or "all").lower()

    result = {}

    try:
        if pipeline in ("rag", "all"):
            from backend.pipelines.rag_pipeline import run as rag_run
            result["rag"] = rag_run(question, qid)

        if pipeline in ("graphrag", "all"):
            from backend.pipelines.graphrag_pipeline import run as grag_run
            result["graphrag"] = grag_run(question, qid)

        if pipeline in ("agentic", "all"):
            from backend.pipelines.agentic_pipeline import run as ag_run
            ag_out = ag_run(question, qid)
            # Serialize (ledger is not JSON-serializable)
            result["agentic"] = {
                "answer": ag_out["answer"],
                "exit_type": ag_out.get("exit_type"),
                "sources": ag_out.get("sources", []),
                "tokens_used": ag_out.get("tokens_used", 0),
                "latency_ms": ag_out.get("latency_ms", 0),
                "trace": ag_out.get("trace", []),
                "stop_reason": ag_out.get("stop_reason"),
                "strategy_changed": ag_out.get("strategy_changed", False),
                "n_chunks_retrieved": ag_out.get("n_chunks_retrieved", 0),
                "n_sources_cited": ag_out.get("n_sources_cited", 0),
            }

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"question": question, "question_id": qid, "results": result}


@app.post("/adaptive")
async def query_adaptive(req: QueryRequest):
    """
    Run a single question through the ADAPTIVE Ladder.
    """
    question = req.question.strip()
    qid = req.question_id or f"adaptive_{hash(question) % 100000}"

    try:
        from backend.core import ladder as ladder_module
        with open("./config/frozen_thresholds.json") as f:
            thresholds = json.load(f)

        rung = ladder_module.route(question, qid, thresholds)
        rung_path = [rung]

        # Run initial rung
        if rung == "RAG":
            from backend.pipelines.rag_pipeline import run as run_fn
        elif rung == "GraphRAG":
            from backend.pipelines.graphrag_pipeline import run as run_fn
        else:
            from backend.pipelines.agentic_pipeline import run as run_fn

        result = run_fn(question, qid)

        # Possible escalation
        if rung != "Agentic" and ladder_module.should_escalate(
                result["answer"], question, rung, thresholds):
            next_r = ladder_module.next_rung(rung)
            rung_path.append(next_r)
            if next_r == "GraphRAG":
                from backend.pipelines.graphrag_pipeline import run as run_fn2
            else:
                from backend.pipelines.agentic_pipeline import run as run_fn2
            result = run_fn2(question, qid)

        return {
            "question": question,
            "rung_path": rung_path,
            "answer": result.get("answer", ""),
            "tokens_used": result.get("tokens_used", 0),
            "latency_ms": result.get("latency_ms", 0),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _resolve_data_path(env_var: str, default_rel: str) -> str:
    """Resolve dataset paths whether running from project/ or repo root."""
    path_val = os.environ.get(env_var, default_rel)
    candidates = [
        path_val,
        Path(path_val),
        Path(".") / path_val.replace("../", ""),
        Path("..") / path_val,
        Path(__file__).parent.parent.parent / path_val.replace("../", ""),
        Path(__file__).parent.parent / path_val.replace("../", ""),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(Path(c).resolve())
    return path_val


# ─── Batch Run Endpoints ──────────────────────────────────────────────────────

@app.post("/benchmark/run")
async def run_benchmark(req: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Launch the BENCHMARK mode runner (background task)."""
    questions_path = req.questions_path or _resolve_data_path(
        "QUESTIONS_PUBLIC_PATH",
        "../questions-20260920T040859Z-1-001/questions/eval_public.jsonl"
    )

    def _run():
        from backend.evaluation.benchmark_runner import run_benchmark as _bench
        _bench(questions_path, limit=req.limit)

    background_tasks.add_task(_run)
    return {"status": "started", "mode": "BENCHMARK",
            "questions_path": questions_path, "limit": req.limit}


@app.post("/adaptive/run")
async def run_adaptive_batch(req: AdaptiveRequest, background_tasks: BackgroundTasks):
    """Launch the ADAPTIVE mode runner (background task)."""
    questions_path = req.questions_path or _resolve_data_path(
        "QUESTIONS_HIDDEN_PATH",
        "../questions-20260920T040859Z-1-001/questions/eval_hidden.jsonl"
    )

    def _run():
        from backend.evaluation.adaptive_runner import run_adaptive
        run_adaptive(questions_path, limit=req.limit)

    background_tasks.add_task(_run)
    return {"status": "started", "mode": "ADAPTIVE",
            "questions_path": questions_path, "limit": req.limit}


# ─── Results Endpoints ────────────────────────────────────────────────────────

@app.get("/benchmark/results")
async def get_benchmark_results(limit: Optional[int] = None):
    """Serve benchmark_results.jsonl as JSON array."""
    return _read_jsonl("./logs/benchmark_results.jsonl", limit)


@app.get("/adaptive/results")
async def get_adaptive_results(limit: Optional[int] = None):
    """Serve adaptive_results.jsonl as JSON array."""
    return _read_jsonl("./logs/adaptive_results.jsonl", limit)


@app.get("/logs/{log_name}")
async def get_log(log_name: str, limit: Optional[int] = 100):
    """Serve any JSONL log file."""
    valid_logs = [
        "benchmark_results", "adaptive_results", "agent_invocations",
        "ledger_writes", "stop_reason_log", "escalation_log",
        "decomposition_revisions", "calibration_log",
        "claim_validation_failures", "completeness_gate_log",
    ]
    if log_name not in valid_logs:
        raise HTTPException(status_code=404, detail=f"Unknown log: {log_name}")
    return _read_jsonl(f"./logs/{log_name}.jsonl", limit)


@app.get("/stats/summary")
async def get_summary_stats():
    """Compute summary statistics from benchmark_results.jsonl."""
    results = _read_jsonl("./logs/benchmark_results.jsonl")
    if not results:
        return {"status": "no_results", "message": "Run /benchmark/run first"}

    n = len(results)
    stats = {
        "total_questions": n,
        "rag": _compute_pipeline_stats([r["rag"] for r in results if "rag" in r]),
        "graphrag": _compute_pipeline_stats([r["graphrag"] for r in results if "graphrag" in r]),
        "agentic": _compute_pipeline_stats([r["agentic"] for r in results if "agentic" in r]),
    }

    # Per question type
    qtypes = {}
    for r in results:
        qt = r.get("qtype", "unknown")
        if qt not in qtypes:
            qtypes[qt] = {"rag": [], "graphrag": [], "agentic": []}
        for p in ("rag", "graphrag", "agentic"):
            if p in r:
                qtypes[qt][p].append(r[p].get("accuracy_score", "FAIL"))

    stats["by_qtype"] = {
        qt: {p: qtypes[qt][p].count("PASS") / max(len(qtypes[qt][p]), 1)
             for p in ("rag", "graphrag", "agentic")}
        for qt in qtypes
    }
    return stats


def _compute_pipeline_stats(records: list[dict]) -> dict:
    if not records:
        return {}
    n = len(records)
    return {
        "accuracy": sum(1 for r in records if r.get("accuracy_score") == "PASS") / n,
        "avg_tokens": sum(r.get("tokens_used", 0) for r in records) / n,
        "avg_latency_ms": sum(r.get("latency_ms", 0) for r in records) / n,
        "avg_bertscore": sum(r.get("bertscore_f1", 0) for r in records) / n,
        "count": n,
    }


def _read_jsonl(path: str, limit: Optional[int] = None) -> list[dict]:
    records = []
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
            if limit and len(records) >= limit:
                break
    return records


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    app_target = "backend.main:app" if (Path.cwd() / "backend").exists() else "main:app"
    try:
        uvicorn.run(app_target, host=host, port=port, reload=True)
    except Exception:
        uvicorn.run(app, host=host, port=port)
