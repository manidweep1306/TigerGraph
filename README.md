# Agentic GraphRAG — TigerGraph Hackathon

> **An end-to-end Agentic GraphRAG system built on TigerGraph Savanna, LangGraph, and Gemini.**
> Benchmarks three approaches — RAG, GraphRAG, and Agentic GraphRAG — on Olympic events corpus.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-2.0--flash-orange.svg)](https://deepmind.google/gemini/)
[![TigerGraph](https://img.shields.io/badge/TigerGraph-Savanna-yellow.svg)](https://tgcloud.io)

---

## Architecture

```
User Query
    │
    ├─► Pipeline A: RAG         → Vector Search → Generate
    ├─► Pipeline B: GraphRAG    → Entity Link + GSQL + Vector → Generate  
    └─► Pipeline C: Agentic     → LangGraph State Machine
                                    │
                                    ├─ Decomposer → Slot Set
                                    ├─ [VoI → Dispatch → Agent → EvidenceEval → Ledger] × N
                                    ├─ Synthesis (ANSWER/PARTIAL/ABSTAIN)
                                    ├─ Claim Validation
                                    └─ Completeness Gate → Final Answer
```

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Node.js 18+
- TigerGraph Savanna account ([tgcloud.io](https://tgcloud.io))
- Google Gemini API key ([aistudio.google.com](https://aistudio.google.com))

### 2. Setup Backend

```bash
cd d:\TigerGraph\project\backend
pip install -r requirements.txt

# Copy and fill in credentials
copy ..\\.env.example ..\\.env
# Edit .env with your TigerGraph Savanna host, credentials, and Gemini API key
```

### 3. Ingest Corpus

```bash
# Dry run first (no writes to TigerGraph)
python backend/scripts/ingest_corpus.py --dry-run --limit 10

# Full ingestion (takes ~30–60 min for all 2951 docs)
python backend/scripts/ingest_corpus.py

# Quick test with 50 docs
python backend/scripts/ingest_corpus.py --limit 50
```

### 4. Install TigerGraph Schema

```bash
# Via TigerGraph Savanna UI: paste contents of backend/db/schema.gsql
# OR use pyTigerGraph:
python -c "
from backend.db.tigergraph_client import get_tg_client
tg = get_tg_client()
tg.install_schema('backend/db/schema.gsql')
"
```

### 5. Start Backend

```bash
python backend/main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 6. Start Frontend

```bash
cd frontend
npm run dev
# Dashboard at http://localhost:3000
```

---

## Running the Benchmark

### BENCHMARK mode (100 public questions, all 3 pipelines)

```bash
# Via API:
curl -X POST http://localhost:8000/benchmark/run

# Via CLI:
python -c "
from backend.evaluation.benchmark_runner import run_benchmark
run_benchmark(
    '../questions-20260920T040859Z-1-001/questions/eval_public.jsonl',
    limit=10  # remove limit for full run
)
"
```

### ADAPTIVE mode (50 hidden questions)

```bash
curl -X POST http://localhost:8000/adaptive/run

# Or CLI:
python -c "
from backend.evaluation.adaptive_runner import run_adaptive
run_adaptive('../questions-20260920T040859Z-1-001/questions/eval_hidden.jsonl')
"
```

---

## Running Tests

```bash
cd d:\TigerGraph\project
pip install pytest
python -m pytest tests/ -v

# Stage 1 only (Ledger — must pass before Stage 2):
python -m pytest tests/test_ledger.py -v

# Stage 5+7 (VoI + Synthesis):
python -m pytest tests/test_voi_synthesis.py -v
```

---

## Project Structure

```
project/
├── backend/
│   ├── main.py                   # FastAPI entrypoint
│   ├── graph.py                  # LangGraph orchestrator
│   ├── pipelines/                # RAG, GraphRAG, Agentic wrappers
│   ├── agents/                   # 6 specialized agents
│   ├── core/                     # Ledger, Decomposer, VoI, Ladder, Synthesis
│   ├── db/                       # TigerGraph + Vector DB clients
│   ├── evaluation/               # LLM judge, benchmark runner, adaptive runner
│   └── scripts/                  # Corpus ingestion
├── frontend/                     # Next.js 14 dashboard
│   └── src/app/page.tsx          # Full dashboard (benchmark, tokens, accuracy, trace)
├── config/
│   ├── frozen_thresholds.json    # VoI thresholds (§10)
│   ├── model_config.json         # Shared model config (§12)
│   └── agent_config.json         # Per-agent params
├── logs/                         # JSONL output files (§11)
└── tests/                        # Unit tests (§13)
```

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| LLM | `gemini-3.6-flash` | Cost-efficient, high-precision reasoning |
| Embeddings | `gemini-embedding-001` (768d) | Native Gemini embedding with 768d output |
| Chunk size | 512 tokens | Retrieval precision balance |
| Agentic framework | LangGraph `StateGraph` | State management + conditional routing |
| Evaluation | LLM-as-judge + BERTScore F1 | Multi-signal accuracy |
| Exit type | Pure function over slot states | No LLM can override (spec invariant #7) |

---

## Scoring Criteria Alignment

| Criterion | Weight | How We Address It |
|-----------|--------|-------------------|
| Investigation accuracy | 30% | LLM-as-judge + BERTScore on 100-question benchmark |
| Evidence quality & explainability | 15% | Ledger claims with provenance chains, source citations |
| Agentic effectiveness & efficiency | 15% | VoI scorer optimizes gain/cost; token tracking |
| Design, engineering & code quality | 15% | Strict spec adherence, full test suite, typed state |
| Innovation | 15% | Adaptive Ladder, slot state machine, evidence evaluation |
| Presentation | 10% | Interactive dashboard, trace viewer |

---

## Log Files (spec §11)

All JSONL logs in `logs/`:

| File | Contents |
|------|----------|
| `benchmark_results.jsonl` | All 3 pipelines × 100 questions |
| `adaptive_results.jsonl` | Ladder routing × 50 hidden questions |
| `agent_invocations.jsonl` | Every agent call with tokens + latency |
| `ledger_writes.jsonl` | Every slot state transition |
| `stop_reason_log.jsonl` | Why each investigation terminated |
| `escalation_log.jsonl` | RAG→GraphRAG→Agentic escalations |
| `calibration_log.jsonl` | Predicted vs actual VoI gains |
| `decomposition_revisions.jsonl` | Stall-triggered slot set revisions |

---

## Attribution

Corpus: English Wikipedia articles, CC BY-SA 4.0. Dataset provided by TigerGraph hackathon organizers.
