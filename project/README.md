# 🐯 Agentic GraphRAG — TigerGraph Hackathon

> **An end-to-end Agentic GraphRAG system built on TigerGraph Savanna, LangGraph, and Google Gemini.**  
> Evaluates and benchmarks three retrieval approaches — **Standard RAG**, **GraphRAG**, and **Agentic GraphRAG with Value-of-Information (VoI) Orchestration** — over an Olympic events corpus.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-3.6--flash-orange.svg)](https://deepmind.google/gemini/)
[![TigerGraph](https://img.shields.io/badge/TigerGraph-Savanna-yellow.svg)](https://tgcloud.io)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)

---

## Architecture Overview

```
User Query
    │
    ├─► Pipeline A: RAG         → Vector Similarity Search → Single-shot Generation
    ├─► Pipeline B: GraphRAG    → Entity Linker + TigerGraph Traversal + Vector Search → Generation
    └─► Pipeline C: Agentic     → LangGraph Investigation State Machine
                                    │
                                    ├─ Decomposer → Central & Context Slot Set
                                    ├─ [VoI Scorer → Dispatcher → Specialized Agent → Evidence Evaluator → Ledger Write] × N
                                    ├─ Dynamic Slot Revision (if stalled)
                                    ├─ Synthesis Engine (Strict Status: ANSWER / PARTIAL / ABSTAIN)
                                    ├─ Claim Validation & Citation Verification
                                    └─ Completeness Gate → Final Answer with Provenance
```

---

## 🚀 Step-by-Step Clone & Setup Guide

Follow this guide to clone, set up, and run the entire application on any machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/manidweep1306/TigerGraph.git
cd TigerGraph
```

---

### Step 2: Set Up Python Virtual Environment

Create and activate an isolated Python environment (Python 3.10, 3.11, 3.12, or 3.13 recommended):

**On Windows (PowerShell / CMD):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Install backend dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Configure Environment Variables

Copy the provided `.env.example` template:

**On Windows:**
```powershell
copy project\.env.example project\.env
copy project\.env.example project\backend\.env
```

**On macOS / Linux:**
```bash
cp project/.env.example project/.env
cp project/.env.example project/backend/.env
```

Open `project/.env` (and `project/backend/.env`) in any text editor and fill in your credentials:

```dotenv
# --- Google Gemini API ---
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# --- TigerGraph Savanna ---
# Obtain these from your TigerGraph Savanna instance at https://tgcloud.io
TG_HOST=https://your-instance.i.tgcloud.io
TG_USERNAME=tigergraph
TG_PASSWORD=your_password
TG_GRAPH_NAME=OlympicGraphRAG
TG_SECRET=your_secret_token

# --- Paths (Pre-configured for the bundled datasets) ---
CORPUS_PATH=../corpus-20260920T042835Z-1-001/corpus/corpus.jsonl
QUESTIONS_PUBLIC_PATH=../questions-20260920T040859Z-1-001/questions/eval_public.jsonl
QUESTIONS_HIDDEN_PATH=../questions-20260920T040859Z-1-001/questions/eval_hidden.jsonl
```

---

### Step 4: Local Datasets Included in the Repository

All required data is **directly tracked in this repository** so you don't need to download external files:

| Directory / File | Contents | Purpose |
|------------------|----------|---------|
| `corpus-20260920T042835Z-1-001/corpus/corpus.jsonl` | 2,951 English Wikipedia articles on Olympic history (~5.4M tokens) | Source-of-truth document corpus for RAG & Graph |
| `questions-20260920T040859Z-1-001/questions/eval_public.jsonl` | 100 labeled evaluation questions with gold answers | Benchmark evaluation dataset (300 runs) |
| `questions-20260920T040859Z-1-001/questions/eval_hidden.jsonl` | 50 unlabeled questions for adaptive routing | Adaptive Ladder evaluation dataset |
| `project/backend/db/schema.gsql` | TigerGraph GSQL Schema | Vertices, edges, indexes, and queries |

---

### Step 5: Install TigerGraph Schema & Ingest Corpus

1. **Install GSQL Schema:**
   - Log into your TigerGraph Savanna dashboard (`tgcloud.io`), open **GraphStudio** or the **GSQL Console**, and execute the script in [`project/backend/db/schema.gsql`](file:///d:/TigerGraph/project/backend/db/schema.gsql).
   - Alternatively, install via pyTigerGraph script.

2. **Ingest Corpus Documents:**
   Navigate into `project`:
   ```bash
   cd project
   ```

   *Dry-run test (parses 10 documents without writing to TigerGraph):*
   ```bash
   python backend/scripts/ingest_corpus.py --dry-run --limit 10
   ```

   *Fast sample ingestion (50 documents for quick testing):*
   ```bash
   python backend/scripts/ingest_corpus.py --limit 50
   ```

   *Full dataset ingestion (all 2,951 documents with entity linking & embeddings):*
   ```bash
   python backend/scripts/ingest_corpus.py
   ```

---

### Step 6: Start the FastAPI Backend Server

From the `project/` directory:
```bash
python backend/main.py
```

* Backend runs at: `http://localhost:8000`
* Interactive API Documentation (Swagger): `http://localhost:8000/docs`
* Health check: `http://localhost:8000/health` (Validates baseline model fairness across all pipelines)

---

### Step 7: Start the Next.js Frontend Dashboard

Open a new terminal window:

```bash
cd project/frontend
npm install
npm run dev
```

* Dashboard UI is live at: `http://localhost:3000`

---

## 🧪 Using the Interactive Dashboard & Running Benchmarks

### 1. Live Query Playground
- Open `http://localhost:3000` and navigate to the **🧪 Live Query** tab.
- Click any of the preloaded Olympic sample questions or type your own question.
- Select **All 3 Pipelines** (or individual ones: RAG, GraphRAG, Agentic) and click **Run Query →**.
- Compare the answers, retrieved passage chunks, graph traversal paths, token costs, and investigation latencies side-by-side.

### 2. Run the 3-Way Benchmark (100 Questions × 3 Pipelines)
- **Via Dashboard:** Navigate to the **📊 Benchmark** tab and trigger the benchmark runner.
- **Via API:**
  ```bash
  curl -X POST http://localhost:8000/benchmark/run
  ```
- **Via Python:**
  ```python
  from backend.evaluation.benchmark_runner import run_benchmark
  run_benchmark('../questions-20260920T040859Z-1-001/questions/eval_public.jsonl', limit=10)
  ```

### 3. Run Adaptive Mode (50 Questions)
- Automatically classifies question difficulty and routes queries up the cost-accuracy ladder (RAG → GraphRAG → Agentic):
  ```bash
  curl -X POST http://localhost:8000/adaptive/run
  ```

---

## 🔬 Running Unit Tests

Run the test suite to verify the State Ledger, Value-of-Information (VoI) Scorer, and Synthesis Engine:

```bash
cd project
python -m pytest tests/ -v
```

* **Test State Ledger invariants:** `python -m pytest tests/test_ledger.py -v`
* **Test VoI & Synthesis:** `python -m pytest tests/test_voi_synthesis.py -v`

---

## 📂 Repository Structure

```
TigerGraph/
├── requirements.txt                              # Root Python dependencies
├── .env.example                                  # Root environment variables template
├── .gitignore                                    # Strict gitignore protecting secrets
├── README.md                                     # System documentation and setup guide
├── adaptive-investigation-engine-implementation-spec.md # Technical spec
├── corpus-20260920T042835Z-1-001/                # Bundled local corpus
│   └── corpus/
│       └── corpus.jsonl                          # 2,951 Wikipedia Olympic articles (22 MB)
├── questions-20260920T040859Z-1-001/             # Bundled evaluation datasets
│   └── questions/
│       ├── eval_public.jsonl                     # 100 benchmark questions with gold answers
│       └── eval_hidden.jsonl                     # 50 adaptive evaluation questions
└── project/
    ├── requirements.txt                          # Project-level dependencies
    ├── .env.example                              # Project-level env template
    ├── config/
    │   ├── model_config.json                     # Shared model config (gemini-3.6-flash, 768d)
    │   ├── frozen_thresholds.json                # VoI & routing thresholds
    │   └── agent_config.json                     # Per-agent execution limits
    ├── backend/
    │   ├── main.py                               # FastAPI application endpoints
    │   ├── graph.py                              # LangGraph state graph & investigation loop
    │   ├── requirements.txt                      # Backend dependencies
    │   ├── agents/                               # 6 specialized investigation agents
    │   │   ├── entity_linker.py                  # Links query mentions to graph vertices
    │   │   ├── graph_traversal.py                # TigerGraph multi-hop neighborhood queries
    │   │   ├── similarity_search.py              # Semantic vector search over chunks
    │   │   ├── document_retrieval.py             # Full document reader
    │   │   ├── aggregation.py                    # Multi-entity count/filter queries
    │   │   └── evidence_evaluator.py             # Evaluates passage relevance & validity
    │   ├── core/
    │   │   ├── decomposer.py                     # Decomposes queries into slot set
    │   │   ├── ledger.py                         # Typed investigation state & slot transitions
    │   │   ├── voi_scorer.py                     # Value-of-Information (gain/token cost) scoring
    │   │   ├── dispatcher.py                     # Dispatches highest-VoI agent
    │   │   ├── synthesis.py                      # Deterministic exit type & answer generation
    │   │   └── ladder.py                         # Adaptive routing ladder
    │   ├── db/
    │   │   ├── schema.gsql                       # TigerGraph GSQL graph schema definition
    │   │   ├── tigergraph_client.py              # pyTigerGraph wrapper client
    │   │   └── vector_index.py                   # Gemini 768d embedding & vector search
    │   ├── evaluation/
    │   │   ├── evaluator.py                      # LLM-as-judge & BERTScore F1 evaluators
    │   │   ├── benchmark_runner.py               # 100 questions × 3 pipelines batch runner
    │   │   └── adaptive_runner.py                # 50 questions adaptive ladder runner
    │   └── scripts/
    │       └── ingest_corpus.py                  # Parses corpus, extracts entities & upserts
    ├── frontend/                                 # Next.js 14 Interactive Dashboard
    │   ├── package.json
    │   └── src/app/
    │       ├── page.tsx                          # Full dashboard tabs (Live Query, Benchmark, etc.)
    │       └── globals.css
    └── tests/
        ├── test_ledger.py                        # Ledger invariant test suite
        └── test_voi_synthesis.py                 # VoI calculation & synthesis tests
```

---

## 🔒 Security & Privacy Notice
* All real `.env` files and API keys are strictly excluded via [`.gitignore`](file:///d:/TigerGraph/.gitignore).
* Never commit production passwords or API tokens to git. Use `.env.example` as your reference.

---

## 🏆 Scoring Criteria Alignment

| Hackathon Criterion | Weight | How Our Implementation Addresses It |
|---------------------|:------:|-------------------------------------|
| **Investigation Accuracy** | 30% | Dual evaluation via LLM-as-judge (factual precision) + BERTScore F1 on 100 Olympic benchmark questions. |
| **Evidence Quality & Provenance** | 15% | All ledger claims carry strict provenance chains, chunk IDs, and exact text spans. Unverified claims cannot trigger an `ANSWER` status. |
| **Agentic Effectiveness & Efficiency** | 15% | Value-of-Information (VoI) engine ranks candidates by expected gain per token cost, preventing redundant traversals and LLM token bloat. |
| **Design, Engineering & Code Quality** | 15% | Strict LangGraph state machine, typed transitions, baseline fairness checks, and 100% reproducible configs. |
| **Innovation** | 15% | Adaptive 3-rung ladder (RAG → GraphRAG → Agentic) with dynamic escalation and deterministic synthesis invariants. |
| **User Experience & Presentation** | 10% | Modern real-time Next.js dashboard featuring live traces, token cost analysis, and interactive query playground. |
