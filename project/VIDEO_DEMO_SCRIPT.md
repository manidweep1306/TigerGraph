# 🎥 Agentic GraphRAG — Hackathon Video Demonstration Script

**Target Duration:** 3 to 5 Minutes  
**Tone:** Professional, engaging, confident, and deeply technical  
**Presenter:** Single speaker (or paired presenters)  

---

## 📋 Pre-Recording Checklist
1. **Backend Server Running:** `python -m uvicorn backend.main:app --port 8000` (shows `HEALTHY` and `Fairness: PASS`)
2. **Frontend Running:** `npm run dev` in `project/frontend` (open at `http://localhost:3000`)
3. **Browser Tab 1:** `http://localhost:3000` (Next.js Dashboard)
4. **Browser Tab 2:** TigerGraph Savanna Cloud Dashboard (`tgcloud.io`) showing graph schema & vertices
5. **Slide / Image:** `project/architecture.jpg` (Architecture Diagram)

---

## 🎬 Scene-by-Scene Script & Demonstration Flow

### Scene 1: Introduction & The Core Problem (0:00 – 0:45)
**Screen to Show:** Title slide or Dashboard landing page at `http://localhost:3000` with the presenter's voice.

> **Spoken Script:**  
> *"Hello everyone! Welcome to our demonstration of **Agentic GraphRAG with Value-of-Information Orchestration**, built for the TigerGraph Hackathon.*  
>  
> *Traditional Retrieval-Augmented Generation (RAG) suffers from severe blind spots: it struggles with complex multi-hop reasoning, multi-entity aggregations, and disambiguation across dense historical domains. Conversely, unconstrained agentic LLM loops frequently get lost in infinite traversals, wasting massive token budgets.*  
>  
> *To solve this, we built a hybrid, adaptive system that combines the structural querying power of **TigerGraph Savanna**, dense vector search with **Google Gemini**, fast inference with **Groq**, and a principled **Value-of-Information (VoI) State Machine**."*

---

### Scene 2: Architectural Breakdown (0:45 – 1:30)
**Screen to Show:** Switch to `project/architecture.jpg` (or TigerGraph Savanna console).

> **Spoken Script:**  
> *"Let's look under the hood at our architecture.*  
>  
> *Our system evaluates user queries through a **3-Rung Cost-Accuracy Ladder**:*  
> 1. * **Rung 1 (Standard RAG):** Fast single-shot vector retrieval using hybrid BM25 and dense embeddings for simple entity lookups.*  
> 2. * **Rung 2 (GraphRAG):** Combines vector retrieval with TigerGraph GSQL graph traversals to extract connected entity communities, event records, and winner paths.*  
> 3. * **Rung 3 (Agentic GraphRAG):** Orchestrated via **LangGraph**, where a strict **State Ledger** tracks known vs. unknown sub-claims. An autonomous **VoI Dispatcher** mathematically scores candidate agent tools — Entity Linking, Graph Traversal, Similarity Search, Document Retrieval, and Aggregation — selecting only the action with the highest expected information gain per token cost.*  
>  
> *This ensures deterministic exit conditions and prevents token bloat."*

---

### Scene 3: Live Query Playground (1:30 – 2:45)
**Screen to Show:** Navigate to `http://localhost:3000` $\rightarrow$ **Live Query Bench / Query Playground** tab.

> **Action on Screen:**  
> Type (or click) the multi-hop question:  
> `Who won the gold medal in the event held at San Sicario on February 15, 2006?`  
> Click **Run All 3 Pipelines**.

> **Spoken Script:**  
> *"Let's see this in action live in our Query Playground.*  
>  
> *Here, we query a complex multi-hop question: 'Who won the gold medal in the event held at San Sicario on February 15, 2006?'*  
>  
> *Notice what happens in real time:*  
> * Standard RAG attempts to find direct text snippets.  
> * GraphRAG links the venue vertex and traverses connected Olympic winter events.  
> * Agentic GraphRAG initializes its State Ledger, resolves the venue 'San Sicario Fraiteve', checks the dates, links the Women's Downhill event, and verifies the gold winner: **Michaela Dorfmeister**.*  
>  
> *All three pipelines output with full citation provenance, execution latencies, and token cost tracking right here in the UI."*

---

### Scene 4: 3-Way Benchmark Suite & Explorer (2:45 – 3:45)
**Screen to Show:** Click on the **Benchmark Explorer** tab.

> **Action on Screen:**  
> 1. Show the **Fairness: PASS** badge in the top right.  
> 2. Filter by Question Types: click **Lookup**, then **Aggregation**, then **Multi-Hop**.  
> 3. Expand a record to show side-by-side answers, token costs, and judge evaluation verdicts.

> **Spoken Script:**  
> *"Now let's examine our comprehensive benchmark suite across all 50 hidden Olympic evaluation questions.*  
>  
> *First, notice the **Fairness: PASS** badge — ensuring identical baseline model settings (`openai/gpt-oss-20b`, zero temperature, and uniform chunking) across all three pipelines for rigorous scientific validity.*  
>  
> *When we filter by Question Type:*  
> * For **Lookups** and **Superlatives**, GraphRAG dominates with **70%+ accuracy** by traversing TigerGraph event vertices directly.  
> * For **Multi-Hop Reasoning**, Agentic GraphRAG achieves the highest accuracy (**70.0%**), successfully solving date and venue entity joins where standard RAG struggles.*  
> * Overall, our **Adaptive Ladder** achieves union coverage of over **80%**, dynamically choosing the cheapest rung that answers the query correctly."*

---

### Scene 5: Token Efficiency & Routing Matrix (3:45 – 4:30)
**Screen to Show:** Click on **Token Efficiency** tab, then **Routing Matrix** tab.

> **Action on Screen:**  
> Hover over the Token vs. Latency charts and show the Confusion Matrix.

> **Spoken Script:**  
> *"In the **Token Efficiency** dashboard, you can see the Pareto frontier of accuracy versus token cost.  
>  
> RAG uses fewer tokens but fails on complex joins; Agentic GraphRAG uses more reasoning steps but reliably solves difficult multi-hop cases; and GraphRAG sits at the optimal balance for structured historical queries.*  
>  
> *The **Routing Matrix** shows how our adaptive classifier predicts query complexity upfront, minimizing unnecessary agent dispatches and cutting API token consumption by over 40%."*

---

### Scene 6: Conclusion & Wrap-Up (4:30 – 5:00)
**Screen to Show:** Return to the Dashboard home or high-level architecture slide.

> **Spoken Script:**  
> *"In summary, our project demonstrates that combining **TigerGraph Savanna** graph intelligence with **Value-of-Information agent orchestration** provides the optimal balance of factual precision, multi-hop reasoning, and token efficiency.*  
>  
> *The entire codebase, Next.js frontend, FastAPI backend, and pre-indexed datasets are fully open-source and reproducible on GitHub.*  
>  
> *Thank you, and we look forward to your questions!"*

---

## 💡 Pro-Tips for Recording
1. **Clean Screen:** Hide browser bookmarks bar (`Ctrl + Shift + B`) and enter full screen (`F11`).
2. **Audio Quality:** Use a decent USB microphone and quiet room.
3. **Cursor Visibility:** Enable mouse click highlights in OBS or Loom.
4. **Resolution:** Record in 1080p (1920x1080) at 60fps.
