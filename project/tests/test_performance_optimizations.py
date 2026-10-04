import os
import sys
import time
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db import vector_index
from backend.core.llm_client import GroqClient, GroqRotator, GroqKeyEntry


def test_vector_in_memory_search():
    # Setup dummy in-memory embeddings
    dim = 768
    num_chunks = 20
    np.random.seed(42)
    dummy_matrix = np.random.randn(num_chunks, dim).astype(np.float32)
    norms = np.linalg.norm(dummy_matrix, axis=1, keepdims=True)
    dummy_matrix /= norms

    dummy_meta = [{"chunk_id": f"chunk_{i}", "text": f"text {i}", "doc_id": f"doc_{i}"} for i in range(num_chunks)]

    vector_index.CHUNK_EMBEDDINGS_MATRIX = dummy_matrix
    vector_index.CHUNK_METADATA = dummy_meta

    # Mock query embedding in cache
    dummy_query = "Who won Olympic gold in 2012?"
    q_vec = dummy_matrix[5].tolist()  # exact match with chunk 5
    vector_index.QUERY_EMBEDDINGS_CACHE[f"RETRIEVAL_QUERY:{dummy_query}"] = q_vec

    vi = vector_index.get_vector_index()
    t0 = time.time()
    results = vi.search(dummy_query, top_k=3)
    latency_ms = (time.time() - t0) * 1000

    assert len(results) == 3
    assert results[0]["chunk_id"] == "chunk_5"
    assert pytest.approx(results[0]["score"], abs=1e-3) == 1.0
    # Vector search directly in RAM must be < 20ms
    assert latency_ms < 20.0


def test_groq_multi_key_rotator():
    keys = ["gsk_key_1", "gsk_key_2", "gsk_key_3"]
    rotator = GroqRotator(keys)

    # Test round robin
    _, entry1, wait1 = rotator.get_client_for_call()
    _, entry2, wait2 = rotator.get_client_for_call()
    _, entry3, wait3 = rotator.get_client_for_call()
    _, entry4, wait4 = rotator.get_client_for_call()

    assert entry1.api_key == "gsk_key_1"
    assert entry2.api_key == "gsk_key_2"
    assert entry3.api_key == "gsk_key_3"
    assert entry4.api_key == "gsk_key_1"
    assert wait1 == 0.0 and wait2 == 0.0 and wait3 == 0.0 and wait4 == 0.0

    # Test cooldown on 429
    entry1.set_cooldown(10.0)
    now = time.time()
    assert not entry1.is_available(now)
    assert entry2.is_available(now)

    # Next call should skip key 1 and give key 2
    _, entry_next, wait_next = rotator.get_client_for_call()
    assert entry_next.api_key == "gsk_key_2"
    assert wait_next == 0.0


def test_benchmark_runner_auto_resume(tmp_path):
    from backend.evaluation import benchmark_runner
    import json

    output_file = tmp_path / "results.jsonl"
    questions_file = tmp_path / "questions.jsonl"

    # Write 3 questions
    q_list = [
        {"qid": "pub-001", "question": "Q1", "answer": ["A1"]},
        {"qid": "pub-002", "question": "Q2", "answer": ["A2"]},
        {"qid": "pub-003", "question": "Q3", "answer": ["A3"]},
    ]
    with open(questions_file, "w") as f:
        for q in q_list:
            f.write(json.dumps(q) + "\n")

    # Pretend pub-001 and pub-002 are already done
    with open(output_file, "w") as f:
        f.write(json.dumps({"question_id": "pub-001", "rag": {"accuracy_score": "PASS", "tokens_used": 10}}) + "\n")
        f.write(json.dumps({"question_id": "pub-002", "rag": {"accuracy_score": "FAIL", "tokens_used": 10}}) + "\n")

    # If limit=2, all are already done so it should return 2 records immediately
    res = benchmark_runner.run_benchmark(str(questions_file), output_path=str(output_file), limit=2)
    assert len(res) == 2
