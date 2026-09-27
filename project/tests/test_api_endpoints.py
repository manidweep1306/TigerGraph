"""
API Endpoint Integration Tests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["baseline_fairness"] == "PASS"
    assert data["model"] == "openai/gpt-oss-120b"
    assert data["fast_model"] == "qwen/qwen3.8-27b"
    assert data["embedding_model"] == "gemini-embedding-001"


def test_config_endpoint():
    client = TestClient(app)
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "model_config" in data
    assert "agent_config" in data
    assert "frozen_thresholds" in data
    assert data["model_config"]["fast_model"] == "qwen/qwen3.8-27b"
    assert data["model_config"]["complex_model"] == "openai/gpt-oss-120b"
