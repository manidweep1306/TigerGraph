"""
Tests for Unified Configuration and LLM/Embeddings Client Layer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from backend.config.unified_config import config, UnifiedConfig
from backend.core.llm_client import GroqClient, GeminiEmbeddingClient, get_groq_client, get_embedding_client


def test_unified_config_defaults():
    assert config.groq.fast_model == "qwen/qwen3.8-27b"
    assert config.groq.complex_model == "openai/gpt-oss-120b"
    assert config.gemini.embedding_model == "gemini-embedding-001"
    assert config.gemini.embedding_dimension == 768


def test_clean_and_parse_json():
    cleaner = GroqClient._clean_and_parse_json

    # Pure JSON
    res1 = cleaner('{"key": "value"}')
    assert res1 == {"key": "value"}

    # Markdown fenced json
    res2 = cleaner('```json\n{"claim_text": "text", "confidence": 0.9}\n```')
    assert res2 == {"claim_text": "text", "confidence": 0.9}

    # Text wrapping json
    res3 = cleaner('Here is the output: {"answer": 42} thank you')
    assert res3 == {"answer": 42}


def test_client_singletons():
    groq1 = get_groq_client()
    groq2 = get_groq_client()
    assert groq1 is groq2

    emb1 = get_embedding_client()
    emb2 = get_embedding_client()
    assert emb1 is emb2


def test_model_config_loading():
    model_cfg = config.model_config_data
    assert model_cfg.get("fast_model") == "qwen/qwen3.8-27b"
    assert model_cfg.get("complex_model") == "openai/gpt-oss-120b"
    assert model_cfg.get("embedding_model") == "gemini-embedding-001"


def test_agent_config_loading():
    agent_cfg = config.agent_config_data
    assert "EvidenceEvaluatorAgent" in agent_cfg
    assert agent_cfg["EvidenceEvaluatorAgent"].get("model") == "qwen/qwen3.8-27b"
    assert agent_cfg["EntityLinkerAgent"].get("model") == "qwen/qwen3.8-27b"
