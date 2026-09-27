"""
Unified Configuration Module for Agentic GraphRAG Backend.
Follows backend-dev-guidelines: centralized, strictly typed, single source of truth.
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Locate the project root directory."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "config").exists() and (current / "config" / "model_config.json").exists():
            return current
        if (current / "project" / "config").exists():
            return current / "project"
        current = current.parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from possible .env paths
for env_path in [
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "project" / ".env",
    Path.cwd() / ".env",
    Path.cwd() / "project" / ".env",
]:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break
else:
    load_dotenv()


class GroqModelConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    fast_model: str = Field(default_factory=lambda: os.environ.get("GROQ_FAST_MODEL", "qwen/qwen3.8-27b"))
    complex_model: str = Field(default_factory=lambda: os.environ.get("GROQ_COMPLEX_MODEL", "openai/gpt-oss-120b"))
    default_model: str = Field(default_factory=lambda: os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))
    temperature: float = Field(default=0.0)
    max_retries: int = Field(default=6)
    initial_delay: float = Field(default=1.0)
    backoff_factor: float = Field(default=2.0)


class GeminiEmbeddingConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    embedding_model: str = Field(default_factory=lambda: os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))
    embedding_dimension: int = Field(default=768)
    max_retries: int = Field(default=6)
    initial_delay: float = Field(default=2.0)
    backoff_factor: float = Field(default=2.0)


class TigerGraphConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.environ.get("TG_HOST", "https://your-instance.i.tgcloud.io"))
    username: str = Field(default_factory=lambda: os.environ.get("TG_USERNAME", "tigergraph"))
    password: str = Field(default_factory=lambda: os.environ.get("TG_PASSWORD", ""))
    graph_name: str = Field(default_factory=lambda: os.environ.get("TG_GRAPH_NAME", "OlympicGraphRAG"))
    secret: str = Field(default_factory=lambda: os.environ.get("TG_SECRET", ""))


class PathConfig(BaseModel):
    project_root: str = str(PROJECT_ROOT)
    config_dir: str = str(CONFIG_DIR)
    log_dir: str = str(LOG_DIR)
    corpus_path: str = Field(default_factory=lambda: os.environ.get("CORPUS_PATH", "data/corpus/corpus.jsonl"))
    questions_public_path: str = Field(default_factory=lambda: os.environ.get("QUESTIONS_PUBLIC_PATH", "data/questions/eval_public.jsonl"))
    questions_hidden_path: str = Field(default_factory=lambda: os.environ.get("QUESTIONS_HIDDEN_PATH", "data/questions/eval_hidden.jsonl"))


class ServerConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.environ.get("API_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.environ.get("API_PORT", 8000)))
    frontend_port: int = Field(default_factory=lambda: int(os.environ.get("FRONTEND_PORT", 3000)))


class UnifiedConfig(BaseModel):
    groq: GroqModelConfig = Field(default_factory=GroqModelConfig)
    gemini: GeminiEmbeddingConfig = Field(default_factory=GeminiEmbeddingConfig)
    tigergraph: TigerGraphConfig = Field(default_factory=TigerGraphConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def load_json_config(self, filename: str) -> dict[str, Any]:
        """Load JSON configuration file safely from config directory."""
        candidates = [
            CONFIG_DIR / f"{filename}.json",
            CONFIG_DIR / filename,
            Path(f"./config/{filename}.json"),
            Path(f"./config/{filename}"),
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading config {p}: {e}")
        return {}

    @property
    def model_config_data(self) -> dict[str, Any]:
        return self.load_json_config("model_config")

    @property
    def agent_config_data(self) -> dict[str, Any]:
        return self.load_json_config("agent_config")

    @property
    def frozen_thresholds_data(self) -> dict[str, Any]:
        return self.load_json_config("frozen_thresholds")


# Global singleton instance of UnifiedConfig
config = UnifiedConfig()
