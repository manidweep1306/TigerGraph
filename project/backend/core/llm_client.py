"""
Production LLM & Embeddings Client Layer.
- Groq for fast & complex LLM inference:
    * Fast Intermediate Agents: llama-3.1-8b-instant (~120ms, 14,400 RPD / 30 RPM)
    * Complex Logic & Synthesis: llama-3.3-70b-versatile (deep reasoning, strict JSON, 1,000 RPD / 30 RPM)
- Google Gemini for 768-dim Vector Embeddings:
    * gemini-embedding-001 loaded into TigerGraph Chunk vertices
"""

import json
import logging
import os
import re
import time
import random
from typing import Any, Optional, Union
from dataclasses import dataclass

from backend.config.unified_config import config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    raw_response: Any = None
    model: str = ""
    latency_ms: int = 0


class GroqClient:
    """
    Robust Groq API client with exponential backoff, rate-limit resilience,
    and role-based model dispatching.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        fast_model: Optional[str] = None,
        complex_model: Optional[str] = None,
    ):
        self.api_key = api_key or config.groq.api_key
        self.fast_model = fast_model or config.groq.fast_model
        self.complex_model = complex_model or config.groq.complex_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            if not self.api_key:
                logger.warning("GROQ_API_KEY is not set. Groq API calls will fail if not authenticated.")
            self._client = Groq(api_key=self.api_key)
        return self._client

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        response_format: Optional[dict[str, str]] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 6,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        **kwargs
    ) -> LLMResponse:
        """
        Execute Groq chat completion with rate-limiting and transient error retry logic.
        """
        client = self._get_client()
        target_model = model or self.complex_model
        delay = initial_delay
        t_start = time.time()

        for attempt in range(1, max_retries + 1):
            try:
                call_kwargs: dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "temperature": temperature,
                    **kwargs
                }
                if response_format is not None:
                    call_kwargs["response_format"] = response_format
                if max_tokens is not None:
                    call_kwargs["max_tokens"] = max_tokens

                response = client.chat.completions.create(**call_kwargs)
                latency_ms = int((time.time() - t_start) * 1000)

                content = response.choices[0].message.content or ""
                total_tokens = 0
                if hasattr(response, "usage") and response.usage:
                    total_tokens = getattr(response.usage, "total_tokens", 0) or 0
                if total_tokens <= 0:
                    # Approximation: ~4 chars per token
                    msg_len = sum(len(m.get("content", "")) for m in messages)
                    total_tokens = max(1, (msg_len + len(content)) // 4)

                return LLMResponse(
                    text=content.strip(),
                    tokens_used=total_tokens,
                    raw_response=response,
                    model=target_model,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "rate_limit" in err_str.lower()
                    or "rate limit" in err_str.lower()
                    or "tokens per minute" in err_str.lower()
                    or "requests per minute" in err_str.lower()
                    or "requests per day" in err_str.lower()
                )
                is_server_error = (
                    "500" in err_str or "502" in err_str or "503" in err_str or "504" in err_str
                    or "service unavailable" in err_str.lower()
                    or "timeout" in err_str.lower()
                )
                is_retryable = is_rate_limit or is_server_error

                if not is_retryable:
                    logger.error(f"Groq API non-retryable error on {target_model}: {e}")
                    raise

                if attempt >= max_retries:
                    logger.error(f"Groq API max retries reached on {target_model}: {e}")
                    raise

                # Parse suggested retry delay if provided in error message
                sleep_time = delay + random.uniform(0.1, 0.5)
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                if match:
                    sleep_time = max(sleep_time, float(match.group(1)) + 0.5)
                else:
                    match_ms = re.search(r"try again in (\d+(?:\.\d+)?)ms", err_str, re.IGNORECASE)
                    if match_ms:
                        sleep_time = max(sleep_time, (float(match_ms.group(1)) / 1000.0) + 0.5)

                logger.warning(
                    f"[Attempt {attempt}/{max_retries}] Groq API transient error on {target_model}: "
                    f"({err_str[:120]}...). Sleeping {sleep_time:.2f}s before retry..."
                )
                time.sleep(sleep_time)
                delay *= backoff_factor

        # Final attempt
        response = client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        content = response.choices[0].message.content or ""
        return LLMResponse(
            text=content.strip(),
            tokens_used=max(1, len(content) // 4),
            raw_response=response,
            model=target_model,
            latency_ms=int((time.time() - t_start) * 1000),
        )

    def call_fast_agent(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """
        Fast intermediate agent call using llama-3.1-8b-instant.
        Used for: EvidenceEvaluatorAgent, EntityLinkerAgent, AggregationAgent.
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response_format = {"type": "json_object"} if json_mode else None
        return self.chat_completion(
            messages=messages,
            model=self.fast_model,
            temperature=temperature,
            response_format=response_format,
            **kwargs
        )

    def call_complex_agent(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """
        Complex logic & synthesis call using llama-3.3-70b-versatile.
        Used for: Decomposer, Synthesis, Evaluator LLM Judge, Ladder Router, RAG / GraphRAG generation.
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response_format = {"type": "json_object"} if json_mode else None
        return self.chat_completion(
            messages=messages,
            model=self.complex_model,
            temperature=temperature,
            response_format=response_format,
            **kwargs
        )

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        use_fast_model: bool = False,
        **kwargs
    ) -> tuple[dict[str, Any], LLMResponse]:
        """
        Execute call and return parsed JSON object with fallback code fence handling.
        """
        target_model = model or (self.fast_model if use_fast_model else self.complex_model)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        # Request JSON object if supported
        res = self.chat_completion(
            messages=messages,
            model=target_model,
            response_format={"type": "json_object"},
            **kwargs
        )

        raw = res.text.strip()
        parsed_data = self._clean_and_parse_json(raw)
        return parsed_data, res

    @staticmethod
    def _clean_and_parse_json(raw_text: str) -> dict[str, Any]:
        """Strip markdown fences and parse json safely."""
        text = raw_text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: attempt to find first '{' and last '}'
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except Exception:
                    pass
            logger.warning(f"Failed to parse JSON from LLM output: {raw_text[:200]}")
            return {}


class GeminiEmbeddingClient:
    """
    Google Gemini Embedding client for 768-dim vector embeddings.
    Model: gemini-embedding-001.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, dimension: int = 768):
        self.api_key = api_key or config.gemini.api_key
        self.model = model or config.gemini.embedding_model
        self.dimension = dimension or config.gemini.embedding_dimension
        self._configured = False

    def _ensure_configured(self):
        if not self._configured:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
            self._configured = True

    def embed_content(
        self,
        content: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
        max_retries: int = 6,
        initial_delay: float = 2.0,
        backoff_factor: float = 2.0,
        **kwargs
    ) -> list[float]:
        """
        Embed a single text using Gemini embeddings API with retry logic.
        """
        self._ensure_configured()
        import google.generativeai as genai
        from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

        model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
        delay = initial_delay

        for attempt in range(1, max_retries + 1):
            try:
                result = genai.embed_content(
                    model=model_name,
                    content=content,
                    task_type=task_type,
                    output_dimensionality=self.dimension,
                    **kwargs
                )
                return result["embedding"]

            except (ResourceExhausted, GoogleAPIError, Exception) as e:
                err_str = str(e)
                is_rate_limit = (
                    isinstance(e, ResourceExhausted)
                    or "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "quota" in err_str.lower()
                    or "rate" in err_str.lower()
                )

                if not is_rate_limit and attempt >= max_retries:
                    logger.error(f"Gemini embed_content failed: {e}")
                    raise

                sleep_time = delay
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                if match:
                    sleep_time = max(sleep_time, float(match.group(1)) + 1.0)

                logger.warning(
                    f"[Attempt {attempt}/{max_retries}] Gemini embed_content rate limit hit. "
                    f"Sleeping for {sleep_time:.1f}s before retry..."
                )
                time.sleep(sleep_time)
                delay *= backoff_factor

        # Fallback if somehow loop exited
        result = genai.embed_content(
            model=model_name,
            content=content,
            task_type=task_type,
            output_dimensionality=self.dimension,
            **kwargs
        )
        return result["embedding"]

    def embed_batch(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 20,
    ) -> list[list[float]]:
        """
        Embed a batch of texts safely with rate limit spacing.
        """
        embeddings = []
        for i, text in enumerate(texts):
            try:
                emb = self.embed_content(text, task_type=task_type)
                embeddings.append(emb)
                if (i + 1) % batch_size == 0:
                    logger.info(f"Embedded {i+1}/{len(texts)} texts")
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Embedding failed for text item {i}: {e}. Falling back to zero vector.")
                embeddings.append([0.0] * self.dimension)
        return embeddings


# Module Singletons
_groq_client: Optional[GroqClient] = None
_embedding_client: Optional[GeminiEmbeddingClient] = None


def get_groq_client() -> GroqClient:
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client


def get_embedding_client() -> GeminiEmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = GeminiEmbeddingClient()
    return _embedding_client
