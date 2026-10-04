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
import threading
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


class GroqKeyEntry:
    """Represents a single Groq API key with individual cooldown tracking."""
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self._client = None
        self.cooldown_until: float = 0.0

    @property
    def client(self):
        if self._client is None:
            try:
                from groq import Groq
                if self.api_key and not self.api_key.startswith("your_"):
                    self._client = Groq(api_key=self.api_key)
                else:
                    self._client = None
            except ImportError:
                self._client = None
        return self._client

    def is_available(self, now: float) -> bool:
        return now >= self.cooldown_until

    def set_cooldown(self, seconds: float):
        self.cooldown_until = time.time() + seconds

    def create_completion(self, **call_kwargs):
        """Invoke completion via Groq SDK or standard library HTTP fallback."""
        if self.client is not None:
            return self.client.chat.completions.create(**call_kwargs)

        import urllib.request
        import urllib.error
        payload = json.dumps(call_kwargs).encode("utf-8")
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

                class _MockChoice:
                    def __init__(self, c):
                        self.message = type("Msg", (), {"content": c.get("message", {}).get("content", "")})()
                class _MockUsage:
                    def __init__(self, u):
                        self.total_tokens = u.get("total_tokens", 0)
                class _MockResp:
                    def __init__(self, d):
                        self.choices = [_MockChoice(c) for c in d.get("choices", [])]
                        self.usage = _MockUsage(d.get("usage", {}))
                return _MockResp(data)
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"Groq HTTP {e.code}: {err_body or e.reason}")


class GroqRotator:
    """Thread-safe round-robin rotator for Groq API keys with 429 cooldown."""
    def __init__(self, api_keys: list[str]):
        clean_keys = [k.strip() for k in api_keys if k.strip() and not k.strip().startswith("your_")]
        if not clean_keys:
            clean_keys = [""]
        self.entries = [GroqKeyEntry(k) for k in clean_keys]
        self._index = 0
        self._lock = threading.Lock()

    def get_client_for_call(self) -> tuple[Optional[Any], GroqKeyEntry, float]:
        """
        Returns (client, entry, wait_seconds).
        wait_seconds > 0 only if ALL keys are currently in cooldown.
        """
        with self._lock:
            now = time.time()
            n = len(self.entries)
            # Find next ready key
            for i in range(n):
                candidate_idx = (self._index + i) % n
                entry = self.entries[candidate_idx]
                if entry.is_available(now):
                    self._index = (candidate_idx + 1) % n
                    return entry.client, entry, 0.0

            # All keys are in cooldown; find the earliest available
            earliest = min(self.entries, key=lambda e: e.cooldown_until)
            wait_time = max(0.1, earliest.cooldown_until - now)
            self._index = (self.entries.index(earliest) + 1) % n
            return earliest.client, earliest, wait_time


class GroqClient:
    """
    Robust Groq API client with thread-safe multi-key rotation, 429 backoff,
    exponential retry, and token-optimized agent dispatches.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        fast_model: Optional[str] = None,
        complex_model: Optional[str] = None,
    ):
        raw_keys = (
            os.environ.get("GROQ_API_KEYS", "")
            or os.environ.get("GROQ_API_KEY", "")
            or config.groq.api_key
        )
        keys_list = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if api_key and api_key not in keys_list:
            keys_list.insert(0, api_key)

        self.rotator = GroqRotator(keys_list)
        # Default fast model to llama-3.1-8b-instant per prompt specifications
        self.fast_model = fast_model or os.environ.get("GROQ_FAST_MODEL") or "llama-3.1-8b-instant"
        self.complex_model = complex_model or config.groq.complex_model

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
        Execute Groq chat completion with multi-key rotation and rate-limit cooldown.
        """
        target_model = model or self.complex_model
        t_start = time.time()
        delay = initial_delay

        normalized_messages = [dict(m) for m in messages]
        if response_format and response_format.get("type") == "json_object":
            has_json = any("json" in m.get("content", "").lower() for m in normalized_messages)
            if not has_json and normalized_messages:
                normalized_messages[-1]["content"] = normalized_messages[-1]["content"] + "\nRespond in valid JSON format."

        for attempt in range(1, max_retries + 1):
            client, entry, wait_time = self.rotator.get_client_for_call()

            if wait_time > 0:
                logger.info(f"All Groq API keys cooling down. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

            if not entry.api_key:
                logger.warning("GROQ_API_KEY is not set. Groq API calls will fail if not authenticated.")

            try:
                call_kwargs: dict[str, Any] = {
                    "model": target_model,
                    "messages": normalized_messages,
                    "temperature": temperature,
                    **kwargs
                }
                if response_format is not None:
                    call_kwargs["response_format"] = response_format
                if max_tokens is not None:
                    call_kwargs["max_tokens"] = max_tokens

                response = entry.create_completion(**call_kwargs)
                latency_ms = int((time.time() - t_start) * 1000)

                content = response.choices[0].message.content or ""
                total_tokens = 0
                if hasattr(response, "usage") and response.usage:
                    total_tokens = getattr(response.usage, "total_tokens", 0) or 0
                if total_tokens <= 0:
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

                if is_rate_limit:
                    cooldown = random.uniform(5.0, 8.0)
                    entry.set_cooldown(cooldown)
                    key_hint = f"...{entry.api_key[-4:]}" if len(entry.api_key) > 4 else ""
                    logger.warning(
                        f"Groq key {key_hint} hit 429 rate limit. "
                        f"Cooling down {cooldown:.1f}s and rotating to next key..."
                    )
                    # Check if another key is immediately ready
                    with self.rotator._lock:
                        now = time.time()
                        ready = any(e.is_available(now) for e in self.rotator.entries)
                    if ready:
                        continue  # Immediate retry with next key!

                is_retryable = is_rate_limit or is_server_error
                if not is_retryable:
                    logger.error(f"Groq API non-retryable error on {target_model}: {e}")
                    raise

                if attempt >= max_retries:
                    if target_model != "openai/gpt-oss-20b":
                        logger.warning(f"Falling back from {target_model} to openai/gpt-oss-20b due to limit/retry.")
                        target_model = "openai/gpt-oss-20b"
                        call_kwargs["model"] = "openai/gpt-oss-20b"
                        try:
                            client, entry, _ = self.rotator.get_client_for_call()
                            response = entry.create_completion(**call_kwargs)
                            content = response.choices[0].message.content or ""
                            return LLMResponse(
                                text=content.strip(),
                                tokens_used=max(1, len(content) // 4),
                                raw_response=response,
                                model=target_model,
                                latency_ms=int((time.time() - t_start) * 1000),
                            )
                        except Exception as e2:
                            logger.error(f"Fallback to 20b also failed: {e2}")
                    logger.error(f"Groq API max retries reached on {target_model}: {e}")
                    raise

                sleep_time = delay + random.uniform(0.1, 0.5)
                logger.warning(
                    f"[Attempt {attempt}/{max_retries}] Groq API error on {target_model}: "
                    f"({err_str[:120]}...). Sleeping {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)
                delay *= backoff_factor

        # Final fallback attempt
        client, entry, _ = self.rotator.get_client_for_call()
        response = entry.create_completion(
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
        max_tokens: int = 150,
        **kwargs
    ) -> LLMResponse:
        """
        Fast intermediate agent call using fast model (default max 150 tokens).
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
            max_tokens=max_tokens,
            **kwargs
        )

    def call_complex_agent(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 350,
        **kwargs
    ) -> LLMResponse:
        """
        Complex logic & synthesis call using complex model (default max 350 tokens).
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
            max_tokens=max_tokens,
            **kwargs
        )

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        use_fast_model: bool = False,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> tuple[dict[str, Any], LLMResponse]:
        """
        Execute call and return parsed JSON object with fallback code fence handling.
        """
        target_model = model or (self.fast_model if use_fast_model else self.complex_model)
        if max_tokens is None:
            max_tokens = 150 if use_fast_model else 350

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        res = self.chat_completion(
            messages=messages,
            model=target_model,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
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
