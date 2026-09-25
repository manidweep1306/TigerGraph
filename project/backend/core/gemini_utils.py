"""
Utility wrapper for Gemini API calls with automatic retry and rate-limiting handling.
Handles 429 / ResourceExhausted errors gracefully with exponential backoff.
"""

import os
import re
import time
import logging
from typing import Any

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

logger = logging.getLogger(__name__)


def generate_content_with_retry(
    model: genai.GenerativeModel,
    prompt: Any,
    max_retries: int = 6,
    initial_delay: float = 3.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> Any:
    """
    Execute model.generate_content with rate-limit and transient error retry logic.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return model.generate_content(prompt, **kwargs)
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
                logger.error(f"Gemini generate_content failed (non-retryable or max retries): {e}")
                raise

            # Try to extract suggested retry delay from error message (e.g., "Please retry in 48.67s")
            sleep_time = delay
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
            if match:
                suggested = float(match.group(1)) + 1.0
                sleep_time = max(sleep_time, suggested)

            logger.warning(
                f"[Attempt {attempt}/{max_retries}] Gemini API rate limit hit ({err_str[:120]}...). "
                f"Sleeping for {sleep_time:.1f}s before retry..."
            )
            time.sleep(sleep_time)
            delay *= backoff_factor

    # Final attempt
    return model.generate_content(prompt, **kwargs)


def embed_content_with_retry(
    model: str,
    content: Any,
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: int = 768,
    max_retries: int = 6,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> Any:
    """
    Execute genai.embed_content with rate-limit retry logic.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return genai.embed_content(
                model=model,
                content=content,
                task_type=task_type,
                output_dimensionality=output_dimensionality,
                **kwargs
            )
        except (ResourceExhausted, GoogleAPIError, Exception) as e:
            err_str = str(e)
            is_rate_limit = (
                isinstance(e, ResourceExhausted)
                or "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "quota" in err_str.lower()
            )

            if not is_rate_limit and attempt >= max_retries:
                logger.error(f"Gemini embed_content failed: {e}")
                raise

            sleep_time = delay
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
            if match:
                suggested = float(match.group(1)) + 1.0
                sleep_time = max(sleep_time, suggested)

            logger.warning(
                f"[Attempt {attempt}/{max_retries}] Gemini embed_content rate limit hit. "
                f"Sleeping for {sleep_time:.1f}s before retry..."
            )
            time.sleep(sleep_time)
            delay *= backoff_factor

    return genai.embed_content(
        model=model,
        content=content,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        **kwargs
    )
