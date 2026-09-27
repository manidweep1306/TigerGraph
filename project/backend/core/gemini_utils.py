"""
Backward-compatibility adapter for gemini_utils.
Delegates to backend.core.llm_client for Groq generation and Gemini embeddings.
"""

import logging
from typing import Any
from backend.core.llm_client import get_groq_client, get_embedding_client

logger = logging.getLogger(__name__)


def generate_content_with_retry(
    model: Any,
    prompt: Any,
    max_retries: int = 6,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> Any:
    """
    Adapter bridging old generate_content calls to the updated LLM client.
    """
    groq = get_groq_client()
    system_instruction = getattr(model, "system_instruction", None)
    if hasattr(system_instruction, "text"):
        system_instruction = system_instruction.text

    prompt_str = prompt if isinstance(prompt, str) else str(prompt)
    res = groq.chat_completion(
        messages=[
            *( [{"role": "system", "content": str(system_instruction)}] if system_instruction else [] ),
            {"role": "user", "content": prompt_str}
        ],
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        **kwargs
    )

    class GeminiResponseAdapter:
        def __init__(self, text: str):
            self.text = text

    return GeminiResponseAdapter(res.text)


def embed_content_with_retry(
    model: str,
    content: Any,
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: int = 768,
    max_retries: int = 6,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> dict[str, Any]:
    """
    Adapter for Gemini embeddings API.
    """
    client = get_embedding_client()
    embedding = client.embed_content(
        content=content if isinstance(content, str) else str(content),
        task_type=task_type,
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        **kwargs
    )
    return {"embedding": embedding}
