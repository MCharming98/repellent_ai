"""Text formatting helpers and local token counting."""

from __future__ import annotations

import os
from typing import Optional


def format_key_to_subheading(key: str) -> str:
    """Convert snake_case keys to title-cased markdown headings."""
    return key.replace("_", " ").strip().title()


def _google_sentencepiece_processor(repo_id: str, filename: str):
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return spm.SentencePieceProcessor(model_file=path)


def estimate_token_count(
    text: str,
    model_provider: str,
    model_context_window: int,
    model_name: Optional[str] = None,
) -> int:
    """Return a token count for ``text``.

    The function first computes a rough estimate (``len(text) // 2``). If that rough
    estimate is at most 50% of ``model_context_window``, it returns immediately.
    Otherwise, it uses provider-specific token counting for higher accuracy.

    - **openai**: ``tiktoken`` (``encoding_for_model(model_name)`` when possible, else ``cl100k_base``).
    - **anthropic**: PyPI ``anthropic-tokenizer``.
    - **google_genai / gemini***: Google GenAI ``count_tokens`` API.
    - **other**: `Estimate token by character count / 2`
    """
    rough_estimate = len(text) // 2
    if rough_estimate <= model_context_window // 2:
        return rough_estimate

    provider = (model_provider or "").strip().lower()
    model = (model_name or "").strip()

    if provider == "google_genai" or model.startswith("gemini"):
        try:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
            token_response = client.models.count_tokens(
                model=model_name or "gemini-2.0-flash",
                contents=text,
            )
            total_tokens = getattr(token_response, "total_tokens", None)
            if total_tokens is not None:
                return int(total_tokens)
        except Exception:
            # Fall back to rough local estimate if remote counting is unavailable.
            return rough_estimate

    if provider == "openai":
        import tiktoken

        if model_name:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    if provider == "anthropic":
        from anthropic_tokenizer import count_tokens as anthropic_count_tokens

        return int(anthropic_count_tokens(text))

    return rough_estimate
