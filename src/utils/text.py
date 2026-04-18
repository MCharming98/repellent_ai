"""Text formatting helpers and local token counting."""

from __future__ import annotations

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
    model_name: Optional[str] = None,
) -> int:
    """Return a token count for ``text`` using local tokenizer libraries.

    - **openai**: ``tiktoken`` (``encoding_for_model(model_name)`` when possible, else ``cl100k_base``).
    - **anthropic**: PyPI ``anthropic-tokenizer``.
    - **other**: `Estimate token by character count / 2`
    """
    provider = (model_provider or "").strip().lower()

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
    
    return len(text) // 2
