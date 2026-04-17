"""LangChain helpers: model initialization and agent factory."""

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

# Built-in web search tools per provider (LangChain + provider docs).
_GEMINI_GOOGLE_SEARCH_TOOL = [{"google_search": {}}]
_OPENAI_WEB_SEARCH_TOOL = [{"type": "web_search_preview"}]
_ANTHROPIC_WEB_SEARCH_TOOL = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
]


def _builtin_web_search_tools(model: str, model_provider: str) -> list[dict[str, Any]] | None:
    """
    Return native web-search tool specs for the provider, or None if unsupported.

    - ``google_genai``: Gemini native ``google_search`` (Google AI Studio).
    - ``openai``: Responses API web search preview (ChatGPT).
    - ``anthropic``: Claude server-side web search.
    """
    if model_provider == "google_genai" and "gemini" in model.lower():
        return _GEMINI_GOOGLE_SEARCH_TOOL
    if model_provider == "openai":
        return _OPENAI_WEB_SEARCH_TOOL
    if model_provider == "anthropic":
        return _ANTHROPIC_WEB_SEARCH_TOOL
    return None


def get_llm_agent(
    model: str,
    model_provider: str,
    api_key: str,
    enable_web_search: bool,
    **create_agent_kwargs: Any,
):
    """
    Build a LangChain agent with the given chat model and optional native web search.

    Pass ``response_format=ToolStrategy(...)`` (and any other ``create_agent`` args)
    via keyword arguments.

    When ``enable_web_search`` is True, the model is bound with provider-native web
    search where supported: Gemini (``google_genai``), OpenAI (``openai``), Anthropic
    (``anthropic``). Other providers are unchanged.
    """
    model_kwargs = {"api_key": api_key}
    if model_provider == "google_genai":
        model_kwargs["google_api_key"] = api_key

    chat_model = init_chat_model(
        model, model_provider=model_provider, **model_kwargs
    )
    if enable_web_search:
        tools = _builtin_web_search_tools(model, model_provider)
        if tools:
            chat_model = chat_model.bind_tools(tools)

    return create_agent(model=chat_model, **create_agent_kwargs)


def aggregate_token_usage_from_messages(messages: list[Any]) -> dict[str, int] | None:
    """
    Sum token counts from ``AIMessage.usage_metadata`` across agent steps (if present).
    """
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    found = False
    for msg in messages or []:
        if not isinstance(msg, AIMessage):
            continue
        um = getattr(msg, "usage_metadata", None)
        if not isinstance(um, dict) or not um:
            continue
        found = True
        it = um.get("input_tokens")
        ot = um.get("output_tokens")
        tt = um.get("total_tokens")
        if it is not None:
            input_tokens += int(it)
        if ot is not None:
            output_tokens += int(ot)
        if tt is not None:
            total_tokens += int(tt)
    if not found:
        return None
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def merge_token_usage_totals(
    current: dict[str, int] | None,
    delta: dict[str, int] | None,
) -> dict[str, int]:
    """Sum two usage dicts (``input_tokens``, ``output_tokens``, ``total_tokens``)."""
    keys = ("input_tokens", "output_tokens", "total_tokens")
    out = {k: int((current or {}).get(k, 0)) for k in keys}
    if delta:
        for k in keys:
            out[k] += int(delta.get(k, 0))
    return out
