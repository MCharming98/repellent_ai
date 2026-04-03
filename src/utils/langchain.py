"""LangChain helpers: model initialization and agent factory."""

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

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
