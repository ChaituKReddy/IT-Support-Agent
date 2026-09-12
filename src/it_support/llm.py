"""Chat model factory.

Two providers are supported and selected with the LLM_PROVIDER setting:

* ``ollama``    - a local model, the default, so the project runs with no API key.
* ``anthropic`` - the Claude API, for stronger tool-calling when a key is set.

Both return a LangChain chat model, so the graph does not care which is active.
"""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the selected provider is not usable with current settings."""


def _build_ollama(settings: Settings) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
        # The chosen model exposes a reasoning mode. It is disabled so the
        # assistant answers directly instead of emitting a thinking preamble.
        reasoning=False,
        validate_model_on_init=False,
    )


def _build_anthropic(settings: Settings) -> BaseChatModel:
    if not settings.anthropic_api_key:
        raise LLMConfigurationError(
            "LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set. "
            "Add the key to .env, or switch LLM_PROVIDER back to 'ollama'."
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )


_BUILDERS = {"ollama": _build_ollama, "anthropic": _build_anthropic}


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Create the chat model for the configured provider."""
    settings = settings or get_settings()
    builder = _BUILDERS.get(settings.llm_provider)
    if builder is None:  # pragma: no cover - guarded by pydantic validation
        raise LLMConfigurationError(f"Unknown LLM provider: {settings.llm_provider}")
    logger.info("Building chat model for provider %s", settings.llm_provider)
    return builder(settings)


def build_agent_model(
    tools: Sequence[BaseTool], settings: Settings | None = None
) -> BaseChatModel:
    """Create the chat model with the agent's tools bound to it."""
    return build_chat_model(settings).bind_tools(list(tools))


def describe_provider(settings: Settings | None = None) -> str:
    """Human-readable provider label, used by the user interface."""
    settings = settings or get_settings()
    if settings.llm_provider == "anthropic":
        return f"Anthropic · {settings.anthropic_model}"
    return f"Ollama · {settings.ollama_model}"
