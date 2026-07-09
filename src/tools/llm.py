"""Shared LLM client factory.

All agents use LangChain's ``ChatOpenAI`` wrapper, which speaks the OpenAI
Chat Completions API. Because that protocol is a de-facto standard, the same
client works against **any OpenAI-compatible provider** — OpenAI itself, Google
Gemini, Groq, OpenRouter, Cerebras, Together, or a local Ollama server.

The provider is chosen entirely through environment variables (see ``.env``):

* ``LLM_API_KEY``  — secret key for the chosen provider (falls back to
  ``OPENAI_API_KEY`` for backward compatibility).
* ``LLM_BASE_URL`` — the provider's OpenAI-compatible endpoint
  (falls back to ``OPENAI_API_BASE``). Leave unset for OpenAI.
* ``LLM_MODEL``    — the model id (falls back to ``OPENAI_MODEL``, then
  :data:`DEFAULT_MODEL`).

Centralizing the factory keeps these knobs in one place and lets us swap
providers without touching every agent.
"""

from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Default model when nothing is configured. Used by OpenAI; other providers
# override via LLM_MODEL (e.g. "gemini-2.0-flash", "llama-3.3-70b-versatile").
DEFAULT_MODEL: str = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"


def _api_key() -> str | None:
    """Return the configured provider API key, or ``None``."""
    return os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")


def _base_url() -> str | None:
    """Return the configured OpenAI-compatible base URL, or ``None``."""
    return os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_API_BASE")


def get_llm(temperature: float = 0.3, model: str | None = None) -> ChatOpenAI:
    """Return a configured ``ChatOpenAI`` instance for the active provider.

    Args:
        temperature: Sampling temperature. Low values (0.0-0.3) keep financial
            analysis deterministic and grounded.
        model: Override model id. Defaults to the ``LLM_MODEL`` env var, then
            :data:`DEFAULT_MODEL`.

    Returns:
        A ``ChatOpenAI`` client bound to the configured provider/model.

    Raises:
        ValueError: If no API key is configured. Agents catch this and degrade
            gracefully (boilerplate text instead of LLM output).
    """
    api_key: str | None = _api_key()
    if not api_key:
        raise ValueError(
            "No LLM API key configured. Set LLM_API_KEY (or OPENAI_API_KEY) in "
            "your environment or .env file."
        )

    kwargs: dict = {
        "model": model or DEFAULT_MODEL,
        "temperature": temperature,
        "api_key": api_key,
    }
    base_url = _base_url()
    if base_url:
        kwargs["base_url"] = base_url

    provider_hint = base_url or "OpenAI (default)"
    logger.debug("Instantiating ChatOpenAI via %s, model=%s", provider_hint, kwargs["model"])
    return ChatOpenAI(**kwargs)


def llm_available() -> bool:
    """Return ``True`` if an LLM API key is configured for any provider."""
    return bool(_api_key())


def llm_provider_label() -> str:
    """Human-readable description of the active provider (for logs/UI)."""
    base = _base_url()
    if not base:
        return "OpenAI"
    base = base.rstrip("/")
    if "generativelanguage.googleapis.com" in base:
        return "Google Gemini"
    if "groq.com" in base:
        return "Groq"
    if "openrouter.ai" in base:
        return "OpenRouter"
    if "localhost:11434" in base or "127.0.0.1:11434" in base:
        return "Ollama (local)"
    return base
