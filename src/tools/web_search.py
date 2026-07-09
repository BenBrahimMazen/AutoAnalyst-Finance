"""Web search tool (Tavily) for the sentiment analyst.

Thin, defensive wrapper around ``langchain_community``'s ``TavilySearchResults``
so the agent doesn't have to worry about missing API keys or transport errors.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def tavily_available() -> bool:
    """Return ``True`` if a Tavily API key is configured."""
    return bool(os.getenv("TAVILY_API_KEY"))


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Run a single Tavily search.

    Args:
        query: Free-text search query.
        max_results: Maximum number of results to return.

    Returns:
        List of result dicts (each with ``title``, ``url``/``link``, ``content``).
        Returns an empty list if Tavily is unavailable or the call fails.
    """
    if not tavily_available():
        logger.warning("TAVILY_API_KEY missing — web search disabled for query %r.", query)
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tool = TavilySearchResults(max_results=max_results)
        raw: Any = tool.invoke(query)
        # Normalize result keys (Tavily returns 'url' + 'content'; some forks use 'link').
        normalized: list[dict] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            normalized.append({
                "title": item.get("title", ""),
                "url": item.get("url") or item.get("link") or "",
                "content": item.get("content", ""),
            })
        return normalized
    except Exception as exc:  # noqa: BLE001 - search must never crash the pipeline
        logger.warning("Tavily search failed for %r: %s", query, exc)
        return []
