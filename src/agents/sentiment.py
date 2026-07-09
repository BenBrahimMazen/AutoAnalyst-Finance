"""Agent 4 — Sentiment Analyst.

Searches recent news about the company (Tavily), scores each article with
FinBERT, computes an aggregate sentiment, and uses GPT-4o-mini to extract the
key topics being discussed. Degrades gracefully when Tavily, FinBERT, or the
LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.prompts.sentiment import KEY_TOPICS_PROMPT
from src.state.schema import AnalysisState
from src.tools.llm import get_llm, llm_available
from src.tools.web_search import search_web

logger = logging.getLogger(__name__)

_DEFAULT_TOPICS: list[str] = [
    "financial results",
    "analyst coverage",
    "market performance",
    "operations",
    "outlook",
]


def _dedupe_by_url(articles: list[dict]) -> list[dict]:
    """Return ``articles`` with duplicates removed (by URL, then by title)."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[dict] = []
    for art in articles:
        url = (art.get("url") or "").strip().lower()
        title = (art.get("title") or "").strip().lower()
        key = url or title
        if not key or key in seen_urls or title in seen_titles:
            continue
        seen_urls.add(key)
        seen_titles.add(title)
        unique.append(art)
    return unique


def _aggregate_sentiment(avg_positive: float, avg_negative: float) -> str:
    """Classify aggregate sentiment per the spec's thresholds."""
    if avg_positive > 0.5:
        return "bullish"
    if avg_negative > 0.4:
        return "bearish"
    return "neutral"


def _extract_topics(company: str, headlines: list[str]) -> list[str]:
    """Ask GPT-4o-mini for 5 key topics, parsing JSON defensively."""
    if not headlines:
        return list(_DEFAULT_TOPICS)
    if not llm_available():
        logger.warning("OPENAI_API_KEY missing — returning default key topics.")
        return list(_DEFAULT_TOPICS)
    try:
        prompt = KEY_TOPICS_PROMPT.format(company=company, headlines="\n".join(headlines[:20]))
        raw = str(get_llm(temperature=0.2).invoke(prompt).content)
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        topics = json.loads(cleaned)
        if isinstance(topics, list):
            parsed = [str(t).strip() for t in topics if str(t).strip()]
            if parsed:
                return parsed[:5]
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse topics JSON: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Topic extraction failed: %s", exc)
    return list(_DEFAULT_TOPICS)


def run_sentiment_analyst(state: AnalysisState) -> AnalysisState:
    """Search, score, and summarize news sentiment for the company.

    Args:
        state: Pipeline state (uses ``company_name`` and ``ticker``).

    Returns:
        Updated state with ``news_articles``, ``aggregate_sentiment``,
        ``sentiment_positive_avg``, ``sentiment_negative_avg`` and ``key_topics``.
    """
    errors: list[str] = list(state.get("errors", []))
    completed_steps: list[str] = list(state.get("completed_steps", []))

    company: str = state.get("company_name") or state.get("ticker", "")
    ticker: str = state.get("ticker", "")

    # 1. Build three targeted queries.
    queries: list[str] = [
        f"{company} earnings results financial performance 2024",
        f"{company} {ticker} analyst rating price target",
        f"{company} risks controversy regulatory investigation",
    ]

    # 2. Search + 3. dedupe.
    raw_articles: list[dict] = []
    try:
        for query in queries:
            raw_articles.extend(search_web(query, max_results=5))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sentiment_analyst: web search failed — {exc}")
        logger.warning("Web search failed: %s", exc)

    articles = _dedupe_by_url(raw_articles)

    # 4. Score each article with FinBERT.
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    try:
        from src.tools.sentiment_model import FinBERTSentiment

        scorer = FinBERTSentiment.get_instance()
        for art in articles:
            text = f"{art.get('title', '')}. {art.get('content', '')}"[:500]
            score = scorer.score(text)
            art["sentiment"] = score["dominant"]
            art["sentiment_scores"] = {k: score[k] for k in ("positive", "negative", "neutral")}
            positive_scores.append(score["positive"])
            negative_scores.append(score["negative"])
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sentiment_analyst: FinBERT scoring failed — {exc}")
        logger.warning("FinBERT scoring failed: %s", exc)

    # 5. Aggregate.
    avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
    avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
    aggregate = _aggregate_sentiment(avg_positive, avg_negative)

    # 6. Key topics.
    headlines = [a.get("title", "") for a in articles if a.get("title")]
    key_topics = _extract_topics(company, headlines)

    completed_steps.append("sentiment_analyst")

    return {
        **state,
        "news_articles": articles,
        "aggregate_sentiment": aggregate,
        "sentiment_positive_avg": round(avg_positive, 4),
        "sentiment_negative_avg": round(avg_negative, 4),
        "key_topics": key_topics,
        "errors": errors,
        "completed_steps": completed_steps,
    }


if __name__ == "__main__":
    from src.agents.data_collector import run_data_collector

    _state = AnalysisState(
        ticker="AAPL", company_name="", analysis_depth="quick",
        price_history={}, financial_statements={}, macro_data={}, peer_tickers=[],
        valuation_ratios={}, profitability_ratios={}, liquidity_ratios={}, dcf_estimate={},
        peer_comparison={}, fundamental_interpretation="", news_articles=[],
        aggregate_sentiment="neutral", sentiment_positive_avg=0.0, sentiment_negative_avg=0.0,
        key_topics=[], red_flags=[], risk_score=0, risk_summary="",
        executive_summary="", full_report_markdown="", pdf_path="",
        messages=[], errors=[], completed_steps=[],
    )
    _state = run_data_collector(_state)
    _out = run_sentiment_analyst(_state)
    print("Aggregate:", _out["aggregate_sentiment"])
    print("Articles:", len(_out["news_articles"]))
    print("Topics:", _out["key_topics"])
