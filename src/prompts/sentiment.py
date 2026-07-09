"""Prompt templates for the sentiment analyst agent."""

from __future__ import annotations

KEY_TOPICS_PROMPT: str = """
Given these news headlines about {company}, identify the 5 most important
topics or themes being discussed. Return ONLY a JSON array of 5 short strings.
No preamble, no explanation, just the JSON array.

Headlines:
{headlines}

Example output: ["earnings beat", "China revenue concerns", "AI investment", "share buyback", "regulatory scrutiny"]
"""
