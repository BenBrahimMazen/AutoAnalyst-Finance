"""FinBERT sentiment scoring tool.

Wraps the ``ProsusAI/finbert`` model in a singleton so the ~1.3 GB model is
downloaded and loaded into memory exactly once per process, then reused for
every article.
"""

from __future__ import annotations

import logging

from transformers import pipeline

try:  # torch is optional at import time but required to actually run inference
    import torch  # noqa: F401
    _TORCH_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# FinBERT emits these three labels.
_LABELS: tuple[str, ...] = ("positive", "negative", "neutral")


class FinBERTSentiment:
    """Singleton wrapper around ``ProsusAI/finbert``.

    The model is loaded lazily on first instantiation and cached on the class,
    so repeated ``get_instance()`` calls return the same object.
    """

    _instance: "FinBERTSentiment | None" = None

    @classmethod
    def get_instance(cls) -> "FinBERTSentiment":
        """Return the shared :class:`FinBERTSentiment` instance, loading once."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Initialize the HuggingFace text-classification pipeline.

        Uses ``top_k=None`` (the modern replacement for the removed
        ``return_all_scores=True``) so every label score is returned. Runs on GPU
        when CUDA is available, otherwise CPU.
        """
        device: int = -1
        try:
            if _TORCH_AVAILABLE:
                import torch

                device = 0 if torch.cuda.is_available() else -1
        except Exception:  # noqa: BLE001
            device = -1

        self.pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,  # return scores for all labels
            device=device,
        )

    def score(self, text: str) -> dict:
        """Score a single piece of text.

        Args:
            text: Headline / article snippet. Truncated to 512 characters.

        Returns:
            Dict like ``{"positive": 0.9, "negative": 0.05, "neutral": 0.05,
            "dominant": "positive"}``.
        """
        text = (text or "")[:512]
        results = self.pipe(text)[0]  # list of {label, score}
        score_dict = {s["label"].lower(): float(s["score"]) for s in results}
        # Guarantee all three labels are present.
        for label in _LABELS:
            score_dict.setdefault(label, 0.0)
        dominant = max(_LABELS, key=lambda lbl: score_dict[lbl])
        return {**score_dict, "dominant": dominant}

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Score a list of texts.

        Args:
            texts: Headlines / snippets to score.

        Returns:
            One result dict per input text (see :meth:`score`).
        """
        return [self.score(t) for t in texts]
