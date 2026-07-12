"""Offline smoke tests for the FinBERT evaluation harness.

These exercise only the *pure* helpers in ``scripts/eval_finbert.py`` — label
canonicalization, the VADER threshold mapping, and the metrics aggregator — on
toy input. They deliberately do **not** download the dataset or load the model,
so they run in the default offline ``pytest -m "not network"`` suite.

The script lives outside the ``src``/``tests`` package tree, so we load it by
path rather than importing it as a module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# importing the module is side-effect-free: heavy deps (datasets, vaderSentiment,
# transformers) are imported lazily inside functions, never at module top level.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_finbert.py"
_spec = importlib.util.spec_from_file_location("eval_finbert", _SCRIPT)
eval_finbert = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(eval_finbert)


def test_canonicalize_label_handles_variants() -> None:
    """Aliases, casing, and whitespace all collapse to the canonical labels."""
    assert eval_finbert.canonicalize_label("Positive") == "positive"
    assert eval_finbert.canonicalize_label(" NEG ") == "negative"
    assert eval_finbert.canonicalize_label("neutral") == "neutral"
    # PhraseBank/FinBERT shape (the app lowercases these too).
    assert eval_finbert.canonicalize_label("positive") == "positive"


def test_canonicalize_label_rejects_unknown() -> None:
    """An unmappable label raises rather than silently producing a mismatch."""
    try:
        eval_finbert.canonicalize_label("kinda_positive")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unrecognized label")


def test_vader_compound_thresholds() -> None:
    """VADER compound maps to three classes via the standard +/-0.05 cutoffs."""
    to_label = eval_finbert.vader_compound_to_label
    assert to_label(0.9) == "positive"
    assert to_label(0.05) == "positive"   # boundary is inclusive
    assert to_label(-0.9) == "negative"
    assert to_label(-0.05) == "negative"  # boundary is inclusive
    assert to_label(0.0) == "neutral"
    assert to_label(0.049) == "neutral"
    assert to_label(-0.049) == "neutral"


def test_compute_metrics_shape_and_perfect_case() -> None:
    """A perfect prediction yields 1.0 everywhere; structure matches the contract."""
    y_true = ["negative", "neutral", "positive", "negative", "positive"]
    perfect = eval_finbert.compute_metrics(y_true, y_true)
    assert perfect["accuracy"] == 1.0
    assert perfect["macro_f1"] == 1.0
    # Every canonical label is present in the per-class breakdown.
    assert set(perfect["per_class"]) == {"negative", "neutral", "positive"}
    # Confusion matrix is 3x3 (one row/col per label), rows sum to supports.
    cm = perfect["confusion_matrix"]
    assert len(cm) == 3 and all(len(row) == 3 for row in cm)
    supports = [perfect["per_class"][lbl]["support"] for lbl in ("negative", "neutral", "positive")]
    assert [sum(row) for row in cm] == supports


def test_compute_metrics_imperfect() -> None:
    """One wrong prediction lowers accuracy but keeps the matrix consistent."""
    y_true = ["positive", "positive", "negative", "negative", "neutral"]
    y_pred = ["positive", "negative", "negative", "negative", "neutral"]  # 1 error
    m = eval_finbert.compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.8
    assert 0.0 < m["macro_f1"] < 1.0
    # Total predictions counted = number of examples.
    total = sum(sum(row) for row in m["confusion_matrix"])
    assert total == len(y_true)
