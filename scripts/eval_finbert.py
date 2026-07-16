"""Evaluate the project's FinBERT sentiment model on Financial PhraseBank.

This is a *reproducible* offline evaluation (not a pytest). It answers a
question the rest of the system never asks: *how good is the FinBERT model the
app actually runs?* To do that it reuses the exact same loader the product uses
(``src.tools.sentiment_model.FinBERTSentiment``) — no second model is
instantiated — and scores every sentence in the Financial PhraseBank benchmark
against a VADER rule-based baseline.

What it computes, for both FinBERT and VADER:

* accuracy and macro-F1,
* per-class precision / recall / F1 (via ``sklearn.classification_report``),
* a confusion matrix (rendered to PNG),
* an error analysis: every FinBERT misclassification is dumped with the true
  label, predicted label, and class probabilities, and a short human-readable
  summary buckets the failure modes.

Artifacts are written to a *tracked* ``evaluation/`` directory (not the
gitignored ``reports/``) so the numbers and plots live in version control.

Usage::

    pip install -r requirements-eval.txt
    python scripts/eval_finbert.py                 # full sentences_75agree set
    python scripts/eval_finbert.py --limit 200     # quick smoke run
    python scripts/eval_finbert.py --config sentences_50agree

The module's top level is import-safe (no dataset download, no model load), so
the pure helpers can be unit-tested in ``tests/test_eval.py`` without network.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Sequence

# sklearn is a hard eval dependency and is light enough to import eagerly so the
# metrics helper can be unit-tested in isolation.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# --- Repository layout -------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
# Ensure the project root is importable as `src` when this script is run
# standalone (mirrors scripts/run_analysis.py). Import-safe: only touches
# sys.path, so importing the module for unit tests has no other side effect.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_EVAL_DIR = _REPO_ROOT / "evaluation"

# Canonical three-class label set. Order is fixed (alphabetical) so that
# confusion matrices and per-class reports are consistent everywhere.
LABELS: tuple[str, ...] = ("negative", "neutral", "positive")

# Same truncation the app applies in FinBERTSentiment.score().
_MAX_CHARS = 512

# Number of (most-confidently-wrong) errors rendered in SUMMARY.md's table. The
# full error set is always written to finbert_errors.json regardless of this.
_ERROR_SAMPLE_SIZE: int = 40


# ---------------------------------------------------------------------------
# Pure, import-safe helpers (unit-tested in tests/test_eval.py)
# ---------------------------------------------------------------------------
def canonicalize_label(label: object) -> str:
    """Normalize a label string to one of :data:`LABELS`.

    FinBERT already emits lowercase ``positive``/``negative``/``neutral`` and
    the app lowercases them again, while Financial PhraseBank's integer labels
    map to the same strings via the dataset's ``ClassLabel.names``. The single
    most common evaluation bug is a silent string mismatch between the two sides
    that collapses accuracy to ~1/3 — centralizing the canonicalization here
    makes that impossible.
    """
    text = str(label).strip().lower()
    aliases = {
        "pos": "positive",
        "bullish": "positive",
        "1": "positive",
        "neg": "negative",
        "bearish": "negative",
        "0": "negative",
        "neu": "neutral",
        "2": "neutral",
    }
    text = aliases.get(text, text)
    if text not in LABELS:
        raise ValueError(f"Unrecognized sentiment label: {label!r} -> {text!r}")
    return text


def vader_compound_to_label(
    compound: float, pos_thresh: float = 0.05, neg_thresh: float = -0.05
) -> str:
    """Map a VADER compound score to a three-class label.

    Uses VADER's standard thresholds: compound >= 0.05 is positive,
    <= -0.05 is negative, otherwise neutral.
    """
    if compound >= pos_thresh:
        return "positive"
    if compound <= neg_thresh:
        return "negative"
    return "neutral"


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """Return accuracy, macro-F1, per-class metrics, and the confusion matrix.

    Args:
        y_true: Canonicalized ground-truth labels.
        y_pred: Canonicalized predicted labels.

    Returns:
        Dict with ``accuracy``, ``macro_f1``, ``per_class`` (one sub-dict per
        class with precision/recall/f1/support), and ``confusion_matrix`` (rows
        = true class, columns = predicted class, ordered as :data:`LABELS`).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    report = classification_report(
        y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0
    )
    per_class = {
        lbl: {
            "precision": report[lbl]["precision"],
            "recall": report[lbl]["recall"],
            "f1": report[lbl]["f1-score"],
            "support": int(report[lbl]["support"]),
        }
        for lbl in LABELS
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro")),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# Financial PhraseBank source: the dataset ships its data as a zip of raw text
# files on the Hub (``takala/financial_phrasebank``). Each config maps to one
# ``Sentences_<agree>.txt`` inside that zip.
_PHRASEBANK_REPO = "takala/financial_phrasebank"
_PHRASEBANK_ZIP = "data/FinancialPhraseBank-v1.0.zip"
_CONFIG_FILES: dict[str, str] = {
    "sentences_50agree": "Sentences_50Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_allagree": "Sentences_AllAgree.txt",
}
# Source lines are ``<sentence>@<label>`` (latin-1). Greedy ``.+`` ensures we
# split on the *last* ``@`` so a stray ``@`` inside a sentence can't fool us.
_LINE_RE = re.compile(r"^(?P<text>.+)@+(?P<label>negative|neutral|positive)\s*$")


def load_phrasebank(config: str = "sentences_75agree") -> tuple[list[str], list[str]]:
    """Load Financial PhraseBank and return ``(texts, canonical_labels)``.

    We read the raw ``Sentences_<agree>.txt`` file from the dataset's source
    archive rather than going through the ``datasets`` loading script: ``datasets``
    4+ no longer runs dataset scripts, and — more importantly — parsing the
    source ourselves yields the labels as plain strings (``negative`` /
    ``neutral`` / ``positive``), so there is no integer<->label mapping to get
    wrong. That mapping is the classic source of the ~0.33-accuracy bug, and
    sidestepping it entirely is more robust than defending against it.

    The zip is fetched once via ``huggingface_hub`` (a transitive dependency of
    ``transformers``) and cached on disk.
    """
    import zipfile

    from huggingface_hub import hf_hub_download

    if config not in _CONFIG_FILES:
        raise ValueError(
            f"Unknown config {config!r}; expected one of {sorted(_CONFIG_FILES)}."
        )
    zip_path = hf_hub_download(
        repo_id=_PHRASEBANK_REPO, repo_type="dataset", filename=_PHRASEBANK_ZIP
    )
    target = _CONFIG_FILES[config]
    with zipfile.ZipFile(zip_path) as zf:
        inner = next(n for n in zf.namelist() if n.endswith(target))
        raw = zf.read(inner).decode("latin-1")  # the source file is latin-1

    texts: list[str] = []
    labels: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = _LINE_RE.match(line)
        if not m:  # skip macOS resource-fork (``__MACOSX/...``) and stray lines
            continue
        texts.append(m.group("text").strip())
        labels.append(canonicalize_label(m.group("label")))
    if not texts:
        raise RuntimeError(
            f"No rows parsed from {target!r} — check the delimiter/encoding."
        )
    return texts, labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def score_finbert(texts: Sequence[str], batch_size: int = 64) -> tuple[list[str], list[dict]]:
    """Score texts with the *app's* FinBERT singleton.

    Reuses ``FinBERTSentiment.get_instance()`` — the same model the product runs
    — and applies the identical label handling as its ``score()`` method
    (``[:512]`` truncation, lowercased labels, argmax over the three classes).
    We drive the singleton's underlying HuggingFace pipeline directly so the
    ~3k sentences are scored in batches rather than one at a time.
    """
    from src.tools.sentiment_model import FinBERTSentiment

    pipe = FinBERTSentiment.get_instance().pipe  # the app's loaded pipeline
    preds: list[str] = []
    probs: list[dict] = []
    truncated = [(t or "")[:_MAX_CHARS] for t in texts]
    n = len(truncated)
    for start in range(0, n, batch_size):
        batch = truncated[start : start + batch_size]
        rows = pipe(batch)  # top_k=None is baked into the pipeline
        for row in rows:
            scores = {s["label"].lower(): float(s["score"]) for s in row}
            for lbl in LABELS:
                scores.setdefault(lbl, 0.0)
            dominant = max(LABELS, key=lambda lbl: scores[lbl])
            preds.append(dominant)
            probs.append({lbl: scores[lbl] for lbl in LABELS})
        done = min(start + batch_size, n)
        print(f"  FinBERT scored {done}/{n} sentences", flush=True)
    return preds, probs


def score_vader(texts: Sequence[str]) -> tuple[list[str], list[float]]:
    """Score texts with VADER; return ``(labels, compound_scores)``."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    compounds = [analyzer.polarity_scores(t or "")["compound"] for t in texts]
    labels = [vader_compound_to_label(c) for c in compounds]
    return labels, compounds


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: Sequence[Sequence[int]], title: str, out_path: Path
) -> None:
    """Render a confusion matrix heatmap to ``out_path`` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")  # headless backend — no display required
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    cm_arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        cbar=True,
        square=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
# Lightweight cue lexicons used to *approximately* bucket misclassifications.
# These are heuristics for surfacing patterns, not ground truth.
_NEGATION_CUES = re.compile(
    r"\b(not|no|never|without|fail(ed|s)? to|unable|neither|nor|despite|although|however|but|yet|rather than|instead of)\b",
    re.IGNORECASE,
)
_POSITIVE_CUES = re.compile(
    r"\b(rose|increase(d|s)?|growth|grew|profit|gain|higher|up|boost|surge|record|strong|improve(d|s)?|beat|exceed|positive|net sales|sold|agreed|acquire)\b",
    re.IGNORECASE,
)
_NEGATIVE_CUES = re.compile(
    r"\b(fell|decrease(d|s)?|loss|drop|lower|down|weak|decline(d|s)?|cut|reduce(d|s)?|negative|war|attack|sued|charges|filed|lowered|missed)\b",
    re.IGNORECASE,
)


def _bucket_error(text: str) -> str:
    """Return a rough failure-mode bucket for one misclassified sentence."""
    has_neg = bool(_NEGATION_CUES.search(text))
    has_pos = bool(_POSITIVE_CUES.search(text))
    has_negcue = bool(_NEGATIVE_CUES.search(text))
    if has_pos and has_negcue:
        return "mixed-signal (both positive and negative cues)"
    if has_neg:
        return "negation / contrast (failed to, despite, however, ...)"
    if has_negcue and not has_pos:
        return "domain vocabulary (financial event verbs)"
    return "other / subtle phrasing"


def build_error_analysis(
    texts: Sequence[str],
    y_true: Sequence[str],
    finbert_pred: Sequence[str],
    finbert_probs: Sequence[dict],
) -> list[dict]:
    """Collect *all* FinBERT misclassifications with context, sorted for inspection.

    Returns every error, most-confidently-wrong first. The full list is what gets
    written to ``finbert_errors.json``; callers that want a short readable sample
    (e.g. for the SUMMARY table) should slice it themselves.
    """
    errors: list[dict] = []
    for i, (t, yt, yp, p) in enumerate(zip(texts, y_true, finbert_pred, finbert_probs)):
        if yt == yp:
            continue
        confidence = p[yp]
        margin = confidence - max(
            p[lbl] for lbl in LABELS if lbl != yp
        )  # how separated the winner is from runner-up
        errors.append(
            {
                "text": t,
                "true": yt,
                "predicted": yp,
                "probabilities": {k: round(v, 4) for k, v in p.items()},
                "confidence": round(confidence, 4),
                "margin": round(margin, 4),
                "bucket": _bucket_error(t),
                "index": i,
            }
        )
    # Sort by how confidently wrong (most over-confident first) for visibility.
    errors.sort(key=lambda e: e["margin"], reverse=True)
    return errors


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def _metrics_table_row(name: str, m: dict) -> str:
    pc = m["per_class"]
    return (
        f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
        f"{pc['negative']['precision']:.2f}/{pc['negative']['recall']:.2f} | "
        f"{pc['neutral']['precision']:.2f}/{pc['neutral']['recall']:.2f} | "
        f"{pc['positive']['precision']:.2f}/{pc['positive']['recall']:.2f} |"
    )


def write_summary(
    out_path: Path,
    config: str,
    n: int,
    label_dist: dict,
    finbert: dict,
    vader: dict,
    error_sample: list[dict],
) -> None:
    """Write a human-readable ``SUMMARY.md`` from the computed numbers."""
    lines: list[str] = []
    lines.append("# FinBERT evaluation — Financial PhraseBank\n")
    lines.append(
        "A reproducible benchmark of the same FinBERT model the application runs "
        "(`src.tools.sentiment_model.FinBERTSentiment`) against the Financial "
        "PhraseBank dataset, with a VADER rule-based baseline for context. The "
        "goal is rigor over demo: a measured, honest picture of where the model "
        "earns its keep and where it breaks.\n"
    )
    lines.append(f"**Dataset:** `financial_phrasebank` / `{config}` — {n} sentences.\n")
    dist_str = ", ".join(f"{k}: {v} ({v / n:.1%})" for k, v in label_dist.items())
    lines.append(f"**Label distribution:** {dist_str}.\n")

    lines.append("## Metrics\n")
    lines.append(
        "| Model | Accuracy | Macro-F1 | Neg P/R | Neu P/R | Pos P/R |"
    )
    lines.append("|---|---|---|---|---|---|")
    lines.append(_metrics_table_row("FinBERT", finbert))
    lines.append(_metrics_table_row("VADER", vader))

    lines.append("\n## Confusion matrices\n")
    lines.append(
        "Rows are the true label, columns the predicted label "
        "(order: negative, neutral, positive).\n"
    )
    lines.append("### FinBERT\n")
    lines.append("![FinBERT confusion matrix](finbert_confusion_matrix.png)\n")
    lines.append("### VADER\n")
    lines.append("![VADER confusion matrix](vader_confusion_matrix.png)\n")

    # Most common confusions from the FinBERT matrix.
    cm = finbert["confusion_matrix"]
    confusions: list[tuple[str, str, int]] = []
    for i, true_lbl in enumerate(LABELS):
        for j, pred_lbl in enumerate(LABELS):
            if i != j and cm[i][j] > 0:
                confusions.append((true_lbl, pred_lbl, cm[i][j]))
    confusions.sort(key=lambda x: x[2], reverse=True)

    lines.append("## Error analysis\n")
    lines.append(
        f"FinBERT misclassified {round((1 - finbert['accuracy']) * n)} of {n} "
        f"sentences (accuracy {finbert['accuracy']:.1%}, macro-F1 "
        f"{finbert['macro_f1']:.1%}). "
        f"VADER reached {vader['accuracy']:.1%} accuracy / "
        f"{vader['macro_f1']:.1%} macro-F1.\n"
    )
    if confusions:
        top = confusions[0]
        lines.append(
            f"The single largest error class is **{top[0]} → {top[1]}** "
            f"({top[2]} cases), i.e. true `{top[0]}` sentences predicted as "
            f"`{top[1]}`.\n"
        )
    # Per-class recall, which is what matters for an imbalanced set.
    pc = finbert["per_class"]
    lines.append("Per-class recall:\n")
    for lbl in LABELS:
        lines.append(f"- `{lbl}`: {pc[lbl]['recall']:.1%} ({pc[lbl]['support']} examples)")

    lines.append("\n### Sample misclassifications\n")
    lines.append(
        "Up to 40 FinBERT errors, sorted by how confidently wrong the model was "
        "(highest margin between the predicted class and the runner-up first). "
        "Buckets are *approximate* keyword heuristics, not ground truth.\n"
    )
    lines.append("| True → Pred | Conf. | Bucket | Sentence |")
    lines.append("|---|---|---|---|")
    for e in error_sample:
        snippet = e["text"].replace("|", "/").replace("\n", " ")
        if len(snippet) > 140:
            snippet = snippet[:137] + "..."
        lines.append(
            f"| {e['true']} → {e['predicted']} | {e['confidence']:.2f} | "
            f"{e['bucket']} | {snippet} |"
        )

    lines.append(
        "\n### Full error dump\n"
        "Every misclassification (text, true label, predicted label, class "
        "probabilities) is in `finbert_errors.json` for deeper inspection.\n"
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_evaluation(config: str, limit: int | None, seed: int) -> dict:
    """Run the full evaluation and write all artifacts to ``evaluation/``."""
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading financial_phrasebank/{config} ...", flush=True)
    texts, y_true = load_phrasebank(config)
    if limit:
        rng = random.Random(seed)
        idx = rng.sample(range(len(texts)), k=min(limit, len(texts)))
        idx.sort()
        texts = [texts[i] for i in idx]
        y_true = [y_true[i] for i in idx]
    n = len(texts)
    label_dist = {lbl: y_true.count(lbl) for lbl in LABELS}
    print(f"  {n} sentences; distribution: {label_dist}", flush=True)

    print("Scoring with FinBERT (the app's model) ...", flush=True)
    finbert_pred, finbert_probs = score_finbert(texts)

    print("Scoring with VADER baseline ...", flush=True)
    vader_pred, _vader_compound = score_vader(texts)

    print("Computing metrics ...", flush=True)
    finbert_metrics = compute_metrics(y_true, finbert_pred)
    vader_metrics = compute_metrics(y_true, vader_pred)

    print("Building error analysis ...", flush=True)
    all_errors = build_error_analysis(texts, y_true, finbert_pred, finbert_probs)
    error_sample = all_errors[:_ERROR_SAMPLE_SIZE]  # short table for SUMMARY.md
    print(f"  {len(all_errors)} misclassifications total", flush=True)

    # --- Persist artifacts ---------------------------------------------------
    import csv

    results_path = _EVAL_DIR / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["text", "true", "finbert_pred", "vader_pred",
             "finbert_positive", "finbert_negative", "finbert_neutral"]
        )
        for t, yt, fp, vp, p in zip(texts, y_true, finbert_pred, vader_pred, finbert_probs):
            writer.writerow(
                [t, yt, fp, vp, p["positive"], p["negative"], p["neutral"]]
            )

    metrics_path = _EVAL_DIR / "metrics.json"
    metrics_payload = {
        "dataset": f"financial_phrasebank/{config}",
        "n_examples": n,
        "labels": list(LABELS),
        "label_distribution": label_dist,
        "seed": seed,
        "finbert": finbert_metrics,
        "vader": vader_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    errors_path = _EVAL_DIR / "finbert_errors.json"
    errors_path.write_text(json.dumps(all_errors, indent=2), encoding="utf-8")

    plot_confusion_matrix(
        finbert_metrics["confusion_matrix"],
        "FinBERT — Financial PhraseBank",
        _EVAL_DIR / "finbert_confusion_matrix.png",
    )
    plot_confusion_matrix(
        vader_metrics["confusion_matrix"],
        "VADER — Financial PhraseBank",
        _EVAL_DIR / "vader_confusion_matrix.png",
    )

    write_summary(
        _EVAL_DIR / "SUMMARY.md",
        config=config,
        n=n,
        label_dist=label_dist,
        finbert=finbert_metrics,
        vader=vader_metrics,
        error_sample=error_sample,
    )

    # --- Sanity check + console report --------------------------------------
    acc = finbert_metrics["accuracy"]
    print("\n=== Results ===", flush=True)
    print(f"FinBERT : accuracy={acc:.3f}  macro_f1={finbert_metrics['macro_f1']:.3f}", flush=True)
    print(f"VADER   : accuracy={vader_metrics['accuracy']:.3f}  macro_f1={vader_metrics['macro_f1']:.3f}", flush=True)
    print(f"Artifacts written to {_EVAL_DIR}", flush=True)
    if acc < 0.6:
        # The canonical symptom of a label-string mismatch between FinBERT and
        # PhraseBank: accuracy collapses to chance (~1/3). If that happens, do
        # not trust any downstream number until the mapping is fixed.
        print(
            "WARNING: FinBERT accuracy is near chance — this usually means a "
            "label-mapping bug. Check canonicalize_label() and the dataset's "
            "ClassLabel.names.",
            flush=True,
        )
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FinBERT on Financial PhraseBank (with a VADER baseline)."
    )
    parser.add_argument(
        "--config",
        default="sentences_75agree",
        help="financial_phrasebank config (default: sentences_75agree).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally subsample N sentences (deterministic, for a quick run).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for subsampling.")
    args = parser.parse_args()

    run_evaluation(config=args.config, limit=args.limit, seed=args.seed)


if __name__ == "__main__":
    main()
