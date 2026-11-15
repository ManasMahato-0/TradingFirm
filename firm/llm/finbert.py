"""FinBERT headline sentiment (local, CPU). Score in [-1, 1] = P(positive) - P(negative).

First use downloads the model (~440MB) from Hugging Face into the user's cache.
"""
from __future__ import annotations

MODEL = "ProsusAI/finbert"
_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        import os

        from transformers import pipeline

        try:  # use the cached copy without asking the Hub every run
            os.environ["HF_HUB_OFFLINE"] = "1"
            _PIPE = pipeline("text-classification", model=MODEL, top_k=None, device=-1)
        except Exception:
            os.environ["HF_HUB_OFFLINE"] = "0"  # first run: download
            _PIPE = pipeline("text-classification", model=MODEL, top_k=None, device=-1)
    return _PIPE


def sentiment(texts: list[str]) -> list[float]:
    if not texts:
        return []
    out = []
    for scores in _pipe()(texts, truncation=True, batch_size=16):
        p = {s["label"].lower(): s["score"] for s in scores}
        out.append(round(p.get("positive", 0.0) - p.get("negative", 0.0), 3))
    return out
