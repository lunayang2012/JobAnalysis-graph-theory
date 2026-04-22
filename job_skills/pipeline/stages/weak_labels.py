from __future__ import annotations

import re
import logging

import numpy as np
import pandas as pd

from ..config import Config

logger = logging.getLogger(__name__)

from typing import Iterable, Tuple

def _compile_union(patterns: Iterable[str]) -> re.Pattern | None:
    """Compile a single regex that ORs together all patterns, or return None if empty."""
    pats = [p for p in patterns if p]
    if not pats:
        return None
    # Wrap each pattern in a non-capturing group to be safe
    union = "|".join(f"(?:{p})" for p in pats)
    return re.compile(union, flags=re.IGNORECASE)

def _mask(text: pd.Series, patterns: Iterable[str]) -> pd.Series:
    """Return a boolean mask: True if any pattern matches the text."""
    text = text.fillna("").astype(str)
    rex = _compile_union(patterns)
    if rex is None:
        return pd.Series(False, index=text.index)
    return text.str.contains(rex, na=False)

def _split_title_patterns(pos_title_patterns: list[str]) -> Tuple[list[str], list[str]]:
    """
    Heuristic split of title patterns into:
      • strong: clearly R&D / lab / scientist / chemist / engineer etc.
      • contextual: generic roles that need lab context (technician, assistant, specialist, etc.)
    """
    context_keywords = [
        "technician",
        "technologist",
        "tech\\b",          # lab tech, etc.
        "assistant",
        "specialist",
        "supervisor",
        "manager",
        "coordinator",
        "analyst",
        "support",
        "associate",        # research associate, lab associate, etc.
    ]

    strong: list[str] = []
    contextual: list[str] = []

    for pat in pos_title_patterns:
        low = pat.lower()
        if any(kw in low for kw in context_keywords):
            contextual.append(pat)
        else:
            strong.append(pat)

    return strong, contextual

def make_weak_labels(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """
    Field-aware weak labels with reduced false negatives:

      • Strong title patterns alone are sufficient for a positive.
      • Contextual title patterns (technician, assistant, specialist, etc.)
        require a lab-ish description match.
      • Negative title patterns still veto positives.
    """
    # ---- TEXT FIELDS ----
    title_text = df.get("title_clean", df.get("title", "")).fillna("").astype(str)
    desc_text = df.get("text_for_bertopic", df.get("description", "")).fillna("").astype(str)

    # ---- POSITIVE PATTERNS ----
    strong_title_patterns, contextual_title_patterns = _split_title_patterns(
        cfg.pos_title_patterns
    )

    pos_title_strong = _mask(title_text, strong_title_patterns)
    pos_title_context = (
        _mask(title_text, contextual_title_patterns)
        if contextual_title_patterns
        else pd.Series(False, index=df.index)
    )
    pos_desc = _mask(desc_text, cfg.pos_desc_patterns)

    # Any positive match (for debugging counters)
    pos_any_title = _mask(title_text, cfg.pos_title_patterns)
    pos_any_desc = pos_desc
    pos_any = pos_any_title | pos_any_desc

    # ---- NEGATIVE PATTERNS ----
    neg_title = _mask(title_text, cfg.neg_title_patterns)
    # Keep neg_desc for future use / debugging but do NOT combine into neg logic
    if hasattr(cfg, "neg_desc_patterns"):
        neg_desc = _mask(desc_text, cfg.neg_desc_patterns)
    else:
        neg_desc = pd.Series(False, index=df.index)

    # ---- COMBINATION LOGIC ----
    # Strong title alone is enough.
    # Contextual title requires lab-ish description.
    #pos = pos_title_strong | (pos_title_context & pos_desc)
    pos = pos_any_title | (pos_any_desc & pos_desc)
    neg = neg_title  # neg_desc intentionally *not* used in label logic

    y = np.zeros(len(df), dtype=int)
    y[pos & ~neg] = 1

    # Persist debug columns for export / inspection
    df["label_weak"] = y
    df["pos_matches"] = pos_any.astype(int)
    df["neg_matches"] = neg.astype(int)

    logger.info(
        "Weak labels: total=%d pos_title_any=%d pos_desc=%d neg_title=%d final_pos=%d",
        len(df),
        int(pos_any_title.sum()),
        int(pos_desc.sum()),
        int(neg_title.sum()),
        int((pos & ~neg).sum()),
    )

    return y

# stages/weak_labels.py
def make_pu_labels(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """
    Build labels for PU training:
      - 1 for high-precision positives (strong title patterns)
      - 0 for unlabeled (everything else)
    """
    title = df[cfg.title_col].fillna("")
    desc  = df[cfg.text_col].fillna("")

    # Reuse or define a subset of your existing patterns as "strong"
    strong_title_mask = _mask(title, cfg.pos_title_patterns_strong_pu)

    # Optionally veto obvious negatives if you want a tiny clean negative set for diagnostics,
    # but don't feed those as full negatives to the PU learner unless you're doing PNU.
    if cfg.neg_title_patterns_forced:
        neg_title_mask = _mask(title, cfg.neg_title_patterns_forced)
    else:
        neg_title_mask = pd.Series(False, index=df.index)

    y_pu = np.zeros(len(df), dtype=int)
    y_pu[strong_title_mask & ~neg_title_mask] = 1

    df["label_pu"] = y_pu  # keep for later inspection
    return y_pu

