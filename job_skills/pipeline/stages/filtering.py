from __future__ import annotations

import re
import logging

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from ..config import Config

from cleanlab.filter import find_label_issues

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

#=================================================================================
# Cleanlab utilities for identifying likely false negatives among weak negatives. 
#=================================================================================
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues

from ..config import Config

logger = logging.getLogger(__name__)


def _load_processed_pair(cfg: Config) -> pd.DataFrame:
    """Load kept + dropped job sets and concatenate them."""
    processed_dir = cfg.processed_dir

    kept_path = processed_dir / "clean_jobs_all.xlsx"
    dropped_path = processed_dir / "clean_jobs_dropped.xlsx"

    if not kept_path.exists():
        raise FileNotFoundError(f"Kept file not found: {kept_path}")
    if not dropped_path.exists():
        raise FileNotFoundError(f"Dropped file not found: {dropped_path}")

    logger.info("Loading kept from %s", kept_path)
    df_kept = pd.read_excel(kept_path)

    logger.info("Loading dropped from %s", dropped_path)
    df_dropped = pd.read_excel(dropped_path)

    df_kept["source_bucket"] = "kept"
    df_dropped["source_bucket"] = "dropped"

    df = pd.concat([df_kept, df_dropped], ignore_index=True, sort=False)
    return df


def select_uncertain_cleanlab(
    cfg: Config,
    n: int = 5000,
    min_p: float = 0.20,
    max_p: float = 0.90,
) -> Tuple[Path, Path]:
    """
    Use Cleanlab to surface the top-N most suspicious rows,
    focusing on label_weak == 0 with mid-range probabilities p_rnd.

    This is designed to give you a high-value pool for manual labeling.
    """
    df = _load_processed_pair(cfg)

    required_cols = ["label_weak", "p_rnd"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Cleanlab: {missing}")

    # Labels: noisy weak labels (0 = non-core, 1 = core)
    y = df["label_weak"].astype(int).to_numpy()

    # Model probabilities from your FP model
    p = df["p_rnd"].astype(float).to_numpy()
    pred_probs = np.column_stack([1.0 - p, p])

    # Run Cleanlab self-confidence ranking
    logger.info("Running Cleanlab label issue detection on %d rows ...", len(df))
    issue_idx = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    # Focus on "suspect negatives": weak_label == 0, mid-range p_rnd
    mask_candidate = (df["label_weak"] == 0) & df["p_rnd"].between(min_p, max_p)
    candidate_idx = [i for i in issue_idx if mask_candidate.iloc[i]]

    top_idx = candidate_idx[:n]
    df_out = df.iloc[top_idx].copy()

    processed_dir = cfg.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = processed_dir / f"clean_jobs_uncertain_{n}.csv"
    jsonl_path = processed_dir / f"clean_jobs_uncertain_{n}.jsonl"

    logger.info(
        "Writing %d uncertain rows to %s and %s",
        len(df_out),
        csv_path,
        jsonl_path,
    )

    df_out.to_csv(csv_path, index=False)
    df_out.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    return csv_path, jsonl_path

def compute_cleanlab_fn_mask(
    df: pd.DataFrame,
    min_p: float = 0.40,
    max_fraction: float = 0.20,
) -> np.ndarray:
    """
    Use Cleanlab to identify likely false negatives among weak negatives.

    Returns a boolean mask (len(df)) where True means:
        "this row is a likely FN candidate (should be treated as core or reviewed)"

    Logic:
      - labels = df['label_weak'] (0/1)  -> noisy labels
      - p_rnd  = prob core from FP model
      - cleanlab.filter.find_label_issues ranks label errors
      - we keep a subset of rows where:
          * label_weak == 0
          * p_rnd >= min_p
          * row is in top 'max_fraction' of ranked label issues
    """

    if "label_weak" not in df.columns or "p_rnd" not in df.columns:
        raise ValueError("compute_cleanlab_fn_mask requires 'label_weak' and 'p_rnd' columns in df.")

    y = df["label_weak"].astype(int).to_numpy()
    p = df["p_rnd"].astype(float).to_numpy()
    pred_probs = np.column_stack([1.0 - p, p])

    # Rank all rows by Cleanlab's self-confidence criterion
    issue_idx = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    n = len(df)
    # At most max_fraction of the dataset will be considered "issues"
    k = int(max_fraction * n)
    if k <= 0:
        return np.zeros(n, dtype=bool)

    top_issue_idx = issue_idx[:k]
    issue_mask = np.zeros(n, dtype=bool)
    issue_mask[top_issue_idx] = True

    # Now restrict to weak negatives with reasonably high model probability
    weak_negative = (y == 0)
    high_prob = (p >= min_p)

    fn_mask = weak_negative & high_prob & issue_mask
    return fn_mask
