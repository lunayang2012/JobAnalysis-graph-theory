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
