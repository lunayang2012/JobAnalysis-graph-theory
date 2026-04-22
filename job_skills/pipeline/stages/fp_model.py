from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from ..config import Config

logger = logging.getLogger(__name__)


def _build_base_estimator(cfg: Config):
    """
    Build the base sklearn estimator used inside CalibratedClassifierCV.
    """
    if cfg.mode == "fast":
        max_features = cfg.max_features_fast
        min_df = cfg.min_df_fast
        cv = cfg.cv_fast
    else:
        max_features = cfg.max_features_precise
        min_df = cfg.min_df_precise
        cv = cfg.cv_precise

    # We return "cv" separately because it's a CalibratedClassifierCV arg
    if cfg.fp_model_type == "logreg":
        pipe = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=0.95,
                max_features=max_features,
                dtype=np.float32,
            ),
            LogisticRegression(
                solver="saga",
                class_weight="balanced",
                max_iter=cfg.max_iter_logreg,
                n_jobs=-1,
            ),
        )
    elif cfg.fp_model_type == "linear_svc":
        # LinearSVC supports decision_function for calibration
        pipe = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=0.95,
                max_features=max_features,
                dtype=np.float32,
            ),
            LinearSVC(),
        )
    else:
        raise ValueError(f"Unsupported fp_model_type: {cfg.fp_model_type}")

    return pipe, cv

def train_and_save_model(df: pd.DataFrame, y_weak: np.ndarray, cfg: Config) -> None:
    texts = (
        df.get("title", "").fillna("").astype(str)
        + " "
        + df.get("text_for_bertopic", "").fillna("").astype(str)
    ).tolist()

    base_estimator, cv = _build_base_estimator(cfg)

    logger.info("Training calibrated FP model...")
    with tqdm(total=1, desc="Training FP model", leave=False) as pbar:
        clf = CalibratedClassifierCV(base_estimator, cv=cv, method="sigmoid")
        clf.fit(texts, y_weak)
        pbar.update(1)

    save_model(clf, cfg)

def train_and_apply(
    df: pd.DataFrame, y_weak: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, CalibratedClassifierCV]:
    texts = (
        df.get("title", "").fillna("").astype(str)
        + " "
        + df.get("text_for_bertopic", "").fillna("").astype(str)
    ).tolist()

    base_estimator, cv = _build_base_estimator(cfg)

    logger.info("Training and applying calibrated FP model...")
    with tqdm(total=1, desc="Training FP model", leave=False) as pbar:
        clf = CalibratedClassifierCV(base_estimator, cv=cv, method="sigmoid")
        clf.fit(texts, y_weak)
        p_rnd = clf.predict_proba(texts)[:, 1]
        pbar.update(1)

    return p_rnd, clf

def save_model(model: CalibratedClassifierCV, cfg: Config) -> Path:
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.models_dir / "rnd_fp_filter.joblib"
    joblib.dump(model, path)
    logger.info(f"Saved FP model to {path}")
    return path
