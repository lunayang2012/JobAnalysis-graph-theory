
from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from rich.progress import Progress

from .config import Config
from .stages import ingest, preprocess, fp_model, export, weak_labels
from .stages.data_quality_report import DatasetSpec, run_data_quality_report

logger = logging.getLogger(__name__)


def _timed(label: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            logger.info("%s took %.2fs", label, dt)
            return out
        return wrapper
    return decorator


@_timed("Preprocess-only")
def preprocess_only(cfg: Config, input_path: Path | None = None) -> pd.DataFrame:
    df = ingest.load_data(cfg, input_path)
    df = preprocess.run_preprocess(df, cfg)
    return df


@_timed("Train FP model only")
def train_fp_model_only(cfg: Config, input_path: Path | None = None) -> None:
    df = preprocess_only(cfg, input_path)
    y = weak_labels.make_weak_labels(df, cfg)
    fp_model.train_and_save_model(df, y, cfg)


@_timed("Data quality report")
def dq_report_only(
    cfg: Config,
    before_path: Path,
    after_path: Path,
    *,
    before_name: str = "before",
    after_name: str = "after",
) -> Dict[str, Path]:
    outputs = run_data_quality_report(
        cfg=cfg,
        before=DatasetSpec(name=before_name, path=before_path, kind="before"),
        after=DatasetSpec(name=after_name, path=after_path, kind="after"),
    )
    return {
        "out_dir": outputs.out_dir,
        "metrics_csv": outputs.metrics_csv,
        "metrics_xlsx": outputs.metrics_xlsx,
        "deltas_csv": outputs.deltas_csv,
        "audit_sample_csv": outputs.audit_sample_csv,
        "error_ledger_csv": outputs.error_ledger_csv,
        "run_metadata_json": outputs.run_metadata_json,
    }


@_timed("Full pipeline")
def run_full_pipeline(cfg: Config, input_path: Path | None = None) -> Dict[str, Path]:
    with Progress() as progress:
        task = progress.add_task("Full pipeline", total=4)

        df = preprocess_only(cfg, input_path)
        progress.advance(task)

        y = weak_labels.make_weak_labels(df, cfg)
        df["label_weak"] = y
        progress.advance(task)

        p_rnd, model = fp_model.train_and_apply(df, y, cfg)
        df["p_rnd"] = p_rnd
        progress.advance(task)

        tokens = df["text_for_bertopic"].astype(str).str.split().map(len)
        neg_matches = df.get("neg_matches", 0).astype(int) > 0
        low_conf_negative = (
            (p_rnd < cfg.fp_tau)
            & neg_matches
            & (tokens >= cfg.min_tokens)
        )

        df_final = df.loc[~low_conf_negative].copy()
        df_dropped = df.loc[low_conf_negative].copy()

        result = export.export_all(df_final, cfg, df_dropped=df_dropped)
        progress.advance(task)

    return {
        "xlsx": result.xlsx,
        "csv": result.csv,
        "jsonl": result.jsonl,
        "survey": result.survey,
    }