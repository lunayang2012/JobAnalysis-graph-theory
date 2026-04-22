from __future__ import annotations

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Tuple
import gzip

from ..config import Config
from job_skills.pipeline.json_utils import json_default

logger = logging.getLogger(__name__)

from dataclasses import dataclass

@dataclass
class ExportResult:
    xlsx: Path
    csv: Path
    survey: Path
    jsonl: Path
    n_json: int
    dropped_csv: Path | None = None
    dropped_xlsx: Path | None = None



def _export_tabular(df: pd.DataFrame, cfg: Config) -> Tuple[Path, Path, Path]:
    """
    Export:
    - clean_jobs_all.xlsx
    - clean_jobs_all.csv
    - survey_luna.xlsx (postings + simple summary)
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = cfg.processed_dir / "clean_jobs_all.xlsx"
    csv_path = cfg.processed_dir / "clean_jobs_all.csv"
    survey_path = cfg.processed_dir / "survey_luna.xlsx"

    # Column selection
    keep_cols = [
        "uid",
        "title",
        "title_clean",
        "company",
        "location",
        "domain",
        "description",
        "text_for_bertopic",
        "min_amount",
        "max_amount",
        "currency",
        "site",
        "job_url",
        "date_posted",
        "p_rnd",
        # weak label metadata (if present)
        "label_weak",
        "pos_matches",
        "neg_matches",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_out = df[keep_cols].copy()

    # Main XLSX + CSV

    with pd.ExcelWriter(
    xlsx_path,
    engine="xlsxwriter",
    engine_kwargs={"options": {"strings_to_urls": False}},
) as writer:
        df_out.to_excel(writer, index=False, sheet_name="jobs")

    df_out.to_csv(csv_path, index=False, float_format="%.8f")

    logger.info(f"Wrote tabular outputs: {xlsx_path}, {csv_path}")

    # Survey workbook for Luna
    with pd.ExcelWriter(survey_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="postings", index=False)

        # Simple stats sheet
        summary = (
            df_out.groupby("company", dropna=False)
            .agg(
                n_postings=("uid", "count"),
                avg_min_amount=("min_amount", "mean"),
                avg_max_amount=("max_amount", "mean"),
            )
            .reset_index()
        )
        summary.to_excel(writer, sheet_name="company_summary", index=False)

    logger.info(f"Wrote survey workbook to {survey_path}")

    return xlsx_path, csv_path, survey_path


def _export_jsonl(df: pd.DataFrame, cfg: Config) -> Tuple[Path, int]:
    """
    Export a gzipped JSONL for large-scale use.

    Writes:
        processed/clean_jobs_all.jsonl.gz
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = cfg.processed_dir / "clean_jobs_all.jsonl.gz"

    cols = [
        "uid",
        "title",
        "title_clean",
        "company",
        "location",
        "domain",
        "text_for_bertopic",
        "description",
        "min_amount",
        "max_amount",
        "currency",
        "site",
        "job_url",
        "date_posted",
        "p_rnd",
    ]
    cols = [c for c in cols if c in df.columns]

    n_written = 0
    with gzip.open(jsonl_path, "wt", encoding="utf-8") as f:
        for row in df[cols].itertuples(index=False):
            rec = row._asdict()
            f.write(json.dumps(rec, ensure_ascii=False, default=json_default) + "\n")
            n_written += 1

    logger.info(f"Wrote gzipped JSONL to {jsonl_path} ({n_written} records)")
    return jsonl_path, n_written


def export_dropped(df_dropped: pd.DataFrame, cfg: Config) -> Tuple[Path, Path]:
    """
    Export the rows dropped by the FP filter for manual inspection.
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = cfg.processed_dir / "clean_jobs_dropped.csv"
    xlsx_path = cfg.processed_dir / "clean_jobs_dropped.xlsx"

    df_dropped.to_csv(csv_path, index=False, float_format="%.8f")

    with pd.ExcelWriter(
        xlsx_path,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}},
    ) as writer:
        df_dropped.to_excel(writer, index=False, sheet_name="dropped")

    logger.info(f"Wrote dropped jobs to: {csv_path}, {xlsx_path}")
    return csv_path, xlsx_path


def export_all(
    df_kept: pd.DataFrame,
    cfg: Config,
    df_dropped: pd.DataFrame | None = None,
) -> ExportResult:
    xlsx_path, csv_path, survey_path = _export_tabular(df_kept, cfg)
    jsonl_path, n_json = _export_jsonl(df_kept, cfg)

    dropped_csv_path: Path | None = None
    dropped_xlsx_path: Path | None = None

    if df_dropped is not None:
        dropped_csv_path, dropped_xlsx_path = export_dropped(df_dropped, cfg)

    return ExportResult(
        xlsx=xlsx_path,
        csv=csv_path,
        survey=survey_path,
        jsonl=jsonl_path,
        n_json=n_json,
        dropped_csv=dropped_csv_path,
        dropped_xlsx=dropped_xlsx_path,
    )


