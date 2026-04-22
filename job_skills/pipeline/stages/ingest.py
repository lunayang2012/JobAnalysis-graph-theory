from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import Config
from ..schemas import validate_main

logger = logging.getLogger(__name__)


def _read_excel(path: Path) -> pd.DataFrame:
    logger.info("Reading Excel: %s", path)
    return pd.read_excel(path)


def _read_csv(path: Path) -> pd.DataFrame:
    logger.info("Reading CSV: %s", path)
    return pd.read_csv(path)


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return _read_excel(path)
    if suf == ".csv":
        return _read_csv(path)
    raise ValueError(f"Unsupported input type: {path}")


def load_data(cfg: Config, input_path: Path | None = None) -> pd.DataFrame:
    """Load job postings data.

    Resolution order:
      1) explicit input_path (file or directory)
      2) cfg.combined_path
      3) cfg.hourly_path + cfg.yearly_path

    Notes:
      - We derive `domain` and `source_domain` from `Source.Name` *before* schema validation
        because downstream stages assume these exist.
    """

    dfs: list[pd.DataFrame] = []

    if input_path is not None:
        if input_path.is_dir():
            # Accept both xlsx and csv in a folder.
            for p in sorted(list(input_path.glob("*.xlsx")) + list(input_path.glob("*.csv"))):
                dfs.append(_read_any(p))
        else:
            dfs.append(_read_any(input_path))
    elif cfg.combined_path and cfg.combined_path.exists():
        dfs.append(_read_any(cfg.combined_path))
    else:
        if cfg.hourly_path and cfg.hourly_path.exists():
            dfs.append(_read_any(cfg.hourly_path))
        if cfg.yearly_path and cfg.yearly_path.exists():
            dfs.append(_read_any(cfg.yearly_path))

    if not dfs:
        raise FileNotFoundError(
            "No input data found. Provide --input or set env vars "
            "JOB_SKILLS_COMBINED / JOB_SKILLS_HOURLY / JOB_SKILLS_YEARLY."
        )

    df = pd.concat(dfs, ignore_index=True)

    # --- derive domain/source_domain from Source.Name before schema validation ---
    if "Source.Name" in df.columns and "source_domain" not in df.columns:
        domain_series = (
            df["Source.Name"]
            .astype(str)
            .str.split("_", n=1)
            .str[0]
            .str.lower()
        )
        # Column required by the schema
        df["domain"] = domain_series
        # Column requested for downstream use
        df["source_domain"] = domain_series

    # Validate / coerce into the relaxed pipeline schema.
    df = validate_main(df)
    return df
