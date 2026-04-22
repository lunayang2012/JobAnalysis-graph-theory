# job_skills/stages/data_quality_report_v3.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import json
import hashlib
import pandas as pd
import logging

# NOTE: this module is intended to live under job_skills/stages/
# so we use relative imports to match the rest of the pipeline.
from ..config import Config
from ..schemas import validate_main

logger = logging.getLogger(__name__)


# ============================================================
# Specs
# ============================================================

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    kind: str  # "before" | "after"


@dataclass(frozen=True)
class ColumnSpec:
    # Id columns (we'll auto-pick + auto-align)
    uid: str = "uid"
    posting_id: str = "Posting ID"
    id_fallback: str = "id"
    job_url: str = "job_url"

    # Core fields (Satish)
    title: str = "title"
    title_clean: str = "title_clean"
    company: str = "company"
    location: str = "location"
    date_posted: str = "date_posted"
    domain: str = "domain"

    # Text (for BERTopic + text quality)
    # We'll auto-detect if this exact name is not present.
    preferred_text: str = "description_cleaned_sPacy_en_lg_100words"


@dataclass(frozen=True)
class ReportOutputs:
    out_dir: Path
    metrics_csv: Path
    metrics_xlsx: Path
    deltas_csv: Path
    audit_sample_csv: Path
    error_ledger_csv: Path
    run_metadata_json: Path


# ============================================================
# IO + normalization
# ============================================================


# =========================
# Overleaf-ready table emitters (BBFB)
# =========================

def write_table_dq_summary_tex(
    out_path: Path,
    metrics_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    *,
    before_label: str = "Before",
    after_label: str = "After",
) -> None:
    """Write a publication-ready LaTeX table summarizing DQ metrics.

    Assumes metrics_df has columns: metric, value_before, value_after, delta
    If your metrics_df is in long form, we will best-effort reshape it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = metrics_df.copy()

    # Best-effort normalization: accept either wide or long input.
    cols = {c.lower(): c for c in df.columns}
    if {"metric", "value_before", "value_after", "delta"}.issubset(set(cols)):
        df = df[[cols["metric"], cols["value_before"], cols["value_after"], cols["delta"]]].copy()
        df.columns = ["Metric", before_label, after_label, "Δ"]
    elif {"metric", "dataset", "value"}.issubset(set(cols)):
        # long: metric/dataset/value -> pivot
        mcol, dcol, vcol = cols["metric"], cols["dataset"], cols["value"]
        piv = df.pivot_table(index=mcol, columns=dcol, values=vcol, aggfunc="first")
        # choose first two columns as before/after if labels not found
        b = before_label if before_label in piv.columns else (piv.columns[0] if len(piv.columns) else before_label)
        a = after_label if after_label in piv.columns else (piv.columns[1] if len(piv.columns) > 1 else after_label)
        out = pd.DataFrame({
            "Metric": piv.index.astype(str),
            before_label: piv[b].values if b in piv.columns else None,
            after_label: piv[a].values if a in piv.columns else None,
        })
        out["Δ"] = out[after_label] - out[before_label]
        df = out
    else:
        # Fallback: do something sane
        df = df.rename(columns={df.columns[0]: "Metric"})
        if before_label not in df.columns and len(df.columns) > 1:
            df = df.rename(columns={df.columns[1]: before_label})
        if after_label not in df.columns and len(df.columns) > 2:
            df = df.rename(columns={df.columns[2]: after_label})
        if "Δ" not in df.columns and {before_label, after_label}.issubset(df.columns):
            b = pd.to_numeric(df[before_label], errors="coerce")
            a = pd.to_numeric(df[after_label], errors="coerce")
            df["Δ"] = a - b
        df = df[["Metric", before_label, after_label, "Δ"]].copy()

    # Row count is BBFB for reviewers; ensure it's present (from deltas_df if needed)
    if "row_count" not in " ".join(df["Metric"].str.lower().tolist()):
        if {"metric", "before", "after", "delta"}.issubset({c.lower() for c in deltas_df.columns}):
            dcols = {c.lower(): c for c in deltas_df.columns}
            row = deltas_df.loc[deltas_df[dcols["metric"]].astype(str).str.contains("row", case=False, na=False)].head(1)
            if len(row):
                df = pd.concat([
                    pd.DataFrame([{
                        "Metric": "Row count",
                        before_label: float(row.iloc[0][dcols["before"]]),
                        after_label: float(row.iloc[0][dcols["after"]]),
                        "Δ": float(row.iloc[0][dcols["delta"]]),
                    }]),
                    df
                ], ignore_index=True)

    # Formatting helpers
    def fmt(x):
        if pd.isna(x):
            return ""
        try:
            fx = float(x)
        except Exception:
            return str(x)
        # Heuristic: percentages typically in [0,1] or [0,100]
        if abs(fx) <= 1.0 and fx != 0:
            return f"{fx:.3f}"
        if abs(fx) <= 100 and ("percent" in str(x).lower() or "rate" in str(x).lower()):
            return f"{fx:.2f}"
        if abs(fx) >= 1000 and fx.is_integer():
            return f"{int(fx)}"
        return f"{fx:.3f}"

    df_out = df.copy()
    for c in [before_label, after_label, "Δ"]:
        if c in df_out.columns:
            df_out[c] = df_out[c].map(fmt)

    # Clean metric names a bit for paper
    df_out["Metric"] = df_out["Metric"].astype(str).str.replace("_", " ").str.strip()

    latex = df_out.to_latex(
        index=False,
        escape=True,
        column_format="lrrr",
        longtable=False,
        caption="Data quality summary (before vs after cleaning).",
        label="tab:dq_summary",
    )
    # Make it look like a real paper table
    latex = latex.replace(
        r"\toprule", 
        r"\toprule\n\multicolumn{4}{l}{\textit{Higher is better for rates; Δ = After − Before.}}\\"
        )
    out_path.write_text(latex, encoding="utf-8")

def _read_dataset(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce into relaxed pipeline schema when available.

    IMPORTANT: DQ reporting should be robust; schema validation must never hard-fail.
    """
    df2 = df.copy()
    try:
        df2 = validate_main(df2)
    except Exception:
        pass
    return df2


def _pick_id_column(df: pd.DataFrame, colspec: ColumnSpec) -> str:
    """Pick a stable id column if present."""
    for c in (colspec.uid, colspec.posting_id, colspec.id_fallback, colspec.job_url, "id"):
        if c in df.columns:
            return c
    return "__index__"


def _ensure_index_id(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col == "__index__":
        out = df.copy()
        out["__index__"] = out.index.astype(str)
        return out
    return df


def _detect_text_col(df: pd.DataFrame, colspec: ColumnSpec) -> Optional[str]:
    candidates = [
        colspec.preferred_text,
        "text_for_bertopic",
        "description_cleaned",
        "description",
        "text",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# Robust date parsing (fix #1)
# ============================================================

def _parse_date_series_robust(s: pd.Series) -> pd.Series:
    """Parse date_posted robustly across:
    - real datetimes
    - Excel serial day numbers (numeric)
    - numeric-like strings containing Excel serials
    - ISO / common datetime strings

    Returns datetime64[ns] with NaT for failures.
    """
    if s is None:
        return pd.to_datetime(pd.Series([], dtype="object"), errors="coerce")

    # If already datetime-ish
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    # Convert to string for mixed types, but keep a numeric candidate
    s_obj = s.copy()
    s_num = pd.to_numeric(s_obj, errors="coerce")

    # Heuristic: Excel serial days are usually in [20000, 80000]
    is_excel_serial = s_num.between(20000, 80000, inclusive="both")

    out = pd.Series(pd.NaT, index=s_obj.index, dtype="datetime64[ns]")

    # Excel serials
    if is_excel_serial.any():
        out.loc[is_excel_serial] = pd.to_datetime(
            s_num.loc[is_excel_serial], unit="D", origin="1899-12-30", errors="coerce"
        )

    # Everything else: normal parse (strings like '2024-01-02', etc.)
    remainder = ~is_excel_serial
    if remainder.any():
        out.loc[remainder] = pd.to_datetime(s_obj.loc[remainder], errors="coerce")

    return out


def _timeliness_summary(df: pd.DataFrame, date_col: str) -> Dict[str, Any]:
    if date_col not in df.columns:
        return {
            "timeliness_parse_rate": 0.0,
            "min_date": None,
            "max_date": None,
            "timeliness_flag": "missing_date_col",
        }

    dt = _parse_date_series_robust(df[date_col])
    parse_rate = float(dt.notna().mean()) * 100.0

    if dt.dropna().empty:
        return {
            "timeliness_parse_rate": parse_rate,
            "min_date": None,
            "max_date": None,
            "timeliness_flag": "all_unparseable",
        }

    mn = dt.min()
    mx = dt.max()

    # Suspicious date diagnostics (e.g., everything becomes 1970)
    flag = "ok"
    if mn.date() == datetime(1970, 1, 1).date() and mx.date() == datetime(1970, 1, 1).date() and parse_rate > 90.0:
        flag = "suspicious_all_1970"

    return {
        "timeliness_parse_rate": parse_rate,
        "min_date": str(mn.date()),
        "max_date": str(mx.date()),
        "timeliness_flag": flag,
    }


# ============================================================
# Data quality metrics (fixes #4, #5, #6)
# ============================================================

def completeness_by_col(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Any]:
    total = len(df)
    out: Dict[str, Any] = {}
    for c in cols:
        if c not in df.columns or total == 0:
            out[f"completeness_{c}"] = 0.0
        else:
            out[f"completeness_{c}"] = 100.0 * float(df[c].notna().mean())
    return out


def completeness_rate_avg(df: pd.DataFrame, cols: Sequence[str]) -> float:
    if len(df) == 0:
        return 0.0
    per = []
    for c in cols:
        if c not in df.columns:
            per.append(0.0)
        else:
            per.append(float(df[c].notna().mean()))
    return 100.0 * float(sum(per) / max(len(per), 1))


def uniqueness_rate(df: pd.DataFrame, key_col: str) -> float:
    total = len(df)
    if total == 0 or key_col not in df.columns:
        return 0.0
    return 100.0 * float(df[key_col].nunique(dropna=False) / total)


def _allowed_domains(cfg: Config, df: pd.DataFrame, domain_col: str) -> Optional[set[str]]:
    # Prefer configured allowlist if present.
    allowlist = getattr(cfg, "domain_allowlist", None)
    if isinstance(allowlist, list) and allowlist:
        return set(str(x).strip().lower() for x in allowlist)

    # Else infer from observed values (top K, non-null)
    if domain_col in df.columns:
        vals = (
            df[domain_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )
        if not vals.empty:
            top = vals.value_counts().head(25).index.tolist()
            return set(top)
    return None


def _location_valid(s: pd.Series) -> pd.Series:
    # Allow common remote markers; otherwise require at least 2 tokens or a comma separated format.
    txt = s.fillna("").astype(str).str.strip()
    low = txt.str.lower()

    remote = low.isin({"remote", "remote (us)", "remote - us", "hybrid", "hybrid (us)"}) | low.str.contains("remote", na=False)
    has_comma = txt.str.contains(",", regex=False, na=False)
    # simple "City ST" patterns
    has_two_tokens = txt.str.split().map(len) >= 2
    return remote | has_comma | has_two_tokens


def validity_rate(df: pd.DataFrame, checks: Dict[str, Callable[[pd.Series], pd.Series]]) -> float:
    total = len(df)
    if total == 0:
        return 0.0

    per_check = []
    for col, fn in checks.items():
        if col not in df.columns:
            per_check.append(0.0)
            continue
        try:
            ok = fn(df[col])
            per_check.append(float(ok.fillna(False).mean()))
        except Exception:
            per_check.append(0.0)

    return 100.0 * float(sum(per_check) / max(len(per_check), 1))


def consistency_rate(df: pd.DataFrame, rules: Dict[str, Callable[[pd.DataFrame], pd.Series]]) -> float:
    if len(df) == 0:
        return 0.0

    scores = []
    for _, rule_fn in rules.items():
        try:
            ok = rule_fn(df)
            scores.append(float(ok.fillna(False).mean()))
        except Exception:
            scores.append(0.0)
    return 100.0 * float(sum(scores) / max(len(scores), 1))


def text_informativeness(df: pd.DataFrame, text_col: Optional[str]) -> Dict[str, Any]:
    if not text_col or text_col not in df.columns:
        return {"text_col_used": text_col, "median_tokens": None, "pct_short_docs_lt_30": None}
    tokens = df[text_col].fillna("").astype(str).str.split().map(len)
    return {
        "text_col_used": text_col,
        "median_tokens": int(tokens.median()) if len(tokens) else None,
        "pct_short_docs_lt_30": 100.0 * float((tokens < 30).mean()) if len(tokens) else None,
    }


def entity_contamination_proxy(df: pd.DataFrame, text_col: Optional[str], company_col: str, location_col: str) -> float:
    if not text_col or any(c not in df.columns for c in (text_col, company_col, location_col)):
        return 0.0
    text = df[text_col].fillna("").astype(str).str.lower()
    comp = df[company_col].fillna("").astype(str).str.lower()
    loc = df[location_col].fillna("").astype(str).str.lower()

    hit = (comp.str.len() > 2) & text.str.contains(comp, regex=False)
    hit |= (loc.str.len() > 2) & text.str.contains(loc, regex=False)
    return 100.0 * float(hit.mean())


# ============================================================
# Record alignment + error ledger (fix #3)
# ============================================================

def _norm_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().lower()
    s = " ".join(s.split())
    return s


def _canon_key_row(row: pd.Series, fields: Sequence[str]) -> str:
    parts = [_norm_str(row.get(f, "")) for f in fields]
    joined = "|".join(parts)
    # Stable short hash
    h = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    return h


def _build_alignment_key(
    df_b: pd.DataFrame,
    df_a: pd.DataFrame,
    colspec: ColumnSpec,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Pick a key that exists in both; else build a canonical key."""
    meta: Dict[str, Any] = {}

    candidate_shared = []
    for c in (colspec.uid, colspec.posting_id, colspec.id_fallback, colspec.job_url, "job_url", "Posting ID", "id"):
        if c in df_b.columns and c in df_a.columns:
            candidate_shared.append(c)

    if candidate_shared:
        key = candidate_shared[0]
        meta["alignment_key"] = key
        meta["alignment_method"] = "shared_column"
        return key, df_b, df_a, meta

    # Fallback: build canonical key from overlapping semantic fields
    fields = [c for c in (colspec.title, colspec.company, colspec.location, colspec.job_url) if c in df_b.columns and c in df_a.columns]
    if not fields:
        # last resort: title+company even if one missing
        fields = [c for c in (colspec.title, colspec.company) if c in df_b.columns or c in df_a.columns]
        fields = [f for f in fields if f]  # type: ignore

    def add_key(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # ensure fields exist
        for f in fields:
            if f not in out.columns:
                out[f] = ""
        out["canon_key"] = out.apply(lambda r: _canon_key_row(r, fields), axis=1)
        return out

    df_b2 = add_key(df_b)
    df_a2 = add_key(df_a)

    meta["alignment_key"] = "canon_key"
    meta["alignment_method"] = f"hash({fields})"
    meta["canon_fields"] = list(fields)
    return "canon_key", df_b2, df_a2, meta


def _build_error_ledger(
    before: pd.DataFrame,
    after: pd.DataFrame,
    core_cols: Sequence[str],
    validity_checks: Dict[str, Callable[[pd.Series], pd.Series]],
    *,
    align_key: str,
) -> pd.DataFrame:
    """Issue-level ledger aligned by an explicit key (NOT row index)."""
    if align_key not in before.columns or align_key not in after.columns:
        return pd.DataFrame()

    b = before.set_index(align_key, drop=False)
    a = after.set_index(align_key, drop=False)

    common = b.index.intersection(a.index)
    b = b.loc[common]
    a = a.loc[common]

    rows: list[dict[str, Any]] = []
    def add_rows(mask_b: pd.Series, mask_a: pd.Series, col: str, issue_type: str):
        # Guard against duplicate index labels (common when align_key is not unique).
        # Collapse to a unique key-level boolean by OR-ing duplicates.
        if mask_b.index.has_duplicates:
            mask_b = mask_b.groupby(level=0).any()
        if mask_a.index.has_duplicates:
            mask_a = mask_a.groupby(level=0).any()

        idx = mask_b.index[mask_b | mask_a]
        if len(idx) == 0:
            return

        # Values: pick the first occurrence per key for display purposes.
        if b.index.has_duplicates:
            b_vals = b.groupby(level=0)[col].first()
        else:
            b_vals = b[col]
        if a.index.has_duplicates:
            a_vals = a.groupby(level=0)[col].first()
        else:
            a_vals = a[col]

        fixed = (mask_b.loc[idx] & (~mask_a.loc[idx]))
        for rid in idx:
            rows.append({
                "record_key": rid,
                "column": col,
                "issue_type": issue_type,
                "before_has_issue": bool(mask_b.loc[rid]),
                "after_has_issue": bool(mask_a.loc[rid]),
                "fixed": bool(fixed.loc[rid]),
                "before_value": None if pd.isna(b_vals.get(rid, None)) else b_vals.get(rid, None),
                "after_value": None if pd.isna(a_vals.get(rid, None)) else a_vals.get(rid, None),
            })

    for col in core_cols:
        if col not in b.columns or col not in a.columns:
            continue

        # Missing
        add_rows(b[col].isna(), a[col].isna(), col, "missing")

        # Invalid
        if col in validity_checks:
            try:
                valid_b = validity_checks[col](b[col])
                valid_a = validity_checks[col](a[col])
                add_rows((~valid_b.fillna(False)), (~valid_a.fillna(False)), col, "invalid")
            except Exception:
                pass

    ledger = pd.DataFrame(rows)
    return ledger


# ============================================================
# Process metrics (fixes #2, #7, #8)
# ============================================================

def issue_volume_fixed(before: pd.DataFrame, after: pd.DataFrame, cols: Sequence[str], *, key_col: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for c in cols:
        b = int(before[c].isna().sum()) if c in before.columns else None
        a = int(after[c].isna().sum()) if c in after.columns else None
        if b is not None and a is not None:
            out[f"nulls_fixed_{c}"] = max(b - a, 0)

    # Duplicate reductions using chosen key_col (must exist in BOTH to be meaningful)
    if key_col in before.columns:
        out["duplicates_before"] = int(len(before) - before[key_col].nunique(dropna=False))
    if key_col in after.columns:
        out["duplicates_after"] = int(len(after) - after[key_col].nunique(dropna=False))

    return out


def _cost_per_cleaned_record(
    runtime_seconds: float,
    records_cleaned: int,
    assumed_hourly_rate_usd: float,
) -> Optional[float]:
    if records_cleaned <= 0:
        return None
    runtime_hours = runtime_seconds / 3600.0
    return (runtime_hours * assumed_hourly_rate_usd) / records_cleaned


def _recurrence_rate_from_ledgers(
    prior_ledger: pd.DataFrame,
    current_ledger: pd.DataFrame,
) -> Optional[float]:
    """Recurrence: issues that were fixed previously but reappear now.

    Needs at least two runs. We compute:
      reappeared / previously_fixed
    """
    if prior_ledger.empty or current_ledger.empty:
        return None
    # Previously fixed issues (by record_key, column, issue_type)
    prior_fixed = prior_ledger.loc[prior_ledger.get("fixed", False) == True]
    if prior_fixed.empty:
        return None

    key_cols = ["record_key", "column", "issue_type"]
    prior_set = set(tuple(x) for x in prior_fixed[key_cols].itertuples(index=False, name=None))

    # Current issues present after cleaning (after_has_issue == True)
    cur_after_issue = current_ledger.loc[current_ledger.get("after_has_issue", False) == True]
    cur_set = set(tuple(x) for x in cur_after_issue[key_cols].itertuples(index=False, name=None))

    reappeared = len(prior_set.intersection(cur_set))
    return 100.0 * (reappeared / max(len(prior_set), 1))


# ============================================================
# Main API
# ============================================================

def run_data_quality_report(
    cfg: Config,
    before: DatasetSpec,
    after: DatasetSpec,
    *,
    colspec: ColumnSpec = ColumnSpec(),
    # Optional instrumentation (preferred: pass real pipeline values)
    pipeline_runtime_seconds: Optional[float] = None,
    records_cleaned_automatically: Optional[int] = None,
    total_records_cleaned: Optional[int] = None,
    assumed_hourly_rate_usd: float = 50.0,
    prior_error_ledger_csv: Optional[Path] = None,
) -> ReportOutputs:
    """Generate Before/After DQ report artifacts.

    Fix coverage:
      1) robust date parsing + suspicious date flag
      2) configurable/defensible validity checks (domains + location)
      3) error ledger aligned by explicit join key (shared id or canonical hash)
      4) completeness outputs include per-column + average
      5) text metrics auto-detect text column
      6) explicit definitions + metadata for reviewers
      7) processing time/cost use pipeline runtime if provided (else NA)
      8) recurrence rate optional via prior ledger
    """
    out_dir = cfg.processed_dir / "data_quality_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()

    df_b_raw = _coerce_columns(_read_dataset(before.path))
    df_a_raw = _coerce_columns(_read_dataset(after.path))

    id_b = _pick_id_column(df_b_raw, colspec)
    id_a = _pick_id_column(df_a_raw, colspec)
    df_b = _ensure_index_id(df_b_raw, id_b)
    df_a = _ensure_index_id(df_a_raw, id_a)

    # Auto-detect text column for text quality metrics
    text_col_b = _detect_text_col(df_b, colspec)
    text_col_a = _detect_text_col(df_a, colspec)

    # Core columns
    core_cols = [colspec.title, colspec.company, colspec.location, colspec.date_posted, colspec.domain]

    # Validity checks (defensible + configurable)
    allowed = _allowed_domains(cfg, df_a, colspec.domain)
    validity_checks: Dict[str, Callable[[pd.Series], pd.Series]] = {
        colspec.date_posted: lambda s: _parse_date_series_robust(s).notna(),
        colspec.location: _location_valid,
    }
    if allowed is not None:
        validity_checks[colspec.domain] = lambda s: (
            s.fillna("").astype(str).str.strip().str.lower().isin(allowed)
        )

    # Consistency rules (real cross-field logic)
    consistency_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
    if colspec.title in df_a.columns and colspec.title_clean in df_a.columns:
        consistency_rules["title_clean_nonempty_if_title_nonempty"] = lambda d: (
            d[colspec.title].fillna("").astype(str).str.strip().eq("") |
            d[colspec.title_clean].fillna("").astype(str).str.strip().ne("")
        )

    # Alignment key for before/after comparisons (error ledger + duplicate deltas)
    align_key, df_b_aligned, df_a_aligned, align_meta = _build_alignment_key(df_b, df_a, colspec)

    # Build metric rows
    def build_row(name: str, df: pd.DataFrame, key_col: str, text_col: Optional[str]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "dataset": name,
            "n_records": int(len(df)),
            "id_col_used": key_col,
            "alignment_key_used": align_key,
        }

        # completeness: both avg + per-col
        row["completeness_rate_avg_core"] = completeness_rate_avg(df, core_cols)
        row.update(completeness_by_col(df, core_cols))

        row["validity_rate_avg_checks"] = validity_rate(df, validity_checks)
        row["uniqueness_rate"] = uniqueness_rate(df, key_col)
        row["consistency_rate_avg_rules"] = consistency_rate(df, consistency_rules) if consistency_rules else None

        row.update(_timeliness_summary(df, colspec.date_posted))

        row["entity_contamination_proxy"] = entity_contamination_proxy(df, text_col, colspec.company, colspec.location)
        row.update(text_informativeness(df, text_col))

        # Accuracy is manual (export audit sample)
        row["accuracy_rate_manual"] = None
        return row

    metrics_df = pd.DataFrame([
        build_row(before.name, df_b, id_b, text_col_b),
        build_row(after.name, df_a, id_a, text_col_a),
    ])

    # Process deltas (null reductions + duplicate deltas)
    deltas = issue_volume_fixed(df_b_aligned, df_a_aligned, core_cols, key_col=align_key)
    deltas_df = pd.DataFrame([{"before": before.name, "after": after.name, **deltas}])

    # Accuracy audit sample (after)
    sample_n = min(250, len(df_a))
    audit_cols = [c for c in [id_a, colspec.title, colspec.company, colspec.location, colspec.date_posted, colspec.domain, text_col_a] if c and c in df_a.columns]
    audit_df = df_a[audit_cols].sample(n=sample_n, random_state=7).copy()
    audit_df["accurate_pass_fail"] = ""
    audit_df["accuracy_notes"] = ""

    # Error ledger (aligned)
    error_ledger = _build_error_ledger(
        df_b_aligned,
        df_a_aligned,
        core_cols=core_cols,
        validity_checks=validity_checks,
        align_key=align_key,
    )

    # Recurrence (optional)
    prior_ledger = pd.DataFrame()
    if prior_error_ledger_csv is not None and prior_error_ledger_csv.exists():
        try:
            prior_ledger = pd.read_csv(prior_error_ledger_csv)
        except Exception:
            prior_ledger = pd.DataFrame()
    recurrence_rate = _recurrence_rate_from_ledgers(prior_ledger, error_ledger) if not prior_ledger.empty else None

    # Compute process metrics from ledger (aligned)
    total_errors_identified = int(error_ledger["before_has_issue"].sum()) if not error_ledger.empty else 0
    total_errors_remaining = int(error_ledger["after_has_issue"].sum()) if not error_ledger.empty else 0
    errors_fixed = int(error_ledger["fixed"].sum()) if not error_ledger.empty else 0

    # Add duplicate reductions to errors_fixed if we computed both sides
    dup_fixed = None
    if "duplicates_before" in deltas and "duplicates_after" in deltas:
        dup_fixed = max(int(deltas["duplicates_before"]) - int(deltas["duplicates_after"]), 0)
        errors_fixed += dup_fixed
        total_errors_identified += int(deltas.get("duplicates_before", 0))

    error_resolution_rate = (100.0 * errors_fixed / total_errors_identified) if total_errors_identified > 0 else None

    # Runtime/cost: prefer pipeline runtime (report runtime is not the cleaning job)
    report_runtime_seconds = float(perf_counter() - t0)
    runtime_for_cost = pipeline_runtime_seconds
    if runtime_for_cost is not None:
        cost_per_record = _cost_per_cleaned_record(float(runtime_for_cost), int(len(df_a)), assumed_hourly_rate_usd)
        processing_time_seconds = float(runtime_for_cost)
        processing_time_source = "pipeline_runtime_seconds"
    else:
        cost_per_record = None
        processing_time_seconds = None
        processing_time_source = "not_provided"

    # Automation rate: only if instrumented
    if records_cleaned_automatically is not None and total_records_cleaned:
        automation_rate = 100.0 * (records_cleaned_automatically / max(int(total_records_cleaned), 1))
        automation_source = "instrumented"
    else:
        automation_rate = None
        automation_source = "not_provided"

    # Append process metrics
    deltas_df["total_errors_identified"] = total_errors_identified
    deltas_df["total_errors_remaining"] = total_errors_remaining
    deltas_df["errors_fixed"] = errors_fixed
    deltas_df["error_resolution_rate"] = error_resolution_rate
    deltas_df["automation_rate"] = automation_rate
    deltas_df["automation_source"] = automation_source
    deltas_df["processing_time_seconds"] = processing_time_seconds
    deltas_df["processing_time_source"] = processing_time_source
    deltas_df["assumed_hourly_rate_usd"] = assumed_hourly_rate_usd if runtime_for_cost is not None else None
    deltas_df["cost_per_cleaned_record_usd"] = cost_per_record
    deltas_df["recurrence_rate"] = recurrence_rate

    # Write artifacts
    metrics_csv = out_dir / "data_quality_metrics.csv"
    metrics_xlsx = out_dir / "data_quality_metrics.xlsx"
    deltas_csv = out_dir / "process_metrics_deltas.csv"
    audit_sample_csv = out_dir / "accuracy_audit_sample.csv"
    error_ledger_csv = out_dir / "error_ledger.csv"
    run_metadata_json = out_dir / "run_metadata.json"

    # Paper-ready table (Overleaf) — safe best-effort, never a failure state
    try:
        table_tex = (out_dir / "tables" / "table_dq_summary.tex")
        write_table_dq_summary_tex(
            table_tex,
            metrics_df,
            deltas_df,
            before_label=str(before.name) if hasattr(before, 'name') else 'Before',
            after_label=str(after.name) if hasattr(after, 'name') else 'After',
        )
    except Exception as e:
        logger.warning(f"Failed to write table_dq_summary.tex: {e}")


    metrics_df.to_csv(metrics_csv, index=False)
    deltas_df.to_csv(deltas_csv, index=False)
    audit_df.to_csv(audit_sample_csv, index=False)
    error_ledger.to_csv(error_ledger_csv, index=False)

    # Excel workbook
    with pd.ExcelWriter(metrics_xlsx, engine="xlsxwriter", engine_kwargs={"options": {"strings_to_urls": False}}) as w:
        metrics_df.to_excel(w, index=False, sheet_name="quality_metrics")
        deltas_df.to_excel(w, index=False, sheet_name="process_metrics")
        audit_df.to_excel(w, index=False, sheet_name="accuracy_audit_sample")
        error_ledger.to_excel(w, index=False, sheet_name="error_ledger")

    # Run metadata (reviewer-facing provenance)
    # Compute a simple match rate for alignment key
    match_rate = None
    if align_key in df_b_aligned.columns and align_key in df_a_aligned.columns:
        b_keys = set(df_b_aligned[align_key].astype(str))
        a_keys = set(df_a_aligned[align_key].astype(str))
        common = len(b_keys.intersection(a_keys))
        match_rate = 100.0 * (common / max(len(a_keys), 1))

    run_metadata = {
        "run_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "before_path": str(before.path),
        "after_path": str(after.path),
        "n_before": int(len(df_b)),
        "n_after": int(len(df_a)),
        "id_before": id_b,
        "id_after": id_a,
        "alignment": {
            **align_meta,
            "match_rate_percent_of_after": match_rate,
        },
        "runtime": {
            "report_runtime_seconds": report_runtime_seconds,
            "pipeline_runtime_seconds": pipeline_runtime_seconds,
        },
        "definitions": {
            "completeness": "Average non-null rate across core fields (plus per-field rates).",
            "validity": "Average validity across checks (date parse, location pattern, optional domain allowlist).",
            "uniqueness": "Unique id values / total records.",
            "consistency": "Average across cross-field rules (e.g., title_clean present when title present).",
            "timeliness": "Date parse rate + min/max date; includes a suspicious-date flag.",
            "accuracy": "Manual audit sample exported; accuracy_rate must be computed from human scoring.",
        },
        "notes": {
            "automation_rate": automation_source,
            "cost_model": "Only computed if pipeline runtime provided; else omitted.",
            "recurrence_rate": "Only computed if prior error ledger provided; else omitted.",
        },
        "artifacts": {
            "metrics_csv": str(metrics_csv),
            "metrics_xlsx": str(metrics_xlsx),
            "deltas_csv": str(deltas_csv),
            "audit_sample_csv": str(audit_sample_csv),
            "error_ledger_csv": str(error_ledger_csv),
        },
    }
    run_metadata_json.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    return ReportOutputs(
        out_dir=out_dir,
        metrics_csv=metrics_csv,
        metrics_xlsx=metrics_xlsx,
        deltas_csv=deltas_csv,
        audit_sample_csv=audit_sample_csv,
        error_ledger_csv=error_ledger_csv,
        run_metadata_json=run_metadata_json,
    )
