from __future__ import annotations

import hashlib
import html
import logging
import re
import unicodedata
import numpy as np
import pandas as pd

from ..config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Regex + small helpers
# ---------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_DASH_LINE_RE = re.compile(r"^[\-\u2013\u2014\.\*\•\s]+$")
_MULTI_DASH_RE = re.compile(r"[\-\u2013\u2014]{2,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")

REMOTE_RE = re.compile(r"\b(remote|work\s*from\s*home|wfh|hybrid)\b", re.I)
US_STATE_RE = re.compile(
    r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b"
)
COUNTRY_HINTS = {
    "usa": "United States",
    "us": "United States",
    "united states": "United States",
    "u.s.": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
    "canada": "Canada",
}


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return html.unescape(text)


def _clean_series(s: pd.Series, *, lower: bool = True) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.map(_normalize_unicode)
    s = s.map(lambda x: _URL_RE.sub(" ", x))
    s = s.str.replace(r"[\r\n\t]+", " ", regex=True)
    s = s.map(lambda x: _HTML_TAG_RE.sub(" ", x))

    def drop_dash_only_chunks(text: str) -> str:
        parts = re.split(r"[•\n\r]", text)
        keep: list[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if _DASH_LINE_RE.match(p):
                continue
            keep.append(p)
        return " ".join(keep)

    s = s.map(drop_dash_only_chunks)
    s = s.map(lambda x: _MULTI_DASH_RE.sub("-", x))
    s = s.map(lambda x: _WHITESPACE_RE.sub(" ", x).strip())
    if lower:
        s = s.str.lower()
    return s


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------

_HEADER_MAP = {
    "job_title": "title",
    "jobtitle": "title",
    "title": "title",
    "company_name": "company",
    "employer": "company",
    "company": "company",
    "location": "location",
    "city": "location",
    "state_city": "location",
    "desc": "description",
    "job_description": "description",
    "description": "description",
    "cleaned_description": "cleaned_description",
    "cleaned_description_2": "cleaned_description_2",
    "domain": "domain",
    "source.name": "source.name",
    "min_salary": "min_amount",
    "max_salary": "max_amount",
    "min_amount": "min_amount",
    "max_amount": "max_amount",
    "currency": "currency",
    "site": "site",
    "job_url": "job_url",
    "date_posted": "date_posted",
    "id": "id",
    "posting id": "posting_id",
    "posting_id": "posting_id",
}


def _normalize_column_name(col: str) -> str:
    col_norm = col.strip().lower().replace(" ", "_")
    return _HEADER_MAP.get(col_norm, col_norm)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_column_name(c) for c in df.columns]

    # Ensure these exist (downstream expects them)
    for col in ("title", "description", "company", "location"):
        if col not in df.columns:
            df[col] = ""

    # Derive unified description source
    if "cleaned_description_2" in df.columns:
        df["description"] = df["cleaned_description_2"]
    elif "cleaned_description" in df.columns:
        df["description"] = df["cleaned_description"]

    for col in ("title", "description", "company", "location"):
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .map(lambda x: _WHITESPACE_RE.sub(" ", x).strip())
        )

    return df


# ---------------------------------------------------------------------
# Derived fields: date, location, company normalization, salary
# ---------------------------------------------------------------------


def parse_date_posted(s: pd.Series) -> pd.Series:
    """Robust parse: Excel serials -> datetime; otherwise normal datetime parse."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def parse_location(loc: object) -> tuple[object, object, object, bool, str]:
    """Return: city, state, country, is_remote, location_parse_quality."""
    if pd.isna(loc):
        return (pd.NA, pd.NA, pd.NA, False, "missing")
    raw = str(loc).strip()
    if raw == "" or raw.lower() in {"nan", "none", "null"}:
        return (pd.NA, pd.NA, pd.NA, False, "empty")

    low = raw.lower()
    is_remote = bool(REMOTE_RE.search(low))

    norm = re.sub(r"\s*[-/|]\s*", ", ", raw)
    norm = re.sub(r"\s+", " ", norm).strip()

    if is_remote and "remote" in low:
        tail = re.sub(r"(?i)\bremote\b[:, ]*", "", norm).strip(" ,")
        tail_low = tail.lower()
        country = state = city = pd.NA

        if tail_low in COUNTRY_HINTS:
            country = COUNTRY_HINTS[tail_low]
        else:
            m = US_STATE_RE.search(tail)
            if m and len(tail) <= 4:
                state = m.group(1)
                country = "United States"

        if country is not pd.NA or state is not pd.NA:
            return (city, state, country, True, "remote_with_region")
        return (pd.NA, pd.NA, pd.NA, True, "remote_only")

    parts = [p.strip() for p in norm.split(",") if p.strip()]
    city = state = country = pd.NA

    if len(parts) == 1:
        tok = parts[0]
        tok_low = tok.lower()
        if tok_low in COUNTRY_HINTS:
            return (pd.NA, pd.NA, COUNTRY_HINTS[tok_low], is_remote, "country_only")
        m = US_STATE_RE.fullmatch(tok.upper())
        if m:
            return (pd.NA, m.group(1), "United States", is_remote, "state_only")
        return (pd.NA, tok, pd.NA, is_remote, "single_token_ambiguous")

    if len(parts) == 2:
        city = parts[0]
        right = parts[1]
        right_low = right.lower()
        m = US_STATE_RE.fullmatch(right.upper())
        if m:
            return (city, m.group(1), "United States", is_remote, "city_state")
        if right_low in COUNTRY_HINTS:
            return (city, pd.NA, COUNTRY_HINTS[right_low], is_remote, "city_country")
        return (city, right, pd.NA, is_remote, "city_region_ambiguous")

    city = parts[0]
    state = parts[1]
    last = parts[-1]
    last_low = last.lower()
    if last_low in COUNTRY_HINTS:
        return (city, state, COUNTRY_HINTS[last_low], is_remote, "city_state_country")

    m = US_STATE_RE.fullmatch(str(state).upper())
    if m:
        return (city, m.group(1), "United States", is_remote, "city_state_assumed_us")

    return (city, state, last, is_remote, "multi_part_unverified")


def normalize_company(name: object) -> object:
    if pd.isna(name):
        return pd.NA
    s = str(name).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    suffixes = {"inc", "incorporated", "llc", "l", "ltd", "limited", "corp", "corporation", "co", "company", "gmbh", "plc"}
    tokens = [t for t in s.split() if t not in suffixes]
    out = " ".join(tokens).strip()
    return out or pd.NA


TITLE_RULES = [
    ("Data Scientist", r"\bdata scientist\b"),
    ("ML Engineer", r"\b(machine learning|ml)\s+engineer\b|\bmlops\b"),
    ("Data Engineer", r"\bdata engineer\b|\banalytics engineer\b"),
    ("Data Analyst", r"\bdata analyst\b|\bbusiness intelligence\b|\bbi analyst\b"),
    ("Research Scientist", r"\b(research|applied)\s+scientist\b|\bscientist\b"),
    ("Software Engineer", r"\bsoftware engineer\b|\b(full[- ]?stack|backend|frontend)\b.*\bengineer\b|\bdeveloper\b"),
    ("Product Manager", r"\bproduct manager\b|\bproduct owner\b"),
    ("Project / Program Manager", r"\b(project|program)\s+manager\b|\bpm\b"),
    ("Statistician", r"\bstatistician\b|\bbiostatistician\b"),
    ("Data Architect", r"\bdata architect\b"),
    ("QA / Test", r"\bqa\b|\btest(ing)?\b|\bsdet\b"),
    ("Other", r".*"),
]
_TITLE_REGEX = [(lab, re.compile(pat, re.I)) for lab, pat in TITLE_RULES]


def title_family(title: object) -> object:
    if pd.isna(title):
        return pd.NA
    s = str(title)
    for lab, rx in _TITLE_REGEX:
        if rx.search(s):
            return lab
    return "Other"


# ---------------------------------------------------------------------
# Optional: spaCy cleaning (behind cfg flags)
# ---------------------------------------------------------------------


def _maybe_spacy_clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add normalized, lemmatized description columns.

    Controlled via Config:
      - enable_spacy_clean (bool)
      - spacy_model (str)
      - spacy_max_words (int, 0=unlimited)
      - spacy_disable (tuple[str,...]) optional spaCy pipeline disables

    Output columns:
      - description_cleaned_spacy
      - description_cleaned_spacy_{N}w (if N>0)
    """
    if not getattr(cfg, "enable_spacy_clean", False):
        return df

    try:
        import spacy  # type: ignore
    except Exception as e:
        logger.warning("spaCy requested but not installed: %s", e)
        return df

    model_name = getattr(cfg, "spacy_model", "en_core_web_lg")
    max_words = int(getattr(cfg, "spacy_max_words", 100))
    disable = tuple(getattr(cfg, "spacy_disable", ()))

    try:
        nlp = spacy.load(model_name, disable=list(disable) if disable else None)
    except Exception as e:
        logger.warning("spaCy model '%s' not available (%s). Skipping spaCy clean.", model_name, e)
        return df

    def clean(text: object) -> str:
        doc = nlp(str(text))
        toks: list[str] = []
        for token in doc:
            if token.is_stop or token.is_punct or token.like_num:
                continue
            lemma = token.lemma_.lower().strip()
            if lemma:
                toks.append(lemma)
        return " ".join(toks)

    out = df.copy()
    out["description_cleaned_spacy"] = out["description"].apply(clean)
    if max_words > 0:
        out[f"description_cleaned_spacy_{max_words}w"] = out["description_cleaned_spacy"].astype(str).str.split().str[:max_words].str.join(" ")
    return out


# ---------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------


def _dedupe_exact(df: pd.DataFrame) -> pd.DataFrame:
    """Stable exact dedupe.

    - Prefer job_url / posting_id / id when available.
    - Canonical key: (company_norm, title_norm, desc_hash).
    """
    out = df.copy()

    out["_company_norm"] = out["company"].fillna("").astype(str).str.strip().str.lower()
    out["_title_norm"] = out["title_clean"].fillna("").astype(str).str.strip().str.lower()

    desc_norm = out["text_for_bertopic"].fillna("").astype(str)
    out["_desc_hash"] = desc_norm.map(_hash_text)

    hard_id = None
    for c in ("job_url", "posting_id", "id"):
        if c in out.columns:
            hard_id = c
            break

    if hard_id is not None:
        before = len(out)
        out = out.drop_duplicates(subset=[hard_id], keep="first").copy()
        logger.info("Dedupe by %s removed %d rows.", hard_id, before - len(out))

    out["_canon_key"] = out["_company_norm"] + " | " + out["_title_norm"] + " | " + out["_desc_hash"]

    before = len(out)
    out = out.sort_index().drop_duplicates("_canon_key", keep="first").copy()
    logger.info("Dedupe by canon_key removed %d rows.", before - len(out))

    if "uid" in out.columns:
        base_id = out["uid"].fillna("").astype(str)
    elif "id" in out.columns:
        base_id = out["id"].fillna("").astype(str)
    else:
        base_id = out.index.astype(str)
    out["uid"] = base_id + "|" + out["_title_norm"]

    out["token_count"] = out["text_for_bertopic"].astype(str).str.split().map(len)
    return out


def _dedupe_title_fuzzy(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Near-duplicate title clustering + optional collapse (Ricky's approach, productionized).

    Config:
      - enable_title_fuzzy_dedupe: bool
      - title_fuzzy_similarity: int (0-100)
      - title_fuzzy_scorer: "token_sort_ratio" | "token_set_ratio"
      - title_fuzzy_collapse: bool  (if True, keeps 1 row per canonical title)
      - title_fuzzy_keep: "first" | "most_recent"  (only used when collapsing)
      - keep_title_cluster_cols: bool (keep audit columns even if collapsing)

    Output (when enabled):
      - title_normalized
      - title_canonical
      - title_cluster_id
      - title_cluster_size
      - title_fuzzy_score (match score to canonical)
      - title_collapsed (bool) if collapsing
    """
    if not getattr(cfg, "enable_title_fuzzy_dedupe", False):
        return df

    try:
        from rapidfuzz import fuzz, process  # type: ignore
    except Exception as e:
        logger.warning("Title fuzzy dedupe requested but rapidfuzz is not installed: %s", e)
        return df

    threshold = int(getattr(cfg, "title_fuzzy_similarity", 90))
    scorer_name = str(getattr(cfg, "title_fuzzy_scorer", "token_sort_ratio")).strip()
    scorer = fuzz.token_sort_ratio if scorer_name == "token_sort_ratio" else fuzz.token_set_ratio

    collapse = bool(getattr(cfg, "title_fuzzy_collapse", False))
    keep_strategy = str(getattr(cfg, "title_fuzzy_keep", "first")).strip()
    keep_cols = bool(getattr(cfg, "keep_title_cluster_cols", True))

    out = df.copy()

    def normalize_title(t: object) -> str:
        if pd.isna(t):
            return ""
        s = str(t).lower().strip()
        s = s.replace(".", "")
        s = s.replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    out["title_normalized"] = out["title"].apply(normalize_title)

    canonical_map: dict[str, str] = {}
    score_map: dict[str, int] = {}
    unique_titles: list[str] = []

    for t in out["title_normalized"].dropna().unique():
        if not unique_titles:
            unique_titles.append(t)
            canonical_map[t] = t
            score_map[t] = 100
            continue
        match, score, _ = process.extractOne(t, unique_titles, scorer=scorer)
        if score >= threshold:
            canonical_map[t] = match
            score_map[t] = int(score)
        else:
            unique_titles.append(t)
            canonical_map[t] = t
            score_map[t] = 100

    out["title_canonical"] = out["title_normalized"].map(canonical_map)
    out["title_fuzzy_score"] = out["title_normalized"].map(score_map).astype("Int64")

    # Stable cluster id from canonical title
    out["title_cluster_id"] = out["title_canonical"].astype(str).map(lambda x: _hash_text(x)[:12])
    out["title_cluster_size"] = out.groupby("title_canonical")["title_canonical"].transform("size").astype("Int64")

    if not collapse:
        return out

    # Collapse to 1 row per canonical title
    before = len(out)

    # If we can keep "most recent", do it (requires date_posted parsed already)
    if keep_strategy == "most_recent" and "date_posted" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date_posted"]):
        out = out.sort_values(["title_canonical", "date_posted"], ascending=[True, False])
    else:
        out = out.sort_values(["title_canonical"]).copy()

    out = out.drop_duplicates(subset=["title_canonical"], keep="first").copy()
    out["title_collapsed"] = True

    removed = before - len(out)
    logger.info("Title fuzzy collapse removed %d rows (threshold=%d, scorer=%s).", removed, threshold, scorer_name)

    if not keep_cols:
        out = out.drop(columns=[c for c in ["title_normalized", "title_canonical", "title_cluster_id", "title_cluster_size", "title_fuzzy_score"] if c in out.columns], errors="ignore")
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def run_preprocess(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Deterministic preprocessing.

    - normalize columns
    - clean title + description into title_clean / text_for_bertopic
    - derive metadata fields for downstream modeling + dashboard
    - optional spaCy cleaned text columns
    - exact dedupe + optional fuzzy-title clustering/collapse
    """
    out = _normalize_columns(df)

    if "domain" not in out.columns:
        out["domain"] = out.get("source_domain", pd.NA)

    out["title_clean"] = _clean_series(out["title"], lower=True)
    out["text_for_bertopic"] = _clean_series(out["description"], lower=True)

    if "date_posted" in out.columns:
        out["date_posted"] = parse_date_posted(out["date_posted"])
        out["date_posted_year"] = out["date_posted"].dt.year.astype("Int64")
        out["date_posted_month"] = out["date_posted"].dt.to_period("M").astype("string")
        out.loc[out["date_posted"].isna(), "date_posted_month"] = pd.NA

        ref = getattr(cfg, "reference_date", None)
        if ref is not None:
            ref_ts = pd.Timestamp(ref)
            out["posting_age_days"] = (ref_ts.normalize() - out["date_posted"].dt.normalize()).dt.days.astype("Int64")

    parsed = out["location"].apply(parse_location)
    out[["city", "state", "country", "is_remote", "location_parse_quality"]] = pd.DataFrame(parsed.tolist(), index=out.index)

    for col in ("min_amount", "max_amount"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.NA

    out["salary_midpoint"] = np.where(
        out["min_amount"].notna() & out["max_amount"].notna(),
        (out["min_amount"] + out["max_amount"]) / 2.0,
        np.where(out["min_amount"].notna(), out["min_amount"], out["max_amount"]),
    )
    out["salary_midpoint"] = pd.to_numeric(out["salary_midpoint"], errors="coerce")
    out["has_salary"] = out[["min_amount", "max_amount", "salary_midpoint"]].notna().any(axis=1)

    out["company_normalized"] = out["company"].apply(normalize_company)

    if "site" in out.columns:
        s = out["site"].astype("string").str.strip().str.lower()
        site_map = {
            "linkedin": "LinkedIn",
            "indeed": "Indeed",
            "glassdoor": "Glassdoor",
            "ziprecruiter": "ZipRecruiter",
            "monster": "Monster",
            "google jobs": "Google Jobs",
        }
        out["site"] = s.map(lambda x: site_map.get(x, x.title() if isinstance(x, str) else x))

    out["job_title"] = out["title_clean"].where(out["title_clean"].ne(""), out["title"])
    out["title_family"] = out["job_title"].apply(title_family)

    if "label_weak" in out.columns:
        out["label_weak_int"] = pd.to_numeric(out["label_weak"], errors="coerce").fillna(0).astype("Int64")
        out["label_weak_bool"] = out["label_weak_int"].astype("Int64").eq(1)

    out = _maybe_spacy_clean(out, cfg)

    out = _dedupe_exact(out)
    out = _dedupe_title_fuzzy(out, cfg)

    out = out[out["token_count"] >= int(getattr(cfg, "min_tokens", 15))].copy()
    return out
