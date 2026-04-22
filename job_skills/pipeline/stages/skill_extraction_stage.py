from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from ..config import Config

logger = logging.getLogger(__name__)


# -----------------------------
# Small utilities
# -----------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\+\-\.#]*")  # keeps C++, C#, SQL, etc.

# Very small alias map to prevent “Excel/excellence” type mistakes
# Expand this over time (or load from a CSV)
SKILL_ALIASES = {
    "excel": ["microsoft excel", "ms excel", "excel"],
    "power bi": ["power bi", "powerbi"],
    "sql": ["sql", "t-sql", "postgresql", "mysql", "sqlite"],
    "python": ["python"],
}

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _contains_grounded_phrase(text: str, phrase: str) -> bool:
    """
    Grounding rule: phrase must appear as whole-word match in the source text.
    For multiword phrases: require all tokens in order with boundaries.
    """
    t = _norm_text(text)
    p = _norm_text(phrase)
    if not p:
        return False

    # simple whole-word regex for the full phrase
    # \b doesn't work perfectly for +/#, so we do token-based check too
    # First try phrase substring with boundary-ish guards
    if re.search(rf"(^|[^A-Za-z0-9]){re.escape(p)}([^A-Za-z0-9]|$)", t):
        return True

    # token-order check
    toks = [tok for tok in _WORD_RE.findall(p)]
    if not toks:
        return False
    idx = 0
    for tok in toks:
        m = re.search(rf"(^|[^A-Za-z0-9]){re.escape(tok)}([^A-Za-z0-9]|$)", t[idx:])
        if not m:
            return False
        idx += m.end()
    return True

def _expand_aliases(phrase: str) -> list[str]:
    """Return alias variants to test for grounding."""
    p = _norm_text(phrase)
    outs = {p}
    # If the phrase is a canonical key
    if p in SKILL_ALIASES:
        outs.update(SKILL_ALIASES[p])
    # If the phrase matches one of the alias values, include its canonical key
    for canon, vals in SKILL_ALIASES.items():
        if p == canon or p in vals:
            outs.add(canon)
            outs.update(vals)
    return sorted(outs)

def _ground_phrase(text: str, phrase: str) -> bool:
    """Ground phrase either directly or via alias expansion."""
    for variant in _expand_aliases(phrase):
        if _contains_grounded_phrase(text, variant):
            return True
    return False

def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for it in items:
        itn = _norm_text(it)
        if not itn or itn in seen:
            continue
        seen.add(itn)
        out.append(it.strip())
    return out


# -----------------------------
# RAKE-style phrase extraction (no nltk download)
# -----------------------------

def _rake_phrases(text: str, *, max_phrases: int = 15) -> list[str]:
    """
    Lightweight RAKE-like keyword extraction using ENGLISH_STOP_WORDS.
    Splits text on stopwords/punct, scores words by degree/frequency, then scores phrases.
    This is RAKE-style (not a dependency-heavy RAKE implementation).
    """
    t = _norm_text(text)
    if not t:
        return []

    # split into candidate phrases
    tokens = re.split(r"[^A-Za-z0-9\+\-\.#]+", t)
    # rebuild phrases by stopword boundaries
    phrases: list[list[str]] = []
    cur: list[str] = []
    for tok in tokens:
        if not tok:
            continue
        if tok in ENGLISH_STOP_WORDS:
            if cur:
                phrases.append(cur)
                cur = []
            continue
        cur.append(tok)
    if cur:
        phrases.append(cur)

    if not phrases:
        return []

    # word scores
    freq = {}
    degree = {}
    for ph in phrases:
        unique_len = len(ph)
        for w in ph:
            freq[w] = freq.get(w, 0) + 1
            degree[w] = degree.get(w, 0) + (unique_len - 1)

    word_score = {w: (degree[w] + freq[w]) / max(freq[w], 1) for w in freq}

    # phrase scores
    scored = []
    for ph in phrases:
        score = sum(word_score.get(w, 0.0) for w in ph)
        phrase = " ".join(ph)
        # basic cleanup: avoid ultra-short junk
        if len(phrase) < 3:
            continue
        scored.append((score, phrase))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in scored[:max_phrases]]
    return _dedupe_keep_order(top)


# -----------------------------
# TF-IDF per-document top terms
# -----------------------------

def _tfidf_top_terms(texts: list[str], *, top_k: int = 15) -> list[list[str]]:
    """
    Fit a TF-IDF vectorizer on the corpus and return top_k terms per doc.
    """
    # Note: keep tokens like "c++" and "c#" reasonably via token_pattern override
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=200_000,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9\+\-\.#]+\b",
        lowercase=True,
    )
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    top_terms_per_doc: list[list[str]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            top_terms_per_doc.append([])
            continue
        idx = row.indices
        data = row.data
        # choose highest weights
        top_idx = idx[np.argsort(data)[::-1][:top_k]]
        top_terms_per_doc.append(_dedupe_keep_order(list(terms[top_idx])))
    return top_terms_per_doc


# -----------------------------
# spaCy-based skills + responsibilities
# -----------------------------

def _load_spacy(cfg: Config):
    try:
        import spacy  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "spaCy is required for Option A responsibilities extraction. "
            "Install it and an English model, e.g.:\n"
            "  pip install spacy\n"
            "  python -m spacy download en_core_web_sm\n"
        ) from e

    model = getattr(cfg, "spacy_model", "en_core_web_sm")
    disable = list(getattr(cfg, "spacy_disable", ()))
    nlp = spacy.load(model, disable=disable if disable else None)
    return nlp


def _spacy_noun_skills(doc, *, max_items: int = 15) -> list[str]:
    """
    Candidate skills from noun chunks + proper nouns (tools/tech).
    Heuristic filters reduce junk.
    """
    out: list[str] = []

    # noun chunks (multiword)
    for chunk in getattr(doc, "noun_chunks", []):
        txt = chunk.text.strip()
        if len(txt) < 3:
            continue
        if chunk.root.is_stop:
            continue
        # avoid generic chunks
        if txt.lower() in {"experience", "ability", "skills", "responsibilities", "requirements"}:
            continue
        out.append(txt)

    # standalone PROPN/NOUN tokens that look tool-ish (SQL, Python, etc.)
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.like_num:
            continue
        if tok.pos_ not in {"PROPN", "NOUN"}:
            continue
        txt = tok.text.strip()
        if len(txt) < 2:
            continue
        # ignore very generic nouns
        if txt.lower() in {"team", "work", "data", "projects", "project", "role"}:
            continue
        out.append(txt)

    return _dedupe_keep_order(out)[:max_items]


def _spacy_responsibilities(doc, *, max_items: int = 12) -> list[str]:
    """
    Responsibilities = action-oriented phrases (verb + object).
    This is a lightweight pattern extractor, good enough for the professor’s request.
    """
    out: list[str] = []
    for tok in doc:
        if tok.pos_ != "VERB":
            continue
        if tok.lemma_.lower() in {"be", "have", "do"}:
            continue

        # Find a direct object, attribute, or object of a preposition
        dobj = None
        for child in tok.children:
            if child.dep_ in {"dobj", "attr", "oprd"}:
                dobj = child
                break

        phrase = None
        if dobj is not None:
            # include dobj subtree (compact)
            obj_txt = " ".join(t.text for t in dobj.subtree)
            phrase = f"{tok.lemma_} {obj_txt}"
        else:
            # fallback: verb + nearby complement (prep pobj)
            pobj = None
            for child in tok.children:
                if child.dep_ == "prep":
                    for g in child.children:
                        if g.dep_ == "pobj":
                            pobj = " ".join(t.text for t in g.subtree)
                            break
                if pobj:
                    break
            if pobj:
                phrase = f"{tok.lemma_} {pobj}"

        if phrase:
            # keep it short
            phrase = phrase.strip()
            if len(phrase) >= 6:
                out.append(phrase)

    return _dedupe_keep_order(out)[:max_items]


# -----------------------------
# Outputs
# -----------------------------

@dataclass(frozen=True)
class SkillExtractionOutputs:
    out_csv: Path
    out_xlsx: Path
    metrics_json: Path


def run_skill_extraction(cfg: Config, *, input_path: Path, out_dir: Path | None = None) -> SkillExtractionOutputs:
    """
    Option A skill extraction stage:
      - skills: TF-IDF terms + RAKE-style phrases + spaCy noun skills
      - responsibilities: spaCy verb phrase patterns
      - grounding/sanity: keep only phrases grounded in the source text (with alias expansion)
    """
    if out_dir is None:
        out_dir = cfg.processed_dir / "skills"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading input: %s", input_path)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input type: {input_path}")

    # Choose the best available text column
    if "text_for_bertopic" in df.columns:
        text_col = "text_for_bertopic"
    elif "description" in df.columns:
        text_col = "description"
    else:
        raise ValueError("No usable text column found (expected 'text_for_bertopic' or 'description').")

    texts = df[text_col].fillna("").astype(str).tolist()

    # TF-IDF + RAKE-style
    logger.info("Computing TF-IDF top terms...")
    tfidf_terms = _tfidf_top_terms(texts, top_k=15)

    logger.info("Computing RAKE-style phrases...")
    rake_terms = [_rake_phrases(t, max_phrases=15) for t in texts]

    # spaCy pass
    logger.info("Loading spaCy model for POS/chunks/responsibilities...")
    nlp = _load_spacy(cfg)

    spacy_skills: list[list[str]] = []
    responsibilities: list[list[str]] = []

    logger.info("Running spaCy over %d documents...", len(texts))
    for doc in nlp.pipe(texts, batch_size=64):
        spacy_skills.append(_spacy_noun_skills(doc, max_items=15))
        responsibilities.append(_spacy_responsibilities(doc, max_items=12))

    # Combine skill candidates and apply grounding filter
    grounded_skills: list[list[str]] = []
    ungrounded_skills: list[list[str]] = []

    logger.info("Applying grounding / sanity filters...")
    for t, a, b, c in zip(texts, tfidf_terms, rake_terms, spacy_skills, strict=True):
        candidates = _dedupe_keep_order([*a, *b, *c])
        g = []
        ug = []
        for ph in candidates:
            if _ground_phrase(t, ph):
                g.append(ph)
            else:
                ug.append(ph)
        grounded_skills.append(g[:25])
        ungrounded_skills.append(ug[:25])

    # Flatten to string columns (comma-separated)
    df_out = df.copy()
    df_out["skills_tfidf"] = [", ".join(x) for x in tfidf_terms]
    df_out["skills_rake"] = [", ".join(x) for x in rake_terms]
    df_out["skills_spacy"] = [", ".join(x) for x in spacy_skills]
    df_out["responsibilities_spacy"] = [", ".join(x) for x in responsibilities]
    df_out["skills_grounded"] = [", ".join(x) for x in grounded_skills]
    df_out["skills_ungrounded"] = [", ".join(x) for x in ungrounded_skills]
    df_out["sanity_flag_excel_like"] = [
        ("excel" in _norm_text(s) and not _ground_phrase(t, "excel"))
        for t, s in zip(texts, df_out["skills_grounded"].tolist(), strict=True)
    ]

    # Metrics
    n = len(df_out)
    grounded_counts = np.array([len(x) for x in grounded_skills], dtype=int)
    ungrounded_counts = np.array([len(x) for x in ungrounded_skills], dtype=int)

    metrics = {
        "n_rows": int(n),
        "text_col": text_col,
        "avg_grounded_skills": float(grounded_counts.mean()) if n else 0.0,
        "avg_ungrounded_candidates": float(ungrounded_counts.mean()) if n else 0.0,
        "pct_rows_with_any_grounded_skills": float((grounded_counts > 0).mean() * 100.0) if n else 0.0,
        "n_rows_flagged_excel_like": int(df_out["sanity_flag_excel_like"].sum()),
        "pct_rows_flagged_excel_like": float(df_out["sanity_flag_excel_like"].mean() * 100.0) if n else 0.0,
    }

    out_csv = out_dir / "jobs_with_skills.csv"
    out_xlsx = out_dir / "jobs_with_skills.xlsx"
    metrics_json = out_dir / "skills_metrics.json"

    logger.info("Writing: %s", out_csv)
    df_out.to_csv(out_csv, index=False)

    logger.info("Writing: %s", out_xlsx)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter", engine_kwargs={"options": {"strings_to_urls": False}}) as w:
        df_out.to_excel(w, index=False, sheet_name="jobs_with_skills")

    logger.info("Writing: %s", metrics_json)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Skill extraction complete.")
    return SkillExtractionOutputs(out_csv=out_csv, out_xlsx=out_xlsx, metrics_json=metrics_json)
