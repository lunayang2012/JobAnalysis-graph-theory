from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json
from hashlib import md5

import logging

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
import plotly as plotly
import plotly.express as px
import plotly.io as pio
from job_skills.pipeline.json_utils import json_default

from ..config import Config, Mode

logger = logging.getLogger(__name__)

@dataclass
class BERTopicOutputs:
    model_dir: Path
    topics_csv: Path
    jobs_with_topics_csv: Path
    html_dashboard: Path
    qlik_jobs_csv: Path | None = None
    metadata_json: Path | None = None

def _build_docs_for_bertopic(df_jobs: pd.DataFrame, cfg: Config) -> list[str]:
    """
    Build the list of documents passed to BERTopic based on config.

    Modes:
      - "legacy": use the existing 'text_for_bertopic' column (unchanged behavior).
      - "combo":  build a richer representation per row using available fields:
                  title, domain, seniority, job_function, description.
    """
    mode = getattr(cfg, "bertopic_text_mode", "legacy")

    if mode == "legacy":
        if "text_for_bertopic" not in df_jobs.columns:
            raise KeyError(
                "Expected 'text_for_bertopic' column in jobs dataframe for BERTopic "
                "when bertopic_text_mode='legacy'."
            )
        return df_jobs["text_for_bertopic"].astype(str).tolist()

    if mode == "combo":
        # We keep this robust: use fields if they exist, skip silently if not.
        def _row_to_text(row: pd.Series) -> str:
            parts: list[str] = []

            if "title" in row and isinstance(row["title"], str) and row["title"].strip():
                parts.append(f"TITLE: {row['title'].strip()}")

            if "domain" in row and isinstance(row["domain"], str) and row["domain"].strip():
                parts.append(f"DOMAIN: {row['domain'].strip()}")

            if "seniority" in row and isinstance(row["seniority"], str) and row["seniority"].strip():
                parts.append(f"SENIORITY: {row['seniority'].strip()}")

            if "job_function" in row and isinstance(row["job_function"], str) and row["job_function"].strip():
                parts.append(f"FUNCTION: {row['job_function'].strip()}")

            # Fallback: use 'description' if available; otherwise fall back to
            # legacy 'text_for_bertopic' so we never drop the main text.
            desc = None
            if "description" in row and isinstance(row["description"], str):
                desc = row["description"]
            elif "text_for_bertopic" in row and isinstance(row["text_for_bertopic"], str):
                desc = row["text_for_bertopic"]

            if desc:
                parts.append(f"DESCRIPTION: {str(desc).strip()}")

            # At minimum we want some text; if everything is empty, return empty string.
            return "  ".join(parts).strip()

        combo_col = "text_for_bertopic_combo"
        df_jobs[combo_col] = df_jobs.apply(_row_to_text, axis=1)
        return df_jobs[combo_col].astype(str).tolist()

    # Fails fast on typos in config
    raise ValueError(f"Unknown bertopic_text_mode={mode!r}. Expected 'legacy' or 'combo'.")


def _embedding_cache_paths(bertopic_dir: Path) -> tuple[Path, Path]:
    """
    Paths for doc-embedding cache for a given BERTopic output directory.
    """
    emb_path = bertopic_dir / "doc_embeddings.npy"
    meta_path = bertopic_dir / "doc_embeddings_meta.json"
    return emb_path, meta_path


def _compute_docs_md5(docs: list[str]) -> str:
    """
    Cheap content hash for a list of documents. Used to ensure that a cached
    embedding matrix actually matches the current docs.
    """
    hasher = md5()
    for text in docs:
        if not isinstance(text, str):
            text = str(text)
        hasher.update(text.encode("utf-8", errors="ignore"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _load_cached_embeddings(
    bertopic_dir: Path,
    expected_model_name: str,
    docs: list[str],
) -> Optional[np.ndarray]:
    """
    Try to load cached embeddings for the current docs from bertopic_dir.

    Valid only if:
      * cache files exist
      * model_name matches
      * n_docs matches
      * docs_md5 matches
    """
    emb_path, meta_path = _embedding_cache_paths(bertopic_dir)
    if not emb_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Failed to read embeddings cache metadata from %s: %s",
            meta_path,
            exc,
        )
        return None

    if meta.get("model_name") != expected_model_name:
        logger.info(
            "Embedding cache model_name mismatch (expected=%r, found=%r); recomputing.",
            expected_model_name,
            meta.get("model_name"),
        )
        return None

    if meta.get("n_docs") != len(docs):
        logger.info(
            "Embedding cache n_docs mismatch (expected=%d, found=%s); recomputing.",
            len(docs),
            meta.get("n_docs"),
        )
        return None

    docs_md5 = _compute_docs_md5(docs)
    if meta.get("docs_md5") != docs_md5:
        logger.info("Embedding cache docs_md5 mismatch; docs changed; recomputing.")
        return None

    try:
        embeddings = np.load(emb_path)
    except Exception as exc:
        logger.warning("Failed to load embeddings cache %s: %s", emb_path, exc)
        return None

    if embeddings.shape[0] != len(docs):
        logger.info(
            "Embedding cache shape mismatch (n_docs=%d, emb_rows=%d); recomputing.",
            len(docs),
            embeddings.shape[0],
        )
        return None

    return embeddings


def _save_embeddings_cache(
    bertopic_dir: Path,
    docs: list[str],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    """
    Save embeddings + metadata so they can be safely reused across runs.
    """
    emb_path, meta_path = _embedding_cache_paths(bertopic_dir)
    bertopic_dir.mkdir(parents=True, exist_ok=True)

    np.save(emb_path, embeddings)

    meta = {
        "model_name": model_name,
        "n_docs": len(docs),
        "docs_md5": _compute_docs_md5(docs),
    }
    meta_path.write_text(json.dumps(meta, default=json_default), encoding="utf-8")


def _get_doc_embeddings_for_run(
    bertopic_dir: Path,
    docs: list[str],
    model_name: str,
    *,
    need_model: bool = True,
    use_cache: bool = True,
    save_cache: bool = True,
) -> tuple[SentenceTransformer | None, np.ndarray]:
    """
    Unified helper for Approach 1 + 3:

      * Within a single run:
          - compute embeddings once and reuse in memory
      * Across runs:
          - cache embeddings on disk under bertopic_dir
            and reload them when docs+model match
    """
    cached = None
    if use_cache:
        cached = _load_cached_embeddings(
        bertopic_dir=bertopic_dir,
        expected_model_name=model_name,
        docs=docs,
    )

    if cached is not None:
        logger.info(
            "Using cached document embeddings from %s (n_docs=%d).",
            bertopic_dir,
            len(docs),
        )
        doc_embeddings = np.asarray(cached, dtype=np.float32)
        embedding_model: SentenceTransformer | None = (
            SentenceTransformer(model_name) if need_model else None
        )
    else:
        logger.info(
            "No valid embedding cache in %s (use_cache=%s). Computing embeddings for %d docs.",
            bertopic_dir,
            use_cache,
            len(docs),
        )
        embedding_model = SentenceTransformer(model_name)
        doc_embeddings = embedding_model.encode(docs, show_progress_bar=True)
        doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
        
    if save_cache:
        _save_embeddings_cache(
            bertopic_dir=bertopic_dir,
            docs=docs,
            embeddings=doc_embeddings,
            model_name=model_name,
            )
    else:
        logger.info("Document embeddings shape: %s", doc_embeddings.shape)
    return embedding_model, doc_embeddings


def _load_input_df(cfg: Config, input_jsonl: Optional[Path]) -> pd.DataFrame:
    """
    Load the JSONL produced by the main pipeline.

    Prefers:
        clean_jobs_all.jsonl.gz
    Falls back to:
        clean_jobs_all.jsonl
    """
    if input_jsonl is None:
        base = cfg.processed_dir / "clean_jobs_all.jsonl"
        gz_path = base.with_suffix(base.suffix + ".gz")  # .jsonl.gz

        if gz_path.exists():
            jsonl_path = gz_path
        elif base.exists():
            jsonl_path = base
        else:
            raise FileNotFoundError(
                f"BERTopic input JSONL not found: {gz_path} or {base}. "
                "Run the main pipeline first to create clean_jobs_all.jsonl.gz."
            )
    else:
        jsonl_path = input_jsonl

        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"BERTopic input JSONL not found: {jsonl_path}."
            )

    logger.info(f"Reading BERTopic input from {jsonl_path}")
    df = pd.read_json(jsonl_path, lines=True)

    if "text_for_bertopic" not in df.columns:
        raise KeyError(
            "Expected 'text_for_bertopic' column in BERTopic input but did not find it."
        )
    if "uid" not in df.columns:
        raise KeyError("Expected 'uid' column in BERTopic input but did not find it.")

    return df


def _fit_bertopic(
    docs: list[str],
    embedding_model: SentenceTransformer,
    doc_embeddings: np.ndarray,
) -> tuple[BERTopic, np.ndarray, Optional[np.ndarray]]:
    """
    Fit a BERTopic model using a pre-loaded embedding model and
    precomputed document embeddings.

    Returns:
      topic_model : fitted BERTopic instance
      topics      : array of topic IDs per document
      probs       : topic probability matrix (or None)
    """
    logger.info("Fitting BERTopic model...")

    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        calculate_probabilities=True,
        verbose=True,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
    )

    topics, probs = topic_model.fit_transform(
        docs,
        embeddings=doc_embeddings,
    )
    topics_arr = np.asarray(topics)

    return topic_model, topics_arr, probs


def _build_topics_table(
    topic_model: BERTopic,
    topics: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a topics table: one row per topic with top words and simple stats.
    """
    topics_dict = topic_model.get_topics()
    if not topics_dict:
        raise RuntimeError("BERTopic returned an empty topics dictionary.")

    rows = []
    for topic_id, words in topics_dict.items():
        if topic_id == -1:
            # -1 is usually the "outlier" topic in BERTopic
            continue

        top_words = [w for w, _ in words[:10]]
        n_docs = int((topics == topic_id).sum())
        rows.append(
            {
                "topic_id": topic_id,
                "top_words": ", ".join(top_words),
                "n_docs": n_docs,
            }
        )

    topics_df = pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)
    return topics_df


def _build_jobs_with_topics_df(
    topic_model: BERTopic,
    topics: np.ndarray,
    probs: Optional[np.ndarray],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join per-job topics/probs with useful metadata.
    """
    df_jobs = df.copy()
    df_jobs["topic_id"] = topics

    if probs is not None:
        # probs is (n_docs, n_topics); use max probability as simple confidence score
        df_jobs["topic_prob"] = probs.max(axis=1)
    else:
        df_jobs["topic_prob"] = None

    # Keep a focused set of columns for downstream use
    keep_cols = [
        "uid",
        "title",
        "company",
        "domain",
        "p_rnd",
        "topic_id",
        "topic_prob",
        "text_for_bertopic",
    ]
    keep_cols = [c for c in keep_cols if c in df_jobs.columns]

    return df_jobs[keep_cols].copy()


def _build_datamapplot_figure(
    topic_model: BERTopic,
    df_jobs: pd.DataFrame,
    doc_embeddings: Optional[np.ndarray] = None,
):
    """
    Build a 2D UMAP + DataMapPlot visualization over the job embeddings.

    Uses the same embedding model BERTopic is using and the new
    `visualize_document_datamap` API.

    If doc_embeddings is provided, it is reused directly and we avoid
    recomputing embeddings; otherwise we fall back to computing them.
    """
    if "text_for_bertopic" not in df_jobs.columns:
        raise KeyError(
            "Expected 'text_for_bertopic' column in jobs dataframe for DataMapPlot."
        )

    docs = df_jobs["text_for_bertopic"].astype(str).tolist()
    if not docs:
        raise ValueError("No documents available in df_jobs for DataMapPlot.")

    # Optional in-run reuse
    if doc_embeddings is not None:
        doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
        if doc_embeddings.shape[0] != len(docs):
            logger.warning(
                "Precomputed embeddings length mismatch for DataMapPlot "
                "(expected n_docs=%d, got=%d); recomputing.",
                len(docs),
                doc_embeddings.shape[0],
            )
            doc_embeddings = None
        else:
            logger.info(
                "Using precomputed document embeddings for DataMapPlot "
                "(n_docs=%d, dim=%d).",
                doc_embeddings.shape[0],
                doc_embeddings.shape[1],
            )

    # Fallback: compute embeddings once if they weren't provided or invalid
    if doc_embeddings is None:
        embedding_model = topic_model.embedding_model
        if embedding_model is None:
            raise RuntimeError(
                "BERTopic model has no embedding_model; cannot build DataMapPlot."
            )

        # Handle both encode() (SentenceTransformers) and embed() (some HF-style models)
        if hasattr(embedding_model, "encode"):
            logger.info("Using embedding_model.encode(.) for document embeddings")
            doc_embeddings = embedding_model.encode(
                docs,
                show_progress_bar=True,
            )
        elif hasattr(embedding_model, "embed"):
            logger.info("Using embedding_model.embed(.) for document embeddings")
            doc_embeddings = embedding_model.embed(
                docs,
                show_progress_bar=True,
            )
        else:
            raise TypeError(
                "embedding_model does not support .encode() or .embed(); "
                "cannot compute document embeddings for DataMapPlot."
            )

        doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)

    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    reduced_embeddings = reducer.fit_transform(doc_embeddings)

    # BERTopic API (document datamap)
    fig = topic_model.visualize_document_datamap(
        docs,
        reduced_embeddings=reduced_embeddings,
    )

    return fig


def _build_additional_figures(
    topics_df: pd.DataFrame, df_jobs: pd.DataFrame
) -> Dict[str, "plotly.graph_objs._figure.Figure"]:
    """
    Create additional Plotly visualizations to embed into the HTML dashboard.
    """
    figs: Dict[str, "plotly.graph_objs._figure.Figure"] = {}


    # llm label preference
    if "topic label" in topics_df.columns:
        topic_name_col = "topic label"
    else:
        topic_name_col = "topic_id"

    # Topic frequency bar chart
    fig_freq = px.bar(
        topics_df,
        x=topic_name_col,
        y="n_docs",
        title="Number of Jobs per Topic",
    )
    figs["topic_frequency"] = fig_freq

    # Average p_rnd by topic (if available)
    if {topic_name_col, "p_rnd"}.issubset(df_jobs.columns):
        avg_p_rnd = (
            df_jobs.dropna(subset=["p_rnd"])
            .groupby(topic_name_col)["p_rnd"]
            .mean()
            .reset_index()
        )
        if not avg_p_rnd.empty:
            fig_prnd = px.bar(
                avg_p_rnd,
                x=topic_name_col,
                y="p_rnd",
                title="Average FP Model Score (p_rnd) per Topic",
            )
            figs["avg_p_rnd_per_topic"] = fig_prnd

    # Topic confidence histogram (topic_prob) if available
    if "topic_prob" in df_jobs.columns:
        fig_prob = px.histogram(
            df_jobs,
            x="topic_prob",
            nbins=30,
            title="Distribution of Topic Assignment Probabilities",
        )
        figs["topic_prob_histogram"] = fig_prob

    return figs


def _build_bertopic_builtin_figures(
    topic_model: BERTopic,
    df_jobs: pd.DataFrame,
) -> Dict[str, "plotly.graph_objs._figure.Figure"]:
    """
    Use BERTopic's built-in Plotly visualizations and return them in a dict.

    All keys will be human-readable labels used in the HTML sections.
    We wrap each call in try/except so that missing optional deps don't
    crash the whole viz stage.
    """
    figs: Dict[str, "plotly.graph_objs._figure.Figure"] = {}

    # Documents text for visualize_documents
    docs = None
    if "text_for_bertopic" in df_jobs.columns:
        docs = df_jobs["text_for_bertopic"].astype(str).tolist()

    def _safe_add(key: str, fn):
        try:
            fig = fn()
            if fig is not None:
                figs[key] = fig
        except Exception as e:
            logger.warning(f"Skipping BERTopic viz '{key}' due to error: {e}")

    # High-level topic overview
    _safe_add("BERTopic - Topic Overview", topic_model.visualize_topics)

    # Topic barchart (most frequent topics)
    _safe_add(
        "BERTopic - Topic Barchart",
        lambda: topic_model.visualize_barchart(top_n_topics=20),
    )

    # Hierarchical topic structure
    _safe_add("BERTopic - Topic Hierarchy", topic_model.visualize_hierarchy)

    # Topic similarity heatmap
    _safe_add("BERTopic - Topic Heatmap", topic_model.visualize_heatmap)

    # Documents scatter plot colored by topic (if we have docs)
    if docs is not None:
        def _viz_docs():
            return topic_model.visualize_documents(
                docs,
                topics=df_jobs["topic_id"].values,
            )
        _safe_add("BERTopic - Documents", _viz_docs)

    return figs


def _write_html_dashboard(
    out_path: Path,
    fig_map,
    extra_figs: Dict[str, "plotly.graph_objs._figure.Figure"],
) -> None:
    """Write a single HTML dashboard combining DataMapPlot + Plotly figs."""
    logger.info(f"Writing BERTopic HTML dashboard to {out_path}")

    sections: list[str] = ["<h1>Job Skills Topic Modeling Dashboard</h1>"]

    # --- 1) DataMapPlot (Matplotlib/DataMapPlot) as PNG + <img> ---
    if fig_map is not None:
        datamap_png = out_path.parent / "bertopic_datamapplot.png"
        fig_map.savefig(datamap_png, dpi=150, bbox_inches="tight")
        sections.append(
            "<section><h2>Job Skill Landscape (DataMapPlot)</h2>"
            f'<img src="{datamap_png.name}" '
            'alt="Job Skill Landscape (DataMapPlot)" '
            'style="max-width:100%;height:auto;" />'
            "</section>"
        )

    # --- 2) Plotly figs via plotly.io.to_html ---
    include_plotlyjs_first = True
    for i, (key, fig) in enumerate(extra_figs.items()):
        fig_html = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=include_plotlyjs_first and i == 0,
        )
        include_plotlyjs_first = False
        sections.append(f"<section><h2>{key}</h2>{fig_html}</section>")

    body = "\n".join(sections)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Job Skills Topic Modeling Dashboard</title>
</head>
<body>
{body}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _write_qlik_exports(
    bertopic_dir: Path,
    df_jobs: pd.DataFrame,
    topics_df: pd.DataFrame,
    *,
    cfg: Config,
    domain: Optional[str],
    n_before: int,
    n_after_filter: int,
    n_after_sample: int,
) -> tuple[Path, Path]:
    """
    Write Qlik-friendly exports:

      - jobs_with_topics_qlik.csv: tidy schema aimed at QlikCloud
      - bertopic_metadata.json: run parameters and row counts
    """
    # 1) Qlik jobs CSV: stable, flat schema
    qlik_jobs_csv = bertopic_dir / "jobs_with_topics_qlik.csv"

    qlik_cols = [
        "uid",
        "title",
        "company",
        "domain",
        "p_rnd",
        "topic_id",
        "topic_prob",
    ]
    qlik_cols = [c for c in qlik_cols if c in df_jobs.columns]

    df_qlik = df_jobs[qlik_cols].copy()
    df_qlik.to_csv(qlik_jobs_csv, index=False)

    # 2) Metadata JSON
    meta = {
        "mode": cfg.mode,
        "domain": domain,
        "embedding_model": cfg.bertopic_embedding_model_name,
        "bertopic_len_text_min": cfg.bertopic_len_text_min,
        "bertopic_p_rnd_cutoff": cfg.bertopic_p_rnd_cutoff,
        "bertopic_require_label_weak": cfg.bertopic_require_label_weak,
        "n_before_all_filters": n_before,
        "n_after_len_core_filter": n_after_filter,
        "n_after_sampling": n_after_sample,
        "n_topics": int(topics_df["topic_id"].nunique()) if not topics_df.empty else 0,
    }

    metadata_json = bertopic_dir / "bertopic_metadata.json"
    metadata_json.write_text(json.dumps(meta, indent=2, default=json_default), encoding="utf-8")

    return qlik_jobs_csv, metadata_json


def run_bertopic(
    cfg: Config,
    input_jsonl: Optional[Path] = None,
    domain: Optional[str] = None,
    no_datamapplot: bool = False,
    no_bertopic_viz: bool = False,
    max_docs: Optional[int] = None,
) -> BERTopicOutputs:
    """
    End-to-end BERTopic stage (optionally per-domain):

      1. Load clean_jobs_all.jsonl (or a provided JSONL).
      2. Optionally filter to a single domain (e.g. biology, chemistry).
      3. Apply minimal cleaning/filtering for BERTopic:
         - len_text >= 200
         - AND (label_weak == 1 OR p_rnd >= 0.5) when those columns exist.
      4. Optionally subsample docs (max_docs) for quick/stable test runs.
      5. Fit BERTopic on text_for_bertopic.
      6. Save:
         - BERTopic model under cfg.models_dir / "bertopic_model[_<domain>]"
         - bertopic_topics.csv / .xlsx under processed/bertopic[/<domain>]
         - jobs_with_topics.csv / .xlsx under processed/bertopic[/<domain>]
         - bertopic_dashboard.html (DataMapPlot + extra charts).
    """
    logger.info("==== BERTopic Stage: Start ====")
    # 1) Load full dataframe from JSONL
    df = _load_input_df(cfg, input_jsonl)

    # 1a) Optional per-domain filter (robust: strip + substring match).
    # If the filter matches nothing, fall back to ALL domains instead of crashing.
    domain_suffix = ""
    if domain is not None:
        dom_lower = domain.strip().lower()
        domain_suffix = f"_{dom_lower.replace(' ', '_')}"
        if "domain" not in df.columns:
            raise KeyError(
                "Requested domain-level BERTopic run, but 'domain' column "
                "is not present in the input dataframe."
            )

        n_before_dom = len(df)

        domain_series = (
            df["domain"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Substring match so 'biology', 'biology / chemistry', etc. all match
        mask_dom = domain_series.str.contains(dom_lower, na=False)
        n_after_dom = int(mask_dom.sum())

        logger.info(
            "Filtering to domain (substring)=%s: n_before=%d, n_after=%d",
            dom_lower,
            n_before_dom,
            n_after_dom,
        )

        if n_after_dom > 0:
            df = df.loc[mask_dom].reset_index(drop=True)
        else:
            logger.warning(
                "Domain filter %r matched 0 rows. "
                "Proceeding with ALL domains instead of raising an error.",
                domain,
            )
            # df stays unfiltered in this case

    
    # 2) Minimal cleaning / filtering for BERTopic
    if "text_for_bertopic" not in df.columns:
        raise KeyError(
            "Expected 'text_for_bertopic' column in input dataframe for BERTopic."
        )

    # How many rows we are starting with after domain filter
    n_before = len(df)

    # Length threshold from config
    len_min = cfg.bertopic_len_text_min
    df["len_text"] = df["text_for_bertopic"].astype(str).str.len()
    mask_len = df["len_text"] >= len_min

    # Optional "core job" mask: label_weak and/or p_rnd
    core_mask = None

    if "label_weak" in df.columns and cfg.bertopic_require_label_weak:
        core_mask = df["label_weak"].astype(int) == 1

    if "p_rnd" in df.columns:
        pr_mask = df["p_rnd"] >= cfg.bertopic_p_rnd_cutoff
        core_mask = pr_mask if core_mask is None else (core_mask | pr_mask)

    # Combine masks + logging that actually matches config
    if core_mask is not None:
        mask = mask_len & core_mask
        logger.info(
            "Filtering docs for BERTopic: len_text>=%d AND core_mask "
            "(bertopic_require_label_weak=%s, p_rnd_cutoff=%.3f) "
            "(n_before=%d, n_after=%d)",
            len_min,
            cfg.bertopic_require_label_weak,
            cfg.bertopic_p_rnd_cutoff,
            n_before,
            int(mask.sum()),
        )
    else:
        mask = mask_len
        logger.info(
            "Filtering docs for BERTopic: len_text>=%d "
            "(n_before=%d, n_after=%d)",
            len_min,
            n_before,
            int(mask.sum()),
        )

    # Apply filters
    df = df.loc[mask].reset_index(drop=True)
    n_after_filter = len(df)

    if df.empty:
        raise ValueError(
            "After filtering on length and core-job mask, no documents remain for BERTopic. "
            "Relax the thresholds (len_text_min / p_rnd_cutoff / require_label_weak) "
            "or check label_weak/p_rnd availability."
        )

    # 3) Optional subsampling for quick stability runs
    if max_docs is not None and len(df) > max_docs:
        logger.info(
            "Subsampling docs for BERTopic: max_docs=%d, original_n=%d",
            max_docs,
            len(df),
        )
        df = df.sample(n=max_docs, random_state=42).reset_index(drop=True)

    n_after_sample = len(df)

    # 3) Build documents for BERTopic
    docs = _build_docs_for_bertopic(df, cfg)

    # 3.1 Domain-specific output directory for BERTopic artifacts
    bertopic_dir = cfg.processed_dir / "bertopic"
    if domain_suffix:
        bertopic_dir = bertopic_dir / domain_suffix.lstrip("_")
    bertopic_dir.mkdir(parents=True, exist_ok=True)

    embedding_model_name = cfg.bertopic_embedding_model_name

    use_cache = cfg.bertopic_use_embeddings_cache
    save_cache = cfg.bertopic_save_embeddings_cache

    embedding_model, doc_embeddings = _get_doc_embeddings_for_run(
        bertopic_dir=bertopic_dir,
        docs=docs,
        model_name=embedding_model_name,
        need_model=True,
        use_cache=use_cache,
        save_cache=save_cache,
    )


    # 4) Fit BERTopic using precomputed embeddings
    topic_model, topics, probs = _fit_bertopic(
        docs=docs,
        embedding_model=embedding_model,
        doc_embeddings=doc_embeddings,
    )

    # 5) Save model (per-domain path if applicable)
    model_dir = cfg.models_dir / f"bertopic_model{domain_suffix}"
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving BERTopic model to {model_dir}")
    topic_model.save(model_dir)

    # 6) Build and save topics table
    topics_df = _build_topics_table(topic_model, topics, df)

    topics_csv = bertopic_dir / "bertopic_topics.csv"
    topics_df.to_csv(topics_csv, index=False)
    logger.info(f"Wrote topics table to {topics_csv}")

    topics_xlsx = bertopic_dir / "bertopic_topics.xlsx"
    with pd.ExcelWriter(
        topics_xlsx,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}},
    ) as writer:
        topics_df.to_excel(writer, index=False, sheet_name="topics")

    logger.info(f"Wrote topics table to: {topics_xlsx}")

    # 7) Build and save jobs_with_topics
    df_jobs = _build_jobs_with_topics_df(topic_model, topics, probs, df)
    jobs_with_topics_csv = bertopic_dir / "jobs_with_topics.csv"
    df_jobs.to_csv(jobs_with_topics_csv, index=False)
    logger.info(f"Wrote jobs-with-topics table to {jobs_with_topics_csv}")

    jobs_with_topics_xlsx = bertopic_dir / "jobs_with_topics.xlsx"
    with pd.ExcelWriter(
        jobs_with_topics_xlsx,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}},
    ) as writer:
        df_jobs.to_excel(writer, index=False, sheet_name="jobs_with_topics")

    logger.info(f"Wrote jobs-with-topics table to {jobs_with_topics_xlsx}")

    # 8) Qlik exports + metadata
    qlik_jobs_csv, metadata_json = _write_qlik_exports(
        bertopic_dir=bertopic_dir,
        df_jobs=df_jobs,
        topics_df=topics_df,
        cfg=cfg,
        domain=domain,
        n_before=n_before,
        n_after_filter=n_after_filter,
        n_after_sample=n_after_sample,
    )

    # 9) Try to enrich topics_df with LLM labels if they already exist
    labeled_topics_csv = bertopic_dir / "bertopic_topics_labeled.csv"
    if labeled_topics_csv.exists():
        try:
            labeled_topics_df = pd.read_csv(labeled_topics_csv)
            join_cols = [
                "topic_id",
                "topic_label",
                "topic_label_long",
                "topic_label_confidence",
            ]
            join_cols = [c for c in join_cols if c in labeled_topics_df.columns]

            if join_cols:
                topics_df = topics_df.merge(
                    labeled_topics_df[join_cols],
                    on="topic_id",
                    how="left",
                )
                logger.info(
                    "Merged LLM topic labels from %s into topics_df (cols=%s)",
                    labeled_topics_csv,
                    join_cols,
                )
        except Exception as exc:
            logger.warning(
                "Failed to merge labeled topics from %s: %s",
                labeled_topics_csv,
                exc,
            )

    # 10) Visualizations: DataMapPlot + extra charts + BERTopic’s built-in viz
    fig_map = None
    extra_figs: Dict[str, "plotly.graph_objs._figure.Figure"] = {}


    if not no_datamapplot:
        logger.info("Building DataMapPlot visualization.")
        fig_map = _build_datamapplot_figure(
            topic_model,
            df_jobs,
            doc_embeddings=doc_embeddings,
        )
    else:
        logger.info("Skipping DataMapPlot (flag: no_datamapplot=True).")


    # 11) Our custom charts
    extra_figs.update(_build_additional_figures(topics_df, df_jobs))

    # 12) BERTopic built-ins
    if not no_bertopic_viz:
        logger.info("Building BERTopic built-in visualizations.")
        bertopic_figs = _build_bertopic_builtin_figures(topic_model, df_jobs)
        extra_figs.update(bertopic_figs)
    else:
        logger.info("Skipping BERTopic built-in visualizations (flag: no_bertopic_viz=True).")

    html_dashboard = bertopic_dir / "bertopic_dashboard.html"
    _write_html_dashboard(html_dashboard, fig_map, extra_figs)
    logger.info(f"Wrote BERTopic HTML dashboard to {html_dashboard}")

    logger.info("==== BERTopic Stage: Done ====")

    return BERTopicOutputs(
        model_dir=model_dir,
        topics_csv=topics_csv,
        jobs_with_topics_csv=jobs_with_topics_csv,
        html_dashboard=html_dashboard,
        qlik_jobs_csv=qlik_jobs_csv,
        metadata_json=metadata_json,
    )


def render_topic_visuals(
    cfg: Config,
    domain: Optional[str] = None,
    no_datamapplot: bool = False,
    no_bertopic_viz: bool = False,
) -> Path:
    """
    Rebuild only the BERTopic HTML dashboard using the already-saved
    model and CSVs. Does NOT re-fit the model.

    If domain is provided, looks under:
      - models/bertopic_model_<domain>
      - data/processed/bertopic/<domain>/
    """
    logger.info("==== BERTopic Viz-only: Start ====")

    embedding_model_name = cfg.bertopic_embedding_model_name

    domain_suffix = ""
    bertopic_subdir = None
    if domain is not None:
        dom_lower = domain.strip().lower()
        domain_suffix = f"_{dom_lower.replace(' ', '_')}"
        bertopic_subdir = dom_lower.replace(" ", "_")

    if bertopic_subdir is None:
        bertopic_dir = cfg.processed_dir / "bertopic"
    else:
        bertopic_dir = cfg.processed_dir / "bertopic" / bertopic_subdir

    model_dir = cfg.models_dir / f"bertopic_model{domain_suffix}"

    topics_csv = bertopic_dir / "bertopic_topics.csv"
    jobs_with_topics_csv = bertopic_dir / "jobs_with_topics.csv"
    html_dashboard = bertopic_dir / "bertopic_dashboard.html"

    # Sanity checks
    if not model_dir.exists():
        raise FileNotFoundError(
            f"BERTopic model directory not found at {model_dir}. "
            "Run `run-bertopic` once for this domain to fit and save the model."
        )
    if not topics_csv.exists():
        raise FileNotFoundError(
            f"Topics CSV not found at {topics_csv}. "
            "Run `run-bertopic` once for this domain to generate it."
        )
    if not jobs_with_topics_csv.exists():
        raise FileNotFoundError(
            f"Jobs-with-topics CSV not found at {jobs_with_topics_csv}. "
            "Run `run-bertopic` once for this domain to generate it."
        )

    logger.info(f"Loading BERTopic model from {model_dir}")
    topic_model = BERTopic.load(model_dir)

    logger.info(f"Loading topics from {topics_csv}")
    topics_df = pd.read_csv(topics_csv)

    logger.info(f"Loading jobs-with-topics from {jobs_with_topics_csv}")
    df_jobs = pd.read_csv(jobs_with_topics_csv)

    use_cache = cfg.bertopic_use_embeddings_cache
    save_cache = False

    # Prepare embeddings (if we are going to build DataMapPlot)
    doc_embeddings = None
    if not no_datamapplot and "text_for_bertopic" in df_jobs.columns:
        docs = df_jobs["text_for_bertopic"].astype(str).tolist()
        # Reuse the same cache mechanism as the full run
        _, doc_embeddings = _get_doc_embeddings_for_run(
            bertopic_dir=bertopic_dir,
            docs=docs,
            model_name=embedding_model_name,
            need_model=False,
            use_cache=use_cache,
            save_cache=save_cache,

        )

    # Rebuild figures
    fig_map = None
    extra_figs: Dict[str, "plotly.graph_objs._figure.Figure"] = {}

    if not no_datamapplot:
        logger.info("Building DataMapPlot visualization...")
        fig_map = _build_datamapplot_figure(
            topic_model, 
            df_jobs,
            doc_embeddings = doc_embeddings,
            )
    else:
        logger.info("Skipping DataMapPlot (flag: no_datamapplot=True).")

    extra_figs.update(_build_additional_figures(topics_df, df_jobs))

    if not no_bertopic_viz:
        logger.info("Building BERTopic built-in visualizations...")
        bertopic_figs = _build_bertopic_builtin_figures(topic_model, df_jobs)
        extra_figs.update(bertopic_figs)
    else:
        logger.info("Skipping BERTopic built-in visualizations (flag: no_bertopic_viz=True).")

    # Rewrite dashboard
    _write_html_dashboard(html_dashboard, fig_map, extra_figs)

    logger.info(f"==== BERTopic Viz-only: Done. Wrote {html_dashboard} ====")
    return html_dashboard

