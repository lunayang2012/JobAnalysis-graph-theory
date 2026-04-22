from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import json
import logging
import os

import pandas as pd
from job_skills.pipeline.json_utils import json_default
from openai import OpenAI  # pip install openai>=1.0

from ..config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM backend config
# ---------------------------------------------------------------------------

Backend = Literal["openai"]  # extend later with "ollama", etc.


@dataclass
class TopicLLMConfig:
    backend: Backend = "openai"
    model: str = "gpt-5-mini"  # or "gpt-5.1" depending on your budget
    max_examples: int = 5
    max_chars_per_example: int = 400
    temperature: float = 0.2
    language: str = "en"

    @classmethod
    def from_config_and_cli(
        cls,
        cfg: Config,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        max_examples: Optional[int] = None,
        max_chars: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> "TopicLLMConfig":
        """
        Build TopicLLMConfig by combining:
          - Config defaults (config.py)
          - Optional CLI overrides
          - Sensible fallback model names
        """
        # Base from Config
        base_backend = cfg.topic_labeling_backend
        base_model = cfg.topic_labeling_model
        base_max_ex = cfg.topic_labeling_max_examples
        base_max_chars = cfg.topic_labeling_max_chars
        base_temp = cfg.topic_labeling_temperature

        resolved_backend = (backend or base_backend or "openai").lower()
        if resolved_backend not in ("openai",):
            raise ValueError(f"Unsupported topic labeling backend: {resolved_backend!r}")

        resolved_model = model or base_model
        if resolved_model is None:
            # Defaults per backend
            if resolved_backend == "openai":
                # Reasonable SOTA-ish default from current OpenAI lineup
                resolved_model = "gpt-5-mini"

        return cls(
            backend=resolved_backend,                        # type: ignore[arg-type]
            model=resolved_model,
            max_examples=max_examples or base_max_ex,
            max_chars_per_example=max_chars or base_max_chars,
            temperature=temperature or base_temp,
            language= cfg.topic_labeling_language,
        )


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _build_openai_client() -> OpenAI:
    """
    Build an OpenAI client using the modern Responses API style.

    Requires OPENAI_API_KEY in the environment.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment to use the OpenAI backend."
        )
    # The Python SDK reads OPENAI_API_KEY automatically, but we instantiate
    # explicitly to make dependencies obvious.
    client = OpenAI()
    return client


def _call_openai_for_topic(
    client: OpenAI,
    llm_cfg: TopicLLMConfig,
    payload: dict,
) -> dict:
    """
    Call OpenAI Responses API with a structured JSON payload and parse
    a JSON response with short_label / long_label / confidence.
    """
    # We follow the Responses API pattern (recommended for GPT-5.*). :contentReference[oaicite:6]{index=6}
    resp = client.responses.create(
        model=llm_cfg.model,
        input=json.dumps(payload, ensure_ascii=False, default=json_default),
        temperature=llm_cfg.temperature,
    )

    # Simple helper from docs: resp.output_text is the concatenated text. :contentReference[oaicite:7]{index=7}
    text = resp.output_text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: treat whole text as a label
        logger.warning("LLM returned non-JSON for topic %s: %s", payload.get("topic_id"), text[:200])
        data = {
            "short_label": text[:80],
            "long_label": text,
            "confidence": 0.5,
        }
    return data


# ---------------------------------------------------------------------------
# Core labeling logic
# ---------------------------------------------------------------------------


def _build_topic_prompt_payload(
    topic_id: int,
    top_words: str,
    df_topic_jobs: pd.DataFrame,
    llm_cfg: TopicLLMConfig,
    domain: Optional[str],
) -> dict:
    """
    Build a JSON payload describing the topic + example postings for the LLM.
    This makes it easy to parse and is robust to prompt changes.
    """
    examples: list[dict] = []
    for _, row in df_topic_jobs.head(llm_cfg.max_examples).iterrows():
        # Build a concise example record
        title = str(row.get("title", "") or "")
        company = str(row.get("company", "") or "")
        desc = str(row.get("text_for_bertopic", row.get("description", "")) or "")

        if len(desc) > llm_cfg.max_chars_per_example:
            desc = desc[: llm_cfg.max_chars_per_example] + " ..."

        examples.append(
            {
                "title": title,
                "company": company,
                "text": desc,
            }
        )

    payload = {
        "task": "label_bertopic_topic",
        "language": llm_cfg.language,
        "domain": domain or "STEM job postings (biology, chemistry, lab, pharma, etc.)",
        "instructions": (
            "You are labeling topics from BERTopic over STEM job postings. "
            "For each topic, you must produce:\n"
            "1. short_label: 3-7 word concise label suitable as a chart legend (e.g., "
            "'Cell culture & immunoassays', 'Analytical chemistry & HPLC').\n"
            "2. long_label: 1-sentence human-friendly description.\n"
            "3. confidence: float in [0,1].\n"
            "Focus on the main technical theme (skills, methods, subfields), "
            "not generic phrases like 'Miscellaneous'."
        ),
        "topic": {
            "topic_id": int(topic_id),
            "top_words": top_words,
        },
        "examples": examples,
        "output_schema": {
            "short_label": "string, 3-7 words",
            "long_label": "string, 1 sentence",
            "confidence": "float in [0,1]",
        },
    }
    return payload


def label_topics_with_llm(
    cfg: Config,
    topics_df: pd.DataFrame,
    df_jobs: pd.DataFrame,
    domain: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    max_examples: Optional[int] = None,
    max_chars: Optional[int] = None,
    temperature: Optional[float] = None,
) -> pd.DataFrame:
    """
    Main entry point: enrich topics_df with LLM-based labels.

    Returns a *new* dataframe with extra columns:
      - topic_label
      - topic_label_long
      - topic_label_confidence
    """
    llm_cfg = TopicLLMConfig.from_config_and_cli(
        cfg=cfg,
        backend=backend,
        model=model,
        max_examples=max_examples,
        max_chars=max_chars,
        temperature=temperature,
    )

    if llm_cfg.backend == "openai":
        client = _build_openai_client()
    else:
        raise ValueError(f"Unsupported backend: {llm_cfg.backend!r}")

    topics_df = topics_df.copy()

    short_labels: list[str] = []
    long_labels: list[str] = []
    confidences: list[float] = []

    logger.info(
        "Starting LLM topic labeling with backend=%s, model=%s, n_topics=%d",
        llm_cfg.backend,
        llm_cfg.model,
        len(topics_df),
    )

    for _, row in topics_df.iterrows():
        topic_id = int(row["topic_id"])
        top_words = str(row.get("top_words", ""))

        df_topic_jobs = df_jobs[df_jobs["topic_id"] == topic_id]

        if df_topic_jobs.empty:
            logger.warning(
                "No jobs found for topic_id=%d; falling back to top_words.",
                topic_id,
            )
            short_labels.append(top_words.split(",")[0].strip() if top_words else f"Topic {topic_id}")
            long_labels.append(f"Topic {topic_id}: {top_words}")
            confidences.append(0.1)
            continue

        payload = _build_topic_prompt_payload(
            topic_id=topic_id,
            top_words=top_words,
            df_topic_jobs=df_topic_jobs,
            llm_cfg=llm_cfg,
            domain=domain,
        )

        try:
            if llm_cfg.backend == "openai":
                data = _call_openai_for_topic(client, llm_cfg, payload)
            else:
                raise ValueError(f"Unsupported backend during call: {llm_cfg.backend!r}")

            short = str(data.get("short_label") or "").strip()
            long = str(data.get("long_label") or "").strip()
            conf_raw = data.get("confidence", 0.7)

            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = 0.7

            # Basic sanitization
            if not short:
                short = top_words.split(",")[0].strip() if top_words else f"Topic {topic_id}"
            if not long:
                long = f"Topic {topic_id}: {top_words}"

            short_labels.append(short[:80])
            long_labels.append(long)
            confidences.append(max(0.0, min(1.0, conf)))

        except Exception as exc:
            logger.exception("LLM labeling failed for topic_id=%d: %s", topic_id, exc)
            short_labels.append(top_words.split(",")[0].strip() if top_words else f"Topic {topic_id}")
            long_labels.append(f"Topic {topic_id}: {top_words}")
            confidences.append(0.0)

    topics_df["topic_label"] = short_labels
    topics_df["topic_label_long"] = long_labels
    topics_df["topic_label_confidence"] = confidences

    return topics_df


# ---------------------------------------------------------------------------
# File-level helper: label topics for a given domain dir
# ---------------------------------------------------------------------------


def label_topics_for_domain(
    cfg: Config,
    domain: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    max_examples: Optional[int] = None,
    max_chars: Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict[str, Path]:
    """
    High-level convenience:

      - Locate bertopic dir for the domain.
      - Load topics + jobs + Qlik CSVs.
      - Run LLM labeler.
      - Write new labeled topics CSV/XLSX and labeled Qlik CSV.

    Returns mapping: {"topics_csv": Path, "topics_xlsx": Path, "qlik_csv": Path}
    """
    # Mirror the directory logic in render_topic_visuals. :contentReference[oaicite:8]{index=8}
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

    topics_csv = bertopic_dir / "bertopic_topics.csv"
    jobs_with_topics_csv = bertopic_dir / "jobs_with_topics.csv"
    qlik_jobs_csv = bertopic_dir / "jobs_with_topics_qlik.csv"

    if not topics_csv.exists():
        raise FileNotFoundError(f"Topics CSV not found at {topics_csv}")
    if not jobs_with_topics_csv.exists():
        raise FileNotFoundError(f"Jobs-with-topics CSV not found at {jobs_with_topics_csv}")
    if not qlik_jobs_csv.exists():
        raise FileNotFoundError(f"Qlik jobs CSV not found at {qlik_jobs_csv}")

    logger.info("Loading topics from %s", topics_csv)
    topics_df = pd.read_csv(topics_csv)

    logger.info("Loading jobs-with-topics from %s", jobs_with_topics_csv)
    df_jobs = pd.read_csv(jobs_with_topics_csv)

    # Run LLM labeler
    labeled_topics_df = label_topics_with_llm(
        cfg=cfg,
        topics_df=topics_df,
        df_jobs=df_jobs,
        domain=domain,
        backend=backend,
        model=model,
        max_examples=max_examples,
        max_chars=max_chars,
        temperature=temperature,
    )

    # Write labeled topics
    labeled_topics_csv = bertopic_dir / "bertopic_topics_labeled.csv"
    labeled_topics_xlsx = bertopic_dir / "bertopic_topics_labeled.xlsx"

    labeled_topics_df.to_csv(labeled_topics_csv, index=False)

    with pd.ExcelWriter(
        labeled_topics_xlsx,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}},
    ) as writer:
        labeled_topics_df.to_excel(writer, index=False, sheet_name="topics_labeled")

    logger.info("Wrote labeled topics to %s and %s", labeled_topics_csv, labeled_topics_xlsx)

    # Update Qlik CSV with labels (left join on topic_id)
    qlik_df = pd.read_csv(qlik_jobs_csv)

    # Only keep minimal columns from topics to avoid cruft
    join_cols = ["topic_id", "topic_label", "topic_label_long", "topic_label_confidence"]
    join_cols = [c for c in join_cols if c in labeled_topics_df.columns]

    labeled_qlik_df = qlik_df.merge(
        labeled_topics_df[join_cols],
        on="topic_id",
        how="left",
    )

    labeled_qlik_csv = bertopic_dir / "jobs_with_topics_qlik_labeled.csv"
    labeled_qlik_df.to_csv(labeled_qlik_csv, index=False)

    logger.info("Wrote labeled Qlik CSV to %s", labeled_qlik_csv)

    return {
        "topics_csv": labeled_topics_csv,
        "topics_xlsx": labeled_topics_xlsx,
        "qlik_csv": labeled_qlik_csv,
    }
