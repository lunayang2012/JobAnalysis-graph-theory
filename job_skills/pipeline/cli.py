from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import logging
import pandas as pd

import typer

from .config import Config, Mode
from .runner import run_full_pipeline, train_fp_model_only, preprocess_only, dq_report_only
from .stages import preprocess as run_preprocess
from .stages.bertopic_stage_v1 import run_bertopic as run_bertopic_stage
from .stages.topic_labeler_llm import label_topics_for_domain
from .stages.skill_extraction_stage import run_skill_extraction





# --------------------------------------------------------------------
# Types
# --------------------------------------------------------------------

mode: Mode = typer.Option(
    Mode.precise,
    "--mode",
    "-m",
    help="Pipeline mode: 'fast' or 'precise'.",
)

# --------------------------------------------------------------------
# Logging + Typer app
# --------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Job Skills pipeline CLI.",
    pretty_exceptions_enable=False,
)


@app.command("dq-report")
def dq_report_cmd(
    mode: Mode = typer.Option(
        Mode.precise,
        "--mode",
        "-m",
        help="Config mode (used mainly for choosing cfg.processed_dir).",
    ),
    before: Path = typer.Option(
        ...,
        "--before",
        help="Path to the BEFORE dataset (.xlsx or .csv).",
    ),
    after: Path = typer.Option(
        ...,
        "--after",
        help="Path to the AFTER dataset (.xlsx or .csv).",
    ),
    before_name: str = typer.Option(
        "before",
        "--before-name",
        help="Label used in the report for the before dataset.",
    ),
    after_name: str = typer.Option(
        "after",
        "--after-name",
        help="Label used in the report for the after dataset.",
    ),
) -> None:
    """Generate the reviewer-facing Data Quality report artifacts."""

    cfg = _build_config(mode)
    outputs = dq_report_only(
        cfg,
        before_path=before,
        after_path=after,
        before_name=before_name,
        after_name=after_name,
    )

    typer.echo("DQ report complete.")
    for k, p in outputs.items():
        typer.echo(f"{k}: {p}")

# --------------------------------------------------------------------
# Shared config helper
# --------------------------------------------------------------------

def _build_config(mode: Mode) -> Config:
    """
    Build Config from environment and apply the requested mode.
    """
    cfg = Config.from_env()
    cfg.mode = mode
    return cfg

# --------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------

@app.command("preprocess")
def preprocess_cmd(
    mode: Mode = typer.Option(
        Mode.precise,
        "--mode",
        "-m",
        help="Pipeline mode: 'fast' or 'precise'.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Optional path to input file or directory. If omitted, config defaults are used.",
    ),
) -> None:
    """
    Ingest and preprocess job postings (legacy runner pipeline).

    Uses the same logic as the full pipeline's preprocessing stage,
    but stops before weak labels / FP model / exports.

    Saves a cleaned preprocessed dataset to both XLSX and CSV in cfg.processed_dir.
    """
    # Build config and log what we're doing
    cfg = _build_config(mode)
    logger.info("Running preprocess in %s mode", cfg.mode)
    if input_path is not None:
        logger.info("Using input: %s", input_path)
    else:
        logger.info("Using default input from config")

    # Call the real pipeline preprocess function
    df = preprocess_only(cfg, input_path)

    # Ensure processed directory exists
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    # Drop internal/helper columns (those starting with "_"),
    # but don't error if those columns are missing
    df_out = df.drop(
        columns=[col for col in df.columns if col.startswith("_")],
        errors="ignore",
    )

    # Paths for output files
    xlsx_path = cfg.processed_dir / "preprocessed.xlsx"
    csv_path = cfg.processed_dir / "preprocessed.csv"

    # Save XLSX
    logger.info("Saving preprocessed data to Excel: %s", xlsx_path)
    with pd.ExcelWriter(
        xlsx_path,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}},
    ) as writer:
        df_out.to_excel(writer, index=False, sheet_name="preprocessed")

    # Save CSV
    logger.info("Saving preprocessed data to CSV: %s", csv_path)
    df_out.to_csv(csv_path, index=False)

    logger.info("Preprocessing complete.")
    typer.echo(f"Preprocessed data saved to:\n  {xlsx_path}\n  {csv_path}")

@app.command()
def train_fp(
    mode: Mode = typer.Option(
        Mode.precise,
        "--mode",
        "-m",
        help="Model mode: 'fast' or 'precise'.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Optional path to input file or directory. If omitted, config defaults are used.",
    ),
) -> None:
    """
    Run preprocessing + weak labels + train the calibrated FP model,
    and save it to the models directory (legacy runner pipeline).
    """
    cfg = _build_config(mode)
    train_fp_model_only(cfg, input_path)
    typer.echo("FP model trained and saved.")

@app.command()
def run_all(
    mode: Mode = typer.Option(
        Mode.precise,
        "--mode",
        "-m",
        help="Pipeline mode: 'fast' or 'precise'.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Optional path to input file or directory. If omitted, config defaults are used.",
    ),
    with_skill_extraction: bool = typer.Option(
        False,
        "--with-skill-extraction",
        help="If set, runs the skill extraction on pipeline's exported xlsx after run-all completes.",
    ),
) -> None:
    """
    Full legacy pipeline:

      1. Ingest + preprocess + dedupe
      2. Weak labels
      3. Train FP model + apply probabilities
      4. Threshold + export CSV/XLSX/JSONL/Survey + save model
    """
    cfg = _build_config(mode)
    outputs = run_full_pipeline(cfg, input_path)
    if with_skill_extraction:
        # Prefer the pipeline XLSX output as the input to skill extraction
        skills_out = run_skill_extraction(cfg, input_path=outputs["xlsx"])
        outputs["skills_csv"] = skills_out.out_csv
        outputs["skills_xlsx"] = skills_out.out_xlsx
        outputs["skills_metrics_json"] = skills_out.metrics_json
    typer.echo("Pipeline complete.")
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-bertopic")
def run_bertopic(
    mode: Mode = typer.Option(
        Mode.precise,
        "--mode",
        "-m",
        help="Pipeline mode: 'fast' or 'precise'.",
    ),
    input_jsonl: Optional[Path] = typer.Option(
        None,
        help="Optional override for the BERTopic input .jsonl file "
             "(defaults to clean_jobs_all.jsonl from the pipeline).",
    ),
    domain: Optional[str] = typer.Option(
        None,
        help="Optional domain filter (e.g. 'bio' or 'chemical'). "
             "Domain wiring & output dirs will be handled in a later step.",
    ),
    max_docs: Optional[int] = typer.Option(
        None,
        help="Optional cap on number of documents passed into BERTopic.",
    ),
    len_text_min: Optional[int] = typer.Option(
        None,
        "--len-text-min",
        help=(
            "Override for cfg.bertopic_len_text_min. "
            "Minimum character length of text_for_bertopic to keep."
        ),
    ),
    p_rnd_cutoff: Optional[float] = typer.Option(
        None,
        "--p-rnd-cutoff",
        help=(
            "Override for cfg.bertopic_p_rnd_cutoff. "
            "Minimum p_rnd when using the (label_weak == 1 OR p_rnd >= cutoff) rule."
        ),
    ),
    require_label_weak: Optional[bool] = typer.Option(
        None,
        "--require-label-weak/--no-require-label-weak",
        help=(
            "If set, overrides cfg.bertopic_require_label_weak. "
            "True → require label_weak==1. "
            "False → allow label_weak==1 OR p_rnd>=cutoff."
        ),
    ),
    no_datamapplot: bool = typer.Option(
        False,
        help="If set, skip building the DataMapPlot visualization.",
    ),
    no_bertopic_viz: bool = typer.Option(
        False,
        help="If set, skip building the BERTopic HTML visualization.",
    ),
    no_embeddings_cache: bool = typer.Option(
        False,
        "--no-embeddings-cache",
        help="If set, do not use cached embeddings; recompute them all.",
    )
) -> None:
    """
    Run the BERTopic stage on the preprocessed dataset.

    This will:

    * Read the clean_jobs_all.jsonl file (or a custom input_jsonl).
    * Filter docs for BERTopic using configurable thresholds from Config
      (optionally overridden on the CLI).
    * Train a BERTopic model and save it under models/bertopic_model.
    * Save a topics table and jobs-with-topics table under data/processed/bertopic.
    * Optionally, build visualizations (BERTopic viz + DataMapPlot).
    """
    cfg = _build_config(mode)

    # --- Apply CLI overrides onto the loaded config -------------------------
    if len_text_min is not None:
        cfg.bertopic_len_text_min = len_text_min

    if p_rnd_cutoff is not None:
        cfg.bertopic_p_rnd_cutoff = p_rnd_cutoff

    if require_label_weak is not None:
        cfg.bertopic_require_label_weak = require_label_weak

    if no_embeddings_cache:
        cfg.bertopic_use_embeddings_cache = False
        cfg.bertopic_save_embeddings_cache = False

    outputs = run_bertopic_stage(
        cfg=cfg,
        input_jsonl=input_jsonl,
        domain=domain,
        no_datamapplot=no_datamapplot,
        no_bertopic_viz=no_bertopic_viz,
        max_docs=max_docs,
    )

    typer.echo("BERTopic stage complete.")
    typer.echo(f"Model dir:           {outputs.model_dir}")
    typer.echo(f"Topics CSV:          {outputs.topics_csv}")
    typer.echo(f"Jobs-with-topics:    {outputs.jobs_with_topics_csv}")
    typer.echo(f"HTML dashboard:      {outputs.html_dashboard}")


@app.command("topics-viz")
def topics_viz(
    mode: Mode = typer.Option(
        Mode.precise,
        "--mode",
        "-m",
        help="Pipeline mode: 'fast' or 'precise'.",
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        "-d",
        help=(
            "Optional domain filter for viz-only rebuild (e.g. 'biology', 'chemistry'). "
            "Must match the domain used when running `run-bertopic`."
        ),
    ),
    no_datamapplot: bool = typer.Option(
        False,
        "--no-datamapplot",
        help="Disable DataMapPlot embedding visualization.",
    ),
    no_bertopic_viz: bool = typer.Option(
        False,
        "--no-bertopic-viz",
        help="Disable BERTopic built-in visualizations.",
    ),
) -> None:
    """
    Rebuild the BERTopic HTML dashboard using the *existing* saved model and CSVs.

    This does NOT re-fit BERTopic. It expects (per domain, if given):

      - models/bertopic_model[_<domain>]
      - data/processed/bertopic[/<domain>]/bertopic_topics.csv
      - data/processed/bertopic[/<domain>]/jobs_with_topics.csv
    """    
    from .stages.bertopic_stage_v1 import render_topic_visuals

    cfg = _build_config(mode)
    out_path = render_topic_visuals(
        cfg,
        domain=domain,
        no_datamapplot=no_datamapplot,
        no_bertopic_viz=no_bertopic_viz,
    )

    typer.echo("BERTopic viz-only stage complete.")
    typer.echo(f"HTML dashboard:     {out_path}")


@app.command("select-uncertain")
def select_uncertain(
    n: int = typer.Option(
        5000,
        "--n",
        "-n",
        help="Number of uncertain rows to export for manual labeling.",
    ),
    min_p: float = typer.Option(
        0.20,
        help="Lower bound on p_rnd for candidate rows (exclude obviously non-core).",
    ),
    max_p: float = typer.Option(
        0.90,
        help="Upper bound on p_rnd for candidate rows (exclude already very confident core).",
    ),
):
    """
    Use Cleanlab + the FP model to export N uncertain rows to CSV/JSONL
    under cfg.processed_dir for Label Studio.
    """
    cfg = Config.from_env()
    from .stages.cleanlab_stage import select_uncertain_cleanlab

    csv_path, jsonl_path = select_uncertain_cleanlab(cfg, n=n, min_p=min_p, max_p=max_p)
    typer.echo(f"Uncertain rows written to:\n  CSV:   {csv_path}\n  JSONL: {jsonl_path}")

@app.command("export-uncertain-simple")
def export_uncertain_simple(
    n: int = typer.Option(
        5000,
        "--n",
        "-n",
        help="Number of mid-range probability rows (likely FNs) to export."
    ),
    min_p: float = typer.Option(
        0.30,
        help="Lower bound probability for selecting FN candidates."
    ),
    max_p: float = typer.Option(
        0.85,
        help="Upper bound probability for selecting FN candidates."
    ),
):
    """
    Export the top-N most likely false negatives using ONLY p_rnd from the dropped dataset.
    No Cleanlab. Minimal, reliable, and stable.
    """
    from .export_uncertain_5k_simple import main as export_main
    export_main(n=n, min_p=min_p, max_p=max_p)


@app.command("label-topics-llm")
def label_topics_llm_cmd(
    mode: Mode = typer.Option(
        Mode.precise,
        "--mode",
        "-m",
        help="Pipeline mode: 'fast' or 'precise'.",
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        "-d",
        help="Optional domain used when running `run-bertopic` (e.g., 'biology').",
    ),
    backend: str = typer.Option(
        "openai",
        "--backend",
        "-b",
        help="LLM backend to use for topic labeling (currently: 'openai').",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-M",
        help="LLM model name (e.g. 'gpt-5-mini', 'gpt-5.1'). Defaults from Config.",
    ),
    max_examples: int = typer.Option(
        5,
        "--max-examples",
        help="Max number of example job postings per topic.",
    ),
    max_chars: int = typer.Option(
        400,
        "--max-chars",
        help="Max characters per example text snippet.",
    ),
    temperature: float = typer.Option(
        0.2,
        "--temperature",
        help="Sampling temperature for the LLM.",
    ),
) -> None:
    """
    Use an LLM (OpenAI Responses API by default) to generate human-readable
    labels for BERTopic topics, and write labeled outputs:

      - bertopic_topics_labeled.csv / .xlsx
      - jobs_with_topics_qlik_labeled.csv

    Requires:
      - run-bertopic already completed for the given domain (if any)
      - OPENAI_API_KEY set when using backend='openai'
    """
    cfg = _build_config(mode)

    outputs = label_topics_for_domain(
        cfg=cfg,
        domain=domain,
        backend=backend,
        model=model,
        max_examples=max_examples,
        max_chars=max_chars,
        temperature=temperature,
    )

    typer.echo("LLM topic labeling complete.")
    typer.echo(f"Labeled topics CSV:     {outputs['topics_csv']}")
    typer.echo(f"Labeled topics XLSX:    {outputs['topics_xlsx']}")
    typer.echo(f"Labeled Qlik CSV:       {outputs['qlik_csv']}")


@app.command("extract-skills")
def extract_skills_cmd(
    mode: Mode = typer.Option(
        Mode.precise,
        "--mode",
        "-m",
        help="Config mode (used mainly for choosing cfg.processed_dir).",
    ),
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to the cleaned dataset (.xlsx or .csv). Must contain 'text_for_bertopic' or 'description'.",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Optional output directory. If omitted, defaults to cfg.processed_dir / 'skills'.",
    ),
) -> None:
    """
    Extract Skillsets + Responsibilities from cleaned job descriptions (Option A).
    Produces:
      - jobs_with_skills.xlsx / .csv
      - skills_metrics.json
    """
    cfg = _build_config(mode)
    outputs = run_skill_extraction(cfg, input_path=input_path, out_dir=out_dir)

    typer.echo("Skill extraction complete.")
    typer.echo(f"out_csv: {outputs.out_csv}")
    typer.echo(f"out_xlsx: {outputs.out_xlsx}")
    typer.echo(f"metrics_json: {outputs.metrics_json}")


if __name__ == "__main__":
    app()
