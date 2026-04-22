# JobAnalysis-graph-theory

This repository contains the data, pipeline code, notebooks, and generated outputs used for processing and analyzing job-posting data in the **job_skills** project. The workflow is designed to take raw job-posting spreadsheets, apply preprocessing and filtering steps, generate intermediate and final datasets, and support downstream topic modeling, labeling, reporting, and exploratory analysis.

## Repository purpose

The repository supports a structured workflow for studying job postings and extracting skill- and topic-level information from them. It combines:

- raw and processed job-posting data,
- a modular Python pipeline,
- notebook-based execution and inspection,
- topic-modeling outputs, and
- data-quality reporting artifacts.

This README is intentionally written to match the current repository structure shown in the project workspace and is suitable for citation or reference alongside a research paper.

## Repository structure

```text
JobAnalysis-graph-theory/
└── job_skills/
    ├── data/
    │   ├── inputs/
    │   │   ├── Data_File_3.xlsx
    │   │   └── Main_Data_File.xlsx
    │   ├── interim/
    │   └── processed/
    │       ├── bertopic\ bio/
    │       │   ├── bertopic_dashboard.html
    │       │   ├── bertopic_datamapplot.png
    │       │   ├── bertopic_metadata.json
    │       │   ├── bertopic_topics_labeled.csv
    │       │   ├── bertopic_topics_labeled.xlsx
    │       │   ├── bertopic_topics.csv
    │       │   ├── bertopic_topics.xlsx
    │       │   ├── doc_embeddings_meta.json
    │       │   ├── doc_embeddings.npy
    │       │   ├── jobs_with_topics_qlik_labeled.csv
    │       │   ├── jobs_with_topics_qlik.csv
    │       │   ├── jobs_with_topics.csv
    │       │   └── jobs_with_topics.xlsx
    │       ├── data_quality_report/
    │       │   ├── tables/
    │       │   ├── data_quality_metrics.csv
    │       │   ├── data_quality_metrics.xlsx
    │       │   ├── process_efficiency_metrics.csv
    │       │   └── process_efficiency_metrics.xlsx
    │       └── CleanJobs_Final_v2.1.xlsx
    ├── documentation/
    ├── models/
    ├── notebooks/
    ├── pipeline/
    │   ├── stages/
    │   │   ├── __init__.py
    │   │   ├── bertopic_stage_v1.py
    │   │   ├── cleanlab_stage.py
    │   │   ├── data_quality_report.py
    │   │   ├── export.py
    │   │   ├── filtering.py
    │   │   ├── fp_model.py
    │   │   ├── ingest.py
    │   │   ├── old_data_quality_report.py
    │   │   ├── preprocess.py
    │   │   ├── skill_extraction_stage.py
    │   │   ├── topic_labeler_llm.py
    │   │   └── weak_labels.py
    │   ├── __init__.py
    │   ├── cli.py
    │   ├── config.py
    │   ├── json_utils.py
    │   ├── runner.py
    │   ├── schemas.py
    │   ├── requirements.txt
    │   └── requirements_current.txt
    ├── dq_recurrence_history.json
    ├── README.md
    ├── run_bertopic.ipynb
    ├── run_pipeline.ipynb
    └── US_Marketing_Jobs_Recoded.xlsx
```

## Main components

### 1. Data directory

The `data/` directory contains the input, intermediate, and processed files used throughout the workflow.

- `data/inputs/` stores the source Excel files used as pipeline inputs.
- `data/interim/` is intended for temporary or transitional datasets produced between major processing steps.
- `data/processed/` stores final or analysis-ready outputs.

Within `data/processed/`, two output groups are visible:

#### BERTopic outputs
The `bertopic\ bio/` folder contains topic-modeling artifacts, including:

- BERTopic dashboard output (`bertopic_dashboard.html`)
- topic maps and visual assets (`bertopic_datamapplot.png`)
- topic metadata (`bertopic_metadata.json`)
- labeled and unlabeled topic tables (`bertopic_topics*.csv`, `bertopic_topics*.xlsx`)
- document embedding outputs (`doc_embeddings.npy`, `doc_embeddings_meta.json`)
- job-level topic assignment files (`jobs_with_topics*.csv`, `jobs_with_topics*.xlsx`)

These files indicate that the repository supports downstream topic discovery and topic labeling on processed job-posting text.

#### Data-quality outputs
The `data_quality_report/` folder contains report tables and exported metrics such as:

- `data_quality_metrics.csv`
- `data_quality_metrics.xlsx`
- `process_efficiency_metrics.csv`
- `process_efficiency_metrics.xlsx`

These outputs suggest that the workflow includes explicit monitoring of dataset quality and processing performance.

#### Final processed dataset
- `CleanJobs_Final_v2.1.xlsx` appears to be a consolidated processed output intended for inspection, analysis, or downstream use.

### 2. Pipeline package

The `pipeline/` directory contains the operational code for the project. Its structure suggests a modular, stage-based pipeline design.

#### Core pipeline files
- `cli.py` provides a command-line interface for running the workflow.
- `runner.py` coordinates execution across stages.
- `config.py` defines pipeline configuration.
- `schemas.py` defines shared data structures or validation schemas.
- `json_utils.py` provides utility functions for JSON handling.

#### Pipeline stages
The `pipeline/stages/` folder contains the main stepwise processing logic:

- `ingest.py` handles loading source data.
- `preprocess.py` handles preprocessing and normalization.
- `filtering.py` applies filtering logic to the records.
- `weak_labels.py` generates weak supervision or heuristic labels.
- `fp_model.py` appears to support modeling or prediction-related workflow steps.
- `cleanlab_stage.py` suggests a label-quality or validation stage.
- `skill_extraction_stage.py` supports extraction of skill-related information from job text.
- `bertopic_stage_v1.py` runs BERTopic-based topic modeling.
- `topic_labeler_llm.py` appears to support topic labeling with an LLM-based step.
- `data_quality_report.py` generates the data-quality report.
- `export.py` writes outputs for downstream use.
- `old_data_quality_report.py` is retained as an earlier version of the reporting stage.

Overall, the file organization indicates an end-to-end workflow from ingestion to processed exports and topic-analysis artifacts.

### 3. Notebooks

The repository includes notebook-based execution and analysis files:

- `run_pipeline.ipynb`
- `run_bertopic.ipynb`

These notebooks likely serve as reproducible entry points for running the main pipeline and BERTopic workflow, as well as for inspection and iterative experimentation.

### 4. Models and documentation

- `models/` is reserved for trained models or model-related artifacts.
- `documentation/` stores supporting documentation for the project.
- `dq_recurrence_history.json` appears to capture historical information related to recurring data-quality checks or runs.

## Workflow summary

Based on the current structure, the workflow can be understood as follows:

1. Raw Excel job-posting files are placed in `data/inputs/`.
2. The pipeline ingests and preprocesses the records.
3. Filtering and weak-labeling stages refine the data for downstream tasks.
4. Additional stages support skill extraction, modeling, topic generation, and topic labeling.
5. Data-quality reporting is generated and exported to `data/processed/data_quality_report/`.
6. Topic-modeling outputs are exported to `data/processed/bertopic\ bio/`.
7. Final cleaned datasets are retained in processed form for analysis and reporting.

## Typical outputs

The repository currently includes the following visible output types:

- cleaned and consolidated Excel datasets,
- BERTopic topic tables,
- job-to-topic mapping files,
- embedding artifacts,
- topic-visualization files,
- data-quality metrics, and
- process-efficiency reports.

These outputs support both methodological transparency and downstream analytical use.

## Reproducibility notes

The repository contains both:
- `requirements.txt`
- `requirements_current.txt`

inside the `pipeline/` folder, which may be used to document the Python environment for execution. For reproducible runs, users should create a virtual environment and install the required dependencies before executing pipeline modules or notebooks.

A typical setup would follow the form:

```bash
cd job_skills/pipeline
pip install -r requirements.txt
```

## Notes for readers

This repository is organized as a working research and analysis codebase. It contains both executable pipeline components and generated artifacts. The structure reflects an applied workflow rather than a minimal software package layout, which is appropriate for a paper-linked repository that needs to preserve data-processing context, intermediate outputs, and analysis deliverables.

## Citation-oriented summary

In summary, this repository provides the code and generated artifacts for a job-posting analysis workflow centered on:

- data ingestion and preprocessing,
- filtering and weak labeling,
- skill extraction,
- topic modeling with BERTopic,
- topic labeling,
- data-quality reporting, and
- export of processed analytical outputs.

