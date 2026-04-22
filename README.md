# Job Skills

An end-to-end data pipeline for processing job postings, identifying science-domain opportunities, and classifying roles as **R&D** or **non-R&D**.

The project is built to transform messy raw job-posting exports into a structured, analysis-ready dataset that can be used for downstream tasks such as topic modeling, exploratory analysis, reporting, and storage in systems like MongoDB.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why This Project Exists](#why-this-project-exists)
- [What the Project Does](#what-the-project-does)
- [Pipeline Summary](#pipeline-summary)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Project Outputs](#project-outputs)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Input Data Expectations](#input-data-expectations)
- [Modeling Approach](#modeling-approach)
- [Use Cases](#use-cases)
- [Future Improvements](#future-improvements)

---

## Project Overview

**Job Skills** is a job-posting data pipeline designed to solve a practical problem:

Raw job listings are often noisy, duplicated, inconsistently formatted, and difficult to analyze at scale.

This project takes those raw job postings and turns them into a **clean, filtered, labeled, and exportable dataset** focused on scientific and laboratory-related job domains. It also adds a machine learning layer that helps distinguish **R&D-oriented roles** from other types of job postings.

The final output is a reusable dataset that can support:
- research and labor-market analysis
- topic modeling with BERTopic
- job trend analysis
- structured storage in MongoDB
- domain-specific recruitment analytics

---

## Why This Project Exists

Job postings contain valuable information about:
- required technical skills
- domain demand
- role types
- research vs non-research work
- industry hiring patterns

However, raw exports from job platforms are usually not ready for analysis. Common issues include:
- duplicated postings
- inconsistent formatting
- mixed salary formats
- irrelevant job domains
- long, noisy descriptions
- no reliable labels for classification tasks

This project was created to build a repeatable workflow that:
1. **cleans and standardizes** raw job data,
2. **filters it to relevant scientific domains**,
3. **deduplicates postings**,
4. **creates weak labels** for role-type classification,
5. **trains a calibrated classifier**, and
6. **exports clean datasets** for analytics and NLP workflows.

---

## What the Project Does

At a high level, the pipeline performs the following steps:

1. Reads raw Excel job-posting files
2. Combines and cleans job titles and descriptions
3. Filters postings to science-related domains such as biology, chemistry, and laboratory work
4. Removes duplicate job postings
5. Creates weak labels using rule-based regex matching
6. Trains a TF-IDF + Logistic Regression classifier
7. Calibrates prediction probabilities
8. Applies a decision threshold to identify likely R&D jobs
9. Exports cleaned data into Excel, CSV, and JSONL formats

---

## Pipeline Summary

### End-to-end pipeline

1. **Ingest raw Excel job postings**
   - accepts hourly, yearly, or combined source files

2. **Normalize and clean text**
   - combines job title and description
   - standardizes spacing, formatting, and missing values

3. **Filter by domain**
   - keeps roles related to biology, chemistry, lab work, and similar scientific areas

4. **Deduplicate postings**
   - removes repeated jobs that may appear across source files

5. **Generate weak labels**
   - uses regex-based heuristics to label postings as **R&D** or **non-R&D**

6. **Train classifier**
   - builds a calibrated **TF-IDF + Logistic Regression** model using `make_pipeline`

7. **Threshold by probability**
   - converts calibrated probabilities into final labels using a selected threshold

8. **Export final datasets**
   - writes processed outputs to Excel, CSV, JSONL, and survey-ready files

---

## How the Pipeline Works

## 1. Data Ingestion

The pipeline starts with raw job postings stored in Excel files. These files may contain:
- hourly job data
- yearly salary job data
- combined datasets

At this stage, the objective is to load the source data into a consistent format for downstream processing.

### Goal
Create a unified starting dataset from multiple raw sources.

---

## 2. Text Normalization and Cleaning

The most important text fields in a job posting are typically:
- job title
- job description

These fields are merged and cleaned so the pipeline has a consistent text representation of each posting.

Cleaning may include:
- lowercasing
- trimming whitespace
- removing malformed or empty text
- normalizing punctuation or spacing
- combining title and description into a single analysis field

### Goal
Produce standardized text that can be used for filtering, labeling, and classification.

---

## 3. Domain Filtering

Not all job postings in raw exports are useful for this project. The pipeline narrows the dataset to job postings relevant to scientific and laboratory contexts.

Example target domains include:
- biology
- chemistry
- laboratory roles
- life sciences
- research-support environments

This domain filtering step helps keep the pipeline focused on relevant job families instead of training on unrelated postings.

### Goal
Retain only the jobs that match the intended scientific/lab-oriented scope of the project.

---

## 4. Deduplication

Raw job exports often contain duplicate records due to:
- repeated scraping
- reposted listings
- platform duplication
- overlapping source files

Deduplication improves dataset quality and prevents the model from being biased by repeated postings.

### Goal
Ensure each job posting is represented once in the processed dataset.

---

## 5. Weak Label Generation

One of the core challenges in this project is identifying whether a job is **R&D-focused** or **not**.

Since manually labeling every posting is time-consuming, the pipeline first creates **weak labels** using regex-based heuristics.

Examples of signals that may be used:
- research-oriented terms
- development-oriented terms
- lab experimentation language
- scientist / associate / research technician style wording
- non-research operational or support terms for contrast

These labels are not perfect, but they provide a useful starting point for supervised modeling.

### Goal
Create a scalable approximation of R&D vs non-R&D labels without manual annotation of the entire dataset.

---

## 6. Model Training

After weak labeling, the project trains a machine learning classifier to generalize beyond exact regex matches.

### Model design
- **TF-IDF vectorization** for converting job text into numerical features
- **Logistic Regression** for binary classification
- **`make_pipeline`** for a simple, reproducible modeling workflow
- **probability calibration** so the model outputs more reliable confidence scores

### Why this approach?
This combination is:
- interpretable
- efficient
- strong for sparse text classification tasks
- easy to reproduce and tune

### Goal
Move from fragile keyword rules to a data-driven classifier that can identify R&D roles more robustly.

---

## 7. Probability Thresholding

The model produces calibrated probabilities, not just hard labels.

A threshold is then applied to determine whether a posting should be treated as R&D or non-R&D.

This makes the system more flexible:
- higher thresholds can improve precision
- lower thresholds can improve recall

### Goal
Control how strict the final classification should be based on project needs.

---

## 8. Data Export

Once the jobs are cleaned, filtered, deduplicated, and classified, the pipeline exports the final dataset into multiple formats.

### Output files

- `data/processed/clean_jobs_all.xlsx`
- `data/processed/clean_jobs_all.csv`
- `data/processed/clean_jobs_all.jsonl`
- `data/processed/survey_luna.xlsx`

### Why multiple formats?

#### Excel
Useful for:
- manual review
- stakeholder sharing
- quick inspection

#### CSV
Useful for:
- lightweight analytics
- interoperability with data tools
- reproducible workflows

#### JSONL
Useful for:
- BERTopic pipelines
- MongoDB ingestion
- NLP workflows and document-based systems

#### Survey Excel
Useful for:
- curated review
- downstream manual validation
- survey or annotation workflows

---

## Project Outputs

The processed outputs are the core deliverables of this repository.

### `clean_jobs_all.xlsx`
Human-readable master dataset for review and sharing.

### `clean_jobs_all.csv`
Analysis-friendly flat file for scripts, notebooks, and data tools.

### `clean_jobs_all.jsonl`
Document-style format used in BERTopic and MongoDB workflows.

### `survey_luna.xlsx`
A curated export intended for survey-based review or validation tasks.

---

## Project Structure

A suggested interpretation of the repository structure is shown below. Update this section if your actual file layout differs.

```text
Job_Skills/
│
├── data/
│   ├── raw/
│   │   └── ... raw Excel job posting files
│   └── processed/
│       ├── clean_jobs_all.xlsx
│       ├── clean_jobs_all.csv
│       ├── clean_jobs_all.jsonl
│       └── survey_luna.xlsx
│
├── notebooks/
│   └── ... exploratory notebooks (if any)
│
├── src/
│   └── ... pipeline, cleaning, labeling, and modeling code
│
├── requirements.txt
├── README.md
└── ...
