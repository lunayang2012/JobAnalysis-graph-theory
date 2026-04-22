# Job Skills Pipeline

End-to-end pipeline for the Job Skills project:

1. Ingest raw Excel job postings (hourly/yearly or combined).
2. Normalize and clean text (title + description).
3. Filter by domain (biology/chem/lab/etc.).
4. Deduplicate postings.
5. Generate weak labels for R&D vs non-R&D (regex-based).
6. Train a calibrated TF-IDF + Logistic Regression classifier using `make_pipeline`.
7. Threshold by calibrated probability.
8. Export:
   - `data/processed/clean_jobs_all.xlsx`
   - `data/processed/clean_jobs_all.csv`
   - `data/processed/clean_jobs_all.jsonl`
   - `data/processed/survey_luna.xlsx`

The JSONL is used for BERTopic and/or MongoDB.

## Install

```bash
cd job_skills
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
