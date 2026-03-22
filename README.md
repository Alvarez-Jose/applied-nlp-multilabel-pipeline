# Palestine Violence Archive — Lab Pipeline

A research pipeline for documenting, classifying, and archiving incidents of violence against Palestinians in the West Bank. Scraped news articles are automatically labeled with event type categories using a trained multilabel classifier, then routed to research assistants for human review. Corrected labels feed back into model retraining.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [RA Review Workflow](#ra-review-workflow)
- [Label Definitions](#label-definitions)
- [Model Details](#model-details)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Retraining](#retraining)

---

## Project Overview

The pipeline operates as a continuous loop:

```
WAFA news site
    -> scrape articles
    -> push to Google Sheets (Reports tab)
    -> export to CSV
    -> run multilabel classifier (baseline or DeBERTa)
    -> flag uncertain predictions
    -> research assistants review in Google Sheets
    -> merge corrections into training data
    -> retrain model
    -> repeat
```

**What gets classified:** Each article is assigned zero or more of 10 binary event-type labels (raid, arrest, physical assault, etc.) according to a structured codebook. The codebook defines precise inclusion/exclusion criteria and is stored in `config/label_codebook.yaml`.

**Who uses this:** A small research team. The pipeline scripts run locally. Research assistants access predictions through Google Sheets for review. The Streamlit dashboard provides a status overview.

---

## Repository Structure

```
palestine-violence-archive/
|
|- run_pipeline.py                         # Master orchestrator -- run this first
|- requirements.txt                        # Python dependencies
|- config/
|   |- label_codebook.yaml                # Codebook: 10 label definitions + constraints
|
|- data_pipeline/
|   |- scraping/
|   |   |- ingest_reports.py              # Crawl WAFA -> parse articles -> Sheets
|   |   |- crawler.py                     # Async URL crawler (robots.txt aware)
|   |   |- scraper.py                     # Article content extractor
|   |- database/
|   |   |- sheets_interface.py            # Google Sheets read/write (gspread)
|   |   |- models.py                      # Report and Incident dataclasses
|   |- cleaning/
|   |   |- deduplicate_events.py          # Event-level deduplication
|   |- export_reports_to_incoming.py      # Sheets Reports tab -> batch CSV
|   |- export_sheets_to_csv.py            # Sheets Incidents tab -> training CSV
|   |- normalize_raw.py                   # Normalize raw RA-coded CSVs (2023-2025)
|   |- convert_oppression_db_to_incidents.py  # Generate rich descriptions from fields
|   |- codebook.py                        # Controlled vocabularies (governorates, etc.)
|
|- modeling/
|   |- predict_and_prepare_review.py      # Run inference + build review sheets
|   |- merge_reviewed_labels.py           # Merge RA corrections into training data
|   |- make_splits.py                     # Train/val/test split
|   |- training/
|   |   |- export_training_data.py        # Consolidate all sources into training CSV
|   |   |- train_baseline_multilabel.py   # TF-IDF + LogisticRegression training
|   |   |- train_transformer_multilabel.py # DeBERTa v3 fine-tuning
|   |- saved_models/
|       |- baseline_rank_based_fixed/     # Baseline model artifacts (pkl + meta)
|       |- deberta_v3_multilabel/         # DeBERTa artifacts (meta JSON; weights on HF Hub)
|
|- streamlit_app/
|   |- app.py                             # Dashboard UI
|   |- requirements.txt                   # Streamlit-only deps (for Cloud deployment)
|
|- utilities/
|   |- date_utils.py                      # Date parsing
|   |- text_utils.py                      # Text cleaning and extraction
|   |- validators.py                      # Field validators
|   |- incident_id.py                     # ID formatting (GOV-YYYYMMDD-NN)
|   |- timezone_conversion.py             # Palestine timezone (Asia/Hebron)
|
|- data/
    |- raw/                               # Original RA-coded Excel files (2023-2025)
    |- processed/                         # Normalized CSVs
    |- incoming/                          # Batch CSVs awaiting model prediction
    |- review_exports/                    # Model output: review sheets + predictions
    |- reviewed/                          # RA-corrected files ready for merge
    |- training/                          # Consolidated training datasets
```

---

## Setup

### 1. Python environment

```bash
pip install -r requirements.txt

# Playwright (for JavaScript-heavy scraping, if needed):
playwright install chromium
```

Tested on Python 3.11. The conda environment used in development is `ml_nlp_env311`.

### 2. Google Sheets credentials

The pipeline reads and writes to a shared Google Spreadsheet. Authentication uses a service account.

- Place `service_account.json` in the project root (this file is gitignored)
- The spreadsheet ID is hardcoded in `data_pipeline/database/sheets_interface.py`
- The service account must have Editor access to the spreadsheet

The spreadsheet has three tabs:
- **Reports** -- scraped articles (written by the scraper)
- **Incidents** -- manually coded events (written by RAs)
- **Review** -- model predictions for RA review (written by the pipeline)

### 3. DeBERTa model weights

The DeBERTa model weights (~738 MB) are hosted on HuggingFace Hub at `jalva182/palestine-violence-deberta` (private repository). They are downloaded automatically on first use.

```bash
huggingface-cli login
# Enter token when prompted
```

Or set the environment variable `HF_TOKEN` before running.

The baseline TF-IDF model is stored locally in `modeling/saved_models/baseline_rank_based_fixed/` and tracked via Git LFS.

---

## Running the Pipeline

All pipeline stages are orchestrated through `run_pipeline.py`.

### Normal daily run (scrape + export + predict)

```bash
python run_pipeline.py
```

This runs three stages in sequence:
1. Scrape WAFA for new articles, push to Google Sheets Reports tab
2. Export new reports from Sheets to `data/incoming/batch_{timestamp}.csv`
3. Run the baseline classifier, write review sheets to `data/review_exports/`

### Common options

```bash
# Use DeBERTa instead of baseline:
python run_pipeline.py --model deberta

# Skip scraping (use what is already in Sheets):
python run_pipeline.py --skip-ingest

# Push predictions directly to the Google Sheets Review tab:
python run_pipeline.py --push-to-sheets

# Re-export all reports, ignoring the already-exported tracker:
python run_pipeline.py --export-all

# Preview each step without writing any files:
python run_pipeline.py --dry-run

# Only retrain (after human review and merge are done):
python run_pipeline.py --retrain-only
```

### Running individual scripts

```bash
# Scrape only:
python data_pipeline/scraping/ingest_reports.py

# Export Reports tab to CSV only:
python data_pipeline/export_reports_to_incoming.py

# Run model on an existing batch CSV:
python modeling/predict_and_prepare_review.py \
    --input data/incoming/batch_20250321_120000.csv \
    --model baseline

# Merge reviewed labels from local file:
python modeling/merge_reviewed_labels.py \
    --reviewed data/reviewed/reviewed_20250321.xlsx \
    --predictions data/review_exports/predictions_20250321_120000.csv

# Merge reviewed labels from Google Sheets:
python modeling/merge_reviewed_labels.py \
    --from-sheets \
    --predictions data/review_exports/predictions_20250321_120000.csv
```

---

## RA Review Workflow

After the pipeline runs, research assistants review model predictions in Google Sheets.

**Step 1 -- Open the Review tab**

Navigate to the shared Google Spreadsheet and open the **Review** tab. New rows will have been pushed by the pipeline.

**Step 2 -- Fill in human labels**

For each row where `needs_review = TRUE`, fill in the `human_{label}` columns:
- `1` if the event type applies
- `0` if it does not
- Leave blank to accept the model prediction

Pay attention to the `codebook_conflict` column -- it flags label combinations that may be inconsistent with the codebook.

**Step 3 -- Mark as reviewed**

Set `review_status = reviewed` when a row is complete. Add any notes to `reviewer_notes`.

**Step 4 -- Merge labels back into training data**

Once a batch is reviewed, a lab admin runs:

```bash
python modeling/merge_reviewed_labels.py \
    --from-sheets \
    --predictions data/review_exports/predictions_{timestamp}.csv
```

This pulls all `review_status = reviewed` rows from Sheets and merges human labels into the training dataset.

---

## Label Definitions

The classifier predicts 10 binary labels per incident. Full definitions, inclusion/exclusion criteria, and examples are in `config/label_codebook.yaml`.

| Label | Description | Train prevalence |
|---|---|---|
| `raid_y` | Organized military or police entry into a Palestinian area | 54.5% |
| `arrest_detention_y` | Apprehension or detention of Palestinians | 38.3% |
| `physical_assault_y` | Direct use of physical force or violence | 38.4% |
| `harm_to_property_y` | Damage or destruction of Palestinian-owned property | 29.9% |
| `dispossession_y` | Seizure or theft of Palestinian land, livestock, or assets | 7.7% |
| `religious_encroachment_y` | Interference with religious sites or observances | 1.1% |
| `restriction_of_freedoms_y` | Restrictions on movement, access, or daily activity | 14.4% |
| `coercive_actions_y` | Intimidation, threats, or psychological coercion | 45.5% |
| `protest_y` | Organized Palestinian protests or civil resistance | 3.6% |
| `multi_community_incident_y` | Incident spanning multiple communities simultaneously | 8.8% |

Labels marked with low prevalence (`religious_encroachment_y`, `protest_y`) are always flagged for human review when predicted positive, regardless of model confidence.

---

## Model Details

### Baseline (TF-IDF + Logistic Regression)

- Input: TF-IDF vectors (unigrams to trigrams, 10k features) built from description + structured metadata fields
- One binary LogisticRegression classifier per label (MultiOutputClassifier)
- Thresholding: rank-based -- the top-k documents by predicted probability are marked positive, where k is calibrated to match the training prevalence of each label
- Artifacts: `modeling/saved_models/baseline_rank_based_fixed/`

### DeBERTa v3

- Backbone: `microsoft/deberta-v3-base` (fine-tuned)
- Input: same structured text format as baseline (`[AREA] ... [GOV] ... [DESC] ...`)
- Per-label sigmoid thresholds tuned on the validation split
- Weights hosted privately on HuggingFace Hub: `jalva182/palestine-violence-deberta`
- Meta JSON stored locally: `modeling/saved_models/deberta_v3_multilabel/deberta_v3_meta.json`

### Uncertainty flagging

A prediction is flagged for human review if the model's confidence score is within `0.10` of the decision threshold. This margin can be adjusted with `--confidence-band` when calling `predict_and_prepare_review.py` directly.

---

## Streamlit Dashboard

A lightweight dashboard for monitoring pipeline status and triggering runs.

```bash
streamlit run streamlit_app/app.py
```

Pages:
- **Dashboard** -- batch counts, flagged doc counts, baseline vs DeBERTa metrics
- **Run Pipeline** -- form to trigger any pipeline configuration; output streams live
- **Review Access** -- link to Google Sheets Review tab; merge-from-Sheets button
- **Retrain** -- per-label metrics table; checkpoint save/restore; retrain button

### Version checkpoints

The Retrain page automatically saves a checkpoint of the current model artifacts before every retrain or merge operation. Checkpoints are stored in `modeling/backups/` (gitignored) and can be restored with one click if a model is accidentally overwritten.

---

## Retraining

Retraining should happen after enough reviewed batches have accumulated (recommended: at least 200 new reviewed incidents).

```bash
# 1. Merge all reviewed batches (if not done already):
python modeling/merge_reviewed_labels.py --from-sheets \
    --predictions data/review_exports/predictions_{timestamp}.csv

# 2. Rebuild consolidated training dataset:
python modeling/training/export_training_data.py

# 3. Retrain baseline:
python modeling/training/train_baseline_multilabel.py

# 4. Evaluate: compare new baseline_meta.json against the previous checkpoint

# 5. If metrics hold, retrain DeBERTa:
python modeling/training/train_transformer_multilabel.py
```

Or use the single-command shortcut:

```bash
python run_pipeline.py --retrain-only
```

---

## Data Sources

Currently ingesting:
- **WAFA** (Wafa News Agency) -- English-language articles filtered to West Bank region

Planned (not yet implemented):
- B'Tselem
- OCHA

---

## Notes for Developers

- All scripts resolve paths relative to `PROJECT_ROOT = Path(__file__).resolve().parent` -- do not rely on the current working directory
- Google Sheets ID is hardcoded in `data_pipeline/database/sheets_interface.py`
- `service_account.json` must be in the project root and is gitignored
- Pipeline logs are written to `logs/pipeline_{timestamp}.log` in addition to stdout
- Large model files use Git LFS; DeBERTa weights are excluded from git entirely and hosted on HF Hub
