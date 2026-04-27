# Human-in-the-Loop Multilabel NLP Pipeline

> A research pipeline for multilabel text classification with margin-based uncertainty routing to human reviewers, end-to-end Google Sheets integration, and an active-retraining loop. Applied in production to a private human-rights research project.

> ⚠️ **Restricted-use repository.** Source-available for review. See [`NOTICE.md`](NOTICE.md) — no license is granted; all rights reserved.
>
> 📦 **Data not included.** This repository ships the engineering only. The research dataset and trained model weights are not distributed here. See [Data not included](#data-not-included) below.

---

## What this is

The engineering side of an end-to-end multilabel text-classification system. Articles are scraped from a configured news source, written to Google Sheets, classified by either a TF-IDF baseline or a fine-tuned DeBERTa v3 multilabel head, and routed to research assistants for review. Corrected labels feed back into training and confirmed records promote to a publication-layer database. A Streamlit dashboard fronts the workflow.

The classification pattern, the human-in-the-loop design, and the Sheets-backed feedback loop are the artifacts I'm sharing — not the underlying dataset.

---

## Architecture

```
1. Scrape articles from configured news source
       |
       v
2. Push to Google Sheets (Reports tab)
       |
       v
3. Export to local batch CSV (data/incoming/)
       |
       v
4. Run multilabel classifier (TF-IDF baseline or DeBERTa v3)
       |
       v
5. Push predictions to Google Sheets (Review tab)
       |
       v
6. Research assistants review and correct labels in Sheets
       |
       v
7. Finalize confirmed records to Incidents tab (publication layer)
       |
       +----> retrain model and repeat
```

**What gets classified:** Each article is assigned zero or more of 10 binary event-type labels according to a structured codebook. The codebook defines precise inclusion/exclusion criteria and is stored in `config/label_codebook.yaml`.

**Google Sheets layout:**
- **Reports** — scraped articles written by the ingestion pipeline
- **Review** — model predictions awaiting RA review (69 columns: metadata, predictions, confidence scores, uncertainty flags, human override columns, workflow columns)
- **Incidents** — finalized, clean records promoted from Review after human confirmation; this is the publication-layer database

---

## Data not included

The research dataset that this pipeline classifies — and the model weights derived from it — are **deliberately excluded** from this public repository.

**Why:** The dataset is curated research data on a sensitive subject, compiled by a small team of research assistants under an internal codebook. Public redistribution of the dataset itself is out of scope for this portfolio repository, and the trained model weights derived from it could leak training-set content through inference. The engineering — pipeline architecture, classifier code, review UI, retraining loop — is the contribution being shared here.

**What this means in practice:**
- `data/raw/`, `data/processed/`, `data/review_exports/`, `data/training/`, and `modeling/training/data/` are gitignored. The directory structure exists (`.gitkeep` files); the contents do not.
- Trained baseline model weights (`modeling/saved_models/baseline_*/*.pkl`, `*.joblib`) are gitignored — they're derived from the private dataset.
- The fine-tuned DeBERTa v3 weights are not in the repo either; see [Model weights](#model-weights) below.

**To run this pipeline on your own data:** drop CSVs into `data/incoming/` matching the schema described in `data_pipeline/database/models.py` (Report dataclass), provide your own Google Sheets and a service account, and the pipeline will run against your data. The codebook in `config/label_codebook.yaml` is project-specific and would need to be adapted to your label set.

**To request research access** to the original dataset and trained weights: email jalva182@ucsc.edu. Access is by separate agreement only.

---

## Repository structure

```
applied-nlp-multilabel-pipeline/
|
|- run_pipeline.py                        # Master orchestrator
|- requirements.txt                        # Core Python dependencies (no GPU libs)
|- requirements-gpu.txt                    # DeBERTa / GPU dependencies (local use only)
|- NOTICE.md                               # All-rights-reserved notice
|
|- config/
|   |- label_codebook.yaml                 # Codebook: 10 label definitions + constraints
|
|- data_pipeline/
|   |- scraping/
|   |   |- ingest_reports.py               # Crawl source -> parse articles -> Sheets
|   |   |- crawler.py                      # Async URL crawler (robots.txt aware)
|   |   |- scraper.py                      # Article content extractor
|   |- database/
|   |   |- sheets_interface.py             # Google Sheets read/write (gspread)
|   |   |- models.py                       # Report and Incident dataclasses
|   |- cleaning/
|   |   |- deduplicate_events.py           # Event-level deduplication
|   |- export_reports_to_incoming.py       # Sheets Reports tab -> batch CSV
|   |- export_sheets_to_csv.py             # Sheets Incidents tab -> training CSV
|   |- export_for_analysis.py              # Export to SPSS (.sav) and R (.rds) formats
|   |- finalize_incidents.py               # Promote reviewed rows -> Incidents tab
|   |- normalize_raw.py                    # Normalize raw RA-coded CSVs
|   |- codebook.py                         # Controlled vocabularies (governorates, etc.)
|
|- modeling/
|   |- predict_and_prepare_review.py       # Run inference + build review sheets
|   |- merge_reviewed_labels.py            # Merge RA corrections into training data
|   |- make_splits.py                      # Train/val/test split
|   |- training/
|   |   |- train_baseline_multilabel.py    # TF-IDF + LogisticRegression training
|   |   |- train_transformer_multilabel.py # DeBERTa v3 fine-tuning
|   |- saved_models/                       # Model artifacts (weights gitignored, see "Data not included")
|
|- analysis/
|   |- descriptive_analysis.R              # R script for descriptive stats
|   |- compute_irr.py                      # Inter-rater reliability (Cohen's kappa)
|
|- streamlit_app/
|   |- app.py                              # Dashboard UI (deployed on Streamlit Cloud)
|
|- docs/
|   |- generate_dashboard_guide.py         # Regenerate the PDF user guide
|   |- Dashboard_User_Guide.pdf            # Current PDF guide (auto-generated)
|
|- utilities/
|   |- date_utils.py
|   |- text_utils.py
|   |- validators.py
|   |- incident_id.py                      # ID formatting (GOV-YYYYMMDD-NN)
|   |- timezone_conversion.py
|
|- check_sheets_structure.py               # Diagnostic: print Sheets tab headers + row counts
|
|- data/                                   # Folder structure only — contents gitignored
    |- incoming/.gitkeep                   # Batch CSVs awaiting model prediction
    |- review_exports/.gitkeep             # Model output: review sheets + predictions
    |- reviewed/.gitkeep                   # RA-corrected files ready for merge
    |- training/.gitkeep                   # Consolidated training datasets
```

---

## Setup

### 1. Python environment

```bash
# Core dependencies (scraping, Sheets, dashboard, baseline model):
pip install -r requirements.txt

# DeBERTa and GPU support (local inference only; not needed for Streamlit Cloud):
pip install -r requirements-gpu.txt

# For CUDA (GPU acceleration):
pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu121
```

Tested on Python 3.11.

### 2. Google Sheets credentials

The pipeline reads and writes to a Google Spreadsheet using a service account.

**Local use:**
- Place `service_account.json` in the project root (gitignored)
- The spreadsheet ID is configured in `data_pipeline/database/sheets_interface.py` — replace with your own
- The service account must have Editor access to the spreadsheet
- Alternatively, set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json`

**Streamlit Cloud deployment:**
- Do not commit `service_account.json` — it is gitignored
- Instead, paste the service account JSON into Streamlit Cloud Secrets in TOML format:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
```

The dashboard writes these credentials to `service_account.json` on startup automatically.

### 3. Bring your own data

Per the [Data not included](#data-not-included) section above, this repository does not ship the original dataset. Place your own CSVs under `data/incoming/` matching the `Report` dataclass schema (`data_pipeline/database/models.py`). Adjust `config/label_codebook.yaml` for your label set.

---

## Running the pipeline

All pipeline stages are orchestrated through `run_pipeline.py`.

### Normal run (scrape + export + predict)

```bash
python run_pipeline.py
```

Three stages in sequence:
1. Scrape configured source for new articles, push to Google Sheets Reports tab
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

---

## Human-in-the-loop review workflow

After the pipeline runs, research assistants review model predictions in Google Sheets.

1. Open the **Review** tab. New rows are added by the pipeline.
2. For each row, fill in `human_{label}` columns: `1` for applies, `0` for does not, blank to accept the model prediction.
3. Rows where `needs_review = TRUE` have low margin or a rare label predicted — prioritize these. The `codebook_conflict` column flags label combinations inconsistent with the codebook.
4. Set `review_status = reviewed` when done. Add notes to `reviewer_notes`.
5. A lab admin merges:
   ```bash
   python modeling/merge_reviewed_labels.py --from-sheets \
       --predictions data/review_exports/predictions_{timestamp}.csv
   ```

---

## Finalizing to the Incidents tab

The **Incidents tab** is the publication-layer database — clean, human-confirmed records, no `pred_*` / `conf_*` / flag columns.

```bash
# Dry-run to preview what will be written:
python data_pipeline/finalize_incidents.py --dry-run

# Write confirmed records:
python data_pipeline/finalize_incidents.py
```

The script:
- Reads all rows with `review_status = reviewed` from the Review tab
- Resolves each label: `human_{label}` if filled; falls back to `pred_{label}`
- Strips all pipeline columns; writes only clean fields to Incidents tab
- Idempotent (skips rows already present)

---

## Label structure

The classifier predicts 10 binary labels per record. Full definitions, inclusion/exclusion criteria, and examples are in `config/label_codebook.yaml`. The label set is **specific to the human-rights research application this pipeline was built for** — adapt the codebook for your own use case.

The label set covers event types like raid, arrest/detention, physical assault, harm to property, dispossession, religious encroachment, restriction of freedoms, coercive actions, protest, and multi-community indicators. Prevalences range from ~1% (rare events, always flagged for human review on positive prediction) to ~55% (common events, threshold-based routing).

Labels with low training prevalence are always flagged for human review when predicted positive, regardless of model confidence — a design choice to compensate for under-representation in training.

---

## Model details

### Baseline (TF-IDF + Logistic Regression)

- Input: TF-IDF vectors (unigrams to trigrams, 10k features) built from text + structured metadata fields
- One binary `LogisticRegression` per label (`MultiOutputClassifier`)
- Thresholding: rank-based — top-k documents by predicted probability are marked positive, where k is calibrated to match training prevalence per label
- Trained artifacts (`*.pkl`, `*.joblib`) are gitignored; train your own with `modeling/training/train_baseline_multilabel.py`

### DeBERTa v3

- Backbone: `microsoft/deberta-v3-base` (fine-tuned)
- Input: structured text format (`[AREA] ... [GOV] ... [DESC] ...`)
- Per-label sigmoid thresholds tuned on the validation split
- See [Model weights](#model-weights) below

### Uncertainty flagging

A prediction is flagged for human review if the model's confidence score is within `0.10` of the decision threshold. Adjust with `--confidence-band` when calling `predict_and_prepare_review.py`.

---

## Evaluation

The pipeline reports three categories of metrics, in order of recruiter relevance:

1. **Headline classification quality** — macro/micro F1 on the held-out test split, plus per-label precision, recall, and F1.
2. **Calibration** — Expected Calibration Error (ECE) and a reliability diagram, so top-1 probabilities mean what they claim.
3. **Routing effectiveness** — the fraction of test examples that can be auto-labeled at a given precision target, and the precision lift from sending margin-low examples to human review. This is the metric that justifies the HITL design.

### Reported numbers

> Numbers from the underlying private deployment (proprietary 10-label codebook, Arabic/English news incidents) are documented on the private Hugging Face model card for `jalva182/palestine-violence-deberta`. Access is granted alongside dataset access by separate agreement.
>
> Public-benchmark numbers (running the same pipeline on a public multilabel dataset — planned: GoEmotions or EUR-Lex) will be filled into the tables below when that benchmark run lands.

**Headline (macro-averaged on held-out test split)**

| Setup | macro-F1 | macro-Precision | macro-Recall | Params |
|---|---|---|---|---|
| TF-IDF + Logistic Regression baseline | _TBD_ | _TBD_ | _TBD_ | ~1M |
| DeBERTa-v3-base, multilabel head | **_TBD_** | _TBD_ | _TBD_ | 184M |

Reported alongside: micro-F1, weighted-F1, and per-label precision/recall/F1/support tables.

**Calibration**

| Setup | ECE | Brier (mean) | Reliability diagram |
|---|---|---|---|
| Baseline | _TBD_ | _TBD_ | `docs/calibration_baseline.png` (forthcoming) |
| DeBERTa-v3-base | _TBD_ | _TBD_ | `docs/calibration_deberta.png` (forthcoming) |

**Routing effectiveness — the HITL win condition**

| Margin threshold τ | Auto-labeled fraction | Auto-labeled precision | Sent to reviewer | Total throughput vs full-manual |
|---|---|---|---|---|
| τ = _TBD_ | _TBD_% | _TBD_% | _TBD_% | _TBD_× |

The point of the table above is to make explicit what a HITL system is supposed to deliver: **most examples get a confident automated label, only the ambiguous ones consume reviewer time.** A model that's accurate on average isn't enough — it has to be calibrated enough that low-margin examples are the genuinely hard ones.

### Methodology notes

- Splits: stratified train/val/test on document-level groupings, fixed seed 42.
- Per-label sigmoid thresholds tuned on the validation split (sweeping on macro-F1, then sanity-checked against per-label precision floors).
- For multilabel calibration, ECE is computed per label and macro-averaged.
- Uncertainty band (`--confidence-band 0.10`) is a top-line knob; the routing-effectiveness table sweeps this to surface the precision/throughput tradeoff.

---

## Model weights

The fine-tuned DeBERTa v3 multilabel weights for the original research project are hosted on the Hugging Face Hub at **[`jalva182/palestine-violence-deberta`](https://huggingface.co/jalva182/palestine-violence-deberta)** as a private model.

The model card describes:
- Base model and fine-tuning configuration
- Per-label F1, precision, and recall on the held-out test split
- Threshold calibration approach
- Known limitations (label prevalence skew, source-specific text patterns)

Access to the model is granted alongside dataset access — by separate agreement only. The training scripts in `modeling/training/` are sufficient to fine-tune your own DeBERTa multilabel head on your own dataset using the same architecture.

---

## Streamlit dashboard

```bash
streamlit run streamlit_app/app.py
```

Pages:
- **Dashboard** — batch counts, flagged doc counts, baseline vs DeBERTa metrics
- **Run Pipeline** — form to trigger any pipeline configuration; output streams live
- **Review Access** — link to Google Sheets Review tab; merge-from-Sheets button
- **Export Analysis** — export to SPSS (.sav) or R (.rds) for statistical analysis
- **Retrain** — per-label metrics table; checkpoint save/restore; retrain button; finalize-to-Incidents section
- **Documentation** — full inline user guide with PDF download

The Retrain page automatically saves a checkpoint of the current model artifacts before every retrain or merge operation. Checkpoints stored in `modeling/backups/` (gitignored), restorable with one click.

**Streamlit Cloud notes:**
- `requirements.txt` for all cloud deps. GPU libs (`torch`, `transformers`) are in `requirements-gpu.txt`, not installed on cloud.
- Service account credentials via Streamlit Cloud Secrets (see [Setup](#2-google-sheets-credentials)).
- Baseline runs fine on cloud; DeBERTa inference requires local/GPU setup.

---

## Retraining

After ~200+ new reviewed records have accumulated:

```bash
# 1. Merge all reviewed batches:
python modeling/merge_reviewed_labels.py --from-sheets \
    --predictions data/review_exports/predictions_{timestamp}.csv

# 2. Retrain baseline:
python modeling/training/train_baseline_multilabel.py

# 3. Compare new metrics against the previous checkpoint

# 4. If metrics hold, retrain DeBERTa (local/GPU only):
python modeling/training/train_transformer_multilabel.py
```

Or:
```bash
python run_pipeline.py --retrain-only
```

---

## Export for statistical analysis

The Incidents tab can be exported to SPSS / R for downstream stats work:

```bash
python data_pipeline/export_for_analysis.py
```

Outputs:
- `incidents.sav` — SPSS format (via `pyreadstat`)
- `incidents.rds` — R data frame (via `rpy2`, requires R installed)

---

## Notes for developers

- All scripts resolve paths relative to `PROJECT_ROOT = Path(__file__).resolve().parent` — do not rely on the current working directory
- Google Sheets ID is configured in `data_pipeline/database/sheets_interface.py` — point it at your own spreadsheet
- `service_account.json` must be in the project root and is gitignored; on Streamlit Cloud it's written from secrets at startup
- Pipeline logs are written to `logs/pipeline_{timestamp}.log` in addition to stdout
- Large model files use Git LFS; DeBERTa weights are excluded from git entirely and hosted on HF Hub
- To regenerate the PDF user guide: `python docs/generate_dashboard_guide.py`

---

## Project context

This pipeline was built for and is currently used in a private research project documenting incidents in Arabic and English news sources. The codebook, ingestion patterns, and label set are tuned for that domain. The engineering pattern (HITL multilabel + margin-based uncertainty routing + Sheets feedback + retraining loop) is generic and adaptable to other multilabel review workflows.

Built with [@Carson1829](https://github.com/Carson1829) on the data pipeline and review workflow side.

---

**Author:** Antonio Alvarez Maciel · M.S. NLP, UC Santa Cruz · [LinkedIn](https://linkedin.com/in/jose-alvarez-maciel) · [Hugging Face](https://huggingface.co/jalva182) · [Email](mailto:jalva182@ucsc.edu)

**License:** None — see [`NOTICE.md`](NOTICE.md). All rights reserved.
