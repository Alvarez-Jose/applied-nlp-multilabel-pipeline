"""
docs/generate_dashboard_guide.py

Generates the Palestine Violence Archive — Dashboard User Guide as a PDF.

Usage:
    python docs/generate_dashboard_guide.py
    # Output: docs/Dashboard_User_Guide.pdf

Can also be called from Python to get raw bytes (used by the Streamlit
Documentation page for the in-browser download button):

    from docs.generate_dashboard_guide import build_pdf_bytes
    pdf_bytes = build_pdf_bytes()
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "docs" / "Dashboard_User_Guide.pdf"


# ---------------------------------------------------------------------------
# Content definition
# ---------------------------------------------------------------------------

TITLE   = "Palestine Violence Archive"
SUBTITLE = "Dashboard & Pipeline User Guide"
VERSION = f"Version 1.0  |  {date.today().strftime('%B %Y')}"

SECTIONS = [
    {
        "heading": "1. Project Overview",
        "body": (
            "The Palestine Violence Archive is a research pipeline built for a political "
            "science lab at UC San Diego. It automatically collects news articles about "
            "human rights incidents affecting Palestinians in the West Bank from the WAFA "
            "news agency, classifies each incident across 10 violence labels using machine "
            "learning, routes uncertain cases to human reviewers, and continuously retrains "
            "the model as reviewed labels accumulate.\n\n"
            "The goal is to build a clean, labeled dataset of incidents (raids, arrests, "
            "property destruction, etc.) that the professor and research assistants (RAs) "
            "can export to SPSS or R for statistical analysis and eventual publication."
        ),
    },
    {
        "heading": "2. Pipeline Flow",
        "body": (
            "The pipeline runs in four stages:\n\n"
            "  1. Ingest  —  Scrapes new WAFA articles and writes them to the Google "
            "Sheets 'Reports' tab. Duplicate URLs are skipped automatically.\n\n"
            "  2. Export  —  Reads unprocessed rows from the 'Reports' tab and saves "
            "them as a timestamped CSV in data/incoming/.\n\n"
            "  3. Predict  —  Runs the baseline (TF-IDF + logistic regression) or "
            "DeBERTa v3 model on the batch CSV. Produces a compact review sheet "
            "(XLSX/CSV) in data/review_exports/ with model predictions, confidence "
            "scores, uncertainty flags, and empty human_* columns for RAs to fill.\n\n"
            "  4. Retrain  —  After RAs review enough batches and labels are merged "
            "back into the training dataset, the baseline model is rebuilt and evaluated. "
            "DeBERTa can then be fine-tuned separately."
        ),
    },
    {
        "heading": "3. Dashboard Pages",
        "body": (
            "The left sidebar lists five pages. Each is described below."
        ),
    },
    {
        "heading": "3a. Dashboard",
        "body": (
            "The landing page gives a live snapshot of the pipeline state.\n\n"
            "  Incoming batch size  —  Number of articles waiting in data/incoming/ "
            "that have not yet been reviewed.\n\n"
            "  Review queue  —  How many rows in the latest review sheet still have "
            "review_status != 'reviewed'.\n\n"
            "  Model metrics  —  Per-label F1, precision, and recall from the last "
            "baseline training run (read from baseline_meta.json).\n\n"
            "  IRR panel  —  Computes inter-rater reliability (Cohen's Kappa) for "
            "double-reviewed rows. A macro Kappa above 0.80 indicates strong agreement "
            "between RAs. Click 'Compute IRR' to refresh after RAs finish a batch.\n\n"
            "  Refresh  —  Clicking 'Refresh' reloads all metrics without re-running "
            "the pipeline."
        ),
    },
    {
        "heading": "3b. Run Pipeline",
        "body": (
            "The main control panel for running the full ingest → export → predict "
            "workflow.\n\n"
            "  Model  —  Choose 'baseline' (fast, CPU-only, good for daily runs) or "
            "'deberta' (slower, GPU-recommended, higher accuracy).\n\n"
            "  Skip ingest  —  Check this if you do not want to scrape new articles. "
            "The pipeline will use whatever is already in Google Sheets.\n\n"
            "  Re-export all reports  —  Forces re-export of all reports, ignoring "
            "the already-exported tracking file. Use if you need to reprocess old data.\n\n"
            "  Push compact review sheet to Sheets  —  After predicting, automatically "
            "uploads the review sheet to the Google Sheets 'Review' tab so RAs can "
            "work collaboratively online.\n\n"
            "  Dry run  —  Prints exactly what each step would do without writing any "
            "files. Always a safe first step on a new machine.\n\n"
            "  Deduplicate batch  —  Removes near-duplicate articles (same date and "
            "governorate, very similar text) before prediction, preventing double-counting.\n\n"
            "  Max rows per batch  —  Caps the batch at N rows. Recommended: 100-150 "
            "to keep review sessions manageable for RAs.\n\n"
            "  Quick Actions  —  Pre-configured shortcuts for the most common runs "
            "(skip ingest + baseline + push to Sheets; dry run preview).\n\n"
            "  Log output  —  Streams live as the pipeline runs. The log persists on "
            "screen after the run completes so you can scroll through it."
        ),
    },
    {
        "heading": "3c. Review Access",
        "body": (
            "Provides access to the latest batch files and links to Google Sheets.\n\n"
            "  Google Sheets link  —  Opens the shared spreadsheet where RAs review "
            "predictions. The 'Reports' tab contains raw scraped articles; the 'Review' "
            "tab contains model predictions with empty human_* columns.\n\n"
            "  Latest review file  —  Shows the most recent review_compact_*.xlsx or "
            ".csv from data/review_exports/. Click to preview the first 20 rows.\n\n"
            "  Latest predictions file  —  Shows the raw inference audit trail "
            "(all confidence scores and flags) from the most recent predict run.\n\n"
            "  RA workflow reminder  —  Step-by-step instructions reminding RAs "
            "what to fill in (human_* columns, review_status = 'reviewed', "
            "reviewer_notes) and where to save the completed file."
        ),
    },
    {
        "heading": "3d. Export Analysis",
        "body": (
            "Exports the cleaned, labeled incident database to formats the professor "
            "can open directly in SPSS, JASP, or R.\n\n"
            "  Years  —  Which years of processed data to include (default: 2023 2024 2025). "
            "Requires data/processed/oppression_{year}.csv to exist; run "
            "normalize_raw.py first if those files are missing.\n\n"
            "  Output directory  —  Where to write the exported files (default: analysis/).\n\n"
            "  Skip SPSS export  —  Skips writing the .sav file. Use this if pyreadstat "
            "is not installed.\n\n"
            "  Exported files  —  After running, the page lists all exported files with "
            "sizes. The .sav file includes full variable labels and value labels "
            "(e.g. 0 = No, 1 = Yes for all binary columns). The codebook_variables.csv "
            "is a variable dictionary showing type, valid N, missing %, and value ranges.\n\n"
            "  R usage  —  An R code snippet at the bottom shows how to load the .sav "
            "with value labels (haven::read_sav) or the plain CSV, and how to run the "
            "bundled descriptive_analysis.R script that produces 11 summary tables and "
            "7 publication-ready figures."
        ),
    },
    {
        "heading": "3e. Retrain  (Admin only)",
        "body": (
            "This page rebuilds the active model. Only run it after enough reviewed "
            "batches have been merged into the training dataset.\n\n"
            "  Training data stats  —  Shows the current row count of "
            "master_reviewed_dataset.csv so you know how much labeled data exists.\n\n"
            "  Checkpoint system  —  Before retraining, the dashboard automatically "
            "creates a timestamped backup of all model artifacts (baseline .pkl files, "
            "DeBERTa meta JSON, master CSV). You can also create a manual checkpoint "
            "at any time and restore any previous checkpoint if something goes wrong.\n\n"
            "  Retrain baseline  —  Runs export_training_data.py followed by "
            "train_baseline_multilabel.py. Takes 1-5 minutes on CPU.\n\n"
            "  Retrain DeBERTa  —  Fine-tunes the DeBERTa v3 transformer. "
            "Requires a GPU; plan for 1-3 hours on a single card.\n\n"
            "  Merge reviewed labels  —  Merges a completed review XLSX back into "
            "the master training CSV. Run this after RAs finish each batch before "
            "retraining. Automatically checkpoints before merging."
        ),
    },
    {
        "heading": "4. RA Review Workflow",
        "body": (
            "Research assistants follow this process for each batch:\n\n"
            "  Step 1  —  Open the review sheet. Either use the Google Sheets link "
            "(Review tab) or download the .xlsx from data/review_exports/.\n\n"
            "  Step 2  —  For each row, read the description column and check or "
            "correct the 10 pred_* columns. Fill in the corresponding human_* column "
            "with 1 (yes) or 0 (no). Leave blank if uncertain and add a note.\n\n"
            "  Step 3  —  For rows marked double_review = TRUE, a second RA should "
            "independently fill the human2_* columns without looking at the first "
            "RA's answers. This enables inter-rater reliability measurement.\n\n"
            "  Step 4  —  Set review_status = 'reviewed' for each completed row. "
            "Optionally fill reviewer_notes with observations.\n\n"
            "  Step 5  —  Save the completed file to data/reviewed/ and notify the "
            "pipeline operator to run the merge step in the Retrain page."
        ),
    },
    {
        "heading": "5. Incident Labels",
        "body": (
            "The model classifies each incident into 10 binary labels (1 = present):\n\n"
            "  Raid  —  Organized entry by Israeli forces into a Palestinian area to "
            "arrest, search, or impose presence.\n\n"
            "  Arrest/Detention  —  Apprehension or detention of Palestinians by "
            "Israeli forces or the Palestinian Authority.\n\n"
            "  Physical Assault  —  Direct violence: shooting, beating, tear gas, "
            "stun grenades, dogs.\n\n"
            "  Coercive Actions  —  Intimidation or harassment without direct assault: "
            "verbal threats, searches, show-of-force deployments.\n\n"
            "  Restriction of Freedoms  —  Curfews, checkpoints, roadblocks, "
            "prevention of agricultural or medical access.\n\n"
            "  Religious Encroachment  —  Incursion into or desecration of mosques, "
            "churches, or holy sites; interference with prayers or observances.\n\n"
            "  Harm to Property  —  Vandalism (graffiti, slashed tires) or destruction "
            "(demolition, arson) of Palestinian property.\n\n"
            "  Dispossession  —  Seizure or theft of land, livestock, crops, or assets.\n\n"
            "  Protest  —  Palestinian organized demonstration, march, or civil "
            "resistance action.\n\n"
            "  Multi-Community Incident  —  Simultaneous or coordinated action "
            "targeting multiple Palestinian villages or towns."
        ),
    },
    {
        "heading": "6. Models",
        "body": (
            "Baseline (TF-IDF + Logistic Regression)\n"
            "  Location: modeling/saved_models/baseline_rank_based_fixed/\n"
            "  Speed: seconds on CPU. Best for daily runs and quick iteration.\n"
            "  Thresholding: rank-based — the model predicts the top-k articles per "
            "label, where k matches the observed training prevalence rate.\n\n"
            "DeBERTa v3 (Transformer)\n"
            "  Location: modeling/saved_models/deberta_v3_multilabel/\n"
            "  Weights: stored on HuggingFace Hub (jalva182/palestine-violence-deberta)\n"
            "  Speed: ~1 min per 100 rows on GPU; much slower on CPU.\n"
            "  Use for final-pass labeling when accuracy matters most.\n\n"
            "Both models output a confidence score (0-1) per label. Rows where "
            "confidence is within 0.10 of the decision threshold are flagged as "
            "needs_review = TRUE and routed to human reviewers."
        ),
    },
    {
        "heading": "7. File Locations",
        "body": (
            "  data/incoming/         — Raw batches exported from Google Sheets\n"
            "  data/review_exports/   — Model predictions + compact review sheets\n"
            "  data/reviewed/         — Completed RA review files\n"
            "  data/training/         — Master labeled dataset (used for retraining)\n"
            "  data/processed/        — Normalized yearly CSVs (2023, 2024, 2025)\n"
            "  analysis/              — SPSS .sav, R CSV, codebook, R analysis script\n"
            "  modeling/saved_models/ — Baseline and DeBERTa model artifacts\n"
            "  modeling/backups/      — Timestamped model checkpoints\n"
            "  logs/                  — Pipeline run logs (pipeline_YYYYMMDD.log)\n"
            "  config/                — Label codebook YAML"
        ),
    },
    {
        "heading": "8. Common Issues",
        "body": (
            "No articles scraped (0 new reports)\n"
            "  Cause: robots.txt block or network issue.\n"
            "  Fix: Check the log for 'robots.txt' warnings. Try --skip-ingest "
            "and re-export from what is already in Sheets.\n\n"
            "pyreadstat not installed\n"
            "  Fix: pip install pyreadstat  (or check 'Skip SPSS export' on the "
            "Export Analysis page).\n\n"
            "DeBERTa weights not found\n"
            "  Fix: Download from HuggingFace Hub into "
            "modeling/saved_models/deberta_v3_multilabel/deberta_v3_best/\n\n"
            "Google Sheets authentication error\n"
            "  Fix: Ensure the service account JSON key path is set in environment "
            "variable GOOGLE_APPLICATION_CREDENTIALS, or in Streamlit secrets.\n\n"
            "Log disappears when clicking buttons\n"
            "  This is fixed — logs are stored in session state and persist across "
            "page interactions."
        ),
    },
]


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Replace characters outside Latin-1 range with ASCII equivalents."""
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2019": "'",   # right single quote
        "\u2018": "'",   # left single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2192": "->",  # right arrow
        "\u2190": "<-",  # left arrow
        "\u2022": "-",   # bullet
        "\u00b7": "-",   # middle dot
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Drop any remaining non-Latin-1 characters
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf_bytes() -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(left=20, top=20, right=20)

    # ---- Cover page ----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(30)
    pdf.multi_cell(0, 12, _clean(TITLE), align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 16)
    pdf.multi_cell(0, 10, _clean(SUBTITLE), align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 7, _clean(VERSION), align="C")
    pdf.ln(14)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(
        0, 6,
        "This guide explains how to use the Streamlit pipeline dashboard,\n"
        "what each page does, how research assistants should review batches,\n"
        "and how to export data for statistical analysis.",
        align="C",
    )
    pdf.set_text_color(0, 0, 0)

    # ---- Content pages ----
    pdf.add_page()

    for section in SECTIONS:
        heading = _clean(section["heading"])
        body    = _clean(section["body"])

        # Determine heading level from the prefix (3a, 3b ... are sub-headings)
        is_sub = heading[1:2] in ("a", "b", "c", "d", "e")

        if is_sub:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_fill_color(230, 240, 255)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(
                0, 8, f"  {heading}", fill=True,
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
        else:
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_fill_color(50, 100, 180)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(
                0, 9, f"  {heading}", fill=True,
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
            pdf.set_text_color(0, 0, 0)

        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)

        # Split body into paragraphs (double-newline separated)
        paragraphs = body.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Indented lines (bullet-style entries)
            if para.startswith("  "):
                for line in para.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    pdf.set_x(24)
                    pdf.multi_cell(0, 5.5, line)
            else:
                pdf.set_x(20)
                pdf.multi_cell(0, 5.5, para)
            pdf.ln(2)

        pdf.ln(4)

        # Page break before long major sections
        if heading in ("4. RA Review Workflow", "6. Models", "8. Common Issues"):
            pdf.add_page()

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = build_pdf_bytes()
    OUTPUT_PATH.write_bytes(data)
    print(f"PDF written: {OUTPUT_PATH}  ({len(data) // 1024} KB)")


if __name__ == "__main__":
    main()
