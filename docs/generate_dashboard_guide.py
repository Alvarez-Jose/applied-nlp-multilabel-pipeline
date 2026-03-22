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

TITLE    = "Palestine Violence Archive"
SUBTITLE = "Dashboard & Pipeline User Guide"
VERSION  = f"Version 2.0  |  {date.today().strftime('%B %Y')}"

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
            "The goal is to produce a clean, labeled dataset of incidents suitable for "
            "statistical analysis in SPSS or R and for export to a permanent research database "
            "in support of peer-reviewed publication."
        ),
    },
    {
        "heading": "2. Google Sheets Structure",
        "body": (
            "The pipeline uses three tabs in one shared Google Spreadsheet:\n\n"
            "  Reports  —  Raw scraped articles (input layer). Written by the scraper. "
            "Contains: report_id, url, title, body_text, published_date, language, "
            "location_raw, actors_raw.\n\n"
            "  Review  —  Model predictions awaiting human verification (working layer). "
            "Written by the pipeline after each prediction run. RAs fill in the human_* "
            "columns here and set review_status = reviewed.\n\n"
            "  Incidents  —  Finalized, clean incident records (publication layer). "
            "Written by the Transfer to Incidents step after human review is complete. "
            "Contains only facts and labels — no model scores or pipeline metadata. "
            "This tab mirrors what a proper database table will look like, making future "
            "migration straightforward."
        ),
    },
    {
        "heading": "3. Pipeline Flow",
        "body": (
            "The pipeline runs in seven steps:\n\n"
            "  Step 1 — Ingest  —  Scrapes new WAFA articles into the Reports tab. "
            "Duplicate URLs are skipped automatically.\n\n"
            "  Step 2 — Export  —  Reads unprocessed rows from Reports and saves them "
            "as a timestamped CSV in data/incoming/.\n\n"
            "  Step 3 — Predict  —  Runs the baseline or DeBERTa model. Produces a "
            "compact review sheet in data/review_exports/ and pushes it to the Review tab.\n\n"
            "  Step 4 — Human Review  —  RAs open the Review tab, fill in human_* columns "
            "(1/0), and set review_status = reviewed. Double-review rows (5% random sample) "
            "are filled by a second RA for inter-rater reliability tracking.\n\n"
            "  Step 5 — Merge  —  Reviewed labels are merged into master_reviewed_dataset.csv "
            "for model retraining.\n\n"
            "  Step 6 — Transfer to Incidents  —  Clean confirmed records are written to "
            "the Incidents tab, stripping all pipeline columns. Run after each merge.\n\n"
            "  Step 7 — Retrain  —  Once enough labeled data accumulates, the baseline "
            "model is rebuilt. DeBERTa can be fine-tuned locally with a GPU."
        ),
    },
    {
        "heading": "4. Dashboard Pages",
        "body": "The left sidebar lists six pages. Each is described below.",
    },
    {
        "heading": "4a. Dashboard",
        "body": (
            "The landing page gives a live snapshot of the pipeline state.\n\n"
            "  Incoming batch size  —  Articles in data/incoming/ not yet reviewed.\n\n"
            "  Review queue  —  Rows in the latest review sheet with review_status not reviewed.\n\n"
            "  Model metrics  —  Per-label F1, precision, and recall from the last baseline "
            "training run (read from baseline_meta.json).\n\n"
            "  IRR panel  —  Cohen's Kappa inter-rater reliability for double-reviewed rows. "
            "A macro Kappa above 0.80 indicates strong RA agreement. "
            "Click 'Compute IRR' to refresh after RAs finish a batch."
        ),
    },
    {
        "heading": "4b. Run Pipeline",
        "body": (
            "The main control panel for Steps 1-3 (ingest, export, predict).\n\n"
            "  Model  —  'baseline' runs on CPU and works on Streamlit Cloud. "
            "'deberta' is more accurate but requires a GPU locally.\n\n"
            "  Skip ingest  —  Use existing Sheets data without scraping new articles.\n\n"
            "  Re-export all  —  Forces re-export ignoring the already-exported tracker.\n\n"
            "  Push to Sheets  —  After predicting, uploads the review sheet to the "
            "Review tab so RAs can work collaboratively online.\n\n"
            "  Dry run  —  Prints what each step would do without writing any files.\n\n"
            "  Deduplicate  —  Removes near-duplicate articles before prediction.\n\n"
            "  Max rows  —  Caps the batch. Recommended: 100-150 for manageable reviews.\n\n"
            "  Log output  —  Streams live and persists after the run so you can scroll it."
        ),
    },
    {
        "heading": "4c. Review Access",
        "body": (
            "Access to review files and the Google Sheets Review tab.\n\n"
            "  Google Sheets link  —  Opens the spreadsheet. Direct RAs to the 'Review' tab.\n\n"
            "  Latest review file  —  Preview of the most recent review_compact_*.csv.\n\n"
            "  Latest predictions  —  Raw inference audit trail with all confidence scores.\n\n"
            "  RA workflow reminder  —  Step-by-step instructions shown on-screen."
        ),
    },
    {
        "heading": "4d. Export Analysis",
        "body": (
            "Exports the cleaned incident database for the professor (SPSS/R).\n\n"
            "  Years  —  Which years to include (default: 2023 2024 2025). "
            "Requires data/processed/oppression_{year}.csv.\n\n"
            "  Output directory  —  Where to write files (default: analysis/).\n\n"
            "  Skip SPSS export  —  Check this if pyreadstat is not installed.\n\n"
            "  Exported files  —  .sav (SPSS/JASP), .csv (R/Excel), and "
            "codebook_variables.csv. The .sav has full variable labels and "
            "0=No / 1=Yes value labels for all binary columns.\n\n"
            "  R analysis script  —  analysis/descriptive_analysis.R produces "
            "11 summary tables and 7 figures. Run with: Rscript analysis/descriptive_analysis.R"
        ),
    },
    {
        "heading": "4e. Retrain  (Admin only)",
        "body": (
            "Rebuilds the model and manages the Incidents tab.\n\n"
            "  Training data stats  —  Current row count of master_reviewed_dataset.csv.\n\n"
            "  Checkpoint system  —  Auto-creates a timestamped backup before any destructive "
            "action. Manually create or restore checkpoints at any time.\n\n"
            "  Merge reviewed labels  —  Pulls reviewed rows from the Sheets Review tab and "
            "merges human labels into the master training CSV. Auto-checkpoints before merging.\n\n"
            "  Transfer to Incidents  —  Copies confirmed records (review_status = reviewed) "
            "from the Review tab into the clean Incidents tab, stripping all pipeline columns. "
            "Run after every merge to keep the Incidents tab current for the database.\n\n"
            "  Retrain baseline  —  Rebuilds the TF-IDF + logistic regression model. "
            "1-5 minutes on CPU.\n\n"
            "  Retrain DeBERTa  —  Fine-tunes the transformer. Requires GPU; plan 1-3 hours."
        ),
    },
    {
        "heading": "4f. Documentation",
        "body": (
            "This page. Contains the full inline user guide and a Download PDF Guide button "
            "that generates and downloads this document."
        ),
    },
    {
        "heading": "5. RA Review Workflow",
        "body": (
            "Research assistants follow this process for each batch:\n\n"
            "  Step 1  —  Open Google Sheets and go to the Review tab (or download the "
            "CSV from data/review_exports/).\n\n"
            "  Step 2  —  For each row, read the description and fill the 10 human_* "
            "columns with 1 (yes) or 0 (no). Leave blank if uncertain; add reviewer_notes.\n\n"
            "  Step 3  —  For rows marked double_review = TRUE, a second RA independently "
            "fills the human2_* columns without seeing the first RA's answers. "
            "This measures inter-rater reliability.\n\n"
            "  Step 4  —  Set review_status = reviewed for each completed row.\n\n"
            "  Step 5  —  Notify the pipeline operator who will run Merge then "
            "Transfer to Incidents."
        ),
    },
    {
        "heading": "6. Incident Labels",
        "body": (
            "The model classifies each incident into 10 binary labels (1 = present):\n\n"
            "  Raid  —  Organized entry by Israeli forces into a Palestinian area.\n\n"
            "  Arrest/Detention  —  Apprehension or detention by Israeli forces or the PA.\n\n"
            "  Physical Assault  —  Shooting, beating, tear gas, stun grenades, dogs.\n\n"
            "  Coercive Actions  —  Intimidation or harassment without direct assault.\n\n"
            "  Restriction of Freedoms  —  Curfews, checkpoints, roadblocks.\n\n"
            "  Religious Encroachment  —  Incursion into or desecration of religious sites.\n\n"
            "  Harm to Property  —  Vandalism or destruction of Palestinian property.\n\n"
            "  Dispossession  —  Seizure or theft of land, livestock, crops, or assets.\n\n"
            "  Protest  —  Palestinian organized demonstration or civil resistance.\n\n"
            "  Multi-Community Incident  —  Coordinated action across multiple communities."
        ),
    },
    {
        "heading": "7. Models",
        "body": (
            "Baseline (TF-IDF + Logistic Regression)\n"
            "  Location: modeling/saved_models/baseline_rank_based_fixed/\n"
            "  Runs on CPU. Used by the Streamlit Cloud dashboard.\n"
            "  Rank-based thresholding: predicts top-k articles per label where k "
            "matches the training prevalence rate.\n\n"
            "DeBERTa v3 (Transformer)\n"
            "  Location: modeling/saved_models/deberta_v3_multilabel/\n"
            "  Weights: HuggingFace Hub (jalva182/palestine-violence-deberta)\n"
            "  Requires GPU. Install with: pip install -r requirements-gpu.txt\n\n"
            "Both models output a confidence score (0-1) per label. Rows within 0.10 "
            "of the threshold are flagged needs_review = TRUE."
        ),
    },
    {
        "heading": "8. Streamlit Cloud Setup",
        "body": (
            "1. Connect the repo at share.streamlit.io. Set main file: streamlit_app/app.py\n\n"
            "2. Add Google credentials in app Settings -> Secrets:\n"
            "   [gcp_service_account]\n"
            "   type = \"service_account\"\n"
            "   project_id = \"your-project-id\"\n"
            "   private_key = \"-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n\"\n"
            "   client_email = \"...@....iam.gserviceaccount.com\"\n"
            "   ... (all other fields from service_account.json)\n\n"
            "3. DeBERTa is not available on Streamlit Cloud (torch too large). "
            "Use the baseline model on Cloud; run DeBERTa locally.\n\n"
            "4. After updating secrets or pushing code changes, use three-dot menu -> "
            "Reboot app to force a full restart."
        ),
    },
    {
        "heading": "9. File Locations",
        "body": (
            "  data/incoming/         — Raw batches exported from Google Sheets\n"
            "  data/review_exports/   — Model predictions + compact review sheets\n"
            "  data/reviewed/         — Completed RA review files (local copies)\n"
            "  data/training/         — Master labeled dataset (used for retraining)\n"
            "  data/processed/        — Normalized yearly CSVs (2023, 2024, 2025)\n"
            "  analysis/              — SPSS .sav, R CSV, codebook, R analysis script\n"
            "  modeling/saved_models/ — Baseline and DeBERTa model artifacts\n"
            "  modeling/backups/      — Timestamped model checkpoints\n"
            "  logs/                  — Pipeline run logs (pipeline_YYYYMMDD.log)\n"
            "  config/                — Label codebook YAML\n"
            "  docs/                  — PDF user guide and generator script\n"
            "  requirements.txt       — Streamlit Cloud dependencies (no torch)\n"
            "  requirements-gpu.txt   — Local DeBERTa dependencies (torch + transformers)"
        ),
    },
    {
        "heading": "10. Common Issues",
        "body": (
            "100% of rows flagged for review\n"
            "  Normal for the first batch before the model is retrained on your data. "
            "Confidence improves as reviewed batches are merged and the model is retrained.\n\n"
            "No articles scraped (0 new reports)\n"
            "  Check the log for robots.txt warnings. Use Skip Ingest and re-run.\n\n"
            "service_account.json not found\n"
            "  Add credentials to Streamlit Cloud Secrets under [gcp_service_account].\n\n"
            "DeBERTa weights not found\n"
            "  Download from HuggingFace Hub into "
            "modeling/saved_models/deberta_v3_multilabel/deberta_v3_best/\n"
            "  Install GPU deps first: pip install -r requirements-gpu.txt\n\n"
            "pyreadstat not installed\n"
            "  pip install pyreadstat  or check 'Skip SPSS export' on the Export page.\n\n"
            "Transfer to Incidents shows 0 rows\n"
            "  No rows have review_status = reviewed yet. RAs must complete their review first."
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
        if heading in ("5. RA Review Workflow", "7. Models", "9. File Locations"):
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
