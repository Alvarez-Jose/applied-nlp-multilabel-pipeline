"""
app.py — Palestine Violence Archive · Pipeline Dashboard

A thin Streamlit UI that orchestrates the existing pipeline scripts.
RAs use Google Sheets to review; this UI is for running and monitoring.

Pages (sidebar):
    Dashboard     — run snapshot, queue size, model metrics
    Run Pipeline  — trigger ingest → predict → (push to Sheets)
    Review        — latest batch summary, Sheets link
    Retrain       — admin-only retraining, per-label metrics

Run locally:
    streamlit run streamlit_app/app.py

Deploy:
    Push to GitHub → connect to Streamlit Cloud → set main file path to
    streamlit_app/app.py. Add Google service account credentials to
    Streamlit secrets if needed.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
# streamlit_app/ lives one level inside the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON       = sys.executable

# ---------------------------------------------------------------------------
# Service account — write from Streamlit secrets if not already on disk
# ---------------------------------------------------------------------------
def _ensure_service_account() -> None:
    """
    On Streamlit Cloud the service_account.json cannot be committed to git.
    Instead, paste the JSON content into the app's Secrets under the key
    [gcp_service_account].  This function writes it to disk once at startup
    so that all pipeline subprocesses can find it at the expected path.
    """
    sa_path = PROJECT_ROOT / "service_account.json"
    if sa_path.exists():
        return
    try:
        sa_info = dict(st.secrets.get("gcp_service_account", {}))
        if sa_info:
            sa_path.write_text(json.dumps(sa_info, indent=2))
    except Exception:
        pass  # will surface as a clear error when the pipeline actually runs

_ensure_service_account()

INCOMING_DIR  = PROJECT_ROOT / "data" / "incoming"
REVIEW_DIR    = PROJECT_ROOT / "data" / "review_exports"
REVIEWED_DIR  = PROJECT_ROOT / "data" / "reviewed"
TRAINING_DIR  = PROJECT_ROOT / "data" / "training"

BASELINE_META = (
    PROJECT_ROOT
    / "modeling" / "saved_models"
    / "baseline_rank_based_fixed" / "baseline_meta.json"
)
DEBERTA_META = (
    PROJECT_ROOT
    / "modeling" / "saved_models"
    / "deberta_v3_multilabel" / "deberta_v3_meta.json"
)

BACKUPS_DIR = PROJECT_ROOT / "modeling" / "backups"

SPREADSHEET_ID = "1Z7zu2JLxOIU1yK3SrXIz8yN3a-2Kzd7d0g0-VNrkwaI"
SHEETS_URL     = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}"

# Human-readable label names (order matches meta JSON)
LABEL_DISPLAY = {
    "raid_y":                    "Raid",
    "arrest_detention_y":        "Arrest/Detention",
    "physical_assault_y":        "Physical Assault",
    "harm_to_property_y":        "Harm to Property",
    "dispossession_y":           "Dispossession",
    "religious_encroachment_y":  "Religious Encroachment",
    "restriction_of_freedoms_y": "Restriction of Freedoms",
    "coercive_actions_y":        "Coercive Actions",
    "protest_y":                 "Protest",
    "multi_community_incident_y":"Multi-Community",
}

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Violence Archive — Pipeline",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Return the most recently modified file matching pattern, or None."""
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def _mtime_str(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return "—"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return raw


def _safe_float(val) -> Optional[float]:
    """Return float or None for NaN/null."""
    try:
        v = float(val)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _review_stats(compact_path: Optional[Path]) -> dict:
    """Read a compact review CSV and return basic counts."""
    if compact_path is None or not compact_path.exists():
        return {}
    try:
        df = pd.read_csv(compact_path, encoding="utf-8-sig")
        n_total    = len(df)
        n_review   = int(df["needs_review"].sum()) if "needs_review" in df.columns else "?"
        n_conflict = int(
            (df["codebook_conflict"].fillna("") != "").sum()
        ) if "codebook_conflict" in df.columns else "?"
        return {"total": n_total, "needs_review": n_review, "conflicts": n_conflict}
    except Exception:
        return {}


def _run_command(cmd: list, placeholder) -> int:
    """
    Stream a subprocess command into a Streamlit placeholder.
    Returns the process exit code.
    """
    placeholder.code("Starting…", language="")
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    for line in proc.stdout:
        lines.append(line)
        placeholder.code("".join(lines[-300:]), language="")
    proc.wait()
    return proc.returncode


def _build_metrics_table(meta: dict, split: str = "test") -> Optional[pd.DataFrame]:
    """
    Build a per-label metrics DataFrame from meta['test']['per_label'].
    per_label format: [label_name, precision, recall, f1, auc, prauc]
    """
    data = meta.get(split, {}).get("per_label")
    if not data:
        return None
    rows = []
    for entry in data:
        label = entry[0]
        prec  = _safe_float(entry[1]) if len(entry) > 1 else None
        rec   = _safe_float(entry[2]) if len(entry) > 2 else None
        f1    = _safe_float(entry[3]) if len(entry) > 3 else None
        auc   = _safe_float(entry[4]) if len(entry) > 4 else None
        rows.append({
            "Label":     LABEL_DISPLAY.get(label, label),
            "Precision": f"{prec:.3f}" if prec is not None else "—",
            "Recall":    f"{rec:.3f}"  if rec  is not None else "—",
            "F1":        f"{f1:.3f}"   if f1   is not None else "—",
            "AUC-ROC":   f"{auc:.3f}"  if auc  is not None else "—",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Checkpoint / backup helpers
# ---------------------------------------------------------------------------

def _create_backup(label: str = "") -> Path:
    """Copy current model artifacts to a timestamped checkpoint folder."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"_{label.strip().replace(' ', '_')}" if label.strip() else ""
    dest = BACKUPS_DIR / f"checkpoint_{ts}{slug}"
    dest.mkdir(parents=True, exist_ok=True)

    # Baseline model (pkl + meta JSON — typically a few MB)
    baseline_src = PROJECT_ROOT / "modeling" / "saved_models" / "baseline_rank_based_fixed"
    if baseline_src.exists():
        shutil.copytree(baseline_src, dest / "baseline_rank_based_fixed")

    # DeBERTa meta JSON only (weights live on HF Hub — too large to copy locally)
    deberta_meta_src = (
        PROJECT_ROOT / "modeling" / "saved_models"
        / "deberta_v3_multilabel" / "deberta_v3_meta.json"
    )
    if deberta_meta_src.exists():
        deb_dir = dest / "deberta_v3_multilabel"
        deb_dir.mkdir(exist_ok=True)
        shutil.copy2(deberta_meta_src, deb_dir / "deberta_v3_meta.json")

    # Master training dataset
    master_src = TRAINING_DIR / "master_reviewed_dataset.csv"
    if master_src.exists():
        shutil.copy2(master_src, dest / "master_reviewed_dataset.csv")

    # Checkpoint manifest
    contents = [str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()]
    (dest / "checkpoint_info.json").write_text(
        json.dumps({"created": ts, "label": label, "contents": contents}, indent=2)
    )
    return dest


def _list_backups() -> list[Path]:
    if not BACKUPS_DIR.exists():
        return []
    return sorted(
        [d for d in BACKUPS_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576


def _restore_backup(checkpoint_dir: Path):
    """Overwrite active model artifacts with contents of a checkpoint."""
    baseline_src  = checkpoint_dir / "baseline_rank_based_fixed"
    baseline_dest = PROJECT_ROOT / "modeling" / "saved_models" / "baseline_rank_based_fixed"
    if baseline_src.exists():
        if baseline_dest.exists():
            shutil.rmtree(baseline_dest)
        shutil.copytree(baseline_src, baseline_dest)

    deberta_meta_src  = checkpoint_dir / "deberta_v3_multilabel" / "deberta_v3_meta.json"
    deberta_meta_dest = (
        PROJECT_ROOT / "modeling" / "saved_models"
        / "deberta_v3_multilabel" / "deberta_v3_meta.json"
    )
    if deberta_meta_src.exists():
        shutil.copy2(deberta_meta_src, deberta_meta_dest)

    master_src  = checkpoint_dir / "master_reviewed_dataset.csv"
    master_dest = TRAINING_DIR / "master_reviewed_dataset.csv"
    if master_src.exists():
        shutil.copy2(master_src, master_dest)


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def page_dashboard():
    st.header("Dashboard")

    latest_batch   = _latest_file(INCOMING_DIR, "batch_*.csv")
    latest_compact = _latest_file(REVIEW_DIR,   "review_compact_*.csv")

    n_batches  = _count_files(INCOMING_DIR, "batch_*.csv")
    n_reviewed = _count_files(REVIEWED_DIR, "reviewed_incidents_*.csv")
    stats      = _review_stats(latest_compact)

    # --- Top metrics ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Incoming batches", n_batches)
        st.caption(f"Latest: {_mtime_str(latest_batch)}")
        if latest_batch:
            st.caption(f"`{latest_batch.name}`")
    with c2:
        st.metric("Docs flagged for review", stats.get("needs_review", "—"))
        st.caption(f"In latest batch ({stats.get('total', '?')} docs)")
    with c3:
        st.metric("Codebook conflicts", stats.get("conflicts", "—"))
        if latest_compact:
            st.caption(f"`{latest_compact.name}`")
    with c4:
        st.metric("Reviewed batches merged", n_reviewed)
        master = TRAINING_DIR / "master_reviewed_dataset.csv"
        if master.exists():
            try:
                n_master = len(pd.read_csv(master))
                st.caption(f"{n_master} rows in master dataset")
            except Exception:
                pass

    st.divider()

    # --- Model comparison ---
    st.subheader("Model Performance (Test Split — 2025 data)")
    baseline_meta = _load_meta(BASELINE_META)
    deberta_meta  = _load_meta(DEBERTA_META)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**Baseline — TF-IDF + LogReg**")
        if baseline_meta:
            b_test = baseline_meta.get("test", {})
            ci     = baseline_meta.get("bootstrap_test", {})
            st.metric(
                "Macro-F1",
                f"{b_test.get('macro_f1', '?'):.3f}",
                help=f"95% CI: {ci.get('macro_f1_ci95', ['?','?'])[0]:.3f} – "
                     f"{ci.get('macro_f1_ci95', ['?','?'])[1]:.3f}",
            )
            st.metric("Micro-F1", f"{b_test.get('micro_f1', '?'):.3f}")
            st.caption(f"Train rows: {baseline_meta.get('split', {}).get('train', {}).get('rows', '?')}")
            st.caption(f"File: {_mtime_str(BASELINE_META)}")
        else:
            st.warning("baseline_meta.json not found.")

    with mc2:
        st.markdown("**DeBERTa v3**")
        if deberta_meta:
            d_test = deberta_meta.get("test", {})
            ci     = deberta_meta.get("bootstrap_test", {})
            st.metric(
                "Macro-F1",
                f"{d_test.get('macro_f1', '?'):.3f}",
                help=f"95% CI: {ci.get('macro_f1_ci95', ['?','?'])[0]:.3f} – "
                     f"{ci.get('macro_f1_ci95', ['?','?'])[1]:.3f}",
            )
            st.metric("Micro-F1", f"{d_test.get('micro_f1', '?'):.3f}")
            st.caption(f"Train rows: {deberta_meta.get('split', {}).get('train', {}).get('rows', '?')}")
            st.caption(f"File: {_mtime_str(DEBERTA_META)}")
        else:
            st.warning("deberta_v3_meta.json not found.")

    st.divider()

    # --- Inter-rater reliability ---
    st.subheader("Inter-Rater Reliability")
    irr_path = REVIEW_DIR / "irr_latest.json"
    irr_col1, irr_col2 = st.columns([3, 1])
    with irr_col1:
        if irr_path.exists():
            try:
                irr = _load_meta(irr_path)
                macro = irr.get("macro_kappa")
                n_rows = irr.get("total_overlap_rows", 0)
                n_eval = irr.get("n_labels_evaluated", 0)
                mc_a, mc_b, mc_c = st.columns(3)
                mc_a.metric(
                    "Macro Kappa",
                    f"{macro:.3f}" if macro is not None else "--",
                    help="Average Cohen's Kappa across all labels. >0.61 is substantial agreement.",
                )
                mc_b.metric("Overlap rows", n_rows)
                mc_c.metric("Labels evaluated", f"{n_eval} / 10")

                per_label = irr.get("per_label", {})
                if per_label:
                    label_display = {
                        "raid": "Raid", "arrest_detention": "Arrest/Detention",
                        "physical_assault": "Physical Assault", "harm_to_property": "Harm to Property",
                        "dispossession": "Dispossession", "religious_encroachment": "Religious Encroachment",
                        "restriction_of_freedoms": "Restriction of Freedoms",
                        "coercive_actions": "Coercive Actions", "protest": "Protest",
                        "multi_community_incident": "Multi-Community",
                    }
                    irr_rows = []
                    for short, stats in per_label.items():
                        kappa = stats.get("kappa")
                        irr_rows.append({
                            "Label":       label_display.get(short, short),
                            "Pairs":       stats.get("n_pairs", 0),
                            "% Agreement": f"{stats['pct_agreement']*100:.1f}%" if stats.get("pct_agreement") is not None else "--",
                            "Kappa":       f"{kappa:.3f}" if kappa is not None else "--",
                        })
                    st.dataframe(pd.DataFrame(irr_rows), use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not load IRR summary: {e}")
        else:
            st.info("No IRR data yet. Run compute_irr.py after RAs complete double-review rows.")

    with irr_col2:
        st.write("")
        st.write("")
        if st.button("Compute IRR"):
            for key, default in [("irr_log", None), ("irr_rc", None)]:
                if key not in st.session_state:
                    st.session_state[key] = default
            live_box = st.empty()
            lines: list[str] = []
            proc = subprocess.Popen(
                [PYTHON, "modeling/compute_irr.py", "--from-sheets"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(PROJECT_ROOT),
            )
            for line in proc.stdout:
                lines.append(line)
                live_box.code("".join(lines[-100:]), language="")
            proc.wait()
            st.session_state.irr_log = "".join(lines)
            st.session_state.irr_rc  = proc.returncode
            live_box.empty()

        if st.session_state.get("irr_log"):
            st.code(st.session_state.irr_log[-3000:], language="")
            if st.session_state.irr_rc == 0:
                st.success("Done.")
            else:
                st.error(f"Exit {st.session_state.irr_rc}.")

    st.divider()

    # --- Recent exports table ---
    st.subheader("Recent Review Exports")
    if REVIEW_DIR.exists():
        recent = sorted(
            REVIEW_DIR.glob("review_compact_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:8]
        if recent:
            rows = []
            for p in recent:
                s = _review_stats(p)
                rows.append({
                    "File":         p.name,
                    "Created":      _mtime_str(p),
                    "Docs":         s.get("total", "?"),
                    "Needs review": s.get("needs_review", "?"),
                    "Conflicts":    s.get("conflicts", "?"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No review exports yet. Run the pipeline to generate batches.")
    else:
        st.info("data/review_exports/ not found.")


# ---------------------------------------------------------------------------
# Page: Run Pipeline
# ---------------------------------------------------------------------------

def page_run():
    st.header("Run Pipeline")
    st.caption(
        "Runs `run_pipeline.py` with the selected options. "
        "Output streams live below."
    )

    # Persist log output across reruns so it doesn't vanish on page interaction
    for key, default in [("run_log", None), ("run_rc", None), ("run_cmd", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    def _run_and_store(cmd: list):
        st.session_state.run_cmd = " ".join(str(c) for c in cmd)
        st.session_state.run_log = None
        st.session_state.run_rc = None
        live_box = st.empty()
        lines: list[str] = []
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        for line in proc.stdout:
            lines.append(line)
            live_box.code("".join(lines[-300:]), language="")
        proc.wait()
        st.session_state.run_log = "".join(lines)
        st.session_state.run_rc = proc.returncode
        live_box.empty()  # replaced by the persistent block below

    with st.form("pipeline_form"):
        col1, col2 = st.columns(2)
        with col1:
            model       = st.selectbox("Model", ["baseline", "deberta"], index=0)
            skip_ingest = st.checkbox("Skip ingest (use existing Sheets data)")
            export_all  = st.checkbox("Re-export all reports (ignore already-exported tracker)")
        with col2:
            push_sheets = st.checkbox("Push compact review sheet to Google Sheets Review tab")
            dry_run     = st.checkbox("Dry run (preview only — no files written)")
            dedup       = st.checkbox("Deduplicate batch before prediction", help="Removes near-duplicate articles (same date + region + high text similarity) before running the model.")
            max_rows    = st.number_input("Max rows per batch (0 = no cap)", min_value=0, value=0, step=50)

        submitted = st.form_submit_button("▶  Run Pipeline", type="primary")

    if submitted:
        cmd = [PYTHON, "run_pipeline.py", "--model", model]
        if skip_ingest:
            cmd.append("--skip-ingest")
        if export_all:
            cmd.append("--export-all")
        if push_sheets:
            cmd.append("--push-to-sheets")
        if dry_run:
            cmd.append("--dry-run")
        if dedup:
            cmd.append("--dedup")
        if max_rows and max_rows > 0:
            cmd.extend(["--max-rows", str(max_rows)])
        _run_and_store(cmd)

    # --- Quick shortcuts ---
    st.divider()
    st.subheader("Quick Actions")
    qc1, qc2 = st.columns(2)
    with qc1:
        if st.button("⏭  Skip ingest · baseline · push to Sheets"):
            _run_and_store([PYTHON, "run_pipeline.py", "--skip-ingest", "--model", "baseline", "--push-to-sheets"])
    with qc2:
        if st.button("🔍  Dry run (full pipeline preview)"):
            _run_and_store([PYTHON, "run_pipeline.py", "--dry-run"])

    # --- Persistent output (survives reruns) ---
    if st.session_state.run_cmd:
        st.divider()
        st.caption(f"Last run: `{st.session_state.run_cmd}`")
        if st.session_state.run_log is not None:
            st.code(st.session_state.run_log[-8000:], language="")
        if st.session_state.run_rc == 0:
            st.success("Pipeline completed.")
        elif st.session_state.run_rc is not None:
            st.error(f"Pipeline exited with code {st.session_state.run_rc}.")


# ---------------------------------------------------------------------------
# Page: Review Access
# ---------------------------------------------------------------------------

def page_review():
    st.header("Review Access")

    latest_compact = _latest_file(REVIEW_DIR, "review_compact_*.csv")
    latest_preds   = _latest_file(REVIEW_DIR, "predictions_*.csv")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Latest Batch")
        if latest_compact:
            stats = _review_stats(latest_compact)
            st.write(f"**File:** `{latest_compact.name}`")
            st.write(f"**Created:** {_mtime_str(latest_compact)}")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total docs",         stats.get("total", "—"))
            mc2.metric("Flagged for review",  stats.get("needs_review", "—"))
            mc3.metric("Codebook conflicts",  stats.get("conflicts", "—"))

            if latest_preds:
                st.caption(f"Predictions audit: `{latest_preds.name}`")
        else:
            st.info("No review batch found. Run the pipeline first.")

    with col2:
        st.subheader("Google Sheets")
        st.markdown(f"[📄 Open Spreadsheet ↗]({SHEETS_URL})")
        st.caption("Navigate to the **Review** tab in the sheet.")
        st.divider()
        st.markdown("**Reviewer instructions:**")
        st.markdown(
            "1. Open the **Review** tab above  \n"
            "2. Fill in `human_*` columns (1 / 0)  \n"
            "3. Set `review_status = reviewed`  \n"
            "4. Add notes in `reviewer_notes` if needed"
        )

    st.divider()

    # --- Merge from Sheets ---
    st.subheader("Merge Reviewed Labels")
    st.caption("Pull reviewed rows from the Sheets Review tab and merge into training data.")

    if latest_preds:
        mc1, mc2 = st.columns([3, 1])
        with mc1:
            st.write(f"Predictions file: `{latest_preds.name}`")
        with mc2:
            merge_btn = st.button("⬇  Merge from Sheets", type="primary")
        if merge_btn:
            with st.spinner("Auto-saving checkpoint before merge…"):
                auto_cp = _create_backup(label="auto-before-merge")
            st.info(f"Auto-checkpoint saved: `{auto_cp.name}`")
            cmd = [
                PYTHON, "modeling/merge_reviewed_labels.py",
                "--from-sheets",
                "--predictions", str(latest_preds),
            ]
            for key, default in [("merge_log", None), ("merge_rc", None), ("merge_cmd", None)]:
                if key not in st.session_state:
                    st.session_state[key] = default
            st.session_state.merge_cmd = " ".join(str(c) for c in cmd)
            live_box = st.empty()
            lines: list[str] = []
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            for line in proc.stdout:
                lines.append(line)
                live_box.code("".join(lines[-300:]), language="")
            proc.wait()
            st.session_state.merge_log = "".join(lines)
            st.session_state.merge_rc = proc.returncode
            live_box.empty()

    for key, default in [("merge_log", None), ("merge_rc", None), ("merge_cmd", None)]:
        if key not in st.session_state:
            st.session_state[key] = default
    if st.session_state.merge_cmd:
        st.caption(f"Last merge: `{st.session_state.merge_cmd}`")
        if st.session_state.merge_log is not None:
            st.code(st.session_state.merge_log[-8000:], language="")
        if st.session_state.merge_rc == 0:
            st.success("Merge complete.")
        elif st.session_state.merge_rc is not None:
            st.error(f"Exit code {st.session_state.merge_rc}.")
    else:
        st.info("No predictions audit file found. Run the pipeline first.")

    st.divider()

    # --- Finalize to Incidents tab ---
    st.subheader("Finalize to Incidents Tab")
    st.caption(
        "Transfers confirmed rows (review_status = 'reviewed') from the Review tab "
        "into the clean Incidents tab, stripping all pipeline columns. "
        "Run this after each merge to keep the Incidents tab up to date for the database."
    )
    fin_col1, fin_col2 = st.columns([2, 1])
    with fin_col1:
        fin_dry = st.checkbox("Dry run (preview only)", key="fin_dry")
    with fin_col2:
        fin_btn = st.button("Transfer to Incidents", type="primary")

    for key, default in [("fin_log", None), ("fin_rc", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    if fin_btn:
        cmd = [PYTHON, "data_pipeline/finalize_incidents.py"]
        if fin_dry:
            cmd.append("--dry-run")
        live_box = st.empty()
        lines: list[str] = []
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(PROJECT_ROOT),
        )
        for line in proc.stdout:
            lines.append(line)
            live_box.code("".join(lines[-200:]), language="")
        proc.wait()
        st.session_state.fin_log = "".join(lines)
        st.session_state.fin_rc  = proc.returncode
        live_box.empty()

    if st.session_state.fin_log is not None:
        st.code(st.session_state.fin_log[-6000:], language="")
        if st.session_state.fin_rc == 0:
            st.success("Transfer complete." if not fin_dry else "Dry run complete.")
        else:
            st.error(f"Exit code {st.session_state.fin_rc}.")

    st.divider()

    # --- Preview latest compact CSV ---
    if latest_compact:
        st.subheader("Batch Preview (first 50 rows)")
        try:
            df = pd.read_csv(latest_compact, encoding="utf-8-sig")
            display_cols = [
                c for c in [
                    "row_id", "date", "governorate", "predicted_labels",
                    "uncertain_labels", "needs_review", "codebook_conflict",
                ]
                if c in df.columns
            ]
            st.dataframe(df[display_cols].head(50), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Could not preview file: {exc}")


# ---------------------------------------------------------------------------
# Page: Documentation
# ---------------------------------------------------------------------------

_DOCS = [
    ("1. Project Overview", (
        "The Palestine Violence Archive is a research pipeline for a political science "
        "lab at UC San Diego. It automatically collects WAFA news articles about human "
        "rights incidents in the West Bank, classifies each incident across 10 violence "
        "labels using ML, routes uncertain cases to human reviewers, and retrains the "
        "model as labeled data accumulates. The goal is a clean, publication-ready dataset "
        "The goal is to produce a clean, labeled dataset suitable for statistical analysis "
        "in SPSS or R and for export to a permanent research database in support of "
        "peer-reviewed publication."
    )),
    ("2. Google Sheets Structure", (
        "The pipeline uses three tabs in one shared Google Spreadsheet:\n\n"
        "- **Reports** - Raw scraped articles (input layer). Written by the scraper. "
        "Columns: report_id, url, title, body_text, published_date, language, location_raw, actors_raw.\n\n"
        "- **Review** - Model predictions awaiting human verification (working layer). "
        "Written by the pipeline after each prediction run. "
        "RAs fill in the `human_*` columns here and set `review_status = reviewed`.\n\n"
        "- **Incidents** - Finalized, clean incident records (publication layer). "
        "Written by the 'Transfer to Incidents' step after human review is complete. "
        "Contains only facts and labels — no model scores or pipeline metadata. "
        "This tab mirrors what a proper database table will look like, making future "
        "migration straightforward."
    )),
    ("3. Pipeline Flow", (
        "**Step 1 — Ingest** - Scrapes new WAFA articles into the Reports tab. "
        "Duplicate URLs are skipped automatically.\n\n"
        "**Step 2 — Export** - Reads unprocessed rows from Reports and saves them as a "
        "timestamped CSV in `data/incoming/`.\n\n"
        "**Step 3 — Predict** - Runs the baseline or DeBERTa model. Produces a compact "
        "review sheet in `data/review_exports/` and pushes it to the Review tab in Sheets.\n\n"
        "**Step 4 — Human Review** - RAs open the Review tab, fill in `human_*` columns "
        "(1/0), and set `review_status = reviewed`. Double-review rows (5% random sample) "
        "are filled by a second RA for inter-rater reliability tracking.\n\n"
        "**Step 5 — Merge** - Reviewed labels are merged into `master_reviewed_dataset.csv` "
        "for model retraining.\n\n"
        "**Step 6 — Transfer to Incidents** - Clean confirmed records are written to the "
        "Incidents tab, stripping all pipeline columns. Run after each merge.\n\n"
        "**Step 7 — Retrain** - Once enough labeled data accumulates, the baseline model "
        "is rebuilt. DeBERTa can be fine-tuned locally with a GPU."
    )),
    ("4. Dashboard Pages", None),
    ("Dashboard", (
        "Live snapshot of the pipeline state:\n\n"
        "- **Incoming batch size** - Articles in `data/incoming/` not yet reviewed\n"
        "- **Review queue** - Rows in the latest sheet with `review_status != 'reviewed'`\n"
        "- **Model metrics** - Per-label F1, precision, recall from the last training run\n"
        "- **IRR panel** - Cohen's Kappa inter-rater reliability for double-reviewed rows. "
        "Kappa above 0.80 = strong agreement. Click 'Compute IRR' to refresh after each batch."
    )),
    ("Run Pipeline", (
        "Main control panel for Steps 1-3 (ingest, export, predict):\n\n"
        "- **Model** - 'baseline' is fast (CPU, runs on Streamlit Cloud). "
        "'deberta' is more accurate but requires a GPU locally.\n"
        "- **Skip ingest** - Use existing Sheets data without scraping new articles.\n"
        "- **Re-export all** - Forces re-export ignoring the already-exported tracker.\n"
        "- **Push to Sheets** - After predicting, uploads the review sheet to the Review tab "
        "so RAs can work online collaboratively.\n"
        "- **Dry run** - Prints what each step would do without writing any files. Always safe.\n"
        "- **Deduplicate** - Removes near-duplicate articles before prediction.\n"
        "- **Max rows** - Caps the batch. Recommended: 100-150 to keep reviews manageable.\n"
        "- **Log output** - Streams live and persists after the run so you can scroll it."
    )),
    ("Review Access", (
        "Access to review files and the Google Sheets Review tab:\n\n"
        "- **Google Sheets link** - Opens the spreadsheet. Direct RAs to the 'Review' tab.\n"
        "- **Latest review file** - Preview of the most recent `review_compact_*.csv`.\n"
        "- **Latest predictions** - Raw inference audit trail (all confidence scores and flags).\n"
        "- **RA workflow reminder** - Step-by-step instructions shown on-screen."
    )),
    ("Export Analysis", (
        "Exports the cleaned incident database for the professor (SPSS/R):\n\n"
        "- **Years** - Which years to include. Requires `data/processed/oppression_{year}.csv`.\n"
        "- **Output directory** - Where to write files (default: `analysis/`).\n"
        "- **Skip SPSS** - Check this if `pyreadstat` is not installed.\n"
        "- **Exported files** - `.sav` (SPSS/JASP), `.csv` (R/Excel), and "
        "`codebook_variables.csv`. The `.sav` has full variable labels and 0=No/1=Yes value labels.\n"
        "- **R usage snippet** - Copy-paste code to load in R with or without value labels.\n"
        "- **R analysis script** - `analysis/descriptive_analysis.R` produces 11 summary "
        "tables and 7 figures (run with `Rscript analysis/descriptive_analysis.R`)."
    )),
    ("Retrain (Admin)", (
        "Rebuilds the model and manages the Incidents tab. Only run after enough reviewed batches:\n\n"
        "- **Training data stats** - Current row count of `master_reviewed_dataset.csv`.\n"
        "- **Checkpoint system** - Auto-creates a timestamped backup before any destructive "
        "action. Manually create or restore checkpoints at any time.\n"
        "- **Merge reviewed labels** - Pulls reviewed rows from the Sheets Review tab and "
        "merges human labels into the master training CSV. Auto-checkpoints before merging.\n"
        "- **Transfer to Incidents** - Copies confirmed records (review_status=reviewed) from "
        "the Review tab into the clean Incidents tab, stripping all pipeline columns. "
        "Run after every merge to keep the Incidents tab current.\n"
        "- **Retrain baseline** - Rebuilds the TF-IDF + logistic regression model. 1-5 min on CPU.\n"
        "- **Retrain DeBERTa** - Fine-tunes the transformer. Requires GPU; plan 1-3 hours."
    )),
    ("5. RA Review Workflow", (
        "**Step 1** - Open the Google Sheets spreadsheet and go to the 'Review' tab. "
        "Or download the `.csv` from `data/review_exports/`.\n\n"
        "**Step 2** - For each row, read the `description` column and fill the 10 `human_*` "
        "columns with 1 (yes) or 0 (no). Leave blank if uncertain and add a `reviewer_notes` comment.\n\n"
        "**Step 3** - For rows marked `double_review = TRUE` (about 5% of each batch), a second "
        "RA should independently fill the `human2_*` columns without seeing the first RA's answers. "
        "This measures inter-rater reliability (IRR).\n\n"
        "**Step 4** - Set `review_status = reviewed` for each completed row.\n\n"
        "**Step 5** - Notify the pipeline operator. They will run Merge then Transfer to Incidents."
    )),
    ("6. Incident Labels", (
        "Each incident is classified across 10 binary labels (1 = present):\n\n"
        "- **Raid** - Organized entry by Israeli forces into a Palestinian area.\n"
        "- **Arrest/Detention** - Apprehension or detention by Israeli forces or the PA.\n"
        "- **Physical Assault** - Shooting, beating, tear gas, stun grenades, dogs.\n"
        "- **Coercive Actions** - Intimidation/harassment without direct assault.\n"
        "- **Restriction of Freedoms** - Curfews, checkpoints, roadblocks.\n"
        "- **Religious Encroachment** - Incursion into or desecration of religious sites.\n"
        "- **Harm to Property** - Vandalism or destruction of Palestinian property.\n"
        "- **Dispossession** - Seizure or theft of land, livestock, crops, or assets.\n"
        "- **Protest** - Organized Palestinian demonstration or civil resistance.\n"
        "- **Multi-Community Incident** - Coordinated action across multiple communities."
    )),
    ("7. Models", (
        "**Baseline (TF-IDF + Logistic Regression)**\n"
        "Location: `modeling/saved_models/baseline_rank_based_fixed/`\n"
        "Runs on CPU. Used by the Streamlit Cloud dashboard. Uses rank-based thresholding: "
        "predicts the top-k articles per label where k matches the training prevalence rate.\n\n"
        "**DeBERTa v3 (Transformer)**\n"
        "Location: `modeling/saved_models/deberta_v3_multilabel/`\n"
        "Weights on HuggingFace Hub (jalva182/palestine-violence-deberta). "
        "Requires GPU for practical use. Install locally with `pip install -r requirements-gpu.txt`.\n\n"
        "Both models output a confidence score (0-1) per label. Rows within 0.10 of the "
        "threshold are flagged `needs_review = TRUE` and routed to human reviewers."
    )),
    ("8. File Locations", (
        "| Path | Contents |\n"
        "|------|----------|\n"
        "| `data/incoming/` | Raw batches exported from Sheets |\n"
        "| `data/review_exports/` | Model predictions + compact review sheets |\n"
        "| `data/reviewed/` | Completed RA review files (local copies) |\n"
        "| `data/training/` | Master labeled dataset for retraining |\n"
        "| `data/processed/` | Normalized yearly CSVs (2023, 2024, 2025) |\n"
        "| `analysis/` | SPSS .sav, R CSV, codebook, R analysis script |\n"
        "| `modeling/saved_models/` | Baseline and DeBERTa artifacts |\n"
        "| `modeling/backups/` | Timestamped model checkpoints |\n"
        "| `logs/` | Pipeline run logs (pipeline_YYYYMMDD.log) |\n"
        "| `config/` | Label codebook YAML |\n"
        "| `docs/` | PDF user guide and generator script |"
    )),
    ("9. Streamlit Cloud Setup", (
        "To deploy this app on Streamlit Cloud:\n\n"
        "**1. Connect the repo** - Link your GitHub repo in share.streamlit.io. "
        "Set the main file to `streamlit_app/app.py`.\n\n"
        "**2. Add Google credentials** - Go to app Settings -> Secrets and paste your "
        "service account JSON in TOML format:\n"
        "```\n[gcp_service_account]\ntype = \"service_account\"\nproject_id = \"...\"\n"
        "private_key = \"-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n\"\n"
        "client_email = \"...@....iam.gserviceaccount.com\"\n... (all other fields)\n```\n\n"
        "**3. DeBERTa on Cloud** - The dashboard uses the baseline model by default. "
        "DeBERTa requires torch which is too large for Streamlit Cloud. "
        "Run DeBERTa locally: `pip install -r requirements-gpu.txt`.\n\n"
        "**4. Rebooting** - If you update secrets or push changes, use the three-dot menu "
        "-> Reboot app to force a full restart."
    )),
    ("10. Common Issues", (
        "**gspread not installed / No module named gspread**\n"
        "The `streamlit_app/requirements.txt` was overriding the root one. Fixed: "
        "only the root `requirements.txt` should exist.\n\n"
        "**service_account.json not found**\n"
        "Add your Google service account credentials to Streamlit Cloud Secrets "
        "under `[gcp_service_account]`. The app writes the file to disk at startup.\n\n"
        "**No articles scraped (0 new reports)**\n"
        "Check the log for robots.txt warnings. Use Skip Ingest and re-run from existing Sheets data.\n\n"
        "**100% of rows flagged for review**\n"
        "Normal for the first batch before the model is retrained on your data. "
        "Confidence improves as reviewed batches are merged and the model is retrained.\n\n"
        "**DeBERTa weights not found**\n"
        "Download from HuggingFace Hub into `modeling/saved_models/deberta_v3_multilabel/deberta_v3_best/`. "
        "Install GPU dependencies first: `pip install -r requirements-gpu.txt`.\n\n"
        "**Log disappears when clicking buttons**\n"
        "Fixed. Logs are stored in session state and persist across page interactions."
    )),
]


def page_docs():
    st.header("Documentation")
    st.caption("User guide for the dashboard, pipeline, and RA review workflow.")

    # Download PDF button
    pdf_path = PROJECT_ROOT / "docs" / "Dashboard_User_Guide.pdf"
    try:
        from docs.generate_dashboard_guide import build_pdf_bytes  # type: ignore
        pdf_bytes = build_pdf_bytes()
    except Exception:
        pdf_bytes = pdf_path.read_bytes() if pdf_path.exists() else None

    if pdf_bytes:
        st.download_button(
            label="Download PDF Guide",
            data=pdf_bytes,
            file_name="Dashboard_User_Guide.pdf",
            mime="application/pdf",
        )
    else:
        st.warning("PDF not found. Run: `python docs/generate_dashboard_guide.py`")

    st.divider()

    # Render inline docs
    for heading, body in _DOCS:
        if body is None:
            st.subheader(heading)
            continue
        # Top-level numbered sections as expanders; sub-items as plain markdown
        if heading[0].isdigit() and "." in heading[:3]:
            with st.expander(heading, expanded=True):
                st.markdown(body)
        else:
            st.markdown(f"**{heading}**")
            st.markdown(body)
            st.divider()


# ---------------------------------------------------------------------------
# Page: Export for Analysis
# ---------------------------------------------------------------------------

def page_export():
    st.header("Export for Analysis")
    st.caption(
        "Exports the cleaned incident database to SPSS (.sav) and R-ready CSV "
        "for use in SPSS, JASP, or R. Also writes a variable codebook CSV."
    )

    ANALYSIS_DIR = PROJECT_ROOT / "analysis"

    # Persist log across reruns
    for key, default in [("export_log", None), ("export_rc", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    def _run_export(cmd: list):
        st.session_state.export_log = None
        st.session_state.export_rc = None
        live_box = st.empty()
        lines: list[str] = []
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        for line in proc.stdout:
            lines.append(line)
            live_box.code("".join(lines[-200:]), language="")
        proc.wait()
        st.session_state.export_log = "".join(lines)
        st.session_state.export_rc = proc.returncode
        live_box.empty()

    # --- Options form ---
    with st.form("export_form"):
        col1, col2 = st.columns(2)
        with col1:
            years_input = st.text_input(
                "Years to include",
                value="2023 2024 2025",
                help="Space-separated list, e.g. '2023 2024 2025'",
            )
            output_dir = st.text_input(
                "Output directory",
                value=str(ANALYSIS_DIR),
                help="Relative or absolute path where files will be written",
            )
        with col2:
            no_spss = st.checkbox(
                "Skip SPSS export",
                value=False,
                help="Use this if pyreadstat is not installed",
            )

        submitted = st.form_submit_button("Export", type="primary")

    if submitted:
        years = years_input.strip().split()
        cmd = [PYTHON, "data_pipeline/export_for_analysis.py", "--output-dir", output_dir]
        if years:
            cmd += ["--years"] + years
        if no_spss:
            cmd.append("--no-spss")
        _run_export(cmd)

    # --- Persistent output ---
    if st.session_state.export_log is not None:
        st.divider()
        st.code(st.session_state.export_log[-6000:], language="")
        if st.session_state.export_rc == 0:
            st.success("Export complete.")
        else:
            st.error(f"Export failed (exit code {st.session_state.export_rc}).")

    # --- Show existing exported files ---
    st.divider()
    st.subheader("Exported Files")

    sav_files = sorted(ANALYSIS_DIR.glob("*.sav"), reverse=True) if ANALYSIS_DIR.exists() else []
    csv_files = sorted(ANALYSIS_DIR.glob("Palestine_Violence_Archive*.csv"), reverse=True) if ANALYSIS_DIR.exists() else []
    codebook  = ANALYSIS_DIR / "codebook_variables.csv"
    r_script  = ANALYSIS_DIR / "descriptive_analysis.R"

    if not sav_files and not csv_files:
        st.info("No exported files yet. Click Export above to generate them.")
    else:
        for f in sav_files:
            size_mb = round(f.stat().st_size / 1_048_576, 1)
            st.markdown(f"**{f.name}** ({size_mb} MB) — SPSS / JASP")
        for f in csv_files:
            size_kb = round(f.stat().st_size / 1024, 0)
            st.markdown(f"**{f.name}** ({size_kb} KB) — R / Excel")
        if codebook.exists():
            st.markdown(f"**{codebook.name}** — variable codebook")
        if r_script.exists():
            st.markdown(f"**{r_script.name}** — R analysis script (run in RStudio)")

    st.divider()
    st.subheader("How to open in SPSS / R")
    st.code(
        "# SPSS:\n"
        "#   File > Open > Data > Palestine_Violence_Archive_2023_2024_2025.sav\n\n"
        "# R:\n"
        "library(haven)\n"
        "df <- read_sav('analysis/Palestine_Violence_Archive_2023_2024_2025.sav')\n\n"
        "# Or plain CSV:\n"
        "df <- read.csv('analysis/Palestine_Violence_Archive_2023_2024_2025.csv')\n\n"
        "# Run full descriptive analysis:\n"
        "source('analysis/descriptive_analysis.R')",
        language="r",
    )


# ---------------------------------------------------------------------------
# Page: Retrain
# ---------------------------------------------------------------------------

def page_retrain():
    st.header("Retrain  —  Admin")
    st.warning(
        "Retraining replaces the active model weights. "
        "Only run this after enough reviewed batches have been merged into the training data.",
        icon="⚠️",
    )

    # --- Current training data stats ---
    master = TRAINING_DIR / "master_reviewed_dataset.csv"
    if master.exists():
        try:
            n_master = len(pd.read_csv(master))
            st.metric("Rows in master_reviewed_dataset.csv", n_master, help=str(master))
        except Exception:
            st.warning("Could not read master_reviewed_dataset.csv.")
    else:
        st.info("master_reviewed_dataset.csv not found. Merge reviewed batches first.")

    st.divider()

    # --- Per-label metrics (test split) ---
    st.subheader("Current Per-Label Metrics — Test Split (2025)")
    tab_bl, tab_db = st.tabs(["Baseline (TF-IDF)", "DeBERTa v3"])

    baseline_meta = _load_meta(BASELINE_META)
    deberta_meta  = _load_meta(DEBERTA_META)

    with tab_bl:
        if baseline_meta:
            df_bl = _build_metrics_table(baseline_meta, "test")
            if df_bl is not None:
                macro = baseline_meta.get("test", {}).get("macro_f1")
                micro = baseline_meta.get("test", {}).get("micro_f1")
                ci    = baseline_meta.get("bootstrap_test", {}).get("macro_f1_ci95", [None, None])
                cols  = st.columns(3)
                cols[0].metric("Macro-F1", f"{macro:.3f}" if macro else "?")
                cols[1].metric("Micro-F1", f"{micro:.3f}" if micro else "?")
                if ci[0]:
                    cols[2].metric("95% CI", f"{ci[0]:.3f} – {ci[1]:.3f}")
                st.dataframe(df_bl, use_container_width=True, hide_index=True)
            else:
                st.info("No per_label data in baseline_meta.json.")
        else:
            st.warning("baseline_meta.json not found.")

    with tab_db:
        if deberta_meta:
            df_db = _build_metrics_table(deberta_meta, "test")
            if df_db is not None:
                macro = deberta_meta.get("test", {}).get("macro_f1")
                micro = deberta_meta.get("test", {}).get("micro_f1")
                ci    = deberta_meta.get("bootstrap_test", {}).get("macro_f1_ci95", [None, None])
                cols  = st.columns(3)
                cols[0].metric("Macro-F1", f"{macro:.3f}" if macro else "?")
                cols[1].metric("Micro-F1", f"{micro:.3f}" if micro else "?")
                if ci[0]:
                    cols[2].metric("95% CI", f"{ci[0]:.3f} – {ci[1]:.3f}")
                st.dataframe(df_db, use_container_width=True, hide_index=True)
            else:
                st.info("No per_label data in deberta_v3_meta.json.")
        else:
            st.warning("deberta_v3_meta.json not found.")

    st.divider()

    # --- Checkpoint management ---
    st.subheader("Version Checkpoints")
    st.caption(
        "A checkpoint saves the current baseline model, DeBERTa meta JSON, and master "
        "training dataset. Restore any checkpoint to undo an accidental retrain."
    )

    chk_col1, chk_col2 = st.columns([2, 1])
    with chk_col1:
        chk_label = st.text_input("Checkpoint label (optional)", placeholder="e.g. before-march-retrain")
    with chk_col2:
        st.write("")
        st.write("")
        if st.button("💾  Create Checkpoint Now"):
            with st.spinner("Saving checkpoint…"):
                saved = _create_backup(label=chk_label)
            st.success(f"Checkpoint saved: `{saved.name}`")

    backups = _list_backups()
    if backups:
        st.markdown("**Existing checkpoints** (newest first):")
        for cp in backups:
            info_file = cp / "checkpoint_info.json"
            info      = json.loads(info_file.read_text()) if info_file.exists() else {}
            size_mb   = _dir_size_mb(cp)
            lbl       = f"  — *{info.get('label')}*" if info.get("label") else ""
            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.markdown(f"**{cp.name}**{lbl}  `{size_mb:.1f} MB`  ·  {len(info.get('contents', []))} files")
            with col_b:
                if st.button("↩ Restore", key=f"restore_{cp.name}"):
                    with st.spinner(f"Restoring {cp.name}…"):
                        _restore_backup(cp)
                    st.success(f"Restored from `{cp.name}`. Reload the page to see updated metrics.")
    else:
        st.info("No checkpoints yet. Create one before retraining.")

    st.divider()

    # --- Retrain button ---
    st.subheader("Run Retraining")
    dry_run = st.checkbox("Dry run (preview steps without writing)")
    for key, default in [("retrain_log", None), ("retrain_rc", None), ("retrain_cmd", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.button("🔁  Retrain Baseline", type="primary"):
        if not dry_run:
            with st.spinner("Auto-saving checkpoint before retraining…"):
                auto_cp = _create_backup(label="auto-before-retrain")
            st.info(f"Auto-checkpoint saved: `{auto_cp.name}`")
        cmd = [PYTHON, "run_pipeline.py", "--retrain-only"]
        if dry_run:
            cmd.append("--dry-run")
        st.session_state.retrain_cmd = " ".join(str(c) for c in cmd)
        live_box = st.empty()
        lines: list[str] = []
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        for line in proc.stdout:
            lines.append(line)
            live_box.code("".join(lines[-300:]), language="")
        proc.wait()
        st.session_state.retrain_log = "".join(lines)
        st.session_state.retrain_rc = proc.returncode
        live_box.empty()

    if st.session_state.retrain_cmd:
        st.caption(f"Last retrain: `{st.session_state.retrain_cmd}`")
        if st.session_state.retrain_log is not None:
            st.code(st.session_state.retrain_log[-8000:], language="")
        if st.session_state.retrain_rc == 0:
            st.success("Retraining complete. Reload the page to see updated metrics.")
        elif st.session_state.retrain_rc is not None:
            st.error(f"Retraining failed (exit code {st.session_state.retrain_rc}).")

    st.caption(
        "After retraining the baseline, run DeBERTa fine-tuning manually:  \n"
        "`python modeling/training/train_transformer_multilabel.py`"
    )


# ---------------------------------------------------------------------------
# Navigation and entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Page: Project Updates / Changelog
# ---------------------------------------------------------------------------

_UPDATES = [
    {
        "date": "2026-03-22",
        "tag": "Codebook",
        "title": "Coding Manual v1.1 — label clarifications",
        "items": [
            "'Protest' label renamed to 'Resistance and Collective Action' to better reflect the broader category.",
            "Restriction of Freedoms: checkpoints and barriers are only coded when newly installed or closed — not when a pre-existing one is mentioned in passing.",
            "Religious Encroachment vs Harm to Property boundary clarified: physical damage to a mosque or church is coded as Harm to Property; Religious Encroachment covers intrusion and interference only.",
            "Harm to Property: property searches with no resulting damage are coded as Coercive Actions (Intimidation) only — not Harm to Property.",
            "Dispossession: intentional killing of livestock confirmed as Dispossession; Attempted Theft/Seizure sub-type added.",
            "Multi-Community Incident: Masafer Yatta sub-communities always coded as a single 'Masafer Yatta (Community)' entry, not as a multi-community incident.",
            "Stabbing added to Physical Assault sub-types.",
            "Place name standardized spellings added (1/16/2026 meeting): ~20 canonical forms with known variant aliases.",
        ],
    },
    {
        "date": "2026-03-21",
        "tag": "Pipeline",
        "title": "Incidents tab repurposed as publication-layer database",
        "items": [
            "The Incidents tab is now the clean, human-confirmed record store — a publication-ready database stripped of all pipeline internals (no pred_*, conf_*, or flag columns).",
            "New script: data_pipeline/finalize_incidents.py promotes reviewed rows from the Review tab to the Incidents tab.",
            "Supports --dry-run flag to preview what will be written before committing.",
            "Finalization also available from the dashboard: Retrain page > Finalize to Incidents Tab.",
            "Label resolution: uses human override if filled; falls back to model prediction.",
        ],
    },
    {
        "date": "2026-03-21",
        "tag": "Dashboard",
        "title": "Export Analysis and Documentation pages added",
        "items": [
            "New Export Analysis page: exports the Incidents tab to SPSS (.sav) and R-ready CSV for academic statistical analysis.",
            "New Documentation page: full inline user guide with downloadable PDF (Dashboard_User_Guide.pdf).",
            "PDF guide generated by docs/generate_dashboard_guide.py using fpdf2.",
        ],
    },
    {
        "date": "2026-03-21",
        "tag": "Infrastructure",
        "title": "Streamlit Cloud deployment stabilized",
        "items": [
            "Removed streamlit_app/requirements.txt — it was shadowing root requirements.txt on Streamlit Cloud, causing gspread and other packages to not install.",
            "Removed torch, transformers, safetensors from requirements.txt (800MB+, caused pip timeout on Streamlit Cloud). Moved to requirements-gpu.txt for local/GPU use only.",
            "Added _ensure_service_account() to app.py: reads Google credentials from Streamlit Secrets at startup and writes service_account.json to disk so all pipeline subprocesses can authenticate.",
            "sheets_interface.py now checks GOOGLE_APPLICATION_CREDENTIALS env var before falling back to service_account.json.",
        ],
    },
    {
        "date": "2026-03-21",
        "tag": "Google Sheets",
        "title": "Google Sheets column names corrected",
        "items": [
            "Fixed column name mismatches in sheets_interface.py: create_incident(), get_next_sequence(), and push_review_batch() were referencing stale column names that no longer exist in the live spreadsheet.",
            "Report ID generation fixed: now uses max existing ID instead of row count, preventing ID collisions when rows are deleted.",
            "MAX_ID_RETRIES raised from 5 to 20.",
            "check_sheets_structure.py added: diagnostic script that connects to Sheets and prints all tab names, column headers, row counts, and sample rows.",
        ],
    },
    {
        "date": "2026-03-21",
        "tag": "Analysis",
        "title": "Statistical analysis exports added",
        "items": [
            "New script: data_pipeline/export_for_analysis.py exports the Incidents tab to SPSS (.sav) and R (.rds) formats.",
            "New script: analysis/compute_irr.py computes inter-rater reliability (Cohen's kappa) across all 10 labels.",
            "New R script: analysis/descriptive_analysis.R for descriptive statistics.",
            "README updated to reflect all new scripts, the three-tab Sheets structure, requirements-gpu.txt, and Streamlit Cloud setup.",
        ],
    },
]

_TAG_COLORS = {
    "Codebook":       "#6c63ff",
    "Pipeline":       "#0ea5e9",
    "Dashboard":      "#10b981",
    "Infrastructure": "#f59e0b",
    "Google Sheets":  "#ef4444",
    "Analysis":       "#8b5cf6",
}


def page_updates():
    st.header("Project Updates")
    st.caption("Changelog of pipeline, dashboard, and codebook changes.")

    for entry in _UPDATES:
        color = _TAG_COLORS.get(entry["tag"], "#888")
        tag_badge = (
            f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:0.75rem;font-weight:600;">'
            f'{entry["tag"]}</span>'
        )
        with st.expander(
            f"{entry['date']}  —  {entry['title']}",
            expanded=False,
        ):
            st.markdown(tag_badge, unsafe_allow_html=True)
            st.markdown("")
            for item in entry["items"]:
                st.markdown(f"- {item}")


PAGES = {
    "📊 Dashboard":        page_dashboard,
    "▶ Run Pipeline":      page_run,
    "📋 Review Access":    page_review,
    "📤 Export Analysis":  page_export,
    "🔁 Retrain":          page_retrain,
    "📖 Documentation":    page_docs,
    "📝 Updates":          page_updates,
}


def main():
    st.sidebar.title("Violence Archive")
    st.sidebar.caption("Lab Pipeline Dashboard · v1")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Project: `{PROJECT_ROOT.name}`")
    st.sidebar.caption(f"Python: `{Path(PYTHON).name}`")

    PAGES[page]()


if __name__ == "__main__":
    main()
