"""
modeling/predict_and_prepare_review.py

Human-in-the-loop inference pipeline for the Oppression Database.

Takes new documents (CSV or JSON), runs the trained multilabel classifier,
applies per-label thresholds, computes uncertainty flags, validates predictions
against the codebook, and exports reviewer-ready CSVs for research assistants.

Usage:
    python modeling/predict_and_prepare_review.py \\
        --input data/raw/new_documents.csv \\
        --model deberta \\
        --output-dir review_output/ \\
        --codebook config/label_codebook.yaml \\
        --confidence-band 0.02   # optional: overrides model-specific default

Outputs (in --output-dir, all timestamped):
    review_{ts}.csv          -- Full traceability sheet (all probs, flags, metadata)
    review_compact_{ts}.csv  -- Slim RA workflow sheet (predicted + uncertain label summaries)
    predictions_{ts}.csv     -- Raw probabilities + thresholds audit trail

Requirements:
    pip install pandas numpy pyyaml scikit-learn
    pip install torch transformers    # only needed for --model deberta
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Project root and path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Labels in the exact order stored in the model metadata JSON files.
# This order is validated against model metadata at load time — do not reorder.
LABELS: List[str] = [
    "raid_y",
    "arrest_detention_y",
    "physical_assault_y",
    "harm_to_property_y",
    "dispossession_y",
    "religious_encroachment_y",
    "restriction_of_freedoms_y",
    "coercive_actions_y",
    "protest_y",
    "multi_community_incident_y",
]

# Short names used in output column names (pred_raid, conf_raid, etc.)
LABEL_SHORT: Dict[str, str] = {
    "raid_y":                      "raid",
    "arrest_detention_y":          "arrest_detention",
    "physical_assault_y":          "physical_assault",
    "harm_to_property_y":          "harm_to_property",
    "dispossession_y":             "dispossession",
    "religious_encroachment_y":    "religious_encroachment",
    "restriction_of_freedoms_y":   "restriction_of_freedoms",
    "coercive_actions_y":          "coercive_actions",
    "protest_y":                   "protest",
    "multi_community_incident_y":  "multi_community_incident",
}

# Metadata columns used to build the enhanced text input (must match training).
META_COLS = ["area", "governorate", "location_type", "perpetrator"]

# Default paths (relative to project root)
BASELINE_MODEL_DIR    = PROJECT_ROOT / "modeling" / "saved_models" / "baseline_rank_based_fixed"
DEBERTA_MODEL_DIR     = PROJECT_ROOT / "modeling" / "saved_models" / "deberta_v3_multilabel" / "deberta_v3_best"
DEBERTA_HF_REPO       = "jalva182/palestine-violence-deberta"  # HuggingFace Hub (private)
DEBERTA_META_PATH     = PROJECT_ROOT / "modeling" / "saved_models" / "deberta_v3_multilabel" / "deberta_v3_meta.json"
DEFAULT_CODEBOOK_PATH = PROJECT_ROOT / "config" / "label_codebook.yaml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ===========================================================================
# Section B — Codebook loading + derived helpers
# ===========================================================================

def load_codebook(path: Path) -> dict:
    """
    Load the label codebook from a YAML file.

    Returns the full dict. Caller accesses label definitions via result["labels"].
    Raises FileNotFoundError with a helpful message if the path is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Codebook not found at: {path}\n"
            f"Expected location: {DEFAULT_CODEBOOK_PATH}\n"
            f"Use --codebook <path> to specify a different location."
        )
    with open(path, "r", encoding="utf-8") as f:
        codebook = yaml.safe_load(f)
    log.info(
        "Loaded codebook v%s from %s",
        codebook.get("metadata", {}).get("version", "?"),
        path,
    )
    return codebook


def get_rare_labels(codebook: dict) -> Set[str]:
    """
    Derive the set of rare label keys from the codebook YAML (is_rare: true).

    Uses the codebook as the single source of truth rather than a hardcoded
    constant, so rarity classification can be updated in one place.
    """
    return {
        label_key
        for label_key, info in codebook.get("labels", {}).items()
        if info.get("is_rare", False)
    }


# ===========================================================================
# Section C — Baseline model loading
# ===========================================================================

def load_baseline_model(model_dir: Path) -> Tuple[object, object, dict]:
    """
    Load the TF-IDF + LogisticRegression baseline model.

    Returns:
        model       -- fitted MultiOutputClassifier (sklearn)
        vectorizer  -- fitted TfidfVectorizer (sklearn)
        meta        -- dict from baseline_meta.json (contains target_rates, labels)
    """
    model_path = model_dir / "baseline_model.pkl"
    vec_path   = model_dir / "tfidf_vectorizer.pkl"
    meta_path  = model_dir / "baseline_meta.json"

    for p in [model_path, vec_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Baseline model artifact not found: {p}")

    log.info("Loading baseline model from %s ...", model_dir)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    log.info(
        "Baseline model loaded. Test macro-F1=%.3f (2025 holdout)",
        meta.get("test", {}).get("macro_f1", float("nan")),
    )
    return model, vectorizer, meta


# ===========================================================================
# Section D — DeBERTa model loading
# ===========================================================================

def load_deberta_model(model_dir: Path) -> Tuple[object, object, dict, str]:
    """
    Load the DeBERTa v3 multilabel transformer model.

    Returns:
        model     -- HuggingFace AutoModelForSequenceClassification (eval mode)
        tokenizer -- HuggingFace AutoTokenizer
        meta      -- dict from deberta_v3_meta.json (contains thresholds, labels)
        device    -- "cuda" or "cpu"
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError(
            "torch and transformers are required for the DeBERTa model.\n"
            "Install with: pip install torch transformers"
        )

    if not DEBERTA_META_PATH.exists():
        raise FileNotFoundError(f"DeBERTa metadata not found: {DEBERTA_META_PATH}")

    with open(DEBERTA_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    # Load from local directory if it exists, otherwise fall back to HF Hub.
    # HF Hub weights are cached after the first download (~738 MB, one-time).
    if model_dir.exists():
        model_source = str(model_dir)
        log.info("Loading DeBERTa from local path: %s", model_dir)
    else:
        model_source = DEBERTA_HF_REPO
        log.info("Local model not found — loading DeBERTa from HF Hub: %s", DEBERTA_HF_REPO)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModelForSequenceClassification.from_pretrained(model_source)
    except OSError as exc:
        if model_source == DEBERTA_HF_REPO:
            raise OSError(
                f"Could not load DeBERTa from HF Hub ({DEBERTA_HF_REPO}).\n"
                "If this is a private repo, authenticate first:\n"
                "  huggingface-cli login\n"
                "or set the HF_TOKEN environment variable."
            ) from exc
        raise
    model.to(device)
    model.eval()

    log.info(
        "DeBERTa model loaded. Test macro-F1=%.3f (2025 holdout)",
        meta.get("test", {}).get("macro_f1", float("nan")),
    )
    return model, tokenizer, meta, device


# ===========================================================================
# Section D2 — Label order validation
# ===========================================================================

def validate_label_order(meta: dict, model_name: str) -> None:
    """
    Verify that the label order in the model metadata exactly matches LABELS.

    The entire pipeline (probs[:,i], thresholds[i], preds[:,i]) assumes a fixed
    label-to-column mapping. If metadata order ever drifts from LABELS, all
    downstream predictions would be silently wrong. This check fails loudly.

    Raises ValueError if there is any mismatch.
    """
    meta_labels = meta.get("labels", [])
    if meta_labels != LABELS:
        raise ValueError(
            f"Label order mismatch in {model_name} metadata!\n"
            f"  Script LABELS:    {LABELS}\n"
            f"  Metadata labels:  {meta_labels}\n"
            f"Update the LABELS constant in this script to match the saved model."
        )
    log.info("Label order validated against %s metadata: OK", model_name)


# ===========================================================================
# Section E — Text preparation (must exactly match training)
# ===========================================================================

def build_text_enhanced(df: pd.DataFrame) -> pd.Series:
    """
    Build enhanced text input from description + metadata columns.

    Replicates the exact format used during model training:
        [AREA] {area} [GOV] {governorate} [LOC_TYPE] {location_type} [PERP] {perpetrator} [DESC] {description}

    Source: modeling/training/train_baseline_multilabel.py:369-386

    Any missing metadata column is filled with an empty string (with a warning).
    """
    for col in META_COLS:
        if col not in df.columns:
            log.warning("Column '%s' not found in input — filling with empty string.", col)
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    if "description" not in df.columns:
        raise ValueError("Input data must have a 'description' column.")
    df["description"] = df["description"].fillna("").astype(str)

    return (
        "[AREA] "        + df["area"]
        + " [GOV] "      + df["governorate"]
        + " [LOC_TYPE] " + df["location_type"]
        + " [PERP] "     + df["perpetrator"]
        + " [DESC] "     + df["description"]
    )


# ===========================================================================
# Section F — Baseline inference (rank-based thresholding)
# ===========================================================================

def run_baseline_inference(
    model, vectorizer, meta: dict, texts: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference with the TF-IDF + LogisticRegression baseline.

    Prediction method: rank-based. For each label, the top-k documents by
    probability are predicted positive, where k = round(target_rate * n_docs).
    This matches the evaluation strategy used during training.

    An implied per-label probability threshold is derived from the rank cut
    (the k-th highest probability), enabling downstream uncertainty flagging.

    Returns:
        probs      -- shape (n_docs, 10), raw LR probabilities per label
        preds      -- shape (n_docs, 10), binary predictions (0/1)
        thresholds -- shape (10,), implied probability threshold per label
    """
    log.info("Vectorizing %d documents ...", len(texts))
    X = vectorizer.transform(texts)

    n_docs   = len(texts)
    n_labels = len(LABELS)
    probs    = np.zeros((n_docs, n_labels), dtype=np.float32)

    # Extract raw probabilities from each per-label LogisticRegression estimator
    for i, estimator in enumerate(model.estimators_):
        # predict_proba returns [prob_neg, prob_pos]; we keep prob_pos
        probs[:, i] = estimator.predict_proba(X)[:, 1]

    # Target rates determine how many documents to label positive per label
    target_rates: dict = meta["target_rates"]["target_test"]
    preds      = np.zeros((n_docs, n_labels), dtype=np.int8)
    thresholds = np.zeros(n_labels, dtype=np.float32)

    for i, label in enumerate(LABELS):
        target_rate  = target_rates.get(label, 0.1)
        k            = max(1, int(round(target_rate * n_docs)))
        sorted_probs = np.sort(probs[:, i])[::-1]
        threshold_i  = float(sorted_probs[k - 1])
        thresholds[i] = threshold_i
        preds[:, i]   = (probs[:, i] >= threshold_i).astype(np.int8)

    log.info("Baseline inference complete. Mean positive rate: %.3f", preds.mean())
    return probs, preds, thresholds


# ===========================================================================
# Section G — DeBERTa inference (threshold-based)
# ===========================================================================

def run_deberta_inference(
    model,
    tokenizer,
    meta: dict,
    texts: pd.Series,
    device: str,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference with the DeBERTa v3 transformer model.

    Prediction method: fixed per-label probability thresholds from training metadata.
    Probabilities are obtained via sigmoid on the model's raw logits.

    Args:
        batch_size: Documents per forward pass. Reduce if GPU runs out of memory.

    Returns:
        probs      -- shape (n_docs, 10), sigmoid probabilities per label
        preds      -- shape (n_docs, 10), binary predictions (0/1)
        thresholds -- shape (10,), per-label thresholds from training metadata
    """
    import torch

    thresholds   = np.array(meta["thresholds"], dtype=np.float32)
    max_len: int = meta.get("max_len", 384)
    n_docs       = len(texts)
    all_probs    = []
    texts_list   = texts.tolist()

    log.info(
        "Running DeBERTa inference on %d documents (batch_size=%d) ...",
        n_docs, batch_size,
    )

    with torch.no_grad():
        for start in range(0, n_docs, batch_size):
            batch_texts = texts_list[start : start + batch_size]
            encoding = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            encoding    = {k: v.to(device) for k, v in encoding.items()}
            logits      = model(**encoding).logits
            probs_batch = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            all_probs.append(probs_batch)
            if (start // batch_size) % 10 == 0:
                log.info(
                    "  Processed %d / %d documents",
                    min(start + batch_size, n_docs), n_docs,
                )

    probs = np.vstack(all_probs)
    preds = (probs >= thresholds).astype(np.int8)

    log.info("DeBERTa inference complete. Mean positive rate: %.3f", preds.mean())
    return probs, preds, thresholds


# ===========================================================================
# Section H — Uncertainty flagging
# ===========================================================================

def compute_uncertainty_flags(
    probs: np.ndarray,
    preds: np.ndarray,
    thresholds: np.ndarray,
    rare_labels: Set[str],
    confidence_band: float = 0.10,
) -> pd.DataFrame:
    """
    Compute per-label uncertainty flags and a preliminary needs_review flag.

    A label is flagged when the model's probability is within `confidence_band`
    of the decision threshold — i.e. the model is not clearly on one side.

    Rare labels (is_rare: true in codebook) always trigger needs_review when
    predicted positive, regardless of confidence margin.

    Note: codebook conflicts also promote needs_review, but that update is
    applied in main() after check_codebook_constraints() has run.

    Args:
        probs:            shape (n_docs, 10), raw probabilities
        preds:            shape (n_docs, 10), binary predictions
        thresholds:       shape (10,), per-label decision thresholds
        rare_labels:      set of label keys with is_rare=true (from codebook)
        confidence_band:  margin around threshold to flag (default 0.10)

    Returns:
        DataFrame with flag_{short} (bool) × 10 and needs_review (bool).
    """
    n_docs       = len(probs)
    flag_data: dict = {}
    rare_indices = [i for i, label in enumerate(LABELS) if label in rare_labels]

    for i, label in enumerate(LABELS):
        short  = LABEL_SHORT[label]
        margin = np.abs(probs[:, i] - thresholds[i])
        flag_data[f"flag_{short}"] = margin < confidence_band

    flags_df = pd.DataFrame(flag_data)

    # needs_review: any label is borderline OR any rare label is predicted positive
    any_flag      = flags_df.values.any(axis=1)
    rare_positive = (
        preds[:, rare_indices].any(axis=1).astype(bool)
        if rare_indices
        else np.zeros(n_docs, dtype=bool)
    )
    flags_df["needs_review"] = any_flag | rare_positive

    log.info(
        "Uncertainty flags: %d / %d documents flagged for review "
        "(confidence_band=%.2f, before codebook conflicts)",
        int(flags_df["needs_review"].sum()), n_docs, confidence_band,
    )
    return flags_df


# ===========================================================================
# Section I — Codebook constraint checking
# ===========================================================================

def check_codebook_constraints(preds: np.ndarray, codebook: dict) -> List[str]:
    """
    Check predictions against soft codebook constraints.

    A constraint fires when all flag_if_predicted conditions match the predicted
    labels for a document. These are warnings, not hard rejections — they surface
    anomalous label combinations for human review.

    Returns:
        List of strings (one per document). Empty string = no conflicts.
        Multiple conflicts on the same document are joined with " | ".
    """
    labels_info  = codebook.get("labels", {})
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    n_docs       = len(preds)
    conflicts    = [""] * n_docs

    for label_key, label_info in labels_info.items():
        for constraint in label_info.get("constraints", []):
            flag_if     = constraint.get("flag_if_predicted", {})
            description = constraint.get("description", "Codebook constraint violated")

            # Build boolean mask: documents matching ALL conditions in flag_if_predicted
            mask = np.ones(n_docs, dtype=bool)
            for col_name, expected_val in flag_if.items():
                if col_name not in label_to_idx:
                    continue
                mask &= preds[:, label_to_idx[col_name]] == expected_val

            for doc_i in np.where(mask)[0]:
                sep = " | " if conflicts[doc_i] else ""
                conflicts[doc_i] += sep + description

    n_conflicts = sum(1 for c in conflicts if c)
    if n_conflicts:
        log.info("Codebook conflicts found in %d / %d documents.", n_conflicts, n_docs)
    return conflicts


# ===========================================================================
# Section J — Format full traceability reviewer sheet
# ===========================================================================

def _build_label_definitions_json(codebook: dict) -> str:
    """
    Build a JSON string of label name, definition, and a concise
    inclusion/exclusion summary for the label_definitions column.

    RAs can read any cell in this column as a quick in-sheet reference.
    Inclusion/exclusion lists are trimmed to the first 3 / 2 items to keep
    the cell readable without losing core guidance.
    """
    labels_info = codebook.get("labels", {})
    defs = {}
    for label_key, info in labels_info.items():
        defs[label_key] = {
            "name":       info.get("name", label_key),
            "definition": (info.get("definition") or "").strip(),
            "include_if": (info.get("inclusion_criteria") or [])[:3],
            "exclude_if": (info.get("exclusion_criteria") or [])[:2],
        }
    return json.dumps(defs, ensure_ascii=False)


def format_full_output(
    input_df: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
    flags_df: pd.DataFrame,
    conflicts: List[str],
    codebook: dict,
    model_name: str,
    run_ts: str,
) -> pd.DataFrame:
    """
    Build the full traceability reviewer sheet.

    Column groups:
        1.  Run metadata        (model_used, run_timestamp)
        2.  Document metadata   (row_id, id, date, governorate, area,
                                 location_name, location_type, perpetrator)
        3.  Description         (description)
        4.  Predictions         (pred_{label} × 10)
        5.  Confidence scores   (conf_{label} × 10, 3 decimal places)
        6.  Uncertainty flags   (flag_{label} × 10, needs_review)
        7.  Codebook warnings   (codebook_conflict)
        8.  Human review cols   (human_{label} × 10)  ← filled by RA
        9.  RA workflow cols    (reviewer_notes, review_status, reviewed_by)
        10. Codebook reference  (label_definitions JSON)
    """
    n_docs = len(input_df)
    out    = pd.DataFrame(index=range(n_docs))

    # 1. Run metadata
    out["model_used"]    = model_name
    out["run_timestamp"] = run_ts

    # 2. Document metadata
    for col in ["row_id", "id", "date", "governorate", "area",
                "location_name", "location_type", "perpetrator"]:
        out[col] = input_df[col].values if col in input_df.columns else ""
    # source_name: prefer source_2 (human-readable name), fall back to source_1 (URL)
    if "source_2" in input_df.columns:
        out["source_name"] = input_df["source_2"].fillna("").values
    elif "source_1" in input_df.columns:
        out["source_name"] = input_df["source_1"].fillna("").values
    else:
        out["source_name"] = ""

    # 3. Description
    out["description"] = input_df["description"].values if "description" in input_df.columns else ""

    # 4. Model predictions (0/1)
    for i, label in enumerate(LABELS):
        out[f"pred_{LABEL_SHORT[label]}"] = preds[:, i].astype(int)

    # 5. Confidence scores (3 decimal places)
    for i, label in enumerate(LABELS):
        out[f"conf_{LABEL_SHORT[label]}"] = np.round(probs[:, i].astype(float), 3)

    # 6. Uncertainty flags
    for label in LABELS:
        short = LABEL_SHORT[label]
        out[f"flag_{short}"] = flags_df[f"flag_{short}"].values
    out["needs_review"] = flags_df["needs_review"].values

    # 7. Codebook conflict warnings
    out["codebook_conflict"] = conflicts

    # 8. Human review columns (empty — filled by RA)
    for label in LABELS:
        out[f"human_{LABEL_SHORT[label]}"] = ""

    # 9. RA workflow columns
    out["reviewer_notes"] = ""
    out["review_status"]  = "pending"
    out["reviewed_by"]    = ""

    # 10. Codebook reference (same JSON in every row for in-sheet lookup)
    out["label_definitions"] = _build_label_definitions_json(codebook)

    return out


# ===========================================================================
# Section J2 — Format compact RA workflow sheet
# ===========================================================================

def format_compact_output(
    input_df: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
    flags_df: pd.DataFrame,
    conflicts: List[str],
    codebook: dict,
    model_name: str,
    run_ts: str,
    overlap_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Build the slim RA workflow sheet for day-to-day review.

    Replaces the 30 individual pred/conf/flag columns with two readable
    summary columns, making the sheet much easier to scan in Excel or Sheets.

    Column groups:
        1. Run metadata        (model_used, run_timestamp)
        2. Document metadata   (row_id, id, date, governorate, area,
                                location_name, location_type, perpetrator)
        3. Description         (description)
        4. Prediction summary  (predicted_labels, uncertain_labels)
        5. Review triggers     (needs_review, codebook_conflict)
        6. Human review cols   (human_{label} × 10)  ← filled by RA
        7. RA workflow cols    (reviewer_notes, review_status, reviewed_by)

    predicted_labels:  comma-separated human-readable names of positive predictions
                       e.g. "Raid, Arrest/Detention, Coercive Actions"
    uncertain_labels:  comma-separated human-readable names of borderline labels
                       e.g. "Coercive Actions, Restriction of Freedoms"
    """
    n_docs = len(input_df)
    out    = pd.DataFrame(index=range(n_docs))

    # Human-readable label names from codebook (fallback to short name)
    cb_labels = codebook.get("labels", {})
    label_display: Dict[str, str] = {
        label: cb_labels.get(label, {}).get("name", LABEL_SHORT[label])
        for label in LABELS
    }

    # 1. Run metadata
    out["model_used"]    = model_name
    out["run_timestamp"] = run_ts

    # 2. Document metadata
    for col in ["row_id", "id", "date", "governorate", "area",
                "location_name", "location_type", "perpetrator"]:
        out[col] = input_df[col].values if col in input_df.columns else ""
    # source_name: prefer source_2 (human-readable name), fall back to source_1 (URL)
    if "source_2" in input_df.columns:
        out["source_name"] = input_df["source_2"].fillna("").values
    elif "source_1" in input_df.columns:
        out["source_name"] = input_df["source_1"].fillna("").values
    else:
        out["source_name"] = ""

    # 3. Description
    out["description"] = input_df["description"].values if "description" in input_df.columns else ""

    # 4. Prediction summary columns
    predicted_labels_col: List[str] = []
    uncertain_labels_col: List[str] = []

    for doc_i in range(n_docs):
        pos_names = [
            label_display[label]
            for j, label in enumerate(LABELS)
            if preds[doc_i, j] == 1
        ]
        uncertain_names = [
            label_display[label]
            for j, label in enumerate(LABELS)
            if flags_df[f"flag_{LABEL_SHORT[label]}"].iloc[doc_i] and preds[doc_i, j] == 1
        ]
        predicted_labels_col.append(", ".join(pos_names) if pos_names else "(none)")
        uncertain_labels_col.append(", ".join(uncertain_names) if uncertain_names else "")

    out["predicted_labels"] = predicted_labels_col
    out["uncertain_labels"] = uncertain_labels_col

    # 5. Review triggers
    out["needs_review"]      = flags_df["needs_review"].values
    out["codebook_conflict"] = conflicts

    # 6. Human review columns (empty — filled by RA)
    for label in LABELS:
        out[f"human_{LABEL_SHORT[label]}"] = ""

    # 7. RA workflow columns
    out["reviewer_notes"] = ""
    out["review_status"]  = "pending"
    out["reviewed_by"]    = ""

    # 8. Inter-rater reliability columns
    # Randomly select overlap_pct of rows for double-review.
    # These rows are reviewed independently by a second RA to measure agreement.
    rng = np.random.default_rng(seed=42)
    n_overlap = max(1, int(round(overlap_pct * n_docs))) if n_docs > 0 else 0
    overlap_idx = set(rng.choice(n_docs, size=min(n_overlap, n_docs), replace=False).tolist())
    out["double_review"] = [i in overlap_idx for i in range(n_docs)]

    # Empty columns for the second rater (only meaningful on double_review=True rows)
    for label in LABELS:
        out[f"human2_{LABEL_SHORT[label]}"] = ""
    out["reviewed_by_2"] = ""

    return out


# ===========================================================================
# Section K — Export outputs
# ===========================================================================

def export_outputs(
    full_df: pd.DataFrame,
    compact_df: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
    thresholds: np.ndarray,
    input_df: pd.DataFrame,
    output_dir: Path,
    run_ts: str,
) -> None:
    """
    Write all three output files to output_dir.

    Files:
        review_{ts}.csv          -- Full traceability sheet (UTF-8 BOM for Excel)
        review_compact_{ts}.csv  -- Slim RA workflow sheet (UTF-8 BOM for Excel)
        predictions_{ts}.csv     -- Raw probabilities + thresholds audit trail
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full traceability sheet (UTF-8 BOM ensures correct Excel rendering on Windows)
    full_path = output_dir / f"review_{run_ts}.csv"
    full_df.to_csv(full_path, index=False, encoding="utf-8-sig")
    log.info(
        "Full reviewer sheet   → %s  (%d rows, %d cols)",
        full_path, len(full_df), len(full_df.columns),
    )

    # Compact RA sheet
    compact_path = output_dir / f"review_compact_{run_ts}.csv"
    compact_df.to_csv(compact_path, index=False, encoding="utf-8-sig")
    log.info(
        "Compact review sheet  → %s  (%d rows, %d cols)",
        compact_path, len(compact_df), len(compact_df.columns),
    )

    # Audit trail: raw probabilities + thresholds used + binary predictions
    id_vals     = input_df["id"].values     if "id"     in input_df.columns else np.arange(len(input_df)).astype(str)
    row_id_vals = input_df["row_id"].values if "row_id" in input_df.columns else np.arange(len(input_df)).astype(str)

    audit_df = pd.DataFrame({"row_id": row_id_vals, "id": id_vals})
    for i, label in enumerate(LABELS):
        short = LABEL_SHORT[label]
        audit_df[f"prob_{short}"]      = np.round(probs[:, i].astype(float), 6)
        audit_df[f"threshold_{short}"] = round(float(thresholds[i]), 6)
        audit_df[f"pred_{short}"]      = preds[:, i].astype(int)

    audit_path = output_dir / f"predictions_{run_ts}.csv"
    audit_df.to_csv(audit_path, index=False, encoding="utf-8")
    log.info("Predictions audit log → %s", audit_path)


# ===========================================================================
# Summary printer
# ===========================================================================

def print_summary(
    input_df: pd.DataFrame,
    preds: np.ndarray,
    flags_df: pd.DataFrame,
    conflicts: List[str],
    rare_labels: Set[str],
    model_name: str,
    output_dir: Path,
    run_ts: str,
) -> None:
    """Print a plain-text run summary to stdout."""
    n_docs     = len(input_df)
    n_review   = int(flags_df["needs_review"].sum())
    n_conflict = sum(1 for c in conflicts if c)

    print("\n" + "=" * 64)
    print(f"  INFERENCE SUMMARY — {model_name.upper()}")
    print("=" * 64)
    print(f"  Documents processed : {n_docs}")
    print(f"  Flagged for review  : {n_review}  ({100 * n_review / max(n_docs, 1):.1f}%)")
    print(f"  Codebook conflicts  : {n_conflict}  ({100 * n_conflict / max(n_docs, 1):.1f}%)")
    print()
    print("  Predicted positive rate per label:")
    for i, label in enumerate(LABELS):
        rate      = preds[:, i].mean()
        rare_mark = " [RARE — always reviewed]" if label in rare_labels else ""
        print(f"    {LABEL_SHORT[label]:32s} {rate * 100:5.1f}%{rare_mark}")
    print()
    print(f"  Output directory: {output_dir}")
    print(f"    review_{run_ts}.csv")
    print(f"    review_compact_{run_ts}.csv")
    print(f"    predictions_{run_ts}.csv")
    print("=" * 64 + "\n")


# ===========================================================================
# Section L — Input loading
# ===========================================================================

def load_input_data(input_path: Path) -> pd.DataFrame:
    """
    Load input documents from a CSV or JSON file.

    JSON files are loaded with json.load() and support two shapes:
        - A list of record dicts:   [{...}, {...}, ...]
        - A wrapped dict format:    {"documents": [{...}, ...]}
          (or any top-level key whose value is a list of dicts)

    CSV files are loaded with pandas read_csv().
    All values are coerced to strings; NaN is replaced with "".
    """
    ext = input_path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(input_path, dtype=str).fillna("")
        log.info("Loaded %d rows from CSV: %s", len(df), input_path)
        return df

    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            # Find the first key whose value is a list (e.g. "documents", "data", "records")
            list_values = [(k, v) for k, v in raw.items() if isinstance(v, list)]
            if not list_values:
                raise ValueError(
                    f"JSON file {input_path} is a dict but contains no list values.\n"
                    f"Expected a list of records or a dict with a list key "
                    f'(e.g. {{"documents": [...]}}).\n'
                    f"Found keys: {list(raw.keys())}"
                )
            key, records = list_values[0]
            if len(list_values) > 1:
                log.warning(
                    "JSON file has multiple list keys: %s. Using '%s'.",
                    [k for k, _ in list_values], key,
                )
        else:
            raise ValueError(
                f"Unexpected JSON structure in {input_path}. "
                f"Expected a list of records or a dict with a list key."
            )

        df = pd.DataFrame(records).fillna("").astype(str)
        log.info("Loaded %d rows from JSON: %s", len(df), input_path)
        return df

    raise ValueError(
        f"Unsupported file format '{ext}'. Provide a .csv or .json file."
    )


# ===========================================================================
# Section M — Main entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multilabel inference and prepare human review sheets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV or JSON file containing documents to classify.",
    )
    parser.add_argument(
        "--model", "-m",
        choices=["baseline", "deberta"],
        default="deberta",
        help="Which model to use for inference. Default: deberta.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(PROJECT_ROOT / "data" / "review_exports"),
        help="Directory to write output files. Default: data/review_exports/.",
    )
    parser.add_argument(
        "--codebook",
        default=str(DEFAULT_CODEBOOK_PATH),
        help="Path to label_codebook.yaml. Default: config/label_codebook.yaml.",
    )
    parser.add_argument(
        "--confidence-band",
        type=float,
        default=None,
        help=(
            "Override the uncertainty margin around the decision threshold. "
            "A label is flagged when |prob - threshold| < confidence_band. "
            "Defaults: baseline=0.02, deberta=0.05 (provisional, tune after checkpoint loads). "
            "Use this flag for experiments."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for DeBERTa inference. Reduce if GPU runs out of memory. Default: 16.",
    )
    parser.add_argument(
        "--push-to-sheets",
        action="store_true",
        help=(
            "After exporting local files, also push the compact review sheet to the "
            "Google Sheets 'Review' tab. Row IDs already present in Sheets are skipped."
        ),
    )
    parser.add_argument(
        "--overlap-pct",
        type=float,
        default=0.05,
        help=(
            "Fraction of rows to mark for double-review (inter-rater reliability). "
            "These rows get a double_review=TRUE flag and empty human2_* columns "
            "for a second rater. Default: 0.05 (5%%)."
        ),
    )
    args = parser.parse_args()

    input_path    = Path(args.input)
    output_dir    = Path(args.output_dir)
    codebook_path = Path(args.codebook)
    run_ts        = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Load codebook (single source of truth for label semantics and rarity) ---
    codebook    = load_codebook(codebook_path)
    rare_labels = get_rare_labels(codebook)
    log.info("Rare labels (from codebook): %s", rare_labels)

    # --- Load and validate input data ---
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    input_df = load_input_data(input_path)

    if "description" not in input_df.columns:
        log.error("Input is missing required column: 'description'")
        sys.exit(1)

    optional_cols = set(META_COLS) | {
        "row_id", "id", "date", "governorate", "area",
        "location_name", "location_type", "perpetrator",
    }
    missing_optional = optional_cols - set(input_df.columns)
    if missing_optional:
        log.warning(
            "Input is missing optional column(s): %s — filling with empty strings.",
            sorted(missing_optional),
        )

    if "row_id" not in input_df.columns:
        input_df["row_id"] = [str(i) for i in range(len(input_df))]

    # --- Build enhanced text (must exactly match training format) ---
    texts = build_text_enhanced(input_df.copy())

    # --- Load model, validate label order, run inference ---
    if args.model == "baseline":
        model, vectorizer, meta = load_baseline_model(BASELINE_MODEL_DIR)
        validate_label_order(meta, "baseline")
        probs, preds, thresholds = run_baseline_inference(model, vectorizer, meta, texts)
    else:  # deberta
        model, tokenizer, meta, device = load_deberta_model(DEBERTA_MODEL_DIR)
        validate_label_order(meta, "deberta")
        probs, preds, thresholds = run_deberta_inference(
            model, tokenizer, meta, texts, device, batch_size=args.batch_size
        )

    # --- Resolve confidence band: model-specific default or CLI override ---
    # baseline: 0.02 (calibrated for production use)
    # deberta:  0.05 (provisional — tune once the checkpoint is confirmed working)
    MODEL_CONFIDENCE_DEFAULTS = {"baseline": 0.02, "deberta": 0.05}
    if args.confidence_band is None:
        confidence_band = MODEL_CONFIDENCE_DEFAULTS[args.model]
        log.info("Using default confidence_band=%.2f for model '%s'", confidence_band, args.model)
    else:
        confidence_band = args.confidence_band
        log.info("Using override confidence_band=%.2f (--confidence-band)", confidence_band)

    # --- Uncertainty flags (preliminary — before codebook conflicts) ---
    flags_df = compute_uncertainty_flags(
        probs, preds, thresholds,
        rare_labels=rare_labels,
        confidence_band=confidence_band,
    )

    # --- Codebook constraint checking ---
    conflicts = check_codebook_constraints(preds, codebook)

    # --- Promote codebook conflicts into needs_review ---
    # A confirmed constraint violation always warrants human review.
    conflict_mask = pd.Series([bool(c) for c in conflicts])
    flags_df["needs_review"] = flags_df["needs_review"] | conflict_mask
    log.info(
        "needs_review after codebook conflicts: %d / %d documents",
        int(flags_df["needs_review"].sum()), len(input_df),
    )

    # --- Format both output sheets ---
    full_df = format_full_output(
        input_df=input_df, probs=probs, preds=preds,
        flags_df=flags_df, conflicts=conflicts, codebook=codebook,
        model_name=args.model, run_ts=run_ts,
    )
    compact_df = format_compact_output(
        input_df=input_df, probs=probs, preds=preds,
        flags_df=flags_df, conflicts=conflicts, codebook=codebook,
        model_name=args.model, run_ts=run_ts,
        overlap_pct=args.overlap_pct,
    )

    # --- Export local files ---
    export_outputs(
        full_df, compact_df, probs, preds, thresholds,
        input_df, output_dir, run_ts,
    )

    # --- Optionally push compact sheet to Google Sheets Review tab ---
    if args.push_to_sheets:
        try:
            from data_pipeline.database.sheets_interface import push_review_batch
            log.info("Pushing %d rows to Google Sheets Review tab ...", len(compact_df))
            n_pushed = push_review_batch(compact_df)
            if n_pushed == 0:
                log.info("No new rows pushed (all row_ids already in Review tab).")
            else:
                log.info("Pushed %d new rows to Review tab.", n_pushed)
        except Exception as exc:
            log.error("Failed to push to Google Sheets: %s", exc)
            log.warning("Local files were written successfully — Sheets push is optional.")

    # --- Summary ---
    print_summary(
        input_df, preds, flags_df, conflicts,
        rare_labels, args.model, output_dir, run_ts,
    )


if __name__ == "__main__":
    main()
