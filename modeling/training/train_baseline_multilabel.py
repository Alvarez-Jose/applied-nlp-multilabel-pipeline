from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression


# =============================
# Paths / Config
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "oppression_2023_2025.csv"

# Output directory for baseline
OUT_DIR = PROJECT_ROOT / "modeling" / "saved_models" / "baseline_rank_based_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Labels in processed file (raw-ish columns)
RAW_BOOL_LABELS = [
    "raid",
    "arrest_detention",
    "religious_encroachment",
    "protest",
    "multi_community_incident",
]
RAW_PRESENCE_LABELS = [
    "physical_assault",
    "harm_to_property",
    "dispossession",
    "restriction_of_freedoms",
    "coercive_actions",
]

# Final labels used by the model (binary)
LABELS = [
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

# Use enhanced text with metadata
USE_ENHANCED_TEXT = True
META_COLS_LITE = ["perpetrator", "governorate", "area", "location_type"]

# Baseline hyperparams
SEED = 42
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 3)
C_REGULARIZATION = 1.0
MAX_ITER = 1000

# Split parameters
DEV_FRAC_2024 = 0.15

# Minimum positives for computing AUC/PR-AUC
MIN_POS_FOR_METRIC = 10

# =============================
# FIXED: Target configuration
# =============================
TARGET_MODE = "meanprob_test_capped_shrunk"  # Using raw probs for targets

# ABSOLUTE CAPS for drifting labels (tighter than ratio-based)
ABSOLUTE_CAPS = {
    "arrest_detention_y": 0.26,      # Much tighter than before
    "coercive_actions_y": 0.50,       # Tighter than 0.569
}

# Default max prediction ratio for other labels
CAP_RATIO_DEFAULT = 1.25

# Shrinkage parameters for meanprob targets
ALPHA_DEFAULT = 0.7  # Weight on meanprob (1-alpha on train prior)
ALPHA_BY_LABEL = {
    "coercive_actions_y": 0.3,   # Shrink more aggressively (was 0.4)
    "arrest_detention_y": 0.4,
    "physical_assault_y": 0.6,
}

# Minimum positives for rare labels (to protect recall)
MIN_POSITIVES = {
    "religious_encroachment_y": 15,   # Force at least 15 positives on test
    "protest_y": 15,
    "multi_community_incident_y": 30,
}

# Generic floor for very rare labels (if not in MIN_POSITIVES)
RARE_LABEL_THRESHOLD = 0.02  # Labels with train prevalence < 2% get floor
RARE_LABEL_FLOOR_RATIO = 0.75  # target >= 0.75 * train_prev

MIN_TARGET = 0.001
MAX_TARGET = 0.95

# Calibration options - FIXED: only used for reporting, NOT for targets/ranking
USE_ISOTONIC_CALIBRATION = False  # DISABLED by default
# If enabled, calibration only affects reporting, not targets or ranking


def get_cap(label: str, train_prev: float) -> float:
    """Get cap for a label (absolute if specified, otherwise ratio-based)."""
    if label in ABSOLUTE_CAPS:
        return ABSOLUTE_CAPS[label]
    return CAP_RATIO_DEFAULT * train_prev


def get_alpha(label: str) -> float:
    """Get shrinkage alpha for a specific label."""
    return ALPHA_BY_LABEL.get(label, ALPHA_DEFAULT)


def apply_floors(target: float, label: str, n_samples: int, train_prev: float) -> float:
    """Apply floors to ensure rare labels get at least some predictions."""
    # Check for minimum positives requirement
    if label in MIN_POSITIVES:
        min_target = MIN_POSITIVES[label] / n_samples
        if target < min_target:
            print(f"      ⚠️  {label}: raising target from {target:.4f} to {min_target:.4f} "
                  f"(min {MIN_POSITIVES[label]} positives)")
            return min_target
    
    # Generic floor for very rare labels
    if train_prev < RARE_LABEL_THRESHOLD:
        floor = RARE_LABEL_FLOOR_RATIO * train_prev
        if target < floor:
            print(f"      ⚠️  {label}: raising target from {target:.4f} to {floor:.4f} (rare label floor)")
            return floor
    
    return target


def clamp(x: float, lo: float = MIN_TARGET, hi: float = MAX_TARGET) -> float:
    """Clamp target rate to safe range."""
    return max(lo, min(hi, x))


# =============================
# Target computation functions - USING RAW PROBS ONLY
# =============================
def compute_meanprob_targets_shrunk(
    train_prev: Dict[str, float],
    probs: np.ndarray,  # Should be RAW probabilities
    label_to_idx: Dict[str, int],
    n_samples: int
) -> Dict[str, float]:
    """
    Compute target rates with shrinkage toward train prior.
    Uses RAW probabilities only.
    Applies:
    - Shrinkage: alpha * mean_prob + (1-alpha) * train
    - Label-specific caps (absolute or ratio-based)
    - Floors for rare labels
    """
    target = {}
    
    print(f"\n=== COMPUTING SHRUNK MEANPROB TARGETS (RAW PROBS) ===")
    for lab in train_prev:
        tr = float(train_prev[lab])
        alpha = get_alpha(lab)
        cap = get_cap(lab, tr)
        
        j = label_to_idx[lab]
        mean_p = float(probs[:, j].mean())
        
        # Shrink toward train prior
        t = alpha * mean_p + (1 - alpha) * tr
        t = min(t, cap)
        t = apply_floors(t, lab, n_samples, tr)
        t = clamp(t)
        
        target[lab] = t
        
        print(f"  {lab:30s} train={tr:.3f}, mean_prob={mean_p:.3f}, alpha={alpha:.2f}, "
              f"cap={cap:.3f} → target={target[lab]:.3f}")
    
    return target


# =============================
# Optional: Isotonic calibration (for reporting only)
# =============================
def fit_isotonic_calibrators(
    y_dev: np.ndarray,
    y_dev_prob: np.ndarray,
    label_names: list[str]
) -> Dict[str, Optional[IsotonicRegression]]:
    """Fit isotonic regression calibrators per label on dev data."""
    calibrators = {}
    
    print("\n=== FITTING ISOTONIC CALIBRATORS (REPORTING ONLY) ===")
    for j, lab in enumerate(label_names):
        y_true = y_dev[:, j]
        y_prob = y_dev_prob[:, j]
        
        if len(set(y_true)) < 2:
            print(f"  {lab:30s} [SKIP] only one class present")
            calibrators[lab] = None
            continue
        
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(y_prob, y_true)
        calibrators[lab] = ir
        
        prob_cal = ir.transform(y_prob)
        brier_raw = np.mean((y_prob - y_true) ** 2)
        brier_cal = np.mean((prob_cal - y_true) ** 2)
        print(f"  {lab:30s} Brier: raw={brier_raw:.4f} → cal={brier_cal:.4f} "
              f"(Δ={brier_raw - brier_cal:.4f})")
    
    return calibrators


def apply_calibration(
    probs: np.ndarray,
    calibrators: Dict[str, Optional[IsotonicRegression]],
    label_names: list[str]
) -> np.ndarray:
    """Apply isotonic calibration to probabilities."""
    probs_cal = probs.copy()
    
    for j, lab in enumerate(label_names):
        ir = calibrators.get(lab)
        if ir is not None:
            probs_cal[:, j] = ir.transform(probs_cal[:, j])
    
    return probs_cal


# =============================
# Label helpers
# =============================
def norm_bool01(x) -> int:
    """Normalize boolean-ish tokens to 0/1."""
    if pd.isna(x):
        return 0
    
    if isinstance(x, (int, float, np.integer, np.floating)):
        return 1 if x > 0 else 0
    
    s = str(x).strip().lower()
    
    try:
        val = float(s)
        return 1 if val > 0 else 0
    except ValueError:
        pass
    
    if s in {"1", "true", "yes", "y", "t"}:
        return 1
    if s in {"0", "false", "no", "n", "f", ""}:
        return 0
    
    return 1 if s else 0


def norm_presence(x) -> int:
    """Any non-empty, non-FALSE/0 token => present."""
    if pd.isna(x):
        return 0
    
    if isinstance(x, (int, float, np.integer, np.floating)):
        return 1 if x > 0 else 0
    
    s = str(x).strip()
    if s == "" or s.upper() == "FALSE" or s == "0":
        return 0
    
    return 1


def safe_auc(y_true_1d: np.ndarray, y_score_1d: np.ndarray) -> float:
    uniq = np.unique(y_true_1d)
    if len(uniq) < 2 or y_true_1d.sum() < MIN_POS_FOR_METRIC:
        return float("nan")
    return float(roc_auc_score(y_true_1d, y_score_1d))


def safe_prauc(y_true_1d: np.ndarray, y_score_1d: np.ndarray) -> float:
    uniq = np.unique(y_true_1d)
    if len(uniq) < 2 or y_true_1d.sum() < MIN_POS_FOR_METRIC:
        return float("nan")
    return float(average_precision_score(y_true_1d, y_score_1d))


# =============================
# Data loading and improved temporal split
# =============================
def load_processed() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = ["date", "description", "id", "row_id", "area", "governorate", "perpetrator", "location_type"]
    for c in required:
        if c not in df.columns:
            df[c] = ""

    df["year"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce").dt.year
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    df["description"] = df["description"].fillna("").astype(str)
    df["location_type"] = df["location_type"].fillna("").astype(str)

    for c in RAW_BOOL_LABELS:
        if c not in df.columns:
            df[c] = 0
        df[c + "_y"] = df[c].apply(norm_bool01).astype(int)

    for c in RAW_PRESENCE_LABELS:
        if c not in df.columns:
            df[c] = ""
        df[c + "_y"] = df[c].apply(norm_presence).astype(int)

    return df


def temporal_split_improved(df: pd.DataFrame, dev_frac_2024: float = DEV_FRAC_2024, seed: int = SEED):
    """Improved temporal split with stratified dev from 2024."""
    train_2023 = df[df["year"] == 2023].copy()
    year_2024 = df[df["year"] == 2024].copy()
    test = df[df["year"] == 2025].copy()

    # Create a proxy label for stratification
    y_proxy = year_2024[LABELS].sum(axis=1).clip(0, 3)

    train_2024, dev_2024 = train_test_split(
        year_2024,
        test_size=dev_frac_2024,
        random_state=seed,
        stratify=y_proxy,
    )

    train = pd.concat([train_2023, train_2024], ignore_index=True)
    
    print(f"\n=== IMPROVED SPLIT ===")
    print(f"  Train: 2023 ({len(train_2023)} rows) + 2024 ({len(train_2024)} rows) = {len(train)} total")
    print(f"  Dev:   2024 ({len(dev_2024)} rows, {dev_frac_2024*100:.1f}% of 2024)")
    print(f"  Test:  2025 ({len(test)} rows)")
    
    return train, dev_2024, test


def build_text_enhanced(df: pd.DataFrame) -> pd.Series:
    """Build enhanced text with metadata."""
    if not USE_ENHANCED_TEXT:
        return df["description"].fillna("").astype(str)

    for c in META_COLS_LITE:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    # Enhanced format with clear markers
    return (
        "[AREA] " + df["area"] +
        " [GOV] " + df["governorate"] +
        " [LOC_TYPE] " + df["location_type"] +
        " [PERP] " + df["perpetrator"] +
        " [DESC] " + df["description"]
    )


def prevalence_and_support(df: pd.DataFrame, name: str, low_support_k: int = MIN_POS_FOR_METRIC) -> None:
    print(f"\n{name} prevalence:")
    for lab in LABELS:
        prev = float(df[lab].mean()) if len(df) else float("nan")
        print(f"  {lab:30s} {prev*100:5.1f}%")

    print(f"\n{name} support (#positives / total):")
    n = len(df)
    for lab in LABELS:
        pos = int(df[lab].sum())
        msg = f"  {lab:30s} {pos:5d} / {n} ({(pos/n*100 if n else 0):.1f}%)"
        if pos < low_support_k:
            msg += " (low support)"
        print(msg)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def shift_analysis(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\n=== DISTRIBUTION SHIFT ANALYSIS (Train → Test) ===")
    p = np.array([train[lab].mean() for lab in LABELS], dtype=float)
    q = np.array([test[lab].mean() for lab in LABELS], dtype=float)

    kl = kl_divergence(p, q)
    print(f"\nKL Divergence (train → test): {kl:.4f}")

    diffs = []
    for i, lab in enumerate(LABELS):
        diffs.append((lab, abs(q[i] - p[i]), (q[i] - p[i])))
    diffs.sort(key=lambda x: x[1], reverse=True)

    print("\nLargest absolute prevalence shifts:")
    for lab, absd, signed in diffs[:3]:
        print(f"  {lab:30s} {signed*100:+.1f} percentage points")


# =============================
# Rank-based prediction (top-k) - USING RAW PROBS
# =============================
def predict_rank_based(y_prob: np.ndarray, target_rates: Dict[str, float], label_names: list[str]) -> np.ndarray:
    """
    Make predictions using rank-based approach.
    Uses RAW probabilities (ranking is monotonic, so calibration unnecessary).
    """
    n_samples = y_prob.shape[0]
    y_pred = np.zeros_like(y_prob, dtype=int)
    
    for i, lab in enumerate(label_names):
        target_rate = target_rates[lab]
        k = int(round(target_rate * n_samples))
        if k == 0:
            continue
        
        # Get indices of top k probabilities
        top_k_indices = np.argpartition(-y_prob[:, i], k)[:k]
        y_pred[top_k_indices, i] = 1
    
    return y_pred


def evaluate_rank_based(
    name: str, 
    y_true: np.ndarray, 
    y_prob: np.ndarray,  # RAW probs for ranking
    y_prob_cal: Optional[np.ndarray],  # Calibrated probs for reporting (optional)
    target_rates: Dict[str, float],
    label_names: list[str]
) -> Dict:
    """Evaluate rank-based predictions."""
    y_pred = predict_rank_based(y_prob, target_rates, label_names)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    aucs, praucs = [], []
    per_label = []
    for i, lab in enumerate(label_names):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        # Use raw probs for AUC/PR-AUC (ranking-based metrics)
        auc = safe_auc(y_true[:, i], y_prob[:, i])
        pr = safe_prauc(y_true[:, i], y_prob[:, i])
        if not math.isnan(auc):
            aucs.append(auc)
        if not math.isnan(pr):
            praucs.append(pr)
        per_label.append((lab, float(p), float(r), float(f1), float(auc), float(pr)))

    mean_auc = float(np.mean(aucs)) if len(aucs) else float("nan")
    mean_prauc = float(np.mean(praucs)) if len(praucs) else float("nan")

    print(f"\n=== {name} RESULTS (RANK-BASED) ===")
    print(f"Input text: {'enhanced with metadata' if USE_ENHANCED_TEXT else 'description-only'}")
    print(f"Macro-F1: {macro_f1:.3f}")
    print(f"Micro-F1: {micro_f1:.3f}")
    print(f"Mean AUC: {mean_auc:.3f}" if not math.isnan(mean_auc) else "Mean AUC: nan")
    print(f"Mean PR-AUC: {mean_prauc:.3f}" if not math.isnan(mean_prauc) else "Mean PR-AUC: nan")

    print("\nPer-label (precision, recall, f1, auc, pr-auc):")
    for lab, p, r, f1, auc, pr in per_label:
        auc_str = "nan" if math.isnan(auc) else f"{auc:.3f}"
        pr_str = "nan" if math.isnan(pr) else f"{pr:.3f}"
        print(f"  {lab:30s} P={p:.3f}  R={r:.3f}  F1={f1:.3f}  AUC={auc_str}  PR-AUC={pr_str}")

    print("\n=== PREDICTED VS TRUE POSITIVE RATES ===")
    for i, lab in enumerate(label_names):
        pred_rate = y_pred[:, i].mean() * 100
        true_rate = y_true[:, i].mean() * 100
        diff = pred_rate - true_rate
        marker = "***" if abs(diff) > 10 else ("*" if abs(diff) > 5 else "")
        print(f"  {lab:30s} pred: {pred_rate:5.1f}%  true: {true_rate:5.1f}%  diff: {diff:+5.1f}% {marker}")

    # Optional: Show calibrated probabilities if available
    if y_prob_cal is not None:
        print("\n=== CALIBRATED PROBABILITIES (for reference) ===")
        for target_lab in ["coercive_actions_y", "arrest_detention_y"]:
            if target_lab in label_names:
                idx = label_names.index(target_lab)
                raw_mean = y_prob[:, idx].mean()
                cal_mean = y_prob_cal[:, idx].mean()
                print(f"  {target_lab}: raw mean={raw_mean:.3f}, cal mean={cal_mean:.3f}")

    # Diagnostic: show raw probability distribution for key labels
    print("\n=== RAW PROBABILITY DISTRIBUTION DIAGNOSTIC ===")
    for target_lab in ["coercive_actions_y", "arrest_detention_y", "religious_encroachment_y"]:
        if target_lab in label_names:
            idx = label_names.index(target_lab)
            probs = y_prob[:, idx]
            print(f"\n{target_lab}:")
            print(f"  Mean prob: {probs.mean():.3f}")
            print(f"  Std prob:  {probs.std():.3f}")
            print(f"  Quantiles: p50={np.percentile(probs, 50):.3f}, "
                  f"p75={np.percentile(probs, 75):.3f}, "
                  f"p90={np.percentile(probs, 90):.3f}, "
                  f"p95={np.percentile(probs, 95):.3f}")

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mean_auc": mean_auc,
        "mean_prauc": mean_prauc,
        "per_label": per_label,
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, 
                 n_boot: int = 500, seed: int = 123) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Bootstrap confidence intervals for predictions."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    macro_scores, micro_scores = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        macro_scores.append(f1_score(yt, yp, average="macro", zero_division=0))
        micro_scores.append(f1_score(yt, yp, average="micro", zero_division=0))

    macro_ci = (float(np.percentile(macro_scores, 2.5)), float(np.percentile(macro_scores, 97.5)))
    micro_ci = (float(np.percentile(micro_scores, 2.5)), float(np.percentile(micro_scores, 97.5)))
    return macro_ci, micro_ci


# =============================
# Main
# =============================
def main():
    # Set random seed
    np.random.seed(SEED)

    print("Loading processed data by year...\n")
    df = load_processed()

    for lab in LABELS:
        if lab not in df.columns:
            raise ValueError(f"Missing label column: {lab}")

    print("Available columns in data:")
    print(list(df.columns)[:10], "...")

    # Use improved temporal split
    train_df, dev_df, test_df = temporal_split_improved(df)

    print("\n=== NEW SPLIT PREVALENCE ===")
    prevalence_and_support(train_df, "Train")
    prevalence_and_support(dev_df, "Dev")
    prevalence_and_support(test_df, "Test")
    shift_analysis(train_df, test_df)

    # Build enhanced text with metadata
    print(f"\nBuilding text with {'enhanced metadata' if USE_ENHANCED_TEXT else 'description only'}...")
    X_train_text = build_text_enhanced(train_df).tolist()
    X_dev_text = build_text_enhanced(dev_df).tolist()
    X_test_text = build_text_enhanced(test_df).tolist()

    y_train = train_df[LABELS].astype(int).values
    y_dev = dev_df[LABELS].astype(int).values
    y_test = test_df[LABELS].astype(int).values

    # Create label to index mapping
    label_to_idx = {lab: i for i, lab in enumerate(LABELS)}

    # Verify label distributions
    print("\n=== LABEL DISTRIBUTION VERIFICATION ===")
    for i, lab in enumerate(LABELS):
        train_pct = y_train[:, i].mean() * 100
        dev_pct = y_dev[:, i].mean() * 100
        test_pct = y_test[:, i].mean() * 100
        print(f"  {lab:30s} train: {train_pct:5.1f}%, dev: {dev_pct:5.1f}%, test: {test_pct:5.1f}%")

    # ===== TF-IDF VECTORIZATION =====
    print("\n=== TF-IDF VECTORIZATION ===")
    print(f"Max features: {MAX_FEATURES}, N-gram range: {NGRAM_RANGE}")
    
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_dev_tfidf = vectorizer.transform(X_dev_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    print(f"Train TF-IDF shape: {X_train_tfidf.shape}")
    print(f"Dev TF-IDF shape:   {X_dev_tfidf.shape}")
    print(f"Test TF-IDF shape:  {X_test_tfidf.shape}")

    # ===== TRAINING =====
    print("\n=== TRAINING MULTI-LABEL LOGISTIC REGRESSION ===")
    print(f"Regularization C: {C_REGULARIZATION}, Max iterations: {MAX_ITER}")
    
    base_lr = LogisticRegression(
        C=C_REGULARIZATION,
        max_iter=MAX_ITER,
        random_state=SEED,
        n_jobs=-1,
    )
    
    model = MultiOutputClassifier(base_lr, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)

    # ===== PREDICTIONS (RAW PROBS) =====
    print("\nGenerating predictions (RAW probabilities)...")
    
    y_train_prob_list = []
    y_dev_prob_list = []
    y_test_prob_list = []
    
    for est in model.estimators_:
        y_train_prob_list.append(est.predict_proba(X_train_tfidf)[:, 1]) # type: ignore
        y_dev_prob_list.append(est.predict_proba(X_dev_tfidf)[:, 1]) # type: ignore
        y_test_prob_list.append(est.predict_proba(X_test_tfidf)[:, 1]) # type: ignore
    
    y_train_prob = np.array(y_train_prob_list).T
    y_dev_prob_raw = np.array(y_dev_prob_list).T
    y_test_prob_raw = np.array(y_test_prob_list).T

    # ===== OPTIONAL: Isotonic calibration (for reporting only) =====
    if USE_ISOTONIC_CALIBRATION:
        calibrators = fit_isotonic_calibrators(y_dev, y_dev_prob_raw, LABELS)
        y_dev_prob_cal = apply_calibration(y_dev_prob_raw, calibrators, LABELS)
        y_test_prob_cal = apply_calibration(y_test_prob_raw, calibrators, LABELS)
        print("\n✓ Isotonic calibration applied (reporting only)")
    else:
        y_dev_prob_cal = None
        y_test_prob_cal = None
        calibrators = None

    # ===== COMPUTE PREVALENCE DICTS =====
    train_prev = {lab: y_train[:, i].mean() for i, lab in enumerate(LABELS)}
    dev_prev = {lab: y_dev[:, i].mean() for i, lab in enumerate(LABELS)}
    test_prev = {lab: y_test[:, i].mean() for i, lab in enumerate(LABELS)}

    # ===== COMPUTE TARGET RATES USING RAW PROBS ONLY =====
    print("\n" + "=" * 70)
    print(f"TARGET RATE COMPUTATION (mode={TARGET_MODE})")
    print("=" * 70)
    print("Using RAW probabilities for target estimation")
    
    target_rates_dev = compute_meanprob_targets_shrunk(
        train_prev, y_dev_prob_raw, label_to_idx, n_samples=len(y_dev)
    )
    target_rates_test = compute_meanprob_targets_shrunk(
        train_prev, y_test_prob_raw, label_to_idx, n_samples=len(y_test)
    )

    # Print comparison with true test prevalence
    print("\n=== FINAL TARGET RATES vs TRUE TEST PREVALENCE ===")
    print(f"{'Label':30s} {'Train':>8s} {'Dev':>8s} {'Test True':>10s} {'Target(Dev)':>12s} {'Target(Test)':>12s}")
    print("-" * 80)
    for lab in LABELS:
        print(f"{lab:30s} {train_prev[lab]:8.3f} {dev_prev[lab]:8.3f} "
              f"{test_prev[lab]:10.3f} {target_rates_dev[lab]:12.3f} {target_rates_test[lab]:12.3f}")

    # ===== EVALUATION (using RAW probs for ranking) =====
    print("\n" + "=" * 70)
    print("RANK-BASED PREDICTION (using RAW probs)")
    print("=" * 70)
    
    dev_metrics = evaluate_rank_based(
        "DEV", y_dev, y_dev_prob_raw, y_dev_prob_cal, target_rates_dev, LABELS
    )
    
    print("\n" + "=" * 70)
    print("TEST (2025)")
    print("=" * 70)
    test_metrics = evaluate_rank_based(
        "TEST", y_test, y_test_prob_raw, y_test_prob_cal, target_rates_test, LABELS
    )
    
    # Get predictions for bootstrap
    y_test_pred = predict_rank_based(y_test_prob_raw, target_rates_test, LABELS)

    # ===== BOOTSTRAP CIs =====
    print("\n" + "=" * 50)
    print("BOOTSTRAP CONFIDENCE INTERVALS (TEST)")
    print("=" * 50)
    macro_ci, micro_ci = bootstrap_ci(y_test, y_test_pred, n_boot=500, seed=123)
    print(f"Macro-F1 95% CI: [{macro_ci[0]:.3f}, {macro_ci[1]:.3f}]")
    print(f"Micro-F1 95% CI: [{micro_ci[0]:.3f}, {micro_ci[1]:.3f}]")

    # ===== SAVE MODEL AND METADATA =====
    print("\nSaving model and metadata...")
    
    # Save model and vectorizer
    model_path = OUT_DIR / "baseline_model.pkl"
    vectorizer_path = OUT_DIR / "tfidf_vectorizer.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

    # Save calibration if used
    if calibrators:
        cal_path = OUT_DIR / "isotonic_calibrators.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump(calibrators, f)
        print(f"Calibrators saved to: {cal_path}")

    # Save metadata
    meta = {
        "model_type": "TF-IDF + MultiOutput LogisticRegression",
        "prediction_method": "rank_based",
        "target_mode": TARGET_MODE,
        "use_isotonic_calibration": USE_ISOTONIC_CALIBRATION,
        "data_path": str(DATA_PATH),
        "labels": LABELS,
        "raw_bool_labels": RAW_BOOL_LABELS,
        "raw_presence_labels": RAW_PRESENCE_LABELS,
        "use_enhanced_text": USE_ENHANCED_TEXT,
        "meta_cols_lite": META_COLS_LITE if USE_ENHANCED_TEXT else None,
        "split": {
            "train": {"years": "2023+2024", "rows": int(len(train_df))},
            "dev": {"year": "2024 (subset)", "rows": int(len(dev_df)), "fraction": DEV_FRAC_2024},
            "test": {"year": 2025, "rows": int(len(test_df))},
        },
        "vectorizer_params": {
            "max_features": MAX_FEATURES,
            "ngram_range": NGRAM_RANGE,
            "stop_words": "english",
            "min_df": 2,
            "max_df": 0.95,
        },
        "model_params": {
            "C": C_REGULARIZATION,
            "max_iter": MAX_ITER,
            "penalty": "l2",
        },
        "target_config": {
            "absolute_caps": ABSOLUTE_CAPS,
            "cap_ratio_default": CAP_RATIO_DEFAULT,
            "alpha_default": ALPHA_DEFAULT,
            "alpha_by_label": ALPHA_BY_LABEL,
            "min_positives": MIN_POSITIVES,
            "rare_label_threshold": RARE_LABEL_THRESHOLD,
            "rare_label_floor_ratio": RARE_LABEL_FLOOR_RATIO,
            "min_target": MIN_TARGET,
            "max_target": MAX_TARGET,
        },
        "target_rates": {
            "train": train_prev,
            "dev": dev_prev,
            "target_dev": target_rates_dev,
            "target_test": target_rates_test,
        },
        "dev": {"rows": int(len(dev_df)), **dev_metrics},
        "test": {"rows": int(len(test_df)), **test_metrics},
        "bootstrap_test": {"macro_f1_ci95": macro_ci, "micro_f1_ci95": micro_ci},
        "seed": SEED,
    }

    meta_path = OUT_DIR / "baseline_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    
    print(f"Metadata saved to: {meta_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Target mode: {TARGET_MODE}")
    print(f"Isotonic calibration: {USE_ISOTONIC_CALIBRATION} (reporting only)")
    print(f"Absolute caps: {ABSOLUTE_CAPS}")
    print(f"Min positives: {MIN_POSITIVES}")
    print(f"Baseline Test Macro-F1: {test_metrics['macro_f1']:.3f} (CI: [{macro_ci[0]:.3f}, {macro_ci[1]:.3f}])")
    print(f"Baseline Test Micro-F1: {test_metrics['micro_f1']:.3f} (CI: [{micro_ci[0]:.3f}, {micro_ci[1]:.3f}])")
    print(f"\nTransformer Test (for comparison):")
    print(f"  Macro-F1: ~0.589")
    print(f"  Micro-F1: ~0.669")


if __name__ == "__main__":
    main()