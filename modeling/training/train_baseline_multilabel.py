import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "modeling" / "training" / "data"
OUT_DIR = PROJECT_ROOT / "modeling" / "saved_models" / "baseline_tfidf_lr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
DEV_CSV = DATA_DIR / "dev.csv"

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


def load_data(path: Path):
    df = pd.read_csv(path)
    # Fill missing descriptions
    df["description"] = df["description"].fillna("").astype(str)
    # Optional: add metadata prefix (helps a bit)
    meta_cols = ["perpetrator", "governorate", "jurisdiction", "location_type", "region"]
    for c in meta_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    # FIXED: Create the text column using vectorized operations
    text_parts = []
    text_parts.append("Perpetrator: " + df["perpetrator"])
    text_parts.append(" | Gov: " + df["governorate"])
    text_parts.append(" | Jur: " + df["jurisdiction"])
    text_parts.append(" | LocType: " + df["location_type"])
    text_parts.append(" | Region: " + df["region"])
    text_parts.append(" | " + df["description"])
    
    df["text"] = text_parts[0]
    for part in text_parts[1:]:
        df["text"] = df["text"] + part

    y = df[LABELS].fillna(0).astype(int).values
    return df["text"].tolist(), y, df


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
    y_pred = (y_prob >= thresholds).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    per_label = []
    for i, lab in enumerate(LABELS):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        per_label.append((lab, float(p), float(r), float(f1)))

    return macro_f1, micro_f1, per_label


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Simple threshold tuning: for each label, pick threshold that maximizes F1 on dev.
    """
    thresholds = np.zeros(len(LABELS), dtype=float)
    for i in range(len(LABELS)):
        best_f1 = -1.0
        best_t = 0.5
        # sweep a small grid
        for t in np.linspace(0.05, 0.95, 19):
            y_pred_i = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], y_pred_i, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[i] = best_t
    return thresholds


def main():
    print("Loading data...")
    X_train, y_train, _ = load_data(TRAIN_CSV)
    X_dev, y_dev, _ = load_data(DEV_CSV)

    print("Training TF-IDF + OneVsRest(LogReg)...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                n_jobs=-1,
                solver="liblinear",
            )
        )),
    ])

    model.fit(X_train, y_train)

    print("Scoring on dev...")
    # OneVsRestClassifier exposes predict_proba
    y_prob = model.predict_proba(X_dev)

    print("Tuning per-label thresholds on dev...")
    thresholds = tune_thresholds(y_dev, y_prob)

    macro_f1, micro_f1, per_label = evaluate(y_dev, y_prob, thresholds)

    print("\n=== DEV RESULTS (threshold-tuned) ===")
    print(f"Macro-F1: {macro_f1:.3f}")
    print(f"Micro-F1: {micro_f1:.3f}")
    print("\nPer-label (precision, recall, f1):")
    for lab, p, r, f1 in per_label:
        print(f"  {lab:28s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  thr={thresholds[LABELS.index(lab)]:.2f}")

    # Save model + thresholds + label list
    import joblib
    joblib.dump(model, OUT_DIR / "model.joblib")

    meta = {
        "labels": LABELS,
        "thresholds": thresholds.tolist(),
        "train_path": str(TRAIN_CSV),
        "dev_path": str(DEV_CSV),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nSaved baseline model to:", OUT_DIR)


if __name__ == "__main__":
    main()
