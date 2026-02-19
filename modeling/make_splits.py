from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/processed/oppression_2023_2025.csv")
OUT_DIR = Path("data/processed/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1  # test will be 0.1

df = pd.read_csv(DATA)

# Basic sanity
assert "id" in df.columns and "description" in df.columns, "Missing required columns"

# Use unique incident IDs for group split
ids = df["id"].dropna().astype(str).unique()
ids = ids.astype(object)  # ensure numpy-friendly dtype
rng = np.random.default_rng(SEED)
rng.shuffle(ids)

n = len(ids)
n_train = int(n * TRAIN_FRAC)
n_val = int(n * VAL_FRAC)

train_ids = set(ids[:n_train])
val_ids = set(ids[n_train:n_train + n_val])
test_ids = set(ids[n_train + n_val:])

train = df[df["id"].isin(train_ids)].reset_index(drop=True)
val = df[df["id"].isin(val_ids)].reset_index(drop=True)
test = df[df["id"].isin(test_ids)].reset_index(drop=True)

# Save
train.to_csv(OUT_DIR / "train.csv", index=False)
val.to_csv(OUT_DIR / "val.csv", index=False)
test.to_csv(OUT_DIR / "test.csv", index=False)

print("Unique ids:", n)
print("Train rows:", train.shape, "Val rows:", val.shape, "Test rows:", test.shape)

# Verify no leakage
assert set(train["id"]).isdisjoint(set(val["id"]))
assert set(train["id"]).isdisjoint(set(test["id"]))
assert set(val["id"]).isdisjoint(set(test["id"]))
print("No id leakage")
