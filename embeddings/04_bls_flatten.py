"""
04_bls_flatten.py
-----------------
PURPOSE:
  Cleans and flattens the BLS (Bureau of Labor Statistics) dataset.
  BLS publishes time-series data on employment, wages, and job statistics
  for different occupation categories in the US.

  Example: "Registered Nurses | January 2023 | Employment: 3,200,000"

  The data is already flat (no joins needed), so this script just does
  minimal cleaning — removing nulls and useless columns.

OUTPUT:
  data/bls/series.jsonl
"""

import pandas as pd
from pathlib import Path

# Path to the raw BLS CSV file
BLS_CSV = Path(__file__).parent / "raw" / "BLS DATASET.csv"

# Where to save the output
OUT_DIR = Path(__file__).parent / "data" / "bls"
OUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading BLS dataset...")
df = pd.read_csv(BLS_CSV, low_memory=False)
print(f"  Raw shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")


# ── STEP 1: NORMALIZE COLUMN NAMES ───────────────────────────────────────────
# Same as the Canada script — convert column names to clean lowercase with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)


# ── STEP 2: DROP ROWS WITH NULL VALUE ────────────────────────────────────────
# "value" is the actual measurement (e.g. employment count, wage).
# A row without a value is useless — there's nothing to embed.
before = len(df)
df = df.dropna(subset=["value"])
print(f"  Dropped {before - len(df)} rows with null value")


# ── STEP 3: DROP THE FOOTNOTES COLUMN IF MOSTLY EMPTY ────────────────────────
# BLS includes a "footnotes" column for special notes (e.g. "data revised").
# If it's almost always empty, it adds no value to embeddings.
if "footnotes" in df.columns and df["footnotes"].notna().sum() / len(df) < 0.05:
    df = df.drop(columns=["footnotes"])
    print("  Dropped footnotes column (>95% empty)")


print(f"  Clean shape: {df.shape}")


# ── SAVE ─────────────────────────────────────────────────────────────────────
# Save as JSONL — one time-series data point per line
out_path = OUT_DIR / "series.jsonl"
df.to_json(out_path, orient="records", lines=True, force_ascii=False)
print(f"\nSaved {len(df)} rows -> {out_path}")
print("BLS flatten complete.")
