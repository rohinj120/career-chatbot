"""
03_canada_flatten.py
--------------------
PURPOSE:
  Cleans and flattens the Canada Job Bank dataset.
  This is real job posting data from Canada — actual listings with job titles,
  NOC occupation codes, salaries, locations, education requirements, etc.

  Unlike O*NET (which is a structured taxonomy), this is messy real-world data,
  so the main job here is cleaning rather than joining tables.

  NOC = National Occupational Classification (Canada's version of O*NET codes)

OUTPUT:
  data/canada/job_postings.jsonl
"""

import pandas as pd
from pathlib import Path

# Path to the raw Canada Job Bank CSV file
CANADA_CSV = Path(__file__).parent / "canada dataset.csv"

# Where to save the output
OUT_DIR = Path(__file__).parent / "data" / "canada"
OUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading Canada dataset...")

# The file is UTF-16 encoded (not the usual UTF-8) and tab-separated (not comma-separated)
# This is common for government data exports from Windows systems
df = pd.read_csv(CANADA_CSV, encoding="utf-16", sep="\t", low_memory=False)
print(f"  Raw shape: {df.shape}")  # shows (rows, columns) before cleaning


# ── STEP 1: NORMALIZE COLUMN NAMES ───────────────────────────────────────────
# Raw column names often have spaces, capital letters, special characters.
# We convert them all to lowercase with underscores so they're easy to use.
# e.g. "Job Title" → "job_title", "NOC 2016 Code Name" → "noc_2016_code_name"
df.columns = (
    df.columns
    .str.strip()                                      # remove leading/trailing spaces
    .str.lower()                                      # convert to lowercase
    .str.replace(r"[^a-z0-9]+", "_", regex=True)     # replace non-alphanumeric chars with _
    .str.strip("_")                                   # remove leading/trailing underscores
)


# ── STEP 2: DROP COMPLETELY EMPTY ROWS ───────────────────────────────────────
# Some rows exist only because of an ID column but have no actual data.
# We drop rows where all columns except the ID are null.
non_id_cols = [c for c in df.columns if c != "wic_job_location_snapshot_id"]
df = df.dropna(subset=non_id_cols, how="all")


# ── STEP 3: DROP NEARLY-EMPTY COLUMNS ────────────────────────────────────────
# If a column is empty for >95% of rows, it's not useful for embeddings.
# Keeping it would just add noise ("Field: None | Field: None | ...").
threshold = 0.05  # a column must have at least 5% non-null values to keep it
n = len(df)
mostly_empty = [c for c in df.columns if df[c].notna().sum() / n < threshold]
print(f"  Dropping {len(mostly_empty)} columns that are >95% empty: {mostly_empty}")
df = df.drop(columns=mostly_empty)


print(f"  Clean shape: {df.shape}")
print(f"  Columns kept: {df.columns.tolist()}")


# ── SAVE ─────────────────────────────────────────────────────────────────────
# Save as JSONL — one job posting per line
out_path = OUT_DIR / "job_postings.jsonl"
df.to_json(out_path, orient="records", lines=True, force_ascii=False)
print(f"\nSaved {len(df)} rows -> {out_path}")
print("Canada flatten complete.")
