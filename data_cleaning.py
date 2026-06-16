"""
data_cleaning.py
================
Conservative cleaning for 3 datasets:
  1. ONET      — merge individual jobs_with_*.csv files into one combined file
  2. ESCO       — clean broaderRelations + conceptSchemes files
  3. LinkedIn   — clean job_postings.csv

Cleaning rules:
  - Remove a COLUMN only when >50% of its entries are empty/null
  - Remove a ROW only when it is an exact duplicate (never for partial nulls)
  - Strip leading/trailing whitespace from string columns
  - No other row removal
"""

import pandas as pd
from pathlib import Path
from functools import reduce

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
ONET_DIR    = BASE / "data" / "onet"
ESCO_DIR    = BASE / "data"
LINKEDIN_DIR = Path(r"C:\Users\rohin jain\Desktop\Linkedin dataset")
OUT_DIR     = BASE / "data" / "cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────
def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.50) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns where null fraction > threshold. Returns (cleaned_df, dropped_col_names)."""
    null_frac = df.isnull().mean()
    drop_cols = null_frac[null_frac > threshold].index.tolist()
    return df.drop(columns=drop_cols), drop_cols


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all string columns."""
    str_cols = df.select_dtypes(include="str").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


def clean(df: pd.DataFrame, name: str, threshold: float = 0.50) -> pd.DataFrame:
    original_rows = len(df)
    original_cols = list(df.columns)

    df = strip_strings(df)
    df, dropped_cols = drop_high_null_columns(df, threshold)
    before_dedup = len(df)
    df = df.drop_duplicates()
    removed_dupes = before_dedup - len(df)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Original  : {original_rows:,} rows × {len(original_cols)} cols")
    if dropped_cols:
        print(f"  Cols removed (>{threshold*100:.0f}% null): {dropped_cols}")
    else:
        print(f"  Cols removed : none (all columns below {threshold*100:.0f}% null)")
    print(f"  Dupe rows removed : {removed_dupes}")
    print(f"  Final     : {len(df):,} rows × {len(df.columns)} cols")
    print(f"  Columns kept: {list(df.columns)}")
    return df


# ── 1. ONET ─────────────────────────────────────────────────────────────────
def clean_onet():
    # Merge all individual attribute files on title + description
    files = {
        "skills":          ONET_DIR / "jobs_with_skills.csv",
        "abilities":       ONET_DIR / "jobs_with_abilities.csv",
        "knowledge":       ONET_DIR / "jobs_with_knowledge.csv",
        "tasks":           ONET_DIR / "jobs_with_tasks.csv",
        "work_activities": ONET_DIR / "jobs_with_work_activities.csv",
        "work_styles":     ONET_DIR / "jobs_with_work_styles.csv",
    }

    base_df = pd.read_csv(ONET_DIR / "onet_job_titles.csv")
    base_df.columns = ["title"]   # single-column file

    dfs = []
    for attr, path in files.items():
        df = pd.read_csv(path)[["title", attr]]
        dfs.append(df)

    # Merge all attribute frames on title
    merged = reduce(lambda left, right: pd.merge(left, right, on="title", how="outer"), dfs)

    out = clean(merged, "ONET — merged jobs")
    out.to_csv(OUT_DIR / "onet_jobs_cleaned.csv", index=False)
    print(f"  Saved -> data/cleaned/onet_jobs_cleaned.csv")


# ── 2. ESCO ─────────────────────────────────────────────────────────────────
def clean_esco():
    datasets = {
        "esco_occ_relations":   ESCO_DIR / "broaderRelationsOccPillar_en.csv",
        "esco_skill_relations": ESCO_DIR / "broaderRelationsSkillPillar_en.csv",
        "esco_concept_schemes": ESCO_DIR / "conceptSchemes_en.csv",
    }
    for name, path in datasets.items():
        df = pd.read_csv(path)
        out = clean(df, f"ESCO — {name}")
        out.to_csv(OUT_DIR / f"{name}_cleaned.csv", index=False)
        print(f"  Saved -> data/cleaned/{name}_cleaned.csv")


# ── 3. LinkedIn ─────────────────────────────────────────────────────────────
def clean_linkedin():
    df = pd.read_csv(LINKEDIN_DIR / "job_postings.csv")
    out = clean(df, "LinkedIn — job_postings")
    out.to_csv(OUT_DIR / "linkedin_job_postings_cleaned.csv", index=False)
    print(f"  Saved -> data/cleaned/linkedin_job_postings_cleaned.csv")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nStarting conservative dataset cleaning...")
    print("Rules: remove columns >50% null | remove exact duplicates | keep all other rows\n")

    clean_onet()
    clean_esco()
    clean_linkedin()

    print(f"\n{'='*55}")
    print("  All cleaned files saved to: data/cleaned/")
    print(f"{'='*55}\n")
