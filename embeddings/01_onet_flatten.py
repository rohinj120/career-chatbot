"""
01_onet_flatten.py
------------------
PURPOSE:
  O*NET data comes as raw SQL files (CREATE TABLE + INSERT statements).
  This script reads those SQL files without needing a real database,
  joins all the lookup/reference tables together, and saves each
  domain (skills, abilities, knowledge, etc.) as a flat JSONL file.

  "Flat" means every row is self-contained — occupation name, element name,
  scale name are all written directly into the row instead of being foreign keys.

OUTPUT:
  data/onet/occupations.jsonl
  data/onet/skills.jsonl
  data/onet/abilities.jsonl
  ... (13 files total)
"""

import re
import os
import json
import pandas as pd
from pathlib import Path

# Where the raw O*NET .sql files live
ONET_DIR = Path(__file__).parent / "ONET"

# Where we save the output .jsonl files
OUT_DIR = Path(__file__).parent / "data" / "onet"
OUT_DIR.mkdir(parents=True, exist_ok=True)  # create the folder if it doesn't exist


# ── SQL PARSER ──────────────────────────────────────────────────────────────
# O*NET gives us .sql files, not a live database.
# These two functions read the SQL text and extract the data rows as Python dicts.

def parse_inserts(sql_text: str) -> list[dict]:
    """
    Reads a SQL file as a string and extracts all INSERT rows as a list of dicts.

    Example SQL input:
        CREATE TABLE skills (onetsoc_code varchar, element_id varchar, data_value float);
        INSERT INTO skills (onetsoc_code, element_id, data_value) VALUES ('15-1252.00', '2.B.1.a', 4.5);

    Returns:
        [{'onetsoc_code': '15-1252.00', 'element_id': '2.B.1.a', 'data_value': 4.5}]
    """

    # Step 1: Extract column names from the CREATE TABLE statement
    create_match = re.search(
        r"CREATE TABLE \w+ \((.*?)\);",
        sql_text, re.DOTALL  # DOTALL lets . match newlines so we can grab multi-line CREATE
    )
    if not create_match:
        return []  # no CREATE TABLE found, nothing to parse

    col_block = create_match.group(1)  # the text inside CREATE TABLE (...)
    cols = []
    for line in col_block.split("\n"):
        line = line.strip()
        # each column definition starts with the column name followed by its type
        m = re.match(r"^(\w+)\s+", line)
        if m and not line.upper().startswith(("PRIMARY", "FOREIGN", "UNIQUE", "KEY", "INDEX", "CONSTRAINT")):
            cols.append(m.group(1))  # save column name, skip constraint lines

    # Step 2: The INSERT statement may specify columns in a different order than CREATE TABLE
    # so we override cols with whatever the INSERT says
    insert_cols_match = re.search(
        r"INSERT INTO \w+ \((.*?)\) VALUES",
        sql_text
    )
    if insert_cols_match:
        cols = [c.strip() for c in insert_cols_match.group(1).split(",")]

    # Step 3: Find every INSERT row and pair its values with the column names
    rows = []
    for m in re.finditer(r"INSERT INTO \w+ \([^)]+\) VALUES (.*?);", sql_text, re.DOTALL):
        raw = m.group(1).strip()   # the VALUES (...) part
        values = _parse_tuple(raw) # parse it into a Python list
        if values and len(values) == len(cols):
            rows.append(dict(zip(cols, values)))  # pair column names with values
    return rows


def _parse_tuple(s: str) -> list:
    """
    Parses a SQL VALUES tuple string into a Python list.

    Handles:
      - String values in single quotes:  'Software Developer'
      - Escaped single quotes ('' means a literal '):  'it''s fine'
      - NULL values → Python None
      - Integer and float numbers

    Example:
        input:  ('15-1252.00', 'Programming', NULL, 4.5)
        output: ['15-1252.00', 'Programming', None, 4.5]
    """
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]  # strip the outer parentheses

    values = []
    i = 0
    while i < len(s):
        s = s[i:].lstrip()  # skip whitespace
        if not s:
            break

        if s.startswith("'"):
            # It's a string value — find the closing quote
            j = 1
            buf = []
            while j < len(s):
                if s[j] == "'" and j + 1 < len(s) and s[j + 1] == "'":
                    # '' is an escaped single quote inside the string
                    buf.append("'")
                    j += 2
                elif s[j] == "'":
                    # closing quote
                    j += 1
                    break
                else:
                    buf.append(s[j])
                    j += 1
            values.append("".join(buf))
            s = s[j:].lstrip()
            if s.startswith(","):
                s = s[1:]
            i = 0

        elif s.upper().startswith("NULL"):
            # SQL NULL becomes Python None
            values.append(None)
            s = s[4:].lstrip()
            if s.startswith(","):
                s = s[1:]
            i = 0

        else:
            # It's a number (integer or float)
            end = 0
            while end < len(s) and s[end] not in (",", ")"):
                end += 1
            token = s[:end].strip()
            try:
                values.append(float(token) if "." in token else int(token))
            except ValueError:
                values.append(token)
            s = s[end:].lstrip()
            if s.startswith(","):
                s = s[1:]
            i = 0

    return values


def load_table(filename: str) -> list[dict]:
    """Read a .sql file and return its rows as a list of dicts."""
    path = ONET_DIR / filename
    text = path.read_text(encoding="utf-8")
    return parse_inserts(text)


def to_df(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of dicts to a pandas DataFrame."""
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── LOAD REFERENCE / LOOKUP TABLES ──────────────────────────────────────────
# These three tables are the "dictionaries" that let us replace IDs with names.
# We load them once and reuse them for every domain table.

print("Loading reference tables...")

# Occupation table: maps onetsoc_code → occupation title + description
# e.g. '15-1252.00' → 'Software Developers'
occ_df = to_df(load_table("03_occupation_data.sql")).set_index("onetsoc_code")

# Content model reference: maps element_id → element name + description
# e.g. '2.B.1.a' → 'Programming'
cmr_df = to_df(load_table("01_content_model_reference.sql")).set_index("element_id")

# Scales reference: maps scale_id → scale name
# e.g. 'LV' → 'Level', 'IM' → 'Importance'
scales_df = to_df(load_table("04_scales_reference.sql")).set_index("scale_id")

# UNSPSC commodity reference: used for technology skills and tools
# maps commodity_code → commodity_title, class_title, family_title
unspsc_df = to_df(load_table("28_unspsc_reference.sql"))
if not unspsc_df.empty:
    unspsc_df["commodity_code"] = unspsc_df["commodity_code"].apply(
        lambda x: int(x) if pd.notna(x) else x
    )
    unspsc_df = unspsc_df.set_index("commodity_code")

print(f"  Occupations: {len(occ_df)}")
print(f"  Content model elements: {len(cmr_df)}")
print(f"  Scales: {len(scales_df)}")


# ── HELPER FUNCTIONS FOR JOINING ─────────────────────────────────────────────
# These three functions do the actual "flattening" — replacing IDs with names.

def add_occ(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace onetsoc_code (e.g. '15-1252.00') with the actual occupation
    title and description (e.g. 'Software Developers').
    Uses a left join so rows without a match are kept but get NaN.
    """
    df = df.join(occ_df[["title", "description"]], on="onetsoc_code", how="left")
    df = df.rename(columns={"title": "occupation_title", "description": "occupation_description"})
    df = df.drop(columns=["onetsoc_code"], errors="ignore")  # remove the raw code
    return df


def add_element(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace element_id (e.g. '2.B.1.a') with the element name
    and description (e.g. 'Programming').
    """
    df = df.join(cmr_df[["element_name", "description"]], on="element_id", how="left")
    df = df.rename(columns={"description": "element_description"})
    df = df.drop(columns=["element_id"], errors="ignore")
    return df


def add_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace scale_id (e.g. 'LV') with the scale name (e.g. 'Level').
    """
    df = df.join(scales_df[["scale_name"]], on="scale_id", how="left")
    df = df.drop(columns=["scale_id"], errors="ignore")
    return df


def save(df: pd.DataFrame, name: str):
    """
    Save a DataFrame to a JSONL file (one JSON object per line).
    JSONL is used instead of CSV because it handles nested/complex values
    and is easy to stream line-by-line in the embedding script.
    """
    if df.empty:
        print(f"  SKIP {name} (empty)")
        return
    df = df.dropna(how="all")  # remove rows where every single field is null
    path = OUT_DIR / f"{name}.jsonl"
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"  Saved {name}: {len(df)} rows -> {path.name}")


# ── FLATTEN EACH DOMAIN ──────────────────────────────────────────────────────
# For each domain we: load the SQL → convert to DataFrame → join lookups → save JSONL
# The pattern is always: load → add_occ → add_element → add_scale → save

# The columns we keep for all attribute tables (skills, abilities, knowledge, etc.)
keep = ["occupation_title", "occupation_description", "element_name", "element_description",
        "scale_name", "data_value", "recommend_suppress", "not_relevant"]


# 1. OCCUPATIONS — the master list of ~1,016 US occupations with titles + descriptions
print("\n[1] Occupations")
occ_out = occ_df.reset_index().rename(columns={"onetsoc_code": "onet_code"})
save(occ_out, "occupations")


# 2. SKILLS — how important each skill is for each occupation
# e.g. "Software Developers | Programming | Level | 4.5"
print("\n[2] Skills")
rows = load_table("16_skills.sql")
df = to_df(rows)
df = add_occ(df)      # replace onetsoc_code with occupation title
df = add_element(df)  # replace element_id with skill name
df = add_scale(df)    # replace scale_id with scale name
save(df[[c for c in keep if c in df.columns]], "skills")


# 3. ABILITIES — cognitive and physical abilities needed per occupation
# e.g. "Surgeons | Manual Dexterity | Importance | 4.8"
print("\n[3] Abilities")
rows = load_table("11_abilities.sql")
df = to_df(rows)
df = add_occ(df)
df = add_element(df)
df = add_scale(df)
save(df[[c for c in keep if c in df.columns]], "abilities")


# 4. KNOWLEDGE — subject matter knowledge required per occupation
# e.g. "Pharmacists | Chemistry | Level | 4.7"
print("\n[4] Knowledge")
rows = load_table("15_knowledge.sql")
df = to_df(rows)
df = add_occ(df)
df = add_element(df)
df = add_scale(df)
save(df[[c for c in keep if c in df.columns]], "knowledge")


# 5. WORK ACTIVITIES — what workers actually do on the job
# e.g. "Architects | Drafting, Laying Out, and Specifying | Importance | 3.2"
print("\n[5] Work Activities")
rows = load_table("19_work_activities.sql")
df = to_df(rows)
df = add_occ(df)
df = add_element(df)
df = add_scale(df)
save(df[[c for c in keep if c in df.columns]], "work_activities")


# 6. WORK STYLES — personality traits and work habits
# e.g. "Nurses | Attention to Detail | Score | 4.9"
print("\n[6] Work Styles")
rows = load_table("21_work_styles.sql")
df = to_df(rows)
df = add_occ(df)
df = add_element(df)
df = add_scale(df)
save(df[[c for c in keep if c in df.columns]], "work_styles")


# 7. WORK VALUES — what workers value in a job (achievement, independence, etc.)
# e.g. "Data Scientists | Achievement | Score | 4.1"
print("\n[7] Work Values")
rows = load_table("22_work_values.sql")
df = to_df(rows)
df = add_occ(df)
df = add_element(df)
df = add_scale(df)
save(df[[c for c in keep if c in df.columns]], "work_values")


# 8. INTERESTS — RIASEC interest profile (Realistic, Investigative, Artistic, etc.)
# e.g. "Physicists | Investigative | Score | 5.0"
print("\n[8] Interests")
rows = load_table("13_interests.sql")
df = to_df(rows)
df = add_occ(df)
df = add_element(df)
df = add_scale(df)
save(df[[c for c in keep if c in df.columns]], "interests")


# 9. TASK STATEMENTS — specific tasks performed in the occupation
# e.g. "Software Developers | Write and test code for new applications"
print("\n[9] Task Statements")
rows = load_table("17_task_statements.sql")
df = to_df(rows)
df = add_occ(df)
# drop columns that are metadata/admin and not useful for embeddings
df = df.drop(columns=["task_id", "incumbents_responding", "date_updated", "domain_source"], errors="ignore")
save(df, "tasks")


# 10. TECHNOLOGY SKILLS — software and tech tools used in each occupation
# e.g. "Software Developers | Python | Programming language software | Hot Technology: Yes"
print("\n[10] Technology Skills")
rows = load_table("31_technology_skills.sql")
df = to_df(rows)
df = add_occ(df)
if not unspsc_df.empty:
    # join the UNSPSC commodity reference to get category names for tech tools
    df["commodity_code"] = df["commodity_code"].apply(lambda x: int(x) if pd.notna(x) else x)
    df = df.join(unspsc_df[["commodity_title", "class_title", "family_title"]], on="commodity_code", how="left")
df = df.drop(columns=["commodity_code"], errors="ignore")
save(df, "technology_skills")


# 11. TOOLS USED — physical tools used in each occupation
# e.g. "Surgeons | Scalpels | Surgical instruments"
print("\n[11] Tools Used")
rows = load_table("32_tools_used.sql")
df = to_df(rows)
df = add_occ(df)
if not unspsc_df.empty:
    df["commodity_code"] = df["commodity_code"].apply(lambda x: int(x) if pd.notna(x) else x)
    df = df.join(unspsc_df[["commodity_title", "class_title", "family_title"]], on="commodity_code", how="left")
df = df.drop(columns=["commodity_code"], errors="ignore")
save(df, "tools_used")


# 12. ALTERNATE TITLES — other job titles that map to this occupation
# e.g. "Software Developers | Coder | App Developer"
print("\n[12] Alternate Titles")
rows = load_table("29_alternate_titles.sql")
df = to_df(rows)
df = add_occ(df)
df = df.drop(columns=["sources"], errors="ignore")
save(df, "alternate_titles")


# 13. EDUCATION / TRAINING / EXPERIENCE — education levels, training, and experience required
# e.g. "Physicians | Required Level of Education | Bachelor's or higher | Score | 5.0"
print("\n[13] Education/Training/Experience")
rows = load_table("12_education_training_experience.sql")
df = to_df(rows)
df = add_occ(df)
df = add_element(df)
df = add_scale(df)
ete_rows = load_table("05_ete_categories.sql")
if ete_rows:
    ete_df = to_df(ete_rows)
    print("    ETE categories cols:", ete_df.columns.tolist())
save(df, "education_training_experience")


print("\nO*NET flatten complete.")
