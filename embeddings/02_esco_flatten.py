"""
02_esco_flatten.py
------------------
PURPOSE:
  ESCO (European Skills, Competences, Qualifications and Occupations) is the
  European equivalent of O*NET. The data comes as CSV files.
  This script joins them together and saves flat JSONL files.

  3 output files:
    - esco_occupations: one row per occupation with ISCO group name
    - esco_occupation_skills: one row per occupation-skill pair (the relationship table)
    - esco_skills: one row per skill with its group/category label

OUTPUT:
  data/esco/occupations.jsonl
  data/esco/occupation_skills.jsonl
  data/esco/skills.jsonl
"""

import pandas as pd
from pathlib import Path

# Where the raw ESCO CSV files live
ESCO_DIR = Path(__file__).parent / "esco dataset"

# Where we save the output JSONL files
OUT_DIR = Path(__file__).parent / "data" / "esco"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, name: str):
    """Save DataFrame to JSONL (one JSON object per line)."""
    df = df.dropna(how="all")  # remove completely empty rows
    path = OUT_DIR / f"{name}.jsonl"
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"  Saved {name}: {len(df)} rows -> {path.name}")


# ── LOAD CORE TABLES ─────────────────────────────────────────────────────────
# Each CSV file is one "table". We load them all first, then join them.

print("Loading ESCO tables...")

# All ESCO occupations (European job titles like "software developer", "nurse")
occ_df = pd.read_csv(ESCO_DIR / "occupations_en.csv")

# All ESCO skills and competences (e.g. "use Python", "communicate with patients")
skills_df = pd.read_csv(ESCO_DIR / "skills_en.csv")

# The relationship table: which skills belong to which occupation
# e.g. "software developer" requires "use Python" as an essential skill
rel_df = pd.read_csv(ESCO_DIR / "occupationSkillRelations_en.csv")

# ISCO groups: the broader job category hierarchy (e.g. ISCO group 2512 = "Software developers")
isco_df = pd.read_csv(ESCO_DIR / "ISCOGroups_en.csv")

# Skill groups: the broader skill category hierarchy
# (e.g. "use Python" belongs to "programming languages" group)
skill_groups_df = pd.read_csv(ESCO_DIR / "skillGroups_en.csv")

print(f"  Occupations: {len(occ_df)}")
print(f"  Skills: {len(skills_df)}")
print(f"  Occ-skill relations: {len(rel_df)}")
print(f"  ISCO groups: {len(isco_df)}")


# ── 1. OCCUPATIONS ───────────────────────────────────────────────────────────
# Goal: one row per occupation, with the ISCO group name added.
# Raw data only has iscoGroup as a code (e.g. 2512), we want the label too.

print("\n[1] ESCO Occupations")

# Build a small lookup: iscoGroup code → iscoGroup label
# e.g. 2512 → "Software developers"
isco_lookup = isco_df[["code", "preferredLabel"]].rename(
    columns={"code": "iscoGroup", "preferredLabel": "iscoGroupLabel"}
)

# Merge the ISCO group label onto each occupation
# (left join: keep all occupations, add the group label where available)
occ_out = occ_df.merge(isco_lookup, on="iscoGroup", how="left")

# Keep only the useful columns (drop internal URIs, database IDs, etc.)
keep_occ = ["preferredLabel", "altLabels", "definition", "description",
            "scopeNote", "code", "naceCode", "iscoGroupLabel", "regulatedProfessionNote"]
occ_out = occ_out[[c for c in keep_occ if c in occ_out.columns]]

# Rename preferredLabel → occupation_title for consistency with O*NET naming
occ_out = occ_out.rename(columns={"preferredLabel": "occupation_title"})

# Drop rows with no title (can't embed something with no name)
occ_out = occ_out[occ_out["occupation_title"].notna()]
save(occ_out, "occupations")


# ── 2. OCCUPATION–SKILL RELATIONS ────────────────────────────────────────────
# Goal: one row per occupation-skill pair, fully labelled.
# This is the most useful table for career matching — it tells you
# exactly which skills each job requires.
#
# Example output row:
#   occupation_title: "Software developer"
#   skill_title: "use Python"
#   relation_type: "essential"  (vs "optional")
#   skillType: "skill/competence"
#   skill_description: "Write and execute Python code..."

print("\n[2] ESCO Occupation–Skill Relations")

# Build a skill lookup table with all the details we want to attach
skill_lookup = skills_df[["conceptUri", "preferredLabel", "skillType",
                           "reuseLevel", "description", "scopeNote", "definition"]].copy()
skill_lookup = skill_lookup.rename(columns={
    "conceptUri": "skillUri",           # the unique ID used to join to rel_df
    "preferredLabel": "skill_title",
    "description": "skill_description",
    "scopeNote": "skill_scopeNote",
    "definition": "skill_definition",
})

# Join: for each occupation-skill relation, attach the full skill details
# rel_df has skillUri which matches skill_lookup's skillUri
rel_out = rel_df.merge(skill_lookup, on="skillUri", how="left")

# Rename for consistency
rel_out = rel_out.rename(columns={
    "occupationLabel": "occupation_title",
    "relationType": "relation_type",    # "essential" or "optional"
})

# Keep only the useful columns
keep_rel = ["occupation_title", "relation_type", "skillType",
            "skill_title", "skill_description", "skill_scopeNote",
            "skill_definition", "reuseLevel"]
rel_out = rel_out[[c for c in keep_rel if c in rel_out.columns]]

# Drop rows missing either the occupation or the skill name
rel_out = rel_out[rel_out["occupation_title"].notna() & rel_out["skill_title"].notna()]
save(rel_out, "occupation_skills")


# ── 3. SKILLS STANDALONE ────────────────────────────────────────────────────
# Goal: one row per skill, with its group/category label attached.
# Useful for skill-only searches (e.g. "find skills similar to machine learning").
#
# ESCO has a skill hierarchy: skills → skill groups → broader groups
# We attach the immediate parent group label to each skill.

print("\n[3] ESCO Skills")

# Build a skill group lookup: broaderUri → group label
# e.g. "http://esco/.../programming" → "Programming languages"
sg_lookup = skill_groups_df[["conceptUri", "preferredLabel"]].rename(
    columns={"conceptUri": "broaderUri", "preferredLabel": "skillGroupLabel"}
)

# broaderRelationsSkillPillar links each skill to its broader (parent) skill group
broader_df = pd.read_csv(ESCO_DIR / "broaderRelationsSkillPillar_en.csv")
print("  broaderRelations cols:", broader_df.columns.tolist())

# Start with core skill fields
skills_out = skills_df[["conceptUri", "preferredLabel", "skillType", "reuseLevel",
                          "description", "scopeNote", "definition", "altLabels"]].copy()
skills_out = skills_out.rename(columns={"preferredLabel": "skill_title"})

if "conceptUri" in broader_df.columns and "broaderUri" in broader_df.columns:
    # Join: attach each skill's parent group URI
    skills_out = skills_out.merge(
        broader_df[["conceptUri", "broaderUri"]].drop_duplicates("conceptUri"),
        on="conceptUri", how="left"
    )
    # Then join: replace the parent group URI with the group's readable label
    skills_out = skills_out.merge(sg_lookup, on="broaderUri", how="left")
    # Drop the raw URIs — we only need the labels
    skills_out = skills_out.drop(columns=["broaderUri", "conceptUri"], errors="ignore")
else:
    skills_out = skills_out.drop(columns=["conceptUri"], errors="ignore")

save(skills_out, "skills")


print("\nESCO flatten complete.")
