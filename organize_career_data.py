"""
organize_career_data.py
=======================

Organizes ESCO and O*NET project datasets into a clean RAG project layout:

    <destination>/
    └── data/
        ├── esco/
        ├── onet/
        └── mappings/

Only genuine data files are copied (CSV, TSV, JSON, JSONL, XLSX, Parquet,
Feather). Virtual environments, caches, notebooks, Python source, embeddings,
FAISS indexes, pickles, and model files are skipped.

Mapping / crosswalk files are detected by keywords in the filename or any
parent folder name and routed to the `mappings/` folder.

USAGE
-----
    python organize_career_data.py

Then paste each folder path when prompted. Windows paths with backslashes and
surrounding quotes are fine, e.g.

    "C:\\Users\\you\\projects\\esco_project"

The script never deletes anything; it only copies.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# File extensions that count as genuine structured datasets.
DATA_EXTENSIONS: set[str] = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
}

# Folder names to skip entirely (case-insensitive). If ANY ancestor folder
# between the source root and the file matches, the file is skipped.
SKIP_DIRS: set[str] = {
    # virtual environments
    ".venv", "venv", "env", ".env",
    # Python / Jupyter junk
    "__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache",
    # version control
    ".git", ".github", ".hg", ".svn",
    # editors / IDEs
    ".idea", ".vscode",
    # JS / build artefacts (just in case)
    "node_modules", "build", "dist",
    # caches and temp
    ".cache", "cache", "tmp", "temp", ".tmp",
    # things we explicitly don't want
    "models", "model", "checkpoints", "weights",
    "embeddings", "embedding",
    "faiss", "faiss_index", "faiss_indexes", "indexes", "index",
}

# File extensions to always skip, even if somehow they slipped in.
SKIP_EXTENSIONS: set[str] = {
    # code / notebooks
    ".py", ".pyc", ".pyo", ".pyd", ".ipynb",
    # pickles & serialized objects
    ".pkl", ".pickle", ".joblib",
    # vector / index files
    ".faiss", ".index", ".idx", ".ann",
    # numerical dumps / model weights
    ".npy", ".npz", ".pt", ".pth", ".bin", ".ckpt", ".safetensors",
    ".h5", ".hdf5", ".onnx",
    # logs & temp
    ".log", ".tmp", ".temp", ".swp", ".bak", ".lock",
    # local DBs
    ".db", ".sqlite", ".sqlite3",
    # archives (let the user unpack first if they really meant it)
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar",
    # binaries
    ".exe", ".dll", ".so", ".dylib",
}

# Keywords that mark a file as a mapping/crosswalk file (case-insensitive).
# Checked against both the filename and every parent folder name.
MAPPING_KEYWORDS: tuple[str, ...] = (
    "crosswalk",
    "mapping",
    "relation",
    "bridge",
    "concordance",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def prompt_path(label: str, must_exist: bool = True) -> Path:
    """Prompt the user for a folder path. Re-prompt until it's valid."""
    while True:
        raw = input(f"{label}: ").strip().strip('"').strip("'")
        if not raw:
            print("  Path cannot be empty. Try again.")
            continue
        path = Path(raw).expanduser().resolve()
        if must_exist:
            if not path.is_dir():
                print(f"  Not a folder (or doesn't exist): {path}")
                continue
        else:
            if path.exists() and not path.is_dir():
                print(f"  Path exists but isn't a folder: {path}")
                continue
        return path


def is_in_skipped_dir(file_path: Path, root: Path) -> bool:
    """True if any folder between `root` and `file_path` is in SKIP_DIRS."""
    try:
        relative = file_path.relative_to(root)
    except ValueError:
        # file_path isn't under root for some reason; treat as skip.
        return True
    # Look at every folder part except the filename itself.
    for part in relative.parts[:-1]:
        if part.lower() in SKIP_DIRS:
            return True
    return False


def is_data_file(file_path: Path) -> bool:
    """True if this file's extension marks it as a structured dataset."""
    ext = file_path.suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return False
    return ext in DATA_EXTENSIONS


def is_mapping_file(file_path: Path) -> bool:
    """True if the filename or any parent folder name contains a mapping keyword."""
    name = file_path.name.lower()
    if any(keyword in name for keyword in MAPPING_KEYWORDS):
        return True
    for parent in file_path.parents:
        if any(keyword in parent.name.lower() for keyword in MAPPING_KEYWORDS):
            return True
    return False


def unique_destination(dest_dir: Path, filename: str) -> Path:
    """Return a non-colliding destination path inside dest_dir.

    If `filename` already exists there, append `__1`, `__2`, ... before the
    extension so we never silently overwrite an existing file.
    """
    target = dest_dir / filename
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    counter = 1
    while True:
        candidate = dest_dir / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def collect_and_copy(
    source: Path,
    default_dest: Path,
    mapping_dest: Path,
    report: dict,
    label: str,
) -> None:
    """Walk `source`, copying qualifying files to `default_dest` or `mapping_dest`."""
    if not source.is_dir():
        print(f"[!] Source folder does not exist: {source}")
        return

    print(f"\nScanning {label.upper()} source: {source}")

    for file_path in source.rglob("*"):
        if not file_path.is_file():
            continue

        if is_in_skipped_dir(file_path, source):
            report["skipped"] += 1
            continue

        if not is_data_file(file_path):
            report["skipped"] += 1
            continue

        if is_mapping_file(file_path):
            target = unique_destination(mapping_dest, file_path.name)
            bucket = "mappings"
        else:
            target = unique_destination(default_dest, file_path.name)
            bucket = label

        try:
            shutil.copy2(file_path, target)
            report[bucket] += 1
            print(f"  [{bucket:>8}] {file_path.name}")
        except OSError as err:
            print(f"  [!] failed to copy {file_path}: {err}")
            report["errors"] += 1


def safety_check(data_root: Path, *sources: Path) -> bool:
    """Refuse to run if the destination is inside any source folder."""
    for src in sources:
        try:
            data_root.relative_to(src)
        except ValueError:
            continue  # data_root is NOT inside src — that's what we want
        print(
            f"[!] Destination ({data_root}) is inside source ({src}).\n"
            f"    This would cause the script to copy its own output. Aborting."
        )
        return False
    return True


def main() -> int:
    print("=" * 64)
    print("  Career RAG dataset organizer")
    print("=" * 64)
    print("Paste a folder path at each prompt. Windows paths and quoted")
    print("paths are both fine. Example:")
    print(r'  C:\Users\you\projects\esco_project')
    print()

    esco_src = prompt_path("ESCO project folder")
    onet_src = prompt_path("O*NET project folder")
    dest_root = prompt_path(
        "Destination project folder (the 'career_rag_system' folder)",
        must_exist=False,
    )

    # Build the destination tree.
    data_root = dest_root / "data"
    esco_dest = data_root / "esco"
    onet_dest = data_root / "onet"
    map_dest = data_root / "mappings"

    if not safety_check(data_root, esco_src, onet_src):
        return 1

    for folder in (esco_dest, onet_dest, map_dest):
        folder.mkdir(parents=True, exist_ok=True)

    report = {"esco": 0, "onet": 0, "mappings": 0, "skipped": 0, "errors": 0}

    collect_and_copy(esco_src, esco_dest, map_dest, report, label="esco")
    collect_and_copy(onet_src, onet_dest, map_dest, report, label="onet")

    # ----- Final report -----
    print("\n" + "=" * 64)
    print("  Done")
    print("=" * 64)
    print(f"ESCO files copied:    {report['esco']}")
    print(f"O*NET files copied:   {report['onet']}")
    print(f"Mapping files copied: {report['mappings']}")
    print(f"Skipped files:        {report['skipped']}")
    if report["errors"]:
        print(f"Errors:               {report['errors']}")
    print(f"\nOutput tree: {data_root}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(130)