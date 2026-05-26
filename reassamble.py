#!/usr/bin/env python3
"""
Requirements
------------
    pip install pandas pyarrow datasets huggingface_hub

For gated datasets, the user must first authenticate:
    huggingface-cli login

Usage
-----
    python reassemble.py \
        --manifest datasets_manifest.csv \
        --scores-dir ./scores \
        --out-dir ./reassembled

Author note
-----------
This is a reproduction utility, not part of the scientific contribution.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# CONFIG — set these three things to match your repository, then the script
# runs unchanged. They are the only values that depend on your file layout.
# --------------------------------------------------------------------------

# 1. The column present in BOTH your released score files and the original
#    HuggingFace datasets, used to join texts to scores (e.g. "uid", "id").
JOIN_KEY = "uid"

# 2. How your released scores are stored. Set ONE of these:
#    - if one score file per dataset, named <paper_key>.parquet:
SCORES_PER_DATASET = True
SCORE_FILE_SUFFIX = ".parquet"        # ".parquet" or ".csv"
#    - if instead a single combined score file, set SCORES_PER_DATASET = False
#      and give its path here, plus the column identifying the dataset:
COMBINED_SCORES_PATH = "scores/all_scores.parquet"
DATASET_ID_COLUMN = "paper_key"       # column in the combined file

# 3. Access values (from the manifest `access` column) that we CAN reassemble
#    automatically. Anything else is skipped with a warning.
REASSEMBLABLE_ACCESS_PREFIXES = ("public",)

# --------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("reassemble")


def load_scores(paper_key: str, scores_dir: Path) -> pd.DataFrame:
    """Load our released scores for one dataset.

    Returns a DataFrame containing at least JOIN_KEY plus the Perspective
    score columns and query metadata. Raises FileNotFoundError if missing.
    """
    if SCORES_PER_DATASET:
        path = scores_dir / f"{paper_key}{SCORE_FILE_SUFFIX}"
        if not path.exists():
            raise FileNotFoundError(f"no score file for '{paper_key}' at {path}")
        return pd.read_parquet(path) if SCORE_FILE_SUFFIX == ".parquet" \
            else pd.read_csv(path)
    # combined-file mode: load once, filter to this dataset
    combined = pd.read_parquet(COMBINED_SCORES_PATH)
    subset = combined[combined[DATASET_ID_COLUMN] == paper_key]
    if subset.empty:
        raise FileNotFoundError(f"no rows for '{paper_key}' in combined scores")
    return subset


def load_texts(hf_id: str) -> pd.DataFrame:
    """Download the original dataset texts from HuggingFace.

    Returns a DataFrame that must contain JOIN_KEY. The `datasets` import is
    local so the script can still run for offline/skipped datasets.
    """
    from datasets import load_dataset

    # trust_remote_code is needed for a few legacy loading scripts (e.g.
    # HateXplain). It only affects datasets that ship their own loader.
    ds = load_dataset(hf_id, trust_remote_code=True)
    # Most datasets expose a single split or a "train" split; concatenate all
    # available splits so no scored row is lost.
    frames = [split.to_pandas() for split in ds.values()]
    return pd.concat(frames, ignore_index=True)


def reassemble_one(row: pd.Series, scores_dir: Path, out_dir: Path) -> bool:
    """Reassemble a single dataset. Returns True on success, False if skipped.

    `row` is one record from the manifest CSV.
    """
    key = row["paper_key"]
    access = str(row.get("access", "")).strip().lower()

    # Skip anything not freely/automatically obtainable.
    if not access.startswith(REASSEMBLABLE_ACCESS_PREFIXES):
        log.warning("SKIP  %-22s access='%s' — obtain this dataset manually",
                    key, access)
        return False
    if not str(row.get("hf_id", "")).strip():
        log.warning("SKIP  %-22s no hf_id in manifest", key)
        return False

    try:
        scores = load_scores(key, scores_dir)
    except FileNotFoundError as e:
        log.warning("SKIP  %-22s %s", key, e)
        return False

    log.info("...   %-22s downloading texts from '%s'", key, row["hf_id"])
    texts = load_texts(row["hf_id"])

    # Both sides must carry the join key.
    for name, df in (("scores", scores), ("texts", texts)):
        if JOIN_KEY not in df.columns:
            log.error("FAIL  %-22s join key '%s' missing from %s",
                      key, JOIN_KEY, name)
            return False

    # Inner join: keep only rows we actually scored. A large drop here means
    # the identifier spaces disagree — worth investigating, so we report it.
    merged = scores.merge(texts, on=JOIN_KEY, how="inner", suffixes=("", "_text"))
    if len(merged) < len(scores):
        log.warning("      %-22s %d/%d scored rows matched a text",
                    key, len(merged), len(scores))

    out_path = out_dir / f"{key}.parquet"
    merged.to_parquet(out_path, index=False)
    log.info("OK    %-22s %d rows -> %s", key, len(merged), out_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="path to the dataset registry CSV")
    parser.add_argument("--scores-dir", required=True, type=Path,
                        help="directory containing the released score files")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="directory for the reassembled per-dataset files")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)

    # Only attempt rows that were actually released (accepted == "yes").
    released = manifest[manifest["accepted"].astype(str).str.lower() == "yes"]
    log.info("manifest: %d datasets marked accepted", len(released))

    ok = sum(reassemble_one(row, args.scores_dir, args.out_dir)
             for _, row in released.iterrows())
    log.info("done: %d/%d datasets reassembled (%d skipped)",
             ok, len(released), len(released) - ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())