"""Package metadata and defaults for postings_classifier."""

from pathlib import Path

# Default locations used across the package and tests
RAW_DATA_PATH = Path("data/raw/fake_real_job_postings_3000x25.csv")
PROCESSED_DATA_DIR = Path("data/processed")

__all__ = ["RAW_DATA_PATH", "PROCESSED_DATA_DIR"]
