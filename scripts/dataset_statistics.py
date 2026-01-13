import sys
from pathlib import Path

import pandas as pd
from loguru import logger


def compute_dataset_statistics(raw_data_path: str = "data/raw/fake_real_job_postings_3000x25.csv") -> dict:
    """
    Compute statistics for the job postings dataset.

    Args:
        raw_data_path: Path to the raw CSV dataset

    Returns:
        Dictionary containing dataset statistics
    """
    raw_path = Path(raw_data_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found at {raw_path}")

    logger.info(f"Loading dataset from {raw_path}")
    df = pd.read_csv(raw_path)

    stats = {
        "total_samples": len(df),
        "num_features": len(df.columns),
        "fraudulent_count": int(df["is_fake"].sum()),
        "legitimate_count": int((df["is_fake"] == 0).sum()),
        "fraudulent_percentage": float(df["is_fake"].mean() * 100),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
    }

    return stats


def print_statistics_report(stats: dict) -> None:
    """
    Print a formatted statistics report.

    Args:
        stats: Dictionary containing dataset statistics
    """
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS REPORT")
    logger.info("=" * 60)
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Number of features: {stats['num_features']}")
    logger.info("-" * 60)
    logger.info("Class Distribution:")
    logger.info(f"  Legitimate postings: {stats['legitimate_count']}")
    logger.info(f"  Fraudulent postings: {stats['fraudulent_count']}")
    logger.info(f"  Fraudulent percentage: {stats['fraudulent_percentage']:.2f}%")
    logger.info("-" * 60)
    logger.info(f"Total missing values: {stats['total_missing']}")

    if stats["total_missing"] > 0:
        logger.info("Missing values per column:")
        for col, count in stats["missing_values"].items():
            if count > 0:
                logger.info(f"  {col}: {count}")

    logger.info("=" * 60)


def print_markdown_report(stats: dict) -> None:
    """
    Print a markdown-formatted statistics report.

    Args:
        stats: Dictionary containing dataset statistics
    """
    print("### Dataset Overview")
    print(f"- **Total samples**: {stats['total_samples']}")
    print(f"- **Number of features**: {stats['num_features']}")
    print()
    print("### Class Distribution")
    print(f"- **Legitimate postings**: {stats['legitimate_count']}")
    print(f"- **Fraudulent postings**: {stats['fraudulent_count']}")
    print(f"- **Fraudulent percentage**: {stats['fraudulent_percentage']:.2f}%")
    print()
    print("### Data Quality")
    print(f"- **Total missing values**: {stats['total_missing']}")

    if stats["total_missing"] > 0:
        print()
        print("**Missing values by column:**")
        print()
        print("| Column | Missing Count |")
        print("|--------|---------------|")
        for col, count in stats["missing_values"].items():
            if count > 0:
                print(f"| {col} | {count} |")


def validate_dataset(stats: dict) -> bool:
    """
    Validate dataset meets quality requirements.

    Args:
        stats: Dictionary containing dataset statistics

    Returns:
        True if validation passes, False otherwise
    """
    validation_passed = True

    if stats["total_samples"] < 100:
        logger.error(f"Dataset too small: {stats['total_samples']} samples (minimum: 100)")
        validation_passed = False

    if stats["fraudulent_count"] == 0 or stats["legitimate_count"] == 0:
        logger.error("Dataset missing one of the classes")
        validation_passed = False

    imbalance_ratio = min(stats["fraudulent_count"], stats["legitimate_count"]) / max(
        stats["fraudulent_count"], stats["legitimate_count"]
    )
    if imbalance_ratio < 0.01:
        logger.warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.3f})")

    if validation_passed:
        logger.success("Dataset validation passed")
    else:
        logger.error("Dataset validation failed")

    return validation_passed


if __name__ == "__main__":
    markdown_mode = "--markdown" in sys.argv

    stats = compute_dataset_statistics()

    if markdown_mode:
        print_markdown_report(stats)
    else:
        print_statistics_report(stats)

    validation_passed = validate_dataset(stats)

    if not validation_passed:
        exit(1)
