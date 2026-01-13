# Data Management with DVC

This project uses [DVC (Data Version Control)](https://dvc.org/) to manage datasets efficiently with Google Cloud Storage as the remote storage backend.

## Quick Start

### First-time Setup (After Cloning)

When you clone this repository, you need to pull the data from Google Cloud Storage:

```bash
# Pull all tracked data
uv run dvc pull
```

This will download:
- Raw dataset to `data/raw/`
- Processed data will be generated locally when you run training

### Regular Workflow

#### 1. After Pulling New Commits

If someone has updated the dataset and you pull their changes:

```bash
# Pull git changes as usual
git pull

# Pull updated data from GCS
uv run dvc pull
```

**Important**: Always run `dvc pull` after `git pull` to ensure your data matches the tracked versions.

#### 2. Updating the Dataset

If you need to modify the raw dataset:

```bash
# 1. Make changes to files in data/raw/
# 2. Add the changes to DVC tracking
uv run dvc add data/raw

# 3. Push data to GCS
uv run dvc push

# 4. Commit the updated .dvc metadata file
git add data/raw.dvc data/.gitignore
git commit -m "Update raw dataset"
git push
```

#### 3. Checking Data Status

To see if your local data matches the tracked version:

```bash
uv run dvc status
```

To see current DVC configuration:

```bash
uv run dvc config --list
```

## Data Directory Structure

```
data/
├── .gitignore          # Ignores raw/ and processed/ from git
├── raw/                # Raw data tracked by DVC
│   └── *.csv          # Raw CSV files
├── raw.dvc            # DVC metadata file (tracked by git)
└── processed/         # Generated data (NOT tracked by DVC)
    └── *.pt           # PyTorch tensor files
```

## Storage Strategy

### What is tracked by DVC (stored in GCS):
- **Raw data** (`data/raw/`): Source of truth, versioned in GCS

### What is NOT tracked (local only):
- **Processed data** (`data/processed/`): Generated locally from raw data during training

This approach ensures:
- Raw data is the single source of truth
- Reproducibility (processed data always generated from tracked raw data)
- Cost efficiency (no duplicate storage)
- Cleaner version control

## Google Cloud Storage Configuration

The project is configured to use:
- **Bucket**: `gs://jop-postings-mlops-data/dvc-store`
- **Version-aware storage**: Enabled (leverages GCS object versioning)

### Authentication

DVC uses your local `gcloud` authentication. Ensure you're authenticated:

```bash
gcloud auth application-default login
```

## Common Commands

| Command | Description |
|---------|-------------|
| `uv run dvc pull` | Download data from GCS |
| `uv run dvc push` | Upload data to GCS |
| `uv run dvc status` | Check if local data matches tracked versions |
| `uv run dvc add data/raw` | Track changes to raw data |
| `uv run dvc checkout` | Restore data to match .dvc file versions |

## Data Validation

The project includes automated data validation via GitHub Actions. When `.dvc` files are modified, the workflow:

1. Pulls data from GCS
2. Runs data statistics checks
3. Posts a report to the PR

You can manually run data validation:

```bash
uv run python scripts/dataset_statistics.py
```

## Troubleshooting

### Data out of sync

If your data doesn't match the expected version:

```bash
# Restore data to match .dvc files
uv run dvc checkout
```

### Missing data

If `data/raw/` is empty:

```bash
# Pull data from remote
uv run dvc pull
```

### Authentication errors

Ensure you're authenticated with GCP:

```bash
gcloud auth application-default login
gcloud auth list
```

### Cache issues

To force re-download from GCS:

```bash
uv run dvc pull --force
```

## Best Practices

1. **Always `dvc pull` after `git pull`** to sync data with code changes
2. **Never commit actual data files** to git (use DVC instead)
3. **Regenerate processed data** rather than tracking it
4. **Test data validation** before pushing dataset changes
5. **Document dataset changes** in commit messages
