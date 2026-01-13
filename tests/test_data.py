import pandas as pd
import torch
from postings_classifier.data import JobPostingsDataModule, JobPostingsDataset


def test_jobpostingsdataset_basic():
    """Test basic behavior of `JobPostingsDataset`."""
    input_ids = torch.randint(0, 100, (10, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 2, (10,))

    ds = JobPostingsDataset(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert len(ds) == 10

    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert torch.equal(item["labels"], labels[0])


def test_datamodule_preprocessing(tmp_path):
    """Test that preprocessing correctly tokenizes and splits data.

    This test creates a small synthetic CSV dataset and verifies:
    - Data is tokenized correctly
    - Train/val/test splits are created with correct proportions
    - Labels are preserved correctly
    - All required files are saved
    """
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    processed_path = tmp_path / "processed"

    test_data = pd.DataFrame(
        {
            "job_title": ["Software Engineer", "Data Scientist", "Product Manager", "Scam Job", "Fake Position"] * 4,
            "job_description": ["Build software"] * 10 + ["Analyze data"] * 10,
            "requirements": ["5 years exp"] * 20,
            "is_fake": [0] * 10 + [1] * 10,
        }
    )
    csv_path = raw_path / "test_data.csv"
    test_data.to_csv(csv_path, index=False)

    dm = JobPostingsDataModule(
        raw_path=str(csv_path),
        processed_path=str(processed_path),
        model_name="distilbert-base-uncased",
        batch_size=4,
        max_length=64,
        train_split=0.6,
        val_split=0.2,
        test_split=0.2,
    )

    dm.prepare_data()

    assert (processed_path / "train_input_ids.pt").exists()
    assert (processed_path / "train_attention_mask.pt").exists()
    assert (processed_path / "train_labels.pt").exists()
    assert (processed_path / "val_input_ids.pt").exists()
    assert (processed_path / "val_attention_mask.pt").exists()
    assert (processed_path / "val_labels.pt").exists()
    assert (processed_path / "test_input_ids.pt").exists()
    assert (processed_path / "test_attention_mask.pt").exists()
    assert (processed_path / "test_labels.pt").exists()

    train_ids = torch.load(processed_path / "train_input_ids.pt", weights_only=True)
    train_labels = torch.load(processed_path / "train_labels.pt", weights_only=True)
    val_ids = torch.load(processed_path / "val_input_ids.pt", weights_only=True)
    val_labels = torch.load(processed_path / "val_labels.pt", weights_only=True)
    test_ids = torch.load(processed_path / "test_input_ids.pt", weights_only=True)
    test_labels = torch.load(processed_path / "test_labels.pt", weights_only=True)

    total_samples = len(train_labels) + len(val_labels) + len(test_labels)
    assert total_samples == 20

    assert len(train_labels) == 12
    assert len(val_labels) == 4
    assert len(test_labels) == 4

    assert train_ids.shape[1] == 64
    assert val_ids.shape[1] == 64
    assert test_ids.shape[1] == 64

    assert train_labels.dtype == torch.long
    assert set(train_labels.tolist() + val_labels.tolist() + test_labels.tolist()).issubset({0, 1})
