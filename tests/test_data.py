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


def test_datamodule_setup_and_dataloader():
    """Smoke-test `JobPostingsDataModule` using the included processed tensors.

    This test avoids network calls by not invoking `prepare_data()` (which
    downloads a tokenizer). The repository already contains processed tensors
    under `data/processed/` so calling `setup()` should load them.
    """
    dm = JobPostingsDataModule(processed_path="data/processed", batch_size=4, num_workers=0)

    # Do not call prepare_data() to avoid network access; the processed files are present.
    dm.setup()

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    # Batch should contain the three tensors
    assert "input_ids" in batch and "attention_mask" in batch and "labels" in batch
    assert batch["input_ids"].ndim == 2 or batch["input_ids"].ndim == 3
    assert batch["labels"].dtype == torch.long
