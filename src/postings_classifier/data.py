from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import lightning as L


class JobPostingsDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


class JobPostingsDataModule(L.LightningDataModule):
    def __init__(
        self,
        raw_path: str = "data/raw/fake_real_job_postings_3000x25.csv",
        processed_path: str = "data/processed",
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 16,
        max_length: int = 256,
        num_workers: int = 0,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> None:
        super().__init__()
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.train_dataset: JobPostingsDataset | None = None
        self.val_dataset: JobPostingsDataset | None = None
        self.test_dataset: JobPostingsDataset | None = None

    def prepare_data(self) -> None:
        """Download tokenizer and preprocess data if needed."""
        AutoTokenizer.from_pretrained(self.model_name)

        if not self._processed_data_exists():
            self._preprocess_and_save()

    def _processed_data_exists(self) -> bool:
        """Check if processed data exists."""
        return (
            (self.processed_path / "train_input_ids.pt").exists()
            and (self.processed_path / "val_input_ids.pt").exists()
            and (self.processed_path / "test_input_ids.pt").exists()
        )

    def _preprocess_and_save(self) -> None:
        """Preprocess raw data and save as tensors."""
        print("Preprocessing data...")
        df = pd.read_csv(self.raw_path)

        texts = (
            df["job_title"].fillna("")
            + " [SEP] "
            + df["job_description"].fillna("")
            + " [SEP] "
            + df["requirements"].fillna("")
        ).tolist()
        labels = df["is_fake"].tolist()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        n = len(labels)
        indices = torch.randperm(n)
        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        self.processed_path.mkdir(parents=True, exist_ok=True)

        torch.save(input_ids[train_idx], self.processed_path / "train_input_ids.pt")
        torch.save(attention_mask[train_idx], self.processed_path / "train_attention_mask.pt")
        torch.save(labels_tensor[train_idx], self.processed_path / "train_labels.pt")

        torch.save(input_ids[val_idx], self.processed_path / "val_input_ids.pt")
        torch.save(attention_mask[val_idx], self.processed_path / "val_attention_mask.pt")
        torch.save(labels_tensor[val_idx], self.processed_path / "val_labels.pt")

        torch.save(input_ids[test_idx], self.processed_path / "test_input_ids.pt")
        torch.save(attention_mask[test_idx], self.processed_path / "test_attention_mask.pt")
        torch.save(labels_tensor[test_idx], self.processed_path / "test_labels.pt")

        print(f"Saved processed data to {self.processed_path}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    def setup(self, stage: str | None = None) -> None:
        if stage == "train" or stage is None:
            self.train_dataset = JobPostingsDataset(
                input_ids=torch.load(self.processed_path / "train_input_ids.pt", weights_only=True),
                attention_mask=torch.load(self.processed_path / "train_attention_mask.pt", weights_only=True),
                labels=torch.load(self.processed_path / "train_labels.pt", weights_only=True),
            )
            self.val_dataset = JobPostingsDataset(
                input_ids=torch.load(self.processed_path / "val_input_ids.pt", weights_only=True),
                attention_mask=torch.load(self.processed_path / "val_attention_mask.pt", weights_only=True),
                labels=torch.load(self.processed_path / "val_labels.pt", weights_only=True),
            )

        if stage == "test" or stage is None:
            self.test_dataset = JobPostingsDataset(
                input_ids=torch.load(self.processed_path / "test_input_ids.pt", weights_only=True),
                attention_mask=torch.load(self.processed_path / "test_attention_mask.pt", weights_only=True),
                labels=torch.load(self.processed_path / "test_labels.pt", weights_only=True),
            )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dm = JobPostingsDataModule()
    dm.prepare_data()
    dm.setup()
    print(f"Train batches: {len(dm.train_dataloader())}")
    print(f"Val batches: {len(dm.val_dataloader())}")
    print(f"Test batches: {len(dm.test_dataloader())}")
