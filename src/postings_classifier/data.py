from pathlib import Path
import time
import multiprocessing

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import lightning as L
from loguru import logger


class JobPostingsDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return int(self.labels.size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a single sample as a dict of tensors."""
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
        logger.info("Preprocessing data from {}", self.raw_path)
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

        logger.info("Saved processed data to {}", self.processed_path)
        logger.info("Splits -> Train: {}, Val: {}, Test: {}", len(train_idx), len(val_idx), len(test_idx))

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

    def _get_dataloader_kwargs(self, shuffle: bool = False) -> dict:
        """Get optimized DataLoader kwargs following DTU MLOps S9.

        Implements distributed data loading best practices:
        - pin_memory: Lock data in host memory for faster GPU transfer
        - persistent_workers: Keep workers alive between epochs
        - prefetch_factor: Control batch queue size

        Returns:
            Dictionary of optimized kwargs for DataLoader.
        """
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
        }

        # GPU optimization: pin memory for faster host->device transfer
        gpu_available = torch.cuda.is_available()
        if self.num_workers > 0:
            kwargs["pin_memory"] = gpu_available
            kwargs["prefetch_factor"] = 2
            kwargs["persistent_workers"] = True

        return kwargs

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader with optimized settings."""
        kwargs = self._get_dataloader_kwargs(shuffle=True)
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader with optimized settings."""
        kwargs = self._get_dataloader_kwargs(shuffle=False)
        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader with optimized settings."""
        kwargs = self._get_dataloader_kwargs(shuffle=False)
        return DataLoader(self.test_dataset, **kwargs)


if __name__ == "__main__":
    dm = JobPostingsDataModule()
    dm.prepare_data()
    dm.setup()
    logger.info("Train batches: {}", len(dm.train_dataloader()))
    logger.info("Val batches: {}", len(dm.val_dataloader()))
    logger.info("Test batches: {}", len(dm.test_dataloader()))


# ============================================================================
# DISTRIBUTED DATA LOADING - DTU MLOps S9
# ============================================================================
# Following: https://skaftenicki.github.io/dtu_mlops/s9_scalable_applications/data_loading/


def get_cpu_cores() -> int:
    """Get the number of CPU cores available.

    Returns:
        Number of CPU cores.
    """
    cores = multiprocessing.cpu_count()
    logger.info(f"Number of cores: {cores}, Number of threads: {2 * cores}")
    return cores


def benchmark_dataloader(
    dataloader: DataLoader,
    num_workers: int,
    num_batches: int = 100,
    num_runs: int = 5,
) -> dict[str, float]:
    """Benchmark dataloader performance with different worker configurations.

    Following the DTU MLOps exercise on distributed data loading.

    Args:
        dataloader: DataLoader to benchmark.
        num_workers: Number of workers used.
        num_batches: Number of batches to process per run.
        num_runs: Number of times to repeat the benchmark.

    Returns:
        Dictionary with timing statistics (mean, std, min, max).
    """
    times = []

    for run in range(num_runs):
        start_time = time.time()
        batch_count = 0

        for batch in dataloader:
            batch_count += 1
            if batch_count >= num_batches:
                break

        elapsed = time.time() - start_time
        times.append(elapsed)
        logger.info(f"Run {run + 1}/{num_runs} with {num_workers} workers: {elapsed:.2f}s")

    return {
        "num_workers": num_workers,
        "mean": sum(times) / len(times),
        "std": (sum((x - (sum(times) / len(times))) ** 2 for x in times) / len(times)) ** 0.5,
        "min": min(times),
        "max": max(times),
    }


def benchmark_different_workers(
    datamodule: JobPostingsDataModule,
    max_workers: int | None = None,
    num_batches: int = 100,
    num_runs: int = 5,
) -> list[dict]:
    """Benchmark dataloader with different numbers of workers.

    This demonstrates the trade-off between parallelization and communication
    overhead as discussed in the DTU MLOps course.

    Args:
        datamodule: JobPostingsDataModule instance.
        max_workers: Maximum number of workers to test. If None, use number of cores.
        num_batches: Number of batches per run.
        num_runs: Number of benchmark runs.

    Returns:
        List of dictionaries with timing results for each worker configuration.
    """
    if max_workers is None:
        max_workers = get_cpu_cores()

    results = []

    logger.info("Starting dataloader benchmarking...")
    logger.info(f"Testing with 1 to {max_workers} workers")
    logger.info(f"Each test: {num_batches} batches x {num_runs} runs")

    for num_workers in range(0, max_workers + 1):
        logger.info(f"\n--- Testing with {num_workers} workers ---")

        # Create dataloader with specific number of workers
        train_loader = DataLoader(
            datamodule.train_dataset,
            batch_size=datamodule.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        result = benchmark_dataloader(train_loader, num_workers, num_batches, num_runs)
        results.append(result)

        logger.info(f"Mean: {result['mean']:.2f}s ± {result['std']:.2f}s")

    return results


def create_optimized_dataloader(
    dataset: JobPostingsDataset,
    batch_size: int,
    num_workers: int | None = None,
    shuffle: bool = True,
    use_pin_memory: bool = True,
    use_persistent_workers: bool = True,
) -> DataLoader:
    """Create an optimized DataLoader following DTU MLOps best practices.

    Key optimizations:
    - num_workers: Parallel data loading (set based on CPU cores)
    - pin_memory: Lock data in host memory for faster GPU transfer
    - persistent_workers: Reduce worker startup overhead
    - prefetch_factor: Control queue size for pipelining

    Args:
        dataset: Dataset to load.
        batch_size: Batch size.
        num_workers: Number of workers. If None, use CPU count / 2.
        shuffle: Whether to shuffle data.
        use_pin_memory: Whether to pin memory (useful for GPU training).
        use_persistent_workers: Keep workers alive between epochs.

    Returns:
        Optimized DataLoader.
    """
    if num_workers is None:
        num_workers = max(0, get_cpu_cores() // 2)

    logger.info(f"Creating DataLoader with {num_workers} workers (batch_size={batch_size})")

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
    }

    # pin_memory is beneficial when using GPU
    if num_workers > 0:
        kwargs["pin_memory"] = use_pin_memory
        kwargs["prefetch_factor"] = 2
        if use_persistent_workers:
            kwargs["persistent_workers"] = True

    return DataLoader(dataset, **kwargs)
