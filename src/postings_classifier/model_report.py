import importlib
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import typer
from loguru import logger


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _try_get_batch(processed_path: str, batch_size: int) -> Optional[dict[str, torch.Tensor]]:
    """Try to load a batch from JobPostingsDataModule if available."""
    try:
        from postings_classifier.data import JobPostingsDataModule  # type: ignore

        dm = JobPostingsDataModule(processed_path=processed_path, batch_size=batch_size, num_workers=0)
        dm.setup()
        loader = dm.val_dataloader()
        batch = next(iter(loader))
        return batch
    except Exception as exc:
        logger.debug(f"Could not load real batch: {exc}")
        return None


def model_statistics(
    model_path: str = "postings_classifier.model.PostingsClassifier",
    checkpoint: Optional[str] = None,
    processed_path: str = "data/processed",
    batch_size: int = 8,
) -> None:
    """Inspect a model: summary, parameter counts, and example outputs.

    Args:
        model_path: Dotted path to the model class (e.g. package.module.ClassName).
        checkpoint: Optional path to a saved state_dict to load.
        processed_path: Path to processed data to attempt a real forward pass.
        batch_size: Batch size to use when fetching a sample batch.
    """
    module_path, _, class_name = model_path.rpartition(".")
    model: torch.nn.Module

    try:
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        model = ModelClass()
        logger.info(f"Imported model class {model_path}")
    except Exception:
        logger.warning(f"Could not import {model_path}, using a small fallback model.")
        model = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )

    if checkpoint:
        try:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            logger.info(f"Loaded checkpoint from {checkpoint}")
        except Exception as exc:
            logger.warning(f"Failed to load checkpoint {checkpoint}: {exc}")

    model.eval()
    total_params, trainable_params = _count_parameters(model)

    summary_lines = [
        f"Model: {model_path}",
        f"Module type: {model.__class__.__module__}.{model.__class__.__name__}",
        f"Total parameters: {total_params}",
        f"Trainable parameters: {trainable_params}",
    ]

    batch = _try_get_batch(processed_path, batch_size)
    if batch is not None:
        if "input_ids" in batch:
            inputs = batch["input_ids"]
            if inputs.ndim == 3:
                example_input = inputs.float().view(inputs.size(0), -1)
            else:
                example_input = inputs.float()
        else:
            example_input = next(iter(batch.values())).float()
        example_input = example_input[:batch_size]
    else:
        example_input = torch.randn(batch_size, 256)

    try:
        with torch.no_grad():
            outputs = model(example_input)
        if isinstance(outputs, torch.Tensor):
            out_np = outputs.detach().cpu().numpy().ravel()
            summary_lines.append(f"Output shape (example): {tuple(outputs.shape)}")
            plt.figure(figsize=(6, 4))
            plt.hist(out_np, bins=30)
            plt.title("Model output distribution (example batch)")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            Path("model_output_hist.png").write_bytes(b"")
            plt.savefig("model_output_hist.png")
            plt.close()
            summary_lines.append("Saved histogram to model_output_hist.png")
    except Exception as exc:
        logger.warning(f"Failed to run example forward pass: {exc}")
        summary_lines.append("Example forward pass failed")

    summary_text = "\n".join(summary_lines)
    Path("model_summary.txt").write_text(summary_text)
    print(summary_text)


if __name__ == "__main__":
    typer.run(model_statistics)
