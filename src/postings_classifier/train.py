from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

from postings_classifier.data import JobPostingsDataModule
from postings_classifier.logging_utils import log_config, setup_logging
from postings_classifier.model import JobPostingsClassifier


def _build_profiler(cfg: DictConfig) -> AdvancedProfiler | PyTorchProfiler | None:
    """Create a Lightning profiler based on config."""
    profiler_type = cfg.trainer.profiler
    if profiler_type is None:
        return None

    profile_dir = cfg.trainer.profiler_dir
    profile_filename = cfg.trainer.profiler_filename

    Path(profile_dir).mkdir(parents=True, exist_ok=True)

    if profiler_type == "advanced":
        return AdvancedProfiler(dirpath=profile_dir, filename=f"{profile_filename}.txt")

    if profiler_type == "pytorch":
        activities = [ProfilerActivity.CPU]
        if str(cfg.trainer.accelerator).lower() in {"gpu", "cuda"}:
            activities.append(ProfilerActivity.CUDA)

        trace_dir = cfg.trainer.profiler_trace_dir
        if trace_dir:
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
        on_trace_ready = tensorboard_trace_handler(trace_dir) if trace_dir else None

        return PyTorchProfiler(
            dirpath=profile_dir,
            filename=profile_filename,
            schedule=schedule(
                wait=cfg.trainer.profiler_wait,
                warmup=cfg.trainer.profiler_warmup,
                active=cfg.trainer.profiler_active,
                repeat=cfg.trainer.profiler_repeat,
            ),
            record_memory=cfg.trainer.profiler_record_memory,
            activities=activities,
            on_trace_ready=on_trace_ready,
        )

    raise ValueError(f"Unsupported profiler type: {profiler_type}")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> float:
    setup_logging(level="INFO")
    log_config(cfg)

    L.seed_everything(cfg.seed)

    datamodule = JobPostingsDataModule(
        raw_path=cfg.data.raw_path,
        processed_path=cfg.data.processed_path,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
    )

    datamodule.prepare_data()
    datamodule.setup(stage=None)

    train_batches = len(datamodule.train_dataloader())
    val_batches = len(datamodule.val_dataloader())
    test_batches = len(datamodule.test_dataloader())

    logger.info(
        "Data prepared. Train/Val/Test batches: {}/{}/{}",
        train_batches,
        val_batches,
        test_batches,
    )

    total_steps = len(datamodule.train_dataloader()) * cfg.trainer.max_epochs

    model = JobPostingsClassifier(
        model_name=cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        freeze_encoder=cfg.model.freeze_encoder,
        total_training_steps=total_steps,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_acc:.4f}",
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=cfg.trainer.early_stopping_patience,
        verbose=True,
    )

    callbacks = [checkpoint_callback, early_stopping_callback]

    profiler = _build_profiler(cfg)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        default_root_dir=cfg.trainer.default_root_dir,
        callbacks=callbacks,
        profiler=profiler,
    )

    logger.info("Starting training for {} epochs", cfg.trainer.max_epochs)
    trainer.fit(model, datamodule=datamodule)

    logger.info("Testing best model")
    datamodule.setup(stage="test")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    logger.info("Best model saved to: {}", checkpoint_callback.best_model_path)
    last_path = checkpoint_callback.last_model_path
    logger.info(
        "Last model saved to: {}",
        last_path if last_path else "N/A (best checkpoint is from final epoch)",
    )

    val_acc = trainer.callback_metrics.get("val_acc", 0.0)
    return val_acc.item() if hasattr(val_acc, "item") else val_acc


if __name__ == "__main__":
    train()
