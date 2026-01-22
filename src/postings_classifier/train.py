import os
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler, PyTorchProfiler
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler
import wandb

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


def _build_wandb_logger(cfg: DictConfig, save_dir: str | None = None) -> WandbLogger | None:
    """Create a WandbLogger if enabled in config."""
    if not cfg.wandb.enabled:
        logger.info("W&B logging disabled")
        return None

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or os.getenv("WANDB_ENTITY"),
        tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model=False,  # We handle artifact upload manually for more control
        save_dir=save_dir,  # Important: use original dir, not Hydra's output dir
    )

    logger.info("W&B logging enabled - project: {}, entity: {}", cfg.wandb.project, cfg.wandb.entity)
    return wandb_logger


def _upload_checkpoint_artifact(
    checkpoint_path: str,
    wandb_logger: WandbLogger,
    artifact_name: str = "model-checkpoint",
    artifact_type: str = "model",
    aliases: list[str] | None = None,
) -> None:
    """Upload a checkpoint file as a W&B artifact."""
    if not checkpoint_path or not Path(checkpoint_path).exists():
        logger.warning("Checkpoint path not found, skipping artifact upload: {}", checkpoint_path)
        return

    aliases = aliases or ["latest"]
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(checkpoint_path, name=Path(checkpoint_path).name)

    wandb_logger.experiment.log_artifact(artifact, aliases=aliases)
    logger.info("Uploaded checkpoint artifact '{}' with aliases {}", artifact_name, aliases)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> float:
    # Store original working directory before Hydra changes it
    original_cwd = hydra.utils.get_original_cwd()

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
    wandb_logger = _build_wandb_logger(cfg, save_dir=original_cwd)

    # Watch model gradients/parameters if enabled
    if wandb_logger and cfg.wandb.watch_model:
        wandb_logger.watch(model, log="all", log_freq=100)

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
        logger=wandb_logger,
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

    # Upload best checkpoint to W&B as artifact
    if wandb_logger and cfg.wandb.log_model:
        _upload_checkpoint_artifact(
            checkpoint_path=checkpoint_callback.best_model_path,
            wandb_logger=wandb_logger,
            artifact_name="model-checkpoint",
            aliases=["best", "latest"],
        )

    # Finish W&B run
    if wandb_logger:
        wandb.finish()

    val_acc = trainer.callback_metrics.get("val_acc", 0.0)
    return float(val_acc.item() if hasattr(val_acc, "item") else val_acc)


if __name__ == "__main__":
    train()
