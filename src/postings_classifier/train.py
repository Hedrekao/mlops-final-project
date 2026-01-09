import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from postings_classifier.data import JobPostingsDataModule
from postings_classifier.model import JobPostingsClassifier


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> float:
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
    datamodule.setup(stage="train")

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

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        default_root_dir=cfg.trainer.default_root_dir,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=datamodule)

    print("\n--- Testing best model ---")
    datamodule.setup(stage="test")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"\nBest model saved to: {checkpoint_callback.best_model_path}")
    last_path = checkpoint_callback.last_model_path
    print(f"Last model saved to: {last_path if last_path else 'N/A (best checkpoint is from final epoch)'}")

    val_acc = trainer.callback_metrics.get("val_acc", 0.0)
    return val_acc.item() if hasattr(val_acc, "item") else val_acc


if __name__ == "__main__":
    train()
