import os
import torch
from torch import nn
import lightning as L
from typing import Any, Dict, Optional
from transformers import AutoModel, get_linear_schedule_with_warmup
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score


class JobPostingsClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        freeze_encoder: bool = False,
        total_training_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        hf_path = os.getenv("HF_MODEL_PATH", model_name)
        self.encoder = AutoModel.from_pretrained(
            hf_path,
            local_files_only=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size, num_labels),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()
        self.encoder.train()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tokenized input IDs.
            attention_mask: Attention mask.

        Returns:
            Logits tensor of shape (batch_size, num_labels).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared step for train/val/test.

        Args:
            batch: Dictionary with input_ids, attention_mask, labels.

        Returns:
            Tuple of (loss, predictions, labels).
        """
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        preds = torch.argmax(logits, dim=1)
        return loss, preds, batch["labels"]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Training loss.
        """
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and scheduler.

        Returns:
            Dictionary with optimizer and lr_scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.total_training_steps:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.total_training_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return {"optimizer": optimizer}
    @classmethod
    def load_from_checkpoint(  # type: ignore[override]
        cls,
        checkpoint_path: str,
        map_location: Optional[str] = "cpu",
        strict: bool = True,
        **kwargs: Any,
    ) -> "JobPostingsClassifier":
        """Load a JobPostingsClassifier from a checkpoint.

        This method is tolerant to different checkpoint formats:
        - If the checkpoint is a Lightning checkpoint (contains hyper-parameters),
          delegate to Lightning's load_from_checkpoint.
        - If the checkpoint is a plain state_dict or contains only "state_dict",
          construct the model from saved hyperparameters if available and load the state dict.

        Args:
            checkpoint_path: Path to the checkpoint file.
            map_location: Map location for torch.load (default: "cpu").
            strict: Passed to load_state_dict when loading a plain state dict.
            **kwargs: Extra kwargs forwarded to Lightning loader.

        Returns:
            Instantiated JobPostingsClassifier with weights loaded.
        """
        chk = torch.load(checkpoint_path, map_location=map_location)

        # Lightning-style checkpoint: delegate to Lightning loader
        if isinstance(chk, dict) and (
            "hyper_parameters" in chk or "hparams" in chk or "state_dict" in chk and "hyper_parameters" in chk
        ):
            return super(JobPostingsClassifier, cls).load_from_checkpoint(
                checkpoint_path, map_location=map_location, **kwargs
            )

        # Plain state_dict or dict containing state_dict
        state_dict = chk.get("state_dict", chk) if isinstance(chk, dict) else chk
        saved_hparams: Dict[str, Any] = {}
        if isinstance(chk, dict):
            saved_hparams = chk.get("hparams", {}) or chk.get("hyper_parameters", {}) or {}

        # Construct model with saved hyperparameters where possible
        model = cls(**{**saved_hparams})
        model.load_state_dict(state_dict, strict=strict)
        return model


if __name__ == "__main__":
    model = JobPostingsClassifier()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 128)),
        "attention_mask": torch.ones(2, 128, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }
    logits = model(batch["input_ids"], batch["attention_mask"])
    print(f"Output shape: {logits.shape}")
