"""FastAPI app for inference using the trained JobPostingsClassifier.

This module lazily attempts to load a trained checkpoint and tokenizer
from the repository. If no usable checkpoint is found the endpoint falls
back to a small rule-based predictor so the API is usable for smoke-tests.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import logging

from transformers import AutoTokenizer

app = FastAPI(title="Postings Classifier")
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.DEBUG)

class TextIn(BaseModel):
    text: str

class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    load_error: str

# Globals populated lazily on first request
_MODEL: Optional[torch.nn.Module] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_LOAD_ERROR: Optional[str] = None
_LABEL_MAP = {0: "real", 1: "fake"}


def _find_checkpoint() -> Optional[str]:
    """Attempt to locate a model checkpoint.

    Search order:
    - `MODEL_CHECKPOINT` env var
    - `models/checkpoints/*` local paths (development)
    - `/gcs/*/models/checkpoints/*` GCS mount paths (Cloud Run)
    - return None if not found
    """
    env_path = os.getenv("MODEL_CHECKPOINT")
    if env_path:
        logger.info("MODEL_CHECKPOINT env var set to: %s", env_path)
        # For GCS mounts, trust the path even if os.path.exists() is unreliable
        if env_path.startswith("/gcs/"):
            logger.info("Using GCS path: %s", env_path)
            return env_path
        if os.path.exists(env_path):
            logger.info("Checkpoint found at env path: %s", env_path)
            return env_path
        logger.warning("MODEL_CHECKPOINT env path does not exist: %s", env_path)

    # Search local checkpoints (development)
    candidates = (
        glob.glob("models/checkpoints/*best*.ckpt")
        + glob.glob("models/checkpoints/*.ckpt")
    )
    if candidates:
        logger.info("Found local checkpoint: %s", candidates[0])
        return candidates[0]

    # Search GCS mount paths (Cloud Run with mounted bucket)
    gcs_candidates = (
        glob.glob("/gcs/*/models/checkpoints/*best*.ckpt")
        + glob.glob("/gcs/*/models/checkpoints/*.ckpt")
    )
    if gcs_candidates:
        logger.info("Found checkpoint in GCS mount: %s", gcs_candidates[0])
        return gcs_candidates[0]

    # Fallback: any checkpoint in models/checkpoints/
    candidates = glob.glob("models/checkpoints/*")
    if candidates:
        logger.info("Found checkpoint (fallback): %s", candidates[0])
        return candidates[0]

    logger.warning("No checkpoint found in local paths or GCS mounts")
    return None


def _load_model_and_tokenizer(device: str = "cpu") -> tuple[Optional[torch.nn.Module], Optional[AutoTokenizer]]:
    """Try to load a Lightning checkpoint and tokenizer.

    Returns (model, tokenizer) or (None, None) on failure.
    """
    try:
        from postings_classifier.model import JobPostingsClassifier

        ckpt = _find_checkpoint()
        if not ckpt:
            logger.warning("No checkpoint path found")
            return None, None

        logger.info("Attempting to load checkpoint: %s", ckpt)

        # Try to use Lightning's loader first (works for standard Lightning checkpoints)
        try:
            model = JobPostingsClassifier.load_from_checkpoint(ckpt)  # type: ignore[arg-type]
            logger.info("Successfully loaded checkpoint using Lightning loader")
        except (TypeError, AttributeError) as e:
            # Fallback: try loading state_dict and construct model from saved hparams
            logger.info("Lightning loader failed, trying manual state_dict loading: %s", str(e))
            chk = torch.load(ckpt, map_location=device)
            state_dict = chk.get("state_dict", chk) if isinstance(chk, dict) else chk
            saved_hparams = {}
            if isinstance(chk, dict):
                saved_hparams = chk.get("hparams", {}) or chk.get("hyper_parameters", {}) or {}

            logger.info("Creating model with hparams: %s", saved_hparams)
            model = JobPostingsClassifier(**{**saved_hparams})
            model.load_state_dict(state_dict)
            logger.info("Successfully loaded state_dict")

        model.to(device)
        model.eval()

        # Determine tokenizer name from saved hparams if available, otherwise default
        tok_model_name = getattr(model.hparams, "model_name", None) if hasattr(model, "hparams") else None
        if not tok_model_name:
            tok_model_name = os.getenv("TOKENIZER_NAME", "distilbert-base-uncased")

        logger.info("Loading tokenizer: %s", tok_model_name)
        tok_path = os.getenv("HF_MODEL_PATH", tok_model_name)
        tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)


        logger.info("Successfully loaded tokenizer")
        return model, tokenizer
    except Exception as e:
        logger.exception("Failed to load model/tokenizer: %s", e)
        return None, None


@app.get("/")
def root():
    return {"service": "postings-classifier", "status": "ok"}


@app.get("/health", response_model=HealthOut)
def health():
    return {
        "status": "ok",
        "model_loaded": _MODEL is not None,
        "load_error": _LOAD_ERROR or "none",
    }


@app.post("/predict")
def predict(payload: TextIn) -> Dict[str, object]:
    """Predict label and score given raw text.

    The endpoint will try to use a trained model if available, otherwise
    it falls back to a simple rule: presence of the word "fake" -> fake.
    """
    global _MODEL, _TOKENIZER, _LOAD_ERROR

    text = (payload.text or "").strip()
    if not text:
        return {"label": "unknown", "score": 0.0, "text": text}

    # Lazy load on first call
    if _MODEL is None or _TOKENIZER is None:
        try:
            _MODEL, _TOKENIZER = _load_model_and_tokenizer(device="cpu")
            if _MODEL is None:
                logger.warning("Model loading returned None, will use fallback predictor")
                _LOAD_ERROR = "Model/tokenizer not available, using fallback"
        except Exception as e:
            logger.exception("Exception during model loading: %s", e)
            _LOAD_ERROR = f"Model loading failed: {str(e)}"
            _MODEL, _TOKENIZER = None, None

    # If model available, run inference
    if _MODEL is not None and _TOKENIZER is not None:
        try:
            inputs = _TOKENIZER(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                logits = _MODEL(inputs["input_ids"], inputs["attention_mask"])  # type: ignore[arg-type]
                probs = F.softmax(logits, dim=-1).cpu().squeeze(0)

            # assume binary. pick top
            if probs.ndim == 0:
                # scalar -> treat as single logit for class 1
                score = float(probs.item())
                label = _LABEL_MAP[1] if score >= 0.5 else _LABEL_MAP[0]
            else:
                top_idx = int(torch.argmax(probs).item())
                score = float(probs[top_idx].item())
                label = _LABEL_MAP.get(top_idx, str(top_idx))

            return {"label": label, "score": score, "text": text}
        except Exception as e:
            logger.exception("Exception during inference: %s", e)
            return {"label": "error", "score": 0.0, "text": text, "error": str(e)}

    # Fallback rule-based predictor
    lower = text.lower()
    if "fake" in lower:
        return {"label": "fake", "score": 0.99, "text": text}
    return {"label": "real", "score": 0.75, "text": text}
