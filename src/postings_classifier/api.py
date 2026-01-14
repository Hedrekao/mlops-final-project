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

from transformers import AutoTokenizer

app = FastAPI(title="Postings Classifier")


class TextIn(BaseModel):
    text: str


# Globals populated lazily on first request
_MODEL: Optional[torch.nn.Module] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_LABEL_MAP = {0: "real", 1: "fake"}


def _find_checkpoint() -> Optional[str]:
    """Attempt to locate a model checkpoint.

    Search order:
    - `MODEL_CHECKPOINT` env var
    - `models/checkpoints/*` best checkpoints
    - return None if not found
    """
    env_path = os.getenv("MODEL_CHECKPOINT")
    if env_path and os.path.exists(env_path):
        return env_path

    # common Lightning extension
    candidates = glob.glob("models/checkpoints/*best*.ckpt") + glob.glob("models/checkpoints/*.ckpt")
    if not candidates:
        candidates = glob.glob("models/checkpoints/*")
    return candidates[0] if candidates else None


def _load_model_and_tokenizer(device: str = "cpu") -> tuple[Optional[torch.nn.Module], Optional[AutoTokenizer]]:
    """Try to load a Lightning checkpoint and tokenizer.

    Returns (model, tokenizer) or (None, None) on failure.
    """
    try:
        from postings_classifier.model import JobPostingsClassifier

        ckpt = _find_checkpoint()
        if not ckpt:
            return None, None

        # Try to use Lightning's loader first (works for standard Lightning checkpoints)
        try:
            model = JobPostingsClassifier.load_from_checkpoint(ckpt, map_location=device)  # type: ignore[arg-type]
        except Exception:
            # Fallback: try loading state_dict and construct model from saved hparams
            chk = torch.load(ckpt, map_location=device)
            state_dict = chk.get("state_dict", chk) if isinstance(chk, dict) else chk
            saved_hparams = {}
            if isinstance(chk, dict):
                saved_hparams = chk.get("hparams", {}) or chk.get("hyper_parameters", {}) or {}

            model = JobPostingsClassifier(**{**saved_hparams})
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        # Determine tokenizer name from saved hparams if available, otherwise default
        tok_name = getattr(model.hparams, "model_name", None) if hasattr(model, "hparams") else None
        if not tok_name:
            tok_name = os.getenv("TOKENIZER_NAME", "distilbert-base-uncased")

        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        return model, tokenizer
    except Exception:
        return None, None


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TextIn) -> Dict[str, object]:
    """Predict label and score given raw text.

    The endpoint will try to use a trained model if available, otherwise
    it falls back to a simple rule: presence of the word "fake" -> fake.
    """
    global _MODEL, _TOKENIZER

    text = (payload.text or "").strip()
    if not text:
        return {"label": "unknown", "score": 0.0, "text": text}

    # Lazy load on first call
    if _MODEL is None or _TOKENIZER is None:
        _MODEL, _TOKENIZER = _load_model_and_tokenizer(device="cpu")

    # If model available, run inference
    if _MODEL is not None and _TOKENIZER is not None:
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

    # Fallback rule-based predictor
    lower = text.lower()
    if "fake" in lower:
        return {"label": "fake", "score": 0.99, "text": text}
    return {"label": "real", "score": 0.75, "text": text}
