"""FastAPI app for inference using the trained JobPostingsClassifier.

This module lazily attempts to load a trained checkpoint and tokenizer
from the repository. If no usable checkpoint is found the endpoint falls
back to a small rule-based predictor so the API is usable for smoke-tests.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
import pandas as pd
from google.cloud import storage

from transformers import AutoTokenizer

app = FastAPI(title="Postings Classifier")
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)  # Changed from DEBUG to INFO


class TextIn(BaseModel):
    text: str


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    load_error: str


# GCS configuration
# Use the existing project bucket for prediction logging.
BUCKET_NAME = "jop-postings-mlops-data"

# Globals populated lazily on first request
_MODEL: Optional[torch.nn.Module] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_LOAD_ERROR: Optional[str] = None
_LABEL_MAP = {0: "real", 1: "fake"}
_PREDICTION_DB = Path("prediction_database.csv")


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
    candidates = glob.glob("models/checkpoints/*best*.ckpt") + glob.glob("models/checkpoints/*.ckpt")
    if candidates:
        logger.info("Found local checkpoint: %s", candidates[0])
        return candidates[0]

    # Search GCS mount paths (Cloud Run with mounted bucket)
    gcs_candidates = glob.glob("/gcs/*/models/checkpoints/*best*.ckpt") + glob.glob("/gcs/*/models/checkpoints/*.ckpt")
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
        tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=False)

        logger.info("Successfully loaded tokenizer")
        return model, tokenizer
    except Exception as e:
        logger.exception("Failed to load model/tokenizer: %s", e)
        return None, None


def _save_prediction(text: str, label: str, score: float) -> None:
    """Save prediction to CSV locally or upload to GCS."""
    try:
        timestamp = datetime.now().isoformat()
        prediction_data = {
            "text": text,
            "label": label,
            "score": score,
            "timestamp": timestamp,
        }

        # Save to GCS (following DTU example)
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
        blob = bucket.blob(f"predictions/prediction_{safe_timestamp}.json")
        blob.upload_from_string(json.dumps(prediction_data))
        logger.info("Prediction saved to GCS bucket.")
    except Exception as e:
        logger.exception(f"Error saving prediction: {e}")


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
def predict(payload: TextIn, background_tasks: BackgroundTasks) -> Dict[str, object]:
    """Predict label and score given raw text.

    The endpoint will try to use a trained model if available, otherwise
    it falls back to a simple rule: presence of the word "fake" -> fake.
    Predictions are saved to a database in the background for monitoring.
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

            background_tasks.add_task(_save_prediction, text, label, score)
            return {"label": label, "score": score, "text": text}
        except Exception as e:
            logger.exception("Exception during inference: %s", e)
            return {"label": "error", "score": 0.0, "text": text, "error": str(e)}

    # Fallback rule-based predictor
    lower = text.lower()
    if "fake" in lower:
        background_tasks.add_task(_save_prediction, text, "fake", 0.99)
        return {"label": "fake", "score": 0.99, "text": text}
    background_tasks.add_task(_save_prediction, text, "real", 0.75)
    return {"label": "real", "score": 0.75, "text": text}


def _load_predictions(n: int = None) -> pd.DataFrame:
    """Load predictions from GCS or local CSV.

    Try GCS first (cloud), fall back to local CSV (development).
    """
    predictions = []

    # Try GCS first
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix="predictions/"))
        blobs.sort(key=lambda x: x.updated, reverse=True)
        if n:
            blobs = blobs[:n]

        for blob in blobs:
            try:
                data = json.loads(blob.download_as_string().decode())
                predictions.append(data)
            except Exception as e:
                logger.warning(f"Failed to parse {blob.name}: {e}")

        if predictions:
            logger.debug(f"Loaded {len(predictions)} predictions from GCS")
            return pd.DataFrame(predictions)
    except Exception as e:
        logger.debug(f"GCS not available, trying local CSV: {e}")

    # Fall back to local CSV
    if _PREDICTION_DB.exists():
        try:
            df = pd.read_csv(_PREDICTION_DB)
            if n:
                df = df.tail(n)
            logger.debug(f"Loaded {len(df)} predictions from local CSV")
            return df
        except Exception as e:
            logger.exception(f"Failed to load local CSV: {e}")

    return pd.DataFrame()


# Monitoring endpoints
@app.get("/monitoring/stats")
def get_monitoring_stats() -> dict:
    """Get statistics about collected predictions (from GCS or local CSV)."""
    try:
        df = _load_predictions()

        if df.empty:
            return {"total_predictions": 0, "label_distribution": {}, "message": "No predictions collected yet"}

        label_counts = df["label"].value_counts().to_dict()

        return {
            "total_predictions": len(df),
            "label_distribution": label_counts,
            "average_score": float(df["score"].mean()),
            "min_score": float(df["score"].min()),
            "max_score": float(df["score"].max()),
        }
    except Exception as e:
        logger.exception(f"Error getting stats: {e}")
        return {"error": str(e)}


@app.get("/monitoring/report")
def get_monitoring_report(n: int = 100) -> HTMLResponse:
    """Generate a drift monitoring report (from GCS or local CSV)."""
    try:
        df = _load_predictions(n=n)

        if df.empty:
            return HTMLResponse(
                content="""
                <html>
                    <head><title>Monitoring - No Data</title></head>
                    <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
                        <h1 style="color: #ff6b6b;">📊 No Predictions Yet</h1>
                        <p style="font-size: 18px;">No prediction data available. Make some predictions first!</p>
                        <p style="color: #666; margin-top: 20px;">Use the /predict endpoint to create predictions.</p>
                    </body>
                </html>
                """,
                status_code=200,
            )

        if len(df) < 10:
            return HTMLResponse(
                content=f"""
                <html>
                    <head><title>Monitoring - Insufficient Data</title></head>
                    <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
                        <h1 style="color: #ffa500;">⚠️ Insufficient Predictions</h1>
                        <p style="font-size: 18px;">Need at least 10 predictions for monitoring.</p>
                        <p style="color: #666; margin-top: 20px;">
                            Current predictions: <strong>{len(df)}</strong><br>
                            Needed: <strong>10</strong><br>
                            Missing: <strong>{10 - len(df)}</strong>
                        </p>
                    </body>
                </html>
                """,
                status_code=200,
            )

        # Get statistics
        recent_df = df.tail(n)
        label_counts = recent_df["label"].value_counts().to_dict()
        avg_score = recent_df["score"].mean()

        # Create simple HTML report
        html_content = f"""
        <html>
            <head>
                <title>Prediction Monitoring Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 40px; background: #f5f5f5; }}
                    .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
                    .stat-box {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
                    .stat-label {{ font-weight: bold; color: #666; }}
                    .stat-value {{ font-size: 24px; color: #333; margin: 5px 0; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background: #4CAF50; color: white; }}
                    tr:hover {{ background: #f5f5f5; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 Prediction Monitoring Report</h1>

                    <div class="stat-box">
                        <div class="stat-label">Total Predictions Analyzed</div>
                        <div class="stat-value">{len(recent_df)}</div>
                    </div>

                    <div class="stat-box">
                        <div class="stat-label">Average Confidence Score</div>
                        <div class="stat-value">{avg_score:.3f}</div>
                    </div>

                    <h2>Label Distribution</h2>
                    <table>
                        <tr>
                            <th>Label</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
        """

        for label, count in label_counts.items():
            percentage = (count / len(recent_df)) * 100
            html_content += f"""
                        <tr>
                            <td>{label}</td>
                            <td>{count}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
            """

        html_content += """
                    </table>

                    <h2>Recent Predictions</h2>
                    <table>
                        <tr>
                            <th>Timestamp</th>
                            <th>Text (preview)</th>
                            <th>Label</th>
                            <th>Score</th>
                        </tr>
        """

        for _, row in recent_df.tail(10).iterrows():
            text_preview = row["text"][:50] + "..." if len(row["text"]) > 50 else row["text"]
            html_content += f"""
                        <tr>
                            <td>{row['timestamp']}</td>
                            <td>{text_preview}</td>
                            <td>{row['label']}</td>
                            <td>{row['score']:.3f}</td>
                        </tr>
            """

        html_content += """
                    </table>
                </div>
            </body>
        </html>
        """

        return HTMLResponse(content=html_content, status_code=200)

    except Exception as e:
        logger.exception(f"Error generating report: {e}")
        return HTMLResponse(
            content=f"""
            <html>
                <body style="font-family: Arial; padding: 40px; text-align: center;">
                    <h1 style="color: #ff6b6b;">❌ Error</h1>
                    <p>Error generating report: {str(e)}</p>
                </body>
            </html>
            """,
            status_code=500,
        )
