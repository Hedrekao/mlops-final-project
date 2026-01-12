"""Minimal FastAPI app for the postings_classifier project.

This provides a lightweight `/health` endpoint and a `/predict` endpoint
that accepts a short piece of text and returns a dummy prediction. The
implementation avoids heavy imports at module import time so the container
can build even before optional ML dependencies are available.
"""
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel


class TextIn(BaseModel):
    text: str


app = FastAPI(title="Postings Classifier (dummy)")


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TextIn) -> Dict[str, object]:
    """Return a simple rule-based dummy prediction.

    If the input text contains the word "fake" (case-insensitive) the
    model returns label "fake"; otherwise "real". This keeps the endpoint
    usable for smoke-tests without requiring a trained model.
    """
    text = payload.text or ""
    lower = text.lower()
    if "fake" in lower:
        label = "fake"
        score = 0.99
    else:
        label = "real"
        score = 0.75

    return {"label": label, "score": score, "text": payload.text}
