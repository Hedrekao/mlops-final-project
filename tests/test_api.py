from fastapi.testclient import TestClient
from postings_classifier.api import app


def test_health_endpoint():
	client = TestClient(app)
	resp = client.get("/health")
	assert resp.status_code == 200
	assert resp.json() == {"status": "ok"}


def test_predict_endpoint():
	client = TestClient(app)
	payload = {"text": "This looks like a fake posting"}
	resp = client.post("/predict", json=payload)
	assert resp.status_code == 200
	data = resp.json()
	assert set(data.keys()) >= {"label", "score", "text"}
	assert data["text"] == payload["text"]
	assert data["label"] in {"fake", "real"}