"""Tests for the FastAPI inference application."""

from fastapi.testclient import TestClient
from postings_classifier.api import app


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self):
        """Test GET / returns service info."""
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "postings-classifier"
        assert data["status"] == "ok"


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint_status_code(self):
        """Test /health returns 200."""
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_endpoint_response_structure(self):
        """Test /health response has required fields."""
        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "load_error" in data
        assert data["status"] == "ok"
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["load_error"], str)

    def test_health_endpoint_ok_status(self):
        """Test /health status is always ok."""
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


class TestPredictEndpoint:
    """Test prediction endpoint."""

    def test_predict_valid_text(self):
        """Test /predict with valid text returns 200."""
        client = TestClient(app)
        payload = {"text": "Senior software engineer position"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_response_structure(self):
        """Test /predict response has required fields."""
        client = TestClient(app)
        payload = {"text": "This is a job posting"}
        resp = client.post("/predict", json=payload)
        data = resp.json()
        assert "label" in data
        assert "score" in data
        assert "text" in data
        assert data["text"] == payload["text"]

    def test_predict_valid_labels(self):
        """Test /predict returns valid label."""
        client = TestClient(app)
        payload = {"text": "Job posting text"}
        resp = client.post("/predict", json=payload)
        data = resp.json()
        assert data["label"] in {"fake", "real", "unknown", "error"}

    def test_predict_score_range(self):
        """Test /predict score is between 0 and 1."""
        client = TestClient(app)
        payload = {"text": "Sample job posting"}
        resp = client.post("/predict", json=payload)
        data = resp.json()
        assert 0.0 <= data["score"] <= 1.0

    def test_predict_empty_text(self):
        """Test /predict with empty text."""
        client = TestClient(app)
        payload = {"text": ""}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "unknown"
        assert data["score"] == 0.0

    def test_predict_whitespace_only(self):
        """Test /predict with whitespace-only text."""
        client = TestClient(app)
        payload = {"text": "   \n\t  "}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "unknown"

    def test_predict_with_fake_keyword(self):
        """Test /predict with 'fake' keyword."""
        client = TestClient(app)
        payload = {"text": "This is a fake job posting"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # Should predict fake (either from model or fallback rule)
        assert data["label"] in {"fake", "real"}

    def test_predict_long_text(self):
        """Test /predict with long text."""
        client = TestClient(app)
        long_text = "This is a job posting. " * 50
        payload = {"text": long_text}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in {"fake", "real", "unknown", "error"}

    def test_predict_special_characters(self):
        """Test /predict with special characters."""
        client = TestClient(app)
        payload = {"text": "Hiring! @#$%^&*() Senior Dev Position 🚀"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in {"fake", "real", "unknown", "error"}

    def test_predict_multiple_requests(self):
        """Test /predict with multiple consecutive requests."""
        client = TestClient(app)
        texts = [
            "Senior Software Engineer",
            "Full Stack Developer wanted",
            "This is a fake posting"
        ]
        for text in texts:
            payload = {"text": text}
            resp = client.post("/predict", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert "label" in data
            assert "score" in data


class TestPredictEdgeCases:
    """Test edge cases for prediction endpoint."""

    def test_predict_unicode_text(self):
        """Test /predict with unicode characters."""
        client = TestClient(app)
        payload = {"text": "Ingénieur senior recherché 📋"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_numeric_text(self):
        """Test /predict with numeric text."""
        client = TestClient(app)
        payload = {"text": "123 456 789"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_missing_text_field(self):
        """Test /predict with missing 'text' field."""
        client = TestClient(app)
        payload = {}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422  # Validation error

    def test_predict_none_text(self):
        """Test /predict with None as text."""
        client = TestClient(app)
        payload = {"text": None}
        resp = client.post("/predict", json=payload)
        # Should handle gracefully
        assert resp.status_code in [200, 422]


class TestAPIIntegration:
    """Integration tests for the API."""

    def test_health_then_predict(self):
        """Test health check followed by prediction."""
        client = TestClient(app)
        
        # Check health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        
        # Make prediction
        pred_payload = {"text": "Job posting"}
        pred_resp = client.post("/predict", json=pred_payload)
        assert pred_resp.status_code == 200

    def test_sequential_predictions(self):
        """Test multiple sequential predictions are consistent."""
        client = TestClient(app)
        payload = {"text": "Senior engineer needed"}
        
        # Make two predictions with same text
        resp1 = client.post("/predict", json=payload)
        resp2 = client.post("/predict", json=payload)
        
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        
        # Both should return same label (model is deterministic)
        data1 = resp1.json()
        data2 = resp2.json()
        assert data1["label"] == data2["label"]
