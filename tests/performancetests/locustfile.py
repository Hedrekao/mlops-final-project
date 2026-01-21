import random

from locust import HttpUser, between, task


class PostingsClassifierUser(HttpUser):
    """Locust user class that simulates users interacting with the Postings Classifier API."""

    wait_time = between(1, 3)

    # Sample job posting texts for realistic testing
    sample_texts = [
        "We are looking for a senior software engineer with 5+ years of experience in Python and machine learning.",
        "URGENT! Work from home opportunity! Make $5000 per week! No experience needed! Contact us now!",
        "Data Scientist position at established tech company. Requirements: PhD in CS, experience with PyTorch and TensorFlow.",
        "Earn money fast! This is a limited time offer! Send us your bank details to get started immediately!",
        "Full-stack developer needed for startup. Must know React, Node.js, and AWS. Competitive salary and equity.",
        "Remote position available. Junior developer role with mentorship. Requirements: Bachelor's degree in Computer Science.",
        "Make thousands working from home! No qualifications required! This is not a scam! Apply now before spots fill up!",
        "Machine Learning Engineer - Work on cutting-edge AI projects. Strong background in NLP required.",
        "Project Manager position in Fortune 500 company. 3+ years experience. PMP certification preferred.",
        "Easy money! Click here! Foreign investment opportunity! Send payment for processing fee!",
    ]

    @task(2)
    def get_root(self) -> None:
        """Test the root endpoint."""
        self.client.get("/", name="GET /")

    @task(1)
    def get_health(self) -> None:
        """Test the health check endpoint."""
        self.client.get("/health", name="GET /health")

    @task(10)
    def predict_posting(self) -> None:
        """Test the predict endpoint with sample job posting text."""
        text = random.choice(self.sample_texts)
        self.client.post(
            "/predict",
            json={"text": text},
            headers={"Content-Type": "application/json"},
            name="POST /predict"
        )

    @task(1)
    def get_monitoring_stats(self) -> None:
        """Test the monitoring stats endpoint."""
        self.client.get("/monitoring/stats", name="GET /monitoring/stats")

    @task(1)
    def get_monitoring_report(self) -> None:
        """Test the monitoring report endpoint."""
        n = random.choice([10, 50, 100])
        self.client.get(f"/monitoring/report?n={n}", name="GET /monitoring/report")