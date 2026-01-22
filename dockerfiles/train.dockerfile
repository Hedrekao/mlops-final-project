FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY uv.lock pyproject.toml ./

# Install dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY src src/
COPY configs configs/
COPY data data/
COPY README.md LICENSE ./

# Install project
RUN uv sync --frozen --no-dev

# W&B configuration - these can be overridden at runtime
# WANDB_API_KEY must be provided at runtime (via --env or secrets)
ENV WANDB_PROJECT="mlops-course"
ENV WANDB_LOG_MODEL="true"

ENTRYPOINT ["uv", "run", "python", "-m", "postings_classifier.train"]
