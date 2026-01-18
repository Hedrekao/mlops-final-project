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
COPY README.md LICENSE ./

# Install project
RUN uv sync --frozen --no-dev

# Set environment variable for HuggingFace cache
ENV HF_HOME=/app/.cache/huggingface

# Pre-cache tokenizer and model during build (before setting offline mode)
RUN mkdir -p $HF_HOME && \
    uv run python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
    AutoModel.from_pretrained('distilbert-base-uncased')" && \
    echo "Models cached successfully"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD uv run python -c "import os,urllib.request; p=os.getenv('PORT','8080'); urllib.request.urlopen(f'http://127.0.0.1:{p}/health')" || exit 1

CMD ["sh", "-c", "uv run uvicorn postings_classifier.api:app --host 0.0.0.0 --port 8080 --workers 1"]
