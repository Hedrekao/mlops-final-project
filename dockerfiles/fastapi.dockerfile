FROM python:3.12-slim

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv /opt/venv && /opt/venv/bin/python -V

# install uv into the venv (not system python)
RUN /opt/venv/bin/pip install --no-cache-dir uv

# make uv install into this environment
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src src/
COPY configs configs/
COPY README.md LICENSE ./

RUN uv sync --frozen --no-dev

EXPOSE 8080

CMD ["sh", "-lc", "/opt/venv/bin/python -m uvicorn postings_classifier.api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
