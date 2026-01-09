# Docker Setup Guide

This document describes the Docker configuration for the Fake vs Real Job Postings ML classifier project.

## Overview

The project includes two main Docker images:
1. **API Image** (`api.dockerfile`) - For running the FastAPI application for model inference
2. **Training Image** (`train.dockerfile`) - For training the model

## Prerequisites

- Docker Desktop installed ([Download](https://www.docker.com/products/docker-desktop))
- Docker version 23.0+ (for BuildKit support)
- Windows/Mac: Docker Desktop includes Docker Engine
- Linux: Install Docker Engine and Docker Compose separately

## Building Docker Images

### Build API Image
```bash
docker build -f dockerfiles/api.dockerfile . -t postings-classifier-api:latest
```

### Build Training Image
```bash
docker build -f dockerfiles/train.dockerfile . -t postings-classifier-train:latest
```

### Build Both Images
```bash
docker-compose build
```

## Running Containers

### Run API Container
```powershell
# PowerShell (single-line)
docker run --rm -p 8000:8000 -v "${PWD}/models:/app/models" -v "${PWD}/configs:/app/configs" postings-classifier-api:latest

# PowerShell (multiline - use backtick ` as continuation)
docker run --rm -p 8000:8000 `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/configs:/app/configs" `
  postings-classifier-api:latest
```

```cmd
# CMD (cmd.exe) - single line
docker run --rm -p 8000:8000 -v %cd%/models:/app/models -v %cd%/configs:/app/configs postings-classifier-api:latest
```

The API will be available at `http://localhost:8000`

### Run Training Container
```bash
docker run --name train-experiment --rm `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/configs:/app/configs" `
  postings-classifier-train:latest
```

### Using Docker Compose

Start the API service:
```bash
docker-compose up api
```

Run training (uses the `training` profile):
```bash
docker-compose run --rm train
```

## Volume Mounts

The containers use volume mounts to share data with the host machine:

| Mount Point | Purpose | Notes |
|---|---|---|
| `/app/data` | Training/evaluation data | Only mounted in training container |
| `/app/models` | Trained models & checkpoints | Mounted in both containers |
| `/app/configs` | Configuration files | Mounted in both containers |

## Environment Variables

Both containers support:
- `PYTHONUNBUFFERED=1` - Ensures Python output is sent straight to logs

## API Endpoints

When running the API container:

- `GET /health` - Health check endpoint
- `POST /predict` - Make predictions (requires model to be present in `/app/models`)
- `GET /docs` - Swagger UI documentation
- `GET /openapi.json` - OpenAPI schema

## Building with Cache

To speed up rebuilds, Docker BuildKit caches dependencies:

```bash
# First build (installs all dependencies)
docker build -f dockerfiles/api.dockerfile . -t postings-classifier-api:latest

# Subsequent builds are much faster due to caching
docker build -f dockerfiles/api.dockerfile . -t postings-classifier-api:latest
```

## Debugging Containers

### Interactive Shell
```bash
docker run --rm -it --entrypoint sh postings-classifier-api:latest
```

### View Logs
```bash
docker logs <container-name>
```

### Copy Files from Container
```bash
docker cp <container-name>:/app/models/. ./models/
```

## Image Optimization

The Docker setup uses:
- **Alpine Linux base** - Minimal image size (~200MB vs 1GB+)
- **uv package manager** - Fast, reliable dependency management
- **Multi-stage layers** - Efficient caching and smaller final image
- **.dockerignore** - Excludes unnecessary files from build context

## Troubleshooting

### "docker: command not found"
- Ensure Docker Desktop is installed and running
- On Windows, ensure WSL 2 is configured
- Restart terminal/IDE after Docker installation

### Build fails with "pyproject.toml not found"
- Ensure you're running docker build from the project root directory
- The `.` argument is important: `docker build -f dockerfiles/api.dockerfile .`

### Port 8000 already in use
- Change the port mapping: `-p 9000:8000`
- Or kill the existing container: `docker ps` then `docker stop <container-id>`

### Permission denied on volume mounts
- Windows: Ensure Docker Desktop has file sharing enabled for your drive
- Linux/Mac: Check directory permissions with `ls -la`

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [DTU MLOps Docker Guide](https://skaftenicki.github.io/dtu_mlops/s3_reproducibility/docker/)
- [FastAPI Docker Documentation](https://fastapi.tiangolo.com/deployment/docker/)
