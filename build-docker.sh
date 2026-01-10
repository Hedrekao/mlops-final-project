#!/bin/bash
# Build script for Docker images
# This script builds both the API and Training Docker images

echo ""
echo "===================================="
echo "Building Postings Classifier Images"
echo "===================================="
echo ""

# Build API Image
echo "[1/2] Building API Image..."
docker build -f dockerfiles/api.dockerfile . -t postings-classifier-api:latest
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build API image"
    exit 1
fi
echo "[OK] API image built successfully"

echo ""

# Build Training Image
echo "[2/2] Building Training Image..."
docker build -f dockerfiles/train.dockerfile . -t postings-classifier-train:latest
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build Training image"
    exit 1
fi
echo "[OK] Training image built successfully"

echo ""
echo "===================================="
echo "Build Complete!"
echo "===================================="
echo ""
echo "Available images:"
docker images | grep postings-classifier
echo ""
echo "To run the API:"
echo "  docker run --rm -p 8000:8000 -v \$(pwd)/models:/app/models postings-classifier-api:latest"
echo ""
echo "To run training:"
echo "  docker run --rm -v \$(pwd)/data:/app/data -v \$(pwd)/models:/app/models postings-classifier-train:latest"
echo ""
