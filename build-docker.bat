@echo off
REM Build script for Docker images
REM This script builds both the API and Training Docker images

echo.
echo ====================================
echo Building Postings Classifier Images
echo ====================================
echo.

REM Build API Image
echo [1/2] Building API Image...
docker build -f dockerfiles/api.dockerfile . -t postings-classifier-api:latest
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to build API image
    exit /b 1
)
echo [OK] API image built successfully

echo.

REM Build Training Image
echo [2/2] Building Training Image...
docker build -f dockerfiles/train.dockerfile . -t postings-classifier-train:latest
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to build Training image
    exit /b 1
)
echo [OK] Training image built successfully

echo.
echo ====================================
echo Build Complete!
echo ====================================
echo.
echo Available images:
docker images | findstr postings-classifier
echo.
echo To run the API:
echo   docker run --rm -p 8000:8000 -v %%cd%%/models:/app/models postings-classifier-api:latest
echo.
echo To run training:
echo   docker run --rm -v %%cd%%/data:/app/data -v %%cd%%/models:/app/models postings-classifier-train:latest
echo.
