# LLM Regression Detection System - Dockerfile
# Uses Python 3.11 slim for a lightweight production image.
# Secrets are injected at runtime via environment variables(never baked in)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
# If requirements.txt hasn't changed, this layer is cached
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY templates/ ./templates/
COPY data/ ./data/

# Create output directories
RUN mkdir -p reports

# Environment variables with defaults
# GOOGLE_API_KEY must be provided at runtime — never set here
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VECTOR_DB_DIR=/app/agri_db \
    EMBEDDINGS_MODEL=models/gemini-embedding-001 \
    LLM_MODEL=gemini-2.5-flash \
    NUM_RETRIEVED_DOCS=6 \
    TEMPERATURE=0.0 \
    WARN_THRESHOLD=0.03 \
    FAIL_THRESHOLD=0.08 \
    REPORTS_DIR=/app/reports \
    GOLDEN_DATASET_PATH=/app/data/golden_dataset.json \
    DATABASE_PATH=/app/data/eval_results.db

# Run the evaluation pipeline
ENTRYPOINT ["python", "src/main.py"]