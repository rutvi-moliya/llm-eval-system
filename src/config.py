"""
src/config.py — Eval System Configuration
Loads all settings from environment variables with validation.
All secrets come from .env — never hardcoded.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# API Keys
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    sys.exit(
        "ERROR: GOOGLE_API_KEY is not set.\n"
        "Add it to your .env file: GOOGLE_API_KEY=your_key_here"
    )



# Paths

# Root of the project
PROJECT_ROOT: Path = Path(__file__).parent.parent

# Vector store from the agri-crop-qa RAG project
VECTOR_DB_DIR: str = os.getenv(
    "VECTOR_DB_DIR",
    str(PROJECT_ROOT / "agri_db")
)

# Golden dataset of 25 test questions
GOLDEN_DATASET_PATH: str = os.getenv(
    "GOLDEN_DATASET_PATH",
    str(PROJECT_ROOT / "data" / "golden_dataset.json")
)

# Reports output directory
REPORTS_DIR: str = os.getenv(
    "REPORTS_DIR",
    str(PROJECT_ROOT / "reports")
)

# SQLite database for storing eval results
DATABASE_PATH: str = os.getenv(
    "DATABASE_PATH",
    str(PROJECT_ROOT / "data" / "eval_results.db")
)


# Model Configuration

EMBEDDINGS_MODEL: str = os.getenv(
    "EMBEDDINGS_MODEL",
    "models/gemini-embedding-001"
)

LLM_MODEL: str = os.getenv(
    "LLM_MODEL",
    "gemini-2.5-flash"
)

NUM_RETRIEVED_DOCS: int = int(os.getenv("NUM_RETRIEVED_DOCS", "6"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))


# Regression Detection Thresholds

# Score drop % that triggers a WARNING
WARN_THRESHOLD: float = float(os.getenv("WARN_THRESHOLD", "0.03"))

# Score drop % that triggers a FAILURE and fails the CI pipeline
FAIL_THRESHOLD: float = float(os.getenv("FAIL_THRESHOLD", "0.08"))


# Ensure output directories exist
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)