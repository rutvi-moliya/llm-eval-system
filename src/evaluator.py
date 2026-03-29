"""
src/evaluator.py — Core Evaluation Engine
Runs all questions from the golden dataset through the RAG pipeline
and collects results for scoring and regression detection.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import GOLDEN_DATASET_PATH
from src.rag_system import ask_question, create_qa_chain, load_vector_store


# Data Classes
@dataclass
class EvalResult:
    """Result for a single question evaluation."""
    question_id: str
    question: str
    expected_answer: str
    actual_answer: str
    source_document: str
    difficulty: str
    category: str
    score: float = 0.0
    status: str = "PENDING"
    error: Optional[str] = None
    retrieved_sources: list[str] = field(default_factory=list)


@dataclass
class EvalRun:
    """Complete results for one full evaluation run."""
    run_id: str
    timestamp: str
    total_questions: int
    completed_questions: int
    failed_questions: int
    average_score: float
    results: list[EvalResult]
    status: str = "PENDING"

# Dataset Loading

def load_golden_dataset(path: str) -> list[dict]:
    """
    Load and return the golden dataset.

    Args:
        path: Path to golden_dataset.json.

    Returns:
        List of question dictionaries.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If dataset is empty.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at '{path}'.\n"
            "Run scripts/validate_dataset.py to check your dataset."
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        raise ValueError("Golden dataset is empty.")

    print(f"Loaded {len(dataset)} questions from golden dataset.")
    return dataset


# Core Evaluation Runner


def run_evaluation(
    dataset_path: str = GOLDEN_DATASET_PATH,
    delay_between_questions: float = 2.0,
) -> EvalRun:
    """
    Run the full evaluation pipeline against all golden dataset questions.

    Loads the vector store, creates the QA chain, runs every question,
    and returns an EvalRun with all results.

    Args:
        dataset_path: Path to the golden dataset JSON file.
        delay_between_questions: Seconds to wait between API calls.
            Prevents rate limiting on free tier.

    Returns:
        EvalRun object containing all results.
    """
    print("=" * 60)
    print("Starting Evaluation Run")
    print("=" * 60)

    # Generate run ID from timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat()

    # Load dataset
    dataset = load_golden_dataset(dataset_path)

    # Load RAG system
    print("\nLoading RAG system...")
    vectorstore = load_vector_store()
    if vectorstore is None:
        raise RuntimeError(
            "Vector store not found. "
            "Run scripts/ingest_documents.py first to build it."
        )

    qa_chain = create_qa_chain(vectorstore)
    print("RAG system ready.\n")

    # Run evaluation
    results: list[EvalResult] = []
    completed = 0
    failed = 0

    for i, entry in enumerate(dataset):
        question_id = entry.get("id", f"Q{i+1:03d}")
        question = entry.get("question", "")
        expected_answer = entry.get("expected_answer", "")
        source_document = entry.get("source_document", "")
        difficulty = entry.get("difficulty", "medium")
        category = entry.get("category", "general")

        print(f"[{i+1:02d}/{len(dataset)}] {question_id}: {question[:60]}...")

        result = EvalResult(
            question_id=question_id,
            question=question,
            expected_answer=expected_answer,
            actual_answer="",
            source_document=source_document,
            difficulty=difficulty,
            category=category,
        )

        try:
            rag_result = ask_question(qa_chain, question)

            if rag_result.get("error"):
                result.actual_answer = ""
                result.error = rag_result["error"]
                result.status = "ERROR"
                failed += 1
                print(f"  ❌ Error: {rag_result['error']}")
            else:
                result.actual_answer = rag_result.get("answer", "")
                result.retrieved_sources = [
                    doc.metadata.get("source", "unknown")
                    for doc in rag_result.get("sources", [])
                ]
                result.status = "COMPLETED"
                completed += 1
                print(f"  ✅ Answered ({len(result.actual_answer)} chars)")

        except Exception as e:
            result.error = str(e)
            result.status = "ERROR"
            failed += 1
            print(f"  ❌ Unexpected error: {e}")

        results.append(result)

        # Rate limit delay between questions
        if i < len(dataset) - 1:
            time.sleep(delay_between_questions)

    # Build EvalRun
    eval_run = EvalRun(
        run_id=run_id,
        timestamp=timestamp,
        total_questions=len(dataset),
        completed_questions=completed,
        failed_questions=failed,
        average_score=0.0,  # Populated by scorer
        results=results,
        status="COMPLETED",
    )

    print("\n" + "=" * 60)
    print(f"Evaluation complete.")
    print(f"  Completed : {completed}/{len(dataset)}")
    print(f"  Failed    : {failed}/{len(dataset)}")
    print("=" * 60)

    return eval_run