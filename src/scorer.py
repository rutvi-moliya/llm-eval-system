"""
src/scorer.py — Semantic Similarity Scorer
Scores RAG answers against expected answers using OpenAI embeddings.
Uses cosine similarity to measure semantic closeness between answers.
Score range: 0.0 (completely different) to 1.0 (identical meaning).

Uses OpenAI text-embedding-3-small for reliable, production-grade scoring.
The RAG pipeline uses Google Gemini — this is a deliberate architectural
decision to use the most reliable embedding model for evaluation scoring.
"""

import math
import os
import time
from typing import Optional

from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.evaluator import EvalResult, EvalRun


# ---------------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
SCORING_MODEL = "text-embedding-3-small"


def _get_client() -> OpenAI:
    """Create and return an OpenAI client."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Embed Text
# ---------------------------------------------------------------------------

def _embed_text(client: OpenAI, text: str) -> list[float]:
    """
    Embed a single text string using OpenAI embeddings.

    Args:
        client: OpenAI client.
        text: Text to embed.

    Returns:
        Embedding vector as list of floats.
    """
    response = client.embeddings.create(
        model=SCORING_MODEL,
        input=text,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns:
        Cosine similarity score between 0.0 and 1.0.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    similarity = dot_product / (magnitude_a * magnitude_b)
    return max(0.0, min(1.0, similarity))


# ---------------------------------------------------------------------------
# Score Single Result
# ---------------------------------------------------------------------------

def score_result(
    result: EvalResult,
    embeddings_client: Optional[OpenAI] = None,
) -> float:
    """
    Score a single EvalResult by comparing actual vs expected answer.

    Args:
        result: EvalResult object with actual and expected answers.
        embeddings_client: Optional pre-created OpenAI client.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if result.error or not result.actual_answer.strip():
        return 0.0

    if (
        "i don't know" in result.actual_answer.lower()
        or len(result.actual_answer.strip()) < 20
    ):
        return 0.1

    client = embeddings_client or _get_client()

    try:
        expected_embedding = _embed_text(client, result.expected_answer)
        actual_embedding = _embed_text(client, result.actual_answer)
        score = _cosine_similarity(expected_embedding, actual_embedding)
        return round(score, 4)

    except Exception as e:
        print(f"  Scoring error for {result.question_id}: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Score Full Eval Run
# ---------------------------------------------------------------------------

def score_eval_run(eval_run: EvalRun) -> EvalRun:
    """
    Score all results in an EvalRun and update the average score.

    Args:
        eval_run: EvalRun object from the evaluator.

    Returns:
        The same EvalRun with scores populated on all results.
    """
    print("\nScoring evaluation results...")

    client = _get_client()
    total_score = 0.0
    scored_count = 0

    for i, result in enumerate(eval_run.results):
        print(f"  Scoring [{i+1:02d}/{len(eval_run.results)}] {result.question_id}...")

        score = score_result(result, embeddings_client=client)
        result.score = score
        total_score += score
        scored_count += 1

        if result.error:
            result.status = "ERROR"
        elif score >= 0.7:
            result.status = "PASS"
        elif score >= 0.4:
            result.status = "WARN"
        else:
            result.status = "FAIL"

        print(f"    Score: {score:.4f} → {result.status}")

    eval_run.average_score = round(
        total_score / scored_count if scored_count > 0 else 0.0,
        4
    )

    print(f"\nScoring complete. Average score: {eval_run.average_score:.4f}")
    return eval_run