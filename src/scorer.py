"""
Semantic Similarity Scorer:
Scores RAG answers against expected answers using Google Gemini embeddings.
Uses cosine similarity to measure semantic closeness between answers.
Score range: 0.0 (completely different) to 1.0 (identical meaning).
"""

import math
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import GOOGLE_API_KEY, EMBEDDINGS_MODEL
from src.evaluator import EvalResult, EvalRun


# Embedding Client
def _get_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    """
    Create and return a Google embeddings client.

    Returns:
        GoogleGenerativeAIEmbeddings instance.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query",
    )


# Cosine Similarity
def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score between 0.0 and 1.0.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    similarity = dot_product / (magnitude_a * magnitude_b)

    # Clamp to [0.0, 1.0] to handle floating point edge cases
    return max(0.0, min(1.0, similarity))


# Score Single Result
def score_result(
    result: EvalResult,
    embeddings_client: Optional[GoogleGenerativeAIEmbeddings] = None,
) -> float:
    """
    Score a single EvalResult by comparing actual vs expected answer.

    Uses semantic similarity via Google embeddings — not exact word matching.
    If the RAG returned an error or empty answer, score is 0.0.

    Args:
        result: EvalResult object with actual and expected answers.
        embeddings_client: Optional pre-created embeddings client.
            If None, creates a new one (less efficient for bulk scoring).

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    # If there was an error or no answer, score is 0
    if result.error or not result.actual_answer.strip():
        return 0.0

    # If the RAG said it doesn't know, score is very low
    if "i don't know" in result.actual_answer.lower():
        return 0.1

    client = embeddings_client or _get_embeddings_client()

    try:
        # Embed both answers
        expected_embedding = client.embed_query(result.expected_answer)
        actual_embedding = client.embed_query(result.actual_answer)

        score = _cosine_similarity(expected_embedding, actual_embedding)
        return round(score, 4)

    except Exception as e:
        print(f"  Scoring error for {result.question_id}: {e}")
        return 0.0

# Score Full Eval Run
def score_eval_run(eval_run: EvalRun) -> EvalRun:
    """
    Score all results in an EvalRun and update the average score.

    Creates a single embeddings client and reuses it for all questions
    to avoid repeated initialisation overhead.

    Args:
        eval_run: EvalRun object from the evaluator.

    Returns:
        The same EvalRun with scores populated on all results.
    """
    print("\nScoring evaluation results...")

    client = _get_embeddings_client()
    total_score = 0.0
    scored_count = 0

    for i, result in enumerate(eval_run.results):
        print(f"  Scoring [{i+1:02d}/{len(eval_run.results)}] {result.question_id}...")

        score = score_result(result, embeddings_client=client)
        result.score = score
        total_score += score
        scored_count += 1

        # Assign status based on score
        if result.error:
            result.status = "ERROR"
        elif score >= 0.7:
            result.status = "PASS"
        elif score >= 0.4:
            result.status = "WARN"
        else:
            result.status = "FAIL"

        print(f"    Score: {score:.4f} → {result.status}")

    # Update run average
    eval_run.average_score = round(
        total_score / scored_count if scored_count > 0 else 0.0,
        4
    )

    print(f"\nScoring complete. Average score: {eval_run.average_score:.4f}")
    return eval_run