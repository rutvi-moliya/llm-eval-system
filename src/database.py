"""
SQLite Results Database:
Persists evaluation runs and results across sessions.
Enables regression detection by storing historical scores.
Uses Python's built-in sqlite3 — no ORM, no extra dependencies.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from src.config import DATABASE_PATH
from src.evaluator import EvalResult, EvalRun

# Schema
CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    total_questions INTEGER NOT NULL,
    completed       INTEGER NOT NULL,
    failed          INTEGER NOT NULL,
    average_score   REAL NOT NULL,
    status          TEXT NOT NULL
);
"""

CREATE_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS eval_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    question_id         TEXT NOT NULL,
    question            TEXT NOT NULL,
    expected_answer     TEXT NOT NULL,
    actual_answer       TEXT NOT NULL,
    source_document     TEXT NOT NULL,
    difficulty          TEXT NOT NULL,
    category            TEXT NOT NULL,
    score               REAL NOT NULL,
    status              TEXT NOT NULL,
    error               TEXT,
    retrieved_sources   TEXT,
    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
);
"""


# Connection Helper
def _get_connection() -> sqlite3.Connection:
    """
    Create and return a SQLite connection.
    Ensures the database directory exists.

    Returns:
        sqlite3.Connection instance.
    """
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialise_database() -> None:
    """
    Create database tables if they don't exist.
    Safe to call multiple times — uses CREATE IF NOT EXISTS.
    """
    with _get_connection() as conn:
        conn.execute(CREATE_RUNS_TABLE)
        conn.execute(CREATE_RESULTS_TABLE)
        conn.commit()
    print(f"Database initialised at '{DATABASE_PATH}'.")


# Save Run
def save_run(eval_run: EvalRun) -> None:
    """
    Save a complete EvalRun to the database.

    Saves the run summary to eval_runs and all individual
    question results to eval_results.

    Args:
        eval_run: Completed and scored EvalRun to persist.
    """
    initialise_database()

    with _get_connection() as conn:
        # Save run summary
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_runs
            (run_id, timestamp, total_questions, completed, failed, average_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eval_run.run_id,
                eval_run.timestamp,
                eval_run.total_questions,
                eval_run.completed_questions,
                eval_run.failed_questions,
                eval_run.average_score,
                eval_run.status,
            ),
        )

        # Save individual results
        for result in eval_run.results:
            conn.execute(
                """
                INSERT INTO eval_results
                (run_id, question_id, question, expected_answer, actual_answer,
                 source_document, difficulty, category, score, status, error, retrieved_sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_run.run_id,
                    result.question_id,
                    result.question,
                    result.expected_answer,
                    result.actual_answer,
                    result.source_document,
                    result.difficulty,
                    result.category,
                    result.score,
                    result.status,
                    result.error,
                    json.dumps(result.retrieved_sources),
                ),
            )

        conn.commit()

    print(f"Saved run '{eval_run.run_id}' to database.")


# Get Last Run (Baseline)
def get_last_run() -> Optional[EvalRun]:
    """
    Retrieve the most recent completed eval run from the database.
    Used as the baseline for regression detection.

    Returns:
        EvalRun object if a previous run exists, otherwise None.
    """
    initialise_database()

    with _get_connection() as conn:
        # Get most recent run
        run_row = conn.execute(
            """
            SELECT * FROM eval_runs
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).fetchone()

        if run_row is None:
            return None

        # Get all results for that run
        result_rows = conn.execute(
            """
            SELECT * FROM eval_results
            WHERE run_id = ?
            ORDER BY question_id
            """,
            (run_row["run_id"],),
        ).fetchall()

    # Reconstruct EvalResult objects
    results = []
    for row in result_rows:
        result = EvalResult(
            question_id=row["question_id"],
            question=row["question"],
            expected_answer=row["expected_answer"],
            actual_answer=row["actual_answer"],
            source_document=row["source_document"],
            difficulty=row["difficulty"],
            category=row["category"],
            score=row["score"],
            status=row["status"],
            error=row["error"],
            retrieved_sources=json.loads(row["retrieved_sources"] or "[]"),
        )
        results.append(result)

    # Reconstruct EvalRun
    eval_run = EvalRun(
        run_id=run_row["run_id"],
        timestamp=run_row["timestamp"],
        total_questions=run_row["total_questions"],
        completed_questions=run_row["completed"],
        failed_questions=run_row["failed"],
        average_score=run_row["average_score"],
        results=results,
        status=run_row["status"],
    )

    return eval_run

# Get Run History
def get_run_history(limit: int = 10) -> list[dict]:
    """
    Retrieve summary of recent eval runs for trend reporting.

    Args:
        limit: Maximum number of runs to return.

    Returns:
        List of run summary dictionaries ordered by timestamp descending.
    """
    initialise_database()

    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT run_id, timestamp, total_questions, completed,
                   failed, average_score, status
            FROM eval_runs
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]