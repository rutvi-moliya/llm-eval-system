"""
Tests for the Core Evaluation Pipeline:
Uses mocking to avoid real API calls during testing.
Run with: pytest tests/test_evaluator.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Fixtures

@pytest.fixture
def sample_dataset(tmp_path):
    """Create a temporary golden dataset for testing."""
    dataset = [
        {
            "id": "Q001",
            "question": "What is crop rotation?",
            "expected_answer": "Crop rotation is the practice of growing different crops sequentially.",
            "source_document": "crop_management.pdf",
            "difficulty": "easy",
            "category": "crop_management"
        },
        {
            "id": "Q002",
            "question": "What is mulching?",
            "expected_answer": "Mulching involves covering soil with organic material.",
            "source_document": "crop_management.pdf",
            "difficulty": "easy",
            "category": "soil_management"
        },
        {
            "id": "Q003",
            "question": "What is soil pH for chickpea?",
            "expected_answer": "Chickpea grows best in soil with pH between 6.0 and 7.5.",
            "source_document": "good_agricultural_practices.pdf",
            "difficulty": "medium",
            "category": "soil_management"
        }
    ]
    dataset_file = tmp_path / "test_dataset.json"
    dataset_file.write_text(json.dumps(dataset), encoding="utf-8")
    return str(dataset_file)


@pytest.fixture
def mock_qa_chain():
    """Create a mock QA chain that returns predictable answers."""
    chain = MagicMock()
    chain.invoke.return_value = {
        "answer": "This is a test answer about agricultural practices.",
        "context": []
    }
    return chain


@pytest.fixture
def mock_vectorstore():
    """Create a mock Chroma vector store."""
    vectorstore = MagicMock()
    vectorstore._collection.count.return_value = 100
    return vectorstore


# Tests - Dataset Loading

def test_load_golden_dataset_success(sample_dataset):
    """Test that golden dataset loads correctly from valid file."""
    from src.evaluator import load_golden_dataset

    dataset = load_golden_dataset(sample_dataset)

    assert len(dataset) == 3
    assert dataset[0]["id"] == "Q001"
    assert dataset[0]["question"] == "What is crop rotation?"


def test_load_golden_dataset_file_not_found():
    """Test that FileNotFoundError is raised for missing dataset."""
    from src.evaluator import load_golden_dataset

    with pytest.raises(FileNotFoundError):
        load_golden_dataset("nonexistent/path/dataset.json")


def test_load_golden_dataset_empty(tmp_path):
    """Test that ValueError is raised for empty dataset."""
    from src.evaluator import load_golden_dataset

    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        load_golden_dataset(str(empty_file))


# Tests - EvalResult
def test_eval_result_defaults():
    """Test EvalResult is created with correct default values."""
    from src.evaluator import EvalResult

    result = EvalResult(
        question_id="Q001",
        question="Test question?",
        expected_answer="Expected answer.",
        actual_answer="Actual answer.",
        source_document="test.pdf",
        difficulty="easy",
        category="test",
    )

    assert result.score == 0.0
    assert result.status == "PENDING"
    assert result.error is None
    assert result.retrieved_sources == []

# Tests - Evaluator
def test_run_evaluation_returns_eval_run(sample_dataset, mock_vectorstore, mock_qa_chain):
    """Test that run_evaluation returns a complete EvalRun object."""
    from src.evaluator import run_evaluation, EvalRun

    with patch("src.evaluator.load_vector_store", return_value=mock_vectorstore), \
         patch("src.evaluator.create_qa_chain", return_value=mock_qa_chain):

        eval_run = run_evaluation(
            dataset_path=sample_dataset,
            delay_between_questions=0.0,
        )

    assert isinstance(eval_run, EvalRun)
    assert eval_run.total_questions == 3
    assert eval_run.completed_questions == 3
    assert eval_run.failed_questions == 0
    assert len(eval_run.results) == 3

def test_run_evaluation_handles_api_error(sample_dataset, mock_vectorstore):
    """Test that evaluator continues when one question fails."""
    from src.evaluator import run_evaluation

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [
        Exception("API Error"),
        "Good answer",
        "Another answer",
    ]

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    mock_qa_chain = {"chain": mock_chain, "retriever": mock_retriever}

    with patch("src.evaluator.load_vector_store", return_value=mock_vectorstore), \
         patch("src.evaluator.create_qa_chain", return_value=mock_qa_chain):

        eval_run = run_evaluation(
            dataset_path=sample_dataset,
            delay_between_questions=0.0,
        )

    assert eval_run.total_questions == 3
    assert eval_run.failed_questions == 1
    assert eval_run.completed_questions == 2

def test_run_evaluation_raises_when_no_vectorstore(sample_dataset):
    """Test that RuntimeError is raised when vector store is missing."""
    from src.evaluator import run_evaluation

    with patch("src.evaluator.load_vector_store", return_value=None):
        with pytest.raises(RuntimeError, match="Vector store not found"):
            run_evaluation(dataset_path=sample_dataset)


# Tests - Scorer

def test_scorer_returns_float_between_0_and_1():
    """Test that scorer returns a valid score in [0.0, 1.0]."""
    from src.evaluator import EvalResult
    from src.scorer import score_result

    result = EvalResult(
        question_id="Q001",
        question="What is crop rotation?",
        expected_answer="Crop rotation grows different crops sequentially.",
        actual_answer="Crop rotation is a farming practice of alternating crops.",
        source_document="test.pdf",
        difficulty="easy",
        category="test",
        status="COMPLETED",
    )

    mock_client = MagicMock()
    mock_client.embed_query.side_effect = [
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.5],
    ]

    score = score_result(result, embeddings_client=mock_client)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_scorer_returns_zero_for_error():
    """Test that scorer returns 0.0 when result has an error."""
    from src.evaluator import EvalResult
    from src.scorer import score_result

    result = EvalResult(
        question_id="Q001",
        question="Test?",
        expected_answer="Expected.",
        actual_answer="",
        source_document="test.pdf",
        difficulty="easy",
        category="test",
        error="API Error",
    )

    score = score_result(result)
    assert score == 0.0


def test_scorer_returns_low_score_for_dont_know():
    """Test that 'I don't know' answers receive a low score."""
    from src.evaluator import EvalResult
    from src.scorer import score_result

    result = EvalResult(
        question_id="Q001",
        question="Test?",
        expected_answer="The answer is X.",
        actual_answer="I don't know based on the documents.",
        source_document="test.pdf",
        difficulty="easy",
        category="test",
        status="COMPLETED",
    )

    score = score_result(result)
    assert score == 0.1


# Tests - Regression Detector
def test_regression_detector_flags_score_drop():
    """Test that a significant score drop is flagged as FAIL."""
    from src.evaluator import EvalResult, EvalRun
    from src.regression_detector import detect_regressions
    from datetime import datetime

    def make_result(qid, score):
        return EvalResult(
            question_id=qid,
            question=f"Question {qid}",
            expected_answer="Expected.",
            actual_answer="Actual.",
            source_document="test.pdf",
            difficulty="easy",
            category="test",
            score=score,
            status="COMPLETED",
        )

    baseline_run = EvalRun(
        run_id="baseline",
        timestamp=datetime.now().isoformat(),
        total_questions=1,
        completed_questions=1,
        failed_questions=0,
        average_score=0.9,
        results=[make_result("Q001", 0.9)],
    )

    current_run = EvalRun(
        run_id="current",
        timestamp=datetime.now().isoformat(),
        total_questions=1,
        completed_questions=1,
        failed_questions=0,
        average_score=0.7,
        results=[make_result("Q001", 0.7)],
    )

    report = detect_regressions(current_run, baseline_run)

    assert report.overall_status == "FAIL"
    assert report.fail_count == 1
    assert report.regression_results[0].status == "FAIL"


def test_regression_detector_no_baseline():
    """Test that NO_BASELINE status is returned when baseline is None."""
    from src.evaluator import EvalResult, EvalRun
    from src.regression_detector import detect_regressions
    from datetime import datetime

    current_run = EvalRun(
        run_id="first_run",
        timestamp=datetime.now().isoformat(),
        total_questions=1,
        completed_questions=1,
        failed_questions=0,
        average_score=0.8,
        results=[
            EvalResult(
                question_id="Q001",
                question="Test?",
                expected_answer="Expected.",
                actual_answer="Actual.",
                source_document="test.pdf",
                difficulty="easy",
                category="test",
                score=0.8,
                status="COMPLETED",
            )
        ],
    )

    report = detect_regressions(current_run, baseline_run=None)

    assert report.overall_status == "NO_BASELINE"
    assert report.new_count == 1


# Tests - Database

def test_database_saves_and_retrieves_run(tmp_path):
    """Test that a run can be saved and retrieved from the database."""
    from src.evaluator import EvalResult, EvalRun
    from src.database import save_run, get_last_run
    from datetime import datetime

    # Patch DATABASE_PATH to use temp directory
    with patch("src.database.DATABASE_PATH", str(tmp_path / "test.db")), \
         patch("src.config.DATABASE_PATH", str(tmp_path / "test.db")):

        run = EvalRun(
            run_id="test_run_001",
            timestamp=datetime.now().isoformat(),
            total_questions=1,
            completed_questions=1,
            failed_questions=0,
            average_score=0.85,
            results=[
                EvalResult(
                    question_id="Q001",
                    question="Test question?",
                    expected_answer="Expected answer.",
                    actual_answer="Actual answer.",
                    source_document="test.pdf",
                    difficulty="easy",
                    category="test",
                    score=0.85,
                    status="PASS",
                )
            ],
            status="COMPLETED",
        )

        save_run(run)
        retrieved = get_last_run()

    assert retrieved is not None
    assert retrieved.run_id == "test_run_001"
    assert retrieved.average_score == 0.85
    assert len(retrieved.results) == 1
    assert retrieved.results[0].question_id == "Q001"