"""
tests/test_rag_system.py — Tests for the RAG System Wrapper
Uses mocking to avoid real API calls during testing.
Run with: pytest tests/test_rag_system.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


# Tests - load_vector_store

def test_load_vector_store_returns_chroma_when_exists(tmp_path):
    """Test that load_vector_store returns a Chroma object when DB exists."""
    from src.rag_system import load_vector_store

    # Create a fake non-empty directory to simulate existing vector store
    fake_db = tmp_path / "agri_db"
    fake_db.mkdir()
    (fake_db / "chroma.sqlite3").write_text("fake")

    mock_chroma = MagicMock()

    with patch("src.rag_system.VECTOR_DB_DIR", str(fake_db)), \
         patch("src.rag_system.Chroma", return_value=mock_chroma), \
         patch("src.rag_system.GoogleGenerativeAIEmbeddings"):

        result = load_vector_store()

    assert result is not None


def test_load_vector_store_returns_none_when_missing(tmp_path):
    """Test that load_vector_store returns None when DB does not exist."""
    from src.rag_system import load_vector_store

    fake_db = tmp_path / "nonexistent_db"

    with patch("src.rag_system.VECTOR_DB_DIR", str(fake_db)):
        result = load_vector_store()

    assert result is None


def test_load_vector_store_returns_none_when_empty(tmp_path):
    """Test that load_vector_store returns None when DB directory is empty."""
    from src.rag_system import load_vector_store

    empty_db = tmp_path / "empty_db"
    empty_db.mkdir()

    with patch("src.rag_system.VECTOR_DB_DIR", str(empty_db)):
        result = load_vector_store()

    assert result is None


# Tests - create_qa_chain
def test_create_qa_chain_returns_dict_with_chain_and_retriever():
    """Test that create_qa_chain returns dict with chain and retriever keys."""
    from src.rag_system import create_qa_chain

    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever

    with patch("src.rag_system.ChatGoogleGenerativeAI"):
        result = create_qa_chain(mock_vectorstore)

    assert isinstance(result, dict)
    assert "chain" in result
    assert "retriever" in result


def test_create_qa_chain_uses_correct_top_k():
    """Test that create_qa_chain passes top_k to the retriever."""
    from src.rag_system import create_qa_chain

    mock_vectorstore = MagicMock()

    with patch("src.rag_system.ChatGoogleGenerativeAI"):
        create_qa_chain(mock_vectorstore, top_k=8)

    mock_vectorstore.as_retriever.assert_called_once_with(
        search_kwargs={"k": 8}
    )


# Tests - ask_question

def test_ask_question_returns_answer_and_sources():
    """Test that ask_question returns answer and sources on success."""
    from src.rag_system import ask_question

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Crop rotation improves soil health."

    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "crop_management.pdf"}

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]

    qa_chain = {"chain": mock_chain, "retriever": mock_retriever}

    result = ask_question(qa_chain, "What is crop rotation?")

    assert result["answer"] == "Crop rotation improves soil health."
    assert result["error"] is None
    assert len(result["sources"]) == 1


def test_ask_question_handles_empty_question():
    """Test that ask_question returns error for empty question."""
    from src.rag_system import ask_question

    qa_chain = {"chain": MagicMock(), "retriever": MagicMock()}

    result = ask_question(qa_chain, "")

    assert result["answer"] == ""
    assert result["error"] == "Empty question provided."
    assert result["sources"] == []


def test_ask_question_handles_whitespace_question():
    """Test that ask_question returns error for whitespace-only question."""
    from src.rag_system import ask_question

    qa_chain = {"chain": MagicMock(), "retriever": MagicMock()}

    result = ask_question(qa_chain, "   ")

    assert result["answer"] == ""
    assert result["error"] == "Empty question provided."


def test_ask_question_handles_api_exception():
    """Test that ask_question handles API exceptions gracefully."""
    from src.rag_system import ask_question

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("API rate limit exceeded")

    qa_chain = {"chain": mock_chain, "retriever": MagicMock()}

    result = ask_question(qa_chain, "What is crop rotation?")

    assert result["answer"] == ""
    assert result["error"] == "API rate limit exceeded"
    assert result["sources"] == []


def test_ask_question_returns_sources_as_list():
    """Test that ask_question always returns sources as a list."""
    from src.rag_system import ask_question

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Some answer"

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    qa_chain = {"chain": mock_chain, "retriever": mock_retriever}

    result = ask_question(qa_chain, "Test question?")

    assert isinstance(result["sources"], list)