"""
scripts/ingest_documents.py — Document Ingestion Script
Loads PDFs, splits them into chunks, embeds them using Gemini,
and persists the vector store to disk.
Run this once when setting up or when adding new documents.
Usage: python scripts/ingest_documents.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Configuration

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
DOCS_DIR: str = os.getenv("DOCS_DIR", "documents")
VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "agri_db")
EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "models/gemini-embedding-001")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

if not GOOGLE_API_KEY:
    sys.exit("ERROR: GOOGLE_API_KEY is not set in .env file.")


# Document Loading

def load_documents(directory: str) -> list[Document]:
    """
    Load all PDF files from a directory.

    Args:
        directory: Path to the directory containing PDF files.

    Returns:
        List of loaded Document objects.
    """
    if not os.path.exists(directory):
        print(f"ERROR: Documents directory '{directory}' does not exist.")
        print(f"Create it and add PDF files: mkdir {directory}")
        sys.exit(1)

    print(f"Scanning '{directory}' for PDFs...")

    loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    documents = loader.load()

    if not documents:
        print(f"ERROR: No PDF files found in '{directory}'.")
        sys.exit(1)

    print(f"Loaded {len(documents)} pages from {len(set(d.metadata.get('source', '') for d in documents))} PDF files.")
    return documents

# Document Splitting
def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks for embedding.

    Args:
        documents: List of Document objects.

    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


# Vector Store Creation
def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    Build and persist a Chroma vector store from document chunks.
    Handles rate limiting for Google AI Studio free tier.

    Args:
        chunks: List of chunked Document objects.

    Returns:
        Chroma vector store instance.

    Raises:
        RuntimeError: If a batch fails after 15 retries.
    """
    print(f"\nBuilding vector store from {len(chunks)} chunks...")
    print("This will take a while due to API rate limiting. Do not interrupt.\n")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document",
    )

    batch_size = 1
    vectorstore: Chroma | None = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"  Embedding chunk {i + 1} / {len(chunks)}...")

        retries = 0
        while retries < 15:
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=VECTOR_DB_DIR,
                    )
                else:
                    vectorstore.add_documents(batch)

                if i + batch_size < len(chunks):
                    time.sleep(12)

                break

            except Exception as e:
                retries += 1
                print(f"  API error: {e}")
                print(f"  Cooling down 180s (attempt {retries}/15)...")
                time.sleep(180)
        else:
            raise RuntimeError(f"Failed to embed chunk {i} after 15 retries.")

    print(f"\nVector store saved to '{VECTOR_DB_DIR}'.")
    print(f"Total chunks stored: {vectorstore._collection.count()}")  # type: ignore
    return vectorstore  # type: ignore

# Entry Point 

if __name__ == "__main__":
    print("=" * 60)
    print("Document Ingestion Script")
    print("=" * 60)

    docs = load_documents(DOCS_DIR)
    chunks = split_documents(docs)
    create_vector_store(chunks)

    print("\nIngestion complete. Vector store is ready for evaluation.")