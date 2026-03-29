"""
src/rag_system.py — RAG System Wrapper
Wraps the agri-crop-qa RAG pipeline and exposes a clean interface
for the evaluation system to use.
"""

import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# Constants - loaded from environment


GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "agri_db")
EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "models/gemini-embedding-001")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
NUM_RETRIEVED_DOCS: int = int(os.getenv("NUM_RETRIEVED_DOCS", "6"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Add it to your .env file: GOOGLE_API_KEY=your_key_here"
    )


# Vector Store

def load_vector_store() -> Optional[Chroma]:
    """
    Load the existing Chroma vector store from disk.

    Returns:
        Chroma instance if the vector store exists, otherwise None.
    """
    if not (os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR)):
        print(f"No vector store found at '{VECTOR_DB_DIR}'.")
        return None

    print(f"Loading vector store from '{VECTOR_DB_DIR}'...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document",
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )

    print("Vector store loaded successfully.")
    return vectorstore


# QA Chain
def create_qa_chain(
    vectorstore: Chroma,
    top_k: int = NUM_RETRIEVED_DOCS,
    temperature: float = TEMPERATURE,
):
    """
    Create a retrieval-augmented generation chain using Gemini.

    Args:
        vectorstore: Chroma vector store to retrieve from.
        top_k: Number of document chunks to retrieve per query.
        temperature: LLM sampling temperature (0.0 = deterministic).

    Returns:
        A LangChain retrieval chain ready for invoke().
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an agricultural assistant. Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say: 'I don't know based on the documents.'\n"
            "Keep answers concise and practical.",
        ),
        (
            "human",
            "Question: {input}\n\nContext:\n{context}",
        ),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return qa_chain

# Ask a Question

def ask_question(qa_chain, question: str) -> dict:
    """
    Run a single question through the RAG pipeline.

    Args:
        qa_chain: A LangChain retrieval chain from create_qa_chain().
        question: The user's natural language question.

    Returns:
        Dictionary with keys:
            "answer" (str): The generated answer text.
            "sources" (list[Document]): Retrieved source chunks used.
            "error" (str | None): Error message if the call failed.
    """
    if not question or not question.strip():
        return {
            "answer": "",
            "sources": [],
            "error": "Empty question provided.",
        }

    try:
        result = qa_chain.invoke({"input": question})
        answer: str = (
            result.get("answer")
            or result.get("output_text")
            or "No answer returned."
        )
        sources: list[Document] = result.get("context", [])
        return {"answer": answer, "sources": sources, "error": None}

    except Exception as e:
        print(f"ERROR during RAG query: {e}")
        return {"answer": "", "sources": [], "error": str(e)}